import asyncio
import websockets
import os
import logging
import json
import httpx
import sys
import uuid
import ssl
import time
import re
import base64
import struct
import math
from websockets.connection import State
from websockets.exceptions import ConnectionClosed, ConnectionClosedOK, ConnectionClosedError

SCRIPT_VERSION = "5.0 - Barge-in Support + Resilient Connections"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info(f"Starte Agent Server - VERSION {SCRIPT_VERSION}")

# ENV-VARIABLEN
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")
ELEVENLABS_AGENT_ID = os.environ.get("ELEVENLABS_AGENT_ID")
POC_PROMPT = os.environ.get("ELEVENLABS_POC_PROMPT")
POC_FIRST_MESSAGE = os.environ.get("ELEVENLABS_POC_FIRST_MESSAGE")
WEBSOCKET_HOST = os.environ.get("WEBSOCKET_HOST", "0.0.0.0")
WEBSOCKET_PORT = int(os.environ.get("WEBSOCKET_PORT", "8080"))
VAD_ENERGY_THRESHOLD = float(os.environ.get("VAD_ENERGY_THRESHOLD", "0.05"))
VAD_SPEECH_DURATION_THRESHOLD = float(os.environ.get("VAD_SPEECH_DURATION_THRESHOLD", "0.2"))
VAD_SILENCE_DURATION_THRESHOLD = float(os.environ.get("VAD_SILENCE_DURATION_THRESHOLD", "0.5"))
LOG_INTERVAL = int(os.environ.get("LOG_INTERVAL", "300"))  # 5 Minuten statt 60 Sekunden

if not ELEVENLABS_API_KEY or not ELEVENLABS_AGENT_ID:
    logger.error("API Key oder Agent ID fehlt! Abbruch.")
    sys.exit(1)

logger.info(f"Konfiguration: Agent-ID={ELEVENLABS_AGENT_ID}, Port={WEBSOCKET_PORT}")
logger.info(f"Prompt: {POC_PROMPT}")
logger.info(f"First Message: {POC_FIRST_MESSAGE or '[leer]'}")
logger.info(f"VAD-Konfiguration: Energy={VAD_ENERGY_THRESHOLD}, Speech={VAD_SPEECH_DURATION_THRESHOLD}s, Silence={VAD_SILENCE_DURATION_THRESHOLD}s")

# μ-LAW DEKODIERUNG
def mulaw_decode(sample):
    """
    Dekodiert einen μ-law-kodierten 8-bit Sample zu einem linearen Float-Wert im Bereich [-1, 1].
    """
    # μ-law Dekodierung
    sample = sample & 0xFF  # Stelle sicher, dass es ein 8-bit Wert ist
    sign = 1 if sample < 128 else -1
    sample = abs(sample - 127) if sign < 0 else sample
    sample = ~sample & 0x7F
    
    # Umwandlung in linearen Wert
    result = sign * (((1 + sample) * 2) ** (1/256) - 1)
    return result

# NETZWERKQUALITÄTSMONITOR
class NetworkQualityMonitor:
    def __init__(self, window_size=100):
        self.packet_times = []
        self.window_size = window_size
        self.expected_interval = 0.02  # 20ms für typische Audio-Chunks
        self.packet_loss_count = 0
        self.total_packets = 0
        self.last_sequence_number = None
    
    def record_packet(self, timestamp, sequence_number=None):
        self.packet_times.append(timestamp)
        self.total_packets += 1
        
        # Paketverlust erkennen, wenn Sequenznummern vorhanden sind
        if sequence_number is not None and self.last_sequence_number is not None:
            expected_seq = (self.last_sequence_number + 1) % 65536  # Typischer Wrap-Around
            if sequence_number != expected_seq:
                # Schätze Anzahl verlorener Pakete
                if sequence_number > expected_seq:
                    lost = sequence_number - expected_seq
                else:
                    lost = (65536 - expected_seq) + sequence_number
                self.packet_loss_count += lost
        
        self.last_sequence_number = sequence_number
        
        if len(self.packet_times) > self.window_size:
            self.packet_times.pop(0)
    
    def record_packet_loss(self):
        self.packet_loss_count += 1
    
    @property
    def packet_loss_rate(self):
        if self.total_packets == 0:
            return 0.0
        return min(1.0, self.packet_loss_count / max(1, self.total_packets))
    
    @property
    def jitter(self):
        if len(self.packet_times) < 2:
            return 0.0
        
        intervals = [self.packet_times[i] - self.packet_times[i-1] 
                    for i in range(1, len(self.packet_times))]
        
        deviations = [abs(interval - self.expected_interval) for interval in intervals]
        return sum(deviations) / len(deviations) * 1000  # in ms

# METRIKEN-SAMMLUNG
class MetricsCollector:
    def __init__(self, log_interval=LOG_INTERVAL):  # 5 Minuten statt 60 Sekunden
        self.metrics = {
            "packets_received": 0,
            "packets_sent": 0,
            "audio_chunks_processed": 0,
            "interruptions_detected": 0,
            "backchanneling_detected": 0,
            "reconnection_attempts": 0,
            "successful_reconnections": 0,
            "average_latency": 0,
            "latency_samples": [],
            "errors": {},
            "vad_triggers": 0,
            "context_switches": 0,
            "context_control_failures": 0
        }
        self.log_interval = log_interval
        self.start_time = time.time()
        self.last_cleanup_time = time.time()
    
    def increment(self, metric, value=1):
        if metric in self.metrics:
            self.metrics[metric] += value
    
    def record_latency(self, latency_ms):
        self.metrics["latency_samples"].append(latency_ms)
        # Berechne gleitenden Durchschnitt
        self.metrics["average_latency"] = sum(self.metrics["latency_samples"]) / len(self.metrics["latency_samples"])
        # Begrenze die Anzahl der Samples auf 50 statt 100
        if len(self.metrics["latency_samples"]) > 50:
            self.metrics["latency_samples"] = self.metrics["latency_samples"][-50:]
    
    def record_error(self, error_type, error_message):
        if error_type not in self.metrics["errors"]:
            self.metrics["errors"][error_type] = 0
        self.metrics["errors"][error_type] += 1
        logger.error(f"{error_type}: {error_message}")
    
    def _cleanup_old_metrics(self):
        """Bereinigt alte Metriken, die nicht mehr benötigt werden."""
        # Hier könnten weitere Bereinigungen hinzugefügt werden
        pass
    
    def _log_critical_metrics(self):
        """Loggt nur die wichtigsten Metriken bei hoher Last."""
        runtime = time.time() - self.start_time
        logger.info(f"KRITISCHE METRIKEN NACH {runtime:.1f}s:")
        logger.info(f"  Pakete: {self.metrics['packets_received']} empfangen, {self.metrics['packets_sent']} gesendet")
        logger.info(f"  Unterbrechungen: {self.metrics['interruptions_detected']}")
        logger.info(f"  Fehler: {sum(self.metrics['errors'].values())}")
    
    def _log_all_metrics(self):
        """Loggt alle Metriken bei normaler Last."""
        runtime = time.time() - self.start_time
        
        # Berechne abgeleitete Metriken
        packets_per_second = self.metrics["packets_received"] / runtime if runtime > 0 else 0
        interruptions_per_minute = (self.metrics["interruptions_detected"] / runtime) * 60 if runtime > 0 else 0
        
        logger.info(f"METRIKEN NACH {runtime:.1f}s:")
        logger.info(f"  Pakete empfangen: {self.metrics['packets_received']} ({packets_per_second:.2f}/s)")
        logger.info(f"  Pakete gesendet: {self.metrics['packets_sent']}")
        logger.info(f"  Audio-Chunks verarbeitet: {self.metrics['audio_chunks_processed']}")
        logger.info(f"  Unterbrechungen erkannt: {self.metrics['interruptions_detected']} ({interruptions_per_minute:.2f}/min)")
        logger.info(f"  Backchanneling erkannt: {self.metrics['backchanneling_detected']}")
        logger.info(f"  Wiederverbindungsversuche: {self.metrics['reconnection_attempts']}")
        logger.info(f"  Erfolgreiche Wiederverbindungen: {self.metrics['successful_reconnections']}")
        logger.info(f"  Durchschnittliche Latenz: {self.metrics['average_latency']:.2f}ms")
        logger.info(f"  VAD-Trigger: {self.metrics['vad_triggers']}")
        logger.info(f"  Kontextwechsel: {self.metrics['context_switches']}")
        logger.info(f"  Kontext-Steuerung Fehler: {self.metrics['context_control_failures']}")
        
        if self.metrics["errors"]:
            logger.info("  Fehler:")
            for error_type, count in self.metrics["errors"].items():
                logger.info(f"    {error_type}: {count}")
    
    async def log_metrics(self):
        while True:
            await asyncio.sleep(self.log_interval)
            
            # Zeitbasierte Bereinigung
            current_time = time.time()
            if current_time - self.last_cleanup_time > 300:  # Alle 5 Minuten
                self._cleanup_old_metrics()
                self.last_cleanup_time = current_time
            
            # Reduziere Logging bei hoher Last
            if self.metrics["packets_received"] > 10000:  # Hohe Last
                self._log_critical_metrics()
            else:
                self._log_all_metrics()

# Globale Metriken-Instanz
metrics = MetricsCollector()

# VOICE ACTIVITY DETECTION
class SimpleVAD:
    def __init__(self, energy_threshold=VAD_ENERGY_THRESHOLD, 
                 speech_duration_threshold=VAD_SPEECH_DURATION_THRESHOLD, 
                 silence_duration_threshold=VAD_SILENCE_DURATION_THRESHOLD):
        self.energy_threshold = energy_threshold
        self.speech_duration_threshold = speech_duration_threshold
        self.silence_duration_threshold = silence_duration_threshold
        self.is_speaking = False
        self.speech_start_time = None
        self.silence_start_time = None
        self.background_energy = None
        self.adaptation_rate = 0.05  # Für adaptive Schwellenwertanpassung
        
    def update_background_energy(self, current_energy):
        if self.background_energy is None:
            self.background_energy = current_energy
        else:
            # Langsame Anpassung an Hintergrundgeräusche
            self.background_energy = (1 - self.adaptation_rate) * self.background_energy + self.adaptation_rate * current_energy
            
    def calculate_energy(self, audio_base64):
        try:
            # Base64 dekodieren
            audio_bytes = base64.b64decode(audio_base64)
            
            # μ-law 8000 Hz Format (8-bit pro Sample)
            format_str = f"{len(audio_bytes)}B"  # 'B' für unsigned char (8-bit)
            samples = struct.unpack(format_str, audio_bytes)
            
            # μ-law Dekodierung und Normalisierung
            normalized_samples = [mulaw_decode(s) for s in samples]
            
            # Berechne RMS-Energie
            if not samples:
                return 0
            
            energy = sum(s*s for s in normalized_samples) / len(normalized_samples)
            return energy
            
        except Exception as e:
            logger.warning(f"Fehler bei Energieberechnung: {e}")
            return 0
        
    def is_voice_active(self, audio_base64, current_time):
        energy = self.calculate_energy(audio_base64)
        
        # Aktualisieren des Hintergrundgeräuschpegels, wenn keine Sprache erkannt wird
        if not self.is_speaking:
            self.update_background_energy(energy)
            
        # Dynamischer Schwellenwert basierend auf Hintergrundgeräuschen
        dynamic_threshold = max(self.energy_threshold, 
                               self.background_energy * 1.3 if self.background_energy else self.energy_threshold)
        
        if energy > dynamic_threshold:
            if not self.is_speaking:
                if self.speech_start_time is None:
                    self.speech_start_time = current_time
                
                if current_time - self.speech_start_time >= self.speech_duration_threshold:
                    self.is_speaking = True
                    self.silence_start_time = None
                    metrics.increment("vad_triggers")
                    return True
            else:
                # Bereits sprechend, setze Silence-Timer zurück
                self.silence_start_time = None
                return True
        else:
            if self.is_speaking:
                if self.silence_start_time is None:
                    self.silence_start_time = current_time
                elif current_time - self.silence_start_time >= self.silence_duration_threshold:
                    self.is_speaking = False
                    self.speech_start_time = None
                    return False
                return True  # Noch als sprechend betrachten während der Silence-Periode
            else:
                self.speech_start_time = None
                    
        return self.is_speaking

# INTERRUPTION CLASSIFIER
class InterruptionClassifier:
    def __init__(self):
        self.backchanneling_duration_threshold = 1.0  # Sekunden
        self.backchanneling_energy_ratio = 0.7  # Im Vergleich zur durchschnittlichen Sprachenergie
        self.average_speech_energy = 0.1  # Startwert, wird dynamisch angepasst
        self.speech_energy_samples = []
        
    def update_average_speech_energy(self, energy):
        self.speech_energy_samples.append(energy)
        if len(self.speech_energy_samples) > 20:  # Behalte die letzten 20 Samples
            self.speech_energy_samples.pop(0)
        
        if self.speech_energy_samples:
            self.average_speech_energy = sum(self.speech_energy_samples) / len(self.speech_energy_samples)
        
    def classify_interruption(self, duration, energy, agent_speaking_duration, agent_in_pause):
        # Aktualisiere durchschnittliche Sprachenergie
        self.update_average_speech_energy(energy)
        
        # Kurze Äußerung mit niedriger Energie während Agent-Pause = Backchanneling
        if duration < self.backchanneling_duration_threshold and energy < self.average_speech_energy * self.backchanneling_energy_ratio:
            if agent_in_pause or agent_speaking_duration > 5.0:  # Nach 5 Sekunden Agentensprache
                return "BACKCHANNELING"
        
        # Längere oder energiereichere Äußerung = echte Unterbrechung
        return "INTERRUPTION"

# KONTEXT-MANAGER FÜR ELEVENLABS
class ContextManager:
    def __init__(self):
        self.current_context_id = None
        self.is_agent_speaking = False
        self.agent_speaking_start_time = None
        self.agent_in_pause = False
        
    @property
    def agent_speaking_duration(self):
        if not self.is_agent_speaking or self.agent_speaking_start_time is None:
            return 0
        return time.time() - self.agent_speaking_start_time
    
    async def start_new_context(self, ws):
        # Generiere eine neue Kontext-ID
        self.current_context_id = f"context_{uuid.uuid4().hex}"
        
        try:
            # Sende Kontext-Initialisierungsnachricht
            await ws.send(json.dumps({
                "type": "context_control",
                "context_control": {
                    "action": "create",
                    "context_id": self.current_context_id
                }
            }))
            
            metrics.increment("context_switches")
            logger.info(f"Neuer Kontext erstellt: {self.current_context_id}")
            return self.current_context_id
        except Exception as e:
            logger.error(f"Fehler beim Erstellen eines neuen Kontexts: {e}")
            metrics.increment("context_control_failures")
            self.current_context_id = None
            return None
    
    async def handle_interruption(self, ws):
        if self.is_agent_speaking and self.current_context_id:
            logger.info(f"Unterbrechung erkannt, stoppe Kontext {self.current_context_id}")
            
            try:
                # Primäre Methode: Kontext-Steuerung
                await ws.send(json.dumps({
                    "type": "context_control",
                    "context_control": {
                        "action": "abort",
                        "context_id": self.current_context_id
                    }
                }))
                
                # Markiere Agent als nicht mehr sprechend
                self.is_agent_speaking = False
                self.agent_speaking_start_time = None
                self.agent_in_pause = False
                
                metrics.increment("interruptions_detected")
                
                # Starte einen neuen Kontext
                await self.start_new_context(ws)
            except Exception as e:
                logger.error(f"Fehler bei Kontext-Steuerung (abort): {e}")
                metrics.increment("context_control_failures")
                
                try:
                    # Fallback: Standard-Interruption
                    logger.warning("Verwende Fallback-Interruption")
                    await ws.send(json.dumps({
                        "type": "interruption"
                    }))
                    
                    # Markiere Agent als nicht mehr sprechend
                    self.is_agent_speaking = False
                    self.agent_speaking_start_time = None
                    self.agent_in_pause = False
                    
                    metrics.increment("interruptions_detected")
                    
                    # Starte einen neuen Kontext
                    await self.start_new_context(ws)
                except Exception as e2:
                    logger.error(f"Auch Fallback-Interruption fehlgeschlagen: {e2}")
                    metrics.record_error("InterruptionError", str(e2))
    
    async def handle_backchanneling(self, ws):
        if self.is_agent_speaking and self.current_context_id and not self.agent_in_pause:
            logger.info(f"Backchanneling erkannt, pausiere Kontext {self.current_context_id}")
            
            try:
                # Sende Pausensignal
                await ws.send(json.dumps({
                    "type": "context_control",
                    "context_control": {
                        "action": "pause",
                        "context_id": self.current_context_id
                    }
                }))
                
                self.agent_in_pause = True
                metrics.increment("backchanneling_detected")
                
                # Kurze Pause einlegen
                await asyncio.sleep(0.5)
                
                # Fortsetzen, wenn kein Abbruch erfolgt ist
                if self.agent_in_pause and self.is_agent_speaking:
                    logger.info(f"Setze Kontext fort: {self.current_context_id}")
                    await ws.send(json.dumps({
                        "type": "context_control",
                        "context_control": {
                            "action": "resume",
                            "context_id": self.current_context_id
                        }
                    }))
                    self.agent_in_pause = False
            except Exception as e:
                logger.error(f"Fehler bei Kontext-Steuerung (pause/resume): {e}")
                metrics.increment("context_control_failures")
                # Bei Fehlern setzen wir den Pause-Status zurück
                self.agent_in_pause = False
    
    def agent_started_speaking(self):
        if not self.is_agent_speaking:
            self.is_agent_speaking = True
            self.agent_speaking_start_time = time.time()
            self.agent_in_pause = False
    
    def agent_stopped_speaking(self):
        self.is_agent_speaking = False
        self.agent_speaking_start_time = None
        self.agent_in_pause = False

# RESILIENT WEBSOCKET CLIENT
class ResilientWebSocketClient:
    def __init__(self, url_provider, max_retries=5, initial_backoff=1.0, max_backoff=60.0):
        self.url_provider = url_provider  # Funktion, die die URL zurückgibt
        self.ws = None
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.max_backoff = max_backoff
        self.connection_state = "DISCONNECTED"
        self.retry_count = 0
        self.last_activity = None
        self.on_reconnect_callback = None
        
    async def connect(self):
        self.connection_state = "CONNECTING"
        backoff = self.initial_backoff
        
        while self.retry_count < self.max_retries:
            try:
                url = await self.url_provider()
                if not url:
                    raise ValueError("Konnte keine gültige URL erhalten")
                
                # Sicherstellen, dass die URL mit wss:// beginnt
                if not url.startswith("wss://"):
                    logger.warning(f"Unsichere WebSocket-URL: {url}")
                
                # Explizite SSL-Kontext-Konfiguration für maximale Sicherheit
                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = True
                ssl_context.verify_mode = ssl.CERT_REQUIRED
                
                self.ws = await websockets.connect(url, ssl=ssl_context if url.startswith("wss://") else None)
                self.connection_state = "CONNECTED"
                self.retry_count = 0
                self.last_activity = time.time()
                logger.info(f"WebSocket-Verbindung hergestellt")
                
                # Starte Heartbeat-Task
                asyncio.create_task(self._heartbeat())
                
                # Wenn es eine Wiederverbindung war, rufe Callback auf
                if self.on_reconnect_callback and self.retry_count > 0:
                    metrics.increment("successful_reconnections")
                    await self.on_reconnect_callback(self.ws)
                
                return self.ws
                
            except Exception as e:
                self.connection_state = "RECONNECTING"
                self.retry_count += 1
                metrics.increment("reconnection_attempts")
                metrics.record_error("WebSocketConnectionError", str(e))
                logger.warning(f"Verbindungsfehler (Versuch {self.retry_count}/{self.max_retries}): {e}")
                
                # Exponentielles Backoff
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, self.max_backoff)
        
        self.connection_state = "FAILED"
        logger.error(f"Maximale Anzahl an Wiederverbindungsversuchen erreicht ({self.max_retries})")
        raise ConnectionError(f"Konnte nach {self.max_retries} Versuchen keine Verbindung herstellen")
    
    async def send(self, message):
        if not self.ws or self.connection_state != "CONNECTED":
            await self.connect()
        
        try:
            await self.ws.send(message)
            self.last_activity = time.time()
            metrics.increment("packets_sent")
        except Exception as e:
            logger.error(f"Fehler beim Senden: {e}")
            metrics.record_error("WebSocketSendError", str(e))
            await self.connect()  # Versuche erneut zu verbinden
            await self.ws.send(message)  # Versuche erneut zu senden
    
    async def receive(self):
        if not self.ws or self.connection_state != "CONNECTED":
            await self.connect()
        
        try:
            message = await self.ws.recv()
            self.last_activity = time.time()
            metrics.increment("packets_received")
            return message
        except Exception as e:
            logger.error(f"Fehler beim Empfangen: {e}")
            metrics.record_error("WebSocketReceiveError", str(e))
            await self.connect()  # Versuche erneut zu verbinden
            return await self.ws.recv()  # Versuche erneut zu empfangen
    
    async def _heartbeat(self):
        while self.connection_state == "CONNECTED":
            try:
                # Sende Ping alle 30 Sekunden, wenn keine Aktivität
                current_time = time.time()
                if self.last_activity and current_time - self.last_activity > 30:
                    await self.ws.ping()
                    logger.debug("Heartbeat-Ping gesendet")
                
                await asyncio.sleep(10)  # Prüfe alle 10 Sekunden
            except Exception as e:
                logger.warning(f"Heartbeat-Fehler: {e}")
                metrics.record_error("HeartbeatError", str(e))
                break
        
        # Wenn wir hier ankommen, ist die Verbindung unterbrochen
        if self.connection_state != "FAILED":
            self.connection_state = "RECONNECTING"
            asyncio.create_task(self.connect())

# WRAPPER FÜR BESTEHENDE WEBSOCKET-VERBINDUNG
class ResilientWebSocketWrapper:
    def __init__(self, ws):
        self.ws = ws
        self.last_activity = time.time()
    
    async def send(self, message):
        try:
            await self.ws.send(message)
            self.last_activity = time.time()
            metrics.increment("packets_sent")
        except Exception as e:
            logger.error(f"Fehler beim Senden an Talkdesk: {e}")
            metrics.record_error("TalkdeskSendError", str(e))
            raise  # Hier können wir nicht viel tun, da die Verbindung extern verwaltet wird
    
    async def receive(self):
        try:
            message = await self.ws.recv()
            self.last_activity = time.time()
            metrics.increment("packets_received")
            return message
        except Exception as e:
            logger.error(f"Fehler beim Empfangen von Talkdesk: {e}")
            metrics.record_error("TalkdeskReceiveError", str(e))
            raise  # Hier können wir nicht viel tun, da die Verbindung extern verwaltet wird

# ADAPTIVE BUFFER FÜR AUDIO-STREAMING
class AdaptiveBuffer:
    def __init__(self, initial_size=3, min_size=2, max_size=10):
        self.buffer = []
        self.target_size = initial_size
        self.min_size = min_size
        self.max_size = max_size
        self.network_quality = 1.0  # 0.0 (schlecht) bis 1.0 (ausgezeichnet)
        
    def update_network_quality(self, packet_loss_rate, jitter):
        # Berechnung der Netzwerkqualität basierend auf Paketverlust und Jitter
        self.network_quality = max(0.0, min(1.0, 1.0 - packet_loss_rate - (jitter / 100.0)))
        
        # Anpassung der Ziel-Puffergröße
        if self.network_quality > 0.8:
            self.target_size = self.min_size
        elif self.network_quality < 0.4:
            self.target_size = self.max_size
        else:
            # Lineare Interpolation zwischen min und max
            quality_factor = (self.network_quality - 0.4) / 0.4
            self.target_size = int(self.max_size - quality_factor * (self.max_size - self.min_size))
    
    async def add_packet(self, packet):
        self.buffer.append(packet)
        
        # Wenn der Puffer die Zielgröße überschreitet, verarbeite Pakete
        if len(self.buffer) >= self.target_size:
            return self.get_packets()
        return None
    
    def get_packets(self):
        packets = self.buffer.copy()
        self.buffer.clear()
        return packets

# SANITIZE LOG FUNCTION
def sanitize_log(message, sensitive_fields=["xi-api-key", "api_key", "password"]):
    """Maskiert sensible Daten in Log-Nachrichten."""
    if isinstance(message, dict):
        sanitized = message.copy()
        for field in sensitive_fields:
            if field in sanitized:
                sanitized[field] = "***REDACTED***"
        return sanitized
    elif isinstance(message, str):
        # Einfache Regex-basierte Maskierung für bekannte Muster
        for field in sensitive_fields:
            pattern = f'["\']?{field}["\']?\\s*[:=]\\s*["\']([^"\']+)["\']'
            message = re.sub(pattern, f'"{field}":"***REDACTED***"', message)
        return message
    return message

# BUILD CONFIG FÜR ELEVENLABS
def build_initial_config():
    config = {"type": "conversation_initiation_client_data"}
    if POC_PROMPT:
        config["conversation_config_override"] = {"agent": {"prompt": {"prompt": POC_PROMPT}}}
    if POC_FIRST_MESSAGE:
        config.setdefault("conversation_config_override", {}).setdefault("agent", {})["first_message"] = POC_FIRST_MESSAGE
    return config

# SIGNED URL LADEN
async def get_elevenlabs_signed_url():
    url = f"https://api.elevenlabs.io/v1/convai/conversation/get_signed_url?agent_id={ELEVENLABS_AGENT_ID}"
    headers = {'xi-api-key': ELEVENLABS_API_KEY}
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(url, headers=headers)
            r.raise_for_status()
            signed_url = r.json().get("signed_url")
            
            # Sicherstellen, dass die URL mit wss:// beginnt
            if signed_url and not signed_url.startswith("wss://"):
                logger.warning(f"Unsichere WebSocket-URL erhalten: {signed_url}")
                
            return signed_url
    except Exception as e:
        logger.error(f"Signed URL Fehler: {e}")
        metrics.record_error("SignedURLError", str(e))
        return None

# AUDIO BRIDGING MIT BARGE-IN UNTERSTÜTZUNG
async def stream_talkdesk_to_elevenlabs(td_ws, el_ws, context_manager):
    vad = SimpleVAD()
    interruption_classifier = InterruptionClassifier()
    speech_start_time = None
    network_monitor = NetworkQualityMonitor()
    adaptive_buffer = AdaptiveBuffer()
    last_packet_time = None
    
    try:
        async for message_str in td_ws:
            data = json.loads(message_str)
            if data.get("event") == "media":
                payload = data.get("media", {}).get("payload")
                if payload:
                    metrics.increment("audio_chunks_processed")
                    current_time = time.time()
                    
                    # Netzwerkqualität messen
                    if last_packet_time:
                        network_monitor.record_packet(current_time)
                    last_packet_time = current_time
                    
                    # Pufferverwaltung
                    adaptive_buffer.update_network_quality(
                        network_monitor.packet_loss_rate,
                        network_monitor.jitter
                    )
                    
                    buffered_packets = await adaptive_buffer.add_packet(payload)
                    
                    # Wenn Puffer voll ist, verarbeite alle Pakete
                    if buffered_packets:
                        for packet in buffered_packets:
                            # VAD-Prüfung
                            if vad.is_voice_active(packet, current_time):
                                if speech_start_time is None:
                                    speech_start_time = current_time
                                
                                # Wenn Agent spricht, klassifiziere die Unterbrechung
                                if context_manager.is_agent_speaking:
                                    speech_duration = current_time - speech_start_time
                                    energy = vad.calculate_energy(packet)
                                    
                                    interruption_type = interruption_classifier.classify_interruption(
                                        speech_duration, 
                                        energy,
                                        context_manager.agent_speaking_duration,
                                        context_manager.agent_in_pause
                                    )
                                    
                                    if interruption_type == "INTERRUPTION":
                                        # Harte Unterbrechung - Stoppe den Agenten und erstelle neuen Kontext
                                        await context_manager.handle_interruption(el_ws)
                                    elif interruption_type == "BACKCHANNELING" and not context_manager.agent_in_pause:
                                        # Weiche Unterbrechung - Pausiere kurz, setze dann fort
                                        await context_manager.handle_backchanneling(el_ws)
                            
                            # Audio an ElevenLabs senden
                            await el_ws.send(json.dumps({"user_audio_chunk": packet}))
                            
                            # Reset, wenn keine Sprache mehr erkannt wird
                            if not vad.is_speaking and speech_start_time is not None:
                                speech_start_time = None
                    else:
                        # Wenn Puffer noch nicht voll, sende direkt
                        # VAD-Prüfung
                        if vad.is_voice_active(payload, current_time):
                            if speech_start_time is None:
                                speech_start_time = current_time
                            
                            # Wenn Agent spricht, klassifiziere die Unterbrechung
                            if context_manager.is_agent_speaking:
                                speech_duration = current_time - speech_start_time
                                energy = vad.calculate_energy(payload)
                                
                                interruption_type = interruption_classifier.classify_interruption(
                                    speech_duration, 
                                    energy,
                                    context_manager.agent_speaking_duration,
                                    context_manager.agent_in_pause
                                )
                                
                                if interruption_type == "INTERRUPTION":
                                    # Harte Unterbrechung - Stoppe den Agenten und erstelle neuen Kontext
                                    await context_manager.handle_interruption(el_ws)
                                elif interruption_type == "BACKCHANNELING" and not context_manager.agent_in_pause:
                                    # Weiche Unterbrechung - Pausiere kurz, setze dann fort
                                    await context_manager.handle_backchanneling(el_ws)
                        
                        # Audio an ElevenLabs senden
                        await el_ws.send(json.dumps({"user_audio_chunk": payload}))
                        
                        # Reset, wenn keine Sprache mehr erkannt wird
                        if not vad.is_speaking and speech_start_time is not None:
                            speech_start_time = None
    except Exception as e:
        logger.error(f"Fehler in stream_talkdesk_to_elevenlabs: {e}", exc_info=True)
        metrics.record_error("TalkdeskToElevenLabsError", str(e))

async def stream_elevenlabs_to_talkdesk(td_ws, el_ws, stream_sid, context_manager):
    try:
        async for message_str in el_ws:
            data = json.loads(message_str)
            
            # Verarbeite Audio-Events
            if data.get("type") == "audio":
                b64_audio = data.get("audio_event", {}).get("audio_base_64")
                if b64_audio:
                    # Markiere Agent als sprechend
                    context_manager.agent_started_speaking()
                    
                    # Sende Audio an Talkdesk
                    await td_ws.send(json.dumps({
                        "event": "media",
                        "streamSid": stream_sid,
                        "media": {"payload": b64_audio}
                    }))
            
            # Verarbeite Kontext-Events
            elif data.get("type") == "context_control_response":
                control_response = data.get("context_control_response", {})
                action = control_response.get("action")
                success = control_response.get("success", False)
                
                if action == "abort" and success:
                    logger.info("Kontext erfolgreich abgebrochen")
                    context_manager.agent_stopped_speaking()
                elif action == "pause" and success:
                    logger.info("Kontext erfolgreich pausiert")
                elif action == "resume" and success:
                    logger.info("Kontext erfolgreich fortgesetzt")
            
            # Verarbeite End-Events
            elif data.get("type") == "end":
                logger.info("Ende des Streams von ElevenLabs")
                context_manager.agent_stopped_speaking()
    except Exception as e:
        logger.error(f"Fehler in stream_elevenlabs_to_talkdesk: {e}", exc_info=True)
        metrics.record_error("ElevenLabsToTalkdeskError", str(e))

# REFAKTORIERTER CONNECTION HANDLER
async def handle_connection(talkdesk_ws):
    remote = f"{talkdesk_ws.remote_address[0]}:{talkdesk_ws.remote_address[1]}"
    logger.info(f"Neue Verbindung von {remote}")

    try:
        stream_sid = None
        # START-NACHRICHT PARSEN
        for _ in range(5):
            try:
                msg = await asyncio.wait_for(talkdesk_ws.recv(), timeout=10)
                data = json.loads(msg)
                if data.get("event") == "start":
                    stream_sid = data["start"].get("streamSid")
                    break
            except asyncio.TimeoutError:
                logger.warning("Timeout beim Warten auf Start-Nachricht")
                continue
            except Exception as e:
                logger.error(f"Fehler beim Parsen der Start-Nachricht: {e}")
                continue
                
        if not stream_sid:
            logger.error("Kein StreamSid erhalten. Abbruch.")
            return

        # Resiliente Wrapper für WebSockets erstellen
        resilient_td_ws = ResilientWebSocketWrapper(talkdesk_ws)
        
        # ElevenLabs WebSocket mit Wiederverbindungslogik
        signed_url = await get_elevenlabs_signed_url()
        if not signed_url:
            logger.error("Konnte keine Signed URL erhalten. Abbruch.")
            return
            
        resilient_el_ws = ResilientWebSocketClient(lambda: asyncio.create_task(get_elevenlabs_signed_url()))
        el_ws = await resilient_el_ws.connect()

        # Initialisierung senden
        initial_config = build_initial_config()
        await resilient_el_ws.send(json.dumps(initial_config))
        
        # Kontext-Manager initialisieren
        context_manager = ContextManager()
        await context_manager.start_new_context(resilient_el_ws)

        # TASKS STARTEN
        task1 = asyncio.create_task(stream_talkdesk_to_elevenlabs(resilient_td_ws, resilient_el_ws, context_manager))
        task2 = asyncio.create_task(stream_elevenlabs_to_talkdesk(resilient_td_ws, resilient_el_ws, stream_sid, context_manager))
        task3 = asyncio.create_task(metrics.log_metrics())
        
        await asyncio.wait([task1, task2], return_when=asyncio.FIRST_COMPLETED)

    except Exception as e:
        logger.error(f"Fehler in Handler: {e}", exc_info=True)
        metrics.record_error("ConnectionHandlerError", str(e))
    finally:
        logger.info("Verbindung wird geschlossen")
        if talkdesk_ws and talkdesk_ws.open:
            await talkdesk_ws.close()

# SERVER START
async def main():
    logger.info(f"Starte WebSocket-Server auf {WEBSOCKET_HOST}:{WEBSOCKET_PORT}")
    
    # Starte Metriken-Logging
    asyncio.create_task(metrics.log_metrics())
    
    try:
        async with websockets.serve(handle_connection, WEBSOCKET_HOST, WEBSOCKET_PORT):
            await asyncio.Future()  # Laufe für immer
    except Exception as e:
        logger.error(f"Fehler beim Starten des Servers: {e}", exc_info=True)
        metrics.record_error("ServerStartError", str(e))

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Manueller Abbruch durch User.")
    except Exception as e:
        logger.error(f"Unbehandelter Fehler: {e}", exc_info=True)
        metrics.record_error("UnhandledError", str(e))

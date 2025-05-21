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

SCRIPT_VERSION = "5.2 - Enhanced Barge-in Support + Debug Logging"
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
VAD_ENERGY_THRESHOLD = float(os.environ.get("VAD_ENERGY_THRESHOLD", "0.01"))  # Niedrigerer Schwellenwert für bessere Erkennung
VAD_SPEECH_DURATION_THRESHOLD = float(os.environ.get("VAD_SPEECH_DURATION_THRESHOLD", "0.1"))  # Kürzere Erkennungszeit
VAD_SILENCE_DURATION_THRESHOLD = float(os.environ.get("VAD_SILENCE_DURATION_THRESHOLD", "0.5"))
LOG_INTERVAL = int(os.environ.get("LOG_INTERVAL", "300"))  # 5 Minuten
DEBUG_LOGGING = os.environ.get("DEBUG_LOGGING", "true").lower() == "true"  # Detaillierte Logs aktivieren

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

# METRIKEN-SAMMLUNG
class MetricsCollector:
    def __init__(self, log_interval=LOG_INTERVAL):
        self.metrics = {
            "packets_received": 0,
            "packets_sent": 0,
            "audio_chunks_processed": 0,
            "interruptions_detected": 0,
            "backchanneling_detected": 0,
            "reconnection_attempts": 0,
            "successful_reconnections": 0,
            "errors": {},
            "vad_triggers": 0,
            "context_switches": 0,
            "context_control_failures": 0
        }
        self.log_interval = log_interval
        self.start_time = time.time()
    
    def increment(self, metric, value=1):
        if metric in self.metrics:
            self.metrics[metric] += value
    
    def record_error(self, error_type, error_message):
        if error_type not in self.metrics["errors"]:
            self.metrics["errors"][error_type] = 0
        self.metrics["errors"][error_type] += 1
        logger.error(f"{error_type}: {error_message}")
    
    def _log_metrics(self):
        """Loggt alle Metriken."""
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
            self._log_metrics()

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
        self.last_energy_log_time = 0
        self.energy_log_interval = 1.0  # Energielevel jede Sekunde loggen
        
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
        
        # Debug-Logging für Energielevel (begrenzt auf Intervalle)
        if DEBUG_LOGGING and current_time - self.last_energy_log_time > self.energy_log_interval:
            self.last_energy_log_time = current_time
            logger.info(f"VAD: Energie={energy:.6f}, Schwelle={self.energy_threshold:.6f}, Hintergrund={self.background_energy:.6f if self.background_energy else 0:.6f}")
        
        # Aktualisieren des Hintergrundgeräuschpegels, wenn keine Sprache erkannt wird
        if not self.is_speaking:
            self.update_background_energy(energy)
            
        # Dynamischer Schwellenwert basierend auf Hintergrundgeräuschen
        dynamic_threshold = max(self.energy_threshold, 
                               self.background_energy * 1.2 if self.background_energy else self.energy_threshold)  # Niedrigerer Multiplikator
        
        if energy > dynamic_threshold:
            if not self.is_speaking:
                if self.speech_start_time is None:
                    self.speech_start_time = current_time
                
                if current_time - self.speech_start_time >= self.speech_duration_threshold:
                    self.is_speaking = True
                    self.silence_start_time = None
                    metrics.increment("vad_triggers")
                    logger.info(f"VAD: Sprache erkannt! Energie={energy:.6f}, Schwelle={dynamic_threshold:.6f}")
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
                    logger.info(f"VAD: Sprache beendet. Stille für {self.silence_duration_threshold}s")
                    return False
                return True  # Noch als sprechend betrachten während der Silence-Periode
            else:
                self.speech_start_time = None
                    
        return self.is_speaking

# INTERRUPTION CLASSIFIER
class InterruptionClassifier:
    def __init__(self):
        self.backchanneling_duration_threshold = 0.8  # Sekunden (kürzer für schnellere Erkennung)
        self.backchanneling_energy_ratio = 0.8  # Im Vergleich zur durchschnittlichen Sprachenergie (höher für aggressivere Erkennung)
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
        
        # Debug-Logging
        if DEBUG_LOGGING:
            logger.info(f"Interruption-Klassifikation: Dauer={duration:.2f}s, Energie={energy:.6f}, Agent-Sprechdauer={agent_speaking_duration:.2f}s")
        
        # Kurze Äußerung mit niedriger Energie während Agent-Pause = Backchanneling
        if duration < self.backchanneling_duration_threshold and energy < self.average_speech_energy * self.backchanneling_energy_ratio:
            if agent_in_pause or agent_speaking_duration > 3.0:  # Nach 3 Sekunden Agentensprache (früher reagieren)
                logger.info(f"Backchanneling erkannt: Dauer={duration:.2f}s, Energie={energy:.6f}")
                return "BACKCHANNELING"
        
        # Längere oder energiereichere Äußerung = echte Unterbrechung
        logger.info(f"Unterbrechung erkannt: Dauer={duration:.2f}s, Energie={energy:.6f}")
        return "INTERRUPTION"

# KONTEXT-MANAGER FÜR ELEVENLABS
class ContextManager:
    def __init__(self):
        self.current_context_id = None
        self.is_agent_speaking = False
        self.agent_speaking_start_time = None
        self.agent_in_pause = False
        self.last_audio_time = 0
        self.audio_timeout = 0.5  # Wenn 0.5s kein Audio, dann spricht der Agent nicht mehr
        
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
            
            # Sende beide Signale gleichzeitig für maximale Wirkung
            try:
                # Sende direkte Interruption (einfacher und schneller)
                await ws.send(json.dumps({
                    "type": "interruption"
                }))
                logger.info("Direktes Interruption-Signal gesendet")
                
                # Sende auch Kontext-Steuerung (für vollständige Kontrolle)
                await ws.send(json.dumps({
                    "type": "context_control",
                    "context_control": {
                        "action": "abort",
                        "context_id": self.current_context_id
                    }
                }))
                logger.info("Kontext-Abort-Signal gesendet")
                
                # Markiere Agent als nicht mehr sprechend
                self.is_agent_speaking = False
                self.agent_speaking_start_time = None
                self.agent_in_pause = False
                
                metrics.increment("interruptions_detected")
                
                # Starte einen neuen Kontext
                await self.start_new_context(ws)
            except Exception as e:
                logger.error(f"Fehler bei Interruption-Signalen: {e}")
                metrics.increment("context_control_failures")
                metrics.record_error("InterruptionError", str(e))
    
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
        current_time = time.time()
        self.last_audio_time = current_time
        
        if not self.is_agent_speaking:
            self.is_agent_speaking = True
            self.agent_speaking_start_time = current_time
            self.agent_in_pause = False
            logger.info("Agent hat begonnen zu sprechen")
    
    def agent_stopped_speaking(self):
        if self.is_agent_speaking:
            logger.info(f"Agent hat aufgehört zu sprechen (nach {self.agent_speaking_duration:.2f}s)")
        
        self.is_agent_speaking = False
        self.agent_speaking_start_time = None
        self.agent_in_pause = False
    
    def update_speaking_status(self, current_time):
        """Aktualisiert den Sprechstatus basierend auf der Zeit seit dem letzten Audio-Chunk"""
        if self.is_agent_speaking and current_time - self.last_audio_time > self.audio_timeout:
            logger.info(f"Kein Audio vom Agent seit {self.audio_timeout}s - setze Status auf 'nicht sprechend'")
            self.agent_stopped_speaking()

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
async def stream_talkdesk_to_elevenlabs(websocket, el_ws, context_manager):
    vad = SimpleVAD()
    interruption_classifier = InterruptionClassifier()
    speech_start_time = None
    last_status_check = time.time()
    status_check_interval = 0.5  # Alle 0.5 Sekunden den Sprechstatus prüfen
    
    try:
        while True:
            try:
                # Direkte Verwendung des Websocket-Objekts ohne Wrapper
                message_str = await websocket.recv()
                data = json.loads(message_str)
                
                current_time = time.time()
                
                # Regelmäßige Überprüfung des Agent-Sprechstatus
                if current_time - last_status_check > status_check_interval:
                    context_manager.update_speaking_status(current_time)
                    last_status_check = current_time
                
                if data.get("event") == "media":
                    payload = data.get("media", {}).get("payload")
                    if payload:
                        metrics.increment("audio_chunks_processed")
                        
                        # VAD-Prüfung
                        if vad.is_voice_active(payload, current_time):
                            if speech_start_time is None:
                                speech_start_time = current_time
                                logger.info("Benutzersprache erkannt")
                            
                            # Wenn Agent spricht, klassifiziere die Unterbrechung
                            if context_manager.is_agent_speaking:
                                speech_duration = current_time - speech_start_time
                                energy = vad.calculate_energy(payload)
                                
                                logger.info(f"Potenzielle Unterbrechung: Benutzer spricht seit {speech_duration:.2f}s, Agent spricht seit {context_manager.agent_speaking_duration:.2f}s")
                                
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
                        metrics.increment("packets_sent")
                        
                        # Reset, wenn keine Sprache mehr erkannt wird
                        if not vad.is_speaking and speech_start_time is not None:
                            logger.info(f"Benutzer hat aufgehört zu sprechen (nach {current_time - speech_start_time:.2f}s)")
                            speech_start_time = None
            except ConnectionClosed:
                logger.info("Talkdesk WebSocket-Verbindung geschlossen")
                break
            except Exception as e:
                logger.error(f"Fehler beim Verarbeiten der Talkdesk-Nachricht: {e}")
                metrics.record_error("TalkdeskMessageError", str(e))
                # Versuche weiterzumachen
                await asyncio.sleep(0.1)
    except Exception as e:
        logger.error(f"Fehler in stream_talkdesk_to_elevenlabs: {e}", exc_info=True)
        metrics.record_error("TalkdeskToElevenLabsError", str(e))

async def stream_elevenlabs_to_talkdesk(websocket, el_ws, stream_sid, context_manager):
    try:
        while True:
            try:
                # Direkte Verwendung des WebSocket-Objekts
                message_str = await el_ws.recv()
                metrics.increment("packets_received")
                
                data = json.loads(message_str)
                current_time = time.time()
                
                # Verarbeite Audio-Events
                if data.get("type") == "audio":
                    b64_audio = data.get("audio_event", {}).get("audio_base_64")
                    if b64_audio:
                        # Markiere Agent als sprechend und aktualisiere Zeitstempel
                        context_manager.agent_started_speaking()
                        
                        # Sende Audio an Talkdesk
                        await websocket.send(json.dumps({
                            "event": "media",
                            "streamSid": stream_sid,
                            "media": {"payload": b64_audio}
                        }))
                        metrics.increment("packets_sent")
                
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
            except ConnectionClosed:
                logger.info("ElevenLabs WebSocket-Verbindung geschlossen")
                break
            except Exception as e:
                logger.error(f"Fehler beim Verarbeiten der ElevenLabs-Nachricht: {e}")
                metrics.record_error("ElevenLabsMessageError", str(e))
                # Versuche weiterzumachen
                await asyncio.sleep(0.1)
    except Exception as e:
        logger.error(f"Fehler in stream_elevenlabs_to_talkdesk: {e}", exc_info=True)
        metrics.record_error("ElevenLabsToTalkdeskError", str(e))

# REFAKTORIERTER CONNECTION HANDLER
async def handle_connection(websocket):
    remote = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
    logger.info(f"Neue Verbindung von {remote}")

    try:
        stream_sid = None
        # START-NACHRICHT PARSEN
        for _ in range(5):
            try:
                msg = await asyncio.wait_for(websocket.recv(), timeout=10)
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

        # ElevenLabs WebSocket-Verbindung herstellen
        signed_url = await get_elevenlabs_signed_url()
        if not signed_url:
            logger.error("Konnte keine Signed URL erhalten. Abbruch.")
            return
            
        # Direkte WebSocket-Verbindung ohne Wrapper
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = True
        ssl_context.verify_mode = ssl.CERT_REQUIRED
        
        el_ws = await websockets.connect(signed_url, ssl=ssl_context if signed_url.startswith("wss://") else None)
        logger.info("WebSocket-Verbindung hergestellt")

        # Initialisierung senden
        initial_config = build_initial_config()
        await el_ws.send(json.dumps(initial_config))
        metrics.increment("packets_sent")
        
        # Kontext-Manager initialisieren
        context_manager = ContextManager()
        await context_manager.start_new_context(el_ws)

        # TASKS STARTEN
        task1 = asyncio.create_task(stream_talkdesk_to_elevenlabs(websocket, el_ws, context_manager))
        task2 = asyncio.create_task(stream_elevenlabs_to_talkdesk(websocket, el_ws, stream_sid, context_manager))
        task3 = asyncio.create_task(metrics.log_metrics())
        
        # Warte auf das Ende einer der Streaming-Tasks
        done, pending = await asyncio.wait(
            [task1, task2], 
            return_when=asyncio.FIRST_COMPLETED
        )
        
        # Bereinige verbleibende Tasks
        for task in pending:
            task.cancel()
            
        # Warte kurz, damit Logs geschrieben werden können
        await asyncio.sleep(0.5)

    except Exception as e:
        logger.error(f"Fehler in Handler: {e}", exc_info=True)
        metrics.record_error("ConnectionHandlerError", str(e))
    finally:
        logger.info("Verbindung wird geschlossen")
        # Prüfe den Status der WebSocket-Verbindung mit der korrekten API
        if websocket.state == State.OPEN:
            await websocket.close()

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

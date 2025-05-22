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
import io
import wave
from websockets.connection import State
from websockets.exceptions import ConnectionClosed, ConnectionClosedOK, ConnectionClosedError
from datetime import datetime
from typing import Dict, Any, Optional, Callable, List, Union, Set

# ELEVENLABS SPEECH-TO-TEXT
async def process_audio_with_elevenlabs_stt(audio_bytes):
    """Verarbeitet Audio mit der ElevenLabs Speech-to-Text API gemäß offizieller Dokumentation."""
    try:
        # HINZUGEFÜGTES LOGGING:
        logger.info(f"Sende {len(audio_bytes)} Bytes WAV-Daten an ElevenLabs STT.")
        
        # ElevenLabs STT API-Endpunkt
        url = "https://api.elevenlabs.io/v1/speech-to-text"
        
        # Bereite die Audiodaten als Datei für multipart/form-data vor
        files = {
            'file': ('audio.wav', audio_bytes, 'audio/wav')
        }
        
        # Bereite die Form-Daten vor
        form_data = {
            "model_id": "scribe_v1",           # Verwende das Scribe v1 Modell
            "language_code": "de",             # Deutsch (ISO-639-1 Code)
            "timestamps_granularity": "word",  # Wort-Level Timestamps
            "diarize": "false",                # Keine Sprechererkennung nötig für unseren Anwendungsfall
            "tag_audio_events": "false"        # Keine Audio-Events nötig für unseren Anwendungsfall
        }
        
        headers = {
            "xi-api-key": ELEVENLABS_API_KEY,
            # Content-Type wird automatisch von httpx gesetzt
        }
        
        # Sende die Anfrage
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url, 
                files=files, 
                data=form_data,  # Verwende data für Form-Felder
                headers=headers
            )
            response.raise_for_status()
            
            # Verarbeite die Antwort
            result = response.json()
            
            # Extrahiere den Text aus der Antwort
            text = result.get("text", "")
            
            if text:
                logger.info(f"ElevenLabs STT erfolgreich: '{text}'")
                return text
            else:
                logger.warning("ElevenLabs STT lieferte keinen Text zurück")
                return None
                
    except Exception as e:
        logger.error(f"Fehler bei ElevenLabs Speech-to-Text: {e}")
        return None

# SPEECH RECOGNIZER
class SpeechRecognizer:
    def __init__(self):
        self.audio_buffer = bytearray()
        self.last_recognition_time = 0
        self.recognition_interval = 0.5  # Reduziert für schnellere Erkennung
        self.min_buffer_size = 12000  # ERHÖHT von 6000 (ca. 1.5 Sekunden Audio)
        self.max_buffer_size = 32000  # Maximale Puffergröße (ca. 4 Sekunden Audio)
        self.is_processing = False
        
    def add_audio(self, audio_base64):
        """Fügt Audio zum Puffer hinzu."""
        try:
            # Base64 dekodieren und zum Puffer hinzufügen
            audio_bytes = base64.b64decode(audio_base64)
            self.audio_buffer.extend(audio_bytes)
            
            # Puffer auf maximale Größe begrenzen (FIFO)
            if len(self.audio_buffer) > self.max_buffer_size:
                excess = len(self.audio_buffer) - self.max_buffer_size
                self.audio_buffer = self.audio_buffer[excess:]
        except Exception as e:
            logger.error(f"Fehler beim Hinzufügen von Audio zum Puffer: {e}")
    
    async def try_recognize(self, current_time):
        """Versucht, Sprache im Puffer zu erkennen, wenn genug Zeit vergangen ist."""
        if self.is_processing:
            return None
            
        # Prüfe, ob genug Zeit vergangen ist und der Puffer groß genug ist
        if (current_time - self.last_recognition_time >= self.recognition_interval and 
                len(self.audio_buffer) >= self.min_buffer_size):
            
            self.is_processing = True
            self.last_recognition_time = current_time
            
            # Kopiere den aktuellen Puffer für die Verarbeitung
            current_buffer = bytes(self.audio_buffer)
            # Leere den Puffer für neue Daten
            self.audio_buffer.clear()
            
            try:
                # Konvertiere μ-law 8kHz zu WAV für die Spracherkennung
                with io.BytesIO() as wav_io:
                    with wave.open(wav_io, 'wb') as wav_file:
                        wav_file.setnchannels(1)
                        wav_file.setsampwidth(1)  # 8-bit
                        wav_file.setframerate(8000)
                        wav_file.writeframes(current_buffer)
                    
                    wav_io.seek(0)
                    wav_data = wav_io.read()
                
                # Verwende ElevenLabs STT
                result = await process_audio_with_elevenlabs_stt(wav_data)
                return result
            except Exception as e:
                logger.error(f"Fehler bei der Spracherkennung: {e}")
            finally:
                self.is_processing = False
                
        return None

SCRIPT_VERSION = "5.8 - Enhanced Barge-in with Reliable Interruption Signaling and WebSocket Monitoring"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info(f"Starte Agent Server - VERSION {SCRIPT_VERSION}")

# Optional imports for speech recognition
try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
    logger.info("Speech Recognition Modul verfügbar. Textbasierte Unterbrechungserkennung aktiviert.")
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False
    logger.warning("speech_recognition Modul nicht verfügbar. Textbasierte Unterbrechungserkennung deaktiviert.")

# ENV-VARIABLEN
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")
ELEVENLABS_AGENT_ID = os.environ.get("ELEVENLABS_AGENT_ID")
POC_PROMPT = os.environ.get("ELEVENLABS_POC_PROMPT")
POC_FIRST_MESSAGE = os.environ.get("ELEVENLABS_POC_FIRST_MESSAGE")
WEBSOCKET_HOST = os.environ.get("WEBSOCKET_HOST", "0.0.0.0")
WEBSOCKET_PORT = int(os.environ.get("WEBSOCKET_PORT", "8080"))
VAD_ENERGY_THRESHOLD = float(os.environ.get("VAD_ENERGY_THRESHOLD", "0.0008"))  # Drastisch reduzierter Schwellenwert basierend auf tatsächlichen Energiewerten
VAD_SPEECH_DURATION_THRESHOLD = float(os.environ.get("VAD_SPEECH_DURATION_THRESHOLD", "0.1"))  # Kürzere Erkennungszeit
VAD_SILENCE_DURATION_THRESHOLD = float(os.environ.get("VAD_SILENCE_DURATION_THRESHOLD", "0.5"))
LOG_INTERVAL = int(os.environ.get("LOG_INTERVAL", "300"))  # 5 Minuten
DEBUG_LOGGING = os.environ.get("DEBUG_LOGGING", "true").lower() == "true"  # Detaillierte Logs aktivieren
USE_SPEECH_RECOGNITION = os.environ.get("USE_SPEECH_RECOGNITION", "true").lower() == "true"  # Spracherkennung für bessere Unterbrechungserkennung

# Liste von Bestätigungswörtern, die nicht als Unterbrechung behandelt werden sollen
ACKNOWLEDGMENT_WORDS = [
    "genau", "ok", "okay", "alles klar", "verstanden", "ja", "jep", "jup", "mhm", "aha", 
    "gut", "super", "prima", "verstehe", "klar", "richtig", "stimmt", "natürlich", "sicher"
]

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

# SPEECH CONTENT ANALYZER
class SpeechContentAnalyzer:
    def __init__(self):
        self.acknowledgment_words = ACKNOWLEDGMENT_WORDS
        
    def is_acknowledgment(self, text):
        """
        Prüft, ob der Text nur eine Bestätigung ist und keine Frage oder Anweisung.
        """
        # Normalisiere den Text (Kleinbuchstaben, entferne Satzzeichen)
        normalized_text = text.lower().strip()
        for punct in ['.', ',', '!', '?', ';', ':', '-']:
            normalized_text = normalized_text.replace(punct, '')
        
        # Prüfe, ob der Text nur aus einem Wort besteht
        words = normalized_text.split()
        if len(words) == 1:
            return True  # Ein-Wort-Antworten als Zustimmung kategorisieren
        
        if not words:
            return False
            
        # Wenn alle Wörter Bestätigungen sind
        for word in words:
            if word not in self.acknowledgment_words:
                return False
                
        return True
        
    def contains_question(self, text):
        """
        Prüft, ob der Text eine Frage enthält.
        """
        # Einfache Heuristik: Endet mit Fragezeichen oder enthält Fragewörter
        question_words = ["wie", "was", "warum", "weshalb", "wo", "wann", "wer", "welche", "welcher", "welches", "können", "kannst", "würdest", "könntest"]
        
        # Normalisiere den Text
        normalized_text = text.lower()
        
        # Prüfe auf Fragezeichen
        if "?" in normalized_text:
            return True
            
        # Prüfe auf Fragewörter am Anfang
        words = normalized_text.split()
        if words and words[0] in question_words:
            return True
            
        return False

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
        # Reduzierte Schwellenwerte für schnellere Erkennung
        self.speech_duration_threshold = speech_duration_threshold * 0.8  # 20% schnellere Erkennung
        self.silence_duration_threshold = silence_duration_threshold
        self.is_speaking = False
        self.speech_start_time = None
        self.silence_start_time = None
        self.background_energy = None
        self.adaptation_rate = 0.05  # Für adaptive Schwellenwertanpassung
        self.last_energy_log_time = 0
        self.energy_log_interval = 1.0  # Energielevel jede Sekunde loggen
        
        # Zusätzliche Parameter für verbesserte Barge-in-Erkennung
        self.agent_speaking_sensitivity_factor = 0.8  # Noch höhere Empfindlichkeit während Agent spricht (von 0.9 reduziert)
        self.energy_history = []  # Speichert Energiewerte für Trend-Analyse
        self.energy_history_max_size = 10  # Anzahl der zu speichernden Energiewerte
        
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
            
            # Berechne direkte Energie ohne μ-law Dekodierung
            # Dies ist robuster und schneller für Echtzeit-Anwendungen
            raw_energy = sum(abs(s - 128) for s in samples) / len(samples)
            
            # Skalierungsfaktor, um die Energie in einen sinnvollen Bereich zu bringen
            # Basierend auf den beobachteten Werten in den Logs
            energy = raw_energy / 255.0
            
            # Alternative Berechnung mit μ-law Dekodierung als Fallback
            if energy < 0.0001:  # Wenn die direkte Energie sehr niedrig ist
                # μ-law Dekodierung und Normalisierung
                normalized_samples = [mulaw_decode(s) for s in samples]
                energy = sum(s*s for s in normalized_samples) / len(normalized_samples)
            
            # Debug-Logging für sehr niedrige Energiewerte
            if DEBUG_LOGGING and energy > 0.001:
                logger.info(f"Erhöhte Energie erkannt: {energy:.6f} (raw: {raw_energy:.2f})")
                
            return energy
            
        except Exception as e:
            logger.warning(f"Fehler bei Energieberechnung: {e}")
            return 0.0005  # Fallback-Wert basierend auf typischen Hintergrundwerten
        
    def is_voice_active(self, audio_base64, current_time, agent_is_speaking=False):
        energy = self.calculate_energy(audio_base64)
        
        # Speichere Energiewerte für Trend-Analyse
        self.energy_history.append(energy)
        if len(self.energy_history) > self.energy_history_max_size:
            self.energy_history.pop(0)
        
        # Debug-Logging für Energielevel (begrenzt auf Intervalle)
        if DEBUG_LOGGING and current_time - self.last_energy_log_time > self.energy_log_interval:
            self.last_energy_log_time = current_time
            bg_energy_str = f"{self.background_energy:.6f}" if self.background_energy is not None else "0.000000"
            logger.info(f"VAD: Energie={energy:.6f}, Schwelle={self.energy_threshold:.6f}, Hintergrund={bg_energy_str}")
        
        # Aktualisieren des Hintergrundgeräuschpegels, wenn keine Sprache erkannt wird
        if not self.is_speaking:
            self.update_background_energy(energy)
        
        # Dynamischer Schwellenwert basierend auf Hintergrundgeräuschen
        # Wenn Agent spricht, verwende einen niedrigeren Schwellenwert für schnellere Barge-in-Erkennung
        threshold_multiplier = 1.1  # Standard: 10% über Hintergrund
        if agent_is_speaking:
            threshold_multiplier = self.agent_speaking_sensitivity_factor  # Reduzierter Schwellenwert während Agent spricht
            
        dynamic_threshold = max(
            self.energy_threshold * (0.9 if agent_is_speaking else 1.0),  # Niedrigerer absoluter Schwellenwert wenn Agent spricht
            self.background_energy * threshold_multiplier if self.background_energy else self.energy_threshold
        )
        
        # Relative Änderung zum Hintergrund berechnen
        relative_change = energy / self.background_energy if self.background_energy else 1.0
        
        # Trend-Analyse: Prüfe, ob die Energie ansteigt (Indikator für beginnende Sprache)
        energy_trend_rising = False
        if len(self.energy_history) >= 3:
            # Berechne durchschnittliche Änderungsrate der letzten Energiewerte
            recent_changes = [self.energy_history[i] - self.energy_history[i-1] for i in range(1, len(self.energy_history))]
            avg_change = sum(recent_changes) / len(recent_changes)
            energy_trend_rising = avg_change > 0.0001  # Positive Änderungsrate deutet auf ansteigende Energie hin
        
        # Sprache erkennen basierend auf absolutem Schwellenwert ODER relativer Änderung
        # Wenn Agent spricht, niedrigere Schwelle für relative Änderung verwenden
        is_above_threshold = energy > dynamic_threshold
        is_significant_change = relative_change > (1.3 if agent_is_speaking else 1.5)  # Niedrigerer Schwellenwert wenn Agent spricht
        
        if is_above_threshold or is_significant_change:
            if not self.is_speaking:
                if self.speech_start_time is None:
                    self.speech_start_time = current_time
                
                if current_time - self.speech_start_time >= self.speech_duration_threshold:
                    self.is_speaking = True
                    self.silence_start_time = None
                    metrics.increment("vad_triggers")
                    logger.info(f"VAD: Sprache erkannt! Energie={energy:.6f}, Schwelle={dynamic_threshold:.6f}, Rel.Änderung={relative_change:.2f}")
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
        self.speech_content_analyzer = SpeechContentAnalyzer()
        self.last_speech_text = ""
        
    def update_average_speech_energy(self, energy):
        self.speech_energy_samples.append(energy)
        if len(self.speech_energy_samples) > 20:  # Behalte die letzten 20 Samples
            self.speech_energy_samples.pop(0)
        
        if self.speech_energy_samples:
            self.average_speech_energy = sum(self.speech_energy_samples) / len(self.speech_energy_samples)
    
    def set_last_speech_text(self, text):
        """Speichert den letzten erkannten Sprachtext für die Klassifikation."""
        if text and isinstance(text, str):
            self.last_speech_text = text
            logger.info(f"Sprachtext erkannt: '{text}'")
        
    def classify_interruption(self, duration, energy, agent_speaking_duration, agent_in_pause):
        try:
            # Aktualisiere durchschnittliche Sprachenergie
            self.update_average_speech_energy(energy)
            
            # Debug-Logging
            if DEBUG_LOGGING:
                logger.info(f"Interruption-Klassifikation: Dauer={duration:.2f}s, Energie={energy:.6f}, Agent-Sprechdauer={agent_speaking_duration:.2f}s")
            
            # Wenn der Agent spricht, prüfe ob es sich um eine Bestätigung handelt
            if agent_speaking_duration > 0.1 and self.last_speech_text:
                # Wenn es eine einfache Bestätigung ist, nicht als Unterbrechung behandeln
                if self.speech_content_analyzer.is_acknowledgment(self.last_speech_text):
                    logger.info(f"Bestätigung erkannt, keine Unterbrechung: '{self.last_speech_text}'")
                    return "BACKCHANNELING"
                
                # Wenn es eine Frage ist, definitiv als Unterbrechung behandeln
                if self.speech_content_analyzer.contains_question(self.last_speech_text):
                    logger.info(f"Frage erkannt, als Unterbrechung behandelt: '{self.last_speech_text}'")
                    return "INTERRUPTION"
                    
                # Standardfall: Als Unterbrechung behandeln
                logger.info(f"Unterbrechung erkannt: Dauer={duration:.2f}s, Energie={energy:.6f}, Text='{self.last_speech_text}'")
                return "INTERRUPTION"
            
            # Wenn der Agent nicht spricht oder keine Spracherkennung vorliegt
            if agent_speaking_duration > 0.1:  # Agent spricht, aber kein Text erkannt
                logger.info(f"Unterbrechung erkannt (ohne Textanalyse): Dauer={duration:.2f}s, Energie={energy:.6f}")
                return "INTERRUPTION"
            
            # Nur wenn der Agent nicht spricht, als Backchanneling klassifizieren
            if agent_in_pause:
                logger.info(f"Backchanneling erkannt: Dauer={duration:.2f}s, Energie={energy:.6f}")
                return "BACKCHANNELING"
            
            # Fallback: Als Unterbrechung behandeln
            logger.info(f"Fallback-Unterbrechung erkannt: Dauer={duration:.2f}s, Energie={energy:.6f}")
            return "INTERRUPTION"
        except Exception as e:
            logger.error(f"Fehler bei Interruption-Klassifikation: {e}")
            # Im Fehlerfall als Unterbrechung behandeln
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
        # Generiere eine neue Kontext-ID (primär für clientseitiges Logging, falls benötigt)
        # Wir senden keine explizite "create context" Nachricht mehr an ElevenLabs.
        # Das Kontextmanagement wird vermutlich serverseitig gehandhabt.
        self.current_context_id = f"context_{uuid.uuid4().hex}"
        metrics.increment("context_switches") # Zählt weiterhin logische Kontextwechsel clientseitig
        logger.info(f"Logischer neuer Kontext clientseitig gestartet: {self.current_context_id}. Es wird keine 'create' Nachricht an ElevenLabs gesendet.")
        return self.current_context_id
    
    async def handle_interruption(self, ws):
        # Diese Methode wird vorerst nicht mehr direkt aufgerufen, um Signale zu senden.
        # Das Stoppen des Agenten-Audios erfolgt nun direkter.
        # Wir behalten die Methode vorerst, falls wir sie für andere Zwecke reaktivieren.
        logger.info(f"handle_interruption aufgerufen für Kontext {self.current_context_id} - AKTION AUSGESETZT")
        # Markiere Agent sofort als nicht mehr sprechend
        self.stop_agent_audio_output_immediately() # Neue, direktere Methode
        metrics.increment("interruptions_detected")

    async def handle_backchanneling(self, ws):
        # Diese Methode wird vorerst nicht mehr direkt aufgerufen.
        logger.info(f"handle_backchanneling aufgerufen für Kontext {self.current_context_id} - AKTION AUSGESETZT")
        # Hier könnte später Logik für serverseitig gemeldetes Backchanneling stehen.

    def stop_agent_audio_output_immediately(self):
        """Stoppt sofort die Annahme, dass der Agent spricht, und setzt relevante Flags."""
        if self.is_agent_speaking:
            logger.info(f"Clientseitiges Stoppen der Agenten-Audioausgabe angefordert (Dauer: {self.agent_speaking_duration:.2f}s). Kontext: {self.current_context_id}")
        self.is_agent_speaking = False
        self.agent_speaking_start_time = None
        self.agent_in_pause = False
        # HINWEIS: Das tatsächliche Leeren der Audio-Queue zum TalkDesk-Client müsste hier
        # oder durch ein Signal an stream_elevenlabs_to_talkdesk erfolgen.
        # Vorerst setzen wir nur die Flags, was das Senden neuer Agent-Audio-Chunks verhindert.

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

# WEBSOCKET MONITORING
class WebSocketMonitor:
    """Überwacht den Status einer WebSocket-Verbindung und protokolliert Aktivitäten."""
    def __init__(self, name="WebSocket"):
        self.name = name
        self.messages_sent = 0
        self.messages_received = 0
        self.last_activity = time.time()
        self.monitoring_task = None
        self.message_handlers = set()
        self.is_closed = False
        
    async def start_monitoring(self, ws):
        """Startet die Überwachung einer WebSocket-Verbindung."""
        self.websocket = ws
        self.monitoring_task = asyncio.create_task(self._monitor())
        logger.info(f"{self.name}-Monitor gestartet")
        
    async def _monitor(self):
        """Überwacht die WebSocket-Verbindung in regelmäßigen Abständen."""
        while not self.is_closed:
            try:
                await asyncio.sleep(5)  # Alle 5 Sekunden prüfen
                
                # Prüfe, ob die Verbindung noch aktiv ist
                if hasattr(self.websocket, 'closed') and self.websocket.closed:
                    logger.warning(f"{self.name} ist geschlossen")
                    self.is_closed = True
                    break
                    
                # Prüfe auf Inaktivität
                inactivity_time = time.time() - self.last_activity
                if inactivity_time > 30:  # 30 Sekunden Inaktivität
                    logger.warning(f"{self.name} ist seit {inactivity_time:.1f}s inaktiv")
                    
                # Protokolliere Statistiken
                logger.debug(f"{self.name} Status: Gesendet={self.messages_sent}, Empfangen={self.messages_received}")
                
            except Exception as e:
                logger.error(f"Fehler bei {self.name}-Überwachung: {e}")
                
    def record_sent(self):
        """Protokolliert eine gesendete Nachricht."""
        self.messages_sent += 1
        self.last_activity = time.time()
        
    def record_received(self):
        """Protokolliert eine empfangene Nachricht."""
        self.messages_received += 1
        self.last_activity = time.time()
        
    def add_message_handler(self, handler):
        """Fügt einen Handler für eingehende Nachrichten hinzu."""
        self.message_handlers.add(handler)
        
    def remove_message_handler(self, handler):
        """Entfernt einen Handler für eingehende Nachrichten."""
        if handler in self.message_handlers:
            self.message_handlers.remove(handler)
        
    def process_message(self, message):
        """Verarbeitet eine eingehende Nachricht mit allen registrierten Handlern."""
        for handler in self.message_handlers:
            try:
                handler(message)
            except Exception as e:
                logger.error(f"Fehler bei Nachrichtenverarbeitung: {e}")
        
    def stop(self):
        """Stoppt die Überwachung."""
        self.is_closed = True
        if self.monitoring_task:
            self.monitoring_task.cancel()
        logger.info(f"{self.name}-Monitor gestoppt")

# LATENCY TRACKER
class LatencyTracker:
    """Misst die Latenz für verschiedene Operationen."""
    def __init__(self):
        self.operations = {}
        self.latency_stats = {}
        
    def start(self, operation_name):
        """Startet die Zeitmessung für eine Operation."""
        self.operations[operation_name] = time.time()
        
    def end(self, operation_name):
        """Beendet die Zeitmessung für eine Operation und protokolliert die Latenz."""
        if operation_name in self.operations:
            duration_ms = (time.time() - self.operations[operation_name]) * 1000
            logger.info(f"Latenz für {operation_name}: {duration_ms:.2f}ms")
            
            # Aktualisiere Statistiken
            if operation_name not in self.latency_stats:
                self.latency_stats[operation_name] = {
                    'count': 0,
                    'total': 0,
                    'min': float('inf'),
                    'max': 0
                }
            
            stats = self.latency_stats[operation_name]
            stats['count'] += 1
            stats['total'] += duration_ms
            stats['min'] = min(stats['min'], duration_ms)
            stats['max'] = max(stats['max'], duration_ms)
            
            del self.operations[operation_name]
            return duration_ms
        return None
    
    def get_stats(self, operation_name=None):
        """Gibt Statistiken für eine bestimmte Operation oder alle Operationen zurück."""
        if operation_name:
            if operation_name in self.latency_stats:
                stats = self.latency_stats[operation_name]
                return {
                    'count': stats['count'],
                    'avg': stats['total'] / stats['count'] if stats['count'] > 0 else 0,
                    'min': stats['min'] if stats['min'] != float('inf') else 0,
                    'max': stats['max']
                }
            return None
        
        # Alle Statistiken zurückgeben
        result = {}
        for op_name, stats in self.latency_stats.items():
            result[op_name] = {
                'count': stats['count'],
                'avg': stats['total'] / stats['count'] if stats['count'] > 0 else 0,
                'min': stats['min'] if stats['min'] != float('inf') else 0,
                'max': stats['max']
            }
        return result

# WEBSOCKET MESSAGE LOGGING
def log_websocket_message(direction, message, is_binary=False):
    """Protokolliert WebSocket-Nachrichten mit Richtung und Zeitstempel."""
    timestamp = time.strftime("%H:%M:%S.%f")[:-3]
    log_prefix = f"[{timestamp}] {direction}"

    if is_binary:
        logger.info(f"{log_prefix} BINARY DATA: {len(message)} bytes")
    elif isinstance(message, (str, bytes, dict)): # dict hinzugefügt
        try:
            if isinstance(message, dict):
                # Wenn die Nachricht bereits ein Dictionary ist (z.B. bei SENT-ELEVENLABS)
                parsed_json = message 
            elif isinstance(message, bytes):
                # Wenn es Bytes sind, zuerst zu String dekodieren
                message_str = message.decode('utf-8')
                parsed_json = json.loads(message_str)
            else: # Es ist ein String
                message_str = message
                parsed_json = json.loads(message_str)
            
            # Sensible Daten maskieren und Audio kürzen
            if isinstance(parsed_json, dict):
                # Kopie für die Modifikation erstellen, um das Original nicht zu verändern
                loggable_json = parsed_json.copy()
                if "user_audio_chunk" in loggable_json:
                    loggable_json["user_audio_chunk"] = f"[AUDIO_CHUNK: {len(loggable_json['user_audio_chunk'])} chars]"
                if loggable_json.get("audio_event", {}).get("audio_base_64"):
                    loggable_json.setdefault("audio_event", {})["audio_base_64"] = f"[AUDIO_B64: {len(loggable_json['audio_event']['audio_base_64'])} chars]"
                if "xi-api-key" in loggable_json:
                    loggable_json["xi-api-key"] = "***REDACTED***"
                
                # Logge den Typ und den gesamten Inhalt der geparsten JSON
                message_type = loggable_json.get("type", "N/A")
                logger.info(f"{log_prefix} JSON (Type: {message_type}): {json.dumps(loggable_json, indent=2, ensure_ascii=False)}")
            else:
                # Falls es valides JSON ist, aber kein Dict (z.B. eine Liste oder ein Skalar)
                logger.info(f"{log_prefix} JSON (Non-dict): {json.dumps(parsed_json, indent=2, ensure_ascii=False)}")
        
        except json.JSONDecodeError:
            # Wenn es kein valides JSON ist, logge es als Text
            # (oder als Hinweis, wenn es Bytes waren, die nicht UTF-8 sind)
            # Dieser Block wird seltener erreicht, wenn message ein Dict ist, da json.loads nicht aufgerufen wird.
            if isinstance(message, bytes): # Gilt nur, wenn message ursprünglich Bytes war
                try:
                    # message_str ist hier definiert, wenn message Bytes war und dekodiert wurde
                    preview = message_str[:200] 
                    logger.info(f"{log_prefix} NON-JSON TEXT (from bytes): {preview}...")
                except UnicodeDecodeError:
                    logger.info(f"{log_prefix} NON-JSON BINARY (preview): {message[:50].hex()}...")
            elif isinstance(message, str): # Gilt nur, wenn message ursprünglich ein String war
                 # message_str ist hier definiert, wenn message ein String war
                logger.info(f"{log_prefix} NON-JSON TEXT: {message_str[:200]}...")
            # Wenn message ein Dict war und json.loads nicht aufgerufen wurde, wird dieser Block nicht erreicht.
        except Exception as e:
            logger.error(f"{log_prefix} Error in log_websocket_message: {e}", exc_info=True)
            # Fallback: Logge den rohen String oder Byte-Preview, wenn möglich
            if isinstance(message, str):
                logger.info(f"{log_prefix} RAW TEXT (after error in logging): {message[:200]}...")
            elif isinstance(message, bytes):
                 logger.info(f"{log_prefix} RAW BYTES (after error in logging, preview): {message[:50].hex()}...")

    else:
        logger.info(f"{log_prefix} UNKNOWN TYPE: {type(message)} - {str(message)[:100]}...")

# SEND WITH CONFIRMATION
async def send_with_confirmation(ws, message, timeout=1.0, max_retries=3, expect_confirmation=True):
    """Sendet eine Nachricht und wartet optional auf Bestätigung."""
    message_id = str(uuid.uuid4())
    confirmation_received = asyncio.Event()
    
    # Erstelle eine Kopie der Nachricht und füge message_id hinzu
    # Stellt sicher, dass message_with_id immer ein Dict oder ein JSON-String ist
    if isinstance(message, str):
        try:
            message_dict = json.loads(message)
        except json.JSONDecodeError:
            logger.warning(f"Nachricht ist kein valides JSON, kann keine message_id hinzufügen: {message[:100]}...")
            # Wenn keine Bestätigung erwartet wird, einfach senden
            if not expect_confirmation:
                await ws.send(message)
                log_websocket_message("SENT (no-confirm)", message)
                return True
            # Ansonsten als Fehler behandeln, da wir ID für Bestätigung brauchen
            logger.error("Kann Nachricht ohne message_id nicht mit Bestätigung senden.")
            return False
    elif isinstance(message, dict):
        message_dict = message.copy()
    else:
        logger.error(f"Ungültiger Nachrichtentyp für send_with_confirmation: {type(message)}")
        return False
        
    message_dict["message_id"] = message_id
    message_with_id_str = json.dumps(message_dict)

    # Spezielle Behandlung für 'interruption'-Nachrichten: Senden und nicht auf Bestätigung warten (oder sehr kurzer Timeout)
    # Die ElevenLabs-Dokumentation legt nahe, dass 'interruption' ein Fire-and-Forget-Signal ist.
    if message_dict.get("type") == "interruption":
        logger.info(f"Sende 'interruption'-Signal {message_id} (fire-and-forget).")
        await ws.send(message_with_id_str)
        log_websocket_message("SENT (interruption)", message_dict)
        # Wir setzen hier expect_confirmation außer Kraft, da für interruption keine explizite Bestätigung erwartet/benötigt wird
        # um den Fluss nicht zu blockieren.
        return True # Sofort als erfolgreich betrachten

    if not expect_confirmation:
        await ws.send(message_with_id_str)
        log_websocket_message("SENT (no-confirm)", message_dict)
        return True

    # Funktion zum Verarbeiten eingehender Nachrichten für Bestätigungen
    def on_message(msg_str):
        try:
            # msg_str sollte hier immer ein String sein, da WebSocket-Nachrichten als Text oder Bytes empfangen werden
            # und log_websocket_message bereits eine JSON-Konvertierung versucht.
            # Für die Bestätigungslogik parsen wir hier erneut, um sicher zu sein.
            data = json.loads(msg_str)
            
            # Erweiterte Prüfung auf message_id in der Antwort
            response_message_id = data.get("message_id")
            if not response_message_id and data.get("type") == "context_control_response":
                response_message_id = data.get("context_control_response", {}).get("message_id")
            
            if response_message_id == message_id:
                logger.debug(f"Bestätigung für {message_id} durch Antwort-message_id erhalten. Typ: {data.get('type')}")
                confirmation_received.set()
            elif data.get("type") == "confirmation" and data.get("confirmed_message_id") == message_id: # Hypothetisches Feld
                logger.debug(f"Bestätigung für {message_id} durch 'confirmation'-Typ erhalten.")
                confirmation_received.set()
            elif data.get("type") == "context_control_response" and not response_message_id:
                 logger.warning(f"Context control response ohne message_id erhalten: {data}")


        except json.JSONDecodeError:
            logger.debug(f"Empfangene Nachricht für Bestätigungsprüfung ist kein JSON: {msg_str[:100]}...")
        except Exception as e:
            logger.error(f"Fehler bei Bestätigungsverarbeitung für {message_id}: {e}", exc_info=True)
    
    # Temporären Handler für WebSocket-Monitor erstellen
    monitor = getattr(ws, '_monitor', None)
    if monitor:
        monitor.add_message_handler(on_message)
    
    # Nachricht senden und auf Bestätigung warten
    for attempt in range(max_retries):
        try:
            await ws.send(message_with_id_str)
            # Loggen der gesendeten Nachricht (das Dict, nicht den String, für bessere Lesbarkeit im Log)
            log_websocket_message("SENT", message_dict) 
            logger.debug(f"Nachricht gesendet (Versuch {attempt+1}/{max_retries}): {message_id}")
            
            try:
                await asyncio.wait_for(confirmation_received.wait(), timeout=timeout)
                logger.info(f"Bestätigung für Nachricht {message_id} (Typ: {message_dict.get('type')}) erhalten.")
                
                # Handler entfernen
                if monitor:
                    monitor.remove_message_handler(on_message)
                    
                return True
            except asyncio.TimeoutError:
                logger.warning(f"Keine Bestätigung für Nachricht {message_id} erhalten (Versuch {attempt+1})")
        except Exception as e:
            logger.error(f"Fehler beim Senden der Nachricht: {e}")
    
    # Handler entfernen
    if monitor:
        monitor.remove_message_handler(on_message)
        
    logger.error(f"Keine Bestätigung für Nachricht {message_id} nach {max_retries} Versuchen")
    return False

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
    speech_recognizer = SpeechRecognizer()
    speech_start_time = None
    last_status_check = time.time()
    status_check_interval = 0.5  # Alle 0.5 Sekunden den Sprechstatus prüfen
    
    # Zugriff auf die WebSocket-Monitore
    talkdesk_monitor = getattr(websocket, '_monitor', None)
    elevenlabs_monitor = getattr(el_ws, '_monitor', None)
    
    # Latenz-Tracking für Audio-Verarbeitung
    latency_tracker = LatencyTracker()
    
    # Statistik für Energiewerte
    energy_samples = []
    last_energy_stats_time = time.time()
    energy_stats_interval = 10.0  # Alle 10 Sekunden Energiestatistiken loggen
    
    # Spracherkennung
    last_recognition_attempt = time.time()
    recognition_attempt_interval = 0.5  # Alle 0.5 Sekunden versuchen, Sprache zu erkennen
    
    try:
        while True:
            try:
                # Empfange Nachricht von Talkdesk
                message_str = await websocket.recv()
                
                # Protokolliere empfangene Nachricht
                if talkdesk_monitor:
                    talkdesk_monitor.record_received()
                log_websocket_message("RECV-TALKDESK", message_str)
                
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
                        
                        try:
                            # Energieberechnung für Statistik
                            try:
                                energy = vad.calculate_energy(payload)
                                energy_samples.append(energy)
                                
                                # Periodisch Energiestatistiken loggen
                                if current_time - last_energy_stats_time > energy_stats_interval and len(energy_samples) > 10:
                                    avg_energy = sum(energy_samples) / len(energy_samples)
                                    max_energy = max(energy_samples)
                                    min_energy = min(energy_samples)
                                    logger.info(f"Energie-Statistik: Avg={avg_energy:.6f}, Min={min_energy:.6f}, Max={max_energy:.6f}, Samples={len(energy_samples)}")
                                    energy_samples = energy_samples[-100:]  # Behalte nur die letzten 100 Samples
                                    last_energy_stats_time = current_time
                            except Exception as e:
                                logger.error(f"Fehler bei Energieberechnung für Statistik: {e}")
                            
                            # VAD-Prüfung mit Fehlerbehandlung
                            is_voice_active = False
                            try:
                                # Übergebe den Agent-Sprechstatus an die VAD für optimierte Barge-in-Erkennung
                                is_voice_active = vad.is_voice_active(
                                    payload, 
                                    current_time, 
                                    agent_is_speaking=context_manager.is_agent_speaking
                                )
                            except Exception as vad_error:
                                logger.error(f"VAD-Fehler: {vad_error}")
                                # Trotz Fehler fortfahren
                            
                            # Wenn Agent spricht, prüfe auf Unterbrechung, auch ohne VAD-Trigger
                            if context_manager.is_agent_speaking:
                                # Direkte Energieprüfung für Unterbrechungserkennung
                                try:
                                    direct_energy = vad.calculate_energy(payload)
                                    # Aggressivere Unterbrechungserkennung: Niedrigerer Schwellenwert für schnellere Reaktion
                                    energy_threshold_factor = 1.2  # Reduziert von 1.3 für schnellere Erkennung
                                    
                                    # Wenn Energie signifikant über Hintergrund, als potenzielle Unterbrechung behandeln
                                    if direct_energy > vad.background_energy * energy_threshold_factor:
                                        if speech_start_time is None:
                                            speech_start_time = current_time
                                            logger.info(f"Potenzielle Unterbrechung durch Energieanstieg: E={direct_energy:.6f}, BG={vad.background_energy:.6f}")
                                        
                                        # Behandle als Unterbrechung nach sehr kurzer Zeit
                                        speech_duration = current_time - speech_start_time
                                        # Reduzierte Dauer für schnellere Unterbrechungserkennung
                                        if speech_duration > 0.15:  # Reduziert von 0.2s für schnellere Reaktion
                                            logger.info(f"Direkte Unterbrechung: Benutzer spricht seit {speech_duration:.2f}s, Agent spricht seit {context_manager.agent_speaking_duration:.2f}s")
                                            # Sofortige Unterbrechung senden
                                            await context_manager.handle_interruption(el_ws)
                                except Exception as e:
                                    logger.error(f"Fehler bei direkter Energieprüfung: {e}")
                            
                            # Standard VAD-basierte Verarbeitung
                            if is_voice_active:
                                if speech_start_time is None:
                                    speech_start_time = current_time
                                    logger.info("Benutzersprache erkannt (stream_talkdesk_to_elevenlabs)") # Eindeutiger machen
                                
                                # Wenn Agent spricht, klassifiziere die Unterbrechung
                                if context_manager.is_agent_speaking: # Agent spricht auch
                                    speech_duration = current_time - speech_start_time
                                    current_chunk_energy = 0.0
                                    try:
                                        current_chunk_energy = vad.calculate_energy(payload) # Energie des aktuellen Chunks
                                    except Exception as e:
                                        logger.error(f"Fehler bei Energieberechnung für Unterbrechungs-Check: {e}")
                                        current_chunk_energy = 0.001  # Fallback
                                    
                                    # NEUES LOGGING (Punkt 1) - Bleibt zur Beobachtung
                                    logger.info(f"[INTERRUPTION_CHECK] User speech detected while agent speaking. "
                                                f"UserSpeechDuration: {speech_duration:.2f}s, "
                                                f"ChunkEnergy: {current_chunk_energy:.6f}, "
                                                f"AgentSpeakingDuration: {context_manager.agent_speaking_duration:.2f}s, "
                                                f"AgentInPause: {context_manager.agent_in_pause}, "
                                                f"LastUserText: '{interruption_classifier.last_speech_text}'")
                                    
                                    # Direkter Stopp der Agenten-Audioausgabe clientseitig
                                    logger.info("Benutzeraktivität während Agent spricht: Stoppe Agenten-Audio clientseitig.")
                                    context_manager.stop_agent_audio_output_immediately()
                                    # Wir senden keine expliziten interruption/abort Befehle mehr.
                                    # Die weitere Handhabung (neue Agentenantwort) wird durch das Senden
                                    # des user_audio_chunk an ElevenLabs und deren serverseitige Logik getriggert.
                            
                            # Füge Audio zum Spracherkennungspuffer hinzu
                            speech_recognizer.add_audio(payload)
                            
                            # Versuche regelmäßig, Sprache zu erkennen
                            if current_time - last_recognition_attempt >= recognition_attempt_interval:
                                last_recognition_attempt = current_time
                                recognized_text = await speech_recognizer.try_recognize(current_time)
                                if recognized_text:
                                    # Aktualisiere den Interruption Classifier mit dem erkannten Text
                                    interruption_classifier.set_last_speech_text(recognized_text)
                                    
                                    # Wenn der Agent spricht, prüfe sofort, ob es eine Unterbrechung ist
                                    if context_manager.is_agent_speaking and speech_start_time is not None:
                                        speech_duration = current_time - speech_start_time
                                        energy = vad.calculate_energy(payload)
                                        
                                        # Die Klassifizierung und das Senden von Steuerbefehlen entfällt hier.
                                        # Wir verlassen uns darauf, dass das Senden des User-Audios an ElevenLabs
                                        # die serverseitige Logik für eine neue Antwort oder Korrektur auslöst.
                                        logger.info(f"Text erkannt ('{recognized_text}') während Agent (vermutlich) schon gestoppt wurde (clientseitig). Sende Audio an ElevenLabs.")

                            # Audio an ElevenLabs senden
                            audio_msg = {"user_audio_chunk": payload}
                            await el_ws.send(json.dumps(audio_msg))
                            if elevenlabs_monitor:
                                elevenlabs_monitor.record_sent()
                            log_websocket_message("SENT-ELEVENLABS", audio_msg)
                            metrics.increment("packets_sent")
                            
                            # Reset, wenn keine Sprache mehr erkannt wird
                            if not vad.is_speaking and speech_start_time is not None:
                                logger.info(f"Benutzer hat aufgehört zu sprechen (nach {current_time - speech_start_time:.2f}s)")
                                speech_start_time = None
                        except Exception as chunk_error:
                            logger.error(f"Fehler bei Audio-Chunk-Verarbeitung: {chunk_error}")
                            # Audio trotzdem senden, um Kontinuität zu gewährleisten
                            await el_ws.send(json.dumps({"user_audio_chunk": payload}))
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
    # Zugriff auf die WebSocket-Monitore
    elevenlabs_monitor = getattr(el_ws, '_monitor', None)
    talkdesk_monitor = getattr(websocket, '_monitor', None)
    
    # Latenz-Tracking für Audio-Übertragung
    latency_tracker = LatencyTracker()
    
    try:
        while True:
            try:
                # Empfange Nachricht von ElevenLabs
                message_str = await el_ws.recv()
                metrics.increment("packets_received")
                
                # Protokolliere empfangene Nachricht
                if elevenlabs_monitor:
                    elevenlabs_monitor.record_received()
                log_websocket_message("RECV-ELEVENLABS", message_str)
                
                data = json.loads(message_str)
                current_time = time.time()
                
                # Verarbeite Audio-Events
                if data.get("type") == "audio":
                    b64_audio = data.get("audio_event", {}).get("audio_base_64")
                    if b64_audio:
                        # Nur Audio senden und Agent als sprechend markieren,
                        # wenn der ContextManager dies aktuell erlaubt.
                        if context_manager.is_agent_speaking:
                            context_manager.agent_started_speaking() # Aktualisiert last_audio_time etc.
                            
                            latency_tracker.start("audio_forward")
                            talkdesk_msg = {
                                "event": "media",
                                "streamSid": stream_sid,
                                "media": {"payload": b64_audio}
                            }
                            await websocket.send(json.dumps(talkdesk_msg))
                            if talkdesk_monitor:
                                talkdesk_monitor.record_sent()
                            log_websocket_message("SENT-TALKDESK", talkdesk_msg)
                            metrics.increment("packets_sent")
                            latency_tracker.end("audio_forward")
                        else:
                            # Wenn is_agent_speaking False ist (z.B. durch clientseitige Unterbrechung),
                            # verwerfen wir dieses Audio und loggen es.
                            logger.info(f"RECV-ELEVENLABS: Audio-Event empfangen, aber Agent soll nicht sprechen (is_agent_speaking=False). Verwerfe {len(b64_audio)} Bytes Audio.")
                            # Es ist wichtig, hier NICHT agent_started_speaking() aufzurufen.
                
                elif data.get("type") == "agent_response_correction":
                    correction_event = data.get("agent_response_correction_event", {})
                    original_response = correction_event.get("original_agent_response")
                    corrected_response = correction_event.get("corrected_agent_response")
                    logger.info(f"Agentenantwort-Korrektur erhalten. Original: '{original_response}', Korrigiert: '{corrected_response}'")
                    # Stoppe die aktuelle Audiowiedergabe des Agenten, da eine Korrektur bedeutet, dass eine Unterbrechung stattgefunden hat.
                    # Dies ist eine zusätzliche Sicherheit, falls das clientseitige Stoppen in stream_talkdesk_to_elevenlabs nicht ausreicht
                    # oder falls ElevenLabs die Unterbrechung rein serverseitig erkennt und dieses Event sendet.
                    context_manager.stop_agent_audio_output_immediately()
                    # Hier müsste idealerweise die Audio-Queue für TalkDesk geleert werden.
                    # TODO: Implementiere Mechanismus zum Leeren der TalkDesk-Audio-Queue.

                # Verarbeite Kontext-Events (obwohl wir sie nicht mehr aktiv senden, loggen wir, falls sie kommen)
                elif data.get("type") == "context_control_response":
                    control_response = data.get("context_control_response", {})
                    action = control_response.get("action")
                    success = control_response.get("success", False)
                    message_id = control_response.get("message_id", "unknown")
                    
                    logger.info(f"Unerwartete Context-Control-Response empfangen: Aktion={action}, Erfolg={success}, ID={message_id}")
                    if action == "abort" and success:
                        context_manager.agent_stopped_speaking() # Sicherstellen, dass der Status korrekt ist
                
                # Verarbeite End-Events
                elif data.get("type") == "end":
                    logger.info("Ende des Streams von ElevenLabs")
                    context_manager.agent_stopped_speaking()
                
                # Verarbeite Bestätigungsnachrichten
                elif data.get("type") == "confirmation":
                    logger.debug(f"Bestätigung erhalten: {data.get('message_id', 'unknown')}")
                
                # Verarbeite andere Nachrichtentypen
                else:
                    logger.debug(f"Unbekannter Nachrichtentyp von ElevenLabs: {data.get('type', 'unknown')}")
                    
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
    
    # Initialisiere Latenz-Tracker
    latency_tracker = LatencyTracker()
    
    # WebSocket-Monitore für Verbindungsüberwachung
    talkdesk_monitor = WebSocketMonitor("Talkdesk-WS")
    elevenlabs_monitor = WebSocketMonitor("ElevenLabs-WS")

    try:
        # Starte Talkdesk-Monitor
        await talkdesk_monitor.start_monitoring(websocket)
        
        stream_sid = None
        # START-NACHRICHT PARSEN
        latency_tracker.start("start_message_wait")
        for _ in range(5):
            try:
                msg = await asyncio.wait_for(websocket.recv(), timeout=10)
                talkdesk_monitor.record_received()
                log_websocket_message("RECV-TALKDESK", msg)
                
                data = json.loads(msg)
                if data.get("event") == "start":
                    stream_sid = data["start"].get("streamSid")
                    latency_tracker.end("start_message_wait")
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
        latency_tracker.start("elevenlabs_connection")
        logger.info("Versuche, Signed URL von ElevenLabs zu erhalten...")
        signed_url = await get_elevenlabs_signed_url()
        if not signed_url:
            logger.error("Konnte keine Signed URL erhalten. Abbruch der Verbindung.")
            return
        logger.info(f"Signed URL erhalten: {signed_url[:70]}...") # Logge nur einen Teil der URL
            
        # Verbesserte WebSocket-Verbindung zu ElevenLabs mit vollständigen Parametern
        el_ws = None
        try:
            logger.info(f"Versuche, WebSocket-Verbindung zu ElevenLabs herzustellen: {signed_url[:70]}...")
            el_ws = await websockets.connect(
                signed_url,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=5
            )
            el_ws._monitor = elevenlabs_monitor # Monitor vor dem Starten hinzufügen
            await elevenlabs_monitor.start_monitoring(el_ws)
            
            latency_tracker.end("elevenlabs_connection")
            logger.info(f"WebSocket-Verbindung zu ElevenLabs erfolgreich hergestellt. Zustand: {el_ws.state}")
        except Exception as e:
            logger.error(f"Fehler beim Verbinden mit ElevenLabs WebSocket: {e}", exc_info=True)
            metrics.record_error("ElevenLabsConnectionError", str(e))
            if el_ws and el_ws.open: # Sicherstellen, dass wir nur schließen, wenn offen
                await el_ws.close()
            return

        # Initialisierung senden
        initial_config = build_initial_config()
        logger.info(f"Sende Initial Config an ElevenLabs: {json.dumps(initial_config)}")
        latency_tracker.start("initial_config_send")
        try:
            await el_ws.send(json.dumps(initial_config))
            elevenlabs_monitor.record_sent()
            log_websocket_message("SENT-ELEVENLABS", initial_config) # Nutzt bereits JSON-Dumps
            metrics.increment("packets_sent")
            latency_tracker.end("initial_config_send")
            logger.info("Initial Config erfolgreich an ElevenLabs gesendet.")
        except Exception as e:
            logger.error(f"Fehler beim Senden der Initial Config an ElevenLabs: {e}", exc_info=True)
            metrics.record_error("ElevenLabsSendError", str(e))
            await el_ws.close()
            return
            
        # Der explizite Empfang der ersten Nachricht wird entfernt.
        # stream_elevenlabs_to_talkdesk wird alle Nachrichten von el_ws handhaben,
        # einschließlich conversation_initiation_metadata und dem ersten Audio.
        logger.info("Initial Config gesendet. Starte Kontext-Manager und Streaming-Tasks.")

        # Kontext-Manager initialisieren
        context_manager = ContextManager()
        # start_new_context sendet keine Nachricht mehr, nur Logging und ID-Generierung
        await context_manager.start_new_context(el_ws) 

        # TASKS STARTEN
        logger.info("Starte Streaming-Tasks...")
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
            
        # Stoppe die WebSocket-Monitore
        talkdesk_monitor.stop()
        elevenlabs_monitor.stop()
        
        # Protokolliere Latenz-Statistiken
        latency_stats = latency_tracker.get_stats()
        if latency_stats:
            logger.info("Latenz-Statistiken für diese Verbindung:")
            for op_name, stats in latency_stats.items():
                logger.info(f"  {op_name}: Avg={stats['avg']:.2f}ms, Min={stats['min']:.2f}ms, Max={stats['max']:.2f}ms, Count={stats['count']}")
            
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

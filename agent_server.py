import asyncio
import websockets
import os
import logging
import json
import base64
# --- NEU: Fehlende Imports hinzufügen ---
import httpx  # Für HTTP-Anfragen (Signed URL)
import sys   # Für sys.exit beim Fehler

# --- Konfiguration ---
# Lese Konfiguration NUR aus Umgebungsvariablen. Beende, wenn kritische Werte fehlen.
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")
ELEVENLABS_AGENT_ID = os.environ.get("ELEVENLABS_AGENT_ID")
# Für den PoC verwenden wir feste Prompts/Nachrichten aus Env Vars
POC_PROMPT = os.environ.get("ELEVENLABS_POC_PROMPT", "You are a test assistant. Just say hello.") # Einfacher Standard-Prompt
POC_FIRST_MESSAGE = os.environ.get("ELEVENLABS_POC_FIRST_MESSAGE", "Hello test call.") # Einfache Standard-Antwort
WEBSOCKET_HOST = "0.0.0.0"
WEBSOCKET_PORT = 8080       # KORREKT für internen Port

# --- Logging Setup ---
# Minimal Level auf DEBUG setzen, um mehr zu sehen
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Kritische Konfiguration prüfen ---
if not ELEVENLABS_API_KEY:
    logger.error("Umgebungsvariable ELEVENLABS_API_KEY nicht gesetzt!")
    sys.exit(1) # Beendet das Skript mit Fehlercode
if not ELEVENLABS_AGENT_ID:
    logger.error("Umgebungsvariable ELEVENLABS_AGENT_ID nicht gesetzt!")
    sys.exit(1) # Beendet das Skript mit Fehlercode
logger.info(f"Konfiguration geladen. Agent ID: {ELEVENLABS_AGENT_ID}, Interner Port: {WEBSOCKET_PORT}")
logger.info(f"PoC Prompt: '{POC_PROMPT}'")
logger.info(f"PoC First Message: '{POC_FIRST_MESSAGE}'")

# --- NEU: Helper Funktion zum Holen der Signed URL ---
async def get_elevenlabs_signed_url():
    """Holt die temporäre, sichere WebSocket URL von ElevenLabs."""
    url = f"https://api.elevenlabs.io/v1/convai/conversation/get_signed_url?agent_id={ELEVENLABS_AGENT_ID}"
    headers = {'xi-api-key': ELEVENLABS_API_KEY}
    try:
        async with httpx.AsyncClient() as client:
            logger.info(f"Rufe Signed URL für Agent {ELEVENLABS_AGENT_ID} ab...")
            response = await client.get(url, headers=headers, timeout=10.0)
            response.raise_for_status()
            data = response.json()
            signed_url = data.get("signed_url")
            if signed_url:
                 logger.info("Signed URL erfolgreich erhalten.")
                 return signed_url
            else:
                 logger.error(f"Keine signed_url in der Antwort von Elevenlabs gefunden. Antwort: {data}")
                 return None
    except httpx.HTTPStatusError as e:
         logger.error(f"HTTP-Fehler beim Abrufen der Signed URL: {e.response.status_code} - {e.response.text}")
         return None
    except httpx.RequestError as e:
         logger.error(f"Netzwerkfehler bei der Anfrage an Elevenlabs (Signed URL): {e}")
         return None
    except Exception as e:
        logger.error(f"Allgemeiner Fehler beim Abrufen der Signed URL: {e}", exc_info=True)
        return None

# --- WebSocket Handler (minimal für PoC) ---
async def handle_connection(talkdesk_ws):
    """Verwaltet eine einzelne WebSocket-Verbindung von TalkDesk."""
    remote_addr = talkdesk_ws.remote_address
    logger.info(f"+++ Neue Verbindung von TalkDesk: {remote_addr}")

    elevenlabs_ws = None
    stream_sid = None # Wichtig für spätere Antworten

    try:
        # 1. Warte auf die erste Nachricht (sollte "start" sein)
        try:
            start_message_str = await asyncio.wait_for(talkdesk_ws.recv(), timeout=15.0)
            logger.info("Erste Nachricht von TalkDesk erhalten.")
            logger.debug(f"Raw Start Message: {start_message_str}")
        except asyncio.TimeoutError:
            logger.error(f"Timeout (15s) beim Warten auf Start-Nachricht von {remote_addr}.")
            return
        except Exception as e:
             logger.error(f"Fehler beim Empfangen der Start-Nachricht: {e}", exc_info=True)
             return

        # Parse Start Nachricht (um stream_sid zu bekommen und Parameter zu loggen)
        try:
            start_data = json.loads(start_message_str)
            event = start_data.get("event")
            if event != "start":
                logger.error(f"Erste Nachricht war kein 'start'-Event, sondern '{event}'. Beende.")
                return

            # TODO: Passe die Schlüssel hier an die *echte* TalkDesk JSON Struktur an!
            start_info = start_data.get("start", {})
            stream_sid = start_info.get("stream_sid")
            call_sid = start_info.get("call_sid", "UnknownCall")
            custom_params = start_info.get("customParameters", {}) # VERMUTUNG!
            media_format = start_info.get("mediaFormat", {})

            if not stream_sid:
                logger.error("Keine 'stream_sid' im Start-Event gefunden. Beende.")
                return

            logger.info(f"Anruf gestartet: CallSid={call_sid}, StreamSid={stream_sid}")
            logger.info(f"Empfangene Media Format Info: {media_format}")
            logger.info(f"Empfangene Custom Parameters (VERMUTUNG!): {custom_params}") # Loggen, was ankommt!

        except json.JSONDecodeError:
            logger.error(f"Konnte Start-Nachricht nicht als JSON parsen: {start_message_str}")
            return
        except Exception as e:
            logger.error(f"Fehler beim Verarbeiten der Start-Nachricht: {e}", exc_info=True)
            return

        # --- Verbindung zu Elevenlabs aufbauen ---
        signed_url = await get_elevenlabs_signed_url()
        if not signed_url:
            raise ConnectionAbortedError("Konnte keine Signed URL von Elevenlabs erhalten.")

        elevenlabs_ws = await websockets.connect(signed_url)
        logger.info("Verbindung zu Elevenlabs WebSocket hergestellt.")

        # --- Initiale Konfiguration an Elevenlabs senden (PoC mit Env Vars) ---
        initial_config = {
            "type": "conversation_initiation_client_data",
            "conversation_config_override": {
                "agent": { "prompt": { "prompt": POC_PROMPT } }
            }
        }
        if POC_FIRST_MESSAGE:
            initial_config["conversation_config_override"]["agent"]["first_message"] = POC_FIRST_MESSAGE

        await elevenlabs_ws.send(json.dumps(initial_config))
        logger.info(f"Initiale Konfiguration an Elevenlabs gesendet: Prompt='{POC_PROMPT}'")

        # --- HIER IST SCHLUSS FÜR DEN PoC ---
        # Wir implementieren das Audio-Handling noch nicht.
        # Wir lassen die Verbindung einfach offen und loggen nur noch, wenn TalkDesk schließt.
        logger.info("Proof of Concept: Verbindung zu TalkDesk und ElevenLabs steht. Warte auf Schließen der Verbindung...")

        # Einfache Schleife, um auf das Schließen durch TalkDesk zu warten
        async for message in talkdesk_ws:
             logger.debug(f"Weitere Nachricht von TalkDesk ignoriert (PoC): {message[:100]}...") # Logge nur Anfang
             pass # Ignoriere weitere Nachrichten im PoC


    except websockets.exceptions.ConnectionClosedOK:
        logger.info(f"Verbindung von TalkDesk {remote_addr} normal geschlossen.")
    except websockets.exceptions.ConnectionClosedError as e:
        logger.error(f"Verbindung von TalkDesk {remote_addr} unerwartet geschlossen: Code={e.code}, Grund='{e.reason}'")
    except ConnectionAbortedError as e:
         logger.error(f"Verbindung abgebrochen für {remote_addr}: {e}")
    except Exception as e:
        logger.error(f"Unerwarteter Fehler im Haupt-Handler für {remote_addr}: {e}", exc_info=True)
    finally:
        logger.info(f"Beende Handler für {remote_addr}. Räume auf...")
        # Elevenlabs WebSocket schließen (falls er geöffnet wurde)
        if elevenlabs_ws and elevenlabs_ws.open:
            logger.info("Schließe Elevenlabs WebSocket Verbindung...")
            await elevenlabs_ws.close(code=1000, reason='PoC Handler finished')
            logger.info("Elevenlabs WebSocket aufgeräumt.")


# --- Hauptfunktion zum Starten des Servers ---
async def main():
    # Prüfe ob httpx installiert ist
    try:
        import httpx
    except ImportError:
         logger.error("Modul 'httpx' nicht gefunden! Stelle sicher, dass es in requirements.txt steht.")
         sys.exit(1)

    logger.info(f"Starte PoC WebSocket Server auf {WEBSOCKET_HOST}:{WEBSOCKET_PORT}")
    try:
        async with websockets.serve(handle_connection, WEBSOCKET_HOST, WEBSOCKET_PORT):
            await asyncio.Future()
    except OSError as e:
         if e.errno == 98:
              logger.error(f"Port {WEBSOCKET_PORT} wird bereits verwendet!")
         else:
              logger.error(f"Server konnte nicht gestartet werden (OS Error): {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Server konnte nicht gestartet werden oder ist abgestürzt: {e}", exc_info=True)

# --- Skript-Einstiegspunkt ---
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server durch Benutzer (Ctrl+C) gestoppt.")
    finally:
        logger.info("Server wird beendet.")
import asyncio
import websockets
import os
import logging
import json
import base64
import httpx  # Für HTTP-Anfragen (Signed URL)
import sys   # Für sys.exit beim Fehler
from websockets.protocol import ConnectionState # <--- ERNEUT KORRIGIERTER IMPORT

# SCRIPT VERSION FÜR LOGGING
SCRIPT_VERSION = "3.3 - ConnectionState Import Fix Attempt 2"

# --- Konfiguration ---
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")
ELEVENLABS_AGENT_ID = os.environ.get("ELEVENLABS_AGENT_ID")
POC_PROMPT = os.environ.get("ELEVENLABS_POC_PROMPT", "You are a test assistant. Just say hello.")
POC_FIRST_MESSAGE = os.environ.get("ELEVENLABS_POC_FIRST_MESSAGE", "Hello test call.")
WEBSOCKET_HOST = "0.0.0.0"
WEBSOCKET_PORT = 8080

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info(f"Starte Agent Server - VERSION {SCRIPT_VERSION}")

if not ELEVENLABS_API_KEY:
    logger.error("Umgebungsvariable ELEVENLABS_API_KEY nicht gesetzt!")
    sys.exit(1)
if not ELEVENLABS_AGENT_ID:
    logger.error("Umgebungsvariable ELEVENLABS_AGENT_ID nicht gesetzt!")
    sys.exit(1)
logger.info(f"Konfiguration geladen. Agent ID: {ELEVENLABS_AGENT_ID}, Interner Port: {WEBSOCKET_PORT}")
logger.info(f"PoC Prompt: '{POC_PROMPT}'")
logger.info(f"PoC First Message: '{POC_FIRST_MESSAGE}'")

async def get_elevenlabs_signed_url():
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

async def handle_connection(talkdesk_ws):
    remote_addr = talkdesk_ws.remote_address
    logger.info(f"+++ Neue Verbindung (möglicherweise TalkDesk): {remote_addr}")
    elevenlabs_ws = None
    stream_sid = None
    try:
        start_data = None
        start_message_str_for_logging = None
        message_count = 0
        max_initial_messages = 5

        while message_count < max_initial_messages:
            message_count += 1
            logger.debug(f"Warte auf Nachricht #{message_count} von Client {remote_addr}...")
            try:
                message_str = await asyncio.wait_for(talkdesk_ws.recv(), timeout=10.0)
                logger.info(f"Nachricht #{message_count} von Client {remote_addr} erhalten.")
                logger.debug(f"Raw Message #{message_count}: {message_str}")
            except asyncio.TimeoutError:
                logger.error(f"Timeout (10s) beim Warten auf Nachricht #{message_count} von {remote_addr}.")
                if message_count == 1 and not start_data: return
                continue
            except websockets.exceptions.ConnectionClosed:
                logger.info(f"Verbindung von Client {remote_addr} geschlossen während Warten auf Nachricht #{message_count}.")
                return
            except Exception as e:
                logger.error(f"Fehler beim Empfangen der Nachricht #{message_count} von {remote_addr}: {e}", exc_info=True)
                return
            try:
                data = json.loads(message_str)
                event = data.get("event")
                if event == "connected":
                    logger.info(f"'{event}'-Event von {remote_addr} empfangen und ignoriert.")
                    continue
                elif event == "start":
                    logger.info(f"'start'-Event von {remote_addr} gefunden!")
                    start_data = data
                    start_message_str_for_logging = message_str
                    break
                elif event == "media":
                    logger.warning(f"Unerwartetes 'media'-Event von {remote_addr} während Initialisierung. Ignoriere.")
                    continue
                else:
                    logger.warning(f"Unbekanntes Event '{event}' von {remote_addr}. Ignoriere: {message_str[:200]}...")
                    continue
            except json.JSONDecodeError:
                logger.error(f"Konnte Nachricht #{message_count} von {remote_addr} nicht als JSON parsen: {message_str[:200]}...")
                continue
            except Exception as e:
                logger.error(f"Fehler Verarbeiten Nachricht #{message_count} von {remote_addr}: {e}", exc_info=True)
                return

        if not start_data:
            logger.error(f"Kein 'start'-Event von {remote_addr} nach {max_initial_messages} Nachrichten. Beende.")
            return

        logger.info(f"--- Vollständige Start-Nachricht von {remote_addr} ---")
        logger.info(start_message_str_for_logging)
        logger.info(f"--- Verarbeitete Start-Daten (pretty) ---")
        logger.info(json.dumps(start_data, indent=2))

        start_info = start_data.get("start", {})
        if not start_info:
            logger.error(f"Start-Event von {remote_addr} ohne 'start'-Objekt: {start_data}")
            return

        stream_sid = start_info.get("streamSid")
        call_sid = start_info.get("callSid", f"UnknownCall_{remote_addr}")
        account_sid_from_start = start_info.get("accountSid")
        media_format = start_info.get("mediaFormat", {})
        custom_params = start_info.get("customParameters", {})
        account_sid_top = start_data.get("account_sid", "UnknownAccount")
        call_sid_top = start_data.get("call_sid", "UnknownCall")

        if not stream_sid:
            logger.error(f"Kein 'streamSid' im 'start'-Objekt von {remote_addr}: {start_info}")
            return

        logger.info(f"Anruf gestartet: CallSid='{call_sid}', StreamSid='{stream_sid}', AccountSid(Start)='{account_sid_from_start}', AccountSid(Top)='{account_sid_top}'")
        logger.info(f"Media Format von {remote_addr}: {media_format}")
        logger.info(f"Custom Parameters von {remote_addr}: {custom_params}")

        signed_url = await get_elevenlabs_signed_url()
        if not signed_url:
            raise ConnectionAbortedError(f"Keine Signed URL für {remote_addr}.")

        elevenlabs_ws = await websockets.connect(signed_url)
        logger.info(f"Verbindung zu Elevenlabs WebSocket für {remote_addr} hergestellt.")

        initial_config = {
            "type": "conversation_initiation_client_data",
            "conversation_config_override": {"agent": {"prompt": {"prompt": POC_PROMPT}}}
        }
        # if POC_FIRST_MESSAGE: # Vorerst auskommentiert lassen
        #    initial_config["conversation_config_override"]["agent"]["first_message"] = POC_FIRST_MESSAGE

        await elevenlabs_ws.send(json.dumps(initial_config))
        logger.info(f"Initiale Konfiguration an Elevenlabs für {remote_addr} gesendet: Prompt='{POC_PROMPT}' (First Message NICHT gesendet!)")

        logger.info(f"PoC für {remote_addr}: Verbindung zu TalkDesk & ElevenLabs steht. Warte...")

        async for message in talkdesk_ws:
            try:
                if isinstance(message, str):
                    msg_data = json.loads(message)
                    evt = msg_data.get("event", "unknown")
                    if evt == "media":
                        payload_preview = msg_data.get("media", {}).get("payload", "")[:20] + "..."
                        logger.debug(f"Media von {remote_addr} (Payload: {payload_preview}) ignoriert (PoC).")
                    elif evt == "stop":
                        logger.info(f"'stop'-Event von TalkDesk {remote_addr} empfangen: {message[:200]}")
                        break
                    else:
                        logger.debug(f"Text-Nachricht von {remote_addr} (Typ: {evt}) ignoriert (PoC): {message[:100]}...")
                elif isinstance(message, bytes):
                    logger.debug(f"Binäre Nachricht von {remote_addr} ignoriert (PoC): {len(message)} bytes")
                else:
                    logger.debug(f"Unbekannte Nachricht (Typ: {type(message)}) von {remote_addr} ignoriert (PoC).")
            except json.JSONDecodeError:
                 logger.warning(f"Konnte Nachricht von {remote_addr} nicht als JSON parsen: {message[:100]}...")
            except Exception as e:
                logger.warning(f"Fehler bei Verarbeitung Nachricht von {remote_addr}: {e}")
            pass

    except websockets.exceptions.ConnectionClosedOK:
        logger.info(f"Verbindung von Client {remote_addr} normal geschlossen (ClosedOK).")
    except websockets.exceptions.ConnectionClosedError as e:
        logger.error(f"Verbindung von Client {remote_addr} unerwartet geschlossen (ClosedError): Code={e.code}, Grund='{e.reason}'")
    except ConnectionAbortedError as e:
         logger.error(f"Verarbeitung für {remote_addr} aktiv abgebrochen: {e}")
    except Exception as e:
        logger.error(f"Unerwarteter Fehler im Haupt-Handler für {remote_addr}: {e}", exc_info=True)
    finally:
        logger.info(f"Beende Handler für {remote_addr}. Räume auf...")
        if elevenlabs_ws:
            if elevenlabs_ws.state == ConnectionState.OPEN:
                logger.info(f"Schließe Elevenlabs WebSocket Verbindung für {remote_addr} (State: OPEN)...")
                try:
                    await asyncio.wait_for(elevenlabs_ws.close(code=1000, reason='Handler finished normally'), timeout=5.0)
                    logger.info(f"Elevenlabs WebSocket für {remote_addr} aufgeräumt.")
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout beim Schließen der Elevenlabs WebSocket für {remote_addr}.")
                except Exception as e:
                    logger.error(f"Fehler beim Schließen der Elevenlabs WebSocket für {remote_addr}: {e}", exc_info=True)
            elif elevenlabs_ws.state == ConnectionState.CLOSED:
                 logger.info(f"Elevenlabs WebSocket für {remote_addr} war bereits geschlossen.")
            else:
                 logger.warning(f"Elevenlabs WebSocket für {remote_addr} in Zustand ({elevenlabs_ws.state}) beim Aufräumen. Versuche trotzdem zu schließen.")
                 try:
                    await asyncio.wait_for(elevenlabs_ws.close(code=1008, reason='Closing from unexpected state'), timeout=5.0)
                    logger.info(f"Elevenlabs WebSocket für {remote_addr} (aus unerwartetem Zustand) aufgeräumt.")
                 except Exception as e:
                    logger.error(f"Fehler beim Schließen des Elevenlabs WebSockets (aus unerwartetem Zustand) für {remote_addr}: {e}", exc_info=True)
        else:
             logger.info(f"Keine (initialisierte) Elevenlabs WebSocket Verbindung zum Schließen für {remote_addr} vorhanden.")

async def main():
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
              logger.error(f"Server konnte nicht gestartet werden (OS Error {e.errno}): {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Server konnte nicht gestartet werden oder ist abgestürzt: {e}", exc_info=True)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server durch Benutzer (Ctrl+C) gestoppt.")
    finally:
        logger.info("Server wird beendet.")

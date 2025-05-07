import asyncio
import websockets # Importiere das Hauptmodul zuerst
import os
import logging
import json
import base64 # Obwohl nicht direkt verwendet, oft nützlich bei WS
import httpx
import sys
from websockets.connection import State # NEU: Korrekter Import für den Verbindungsstatus

# SCRIPT VERSION FÜR LOGGING
SCRIPT_VERSION = "3.13 - Finally Block with .state & Pinned Lib Version"

# --- Logging Setup ---
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info(f"Starte Agent Server - VERSION {SCRIPT_VERSION}")
try:
    logger.info(f"WEBSOCKETS LIBRARY VERSION (aus Umgebung): {websockets.__version__}")
except Exception as e:
    logger.error(f"Konnte websockets.__version__ nicht abrufen: {e}")

# --- Konfiguration ---
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")
ELEVENLABS_AGENT_ID = os.environ.get("ELEVENLABS_AGENT_ID")
POC_PROMPT = os.environ.get("ELEVENLABS_POC_PROMPT", "You are a test assistant. Just say hello.")
POC_FIRST_MESSAGE = os.environ.get("ELEVENLABS_POC_FIRST_MESSAGE", "Hello test call.")
WEBSOCKET_HOST = "0.0.0.0"
WEBSOCKET_PORT = 8080

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
                logger.error(f"Keine signed_url in der Antwort von Elevenlabs. Antwort: {data}")
                return None
    except Exception as e:
        logger.error(f"Fehler beim Abrufen der Signed URL: {e}", exc_info=True)
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
                if message_count == 1 and not start_data: return # Wenn erste Nachricht ausbleibt, beenden
                continue # Ansonsten bei weiteren Timeouts im Loop bleiben, falls start schon da ist
            except websockets.exceptions.ConnectionClosed:
                logger.info(f"Verbindung von Client {remote_addr} geschlossen während Warten auf Nachricht #{message_count}.")
                return
            except Exception as e:
                logger.error(f"Fehler Empfangen Nachricht #{message_count} von {remote_addr}: {e}", exc_info=True)
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
                    start_message_str_for_logging = message_str # Für späteres Logging speichern
                    break # Start-Event gefunden, Schleife verlassen
                elif event == "media":
                    logger.warning(f"Unerwartetes 'media'-Event von {remote_addr} während Initialisierung. Ignoriere.")
                    continue
                else:
                    logger.warning(f"Unbekanntes Event '{event}' von {remote_addr}. Ignoriere: {message_str[:200]}...")
                    continue
            except json.JSONDecodeError:
                logger.error(f"Konnte Nachricht #{message_count} von {remote_addr} nicht als JSON parsen: {message_str[:200]}...")
                continue
            except Exception as e: # Andere Fehler beim Verarbeiten der Nachricht
                logger.error(f"Fehler Verarbeiten Nachricht #{message_count} von {remote_addr}: {e}", exc_info=True)
                return

        if not start_data:
            logger.error(f"Kein 'start'-Event von {remote_addr} nach {max_initial_messages} Nachrichten. Beende.")
            return

        logger.info(f"--- Vollständige Start-Nachricht von {remote_addr} ---")
        logger.info(start_message_str_for_logging) # Geloggte Start-Nachricht
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
        # account_sid_top = start_data.get("account_sid", "UnknownAccount") # `account_sid` ist im `start` Objekt, nicht auf Top-Level

        if not stream_sid:
            logger.error(f"Kein 'streamSid' im 'start'-Objekt von {remote_addr}: {start_info}")
            return

        logger.info(f"Anruf gestartet: CallSid='{call_sid}', StreamSid='{stream_sid}', AccountSid(Start)='{account_sid_from_start}'")
        logger.info(f"Media Format von {remote_addr}: {media_format}")
        logger.info(f"Custom Parameters von {remote_addr}: {custom_params}")

        signed_url = await get_elevenlabs_signed_url()
        if not signed_url:
            # Fehler wurde bereits in get_elevenlabs_signed_url geloggt.
            # Wir könnten hier eine spezifischere Exception werfen, wenn nötig,
            # oder einfach den Handler beenden, da ohne URL nichts geht.
            logger.error(f"Abbruch, da keine Signed URL für {remote_addr} erhalten werden konnte.")
            return # Oder raise ConnectionAbortedError(...)

        elevenlabs_ws = await websockets.connect(signed_url)
        logger.info(f"Verbindung zu Elevenlabs WebSocket für {remote_addr} hergestellt.")

        initial_config = {
            "type": "conversation_initiation_client_data",
            "conversation_config_override": {"agent": {"prompt": {"prompt": POC_PROMPT}}}
        }
        if POC_FIRST_MESSAGE:
            initial_config["conversation_config_override"]["agent"]["first_message"] = POC_FIRST_MESSAGE

        await elevenlabs_ws.send(json.dumps(initial_config))
        if POC_FIRST_MESSAGE and "first_message" in initial_config["conversation_config_override"]["agent"]:
            logger.info(f"Initiale Konfiguration an Elevenlabs für {remote_addr} gesendet: Prompt='{POC_PROMPT}', First Message='{POC_FIRST_MESSAGE}'")
        else:
            logger.info(f"Initiale Konfiguration an Elevenlabs für {remote_addr} gesendet: Prompt='{POC_PROMPT}' (First Message NICHT gesendet oder war leer)")

        logger.info(f"PoC für {remote_addr}: Verbindung zu TalkDesk & ElevenLabs steht. Warte auf Nachrichten...")

        # Hauptschleife zum Verarbeiten von Nachrichten von TalkDesk
        async for message in talkdesk_ws:
            try:
                if isinstance(message, str):
                    msg_data = json.loads(message)
                    evt = msg_data.get("event")
                    if evt == "stop":
                        logger.info(f"'stop'-Event von TalkDesk {remote_addr} empfangen: {message[:200]}")
                        break # Beendet die Schleife und geht zum finally-Block
                    logger.debug(f"Ignoriere Text-Event '{evt}' von TalkDesk {remote_addr}")
                else: # Annahme: Binäre Nachrichten sind Audio-Daten
                    logger.debug(f"Ignoriere Binär-Nachricht von TalkDesk {remote_addr}")
                    # Hier später Audio an elevenlabs_ws weiterleiten:
                    # if elevenlabs_ws and elevenlabs_ws.state == State.OPEN:
                    #     await elevenlabs_ws.send(message)
            except json.JSONDecodeError:
                logger.warning(f"Konnte Nachricht von TalkDesk nicht als JSON parsen (ignoriere): {message[:200]}")
            except Exception as e:
                logger.warning(f"Fehler bei Verarbeitung weiterer Nachricht von TalkDesk {remote_addr}: {e}", exc_info=True)
        # Schleife beendet (entweder durch 'stop' oder weil TalkDesk die Verbindung geschlossen hat)

    except websockets.exceptions.ConnectionClosedOK:
        logger.info(f"TalkDesk WebSocket Verbindung zu {remote_addr} wurde sauber geschlossen (OK).")
    except websockets.exceptions.ConnectionClosedError as e:
        logger.error(f"TalkDesk WebSocket Verbindung zu {remote_addr} wurde unerwartet geschlossen (Error): {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Unerwarteter Fehler im Haupt-Handler für {remote_addr}: {e}", exc_info=True)
    finally:
        logger.info(f"Beende Handler für {remote_addr}. Räume auf...")
        if elevenlabs_ws:
            try:
                current_state = elevenlabs_ws.state # KORREKTES ATTRIBUT
                logger.info(f"Elevenlabs WebSocket state für {remote_addr}: {current_state} (OPEN ist {State.OPEN}, CLOSED ist {State.CLOSED})")

                if current_state == State.OPEN:
                    logger.info(f"Schließe Elevenlabs WebSocket (state == OPEN) für {remote_addr}...")
                    await asyncio.wait_for(elevenlabs_ws.close(code=1000, reason='Handler finished normally'), timeout=5.0)
                    logger.info(f"Elevenlabs WebSocket .close() für {remote_addr} aufgerufen. Warte auf Bestätigung...")
                    await asyncio.wait_for(elevenlabs_ws.wait_closed(), timeout=5.0) # Warten bis wirklich geschlossen
                    logger.info(f"Elevenlabs WebSocket für {remote_addr} erfolgreich geschlossen (via state und wait_closed).")
                elif current_state == State.CLOSING: # Wenn schon im Schließvorgang
                    logger.warning(f"Elevenlabs WebSocket für {remote_addr} in state {current_state} beim Aufräumen. Warte auf Abschluss...")
                    try:
                        await asyncio.wait_for(elevenlabs_ws.wait_closed(), timeout=5.0)
                        logger.info(f"Elevenlabs WebSocket für {remote_addr} ist nun nach Warten geschlossen (war {current_state}).")
                    except asyncio.TimeoutError:
                        logger.warning(f"Timeout beim Warten auf das Schließen des Elevenlabs WebSocket (war {current_state}) für {remote_addr}.")
                    except Exception as e_wait: # Andere Fehler beim Warten
                        logger.error(f"Fehler beim Warten auf Schließen des Elevenlabs WebSocket (war {current_state}) für {remote_addr}: {e_wait}", exc_info=True)
                elif current_state == State.CLOSED:
                    logger.info(f"Elevenlabs WebSocket für {remote_addr} war bereits geschlossen (state={current_state}).")
                elif current_state == State.CONNECTING:
                    logger.warning(f"Elevenlabs WebSocket für {remote_addr} war noch im Status CONNECTING. Versuche zu schließen und warte.")
                    try:
                        await asyncio.wait_for(elevenlabs_ws.close(code=1001, reason='Closing while still connecting'), timeout=5.0)
                        await asyncio.wait_for(elevenlabs_ws.wait_closed(), timeout=5.0)
                        logger.info(f"Elevenlabs WebSocket (war CONNECTING) für {remote_addr} geschlossen.")
                    except Exception as e_close_conn: # Fehler beim Schließen/Warten einer CONNECTING Verbindung
                        logger.error(f"Fehler beim Schließen/Warten einer CONNECTING Elevenlabs WS für {remote_addr}: {e_close_conn}", exc_info=True)
                else: # Unerwarteter Status
                    logger.warning(f"Elevenlabs WebSocket für {remote_addr} in unerwartetem state={current_state}. Prüfe .closed.done() als Fallback.")
                    if hasattr(elevenlabs_ws, 'closed') and not elevenlabs_ws.closed.done():
                        logger.info(f"Versuche Elevenlabs WebSocket (Fallback nach unerwartetem state) für {remote_addr} zu schließen...")
                        try:
                            await asyncio.wait_for(elevenlabs_ws.close(code=1008, reason='Closing from unexpected state - fallback'), timeout=5.0)
                            await asyncio.wait_for(elevenlabs_ws.wait_closed(), timeout=5.0) # Auch hier warten
                            logger.info(f"Elevenlabs WebSocket (Fallback nach unerwartetem state) für {remote_addr} geschlossen.")
                        except Exception as e_close_fb_unexpected: # Fehler im Fallback-Schließen
                            logger.error(f"Fehler beim Schließen/Warten (Fallback nach unerwartetem state) der Elevenlabs WS für {remote_addr}: {e_close_fb_unexpected}", exc_info=True)
                    else:
                        logger.info(f"Elevenlabs WebSocket (Fallback nach unerwartetem state) für {remote_addr} war bereits durch .closed.done() als geschlossen markiert oder .closed Attribut fehlt.")

            except websockets.exceptions.ConnectionClosed: # Fängt sowohl ConnectionClosedOK als ConnectionClosedError
                logger.info(f"Elevenlabs WebSocket für {remote_addr} war bereits geschlossen (fing ConnectionClosed Exception). State: {getattr(elevenlabs_ws, 'state', 'N/A')}")
            except AttributeError as e_attr: # Falls .state selbst fehlen sollte (sehr unwahrscheinlich für ein gültiges WS-Objekt)
                logger.error(f"Schwerwiegender AttributeError im ElevenLabs WS Cleanup für {remote_addr}: {e_attr}. elevenlabs_ws={elevenlabs_ws}", exc_info=True)
            except Exception as e_final_cleanup: # Andere, generelle Fehler beim ElevenLabs WS Cleanup
                logger.error(f"Genereller Fehler im ElevenLabs WS Cleanup für {remote_addr}: {e_final_cleanup}", exc_info=True)
        else:
            logger.info(f"Keine (initialisierte) Elevenlabs WebSocket Verbindung zum Schließen für {remote_addr} vorhanden.")

        # Aufräumen der TalkDesk-Verbindung
        # Dies ist oft nicht streng notwendig, wenn der Handler normal endet oder eine Exception wirft,
        # da das `websockets.serve` Framework die Verbindung schließt.
        # Aber zur expliziten Klarheit und für Fälle, wo der Handler "hängen" könnte ohne Exception:
        if talkdesk_ws and talkdesk_ws.open: # talkdesk_ws.open prüft, ob die Verbindung noch offen ist
            try:
                logger.info(f"Sicherstellen, dass die TalkDesk WebSocket Verbindung zu {remote_addr} geschlossen wird.")
                await asyncio.wait_for(talkdesk_ws.close(code=1000, reason="Handler cleanup complete"), timeout=2.0)
                logger.info(f"TalkDesk WebSocket Verbindung zu {remote_addr} explizit geschlossen.")
            except asyncio.TimeoutError:
                logger.warning(f"Timeout beim expliziten Schließen der TalkDesk Verbindung zu {remote_addr}.")
            except websockets.exceptions.ConnectionClosed:
                logger.info(f"TalkDesk WebSocket Verbindung zu {remote_addr} war bereits geschlossen, als explizites Schließen versucht wurde.")
            except Exception as e_td_close: # Andere Fehler beim Schließen der TalkDesk-Verbindung
                logger.error(f"Fehler beim expliziten Schließen der TalkDesk Verbindung zu {remote_addr}: {e_td_close}", exc_info=True)
        else:
            logger.info(f"TalkDesk WebSocket Verbindung zu {remote_addr} war beim finalen Aufräumen bereits geschlossen oder nicht initialisiert.")


async def main():
    try:
        import httpx # Sicherstellen, dass httpx verfügbar ist
    except ImportError:
        logger.error("Modul 'httpx' nicht gefunden! Stelle sicher, dass es in requirements.txt steht und installiert ist.")
        sys.exit(1)

    logger.info(f"Starte PoC WebSocket Server auf {WEBSOCKET_HOST}:{WEBSOCKET_PORT}")
    try:
        async with websockets.serve(handle_connection, WEBSOCKET_HOST, WEBSOCKET_PORT):
            await asyncio.Future()  # Hält den Server am Laufen, bis er gestoppt wird
    except OSError as e:
        if e.errno == 98: # EADDRINUSE
            logger.error(f"Port {WEBSOCKET_PORT} wird bereits verwendet!")
        else:
            logger.error(f"Server konnte nicht gestartet werden (OS Error {e.errno}): {e}", exc_info=True)
        sys.exit(1) # Beende das Skript bei Server-Startfehler
    except Exception as e:
        logger.error(f"Server konnte nicht gestartet werden oder ist abgestürzt: {e}", exc_info=True)
        sys.exit(1) # Beende das Skript bei Server-Startfehler

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server durch Benutzer (Ctrl+C) gestoppt.")
    finally:
        logger.info("Server wird beendet.")

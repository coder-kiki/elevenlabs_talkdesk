import asyncio
import websockets # Importiere das Hauptmodul zuerst
import os
import logging
import json
import base64 # Wird jetzt benötigt für Audio
import httpx
import sys
from websockets.connection import State
from websockets.exceptions import ConnectionClosed, ConnectionClosedOK, ConnectionClosedError

# SCRIPT VERSION FÜR LOGGING
SCRIPT_VERSION = "3.16 - Implemented Audio Streaming Logic" # NEU

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

# === AUDIO STREAMING TASK FUNKTIONEN (Innerhalb von handle_connection definiert) ===
async def stream_talkdesk_to_elevenlabs(td_ws, el_ws, remote_addr_log):
    """Empfängt Audio von TalkDesk und sendet es an ElevenLabs."""
    logger.info(f"[{remote_addr_log}] Starte Task: TalkDesk -> ElevenLabs Audio Streaming")
    message_counter = 0
    try:
        async for message_str in td_ws:
            message_counter += 1
            try:
                data = json.loads(message_str)
                event = data.get("event")

                if event == "media":
                    payload = data.get("media", {}).get("payload")
                    if payload:
                        try:
                            # Schritt 1: Base64 dekodieren (rohe µ-law Bytes)
                            # raw_audio_bytes = base64.b64decode(payload)
                            # Schritt 2: Für ElevenLabs vorbereiten (JSON mit Base64)
                            # Wichtig: Wir verwenden den *originalen* Base64-String von TalkDesk!
                            #          Kein Dekodieren/Rekodieren nötig, wenn EL Base64 erwartet.
                            elevenlabs_message = {
                                "user_audio_chunk": payload # Direkte Weitergabe des Base64-Strings
                            }
                            # Schritt 3: An ElevenLabs senden, wenn Verbindung offen
                            if el_ws and el_ws.state == State.OPEN:
                                await el_ws.send(json.dumps(elevenlabs_message))
                                logger.debug(f"[{remote_addr_log}] TD -> EL: Media Chunk #{message_counter} (Payload L: {len(payload)}) an ElevenLabs gesendet.")
                            else:
                                logger.warning(f"[{remote_addr_log}] TD -> EL: ElevenLabs WS nicht offen (State: {el_ws.state if el_ws else 'None'}), konnte Chunk #{message_counter} nicht senden.")
                                # Optional: Task beenden, wenn EL nicht mehr offen ist
                                # break
                        except Exception as e_inner:
                             logger.error(f"[{remote_addr_log}] TD -> EL: Fehler bei Verarbeitung/Senden von Media Chunk #{message_counter}: {e_inner}", exc_info=True)

                elif event == "stop":
                    logger.info(f"[{remote_addr_log}] TD -> EL: Stop Event in Stream-Task empfangen, beende Task.")
                    break # Task beenden

                # Andere Events wie 'connected', 'start' werden ignoriert
                # (sie wurden bereits im Hauptteil behandelt oder sind hier irrelevant)

            except json.JSONDecodeError:
                 logger.warning(f"[{remote_addr_log}] TD -> EL: Konnte Nachricht #{message_counter} nicht als JSON parsen: {message_str[:100]}")
            except Exception as e:
                 logger.error(f"[{remote_addr_log}] TD -> EL: Fehler bei Nachrichtenverarbeitung #{message_counter}: {e}", exc_info=True)

    except ConnectionClosedOK:
        logger.info(f"[{remote_addr_log}] TD -> EL: TalkDesk Verbindung sauber geschlossen (OK).")
    except ConnectionClosedError as e:
        logger.warning(f"[{remote_addr_log}] TD -> EL: TalkDesk Verbindung unerwartet geschlossen (Error): {e}")
    except Exception as e:
        logger.error(f"[{remote_addr_log}] TD -> EL: Unerwarteter Fehler im Streaming Task: {e}", exc_info=True)
    finally:
        logger.info(f"[{remote_addr_log}] Beende Task: TalkDesk -> ElevenLabs Audio Streaming (Nach {message_counter} Nachrichten)")

async def stream_elevenlabs_to_talkdesk(td_ws, el_ws, stream_sid_from_td, remote_addr_log):
    """Empfängt Audio von ElevenLabs und sendet es an TalkDesk."""
    logger.info(f"[{remote_addr_log}] Starte Task: ElevenLabs -> TalkDesk Audio Streaming")
    message_counter = 0
    try:
        async for message_str in el_ws:
            message_counter += 1
            try:
                data = json.loads(message_str)
                msg_type = data.get("type")

                if msg_type == "audio":
                    b64_audio = data.get("audio_event", {}).get("audio_base_64")
                    if b64_audio:
                        # Schritt 1: TalkDesk Nachricht erstellen
                        talkdesk_message = {
                           "event": "media",
                           "streamSid": stream_sid_from_td, # Wichtig: Den korrekten StreamSid verwenden
                           "media": {"payload": b64_audio} # Direkte Weitergabe des Base64-Strings
                        }
                        # Schritt 2: An TalkDesk senden, wenn Verbindung offen
                        if td_ws and td_ws.state == State.OPEN:
                           await td_ws.send(json.dumps(talkdesk_message))
                           logger.debug(f"[{remote_addr_log}] EL -> TD: Audio Chunk #{message_counter} (Payload L: {len(b64_audio)}) an TalkDesk gesendet.")
                        else:
                            logger.warning(f"[{remote_addr_log}] EL -> TD: TalkDesk WS nicht offen (State: {td_ws.state if td_ws else 'None'}), konnte Audio Chunk #{message_counter} nicht senden.")
                            # Task beenden, wenn TalkDesk nicht mehr offen ist
                            break
                    else:
                        logger.warning(f"[{remote_addr_log}] EL -> TD: 'audio' Event ohne 'audio_base_64' empfangen: {data}")

                elif msg_type == "agent_response":
                     logger.info(f"[{remote_addr_log}] EL -> TD: Agent Response Text: {data.get('agent_response_event', {}).get('agent_response')}")
                elif msg_type == "user_transcript":
                     logger.info(f"[{remote_addr_log}] EL -> TD: User Transcript: {data.get('user_transcription_event', {}).get('user_transcript')}")
                elif msg_type == "ping":
                     # Optional: Ping-Nachrichten loggen oder ignorieren
                     logger.debug(f"[{remote_addr_log}] EL -> TD: Ping Event empfangen: {data}")
                     # Hinweis: Der websockets Client sollte Pongs automatisch senden.
                # Andere Events von EL (interruption, vad_score, etc.) könnten hier behandelt werden
                else:
                    logger.debug(f"[{remote_addr_log}] EL -> TD: Ignoriere Nachrichtentyp '{msg_type}': {data}")

            except json.JSONDecodeError:
                 logger.warning(f"[{remote_addr_log}] EL -> TD: Konnte Nachricht #{message_counter} nicht als JSON parsen: {message_str[:100]}")
            except Exception as e:
                 logger.error(f"[{remote_addr_log}] EL -> TD: Fehler bei Nachrichtenverarbeitung #{message_counter}: {e}", exc_info=True)

    except ConnectionClosedOK:
        logger.info(f"[{remote_addr_log}] EL -> TD: ElevenLabs Verbindung sauber geschlossen (OK).")
    except ConnectionClosedError as e:
        logger.warning(f"[{remote_addr_log}] EL -> TD: ElevenLabs Verbindung unerwartet geschlossen (Error): {e}")
    except Exception as e:
        logger.error(f"[{remote_addr_log}] EL -> TD: Unerwarteter Fehler im Streaming Task: {e}", exc_info=True)
    finally:
        logger.info(f"[{remote_addr_log}] Beende Task: ElevenLabs -> TalkDesk Audio Streaming (Nach {message_counter} Nachrichten)")
# === ENDE AUDIO STREAMING TASK FUNKTIONEN ===

async def handle_connection(talkdesk_ws):
    remote_addr = talkdesk_ws.remote_address
    remote_addr_log = f"{remote_addr[0]}:{remote_addr[1]}"
    logger.info(f"+++ Neue Verbindung (möglicherweise TalkDesk): {remote_addr_log}")
    elevenlabs_ws = None
    stream_sid = None
    audio_tasks = []

    try:
        # === Initialisierungsphase (wie in v3.15) ===
        start_data = None
        start_message_str_for_logging = None
        message_count = 0
        max_initial_messages = 5

        while message_count < max_initial_messages:
            message_count += 1
            logger.debug(f"Warte auf Nachricht #{message_count} von Client {remote_addr_log}...")
            try:
                message_str = await asyncio.wait_for(talkdesk_ws.recv(), timeout=10.0)
                logger.info(f"Nachricht #{message_count} von Client {remote_addr_log} erhalten.")
                logger.debug(f"Raw Message #{message_count}: {message_str}")
            except asyncio.TimeoutError:
                logger.error(f"Timeout (10s) beim Warten auf Nachricht #{message_count} von {remote_addr_log}.")
                if message_count == 1 and not start_data: return
                continue
            except ConnectionClosed:
                logger.info(f"Verbindung von Client {remote_addr_log} geschlossen während Warten auf Nachricht #{message_count}.")
                return
            except Exception as e:
                logger.error(f"Fehler Empfangen Nachricht #{message_count} von {remote_addr_log}: {e}", exc_info=True)
                return
            try:
                data = json.loads(message_str)
                event = data.get("event")
                if event == "connected":
                    logger.info(f"'{event}'-Event von {remote_addr_log} empfangen und ignoriert.")
                    continue
                elif event == "start":
                    logger.info(f"'start'-Event von {remote_addr_log} gefunden!")
                    start_data = data
                    start_message_str_for_logging = message_str
                    break
                elif event == "media":
                    logger.warning(f"Unerwartetes 'media'-Event von {remote_addr_log} während Initialisierung. Ignoriere.")
                    continue
                else:
                    logger.warning(f"Unbekanntes Event '{event}' von {remote_addr_log}. Ignoriere: {message_str[:200]}...")
                    continue
            except json.JSONDecodeError:
                logger.error(f"Konnte Nachricht #{message_count} von {remote_addr_log} nicht als JSON parsen: {message_str[:200]}...")
                continue
            except Exception as e:
                logger.error(f"Fehler Verarbeiten Nachricht #{message_count} von {remote_addr_log}: {e}", exc_info=True)
                return

        if not start_data:
            logger.error(f"Kein 'start'-Event von {remote_addr_log} nach {max_initial_messages} Nachrichten. Beende.")
            return

        logger.info(f"--- Vollständige Start-Nachricht von {remote_addr_log} ---")
        logger.info(start_message_str_for_logging)
        logger.info(f"--- Verarbeitete Start-Daten (pretty) ---")
        logger.info(json.dumps(start_data, indent=2))

        start_info = start_data.get("start", {})
        if not start_info:
            logger.error(f"Start-Event von {remote_addr_log} ohne 'start'-Objekt: {start_data}")
            return

        stream_sid = start_info.get("streamSid")
        call_sid = start_info.get("callSid", f"UnknownCall_{remote_addr_log}")
        account_sid_from_start = start_info.get("accountSid")
        media_format = start_info.get("mediaFormat", {})
        custom_params = start_info.get("customParameters", {})

        if not stream_sid:
            logger.error(f"Kein 'streamSid' im 'start'-Objekt von {remote_addr_log}: {start_info}")
            return

        logger.info(f"Anruf gestartet: CallSid='{call_sid}', StreamSid='{stream_sid}', AccountSid(Start)='{account_sid_from_start}'")
        logger.info(f"Media Format von {remote_addr_log}: {media_format}")
        logger.info(f"Custom Parameters von {remote_addr_log}: {custom_params}")

        signed_url = await get_elevenlabs_signed_url()
        if not signed_url:
            logger.error(f"Abbruch, da keine Signed URL für {remote_addr_log} erhalten werden konnte.")
            return

        elevenlabs_ws = await websockets.connect(signed_url)
        logger.info(f"Verbindung zu Elevenlabs WebSocket für {remote_addr_log} hergestellt.")

        initial_config = {
            "type": "conversation_initiation_client_data",
            "conversation_config_override": {"agent": {"prompt": {"prompt": POC_PROMPT}}}
        }
        if POC_FIRST_MESSAGE:
            initial_config["conversation_config_override"]["agent"]["first_message"] = POC_FIRST_MESSAGE

        await elevenlabs_ws.send(json.dumps(initial_config))
        if POC_FIRST_MESSAGE and "first_message" in initial_config["conversation_config_override"]["agent"]:
            logger.info(f"Initiale Konfiguration an Elevenlabs für {remote_addr_log} gesendet: Prompt='{POC_PROMPT}', First Message='{POC_FIRST_MESSAGE}'")
        else:
            logger.info(f"Initiale Konfiguration an Elevenlabs für {remote_addr_log} gesendet: Prompt='{POC_PROMPT}' (First Message NICHT gesendet oder war leer)")

        logger.info(f"Verbindungen stehen für {remote_addr_log}. Starte Audio-Streaming-Tasks...")

        # Starte die nebenläufigen Audio-Streaming-Tasks
        task_td_to_el = asyncio.create_task(
            stream_talkdesk_to_elevenlabs(talkdesk_ws, elevenlabs_ws, remote_addr_log),
            name=f"TD_to_EL_{remote_addr_log}" # Optional: Name für besseres Debugging
        )
        task_el_to_td = asyncio.create_task(
            stream_elevenlabs_to_talkdesk(talkdesk_ws, elevenlabs_ws, stream_sid, remote_addr_log),
            name=f"EL_to_TD_{remote_addr_log}" # Optional: Name für besseres Debugging
        )
        audio_tasks = [task_td_to_el, task_el_to_td]

        # Warte, bis eine der Tasks endet (oder Hauptverbindung schließt)
        logger.info(f"[{remote_addr_log}] Audio-Tasks gestartet. Warte auf Beendigung einer Task oder Verbindungsschluss.")
        done, pending = await asyncio.wait(audio_tasks, return_when=asyncio.FIRST_COMPLETED)

        logger.info(f"[{remote_addr_log}] Mindestens eine Audio-Task beendet. Done: {len(done)}, Pending: {len(pending)}")
        for task in done:
            try:
                 result = task.result() # Holt das Ergebnis oder wirft die Exception der Task erneut
                 logger.info(f"[{remote_addr_log}] Abgeschlossene Task {task.get_name()} Ergebnis: {result}")
            except asyncio.CancelledError:
                 logger.info(f"[{remote_addr_log}] Abgeschlossene Task {task.get_name()} wurde gecancelt.")
            except Exception as task_exc:
                 logger.error(f"[{remote_addr_log}] Abgeschlossene Task {task.get_name()} endete mit Fehler: {task_exc}", exc_info=True)

        # Beende die verbleibenden Tasks, falls eine Task normal oder mit Fehler endete
        for task in pending:
            if not task.done():
                logger.info(f"[{remote_addr_log}] Cancelling pending audio task: {task.get_name()}")
                task.cancel()

    except ConnectionClosedOK:
        logger.info(f"Haupt-TalkDesk WebSocket Verbindung zu {remote_addr_log} wurde sauber geschlossen (OK).")
    except ConnectionClosedError as e:
        logger.error(f"Haupt-TalkDesk WebSocket Verbindung zu {remote_addr_log} wurde unerwartet geschlossen (Error): {e}", exc_info=False)
    except Exception as e:
        logger.error(f"Unerwarteter Fehler im Haupt-Handler für {remote_addr_log}: {e}", exc_info=True)
    finally:
        logger.info(f"Beende Handler für {remote_addr_log}. Räume auf...")

        # Sicherstellen, dass Audio-Tasks beendet werden
        logger.info(f"[{remote_addr_log}] Sicherstellen, dass Audio-Tasks beendet werden im finally Block...")
        if audio_tasks: # Nur wenn die Liste nicht leer ist
            for task in audio_tasks:
                if task and not task.done():
                    logger.info(f"[{remote_addr_log}] Cancelling audio task im finally: {task.get_name()}")
                    task.cancel()
            # Warte darauf, dass alle Tasks (auch die gecancelten) abgeschlossen sind
            results = await asyncio.gather(*audio_tasks, return_exceptions=True)
            logger.info(f"[{remote_addr_log}] Audio streaming task results after final cleanup gather: {results}")
        else:
             logger.info(f"[{remote_addr_log}] Keine Audio-Tasks zum finalen Aufräumen vorhanden.")

        # Aufräumen der ElevenLabs-Verbindung (Code von v3.14)
        if elevenlabs_ws:
            try:
                current_state = elevenlabs_ws.state
                logger.info(f"Elevenlabs WebSocket state für {remote_addr_log}: {current_state} (OPEN ist {State.OPEN}, CLOSED ist {State.CLOSED})")

                if current_state == State.OPEN:
                    logger.info(f"Schließe Elevenlabs WebSocket (state == OPEN) für {remote_addr_log}...")
                    await asyncio.wait_for(elevenlabs_ws.close(code=1000, reason='Handler finished normally'), timeout=5.0)
                    logger.info(f"Elevenlabs WebSocket .close() für {remote_addr_log} aufgerufen. Warte auf Bestätigung...")
                    await asyncio.wait_for(elevenlabs_ws.wait_closed(), timeout=5.0)
                    logger.info(f"Elevenlabs WebSocket für {remote_addr_log} erfolgreich geschlossen (via state und wait_closed).")
                elif current_state == State.CLOSING:
                    logger.warning(f"Elevenlabs WebSocket für {remote_addr_log} in state {current_state} beim Aufräumen. Warte auf Abschluss...")
                    try:
                        await asyncio.wait_for(elevenlabs_ws.wait_closed(), timeout=5.0)
                        logger.info(f"Elevenlabs WebSocket für {remote_addr_log} ist nun nach Warten geschlossen (war {current_state}).")
                    except asyncio.TimeoutError:
                        logger.warning(f"Timeout beim Warten auf das Schließen des Elevenlabs WebSocket (war {current_state}) für {remote_addr_log}.")
                    except Exception as e_wait:
                        logger.error(f"Fehler beim Warten auf Schließen des Elevenlabs WebSocket (war {current_state}) für {remote_addr_log}: {e_wait}", exc_info=True)
                elif current_state == State.CLOSED:
                    logger.info(f"Elevenlabs WebSocket für {remote_addr_log} war bereits geschlossen (state={current_state}).")
                elif current_state == State.CONNECTING:
                    logger.warning(f"Elevenlabs WebSocket für {remote_addr_log} war noch im Status CONNECTING. Versuche zu schließen und warte.")
                    try:
                        await asyncio.wait_for(elevenlabs_ws.close(code=1001, reason='Closing while still connecting'), timeout=5.0)
                        await asyncio.wait_for(elevenlabs_ws.wait_closed(), timeout=5.0)
                        logger.info(f"Elevenlabs WebSocket (war CONNECTING) für {remote_addr_log} geschlossen.")
                    except Exception as e_close_conn:
                        logger.error(f"Fehler beim Schließen/Warten einer CONNECTING Elevenlabs WS für {remote_addr_log}: {e_close_conn}", exc_info=True)
                else:
                    logger.warning(f"Elevenlabs WebSocket für {remote_addr_log} in unerwartetem state={current_state}. Prüfe .closed.done() als Fallback.")
                    if hasattr(elevenlabs_ws, 'closed') and not elevenlabs_ws.closed.done():
                        logger.info(f"Versuche Elevenlabs WebSocket (Fallback nach unerwartetem state) für {remote_addr_log} zu schließen...")
                        try:
                            await asyncio.wait_for(elevenlabs_ws.close(code=1008, reason='Closing from unexpected state - fallback'), timeout=5.0)
                            await asyncio.wait_for(elevenlabs_ws.wait_closed(), timeout=5.0)
                            logger.info(f"Elevenlabs WebSocket (Fallback nach unerwartetem state) für {remote_addr_log} geschlossen.")
                        except Exception as e_close_fb_unexpected:
                            logger.error(f"Fehler beim Schließen/Warten (Fallback nach unerwartetem state) der Elevenlabs WS für {remote_addr_log}: {e_close_fb_unexpected}", exc_info=True)
                    else:
                        logger.info(f"Elevenlabs WebSocket (Fallback nach unerwartetem state) für {remote_addr_log} war bereits durch .closed.done() als geschlossen markiert oder .closed fehlt.")

            except ConnectionClosed:
                logger.info(f"Elevenlabs WebSocket für {remote_addr_log} war bereits geschlossen (fing ConnectionClosed Exception). State: {getattr(elevenlabs_ws, 'state', 'N/A')}")
            except AttributeError as e_attr:
                logger.error(f"Schwerwiegender AttributeError im ElevenLabs WS Cleanup für {remote_addr_log}: {e_attr}. elevenlabs_ws={elevenlabs_ws}", exc_info=True)
            except Exception as e_final_cleanup:
                logger.error(f"Genereller Fehler im ElevenLabs WS Cleanup für {remote_addr_log}: {e_final_cleanup}", exc_info=True)
        else:
            logger.info(f"Keine (initialisierte) Elevenlabs WebSocket Verbindung zum Schließen für {remote_addr_log} vorhanden.")

        # Aufräumen der TalkDesk-Verbindung (Code von v3.14)
        if talkdesk_ws:
            try:
                if talkdesk_ws.state == State.OPEN:
                    logger.info(f"TalkDesk WebSocket (state == OPEN) zu {remote_addr_log} wird explizit geschlossen.")
                    await asyncio.wait_for(talkdesk_ws.close(code=1000, reason="Handler cleanup complete"), timeout=2.0)
                    logger.info(f"TalkDesk WebSocket Verbindung zu {remote_addr_log} explizit geschlossen.")
                elif talkdesk_ws.state == State.CLOSED:
                    logger.info(f"TalkDesk WebSocket Verbindung zu {remote_addr_log} war bereits geschlossen (state={talkdesk_ws.state}).")
                else:
                    logger.warning(f"TalkDesk WebSocket zu {remote_addr_log} in unerwartetem state={talkdesk_ws.state} beim Aufräumen. Übergabe an Server-Framework zum Abschluss.")
            except AttributeError as e_attr_td:
                 logger.error(f"AttributeError beim Zugriff auf talkdesk_ws.state für {remote_addr_log}. talkdesk_ws={talkdesk_ws}", exc_info=True)
            except ConnectionClosed:
                logger.info(f"TalkDesk WebSocket Verbindung zu {remote_addr_log} war bereits geschlossen (ConnectionClosed Exception), als explizites Schließen versucht wurde.")
            except asyncio.TimeoutError:
                logger.warning(f"Timeout beim expliziten Schließen der TalkDesk Verbindung zu {remote_addr_log}.")
            except Exception as e_td_close:
                logger.error(f"Genereller Fehler beim expliziten Schließen der TalkDesk Verbindung zu {remote_addr_log}: {e_td_close}", exc_info=True)
        else:
            logger.info(f"TalkDesk WebSocket Verbindung zu {remote_addr_log} war beim finalen Aufräumen nicht initialisiert (None).")


async def main():
    try:
        import httpx
    except ImportError:
        logger.error("Modul 'httpx' nicht gefunden! Stelle sicher, dass es in requirements.txt steht und installiert ist.")
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
        sys.exit(1)
    except Exception as e:
        logger.error(f"Server konnte nicht gestartet werden oder ist abgestürzt: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server durch Benutzer (Ctrl+C) gestoppt.")
    finally:
        logger.info("Server wird beendet.")

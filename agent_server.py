import asyncio
import websockets
import os
import logging
import json
import httpx
import sys
from websockets.connection import State
from websockets.exceptions import ConnectionClosed, ConnectionClosedOK, ConnectionClosedError

SCRIPT_VERSION = "4.1 - ENV-Ready Multi-Agent Support + Refactored Handler"
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

if not ELEVENLABS_API_KEY or not ELEVENLABS_AGENT_ID:
    logger.error("API Key oder Agent ID fehlt! Abbruch.")
    sys.exit(1)

logger.info(f"Konfiguration: Agent-ID={ELEVENLABS_AGENT_ID}, Port={WEBSOCKET_PORT}")
logger.info(f"Prompt: {POC_PROMPT}")
logger.info(f"First Message: {POC_FIRST_MESSAGE or '[leer]'}")

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
            return r.json().get("signed_url")
    except Exception as e:
        logger.error(f"Signed URL Fehler: {e}")
        return None

# AUDIO BRIDGING
async def stream_talkdesk_to_elevenlabs(td_ws, el_ws):
    try:
        async for message_str in td_ws:
            data = json.loads(message_str)
            if data.get("event") == "media":
                payload = data.get("media", {}).get("payload")
                if payload:
                    await el_ws.send(json.dumps({"user_audio_chunk": payload}))
    except Exception as e:
        logger.error(f"Fehler Talkdesk->ElevenLabs: {e}")

async def stream_elevenlabs_to_talkdesk(td_ws, el_ws, stream_sid):
    try:
        async for message_str in el_ws:
            data = json.loads(message_str)
            if data.get("type") == "audio":
                b64_audio = data.get("audio_event", {}).get("audio_base_64")
                if b64_audio:
                    await td_ws.send(json.dumps({
                        "event": "media",
                        "streamSid": stream_sid,
                        "media": {"payload": b64_audio}
                    }))
    except Exception as e:
        logger.error(f"Fehler ElevenLabs->Talkdesk: {e}")

# REFACOTRIERTER CONNECTION HANDLER
async def handle_connection(talkdesk_ws):
    remote = f"{talkdesk_ws.remote_address[0]}:{talkdesk_ws.remote_address[1]}"
    logger.info(f"Neue Verbindung von {remote}")

    try:
        stream_sid = None
        # START-NACHRICHT PARSEN
        for _ in range(5):
            msg = await asyncio.wait_for(talkdesk_ws.recv(), timeout=10)
            data = json.loads(msg)
            if data.get("event") == "start":
                stream_sid = data["start"].get("streamSid")
                break
        if not stream_sid:
            logger.error("Kein StreamSid erhalten. Abbruch.")
            return

        signed_url = await get_elevenlabs_signed_url()
        if not signed_url:
            return
        el_ws = await websockets.connect(signed_url)

        initial_config = build_initial_config()
        await el_ws.send(json.dumps(initial_config))

        # TASKS STARTEN
        task1 = asyncio.create_task(stream_talkdesk_to_elevenlabs(talkdesk_ws, el_ws))
        task2 = asyncio.create_task(stream_elevenlabs_to_talkdesk(talkdesk_ws, el_ws, stream_sid))
        await asyncio.wait([task1, task2], return_when=asyncio.FIRST_COMPLETED)

    except Exception as e:
        logger.error(f"Fehler in Handler: {e}", exc_info=True)
    finally:
        logger.info("Verbindung wird geschlossen")
        if talkdesk_ws and talkdesk_ws.open:
            await talkdesk_ws.close()

# SERVER START
async def main():
    logger.info(f"Starte WebSocket-Server auf {WEBSOCKET_HOST}:{WEBSOCKET_PORT}")
    async with websockets.serve(handle_connection, WEBSOCKET_HOST, WEBSOCKET_PORT):
        await asyncio.Future()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Manueller Abbruch durch User.")

"""Shliach agent: voice server for Plivo Audio Streaming.

Flow: the 845 number's Plivo application fetches /voice, Plivo opens a bidirectional
websocket to /media, this server detects when the caller stops talking, transcribes
with local Whisper, asks Claude (or Gemini) what to do, synthesizes the reply, and
streams it back to the caller as base64 audio frames.
"""
import asyncio
import base64
import io
import json
import os
import wave

import audioop
from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect

from agent import Agent
from tts import synthesize_wav

PUBLIC_WSS_URL = os.environ.get("PUBLIC_WSS_URL", "wss://REPLACE-ME.example/media")
SILENCE_MS = int(os.environ.get("SILENCE_MS", "800"))
GREETING = os.environ.get("GREETING", "Hello, how can I help you today?")
RMS_THRESHOLD = 500          # mulaw energy above which the caller is "talking"
CHUNK_MS = 20                # 20ms frames at 8kHz

app = FastAPI()
transcriber = None           # lazy: Whisper loads on first call
agent = None                 # lazy: needs an API key in the env


def mulaw_to_wav(ulaw: bytes, rate: int = 8000) -> bytes:
    pcm = audioop.ulaw2lin(ulaw, 2)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate); w.writeframes(pcm)
    return buf.getvalue()


def wav_to_mulaw(wav_bytes: bytes) -> bytes:
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        pcm = w.readframes(w.getnframes())
        rate = w.getframerate()
    pcm8, _ = audioop.ratecv(pcm, 2, 1, rate, 8000, None)
    return audioop.lin2ulaw(pcm8, 2)


@app.get("/voice")
@app.post("/voice")
async def voice(_: Request) -> Response:
    # Plivo XML: speak the greeting, then hand the call to the bidirectional stream.
    speak_voice = os.environ.get("PLIVO_SPEAK_VOICE", "Polly.Matthew")
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<Response><Speak voice="{speak_voice}" language="en-US">{GREETING}</Speak>'
        f'<Stream bidirectional="true" keepCallAlive="true" '
        f'contentType="audio/x-mulaw;rate=8000">{PUBLIC_WSS_URL}</Stream></Response>'
    )
    return Response(xml, media_type="text/xml")


def get_agent():
    global agent
    if agent is None:
        agent = Agent()
    return agent


async def respond(ws: WebSocket, caller_ulaw: bytes) -> None:
    global transcriber
    wav = mulaw_to_wav(caller_ulaw)
    if transcriber is None:
        from stt import Transcriber
        transcriber = Transcriber()
    heard = await asyncio.to_thread(transcriber.transcribe, wav)
    if not heard:
        return
    print(f"caller said: {heard}")
    reply = await asyncio.to_thread(get_agent().run, heard)
    if not reply:
        return
    print(f"agent says: {reply}")
    mulaw = wav_to_mulaw(await asyncio.to_thread(synthesize_wav, reply))
    for i in range(0, len(mulaw), 160):  # 20ms frames of 8kHz mulaw
        await ws.send_text(json.dumps({
            "event": "playAudio",
            "media": {"contentType": "audio/x-l16", "sampleRate": "8000",
                      "payload": base64.b64encode(mulaw[i:i + 160]).decode()},
        }))


async def handle_turn(ws, spoken, state):
    try:
        await respond(ws, spoken)
    finally:
        state["busy"] = False


@app.websocket("/media")
async def media(ws: WebSocket) -> None:
    await ws.accept()
    buf, silence_ms = b"", 0
    state = {"busy": False}
    try:
        while True:
            msg = json.loads(await ws.receive_text())
            ev = msg.get("event")
            if ev == "start":          # Plivo connection event; nothing to set up
                continue
            elif ev == "stop":
                break
            elif ev == "media":
                chunk = base64.b64decode(msg["media"]["payload"])
                if audioop.rms(chunk, 1) > RMS_THRESHOLD:
                    buf, silence_ms = buf + chunk, 0
                elif buf:
                    silence_ms += CHUNK_MS
                    if silence_ms >= SILENCE_MS and not state["busy"]:
                        state["busy"] = True
                        spoken, buf, silence_ms = buf, b"", 0
                        asyncio.create_task(handle_turn(ws, spoken, state))
    except WebSocketDisconnect:
        pass

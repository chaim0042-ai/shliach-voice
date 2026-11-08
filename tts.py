"""Text-to-speech. Gemini TTS speaks Yiddish well; gTTS is the free English fallback."""
import base64
import io
import os
import subprocess
import tempfile
import wave

import requests

GEMINI_TTS_MODEL = "gemini-2.5-flash-preview-tts"


def synthesize_wav(text: str) -> bytes:
    if os.environ.get("GOOGLE_API_KEY"):
        return gemini_tts(text)
    return gtts_wav(text)


def gemini_tts(text: str) -> bytes:
    r = requests.post(
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_TTS_MODEL}:generateContent",
        params={"key": os.environ["GOOGLE_API_KEY"]},
        json={"contents": [{"parts": [{"text": text}]}],
              "generationConfig": {"responseModalities": ["AUDIO"],
                                   "speechConfig": {"voiceConfig": {
                                       "prebuiltVoiceConfig": {"voiceName": "Kore"}}}}},
        timeout=60,
    )
    r.raise_for_status()
    pcm = base64.b64decode(
        r.json()["candidates"][0]["content"]["parts"][0]["inlineData"]["data"])
    return pcm_to_wav(pcm, 24000)


def pcm_to_wav(pcm: bytes, rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate); w.writeframes(pcm)
    return buf.getvalue()


def gtts_wav(text: str, lang: str = "en") -> bytes:
    from gtts import gTTS
    fp = io.BytesIO()
    gTTS(text=text, lang=lang).write_to_fp(fp)
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(fp.getvalue())
        mp3_path = f.name
    return subprocess.run(
        ["ffmpeg", "-y", "-i", mp3_path, "-ar", "8000", "-ac", "1", "-f", "wav", "pipe:1"],
        capture_output=True, check=True).stdout

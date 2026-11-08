"""Speech-to-text: local Whisper (faster-whisper). Free, runs on CPU, understands Yiddish."""
import io
import os

from faster_whisper import WhisperModel


class Transcriber:
    def __init__(self, model_size=None):
        self.model = WhisperModel(
            model_size or os.environ.get("STT_MODEL", "base"),
            device="cpu",
            compute_type="int8",
        )

    def transcribe(self, wav_bytes: bytes) -> str:
        segments, _ = self.model.transcribe(io.BytesIO(wav_bytes))
        return " ".join(s.text.strip() for s in segments).strip()

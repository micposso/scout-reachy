"""Speech-to-text via faster-whisper (local, offline).

Whisper expects 16 kHz mono float32; we resample any incoming AudioClip to that.
The model downloads from HuggingFace on first use, then is cached locally.
"""

from __future__ import annotations

import logging
import os
import time

import numpy as np

# Windows without Developer Mode can't symlink; the HF cache still works fine, so
# silence the noisy warning.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from ..config import settings
from .audio import AudioClip

logger = logging.getLogger(__name__)

WHISPER_RATE = 16000


class STTError(RuntimeError):
    pass


class SpeechToText:
    def __init__(
        self,
        model_name: str | None = None,
        *,
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:
        from faster_whisper import WhisperModel  # lazy import

        self.model_name = model_name or settings.whisper_model
        logger.info("Loading faster-whisper model %s (%s/%s)",
                    self.model_name, device, compute_type)
        # First load downloads from HuggingFace, which can intermittently reset
        # the connection on some networks. Retry transient failures; once cached,
        # subsequent loads are offline and instant.
        attempts = 4
        last_exc: Exception | None = None
        for i in range(attempts):
            try:
                self._model = WhisperModel(
                    self.model_name, device=device, compute_type=compute_type
                )
                break
            except Exception as exc:  # download failure, bad model name, etc.
                last_exc = exc
                if i < attempts - 1:
                    logger.warning("Whisper load attempt %d/%d failed: %s",
                                   i + 1, attempts, str(exc)[:120])
                    time.sleep(2 * (i + 1))
        else:
            raise STTError(
                f"Could not load Whisper model '{self.model_name}' after "
                f"{attempts} attempts: {last_exc}"
            ) from last_exc

    def transcribe(self, clip: AudioClip, *, language: str | None = "en") -> str:
        """Transcribe a mono float32 AudioClip to text."""
        if len(clip.samples) == 0:
            return ""
        audio = clip.resampled(WHISPER_RATE).samples.astype(np.float32)
        segments, _info = self._model.transcribe(
            audio,
            language=language,
            beam_size=5,
            vad_filter=True,  # drop leading/trailing silence
        )
        return "".join(seg.text for seg in segments).strip()

    def transcribe_file(self, path: str, *, language: str | None = "en") -> str:
        segments, _info = self._model.transcribe(path, language=language, beam_size=5)
        return "".join(seg.text for seg in segments).strip()

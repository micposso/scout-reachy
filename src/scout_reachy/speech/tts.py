"""ElevenLabs announcer TTS with character-level timestamps.

We request pcm_16000 (raw 16 kHz 16-bit PCM) so the audio matches the robot
speaker rate exactly — no decode, no resample. If the account tier rejects PCM
output we fall back to mp3 + decode + resample automatically.

All ElevenLabs SDK field access is isolated here on purpose.
"""

from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass

import numpy as np

from ..config import settings
from ..voice.audio import AudioClip

logger = logging.getLogger(__name__)

ROBOT_RATE = 16000

# Announcer delivery, tuned for INTELLIGIBILITY over theatrics: `style` is the
# exaggeration dial — high style over-acts the voice and slurs its accent into
# something hard to follow, so it's kept low. Higher `stability` keeps delivery
# consistent (low stability makes it wander), and speed < 1.0 slows the pace.
# If it's still hard to understand, the voice itself may be accented — try a
# clear English voice in ELEVENLABS_VOICE_ID. Tune in scripts/smoke_elevenlabs.py.
VOICE_SETTINGS = {
    "stability": 0.7,
    "similarity_boost": 0.75,
    "style": 0.1,
    "use_speaker_boost": True,
    "speed": 0.9,
}


class TTSError(RuntimeError):
    pass


@dataclass
class TimedSpeech:
    clip: AudioClip
    # char_starts[i] = seconds at which clean_text[i] starts being spoken
    char_starts: list[float]


class AnnouncerTTS:
    def __init__(self) -> None:
        if not settings.elevenlabs_api_key:
            raise TTSError("ELEVENLABS_API_KEY is not set (.env)")
        if not settings.elevenlabs_voice_id:
            raise TTSError(
                "ELEVENLABS_VOICE_ID is not set (.env). Pick a high-energy "
                "voice in the ElevenLabs Voice Library and paste its ID."
            )
        from elevenlabs.client import ElevenLabs

        self._client = ElevenLabs(api_key=settings.elevenlabs_api_key)
        self._pcm_ok: bool | None = None  # learned on first call

    def synthesize(self, clean_text: str) -> TimedSpeech:
        """Speak clean (tag-free) text; return audio + per-character timing."""
        res, rate = self._convert(clean_text)
        samples = self._decode(res.audio_base_64, rate)

        align = res.alignment
        if align is None:
            raise TTSError("ElevenLabs returned no alignment data")
        starts = list(align.character_start_times_seconds or [])
        chars = align.characters or []
        # Alignment is for the exact text we sent; tolerate small mismatches.
        if len(starts) < len(clean_text):
            logger.warning(
                "Alignment covers %d/%d chars; padding with last timestamp",
                len(starts), len(clean_text),
            )
            last = starts[-1] if starts else 0.0
            starts += [last] * (len(clean_text) - len(starts))
        if chars and len(chars) != len(clean_text):
            logger.debug("Alignment char count %d != text %d", len(chars), len(clean_text))

        return TimedSpeech(AudioClip(samples, ROBOT_RATE), starts)

    def say_quick(self, text: str) -> AudioClip:
        """Short utterances (greetings, questions) — no timestamps needed."""
        res, rate = self._convert(text)
        return AudioClip(self._decode(res.audio_base_64, rate), ROBOT_RATE)

    # -- internals ---------------------------------------------------------

    def _convert(self, text: str):
        """Try pcm_16000 first; fall back to mp3 if the tier rejects PCM."""
        formats = ["pcm_16000", "mp3_22050_32"] if self._pcm_ok is None else (
            ["pcm_16000"] if self._pcm_ok else ["mp3_22050_32"]
        )
        last_exc: Exception | None = None
        for fmt in formats:
            try:
                res = self._client.text_to_speech.convert_with_timestamps(
                    voice_id=settings.elevenlabs_voice_id,
                    text=text,
                    model_id=settings.elevenlabs_tts_model,
                    output_format=fmt,
                    voice_settings=VOICE_SETTINGS,
                )
                self._pcm_ok = fmt == "pcm_16000"
                if not self._pcm_ok:
                    logger.info("Using mp3 fallback (pcm_16000 unavailable on this tier)")
                return res, fmt
            except Exception as exc:
                last_exc = exc
                logger.warning("TTS with %s failed: %s", fmt, str(exc)[:200])
        raise TTSError(f"ElevenLabs TTS failed: {last_exc}") from last_exc

    @staticmethod
    def _decode(audio_b64: str, fmt: str) -> np.ndarray:
        raw = base64.b64decode(audio_b64)
        if fmt.startswith("pcm_"):
            pcm = np.frombuffer(raw, dtype="<i2")
            return (pcm.astype(np.float32) / 32768.0).reshape(-1)
        # mp3 fallback — decode + resample to the robot rate
        import soundfile as sf

        data, rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        arr = np.asarray(data, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr.mean(axis=1)
        return AudioClip(arr.reshape(-1), int(rate)).resampled(ROBOT_RATE).samples

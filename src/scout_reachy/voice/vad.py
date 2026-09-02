"""Voice activity detection + utterance capture via webrtcvad.

webrtcvad classifies short (10/20/30 ms) frames of 16-bit mono PCM as speech or
not. We wrap it to (a) judge a single frame and (b) record one full spoken
utterance from a frame stream: wait for speech, then stop after trailing silence.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Iterator

import numpy as np

from .audio import AudioClip

logger = logging.getLogger(__name__)

VAD_RATE = 16000  # webrtcvad supports 8/16/32/48 kHz; we standardise on 16 kHz
FRAME_MS = 20
FRAME_SAMPLES = VAD_RATE * FRAME_MS // 1000  # 320 samples per 20 ms frame


def float_to_pcm16(frame: np.ndarray) -> bytes:
    clipped = np.clip(frame, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


class VoiceActivityDetector:
    """Thin webrtcvad wrapper. aggressiveness 0 (lax) .. 3 (strict)."""

    def __init__(self, aggressiveness: int = 2) -> None:
        import webrtcvad

        self._vad = webrtcvad.Vad(aggressiveness)

    def is_speech(self, frame: np.ndarray) -> bool:
        """frame: float32 mono, exactly FRAME_SAMPLES long, at 16 kHz."""
        if len(frame) != FRAME_SAMPLES:
            raise ValueError(f"VAD frame must be {FRAME_SAMPLES} samples, got {len(frame)}")
        return self._vad.is_speech(float_to_pcm16(frame), VAD_RATE)


class UtteranceRecorder:
    """Capture one utterance from a 16 kHz float32 frame stream.

    Strategy: buffer a little pre-roll, trigger when several consecutive frames
    are speech, then keep recording until `silence_ms` of trailing non-speech.
    """

    def __init__(
        self,
        aggressiveness: int = 2,
        *,
        start_frames: int = 3,      # consecutive speech frames to start (~60 ms)
        silence_ms: int = 700,      # trailing silence that ends the utterance
        max_seconds: float = 15.0,  # hard cap
        preroll_ms: int = 200,      # audio kept from just before the trigger
    ) -> None:
        self.vad = VoiceActivityDetector(aggressiveness)
        self.start_frames = start_frames
        self.silence_frames = silence_ms // FRAME_MS
        self.max_frames = int(max_seconds * 1000 // FRAME_MS)
        self.preroll = deque(maxlen=preroll_ms // FRAME_MS)

    def record(self, frames: Iterator[np.ndarray]) -> AudioClip:
        """Consume frames until an utterance completes; return it as a clip."""
        triggered = False
        voiced: list[np.ndarray] = []
        run_speech = 0
        run_silence = 0

        for i, frame in enumerate(frames):
            speech = self.vad.is_speech(frame)
            if not triggered:
                self.preroll.append(frame)
                run_speech = run_speech + 1 if speech else 0
                if run_speech >= self.start_frames:
                    triggered = True
                    voiced.extend(self.preroll)
                    self.preroll.clear()
                    logger.debug("Utterance started")
            else:
                voiced.append(frame)
                run_silence = 0 if speech else run_silence + 1
                if run_silence >= self.silence_frames:
                    logger.debug("Utterance ended (trailing silence)")
                    break
                if len(voiced) >= self.max_frames:
                    logger.debug("Utterance hit max length")
                    break

        if not voiced:
            return AudioClip(np.zeros(0, dtype=np.float32), VAD_RATE)
        return AudioClip(np.concatenate(voiced).astype(np.float32), VAD_RATE)

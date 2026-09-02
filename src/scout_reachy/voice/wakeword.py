"""Wake-word detection via openWakeWord (local, offline once models are cached).

openWakeWord consumes 16 kHz 16-bit PCM in ~80 ms chunks (1280 samples) and
returns a per-model confidence score. We wrap it to load a configured wake word
and block until it fires on a frame stream.
"""

from __future__ import annotations

import logging
from typing import Iterator

import numpy as np

from ..config import settings

logger = logging.getLogger(__name__)

WAKE_RATE = 16000
CHUNK_SAMPLES = 1280  # 80 ms at 16 kHz — openWakeWord's expected chunk size


def ensure_models() -> None:
    """Download openWakeWord's base + pretrained models if missing (one time)."""
    import openwakeword

    openwakeword.utils.download_models()


class WakeWord:
    def __init__(
        self,
        model_name: str | None = None,
        *,
        threshold: float | None = None,
        auto_download: bool = True,
    ) -> None:
        from openwakeword.model import Model

        self.model_name = model_name or settings.wakeword_model
        self.threshold = settings.wakeword_threshold if threshold is None else threshold
        if auto_download:
            ensure_models()
        logger.info("Loading wake word model %s", self.model_name)
        # Accept either a bundled name ("hey_jarvis") or a path to a custom .onnx.
        self._model = Model(wakeword_models=[self.model_name], inference_framework="onnx")

    def score(self, chunk: np.ndarray) -> float:
        """Score one ~80 ms float32 chunk; returns max confidence across models."""
        pcm16 = (np.clip(chunk, -1.0, 1.0) * 32767.0).astype(np.int16)
        preds = self._model.predict(pcm16)
        return max(preds.values()) if preds else 0.0

    def reset(self) -> None:
        """Clear the model's internal audio buffer/state. Call before starting a
        fresh listen loop that feeds chunks via score() (wait_for_wake does this
        itself)."""
        self._model.reset()

    def wait_for_wake(self, chunks: Iterator[np.ndarray]) -> bool:
        """Block until a chunk scores above threshold. Returns True when woken."""
        self._model.reset()
        for chunk in chunks:
            if self.score(chunk) >= self.threshold:
                logger.info("Wake word detected")
                return True
        return False

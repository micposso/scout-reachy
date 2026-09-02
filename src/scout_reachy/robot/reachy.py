"""Reachy Mini wrapper: robot mic + speaker as AudioSource/AudioSink.

Connects to the local daemon (reachy-mini-daemon) over localhost and exposes the
robot's mic array and speaker through the same small interfaces the rest of the
voice stack uses, so triage/agent code never touches the SDK directly.

The robot audio is stereo 16 kHz; we downmix the mic to mono for STT/VAD and push
mono TTS to the speaker.
"""

from __future__ import annotations

import logging
import time
from typing import Iterator

import numpy as np

from ..config import settings
from ..vision.frame import Frame
from ..voice.audio import AudioClip

logger = logging.getLogger(__name__)

_POLL_SLEEP = 0.005  # seconds to wait when the mic has no sample ready yet


def _downmix(sample) -> np.ndarray:
    arr = np.asarray(sample, dtype=np.float32)
    return arr.mean(axis=1) if arr.ndim == 2 else arr.reshape(-1)


class ReachyMic:
    """Robot microphone as an AudioSource (mono float32)."""

    def __init__(self, media) -> None:
        self._m = media
        self.sample_rate = int(media.get_input_audio_samplerate())

    def read(self, seconds: float) -> AudioClip:
        self._m.start_recording()
        collected: list[np.ndarray] = []
        t0 = time.time()
        try:
            while time.time() - t0 < seconds:
                s = self._m.get_audio_sample()
                if s is not None:
                    collected.append(_downmix(s))
                else:
                    time.sleep(_POLL_SLEEP)
        finally:
            self._m.stop_recording()
        if not collected:
            return AudioClip(np.zeros(0, dtype=np.float32), self.sample_rate)
        return AudioClip(np.concatenate(collected), self.sample_rate)

    def frames(self, frame_samples: int) -> Iterator[np.ndarray]:
        """Yield fixed-size mono frames until the consumer stops iterating."""
        self._m.start_recording()
        buf = np.zeros(0, dtype=np.float32)
        try:
            while True:
                s = self._m.get_audio_sample()
                if s is None:
                    time.sleep(_POLL_SLEEP)
                    continue
                buf = np.concatenate([buf, _downmix(s)])
                while len(buf) >= frame_samples:
                    yield buf[:frame_samples]
                    buf = buf[frame_samples:]
        finally:
            self._m.stop_recording()

    def doa(self):
        """(direction_of_arrival_radians, is_speech_detected) from the mic array."""
        return self._m.get_DoA()


class ReachyCamera:
    """Robot head camera as a Frame source.

    Wraps the daemon MediaManager's `get_frame()`, which returns the latest BGR
    frame (h, w, 3) uint8, or None when the camera is unavailable or the
    GStreamer pipeline hasn't decoded a frame yet (it warms up for a beat after
    connect). Mirrors ReachyMic: the daemon owns the hardware, this just reads.
    """

    def __init__(self, media) -> None:
        self._m = media

    @property
    def available(self) -> bool:
        """True if the daemon initialized a camera device at all."""
        return getattr(self._m, "camera", None) is not None

    @property
    def resolution(self) -> tuple[int, int] | None:
        """(width, height) in pixels, or None if not yet known."""
        cam = getattr(self._m, "camera", None)
        if cam is None:
            return None
        try:
            return cam.resolution
        except RuntimeError:  # resolution not set until the pipeline opens
            return None

    def read(self, *, timeout: float = 2.0) -> Frame | None:
        """Poll for the next decoded frame, up to `timeout` seconds. Returns
        None if the camera is unavailable or no frame arrives in time (e.g. the
        media was released, or the pipeline never warmed up)."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            arr = self._m.get_frame()
            if arr is not None:
                return Frame(arr)
            time.sleep(_POLL_SLEEP)
        return None

    def grab(self) -> Frame | None:
        """Return the latest decoded frame without blocking, or None if none is
        ready yet. Cheaper than read() for poll loops (e.g. presence detection)
        that interleave camera checks with other work on the shared connection."""
        arr = self._m.get_frame()
        return Frame(arr) if arr is not None else None

    def frames(self) -> Iterator[Frame]:
        """Yield decoded frames as they arrive until the consumer stops
        iterating. Skips the warm-up gaps where get_frame() returns None."""
        while True:
            arr = self._m.get_frame()
            if arr is None:
                time.sleep(_POLL_SLEEP)
                continue
            yield Frame(arr)


class ReachySpeaker:
    """Robot speaker as an AudioSink. Resamples clips to the output rate and
    applies an output gain (settings.reachy_speaker_volume) so the whole robot
    can be made quieter without touching each call site."""

    def __init__(self, media, *, gain: float | None = None) -> None:
        self._m = media
        self.sample_rate = int(media.get_output_audio_samplerate())
        self.gain = float(settings.reachy_speaker_volume if gain is None else gain)

    def play(self, clip: AudioClip) -> None:
        if len(clip.samples) == 0:
            return
        c = clip.resampled(self.sample_rate)
        samples = c.samples * self.gain
        np.clip(samples, -1.0, 1.0, out=samples)  # guard against clipping
        mono = samples.reshape(-1, 1).astype(np.float32)
        self._m.start_playing()
        try:
            for i in range(0, len(mono), 1024):
                self._m.push_audio_sample(mono[i : i + 1024])
            time.sleep(c.duration + 0.3)  # let the buffer drain
        finally:
            self._m.stop_playing()


class ReachyRobot:
    """Context manager owning the ReachyMini connection + audio endpoints."""

    def __init__(
        self,
        *,
        connection_mode: str | None = None,
        media_backend: str | None = None,
    ) -> None:
        self.connection_mode = connection_mode or settings.reachy_connection_mode
        self.media_backend = media_backend or settings.reachy_media_backend
        self._mini = None
        self.mic: ReachyMic | None = None
        self.speaker: ReachySpeaker | None = None
        self.camera: ReachyCamera | None = None
        self._expressions = None

    def __enter__(self) -> "ReachyRobot":
        from reachy_mini import ReachyMini

        logger.info("Connecting to Reachy Mini (%s, media=%s)",
                    self.connection_mode, self.media_backend)
        self._mini = ReachyMini(
            connection_mode=self.connection_mode,
            media_backend=self.media_backend,
        ).__enter__()
        self.mic = ReachyMic(self._mini.media)
        self.speaker = ReachySpeaker(self._mini.media)
        self.camera = ReachyCamera(self._mini.media)
        return self

    def __exit__(self, *exc) -> None:
        if self._mini is not None:
            self._mini.__exit__(*exc)
            self._mini = None

    @property
    def mini(self):
        """The underlying ReachyMini (for head/antenna control)."""
        return self._mini

    @property
    def expressions(self):
        """Head/antenna gesture controller (lazily created)."""
        if self._expressions is None:
            from .expressions import Expressions

            self._expressions = Expressions(self._mini)
        return self._expressions

"""Audio primitives shared by the voice stack.

A small `AudioClip` value type plus resampling and a host-speaker sink. The robot
speaker/mic backends (M2 hardware step) implement the same Sink/Source protocols
so the rest of the code is backend-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.signal import resample_poly


@dataclass
class AudioClip:
    """Mono float32 PCM with an explicit sample rate. Range roughly [-1, 1]."""

    samples: np.ndarray  # shape (n,), dtype float32
    sample_rate: int

    def __post_init__(self) -> None:
        arr = np.asarray(self.samples, dtype=np.float32).reshape(-1)
        self.samples = arr

    @property
    def duration(self) -> float:
        return len(self.samples) / float(self.sample_rate) if self.sample_rate else 0.0

    def resampled(self, target_rate: int) -> "AudioClip":
        if target_rate == self.sample_rate or len(self.samples) == 0:
            return AudioClip(self.samples, target_rate)
        # Rational resampling: good quality, no external state.
        g = np.gcd(int(self.sample_rate), int(target_rate))
        up, down = target_rate // g, self.sample_rate // g
        out = resample_poly(self.samples, up, down).astype(np.float32)
        return AudioClip(out, target_rate)

    def peak_normalized(self, headroom: float = 0.97) -> "AudioClip":
        peak = float(np.max(np.abs(self.samples))) if len(self.samples) else 0.0
        if peak <= 1e-6:
            return self
        return AudioClip(self.samples * (headroom / peak), self.sample_rate)


def load_audio(path) -> AudioClip:
    """Load an audio file (WAV/FLAC/OGG/AIFF — anything libsndfile reads) as a
    mono float32 AudioClip at the file's own sample rate. Stereo is downmixed;
    the speaker resamples to its output rate on play, so any rate is fine."""
    import soundfile as sf  # lazy; libsndfile-backed

    data, rate = sf.read(str(path), dtype="float32", always_2d=False)
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 2:  # (frames, channels) -> mono
        arr = arr.mean(axis=1)
    return AudioClip(arr.reshape(-1), int(rate))


def tone(
    freq: float,
    seconds: float,
    *,
    sample_rate: int = 16000,
    volume: float = 0.22,
    fade: float = 0.008,
) -> AudioClip:
    """A simple sine tone with short fades in/out to avoid clicks."""
    n = int(seconds * sample_rate)
    if n <= 0:
        return AudioClip(np.zeros(0, dtype=np.float32), sample_rate)
    t = np.arange(n, dtype=np.float32) / sample_rate
    sig = (np.sin(2 * np.pi * freq * t) * volume).astype(np.float32)
    f = max(1, int(fade * sample_rate))
    if f * 2 < n:
        ramp = np.linspace(0.0, 1.0, f, dtype=np.float32)
        sig[:f] *= ramp
        sig[-f:] *= ramp[::-1]
    return AudioClip(sig, sample_rate)


def chime(
    notes: list[tuple[float, float]],
    *,
    sample_rate: int = 16000,
    volume: float = 0.22,
    gap: float = 0.0,
) -> AudioClip:
    """Concatenate a sequence of (freq_hz, seconds) notes into one short cue."""
    parts: list[np.ndarray] = []
    for freq, dur in notes:
        parts.append(tone(freq, dur, sample_rate=sample_rate, volume=volume).samples)
        if gap > 0:
            parts.append(np.zeros(int(gap * sample_rate), dtype=np.float32))
    samples = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
    return AudioClip(samples, sample_rate)


class AudioSink(Protocol):
    """Anything that can play a mono float32 clip (host speaker or robot)."""

    def play(self, clip: AudioClip) -> None: ...


class AudioSource(Protocol):
    """Anything that yields mono float32 audio at a known sample rate."""

    @property
    def sample_rate(self) -> int: ...

    def read(self, seconds: float) -> AudioClip: ...


class HostSpeaker:
    """Plays audio through the host PC's default output via sounddevice.

    Used for development/testing before (or instead of) routing to the robot.
    """

    def play(self, clip: AudioClip) -> None:
        import sounddevice as sd  # imported lazily so module import stays cheap

        sd.play(clip.samples, clip.sample_rate)
        sd.wait()


class HostMicrophone:
    """Captures audio from the host PC's default input via sounddevice.

    Yields fixed-size mono float32 frames at `sample_rate`. Used to test the VAD
    / wake-word / STT path before routing capture through the robot mic array.
    """

    def __init__(self, sample_rate: int = 16000) -> None:
        self.sample_rate = sample_rate

    def read(self, seconds: float) -> AudioClip:
        import sounddevice as sd

        n = int(seconds * self.sample_rate)
        buf = sd.rec(n, samplerate=self.sample_rate, channels=1, dtype="float32")
        sd.wait()
        return AudioClip(buf.reshape(-1), self.sample_rate)

    def frames(self, frame_samples: int):
        """Open the mic and yield successive float32 frames until the consumer
        stops iterating (e.g. breaks out of the loop)."""
        import sounddevice as sd

        with sd.InputStream(
            samplerate=self.sample_rate, channels=1, dtype="float32",
            blocksize=frame_samples,
        ) as stream:
            while True:
                data, _overflowed = stream.read(frame_samples)
                yield data.reshape(-1)

"""A single camera frame value type — the vision counterpart to AudioClip.

Deliberately dependency-free (numpy only): the daemon hands us a BGR uint8 array
from GStreamer, and everything here is plain array work. Stills are saved as
binary PPM (P6) so a frame can be inspected with no image library installed —
important on this machine, where some compiled image wheels are blocked.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Frame:
    """One camera frame: BGR uint8, shape (height, width, 3) — the layout the
    Reachy daemon's MediaManager.get_frame() returns (OpenCV channel order)."""

    bgr: np.ndarray

    def __post_init__(self) -> None:
        arr = np.asarray(self.bgr, dtype=np.uint8)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Frame expects (h, w, 3) BGR, got shape {arr.shape}")
        self.bgr = arr

    @property
    def height(self) -> int:
        return int(self.bgr.shape[0])

    @property
    def width(self) -> int:
        return int(self.bgr.shape[1])

    @property
    def rgb(self) -> np.ndarray:
        """RGB view (channel-reversed) — what most viewers/models expect."""
        return self.bgr[:, :, ::-1]

    @property
    def mean_brightness(self) -> float:
        """Mean pixel value in [0, 255]. Cheap 'is the lens covered / dark?'
        signal — a near-zero value usually means the cap is on or the room is
        dark, which is the first thing to check when perception misbehaves."""
        return float(self.bgr.mean())

    def save_ppm(self, path) -> Path:
        """Write the frame as a binary PPM (P6) — no image library required.
        Most image viewers and IrfanView/GIMP open PPM directly."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        header = f"P6\n{self.width} {self.height}\n255\n".encode("ascii")
        # PPM is RGB, row-major, 8-bit/channel — ascontiguousarray for tobytes().
        body = np.ascontiguousarray(self.rgb).tobytes()
        p.write_bytes(header + body)
        return p


def load_ppm(path) -> Frame:
    """Read a binary PPM (P6) back into a Frame — the inverse of save_ppm, with
    no image library. Lets tests/prototypes run against a saved camera snapshot
    off-robot. Only P6, maxval 255 (what save_ppm writes) is supported."""
    raw = Path(path).read_bytes()
    if not raw.startswith(b"P6"):
        raise ValueError(f"{path}: not a P6 PPM")
    # Header is three whitespace-separated ASCII tokens after the magic: width,
    # height, maxval — then a single whitespace byte, then the binary body.
    fields: list[bytes] = []
    i = 2
    while len(fields) < 3:
        while i < len(raw) and raw[i:i + 1].isspace():
            i += 1
        if i < len(raw) and raw[i:i + 1] == b"#":  # comment line
            while i < len(raw) and raw[i:i + 1] not in (b"\n", b"\r"):
                i += 1
            continue
        start = i
        while i < len(raw) and not raw[i:i + 1].isspace():
            i += 1
        fields.append(raw[start:i])
    width, height, maxval = (int(f) for f in fields)
    if maxval != 255:
        raise ValueError(f"{path}: only maxval 255 supported, got {maxval}")
    body = raw[i + 1:]  # skip the single whitespace after maxval
    rgb = np.frombuffer(body[: width * height * 3], dtype=np.uint8).reshape(height, width, 3)
    return Frame(rgb[:, :, ::-1])  # store as BGR

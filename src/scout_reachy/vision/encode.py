"""Encode a camera Frame into a base64 PNG for an Anthropic image block.

Why hand-rolled: Anthropic needs JPEG/PNG bytes, but the compiled image wheels
(OpenCV, and sometimes Pillow's _imaging.pyd) are exactly what BitDefender
quarantines on this machine — the same reason vision/frame.py saves PPM by hand.
So we build a minimal PNG here with only stdlib zlib + struct + numpy, which
cannot be quarantined. PNG (lossless) is a fine fit: signs are line art and text
where JPEG blocking would hurt legibility, and we downsample first to bound size.

Public API:
    frame_to_png_b64(frame, max_width=...) -> (base64_str, media_type)
"""

from __future__ import annotations

import base64
import struct
import zlib

import numpy as np

from ..config import settings
from .frame import Frame

PNG_MEDIA_TYPE = "image/png"
_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _chunk(tag: bytes, data: bytes) -> bytes:
    """One PNG chunk: length + type + data + CRC32(type+data)."""
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def _downsample(rgb: np.ndarray, max_width: int) -> np.ndarray:
    """Nearest-neighbour shrink so the longest side <= max_width. Cheap integer
    index subsampling (same approach as presence.py) — no interpolation lib."""
    h, w = rgb.shape[:2]
    if max_width <= 0 or w <= max_width:
        return rgb
    step = int(np.ceil(w / max_width))
    return rgb[::step, ::step, :]


def encode_png(rgb: np.ndarray) -> bytes:
    """Encode an (h, w, 3) uint8 RGB array as PNG bytes (truecolor, no alpha)."""
    arr = np.ascontiguousarray(rgb, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"encode_png expects (h, w, 3) RGB, got {arr.shape}")
    h, w = arr.shape[:2]

    # Prepend the per-scanline filter byte (0 = None) to each row, then deflate.
    filtered = np.concatenate(
        [np.zeros((h, 1), dtype=np.uint8), arr.reshape(h, w * 3)], axis=1
    )
    compressed = zlib.compress(filtered.tobytes(), level=6)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)  # 8-bit, colour type 2
    return (
        _SIGNATURE
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", compressed)
        + _chunk(b"IEND", b"")
    )


def frame_to_png_b64(
    frame: Frame, *, max_width: int | None = None
) -> tuple[str, str]:
    """Frame -> (base64 PNG string, media type) ready for an Anthropic image
    source block. Downsamples so the longest side is <= max_width first."""
    limit = settings.vision_max_width if max_width is None else max_width
    rgb = _downsample(frame.rgb, limit)
    png = encode_png(rgb)
    return base64.b64encode(png).decode("ascii"), PNG_MEDIA_TYPE

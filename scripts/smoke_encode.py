"""Smoke test: the pure-Python PNG encoder — fully OFFLINE (no keys, no robot).

    .venv\\Scripts\\python.exe scripts\\smoke_encode.py

Builds a synthetic frame, encodes it with vision/encode.py, then decodes the PNG
back by hand (parse chunks, inflate IDAT, strip filter bytes) and checks the
pixels round-trip exactly. Also checks the downsample cap. This proves the
encoder is correct without any compiled image library (Pillow/OpenCV), which is
the whole point on this machine.
"""

from __future__ import annotations

import base64
import struct
import zlib

import numpy as np

from scout_reachy.vision.encode import PNG_MEDIA_TYPE, frame_to_png_b64
from scout_reachy.vision.frame import Frame


def decode_png(png: bytes) -> tuple[int, int, np.ndarray]:
    """Minimal PNG decoder for our own truecolor/filter-0 output."""
    assert png[:8] == b"\x89PNG\r\n\x1a\n", "bad signature"
    pos = 8
    width = height = 0
    idat = b""
    while pos < len(png):
        (length,) = struct.unpack(">I", png[pos:pos + 4])
        tag = png[pos + 4:pos + 8]
        data = png[pos + 8:pos + 8 + length]
        crc = struct.unpack(">I", png[pos + 8 + length:pos + 12 + length])[0]
        assert crc == (zlib.crc32(tag + data) & 0xFFFFFFFF), f"CRC fail on {tag}"
        if tag == b"IHDR":
            width, height, depth, ctype = struct.unpack(">IIBB", data[:10])
            assert (depth, ctype) == (8, 2), "expected 8-bit truecolor"
        elif tag == b"IDAT":
            idat += data
        pos += 12 + length
    raw = zlib.decompress(idat)
    stride = width * 3 + 1
    rows = []
    for y in range(height):
        line = raw[y * stride:(y + 1) * stride]
        assert line[0] == 0, "expected filter type 0 (None)"
        rows.append(np.frombuffer(line[1:], dtype=np.uint8).reshape(width, 3))
    return width, height, np.array(rows, dtype=np.uint8)


def main() -> int:
    failures = 0

    # A small RGB gradient so any channel swap or row/col mixup shows up.
    h, w = 12, 20
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[..., 0] = np.linspace(0, 255, w, dtype=np.uint8)[None, :]  # R varies by col
    rgb[..., 1] = np.linspace(0, 255, h, dtype=np.uint8)[:, None]  # G varies by row
    rgb[..., 2] = 128
    frame = Frame(rgb[:, :, ::-1])  # Frame stores BGR; give it BGR of our RGB

    b64, media = frame_to_png_b64(frame, max_width=1024)  # no downsample at this size
    if media != PNG_MEDIA_TYPE:
        print(f"[FAIL] wrong media type {media!r}")
        failures += 1
    png = base64.b64decode(b64)
    dw, dh, decoded = decode_png(png)
    if (dw, dh) != (w, h):
        print(f"[FAIL] dimensions {dw}x{dh} != {w}x{h}")
        failures += 1
    elif not np.array_equal(decoded, rgb):
        print("[FAIL] pixels did not round-trip (channel/row order bug)")
        failures += 1
    else:
        print(f"[OK] {w}x{h} PNG round-trips exactly ({len(png)} bytes)")

    # Downsample cap: a wide frame must shrink so its width <= max_width.
    wide = Frame(np.zeros((100, 800, 3), dtype=np.uint8))
    b64w, _ = frame_to_png_b64(wide, max_width=200)
    dw2, _, _ = decode_png(base64.b64decode(b64w))
    if dw2 > 200:
        print(f"[FAIL] downsample cap not applied: width {dw2} > 200")
        failures += 1
    else:
        print(f"[OK] downsample cap applied: 800px -> {dw2}px")

    if failures:
        print(f"\n[FAIL] {failures} check(s) failed")
        return 1
    print("\n[OK] PNG encoder smoke complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

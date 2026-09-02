"""Vision stack: the camera Frame value type and PNG encoding for the model.

The camera *hardware* wrapper lives in `robot.reachy` next to the mic/speaker
(they share one robot connection). Encoding stays dependency-free (numpy +
stdlib zlib) because this machine's BitDefender blocks some compiled image
wheels — the same reason frame.py reads/writes PPM by hand.
"""

from __future__ import annotations

from .encode import frame_to_png_b64
from .frame import Frame, load_ppm

__all__ = ["Frame", "load_ppm", "frame_to_png_b64"]

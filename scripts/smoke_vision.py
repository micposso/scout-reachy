"""Smoke test: prove the Reachy Mini head camera delivers frames.

Requires: Reachy Mini Lite on COM3, reachy-mini-daemon running, reachy-mini
installed in this venv (see README). No API keys, no extra image libraries.

    .venv\\Scripts\\python.exe scripts\\smoke_vision.py

Saves the first and last frame as PPM stills under .app_data/vision/ — feed one
to smoke_locate.py --ppm to test the vision pipeline off-robot afterward.
"""

from __future__ import annotations

import logging
import time

from scout_reachy.robot.reachy import ReachyRobot

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

SNAPSHOT_DIR = ".app_data/vision"
BURST_SECONDS = 3.0


def main() -> int:
    print("Connecting to Reachy Mini (daemon must be running on localhost:8000)...")
    with ReachyRobot() as robot:
        cam = robot.camera
        assert cam is not None

        if not cam.available:
            print("[FAIL] No camera device. Check the camera is plugged in and "
                  "that media isn't released to direct access.")
            return 1
        print(f"[OK] camera present — resolution {cam.resolution}")

        first = cam.read(timeout=5.0)
        if first is None:
            print("[FAIL] camera present but no frame decoded within 5s.")
            return 1
        p_first = first.save_ppm(f"{SNAPSHOT_DIR}/first.ppm")
        print(f"[OK] first frame {first.width}x{first.height} "
              f"brightness={first.mean_brightness:.1f}/255 -> {p_first}")
        if first.mean_brightness < 2.0:
            print("[WARN] frame is almost black — lens cap on, or the room is dark?")

        count = 0
        last = first
        t0 = time.time()
        for frame in cam.frames():
            count += 1
            last = frame
            if time.time() - t0 >= BURST_SECONDS:
                break
        fps = count / max(time.time() - t0, 1e-6)
        p_last = last.save_ppm(f"{SNAPSHOT_DIR}/last.ppm")
        print(f"[OK] streamed {count} frames in {BURST_SECONDS:.0f}s (~{fps:.1f} fps) -> {p_last}")
        if count == 0:
            print("[FAIL] stream stalled after the first frame.")
            return 1

    print("[OK] vision smoke complete — camera is live")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

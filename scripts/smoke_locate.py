"""Smoke test: the full look-and-locate pipeline (encode -> vision agent -> OSM
-> guard). Needs ANTHROPIC_API_KEY + network. Image comes from a saved PPM
(--ppm) or the robot camera (daemon + reachy-mini required).

    # off-robot, against a saved snapshot near a known place:
    .venv\\Scripts\\python.exe scripts\\smoke_locate.py --ppm shot.ppm --lat 48.8584 --lon 2.2945
    # on-robot, live camera + resolved device location:
    .venv\\Scripts\\python.exe scripts\\smoke_locate.py

Prints the guarded LocationAnswer and asserts the core invariant: a located
answer (source != unknown) is always backed by a real osm_id + coordinates.
"""

from __future__ import annotations

import argparse
import json

from dotenv import load_dotenv

load_dotenv()

from scout_reachy.agent.scout import ScoutAgent
from scout_reachy.location import resolve_device_location
from scout_reachy.vision.frame import load_ppm


def get_frame(ppm: str | None):
    if ppm:
        print(f"Loading frame from {ppm}")
        return load_ppm(ppm), None  # no robot to clean up
    from scout_reachy.robot.reachy import ReachyRobot

    robot = ReachyRobot().__enter__()
    frame = robot.camera.read(timeout=5.0)
    if frame is None:
        robot.__exit__(None, None, None)
        raise SystemExit("[FAIL] camera returned no frame")
    return frame, robot


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ppm", help="path to a P6 PPM snapshot (else use robot camera)")
    ap.add_argument("--lat", type=float, help="override device latitude")
    ap.add_argument("--lon", type=float, help="override device longitude")
    args = ap.parse_args()

    if args.lat is not None and args.lon is not None:
        device = (args.lat, args.lon)
    else:
        coord = resolve_device_location()
        device = coord.as_tuple() if coord else None
    print(f"Device location: {device}")

    frame, robot = get_frame(args.ppm)
    try:
        print(f"Frame {frame.width}x{frame.height}, brightness {frame.mean_brightness:.1f}")
        answer = ScoutAgent().identify(frame, device)
    finally:
        if robot is not None:
            robot.__exit__(None, None, None)

    print("\nLocationAnswer:")
    print(json.dumps(answer.model_dump(), indent=2, ensure_ascii=False))

    # Invariant: a located answer must carry a real osm_id + coordinates.
    if answer.source != "unknown":
        if not answer.osm_id or answer.lat is None or answer.lon is None:
            print("\n[FAIL] located answer missing osm_id/coords — guard breach")
            return 1
        print(f"\n[OK] located + confirmed: {answer.identified_place} <{answer.osm_id}>")
    else:
        print("\n[OK] honest unknown (no confirmed location) — guard held")
    print(f'\nSpoken: "{answer.spoken_summary}"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

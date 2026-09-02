"""Smoke test: device location resolution.

    .venv\\Scripts\\python.exe scripts\\smoke_location.py

Tries the Windows Geolocator (winrt) first, then the .env fallbacks. Passes as
long as SOMETHING resolves or the fallback chain is intentionally empty — the
point is to see WHICH source fired and confirm the guarded import doesn't crash.
No robot, no Anthropic key. On a desktop with no manual location set, expect
'None' — set DEVICE_LAT/DEVICE_LON or DEVICE_ADDRESS in .env to exercise it.
"""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

from scout_reachy.config import settings
from scout_reachy.location import resolve_device_location


def main() -> int:
    print(f"use_windows_location={settings.use_windows_location} "
          f"device_lat={settings.device_lat} device_lon={settings.device_lon} "
          f"device_address={settings.device_address!r}")
    coord = resolve_device_location()
    if coord is None:
        print("[OK] no location resolved (guarded import held; set a fallback in "
              ".env to get a fix). The agent will rely on sign text alone.")
        return 0
    print(f"[OK] resolved {coord.lat:.5f},{coord.lon:.5f} via '{coord.source}'"
          + (f" (+/-{coord.accuracy_m:.0f} m)" if coord.accuracy_m else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

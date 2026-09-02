# Scout Reachy

A voice app for a **Reachy Mini Lite** (USB, COM3, daemon on localhost:8000) that
uses its camera to **read a sign or landmark, look it up on a map, and speak where
you are** — carried around on a WiFi laptop + portable battery. Sibling of
`WORLD-CUP-REACHY` / `AGENTMAIL-REACHY`; reuses their proven robot/voice/speech
modules unchanged.

## Pipeline (one line per stage)

trigger (push-to-talk `Enter`, or the wake word) → capture one camera frame
(`robot/reachy.py: ReachyCamera.read`) → **encode** the frame to a base64 PNG
(`vision/encode.py`, pure-Python — no compiled image wheel) → **vision agent**
(`agent/scout.py`: Anthropic `tool_runner`, the image + device location go in;
Claude reads the sign and calls OSM tools) → **maps tools** (`agent/tools.py` →
`data/osm.py`: Nominatim `geocode`/`reverse_geocode`, Overpass `nearby`) →
**guard** (`agent/guard.py`: drops any location the maps didn't return) →
structured `LocationAnswer` (`agent/schema.py`) → **TTS** (`speech/tts.py`) +
sparse gestures (`robot/expressions.py`) → **local visit memory**
(`data/memory.py`, SQLite) → optional spoken follow-ups. Device
location comes from `location.py` (Windows Geolocator → `.env` manual fallback).

## Invariants — do not break these

1. **No invented locations.** A `LocationAnswer` with `source != "unknown"` MUST
   carry an `osm_id` (and coordinates) that a maps tool actually returned this
   turn; `agent/guard.py` checks this in plain code and downgrades anything else
   to an honest "I can read the sign but can't confirm the spot." The model never
   locates itself. (Vision counterpart of the sibling project's fact-check gate.)
2. **Coordinates only from the OSM tools**, never guessed from the image. The
   system prompt enforces it; the guard is the backstop.
3. **Motors and audio share one robot connection** — gestures during playback stay
   sparse (`talk_beat`, ≥1s spacing). 0.4s echo settle before opening the mic.
4. **Be polite to OpenStreetMap.** `data/osm.py` sets a descriptive `User-Agent`
   (required — set `OSM_USER_AGENT`), throttles to ~1 req/s, and caches (60s TTL).
   No bulk queries. It's a donated free service.
5. **No compiled image wheels.** BitDefender quarantines OpenCV / Pillow `.pyd` on
   this machine (same reason zeroconf is stubbed). PNG encoding is pure-Python;
   the `winrt` location import is guarded and degrades to the manual fallback.
6. **Memory is context, not location proof.** Only nearby, map-confirmed initial
   answers are stored in `.app_data/scout_memory.db`. Recalled places are never
   added to the guard's per-turn map registry, so a memory cannot validate where
   the robot is now.

## Commands

```powershell
.venv\Scripts\reachy-mini-daemon.exe                  # FIRST: start the robot daemon
.venv\Scripts\python.exe -m scout_reachy.app          # run the app
uv pip install -e . --python .venv\Scripts\python.exe # after dep changes
```

Python 3.12 (reachy-mini requires 3.10–3.12). `reachy-mini` is installed
separately (it pins zeroconf, stubbed here for BitDefender). Copy the stub from a
sibling project's `.venv\Lib\site-packages\zeroconf\__init__.py`.

## Smoke tests (no pytest; each returns non-zero on failure)

| Script | Needs | Checks |
|---|---|---|
| `smoke_encode.py`   | nothing (offline)        | PNG encoder round-trips pixels + downsample cap |
| `smoke_guard.py`    | nothing (offline)        | guard passes confirmed / downgrades invented locations |
| `smoke_memory.py`   | nothing (offline)        | SQLite visits persist, increment, search, and recall nearby |
| `smoke_location.py` | `.env` (no robot)        | device location resolves (winrt or fallback) |
| `smoke_osm.py`      | network (no key)         | geocode / reverse / nearby against live OSM |
| `smoke_vision.py`   | robot + daemon           | camera streams frames; saves PPM snapshots |
| `smoke_locate.py`   | Anthropic key + image    | full pipeline; `--ppm shot.ppm --lat .. --lon ..` off-robot |

Recommended first run (no robot, no camera): `smoke_encode` → `smoke_guard` →
`smoke_memory` → `smoke_osm` → `smoke_location`. Then `smoke_vision` on the robot, feed a saved
`.app_data/vision/first.ppm` into `smoke_locate --ppm`.

## Keys (.env, gitignored)

`ANTHROPIC_API_KEY` + `ANTHROPIC_MODEL` (vision-capable), `ELEVENLABS_API_KEY` /
`ELEVENLABS_VOICE_ID`, `OSM_USER_AGENT` (required by Nominatim). Optional:
`DEVICE_LAT`/`DEVICE_LON`/`DEVICE_ADDRESS` (manual location), `TRIGGER_MODE`,
`GOOGLE_MAPS_API_KEY` (future backend seam). OSM needs no key.

## Persistent place memory

After the guard confirms an initial location, Scout records it as a visit only
when the laptop's device location is within `MEMORY_VISIT_RADIUS_M` (500 m by
default). This avoids treating a photo of a distant landmark or poster as a
visit. Places, timestamps, counts, and the last observed sign text are kept in a
local, gitignored SQLite database at `.app_data/scout_memory.db`. The agent can
use `recall_nearby` and `recall_place` for questions such as "have we been here
before?". Set `MEMORY_ENABLED=false` to disable it; delete the database to erase
the history.

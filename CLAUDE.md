# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Scout Reachy is a voice app for a **Reachy Mini Lite** (USB, COM3, daemon on
`localhost:8000`). It uses the robot's camera to read a sign or landmark, look it
up on a map, and speak where you are — carried on a WiFi laptop + battery. It is a
sibling of `WORLD-CUP-REACHY` / `AGENTMAIL-REACHY` and reuses their proven
robot/voice/speech modules with parallel field names, so keep changes consistent
with those projects when touching shared-shape code (`config.py`, `robot/`,
`voice/`, `speech/`).

## Commands

```powershell
.venv\Scripts\reachy-mini-daemon.exe                  # FIRST: start the robot daemon
.venv\Scripts\python.exe -m scout_reachy.app          # run the app
uv pip install -e . --python .venv\Scripts\python.exe # after dependency changes
```

- Python 3.12 (reachy-mini requires 3.10–3.12).
- `reachy-mini` is installed **separately** and must NOT be added to
  `pyproject.toml` — it pins `zeroconf`, which is stubbed on this machine for
  BitDefender reasons. Copy the stub from a sibling project's
  `.venv\Lib\site-packages\zeroconf\__init__.py`.

### Tests (smoke scripts, no pytest)

There is no pytest suite. Each `scripts/smoke_*.py` is a standalone check that
exits non-zero on failure. Run one directly:

```powershell
.venv\Scripts\python.exe scripts\smoke_encode.py     # PNG encoder round-trip (offline)
.venv\Scripts\python.exe scripts\smoke_guard.py      # guard confirms/downgrades (offline)
.venv\Scripts\python.exe scripts\smoke_osm.py        # live OSM geocode/reverse/nearby (network)
.venv\Scripts\python.exe scripts\smoke_location.py   # device location resolves (.env)
.venv\Scripts\python.exe scripts\smoke_vision.py     # camera streams frames (robot + daemon)
.venv\Scripts\python.exe scripts\smoke_locate.py --ppm shot.ppm --lat 48.8584 --lon 2.2945  # full pipeline off-robot
```

Recommended offline-first order: `smoke_encode` → `smoke_guard` → `smoke_osm` →
`smoke_location`. Then `smoke_vision` on the robot; save a
`.app_data/vision/*.ppm` snapshot and feed it into `smoke_locate --ppm`.

## Architecture

One "look" is a single-turn pipeline. `app.py` (`ScoutReachyApp`) orchestrates
the robot loop; the intelligence lives in `agent/`.

```
trigger (Enter or wake word)          app.py: _wait_for_trigger
  → capture one camera frame          robot/reachy.py: ReachyCamera.read → vision/frame.py: Frame (BGR)
  → encode to base64 PNG              vision/encode.py: frame_to_png_b64 (pure-Python, downsamples)
  → vision agent (image + location)   agent/scout.py: ScoutAgent.identify → Anthropic tool_runner
      → maps tools                    agent/tools.py → data/osm.py: geocode / reverse / nearby
  → guard                             agent/guard.py: guard_answer (drops unconfirmed locations)
  → LocationAnswer                    agent/schema.py (strict JSON schema)
  → TTS + sparse gestures             speech/tts.py + robot/expressions.py
```

Device location (`location.py`) is resolved per look and injected into the
tool context — the model never sends its own "near me" coordinates.

### The anti-hallucination invariant (the core design)

**No invented locations.** The model is never trusted to locate anything itself.
This is enforced in plain code, not by the prompt:

1. Every place returned by an OSM tool is recorded in the per-look
   `ScoutContext.seen` registry (`agent/tools.py`), keyed by its
   `<osm_type/osm_id>` token.
2. A `LocationAnswer` with `source != "unknown"` MUST carry an `osm_id` that a
   maps tool actually returned this turn (or coordinates within 50 m of a seen
   place). `agent/guard.py: guard_answer` checks this against `seen`.
3. If the claim can't be backed, the guard **downgrades** it to an honest "I can
   read the sign but can't confirm the spot" — nulling `osm_id`/`lat`/`lon` and
   setting `source = "unknown"`. Invented coordinates never reach the speaker.

The system prompt (`agent/prompts.py`) instructs the model to copy the exact
`<osm_type/id>` token and lat/lon from tool results; the guard is the backstop.
When changing the schema or tool output format, preserve the `<osm_type/id>`
token in `summarize_places` (`data/osm.py`) and the `osm_id` field — the guard
depends on both. `smoke_guard.py` and the `smoke_locate.py` invariant check
guard this.

### Hardware constraints inherited from the sibling projects

- **Motors and audio share one robot connection.** Gestures during playback stay
  sparse (`talk_beat`, ~0.1s spacing). `ECHO_SETTLE = 0.4s` after the robot
  speaks before opening the mic (`app.py`).
- **No compiled image wheels.** BitDefender quarantines OpenCV / Pillow `.pyd` on
  this machine. So: `vision/encode.py` hand-builds PNG with stdlib
  `zlib`/`struct` + numpy only; `vision/frame.py` saves stills as raw PPM (P6);
  the `winrt` location import in `location.py` is guarded and degrades to the
  `.env` manual fallback. Do not introduce OpenCV/Pillow.
- **Be polite to OpenStreetMap** (`data/osm.py`): it's a donated free service. A
  descriptive `User-Agent` is required (`OSM_USER_AGENT`), requests are throttled
  to ~1 req/s, and results are cached (60s TTL). No bulk queries.

### Config and keys

All runtime knobs live in `config.py` (`Settings`, pydantic-settings from `.env`);
nothing else reads `os.environ` directly — import `from scout_reachy.config import
settings`. See `.env.example` for the full list. The model is
`settings.scout_model` (= `VISION_MODEL` or `ANTHROPIC_MODEL`, must be
vision-capable). Required keys: `ANTHROPIC_API_KEY`, `OSM_USER_AGENT`,
`ELEVENLABS_API_KEY`/`ELEVENLABS_VOICE_ID`. `.env` is gitignored. OSM needs no key.
`GOOGLE_MAPS_API_KEY` is an unused future-backend seam.

## Follow-ups and session state

`ScoutAgent` keeps conversation history across a look session so spoken follow-ups
("how far is that?") reuse the image already in context via `ScoutAgent.ask`
(text-only, no new frame). History is cleared with `ScoutAgent.reset()` when a
session ends, and truncated on a failed turn to stay clean.

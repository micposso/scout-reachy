"""Smoke test: the anti-hallucination guard — fully OFFLINE (no keys, no robot).

    .venv\\Scripts\\python.exe scripts\\smoke_guard.py

The guard must PASS answers whose location is backed by a place a maps tool
surfaced this turn, and DOWNGRADE (to source=unknown, coords stripped) any
answer that invents a location. Mirrors the sibling project's smoke_verify.
"""

from __future__ import annotations

from scout_reachy.agent.guard import guard_answer
from scout_reachy.agent.schema import LocationAnswer
from scout_reachy.agent.tools import ScoutContext
from scout_reachy.data.osm import OSMClient, Place


def ctx_with(*places: Place) -> ScoutContext:
    c = ScoutContext(osm=object.__new__(OSMClient), device=(48.8584, 2.2945))
    c.record(list(places))
    return c


EIFFEL = Place(
    name="Eiffel Tower", address="Champ de Mars, Paris",
    lat=48.8584, lon=2.2945, category="tourism/attraction", osm_id="way/5013364",
)


def answer(**kw) -> LocationAnswer:
    base = dict(
        sign_text="Tour Eiffel", identified_place="Eiffel Tower",
        address="Champ de Mars, Paris", city="Paris", lat=48.8584, lon=2.2945,
        osm_id="way/5013364", distance_m=5.0, confidence="high", source="geocode",
        spoken_summary="You're at the Eiffel Tower.",
    )
    base.update(kw)
    return LocationAnswer(**base)


CASES = [
    # (description, answer, context, expect_source_after)
    ("confirmed by osm_id", answer(), ctx_with(EIFFEL), "geocode"),
    ("confirmed by coord match (token dropped)",
     answer(osm_id=None), ctx_with(EIFFEL), "geocode"),
    ("invented place (nothing seen)",
     answer(osm_id="way/999999", lat=40.0, lon=-100.0), ctx_with(), "unknown"),
    ("invented coords not near any result",
     answer(osm_id="way/999999", lat=51.5, lon=-0.12), ctx_with(EIFFEL), "unknown"),
    ("honest unknown stays unknown",
     answer(source="unknown", osm_id=None, lat=None, lon=None), ctx_with(), "unknown"),
]


def main() -> int:
    failures = 0
    for desc, ans, ctx, expect in CASES:
        out = guard_answer(ans, ctx)
        ok = out.source == expect
        # A downgraded answer must not leak coordinates.
        if expect == "unknown" and (out.lat is not None or out.osm_id is not None):
            ok = False
        mark = "OK  " if ok else "FAIL"
        print(f"[{mark}] {desc}: -> source={out.source}")
        if not ok:
            failures += 1
    if failures:
        print(f"\n[FAIL] {failures}/{len(CASES)} cases wrong")
        return 1
    print(f"\n[OK] all {len(CASES)} guard cases behave correctly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

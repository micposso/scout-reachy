"""Smoke test: the OpenStreetMap client — NETWORK, but no API key, no robot.

    .venv\\Scripts\\python.exe scripts\\smoke_osm.py

Hits the public Nominatim + Overpass endpoints (respecting the ~1 req/s
throttle). Checks: a known landmark geocodes near its real coordinates; reverse
geocoding those coordinates yields a plausible address; a nearby query returns
named POIs. Set OSM_USER_AGENT in .env first — Nominatim blocks the default.
"""

from __future__ import annotations

from scout_reachy.data.osm import OSMClient, summarize_places

# Eiffel Tower ground truth (approx).
EIFFEL = (48.8584, 2.2945)


def main() -> int:
    client = OSMClient()
    failures = 0

    print("1) geocode('Eiffel Tower, Paris')")
    hits = client.geocode("Eiffel Tower, Paris", limit=3)
    if not hits:
        print("[FAIL] no geocode results")
        return 1
    top = hits[0]
    d = top.distance_m(*EIFFEL)
    print(f"    top: {top.name} ({top.lat:.4f},{top.lon:.4f}) <{top.osm_id}> — {d:.0f} m off")
    if d > 2000:
        print(f"[FAIL] top result {d:.0f} m from the real Eiffel Tower")
        failures += 1
    else:
        print("[OK] geocode landed on the landmark")

    print("2) reverse(Eiffel Tower coords)")
    place = client.reverse(*EIFFEL)
    if place is None or not place.address:
        print("[FAIL] reverse geocode returned nothing")
        failures += 1
    else:
        print(f"[OK] reverse -> {place.address[:80]} (city={place.city or 'unavailable'})")

    print("3) nearby(category='tourism')")
    near = client.nearby(*EIFFEL, category="tourism", radius_m=400, limit=8)
    if not near:
        print("[FAIL] nearby returned no tourism POIs")
        failures += 1
    else:
        print("[OK] nearby POIs:")
        print("    " + summarize_places(near, origin=EIFFEL).replace("\n", "\n    "))

    if failures:
        print(f"\n[FAIL] {failures} check(s) failed")
        return 1
    print("\n[OK] OSM smoke complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

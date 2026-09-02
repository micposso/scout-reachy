"""OpenStreetMap client: Nominatim (geocode / reverse) + Overpass (nearby POIs).

Free, no API key. In return we MUST be polite or we get blocked:
  - a descriptive User-Agent identifying the app + a contact (settings.osm_user_agent),
  - at most ~1 request/second (a hard throttle below),
  - cache results on our side (60s TTL, like the sibling football client).
See https://operations.osmfoundation.org/policies/nominatim/

Methods hand back compact `Place` values and human-readable summary strings, not
raw JSON — fewer tokens for the agent, nothing to misparse. Modeled on the
sibling project's data/football.py.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass

import httpx

from ..config import settings

logger = logging.getLogger(__name__)


class OSMError(RuntimeError):
    pass


@dataclass
class Place:
    """One located place from OSM."""

    name: str
    address: str  # human-readable full address / display name
    lat: float
    lon: float
    category: str  # OSM class/type, e.g. "tourism/museum" or "highway/residential"
    osm_id: str  # "{osm_type}/{osm_id}", stable identity for the guard
    city: str = ""  # explicit locality for concise "what city?" answers

    def distance_m(self, lat: float, lon: float) -> float:
        return _haversine_m(self.lat, self.lon, lat, lon)


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres."""
    r = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


# (endpoint, sorted-params-tuple) -> (timestamp, list[Place])
_cache: dict[tuple, tuple[float, list[Place]]] = {}
_CACHE_TTL = 60.0
_MIN_INTERVAL = 1.0  # seconds between outbound requests (Nominatim policy)
_last_request_at = 0.0


def _throttle() -> None:
    global _last_request_at
    wait = _MIN_INTERVAL - (time.time() - _last_request_at)
    if wait > 0:
        time.sleep(wait)
    _last_request_at = time.time()


class OSMClient:
    def __init__(self) -> None:
        self._http = httpx.Client(
            headers={"User-Agent": settings.osm_user_agent},
            timeout=20.0,
            follow_redirects=True,
        )

    # -- Nominatim ---------------------------------------------------------

    def geocode(
        self,
        query: str,
        *,
        near: tuple[float, float] | None = None,
        limit: int = 5,
    ) -> list[Place]:
        """Name/address -> candidate places. When `near` (lat, lon) is given the
        search is biased to a box around that point (prefer_close), so an
        ambiguous sign like 'Main Street' resolves to the nearby one."""
        params: dict[str, str] = {
            "q": query,
            "format": "jsonv2",
            "addressdetails": "1",
            "limit": str(limit),
        }
        if near is not None:
            lat, lon = near
            d = 0.5  # ~50 km half-box; bias without hard-excluding a good hit
            params["viewbox"] = f"{lon - d},{lat + d},{lon + d},{lat - d}"
            # bounded=0: prefer the box but still allow far matches as a fallback.
            params["bounded"] = "0"
        rows = self._get_json(f"{settings.nominatim_base}/search", params)
        return [self._to_place(r) for r in rows]

    def reverse(self, lat: float, lon: float) -> Place | None:
        """Coordinates -> the address of that spot (where you're standing)."""
        params = {
            "lat": f"{lat}",
            "lon": f"{lon}",
            "format": "jsonv2",
            "addressdetails": "1",
        }
        row = self._get_json(f"{settings.nominatim_base}/reverse", params)
        if isinstance(row, dict) and row.get("error"):
            return None
        if isinstance(row, list):
            return self._to_place(row[0]) if row else None
        return self._to_place(row)

    # -- Overpass ----------------------------------------------------------

    def nearby(
        self,
        lat: float,
        lon: float,
        *,
        category: str | None = None,
        radius_m: int | None = None,
        limit: int = 12,
    ) -> list[Place]:
        """Named POIs around a point. `category` is an OSM key like 'amenity',
        'tourism', 'shop'; None gets any named feature. Sorted by distance."""
        radius = radius_m or settings.osm_nearby_radius_m
        key = category or "name"
        selector = f'[{category}]' if category else '["name"]'
        q = (
            f"[out:json][timeout:25];"
            f'(node{selector}(around:{radius},{lat},{lon});'
            f'way{selector}(around:{radius},{lat},{lon}););'
            f"out center {limit * 3};"
        )
        cache_key = ("overpass", key, radius, round(lat, 5), round(lon, 5), limit)
        cached = _cache.get(cache_key)
        if cached and time.time() - cached[0] < _CACHE_TTL:
            return cached[1]
        _throttle()
        try:
            resp = self._http.post(settings.overpass_base, data={"data": q})
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise OSMError(f"Overpass request failed: {exc}") from exc
        places: list[Place] = []
        for el in resp.json().get("elements", []):
            tags = el.get("tags") or {}
            name = tags.get("name")
            if not name:
                continue
            center = el if "lat" in el else el.get("center", {})
            plat, plon = center.get("lat"), center.get("lon")
            if plat is None or plon is None:
                continue
            cat = next(
                (f"{k}/{tags[k]}" for k in ("amenity", "tourism", "shop", "highway")
                 if k in tags),
                tags.get("amenity") or "poi",
            )
            places.append(Place(
                name=name, address=tags.get("addr:street", ""), lat=plat, lon=plon,
                category=cat, osm_id=f"{el.get('type')}/{el.get('id')}",
                city=_locality(tags),
            ))
        places.sort(key=lambda p: p.distance_m(lat, lon))
        places = places[:limit]
        _cache[cache_key] = (time.time(), places)
        return places

    # -- internals ---------------------------------------------------------

    def _get_json(self, url: str, params: dict):
        cache_key = (url, tuple(sorted(params.items())))
        cached = _cache.get(cache_key)
        if cached and time.time() - cached[0] < _CACHE_TTL:
            return cached[1]
        _throttle()
        try:
            resp = self._http.get(url, params=params)
        except httpx.HTTPError as exc:
            raise OSMError(f"Nominatim request failed: {exc}") from exc
        if resp.status_code == 429:
            raise OSMError("Nominatim rate limit hit (be gentler; ~1 req/s).")
        if resp.status_code >= 400:
            raise OSMError(f"Nominatim returned {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        # Only list responses (search) get memoized as Place lists elsewhere; the
        # reverse endpoint returns a dict and is cached here as-is.
        _cache[cache_key] = (time.time(), data)
        return data

    @staticmethod
    def _to_place(row: dict) -> Place:
        cls, typ = row.get("class", ""), row.get("type", "")
        address = row.get("address") or {}
        return Place(
            name=row.get("name") or (row.get("display_name") or "").split(",")[0],
            address=row.get("display_name", ""),
            lat=float(row["lat"]),
            lon=float(row["lon"]),
            category=f"{cls}/{typ}".strip("/"),
            osm_id=f"{row.get('osm_type', '?')}/{row.get('osm_id', '?')}",
            city=_locality(address),
        )


def _locality(address: dict) -> str:
    """Extract a human city/locality from Nominatim address details or OSM tags."""
    value = next(
        (
            str(address[key]).strip()
            for key in ("city", "addr:city", "town", "village", "municipality")
            if address.get(key)
        ),
        "",
    )
    # Nominatim conventionally calls the municipality "New York". In speech,
    # distinguish the city from New York State.
    if value in ("New York", "City of New York"):
        return "New York City"
    if not value and address.get("borough") in {
        "Manhattan", "Brooklyn", "Queens", "The Bronx", "Staten Island"
    }:
        return "New York City"
    return value


def summarize_places(
    places: list[Place], *, origin: tuple[float, float] | None = None
) -> str:
    """Compact one-line-per-place text for an agent tool result. When `origin`
    is given, each line carries the distance + rough bearing from it."""
    if not places:
        return "No matching places found."
    lines = []
    for p in places:
        extra = f" [{p.category}]" if p.category else ""
        if p.city:
            extra += f" — city: {p.city}"
        loc = f" ({p.lat:.5f},{p.lon:.5f})"
        if origin is not None:
            d = p.distance_m(*origin)
            bearing = _bearing(origin[0], origin[1], p.lat, p.lon)
            extra += f" — {_fmt_dist(d)} {bearing}"
        addr = f" — {p.address}" if p.address else ""
        lines.append(f"{p.name}{loc}{extra}{addr} <{p.osm_id}>")
    return "\n".join(lines)


def _fmt_dist(m: float) -> str:
    return f"{m:.0f} m" if m < 1000 else f"{m / 1000:.1f} km"


def _bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> str:
    """Rough compass direction from point 1 to point 2."""
    dlon = math.radians(lon2 - lon1)
    y = math.sin(dlon) * math.cos(math.radians(lat2))
    x = math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) - math.sin(
        math.radians(lat1)
    ) * math.cos(math.radians(lat2)) * math.cos(dlon)
    deg = (math.degrees(math.atan2(y, x)) + 360) % 360
    dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    return dirs[round(deg / 45) % 8]

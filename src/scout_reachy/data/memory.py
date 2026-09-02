"""Persistent, local memory of places Scout has physically visited.

Only map-confirmed places are written here, and only when the robot's device
location is close to the place.  Memory is useful context for the agent, but it
is deliberately not part of the live-location guard: remembering a place is not
proof that the robot is there now.
"""

from __future__ import annotations

import sqlite3
import math
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol


class PlaceLike(Protocol):
    osm_id: str
    name: str
    address: str
    lat: float
    lon: float


def _distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance, kept local so memory itself is stdlib-only."""
    radius = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * radius * math.asin(min(1.0, math.sqrt(a)))


@dataclass(frozen=True)
class RememberedPlace:
    osm_id: str
    name: str
    address: str
    lat: float
    lon: float
    first_visited_at: str
    last_visited_at: str
    visit_count: int
    last_sign_text: str

    def distance_m(self, lat: float, lon: float) -> float:
        return _distance_m(self.lat, self.lon, lat, lon)


class PlaceMemory:
    """Small SQLite-backed visit history.

    A connection is opened per operation so the store is safe to use from the
    app's short-lived motion/audio threads and survives process restarts.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    @contextmanager
    def _connection(self):
        """Yield a transactional connection and always release its file handle."""
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _initialize(self) -> None:
        with self._connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS places (
                    osm_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    address TEXT NOT NULL,
                    lat REAL NOT NULL,
                    lon REAL NOT NULL,
                    first_visited_at TEXT NOT NULL,
                    last_visited_at TEXT NOT NULL,
                    visit_count INTEGER NOT NULL DEFAULT 1,
                    last_sign_text TEXT NOT NULL DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS visits (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    osm_id TEXT NOT NULL REFERENCES places(osm_id),
                    visited_at TEXT NOT NULL,
                    sign_text TEXT NOT NULL DEFAULT ''
                );

                CREATE INDEX IF NOT EXISTS visits_osm_id_idx
                    ON visits(osm_id, visited_at);
                """
            )

    @staticmethod
    def _from_row(row: sqlite3.Row) -> RememberedPlace:
        return RememberedPlace(
            osm_id=row["osm_id"],
            name=row["name"],
            address=row["address"],
            lat=float(row["lat"]),
            lon=float(row["lon"]),
            first_visited_at=row["first_visited_at"],
            last_visited_at=row["last_visited_at"],
            visit_count=int(row["visit_count"]),
            last_sign_text=row["last_sign_text"],
        )

    def get(self, osm_id: str) -> RememberedPlace | None:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM places WHERE osm_id = ?", (osm_id,)
            ).fetchone()
        return self._from_row(row) if row is not None else None

    def record_visit(
        self,
        place: PlaceLike,
        *,
        sign_text: str = "",
        visited_at: datetime | None = None,
    ) -> RememberedPlace:
        """Record one confirmed visit and return the updated place memory."""
        when = (visited_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
        stamp = when.isoformat(timespec="seconds")
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO places (
                    osm_id, name, address, lat, lon, first_visited_at,
                    last_visited_at, visit_count, last_sign_text
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?)
                ON CONFLICT(osm_id) DO UPDATE SET
                    name = excluded.name,
                    address = excluded.address,
                    lat = excluded.lat,
                    lon = excluded.lon,
                    last_visited_at = excluded.last_visited_at,
                    visit_count = places.visit_count + 1,
                    last_sign_text = excluded.last_sign_text
                """,
                (
                    place.osm_id, place.name, place.address, place.lat, place.lon,
                    stamp, stamp, sign_text.strip(),
                ),
            )
            conn.execute(
                "INSERT INTO visits (osm_id, visited_at, sign_text) VALUES (?, ?, ?)",
                (place.osm_id, stamp, sign_text.strip()),
            )
        remembered = self.get(place.osm_id)
        assert remembered is not None
        return remembered

    def search(self, query: str, *, limit: int = 8) -> list[RememberedPlace]:
        """Find remembered places by name, address, or observed sign text."""
        needle = query.strip().casefold()
        if not needle:
            return []
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM places
                WHERE instr(lower(name), ?) > 0
                   OR instr(lower(address), ?) > 0
                   OR instr(lower(last_sign_text), ?) > 0
                ORDER BY last_visited_at DESC
                LIMIT ?
                """,
                (needle, needle, needle, limit),
            ).fetchall()
        return [self._from_row(row) for row in rows]

    def nearby(
        self, lat: float, lon: float, *, radius_m: float, limit: int = 8
    ) -> list[RememberedPlace]:
        """Return memories within a radius, nearest first."""
        with self._connection() as conn:
            rows = conn.execute("SELECT * FROM places").fetchall()
        places = [self._from_row(row) for row in rows]
        places = [p for p in places if p.distance_m(lat, lon) <= radius_m]
        places.sort(key=lambda p: p.distance_m(lat, lon))
        return places[:limit]


def summarize_memories(
    places: list[RememberedPlace], *, origin: tuple[float, float] | None = None
) -> str:
    """Compact agent-facing text that clearly labels memory as non-authoritative."""
    if not places:
        return "No matching places in local visit memory."
    lines = ["LOCAL MEMORY ONLY (not proof of the robot's current location):"]
    for place in places:
        distance = ""
        if origin is not None:
            distance = f", {place.distance_m(*origin):.0f} m from current device location"
        times = "once" if place.visit_count == 1 else f"{place.visit_count} times"
        lines.append(
            f"{place.name} — visited {times}, last {place.last_visited_at}{distance}; "
            f"{place.address} [remembered OSM {place.osm_id}]"
        )
    return "\n".join(lines)

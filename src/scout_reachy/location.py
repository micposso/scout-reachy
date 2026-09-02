"""Where the robot/laptop is — needed to bias 'near me' lookups.

The Reachy Mini Lite has no GPS; it rides on a laptop. Best available signal is
the Windows location service (WiFi triangulation via winrt). That wheel is
compiled, so the import is guarded — if BitDefender quarantines it, or the user
denies the permission, or we're on a desktop, we fall back cleanly:

    Windows Geolocator  ->  .env DEVICE_LAT/DEVICE_LON  ->  .env DEVICE_ADDRESS
    (geocoded once)  ->  None (agent then relies on the sign text itself).

`resolve_device_location()` returns a Coord or None and never raises.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .config import settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Coord:
    lat: float
    lon: float
    source: str  # "windows" | "manual" | "address" — for logging/telemetry
    accuracy_m: float | None = None

    def as_tuple(self) -> tuple[float, float]:
        return (self.lat, self.lon)


def _windows_location() -> Coord | None:
    """WiFi/GNSS location via winrt. Returns None on any failure (no wheel,
    permission denied, no fix)."""
    try:
        import asyncio

        from winrt.windows.devices.geolocation import (  # type: ignore
            GeolocationAccessStatus,
            Geolocator,
            PositionAccuracy,
        )
    except Exception as exc:  # ImportError, or a quarantined .pyd
        logger.info("Windows location unavailable (%s); using fallback.", exc)
        return None

    async def _get() -> Coord | None:
        access = await Geolocator.request_access_async()
        if access != GeolocationAccessStatus.ALLOWED:
            logger.warning("Windows location permission not granted (%s).", access)
            return None
        geo = Geolocator()
        geo.desired_accuracy = PositionAccuracy.DEFAULT
        pos = await geo.get_geoposition_async()
        c = pos.coordinate
        point = c.point.position  # BasicGeoposition: latitude, longitude
        acc = getattr(c, "accuracy", None)
        return Coord(
            lat=float(point.latitude),
            lon=float(point.longitude),
            source="windows",
            accuracy_m=float(acc) if acc is not None else None,
        )

    try:
        return asyncio.run(_get())
    except Exception as exc:
        logger.warning("Windows location lookup failed (%s); using fallback.", exc)
        return None


def _manual_location() -> Coord | None:
    if settings.device_lat is not None and settings.device_lon is not None:
        return Coord(float(settings.device_lat), float(settings.device_lon), "manual")
    return None


def _address_location() -> Coord | None:
    """Geocode the .env DEVICE_ADDRESS once, if set."""
    if not settings.device_address:
        return None
    try:
        from .data.osm import OSMClient

        hits = OSMClient().geocode(settings.device_address, limit=1)
    except Exception as exc:
        logger.warning("Could not geocode DEVICE_ADDRESS (%s).", exc)
        return None
    if not hits:
        return None
    return Coord(hits[0].lat, hits[0].lon, "address")


def resolve_device_location() -> Coord | None:
    """Best available device location, or None. Never raises."""
    if settings.use_windows_location:
        coord = _windows_location()
        if coord is not None:
            logger.info("Device location: %.5f,%.5f (windows, +/-%sm)",
                        coord.lat, coord.lon, coord.accuracy_m)
            return coord
    for fallback in (_manual_location, _address_location):
        coord = fallback()
        if coord is not None:
            logger.info("Device location: %.5f,%.5f (%s)",
                        coord.lat, coord.lon, coord.source)
            return coord
    logger.info("No device location available; relying on sign text alone.")
    return None

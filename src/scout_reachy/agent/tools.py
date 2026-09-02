"""Agent tools: OpenStreetMap geocode / reverse / nearby.

The custom tools return compact human-readable strings, not JSON. Every place a
tool surfaces is recorded in the active ScoutContext so the guard can later
confirm the model's answer refers to a place the maps actually returned (no
invented coordinates). The context is set per look by ScoutAgent.

The device location is injected via the context, not passed by the model, so the
model can't spoof "near me" — biasing is always the robot's real position.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from anthropic import beta_tool

from ..data.osm import OSMClient, OSMError, Place, summarize_places


@dataclass
class ScoutContext:
    """Per-look tool state: the OSM client, the robot's device location, and a
    registry of every place surfaced this turn (keyed by <osm_type/id>)."""

    osm: OSMClient
    device: tuple[float, float] | None = None
    seen: dict[str, Place] = field(default_factory=dict)

    def record(self, places: list[Place]) -> None:
        for p in places:
            self.seen[p.osm_id] = p


_ctx: ScoutContext | None = None


def set_context(ctx: ScoutContext | None) -> None:
    """Install the context the tools operate against for the next turn."""
    global _ctx
    _ctx = ctx


def _c() -> ScoutContext:
    if _ctx is None:
        raise RuntimeError("ScoutContext not set — call set_context() before the turn.")
    return _ctx


@beta_tool
def geocode(query: str) -> str:
    """Locate a place by the text you read on a sign, a street name, a business
    name, or a landmark. Results are automatically biased to the robot's current
    location so nearby matches rank first. Call this to turn what you SEE into a
    real spot on the map.

    Args:
        query: The sign / street / landmark / business text to locate.
    """
    ctx = _c()
    try:
        places = ctx.osm.geocode(query, near=ctx.device)
    except OSMError as exc:
        return f"ERROR: {exc}"
    ctx.record(places)
    return summarize_places(places, origin=ctx.device)


@beta_tool
def reverse_geocode() -> str:
    """Get the street address of where the robot is standing right now, from its
    own device location. Use this to answer "where am I?".
    """
    ctx = _c()
    if ctx.device is None:
        return "The robot's location is unknown, so I can't reverse-geocode a spot."
    try:
        place = ctx.osm.reverse(*ctx.device)
    except OSMError as exc:
        return f"ERROR: {exc}"
    if place is None:
        return "No address found for the current location."
    ctx.record([place])
    return summarize_places([place], origin=ctx.device)


@beta_tool
def nearby(category: str) -> str:
    """List named places around the robot's current location, nearest first.
    Use this for "what's around me" or to confirm which nearby place a sign
    refers to.

    Args:
        category: An OSM category key: amenity, tourism, shop, highway — or
            "any" for all named features.
    """
    ctx = _c()
    if ctx.device is None:
        return "The robot's location is unknown, so I can't list nearby places."
    cat = None if category.strip().lower() in ("any", "all", "") else category.strip()
    try:
        places = ctx.osm.nearby(*ctx.device, category=cat)
    except OSMError as exc:
        return f"ERROR: {exc}"
    ctx.record(places)
    return summarize_places(places, origin=ctx.device)


TOOLS = [geocode, reverse_geocode, nearby]

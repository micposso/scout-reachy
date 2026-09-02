"""Anti-hallucination gate for the scout answer — the vision counterpart to the
sibling project's fact-check verifier.

A LocationAnswer that claims a real location (source != "unknown") MUST be
backed by a place a maps tool actually returned this turn. We confirm that in
plain code against the ScoutContext registry — the model is never trusted to
locate itself. If the claim can't be tied to a surfaced place, we DOWNGRADE it
to an honest "I can read the sign but can't confirm the spot" answer rather than
letting invented coordinates reach the speaker.
"""

from __future__ import annotations

import logging

from ..data.osm import Place, _haversine_m
from .schema import LocationAnswer
from .tools import ScoutContext

logger = logging.getLogger(__name__)

# A copied coordinate may round slightly; accept a near match to a seen place.
_COORD_TOLERANCE_M = 50.0


def backing_place(answer: LocationAnswer, ctx: ScoutContext) -> Place | None:
    """Return the map place backing this answer, if one surfaced this turn."""
    if answer.osm_id and answer.osm_id in ctx.seen:
        return ctx.seen[answer.osm_id]
    # Fall back to a coordinate match, in case the model dropped the token but
    # copied real coordinates from a result.
    if answer.lat is not None and answer.lon is not None:
        for place in ctx.seen.values():
            if _haversine_m(answer.lat, answer.lon, place.lat, place.lon) <= _COORD_TOLERANCE_M:
                return place
    return None


def guard_answer(answer: LocationAnswer, ctx: ScoutContext) -> LocationAnswer:
    """Return a trustworthy answer: the original if its location is backed by a
    maps result, otherwise a downgraded 'unconfirmed' version."""
    if answer.source == "unknown":
        return answer  # nothing located, nothing to confirm
    if backing_place(answer, ctx) is not None:
        logger.info("Location confirmed against maps result: %s", answer.osm_id)
        return answer

    logger.warning(
        "Rejecting unconfirmed location (osm_id=%r, %s,%s) — not in tool results; "
        "downgrading to sign-only.",
        answer.osm_id, answer.lat, answer.lon,
    )
    read = answer.sign_text.strip()
    if read:
        spoken = (
            f"I can read \"{read}\", but I couldn't confirm that spot on the map, "
            "so I won't guess where it is."
        )
    else:
        spoken = (
            "I couldn't confirm a location on the map for what I'm seeing, "
            "so I won't guess."
        )
    return answer.model_copy(update={
        "identified_place": None,
        "address": None,
        "city": None,
        "lat": None,
        "lon": None,
        "osm_id": None,
        "distance_m": None,
        "confidence": "low",
        "source": "unknown",
        "spoken_summary": spoken,
    })

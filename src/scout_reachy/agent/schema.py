"""Structured output schema for the scout vision agent.

The agent returns one LocationAnswer per look. Optional fields follow the
required-but-nullable convention (Optional[...] with no default) so the strict
JSON schema lists every key. The `osm_id` field is the linchpin of the guard:
a located answer MUST carry the <osm_type/id> token from a maps tool result.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class LocationAnswer(BaseModel):
    sign_text: str = Field(
        description=(
            "The text you read on any sign, or the landmark you recognized, in "
            "the image. Empty string if you see no readable sign or known place."
        )
    )
    identified_place: Optional[str] = Field(
        description="Best-guess place name for what you see, or null if unsure."
    )
    address: Optional[str] = Field(
        description="Full address of the place, copied from a maps tool result; null if none."
    )
    lat: Optional[float] = Field(
        description="Latitude, copied EXACTLY from the backing maps tool result; null if unconfirmed."
    )
    lon: Optional[float] = Field(
        description="Longitude, copied EXACTLY from the backing maps tool result; null if unconfirmed."
    )
    osm_id: Optional[str] = Field(
        description=(
            "The <osm_type/id> token (shown in angle brackets in the tool result) "
            "of the place this answer refers to. REQUIRED whenever source is not "
            "'unknown'. null only when no maps tool confirmed a location."
        )
    )
    distance_m: Optional[float] = Field(
        description="Approximate distance in metres from the robot, or null if unknown."
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="Your confidence that identified_place is correct."
    )
    source: Literal["geocode", "reverse", "nearby", "unknown"] = Field(
        description=(
            "Which maps tool backs this answer. 'unknown' means you could read the "
            "sign but no tool confirmed a real location — then osm_id/lat/lon are null."
        )
    )
    spoken_summary: str = Field(
        description=(
            "1-3 natural sentences for text-to-speech describing what you see and "
            "where it is. No markdown, no raw coordinates read aloud."
        )
    )


def _strictify(node) -> None:
    """Set additionalProperties: false on every object node (incl. $defs)."""
    if isinstance(node, dict):
        if node.get("type") == "object":
            node["additionalProperties"] = False
        for value in node.values():
            _strictify(value)
    elif isinstance(node, list):
        for value in node:
            _strictify(value)


def location_schema() -> dict:
    """JSON schema for output_config.format (strict: no extra properties)."""
    schema = LocationAnswer.model_json_schema()
    _strictify(schema)
    return schema

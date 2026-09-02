"""The scout vision agent: one tool-runner turn per look.

ScoutAgent.identify(frame, device) sends the camera frame (as a PNG image block)
plus the robot's location to the model, which reads any sign/landmark and calls
the OSM tools to locate it, then returns a validated LocationAnswer. Every
answer passes through the guard, so an invented location can never reach the
speaker. Conversation history persists across a session for follow-ups
("how far is that?") that don't need a fresh image.

Modeled on the sibling project's agent/session.py (same tool_runner + strict
structured-output pattern), with an image content block added.
"""

from __future__ import annotations

import logging

import anthropic

from ..config import settings
from ..data.osm import OSMClient
from ..vision.encode import frame_to_png_b64
from ..vision.frame import Frame
from .guard import guard_answer
from .prompts import SYSTEM_PROMPT
from .schema import LocationAnswer, location_schema
from .tools import TOOLS, ScoutContext, set_context

logger = logging.getLogger(__name__)


class AgentError(RuntimeError):
    pass


class ScoutAgent:
    def __init__(self) -> None:
        self.client = anthropic.Anthropic()  # ANTHROPIC_API_KEY from env
        self.osm = OSMClient()
        self.messages: list[dict] = []

    def reset(self) -> None:
        """Forget the conversation (called when a look-session ends)."""
        self.messages = []

    def identify(
        self,
        frame: Frame,
        device: tuple[float, float] | None,
        *,
        question: str | None = None,
    ) -> LocationAnswer:
        """Look at one frame and return a guarded LocationAnswer."""
        b64, media_type = frame_to_png_b64(frame)
        loc_line = (
            f"The robot's current location is {device[0]:.5f}, {device[1]:.5f}."
            if device is not None
            else "The robot's current location is unknown."
        )
        ask = question or "What am I looking at, and where is it?"
        self.messages.append({
            "role": "user",
            "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": media_type, "data": b64}},
                {"type": "text", "text": f"{loc_line}\n{ask}"},
            ],
        })
        return self._run(device)

    def ask(self, text: str, device: tuple[float, float] | None) -> LocationAnswer:
        """Text-only follow-up on the image already in history."""
        self.messages.append({"role": "user", "content": text})
        return self._run(device)

    # -- internals ---------------------------------------------------------

    def _run(self, device: tuple[float, float] | None) -> LocationAnswer:
        ctx = ScoutContext(osm=self.osm, device=device)
        set_context(ctx)
        history_len = len(self.messages)
        try:
            answer = self._run_turn()
        except Exception:
            del self.messages[history_len:]  # keep history clean on failure
            raise
        finally:
            set_context(None)
        return guard_answer(answer, ctx)

    def _run_turn(self) -> LocationAnswer:
        runner = self.client.beta.messages.tool_runner(
            model=settings.scout_model,
            max_tokens=4000,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=self.messages,
            output_config={"format": {"type": "json_schema", "schema": location_schema()}},
        )
        final = None
        for message in runner:  # runner executes the OSM tools itself
            final = message
        if final is None:
            raise AgentError("Agent produced no response.")
        text = next((b.text for b in final.content if b.type == "text"), None)
        if text is None:
            raise AgentError(f"No text block in final response (stop: {final.stop_reason})")
        answer = LocationAnswer.model_validate_json(text)
        # History keeps the plain structured text — enough for follow-ups.
        self.messages.append({"role": "assistant", "content": text})
        return answer

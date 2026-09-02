"""Lightweight yes/no intent detection from a transcript.

Used for spoken confirmations ("read it?", "send it?"). Keyword-based so it's
instant and offline; the LLM is reserved for richer understanding.
"""

from __future__ import annotations

from enum import Enum

_AFFIRM = {
    "yes", "yeah", "yep", "yup", "sure", "ok", "okay", "please", "go", "ahead",
    "do", "read", "send", "confirm", "affirmative", "absolutely", "definitely",
    "correct", "right", "fine", "proceed",
}
_NEGATE = {
    "no", "nope", "nah", "don't", "dont", "stop", "cancel", "skip", "later",
    "negative", "never", "wait", "hold",
}


class Intent(str, Enum):
    YES = "yes"
    NO = "no"
    UNCLEAR = "unclear"


def _tokens(text: str) -> set[str]:
    return {
        "".join(c for c in w if c.isalpha())
        for w in (text or "").lower().split()
    }


def classify(text: str) -> Intent:
    """Classify a short spoken response as YES / NO / UNCLEAR."""
    words = _tokens(text)
    yes = bool(words & _AFFIRM)
    no = bool(words & _NEGATE)
    if yes and not no:
        return Intent.YES
    if no and not yes:
        return Intent.NO
    # Both or neither: lean on a couple of strong phrases, else unclear.
    low = (text or "").lower()
    if any(p in low for p in ("go ahead", "read it", "send it", "yes please")):
        return Intent.YES
    if any(p in low for p in ("don't", "do not", "not now", "leave it")):
        return Intent.NO
    return Intent.UNCLEAR


def is_affirmative(text: str) -> bool:
    return classify(text) is Intent.YES


class Gate(str, Enum):
    """Decision at the send-confirmation gate."""

    SEND = "send"
    REVISE = "revise"
    CANCEL = "cancel"
    UNCLEAR = "unclear"


_SEND = {"send", "yes", "yeah", "yep", "sure", "ok", "okay", "go", "ahead",
         "confirm", "approved", "approve", "ship", "fire"}
_CANCEL = {"cancel", "forget", "discard", "delete", "drop", "scrap", "stop",
           "abort", "nevermind"}
_REVISE = {"change", "revise", "edit", "redo", "rewrite", "rephrase", "fix",
           "different", "instead", "modify", "adjust", "tweak", "no", "nope"}


def gate_decision(text: str) -> Gate:
    """Three-way classify a reply to 'Should I send it?'.

    SEND only on a clear affirmative — this is the safety gate. 'no'/'change'
    routes to REVISE (offer to change), explicit cancel words to CANCEL.
    """
    low = (text or "").lower()
    words = _tokens(text)
    if any(p in low for p in ("send it", "go ahead", "yes please", "sounds good")):
        return Gate.SEND
    if any(p in low for p in ("leave it", "never mind", "not now", "don't send",
                              "do not send", "forget it")):
        return Gate.CANCEL
    send = bool(words & _SEND)
    cancel = bool(words & _CANCEL)
    revise = bool(words & _REVISE)
    # Prefer the safe/explicit signals; 'no' lives in REVISE so we offer a change.
    if send and not (cancel or revise):
        return Gate.SEND
    if cancel and not send:
        return Gate.CANCEL
    if revise and not send:
        return Gate.REVISE
    return Gate.UNCLEAR

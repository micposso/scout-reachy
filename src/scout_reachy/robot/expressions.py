"""Expressive head/antenna gestures for the Reachy Mini.

Maps assistant states (idle, listening, thinking, speaking, awaiting-confirm,
happy, sad) to head poses + antenna positions, plus a few dynamic gestures
(nod yes, shake no, celebrate). Built on goto_target with create_head_pose.

Conventions:
- Head pose angles are in DEGREES (create_head_pose(degrees=True)).
- Antenna positions are in RADIANS, as [left, right].
Magnitudes are kept modest/safe; flip a sign here if a gesture reads backwards
on the hardware.
"""

from __future__ import annotations

import logging
import random
from enum import Enum

import numpy as np

from reachy_mini.utils import create_head_pose
from reachy_mini.utils.interpolation import InterpolationTechnique as IT

logger = logging.getLogger(__name__)


def _ant(left_deg: float, right_deg: float) -> list[float]:
    """Antenna pose from degrees -> radians [left, right]."""
    return [np.deg2rad(left_deg), np.deg2rad(right_deg)]


# --- Announcer talk-beat flavours --------------------------------------------
# Each returns one pose tuple: (head kwargs in degrees, antennas [L,R] degrees,
# body_yaw in degrees, duration s). talk_beat() picks one at random and fires it
# as a SINGLE goto_target, so speech stays smooth (head/audio share one
# connection) while the variety + big amplitudes + whole-body sway read as a
# passionate sportscaster. Durations are >= the talk cadence so the motion tiles
# with no frozen gaps. Antenna swings stay under the celebrate/goal_dance range
# so the cue gestures still land as the loudest moments.


def _beat_point():
    """Emphatic chin-thrust toward the listener — 'WHAT a goal!'"""
    return (
        dict(pitch=random.uniform(-12, -6), roll=random.uniform(-4, 4),
             yaw=random.uniform(-6, 6)),
        [random.uniform(30, 42), random.uniform(30, 42)],
        random.uniform(-5, 5), random.uniform(0.85, 1.1),
    )


def _beat_crowd_glance():
    """Big turn to address the 'crowd' on one side, antennas leading."""
    side = random.choice((-1.0, 1.0))
    return (
        dict(pitch=random.uniform(-4, 4), roll=side * random.uniform(3, 7),
             yaw=side * random.uniform(14, 24)),
        [random.uniform(20, 38), random.uniform(20, 38)],
        side * random.uniform(6, 14), random.uniform(0.95, 1.3),
    )


def _beat_swagger():
    """Loose, confident whole-body roll sway."""
    side = random.choice((-1.0, 1.0))
    return (
        dict(pitch=random.uniform(-6, 2), roll=side * random.uniform(6, 11),
             yaw=side * random.uniform(-6, 6)),
        [random.uniform(15, 30), random.uniform(15, 30)],
        side * random.uniform(8, 16), random.uniform(0.95, 1.3),
    )


def _beat_lean_in():
    """Intimate confide — chin forward, calmer, a breather between hype."""
    return (
        dict(pitch=random.uniform(4, 9), roll=random.uniform(-4, 4),
             yaw=random.uniform(-8, 8)),
        [random.uniform(12, 22), random.uniform(12, 22)],
        random.uniform(-4, 4), random.uniform(0.9, 1.2),
    )


def _beat_hype_bob():
    """Quick up-bob with antenna asymmetry — building excitement."""
    hot = [random.uniform(28, 40), random.uniform(10, 24)]
    if random.random() < 0.5:
        hot.reverse()
    return (
        dict(pitch=random.uniform(-10, -4), roll=random.uniform(-7, 7),
             yaw=random.uniform(-10, 10)),
        hot, random.uniform(-6, 6), random.uniform(0.85, 1.05),
    )


# Weighted toward the punchy flavours; lean_in is the occasional breather.
_TALK_BEATS = [
    _beat_point, _beat_point,
    _beat_hype_bob, _beat_hype_bob,
    _beat_crowd_glance, _beat_crowd_glance,
    _beat_swagger,
    _beat_lean_in,
]


class State(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    AWAITING_CONFIRM = "awaiting_confirm"
    HAPPY = "happy"
    SAD = "sad"


# Static poses per state: (head kwargs in degrees, antennas [L,R] degrees).
_POSES: dict[State, tuple[dict, list[float]]] = {
    State.IDLE: (dict(roll=0, pitch=0, yaw=0), [0, 0]),
    # lean in, antennas perked up = attentive
    State.LISTENING: (dict(roll=0, pitch=8, yaw=0), [28, 28]),
    # head cocked, one antenna up one down = pondering
    State.THINKING: (dict(roll=10, pitch=4, yaw=10), [32, -8]),
    # chin up, antennas perked — an eager sportscaster ready to call it
    # (talk_beat layers big, varied motion on top of this base)
    State.SPEAKING: (dict(roll=0, pitch=-5, yaw=0), [22, 22]),
    # curious head tilt, antennas high = "well?"
    State.AWAITING_CONFIRM: (dict(roll=14, pitch=2, yaw=0), [36, 36]),
    # chin up, antennas high = pleased
    State.HAPPY: (dict(roll=0, pitch=-10, yaw=0), [40, 40]),
    # head down, antennas drooped = sorry
    State.SAD: (dict(roll=0, pitch=12, yaw=0), [-35, -35]),
}


class Expressions:
    """Drives head/antenna gestures on a connected ReachyMini."""

    def __init__(self, mini) -> None:
        self._mini = mini

    # -- lifecycle -----------------------------------------------------------

    def wake(self) -> None:
        """Enable motors and come to life (gentle)."""
        self._mini.enable_motors()
        self.set_state(State.IDLE, duration=1.2, method=IT.MIN_JERK)

    def rest(self) -> None:
        """Return to neutral and go to sleep."""
        try:
            self.set_state(State.IDLE, duration=1.0)
        finally:
            self._mini.goto_sleep()

    # -- static states -------------------------------------------------------

    def set_state(
        self,
        state: State,
        *,
        duration: float = 0.6,
        method: IT = IT.CARTOON,
        body_yaw: float = 0.0,
    ) -> None:
        head_kw, ant_deg = _POSES[state]
        head = create_head_pose(**head_kw)
        self._mini.goto_target(
            head=head,
            antennas=_ant(*ant_deg),
            body_yaw=body_yaw,
            duration=duration,
            method=method,
        )

    # -- dynamic gestures ----------------------------------------------------

    def nod_yes(self, times: int = 2) -> None:
        """Nod the head down/up to signal yes/affirmation."""
        for _ in range(times):
            self._mini.goto_target(
                head=create_head_pose(pitch=16), antennas=_ant(20, 20),
                duration=0.28, method=IT.CARTOON,
            )
            self._mini.goto_target(
                head=create_head_pose(pitch=-4), antennas=_ant(24, 24),
                duration=0.28, method=IT.CARTOON,
            )
        self.set_state(State.IDLE, duration=0.4)

    def shake_no(self, times: int = 2) -> None:
        """Shake the head left/right to signal no/cancel."""
        for _ in range(times):
            self._mini.goto_target(
                head=create_head_pose(yaw=18), antennas=_ant(-10, -10),
                duration=0.22, method=IT.CARTOON,
            )
            self._mini.goto_target(
                head=create_head_pose(yaw=-18), antennas=_ant(-10, -10),
                duration=0.22, method=IT.CARTOON,
            )
        self.set_state(State.IDLE, duration=0.4)

    def celebrate(self) -> None:
        """Happy wiggle: chin up, antennas flapping, body sway."""
        self.set_state(State.HAPPY, duration=0.4)
        for i in range(3):
            yaw = 14 if i % 2 == 0 else -14
            self._mini.goto_target(
                head=create_head_pose(pitch=-10, roll=yaw / 2),
                antennas=_ant(45 if i % 2 == 0 else 25, 25 if i % 2 == 0 else 45),
                body_yaw=np.deg2rad(yaw),
                duration=0.22, method=IT.CARTOON,
            )
        self.set_state(State.IDLE, duration=0.5)

    def look_around(self, sweeps: int = 2) -> None:
        """Gentle 'waking up and looking around' gesture: the head turns slowly
        from one side to the other while the antennas sway, as if curiously
        scanning the room for something. Smooth and unhurried (MIN_JERK), meant
        as a startup flourish so the robot visibly comes to life instead of just
        settling straight into idle."""
        for _ in range(sweeps):
            self._mini.goto_target(
                head=create_head_pose(yaw=24, pitch=3, roll=4),
                antennas=_ant(30, 10), duration=0.9, method=IT.MIN_JERK,
            )
            self._mini.goto_target(
                head=create_head_pose(yaw=-24, pitch=3, roll=-4),
                antennas=_ant(10, 30), duration=0.9, method=IT.MIN_JERK,
            )
        # settle back to a centered, neutral idle pose
        self.set_state(State.IDLE, duration=0.7, method=IT.MIN_JERK)

    def antenna_life(self, *, hold_head: bool = False) -> None:
        """One gentle antenna sway — the camera-safe building block of Pichi's
        'never fully still' idle life. Antennas always move; the head is touched
        only when hold_head=True, and then it is re-commanded to EXACT neutral so
        it neither sags nor shifts the head camera (a moving camera reads as
        motion to the presence detector). With hold_head=False the head is left
        wherever it is (e.g. the listening pose), so this keeps Pichi alive
        during any phase without disturbing the current head pose."""
        kwargs: dict = {}
        if hold_head:
            kwargs["head"] = create_head_pose(pitch=0, roll=0, yaw=0)
        a, b = random.uniform(8, 22), random.uniform(2, 14)
        if random.random() < 0.5:
            a, b = b, a
        self._mini.goto_target(
            antennas=_ant(a, b),
            duration=random.uniform(0.6, 0.95), method=IT.MIN_JERK, **kwargs,
        )

    def glance(self) -> None:
        """A subtle idle 'look around': one slow, small head turn to a random
        side with a soft antenna sway, then ease back to neutral. Gentler and
        shorter than look_around()'s full sweep — sprinkled into the idle
        breathing so Pichi keeps idly scanning the room without big motion."""
        side = random.choice((-1.0, 1.0))
        self._mini.goto_target(
            head=create_head_pose(
                yaw=side * random.uniform(10, 18),
                pitch=random.uniform(-2, 3),
                roll=side * random.uniform(2, 5),
            ),
            antennas=_ant(random.uniform(8, 18), random.uniform(8, 18)),
            duration=random.uniform(1.0, 1.4), method=IT.MIN_JERK,
        )
        self.set_state(State.IDLE, duration=random.uniform(0.8, 1.1), method=IT.MIN_JERK)

    def talk_beat(self) -> None:
        """One lively announcer beat — a single, bold conversational move with
        real passion: an emphatic chin-thrust point, a big glance out to the
        crowd, a swaggering whole-body sway, an excited hype-bob, or the
        occasional lean-in breather. A random flavour is picked each call
        (`_TALK_BEATS`, weighted toward the punchy ones) so back-to-back beats
        never look mechanical, with antennas dialled up high for energy.

        STILL exactly ONE goto_target per call on a smooth MIN_JERK ease
        (≈0.85–1.3s): head and audio share one connection, so the energy comes
        from *bigger, varied poses and whole-body sway* — never from flooding
        more commands. This keeps the ≥1s spacing / one-command-per-beat rule
        that keeps speech smooth (same lesson as the typewriter cue), so Pichi
        reads as an excited sportscaster instead of a gentle nodder."""
        head_kw, ant, body_yaw_deg, dur = random.choice(_TALK_BEATS)()
        self._mini.goto_target(
            head=create_head_pose(**head_kw),
            antennas=_ant(*ant),
            body_yaw=np.deg2rad(body_yaw_deg),
            duration=dur, method=IT.MIN_JERK,
        )

    def antenna_wiggle(self, times: int = 2) -> None:
        """Quick antenna wiggle — good as a 'thinking' flourish."""
        for _ in range(times):
            self._mini.goto_target(antennas=_ant(35, -10), duration=0.2, method=IT.CARTOON)
            self._mini.goto_target(antennas=_ant(-10, 35), duration=0.2, method=IT.CARTOON)

    def work_wobble(self, times: int = 1) -> None:
        """'Writing' cue motion: head held UP, drifting *very subtly* side to
        side with a small antenna sway — like a calm typist, not a head-bang.

        Deliberately gentle: small angles, slow MIN_JERK easing, few commands.
        It keeps *commanding the head every cycle* (chin up, pitch negative) so
        the head never sags during a long draft, while staying soft enough to
        read as "quietly working" — and sparse enough not to starve the audio
        cue that shares the robot connection."""
        for _ in range(times):
            self._mini.goto_target(
                head=create_head_pose(pitch=-4, roll=3, yaw=2),
                antennas=_ant(18, 10), duration=0.7, method=IT.MIN_JERK,
            )
            self._mini.goto_target(
                head=create_head_pose(pitch=-4, roll=-3, yaw=-2),
                antennas=_ant(10, 18), duration=0.7, method=IT.MIN_JERK,
            )

    def stadium_wave(self) -> None:
        """'The wave' — the crowd motion while Pichi fetches data: the head sways
        side to side (yaw + roll) with a whole-body lean while the antennas pump
        up and down out of phase, like fans rising in sequence. Bigger and more
        energetic than work_wobble, but still just TWO bounded commands per call
        (~1.2-1.6s) so it stays sparse enough not to make the stadium audio choppy
        on the shared connection (same ≥1s-per-burst rule as talk_beat)."""
        for side in (1.0, -1.0):
            lead_up = side > 0  # which antenna is 'up' leads the wave each side
            self._mini.goto_target(
                head=create_head_pose(
                    yaw=side * random.uniform(12, 20),
                    roll=side * random.uniform(6, 12),
                    pitch=random.uniform(-6, -2),
                ),
                antennas=_ant(*(
                    (random.uniform(38, 52), random.uniform(8, 20)) if lead_up
                    else (random.uniform(8, 20), random.uniform(38, 52))
                )),
                body_yaw=np.deg2rad(side * random.uniform(6, 12)),
                duration=random.uniform(0.6, 0.8), method=IT.MIN_JERK,
            )

    def breathe(self) -> None:
        """One slow 'breath': a barely-there pitch rise and fall with a soft
        antenna sway. Looped sparsely while idle so the robot reads as alive
        (and the motors never sag) without drawing attention."""
        self._mini.goto_target(
            head=create_head_pose(pitch=-2), antennas=_ant(6, 6),
            duration=1.4, method=IT.MIN_JERK,
        )
        self._mini.goto_target(
            head=create_head_pose(pitch=2), antennas=_ant(2, 2),
            duration=1.4, method=IT.MIN_JERK,
        )

    def goal_dance(self) -> None:
        """GOOOAL! A bigger celebrate: chin thrown up, antennas flapping wide,
        body swinging side to side. The loudest gesture in the repertoire —
        reserved for [GOAL] cues. Still bounded (~1.6s of commands) so it never
        starves the audio channel mid-narration."""
        self._mini.goto_target(
            head=create_head_pose(pitch=-14), antennas=_ant(55, 55),
            duration=0.25, method=IT.CARTOON,
        )
        for i in range(3):
            yaw = 20 if i % 2 == 0 else -20
            self._mini.goto_target(
                head=create_head_pose(pitch=-12, roll=yaw / 2),
                antennas=_ant(60 if i % 2 == 0 else 20, 20 if i % 2 == 0 else 60),
                body_yaw=np.deg2rad(yaw),
                duration=0.25, method=IT.CARTOON,
            )
        self.set_state(State.SPEAKING, duration=0.4)

    def slump(self) -> None:
        """Disappointment: head drops, antennas droop slowly — the crowd's
        'awww'. Used for [SAD] cues, then recovers toward speaking pose so the
        narration doesn't stay gloomy."""
        self._mini.goto_target(
            head=create_head_pose(pitch=14, roll=4),
            antennas=_ant(-35, -35), duration=0.9, method=IT.MIN_JERK,
        )
        self._mini.goto_target(
            head=create_head_pose(pitch=2),
            antennas=_ant(8, 8), duration=0.7, method=IT.MIN_JERK,
        )

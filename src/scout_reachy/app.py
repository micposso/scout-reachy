"""Scout Reachy — the look-and-locate voice loop.

    .venv\\Scripts\\python.exe -m scout_reachy.app

Flow per look:
    IDLE (gentle motion) -> trigger (press Enter, or the wake word) -> capture
    one camera frame -> ScoutAgent.identify (reads the sign, calls OSM tools) ->
    guard (drops any location the maps didn't confirm) -> speak the answer ->
    optional spoken follow-ups ("how far is that?") -> back to IDLE.

Hard constraints inherited from the sibling project's hardware lessons:
    - 0.4s echo settle after the robot speaks before opening the mic.
    - Gestures and audio share one connection; motion stays sparse.
"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager, nullcontext

from dotenv import load_dotenv

load_dotenv()  # the Anthropic SDK reads ANTHROPIC_API_KEY from raw env

from .agent.scout import ScoutAgent
from .config import settings
from .location import resolve_device_location
from .robot.expressions import State
from .robot.reachy import ReachyRobot
from .speech.tts import AnnouncerTTS
from .voice.stt import SpeechToText
from .voice.vad import FRAME_SAMPLES, UtteranceRecorder
from .voice.wakeword import CHUNK_SAMPLES, WakeWord

logger = logging.getLogger(__name__)

ECHO_SETTLE = 0.4  # seconds between robot speech and opening the mic
MAX_FOLLOWUPS = 3  # spoken follow-up questions allowed per look
MOVE_WHILE_LISTENING = True


class ScoutReachyApp:
    def __init__(self) -> None:
        logger.info("Loading components (STT, TTS, vision agent)...")
        self.stt = SpeechToText()
        self.tts = AnnouncerTTS()
        self.agent = ScoutAgent()
        self.wake = WakeWord() if settings.trigger_mode == "wakeword" else None
        self._phrase_cache: dict[str, object] = {}
        self.robot: ReachyRobot | None = None

    # ----- motion + speech helpers (shared pattern with the sibling app) ----

    @contextmanager
    def _motion(self, gesture, *, gap: float = 0.0):
        """Keep the robot moving during an otherwise-still phase."""
        stop = threading.Event()

        def _loop() -> None:
            while not stop.is_set():
                try:
                    gesture()
                except Exception as exc:
                    logger.debug("Motion loop ended: %s", exc)
                    return
                if gap:
                    stop.wait(gap)

        thread = threading.Thread(target=_loop, daemon=True)
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join(timeout=2.0)

    def say(self, text: str) -> None:
        """Speak a phrase (cached per phrase), keeping the robot gently moving."""
        clip = self._phrase_cache.get(text)
        if clip is None:
            clip = self.tts.say_quick(text)
            self._phrase_cache[text] = clip
        self._play_with_motion(clip)

    def _play_with_motion(self, clip) -> None:
        stop = threading.Event()

        def _move() -> None:
            try:
                while not stop.is_set():
                    self.robot.expressions.talk_beat()
                    stop.wait(0.1)
            except Exception as exc:
                logger.debug("Speak motion ended: %s", exc)

        mover = threading.Thread(target=_move, daemon=True)
        mover.start()
        try:
            self.robot.speaker.play(clip)
        finally:
            stop.set()
            mover.join(timeout=2.0)

    def listen(self, max_seconds: float = 8.0) -> str:
        """Echo-settle, record one utterance, transcribe it."""
        time.sleep(ECHO_SETTLE)
        rec = UtteranceRecorder(aggressiveness=2, silence_ms=700, max_seconds=max_seconds)
        gen = self.robot.mic.frames(FRAME_SAMPLES)
        expr = self.robot.expressions
        motion = (self._motion(lambda: expr.antenna_life(hold_head=False))
                  if MOVE_WHILE_LISTENING else nullcontext())
        try:
            with motion:
                clip = rec.record(gen)
        finally:
            gen.close()
        text = self.stt.transcribe(clip)
        logger.info("Heard: %r", text)
        return text

    # ----- capture + look ---------------------------------------------------

    def _capture(self):
        cam = self.robot.camera
        if cam is None or not cam.available:
            return None
        return cam.read(timeout=3.0)

    def _speak_answer(self, answer) -> None:
        """Voice the guarded answer with a matching gesture."""
        expr = self.robot.expressions
        if answer.source != "unknown" and answer.lat is not None:
            expr.nod_yes(times=1)
        else:
            expr.set_state(State.SAD, duration=0.4)
        self.say(answer.spoken_summary)

    def run_look_session(self, device) -> None:
        """One capture -> identify -> speak, then a short spoken follow-up loop."""
        expr = self.robot.expressions
        expr.set_state(State.THINKING, duration=0.5)
        frame = self._capture()
        if frame is None:
            self.say("I can't see anything — my camera isn't available right now.")
            return
        if frame.mean_brightness < 2.0:
            self.say("It's too dark to see — is the lens covered?")
            return

        self.say("Let me take a look...")
        try:
            with self._motion(expr.work_wobble):
                answer = self.agent.identify(frame, device)
        except Exception as exc:
            logger.exception("Vision agent failed: %s", exc)
            self.say("Something went wrong while I was looking. Let's try again.")
            return
        self._speak_answer(answer)

        # Spoken follow-ups on the same image ("how far is that?", "what's near?").
        for _ in range(MAX_FOLLOWUPS):
            self.say("Anything else about this?")
            reply = self.listen(max_seconds=8.0).strip()
            if not reply or reply.lower() in ("no", "nope", "no thanks", "that's all"):
                break
            try:
                with self._motion(expr.work_wobble):
                    follow = self.agent.ask(reply, device)
            except Exception as exc:
                logger.exception("Follow-up failed: %s", exc)
                self.say("I couldn't work that one out, sorry.")
                break
            self._speak_answer(follow)

    # ----- trigger ----------------------------------------------------------

    def _wait_for_trigger(self) -> str | None:
        """Block until a look should start. Returns 'go', or None to quit."""
        expr = self.robot.expressions
        if self.wake is not None:
            self.wake.reset()
            with self._motion(lambda: expr.antenna_life(hold_head=True)):
                mic_gen = self.robot.mic.frames(CHUNK_SAMPLES)
                try:
                    for chunk in mic_gen:
                        if self.wake.score(chunk) >= self.wake.threshold:
                            logger.info("Wake word detected")
                            return "go"
                finally:
                    mic_gen.close()
            return "go"
        # push-to-talk: block on Enter while the antennas keep swaying.
        with self._motion(lambda: expr.antenna_life(hold_head=False)):
            try:
                input()
            except (EOFError, KeyboardInterrupt):
                return None
        return "go"

    # ----- main loop --------------------------------------------------------

    def run(self) -> None:
        with ReachyRobot() as robot:
            self.robot = robot
            expr = robot.expressions
            expr.wake()
            expr.look_around(sweeps=1)
            if settings.trigger_mode == "wakeword":
                hint = f"Say '{settings.wakeword_model}' and point me at a sign."
            else:
                hint = "Point me at a sign and press Enter. Ctrl+C to quit."
            logger.info("Ready. %s", hint)
            self.say("Scout ready!")

            while True:
                trigger = self._wait_for_trigger()
                if trigger is None:
                    break
                device = resolve_device_location()
                try:
                    self.run_look_session(device.as_tuple() if device else None)
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    logger.exception("Look session error: %s", exc)
                    try:
                        self.say("We hit a snag — let's try that again.")
                    except Exception:
                        pass
                finally:
                    self.agent.reset()
                    expr.set_state(State.IDLE, duration=0.8)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        ScoutReachyApp().run()
    except KeyboardInterrupt:
        print("\nScout signing off. Goodbye!")


if __name__ == "__main__":
    main()

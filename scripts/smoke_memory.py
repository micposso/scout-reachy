r"""Smoke test: persistent place memory — fully offline.

    .venv\Scripts\python.exe scripts\smoke_memory.py
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from scout_reachy.data.memory import PlaceMemory, summarize_memories


def main() -> int:
    eiffel = SimpleNamespace(
        name="Eiffel Tower",
        address="Champ de Mars, Paris",
        lat=48.8584,
        lon=2.2945,
        category="tourism/attraction",
        osm_id="way/5013364",
    )
    Path(".app_data").mkdir(exist_ok=True)
    with TemporaryDirectory(dir=".app_data") as tmp:
        path = f"{tmp}/memory.db"
        memory = PlaceMemory(path)
        memory.record_visit(
            eiffel,
            sign_text="Tour Eiffel",
            visited_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        memory.record_visit(
            eiffel,
            sign_text="Tour Eiffel",
            visited_at=datetime(2026, 2, 1, tzinfo=timezone.utc),
        )

        # Reopen the database to prove the memory survives process instances.
        reopened = PlaceMemory(path)
        saved = reopened.get(eiffel.osm_id)
        nearby = reopened.nearby(48.8584, 2.2945, radius_m=50)
        searched = reopened.search("Eiffel")

        assert saved is not None and saved.visit_count == 2
        assert saved.first_visited_at.startswith("2026-01-01")
        assert saved.last_visited_at.startswith("2026-02-01")
        assert len(nearby) == 1 and len(searched) == 1
        summary = summarize_memories(nearby, origin=(48.8584, 2.2945))
        assert "LOCAL MEMORY ONLY" in summary and "visited 2 times" in summary

    print("[OK] visits persist, increment, and can be recalled safely")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

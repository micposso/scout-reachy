"""Central configuration, loaded from environment / .env file.

All runtime knobs live here so the rest of the codebase never reads os.environ
directly. Import the singleton via `from scout_reachy.config import settings`.
Kept intentionally parallel to the sibling WORLD-CUP-REACHY project so the shared
robot/voice/speech modules read the same field names.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings sourced from environment variables / .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---------- Anthropic (vision + reasoning brain) ----------
    # ANTHROPIC_API_KEY is read by the SDK directly from the environment.
    anthropic_model: str = Field(default="claude-sonnet-4-6")
    # The model that reads the camera frame. Must be VISION-capable. Defaults to
    # anthropic_model; override only if you want a different tier for vision.
    vision_model: str = Field(default="")

    @property
    def scout_model(self) -> str:
        """The model actually used for the vision turn."""
        return self.vision_model or self.anthropic_model

    # ---------- ElevenLabs (spoken output) ----------
    elevenlabs_api_key: str = Field(default="")
    elevenlabs_voice_id: str = Field(default="")
    elevenlabs_tts_model: str = Field(default="eleven_multilingual_v2")

    # ---------- OpenStreetMap (free maps backend) ----------
    # Nominatim REQUIRES a descriptive User-Agent identifying the app + a contact.
    # See https://operations.osmfoundation.org/policies/nominatim/
    osm_user_agent: str = Field(
        default="scout-reachy/0.1 (https://github.com/; personal robot project)"
    )
    nominatim_base: str = Field(default="https://nominatim.openstreetmap.org")
    overpass_base: str = Field(default="https://overpass-api.de/api/interpreter")
    # Radius for "what's around me" POI queries.
    osm_nearby_radius_m: int = Field(default=250)
    # Future higher-quality backend seam; unused while empty.
    google_maps_api_key: str = Field(default="")

    # ---------- Device location (where the robot/laptop is) ----------
    # Manual fallback used when the Windows Geolocator is unavailable or denied.
    # Set either lat+lon, or a free-text address that gets geocoded once.
    device_lat: float | None = Field(default=None)
    device_lon: float | None = Field(default=None)
    device_address: str = Field(default="")
    # Try the Windows WiFi-triangulated location first (winrt). Off => manual only.
    use_windows_location: bool = Field(default=True)

    # ---------- Trigger ----------
    # "push_to_talk" (press Enter, best outdoors) or "wakeword" (hands-free).
    trigger_mode: str = Field(default="push_to_talk")

    # ---------- Voice stack ----------
    whisper_model: str = Field(default="base.en")
    # Bundled name ("hey_jarvis") or a path to a custom .onnx.
    wakeword_model: str = Field(default="hey_jarvis")
    wakeword_threshold: float = Field(default=0.5)

    # ---------- Reachy Mini robot ----------
    reachy_connection_mode: str = Field(default="localhost_only")
    reachy_media_backend: str = Field(default="default")
    reachy_speaker_volume: float = Field(default=0.55)

    # ---------- Vision tuning ----------
    # Cap the frame width sent to the model (downsampled) to bound tokens/latency.
    vision_max_width: int = Field(default=1024)

    # ---------- Persistent place memory (local SQLite) ----------
    memory_enabled: bool = Field(default=True)
    memory_db_path: str = Field(default=".app_data/scout_memory.db")
    # A map result counts as a visit only when it is this close to the laptop's
    # own location. This prevents a photographed poster from becoming a visit.
    memory_visit_radius_m: float = Field(default=500.0)
    memory_recall_radius_m: float = Field(default=500.0)

    @field_validator("device_lat", "device_lon", mode="before")
    @classmethod
    def _blank_coordinate_is_none(cls, value):
        """Allow the documented blank .env values for optional coordinates."""
        return None if isinstance(value, str) and not value.strip() else value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached Settings singleton."""
    return Settings()


# Convenient module-level handle.
settings = get_settings()

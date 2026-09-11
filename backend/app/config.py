"""Runtime configuration. No magic numbers in pipeline code (AGENTS.md code rules)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]

# Invariant 11: the server binds loopback only. Remote access is a `tailscale serve` proxy
# onto loopback, never a tailnet bind.
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Paths
    db_path: Path = REPO_ROOT / "data" / "facematch.db"
    media_dir: Path = REPO_ROOT / "data" / "media"
    crops_dir: Path = REPO_ROOT / "data" / "crops"
    models_dir: Path = REPO_ROOT / "models"
    frontend_dist: Path = REPO_ROOT / "frontend" / "dist"
    fixtures_dir: Path = REPO_ROOT / "fixtures"

    # Server
    host: str = "127.0.0.1"
    port: int = 8000
    operator_name: str = "operator"

    # Models
    detector_model: str = "yunet-2023mar"
    # SFace is the shipped default so a fresh clone runs a permissively licensed model.
    # buffalo_l requires allow_noncommercial_models (C7).
    embedder_model: str = "sface-2021dec"
    allow_noncommercial_models: bool = False
    execution_provider: Literal[
        "CPUExecutionProvider", "CoreMLExecutionProvider"
    ] = "CPUExecutionProvider"

    # Pipeline (spec 6.2)
    sample_fps: float = 3.0
    min_embed_px: int = 80
    max_yaw: float = 35.0
    min_sharpness: float = 40.0
    min_det_score: float = 0.6
    embed_k: int = 5
    top_k: int = 3

    # Calibration (spec 10)
    fpir_target: float = 1e-3

    # Worker
    worker_poll_seconds: float = 0.5

    @field_validator("host")
    @classmethod
    def _loopback_only(cls, value: str) -> str:
        if value not in LOOPBACK_HOSTS:
            raise ValueError(
                f"HOST must be loopback (invariant 11), got {value!r}. "
                "Use `tailscale serve` for remote access."
            )
        return value

    def ensure_dirs(self) -> None:
        for path in (self.db_path.parent, self.media_dir, self.crops_dir, self.models_dir):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

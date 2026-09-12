"""Runtime configuration. No magic numbers in pipeline code (AGENTS.md code rules)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
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

    # Pipeline (spec 6.2). The editable ones carry their bounds here rather than in the
    # config API: PATCH /api/config validates by constructing a Settings, so one statement
    # of each bound serves the environment, the .env file and the wire.
    sample_fps: float = Field(default=3.0, gt=0.0)
    min_embed_px: int = Field(default=80, ge=1)
    max_yaw: float = Field(default=35.0, ge=0.0, le=90.0)
    # Laplacian variance of the detection box resampled to the 112x112 embed size, so the
    # number is a focus measure and not a resolution measure (see pipeline/quality.py).
    # 40.0 keeps 100% of 58 fixture faces at 1x and 98% upscaled 2.5x, while rejecting
    # every face blurred at gaussian radius >= 1.5.
    min_sharpness: float = Field(default=40.0, ge=0.0)
    min_det_score: float = Field(default=0.6, ge=0.0, le=1.0)
    nms_iou: float = Field(default=0.3, ge=0.0, le=1.0)
    embed_k: int = Field(default=5, ge=1)
    top_k: int = Field(default=3, ge=1)
    # Spec 6.4 offers mean-of-top-3 once a person has 5+ templates; max is the default
    # because it is what the calibration harness scores against.
    person_score_mode: Literal["max", "mean_top3"] = "max"
    # Re-match scores track means against the gallery in blocks (spec 6.2), never row by row.
    rematch_block_size: int = Field(default=4096, ge=1)

    # Ingest
    max_upload_bytes: int = 256 * 1024 * 1024
    # Screen capture (macOS). An interactive selection waits on the operator, so the
    # ceiling is a UX timeout, not a machine one: past it we report a cancelled capture.
    capture_timeout_seconds: float = 120.0

    # Calibration (spec 10)
    fpir_target: float = 1e-3
    eval_min_identities: int = 5

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

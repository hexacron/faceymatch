#!/usr/bin/env python3
"""Measure the fixed costs of the pipeline, so a performance change is proved not asserted.

Run from `backend/` so the project venv is active:

    cd backend && uv run python ../tools/bench_perf.py > /tmp/perf-before.json

Everything happens in a throwaway directory: a fresh database, a synthetic gallery, and
JPEG frames built by pasting fixture faces onto noise. `data/facematch.db` is never opened,
so the operator's evidence is not a benchmark fixture.

The numbers here are medians of repeated runs on one machine. They are comparable against
another run of *this* script on the same machine, and against nothing else.
"""

from __future__ import annotations

import argparse
import io
import json
import sqlite3
import statistics
import sys
import tempfile
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app import audit, models_lock, runtime_config
from app.config import Settings
from app.core import vectors
from app.core.registry import ActiveModels, get_active_models
from app.db.conn import connect, transaction
from app.db.migrate import migrate
from app.ids import new_id
from app.main import sync_models_table
from app.pipeline import align, decode, ingest, live, quality
from app.pipeline.matching import rematch
from app.pipeline.process import process_image
from app.thresholds import seed_default_threshold_set

REPO_ROOT = Path(__file__).resolve().parents[1]

# The gallery the plan measures against: big enough that a per-template Python reduction is
# visible, small enough to build in a second.
GALLERY_PERSONS = 1000
TEMPLATES_PER_PERSON = 5
# Stored track means to re-match. A rematch rewrites every one of these rows.
TRACK_COUNT = 500

# Frame sizes: 1080p, a 3K laptop share, and a 5K display.
FRAME_SIZES: tuple[tuple[int, int], ...] = ((1920, 1080), (3024, 1612), (5120, 2880))
FACE_COUNTS: tuple[int, ...] = (0, 1, 3)
# Fixture faces are 250x250 LFW crops; 2x puts the face itself well above `min_embed_px`.
FACE_SCALE = 2
# The live view encodes at q0.92 (frontend/src/lib/liveMatch.ts).
JPEG_QUALITY = 92

WARMUP = 1
REPEATS = 15
# Weight verification and the whole-image pipeline are slow enough that fewer runs settle.
SLOW_REPEATS = 5

ACTOR = "bench"


def median_ms(samples: list[float]) -> float:
    return round(statistics.median(samples), 3)


def timed(call: Callable[[], object], *, repeats: int) -> list[float]:
    """Wall time of `repeats` calls in ms, after one untimed warm-up."""
    for _ in range(WARMUP):
        call()
    samples: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        call()
        samples.append((time.perf_counter() - start) * 1000.0)
    return samples


def fixture_images(fixtures_dir: Path) -> list[Path]:
    """Fixture faces in a stable order, one per identity first so faces differ."""
    by_identity: list[list[Path]] = []
    for identity in sorted(path for path in fixtures_dir.iterdir() if path.is_dir()):
        images = sorted(path for path in identity.iterdir() if decode.is_supported_image(path))
        if images:
            by_identity.append(images)
    ordered: list[Path] = []
    depth = 0
    while any(len(images) > depth for images in by_identity):
        for images in by_identity:
            if len(images) > depth:
                ordered.append(images[depth])
        depth += 1
    return ordered


def synth_frame(
    faces: list[np.ndarray], *, width: int, height: int, count: int, rng: np.random.Generator
) -> bytes:
    """A JPEG of `count` fixture faces pasted onto noise, at the live view's quality.

    Noise rather than a flat fill: a uniform background compresses to almost nothing, and
    the decode cost of that JPEG would not resemble a real screen share.
    """
    canvas = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    for index in range(count):
        face = faces[index % len(faces)]
        tile_h, tile_w = face.shape[:2]
        # Spread the faces across the frame so the detector cannot merge them.
        left = min(width - tile_w, 40 + index * (tile_w + 60))
        top = min(height - tile_h, 40)
        if left < 0 or top < 0:
            raise ValueError(f"{count} faces do not fit in a {width}x{height} frame")
        canvas[top : top + tile_h, left : left + tile_w] = face
    buffer = io.BytesIO()
    Image.fromarray(canvas).save(buffer, format="JPEG", quality=JPEG_QUALITY)
    return buffer.getvalue()


def scaled_face(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    scaled = image.resize(
        (image.width * FACE_SCALE, image.height * FACE_SCALE), Image.Resampling.LANCZOS
    )
    return np.asarray(scaled, dtype=np.uint8)


def aligned_crops(paths: list[Path], models: ActiveModels, settings: Settings) -> np.ndarray:
    """Real aligned 112x112 crops, cycled up to eight, for the embedder benchmark."""
    crops: list[np.ndarray] = []
    for path in paths:
        image = decode.decode_image(path)
        passing = [
            detection
            for detection in models.detector.detect(image)
            if quality.evaluate(image, detection, settings).passed
        ]
        if not passing:
            continue
        best = max(passing, key=lambda item: item.score)
        crops.append(align.align_crop(image, best.landmarks))
        if len(crops) == 8:
            break
    if not crops:
        raise SystemExit("no fixture face passed the quality gate; cannot benchmark embed")
    distinct = len(crops)
    while len(crops) < 8:
        crops.append(crops[len(crops) % distinct])
    return np.stack(crops)


def seed_gallery(
    conn: sqlite3.Connection, *, embedder_model_id: str, dim: int, rng: np.random.Generator
) -> None:
    """A synthetic gallery and stored track means: the shapes matching actually reduces over."""
    now = audit.now_ts()
    case_id = new_id()
    media_id = new_id()
    with transaction(conn):
        conn.execute(
            "INSERT INTO cases (id, name, authorization_basis, created_at, created_by) "
            "VALUES (?, ?, ?, ?, ?)",
            (case_id, "bench", "benchmark fixture", now, ACTOR),
        )
        conn.execute(
            "INSERT INTO media (id, case_id, sha256, kind, path, source_url, acquired_at, "
            "width, height, duration_ms, fps, ingested_at, status) "
            "VALUES (?, ?, ?, 'image', ?, NULL, NULL, 1920, 1080, NULL, NULL, ?, 'done')",
            (media_id, case_id, "0" * 64, "bench/frame.jpg", now),
        )
        for person_index in range(GALLERY_PERSONS):
            person_id = new_id()
            conn.execute(
                "INSERT INTO persons (id, display_name, notes, do_not_enroll, status, "
                "enrolled_in_case_id, created_at, created_by) "
                "VALUES (?, ?, NULL, 0, 'enrolled', ?, ?, ?)",
                (person_id, f"bench-{person_index:04d}", case_id, now, ACTOR),
            )
            raw = rng.standard_normal((TEMPLATES_PER_PERSON, dim)).astype(np.float32)
            for row in vectors.l2_normalize(raw, axis=1):
                conn.execute(
                    "INSERT INTO templates (id, person_id, detection_id, source_case_id, "
                    "embedding, embedder_model_id, quality, status, created_at, created_by) "
                    "VALUES (?, ?, NULL, ?, ?, ?, NULL, 'active', ?, ?)",
                    (
                        new_id(),
                        person_id,
                        case_id,
                        vectors.to_blob(row),
                        embedder_model_id,
                        now,
                        ACTOR,
                    ),
                )
        means = vectors.l2_normalize(
            rng.standard_normal((TRACK_COUNT, dim)).astype(np.float32), axis=1
        )
        for mean in means:
            conn.execute(
                "INSERT INTO tracks (id, media_id, start_ms, end_ms, best_detection_id, "
                "cluster_id, embedding_mean, embedder_model_id) "
                "VALUES (?, ?, 0, 0, NULL, NULL, ?, ?)",
                (new_id(), media_id, vectors.to_blob(mean), embedder_model_id),
            )


def bench_frames(
    conn: sqlite3.Connection,
    settings: Settings,
    models: ActiveModels,
    faces: list[np.ndarray],
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(20260912)
    rows: list[dict[str, Any]] = []
    for width, height in FRAME_SIZES:
        for count in FACE_COUNTS:
            frame = synth_frame(faces, width=width, height=height, count=count, rng=rng)
            elapsed: list[float] = []
            stages: dict[str, list[float]] = {
                "decode": [],
                "detect": [],
                "quality_align": [],
                "embed": [],
                "match": [],
            }
            detected = 0
            for index in range(WARMUP + REPEATS):
                result = live.match_frame(conn, settings, models, frame=frame)
                if index < WARMUP:
                    continue
                detected = len(result.faces)
                elapsed.append(float(result.elapsed_ms))
                for key, value in result.timings.as_dict().items():
                    stages[key].append(value)
            rows.append(
                {
                    "width": width,
                    "height": height,
                    "faces_pasted": count,
                    "faces_detected": detected,
                    "bytes": len(frame),
                    "elapsed_ms": median_ms(elapsed),
                    **{key: median_ms(values) for key, values in stages.items()},
                }
            )
    return rows


def bench_embed(models: ActiveModels, crops: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for count in (1, 3, 8):
        batch = crops[:count]
        samples = timed(partial(models.embedder.embed, batch), repeats=REPEATS)
        rows.append({"crops": count, "ms": median_ms(samples)})
    return rows


def bench_process(
    conn: sqlite3.Connection, settings: Settings, models: ActiveModels, paths: list[Path]
) -> float:
    """Ingest and process distinct fixture images; a processed media is never reprocessed."""
    case_id = new_id()
    with transaction(conn):
        conn.execute(
            "INSERT INTO cases (id, name, authorization_basis, created_at, created_by) "
            "VALUES (?, ?, ?, ?, ?)",
            (case_id, "bench-process", "benchmark fixture", audit.now_ts(), ACTOR),
        )
    samples: list[float] = []
    for path in paths[: SLOW_REPEATS + WARMUP]:
        result = ingest.ingest_file(conn, settings, case_id=case_id, src=path, actor=ACTOR)
        start = time.perf_counter()
        process_image(conn, settings, models, media_id=result.media_id, actor=ACTOR)
        samples.append((time.perf_counter() - start) * 1000.0)
    return median_ms(samples[WARMUP:] or samples)


def bench_blocked_reason(
    settings: Settings, lock: models_lock.ModelsLock
) -> dict[str, Any]:
    """First and second call per lock entry: the second is the one a cache would shorten."""
    entries: dict[str, dict[str, float]] = {}
    for entry in lock.models:
        calls: list[float] = []
        for _ in range(2):
            start = time.perf_counter()
            runtime_config.blocked_reason(lock, entry.id, kind=entry.kind, settings=settings)
            calls.append(round((time.perf_counter() - start) * 1000.0, 3))
        entries[entry.id] = {"first": calls[0], "second": calls[1]}
    return {
        "entries": entries,
        "first_total": round(sum(item["first"] for item in entries.values()), 3),
        "second_total": round(sum(item["second"] for item in entries.values()), 3),
    }


def providers_of(models: ActiveModels) -> dict[str, Any]:
    """The providers the built sessions actually run on, so a CPU fallback is not silent."""

    def session_providers(adapter: object) -> list[str]:
        session = getattr(adapter, "_session", None)
        return [] if session is None else list(session.get_providers())

    return {
        "detector": session_providers(models.detector),
        "embedder": session_providers(models.embedder),
        "active": models.execution_provider,
    }


def scratch_settings(root: Path) -> Settings:
    """A throwaway install pointed at the real weights and the real fixtures."""
    return Settings(
        db_path=root / "data" / "facematch.db",
        media_dir=root / "media",
        crops_dir=root / "crops",
        frontend_dist=root / "dist",
    )


def run(root: Path) -> dict[str, Any]:
    settings = scratch_settings(root)
    settings.ensure_dirs()
    if not settings.fixtures_dir.is_dir():
        raise SystemExit(f"{settings.fixtures_dir} does not exist; set FIXTURES_DIR")

    lock_samples = timed(
        lambda: models_lock.verify(settings.models_dir), repeats=SLOW_REPEATS
    )
    lock = models_lock.verify(settings.models_dir)
    models = get_active_models(settings, lock)

    conn = connect(settings.db_path)
    try:
        migrate(conn)
        sync_models_table(conn, lock, ACTOR)
        with transaction(conn):
            seed_default_threshold_set(conn, model_id=settings.embedder_model, actor=ACTOR)
        seed_gallery(
            conn,
            embedder_model_id=models.embedder_model_id,
            dim=models.embedder.dim,
            rng=np.random.default_rng(1),
        )
        live.clear_gallery_cache()

        paths = fixture_images(settings.fixtures_dir)
        faces = [scaled_face(path) for path in paths[:3]]
        crops = aligned_crops(paths, models, settings)

        frames = bench_frames(conn, settings, models, faces)
        embed = bench_embed(models, crops)
        rematch_samples = timed(
            lambda: rematch(
                conn,
                settings,
                embedder_model_id=models.embedder_model_id,
                execution_provider=models.execution_provider,
                actor=ACTOR,
            ),
            repeats=SLOW_REPEATS,
        )
        process_ms = bench_process(conn, settings, models, paths)
    finally:
        conn.close()

    return {
        "providers": providers_of(models),
        "gallery": {
            "persons": GALLERY_PERSONS,
            "templates": GALLERY_PERSONS * TEMPLATES_PER_PERSON,
            "tracks": TRACK_COUNT,
        },
        "frames": frames,
        "embed": embed,
        "process_ms": process_ms,
        "rematch_ms": median_ms(rematch_samples),
        "lock_verify_ms": median_ms(lock_samples),
        "blocked_reason_ms": bench_blocked_reason(settings, lock),
    }


@contextmanager
def scratch_root(keep: Path | None) -> Iterator[Path]:
    if keep is not None:
        keep.mkdir(parents=True, exist_ok=True)
        yield keep
        return
    with tempfile.TemporaryDirectory(prefix="facematch-bench-") as name:
        yield Path(name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keep",
        type=Path,
        default=None,
        help="scratch directory to keep instead of a temp dir that is deleted",
    )
    args = parser.parse_args(argv)
    with scratch_root(args.keep) as root:
        print(json.dumps(run(root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())

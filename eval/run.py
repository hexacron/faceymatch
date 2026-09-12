"""Run detector/embedder calibration over labeled identity folders (spec section 10)."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import metrics
from app import audit, models_lock
from app.config import Settings
from app.core.registry import get_active_models
from app.db.conn import connect, transaction
from app.ids import new_id
from app.main import startup_checks
from app.pipeline import align, decode, quality


@dataclass(frozen=True, slots=True)
class Sample:
    label: str
    path: Path
    embedding: np.ndarray


def collect_samples(root: Path, settings: Settings) -> list[Sample]:
    """Embed the best quality-passing face in every image under `<root>/<identity>/`."""
    lock = models_lock.verify(settings.models_dir)
    active = get_active_models(settings, lock)
    samples: list[Sample] = []
    for identity_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for path in sorted(identity_dir.rglob("*")):
            if not path.is_file() or not decode.is_supported_image(path):
                continue
            image = decode.decode_image(path)
            passing = [
                detection
                for detection in active.detector.detect(image)
                if quality.evaluate(image, detection, settings).passed
            ]
            if not passing:
                continue
            detection = max(passing, key=lambda item: item.score)
            crop = align.align_crop(image, detection.landmarks)
            embedding = active.embedder.embed(crop)[0]
            samples.append(Sample(identity_dir.name, path, embedding))
    return samples


def evaluate_samples(samples: list[Sample], settings: Settings) -> dict[str, Any]:
    """Build deterministic verification and open-set populations from embedded samples."""
    by_label: dict[str, list[Sample]] = {}
    for sample in samples:
        by_label.setdefault(sample.label, []).append(sample)
    eligible = {label: rows for label, rows in by_label.items() if len(rows) >= 2}
    if len(eligible) < settings.eval_min_identities:
        raise ValueError(
            f"need at least {settings.eval_min_identities} identities with 2 passing images; "
            f"found {len(eligible)}"
        )

    labels = sorted(eligible)
    nonmated_count = max(1, len(labels) // 5)
    enrolled_labels = labels[:-nonmated_count]
    nonmated_labels = labels[-nonmated_count:]
    if not enrolled_labels:
        raise ValueError("calibration needs both enrolled and non-mated identities")

    gallery_vectors: list[np.ndarray] = []
    gallery_labels: list[str] = []
    for label in enrolled_labels:
        gallery_vectors.append(eligible[label][0].embedding)
        gallery_labels.append(label)
    gallery = np.stack(gallery_vectors).astype(np.float32)

    mated_scores: list[float] = []
    mated_correct: list[bool] = []
    genuine_gaps: list[float] = []
    for label in enrolled_labels:
        for sample in eligible[label][1:]:
            person_scores = sample.embedding @ gallery.T
            order = np.argsort(person_scores)[::-1]
            top = int(order[0])
            second_score = float(person_scores[order[1]]) if len(order) > 1 else 0.0
            mated_scores.append(float(person_scores[top]))
            correct = gallery_labels[top] == label
            mated_correct.append(correct)
            if correct:
                genuine_gaps.append(float(person_scores[top]) - second_score)

    nonmated_scores: list[float] = []
    impostor_gaps: list[float] = []
    for label in nonmated_labels:
        for sample in eligible[label]:
            person_scores = sample.embedding @ gallery.T
            order = np.argsort(person_scores)[::-1]
            top_score = float(person_scores[order[0]])
            second_score = float(person_scores[order[1]]) if len(order) > 1 else 0.0
            nonmated_scores.append(top_score)
            impostor_gaps.append(top_score - second_score)

    all_vectors = np.stack([sample.embedding for sample in samples]).astype(np.float32)
    all_labels = [sample.label for sample in samples]
    genuine, impostor = metrics.genuine_impostor_scores(all_vectors, all_labels)
    verification = {
        f"fnmr_at_fmr_{target:g}": {
            "threshold": threshold,
            "fnmr": fnmr,
        }
        for target in (1e-3, 1e-4)
        for threshold, fnmr in [metrics.fnmr_at_fmr(genuine, impostor, target)]
    }
    choice = metrics.choose_thresholds(
        mated_top1=mated_scores,
        mated_rank1_correct=mated_correct,
        nonmated_top1=nonmated_scores,
        gallery_size=len(enrolled_labels),
        fpir_target=settings.fpir_target,
        genuine_gaps=genuine_gaps,
        impostor_gaps=impostor_gaps,
        # A fixture set never has the ~1/target non-mated probes a 1e-3 FPIR needs, so
        # the policy falls back to the impostor-pair tail. Both populations are handed
        # over and the report records which route bound t_strong.
        impostor=impostor,
        # This gallery holds exactly one template per enrolled identity, so a probe
        # makes one comparison per person. The live gallery holds several per person;
        # the gate's gallery-size rule is what covers that drift (spec 10).
        comparisons_per_probe=len(enrolled_labels),
    )
    curve = metrics.fpir_fnir_curve(
        mated_scores,
        mated_correct,
        nonmated_scores,
        gallery_size=len(enrolled_labels),
    )
    return {
        "dataset": {
            "identities": len(eligible),
            "samples": len(samples),
            "enrolled_identities": len(enrolled_labels),
            "nonmated_identities": len(nonmated_labels),
        },
        "verification": verification,
        "verification_distributions": {
            "genuine": metrics.summarize_scores(genuine),
            "impostor": metrics.summarize_scores(impostor),
        },
        "open_set": curve.as_json(),
        "thresholds": choice.as_json(),
    }


def persist_report(
    report: dict[str, Any], settings: Settings, output: Path
) -> tuple[str, str]:
    """Write report bytes and an inactive calibrated threshold row, with one audit tx."""
    encoded = json.dumps(
        report, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(encoded + b"\n")
    digest = hashlib.sha256(encoded + b"\n").hexdigest()
    values = report["thresholds"]
    if not isinstance(values, dict):
        raise TypeError("threshold report is not an object")
    threshold_set_id = new_id()
    now = audit.now_ts()
    conn = connect(settings.db_path)
    try:
        with transaction(conn):
            conn.execute(
                "INSERT INTO threshold_sets (id, model_id, t_strong, t_possible, margin, "
                "calibrated, calibrated_at, eval_report_sha256, gallery_size, "
                "execution_provider, active, created_at) "
                "VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?, ?, 0, ?)",
                (
                    threshold_set_id,
                    settings.embedder_model,
                    float(values["t_strong"]),
                    float(values["t_possible"]),
                    float(values["margin"]),
                    now,
                    digest,
                    int(values["gallery_size"]),
                    settings.execution_provider,
                    now,
                ),
            )
            audit.append(
                conn,
                actor=settings.operator_name,
                action="threshold_set.create_calibrated",
                object_type="threshold_set",
                object_id=threshold_set_id,
                payload={
                    "model_id": settings.embedder_model,
                    "eval_report_sha256": digest,
                    "gallery_size": int(values["gallery_size"]),
                    "execution_provider": settings.execution_provider,
                    "active": False,
                },
            )
    finally:
        conn.close()
    return threshold_set_id, digest


def run(dataset: Path, settings: Settings, output: Path) -> dict[str, Any]:
    startup_checks(settings)
    lock = models_lock.verify(settings.models_dir)
    active = get_active_models(settings, lock)
    samples = collect_samples(dataset, settings)
    report = evaluate_samples(samples, settings)
    report.update(
        {
            "created_at": audit.now_ts(),
            "dataset_path": str(dataset.resolve()),
            "detector_model_id": active.detector_model_id,
            "embedder_model_id": active.embedder_model_id,
            "execution_provider": active.execution_provider,
        }
    )
    # Persist the actual provider used, not merely the configured preference.
    effective = settings.model_copy(update={"execution_provider": active.execution_provider})
    threshold_set_id, digest = persist_report(report, effective, output)
    return {
        "report": str(output),
        "sha256": digest,
        "threshold_set_id": threshold_set_id,
        "active": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="folder containing one subfolder per identity")
    parser.add_argument("--model", dest="embedder_model", help="models.lock embedder id")
    parser.add_argument("--output", type=Path, default=Path("eval/report.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = Settings()
    if args.embedder_model:
        settings = settings.model_copy(update={"embedder_model": args.embedder_model})
    result = run(args.dataset, settings, args.output)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

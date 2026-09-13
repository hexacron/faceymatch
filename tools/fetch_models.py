#!/usr/bin/env python3
"""Provision model weights into models/ and refresh models.lock.

This is BUILD-TIME tooling, run by an operator on purpose. It is deliberately a standalone
script outside the `app` package and imports nothing from it: invariant 1 forbids outbound
calls at runtime, and the way to keep that true is for the runtime package to contain no
download code at all.

Usage (from the repo root):

    python3 tools/fetch_models.py                        # permissive models only
    python3 tools/fetch_models.py --allow-noncommercial  # adds buffalo_l (C7)
    python3 tools/fetch_models.py --verify               # no downloads, check what is here

Every artifact is pinned by SHA-256. A digest mismatch is a hard failure: it means the
remote file changed under a stable URL, which is exactly the supply-chain event
invariant 8 exists to catch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"
LOCK_PATH = MODELS_DIR / "models.lock"
CHUNK = 1024 * 1024

ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models"

# InsightFace ships buffalo_l as a pack that also contains age/gender and dense-landmark
# models. Invariant 10 forbids attribute estimation outright, so only the recognition model
# is extracted and the rest of the archive is discarded, never written to models/.
BUFFALO_KEEP = "w600k_r50.onnx"
BUFFALO_DETECTOR = "det_10g.onnx"
# Age/gender and dense-landmark models are forbidden outright (invariant 10) and are never
# written to models/. det_10g is the SCRFD-10GF detector and is now extracted (D4).
BUFFALO_DISCARD = ("genderage.onnx", "1k3d68.onnx", "2d106det.onnx")
# antelopev2 ships its own SCRFD copy; the detector comes from buffalo_l, which is already
# fetched, so a build that only wants the detector needs one archive and not two.
ANTELOPE_DISCARD = ("genderage.onnx", "1k3d68.onnx", "2d106det.onnx", "scrfd_10g_bnkps.onnx")


@dataclass(frozen=True, slots=True)
class Source:
    id: str
    name: str
    version: str
    kind: str
    license: str
    license_url: str
    commercial_use: bool
    url: str
    filename: str
    sha256: str
    dim: int | None = None
    archive_member: str | None = None
    notes: str = ""
    discards: tuple[str, ...] = field(default_factory=tuple)


SOURCES: tuple[Source, ...] = (
    Source(
        id="yunet-2023mar",
        name="YuNet",
        version="2023mar",
        kind="detector",
        license="MIT",
        license_url=f"{ZOO}/face_detection_yunet/LICENSE",
        commercial_use=True,
        url=f"{ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        filename="face_detection_yunet_2023mar.onnx",
        sha256="8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
        notes="D4 detector. 5 landmarks, CPU-fast, permissive.",
    ),
    Source(
        id="sface-2021dec",
        name="SFace",
        version="2021dec",
        kind="embedder",
        dim=128,
        license="Apache-2.0",
        license_url=f"{ZOO}/face_recognition_sface/LICENSE",
        commercial_use=True,
        url=f"{ZOO}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        filename="face_recognition_sface_2021dec.onnx",
        sha256="0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
        notes="Shipped default embedder (D2 swap target). 128-d, L2-normalized.",
    ),
    Source(
        id="buffalo_l-w600k_r50",
        name="buffalo_l w600k_r50",
        version="v0.7",
        kind="embedder",
        dim=512,
        license="InsightFace model license: non-commercial research only",
        license_url="https://github.com/deepinsight/insightface/tree/master/model_zoo",
        commercial_use=False,
        url="https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
        filename="w600k_r50.onnx",
        archive_member=BUFFALO_KEEP,
        sha256="4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43",
        discards=BUFFALO_DISCARD,
        notes=(
            "Lab prototype only (D2, C7). Loading it needs allow_noncommercial_models, "
            "an audited operator decision. "
            "Attribute models in the pack are discarded, never stored (invariant 10)."
        ),
    ),
    Source(
        id="buffalo_l-det_10g",
        name="buffalo_l det_10g (SCRFD-10GF)",
        version="v0.7",
        kind="detector",
        license="InsightFace model license: non-commercial research only",
        license_url="https://github.com/deepinsight/insightface/tree/master/model_zoo",
        commercial_use=False,
        url="https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
        filename="det_10g.onnx",
        archive_member=BUFFALO_DETECTOR,
        sha256="5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91",
        discards=BUFFALO_DISCARD,
        notes=(
            "D4 SCRFD-10GF detector, 5 landmarks. Same pack and same licence as "
            "w600k_r50: loading needs allow_noncommercial_models (C7, invariant 9)."
        ),
    ),
    Source(
        id="antelopev2-glintr100",
        name="antelopev2 glintr100",
        version="v0.7",
        kind="embedder",
        dim=512,
        license="InsightFace model license: non-commercial research only",
        license_url="https://github.com/deepinsight/insightface/tree/master/model_zoo",
        commercial_use=False,
        url="https://github.com/deepinsight/insightface/releases/download/v0.7/antelopev2.zip",
        filename="glintr100.onnx",
        archive_member="glintr100.onnx",
        sha256="4ab1d6435d639628a6f3e5008dd4f929edf4c4124b1a7169e1048f9fef534cdf",
        discards=ANTELOPE_DISCARD,
        notes=(
            "ArcFace R100, 512-d. Same preprocessing as w600k_r50, so it runs on the same "
            "adapter. Non-commercial (C7); a switch re-embeds and needs recalibration."
        ),
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    print(f"  fetching {url}")
    with urllib.request.urlopen(url) as response, destination.open("wb") as out:  # noqa: S310
        shutil.copyfileobj(response, out)


def extract_member(
    archive: Path, member: str, destination: Path, discards: tuple[str, ...]
) -> None:
    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
        matches = [n for n in names if Path(n).name == member]
        if not matches:
            raise SystemExit(f"{member} not found in {archive.name}: {names}")
        dropped = sorted({Path(n).name for n in names if Path(n).name in discards})
        if dropped:
            print(f"  discarding (invariant 10 / not needed): {', '.join(dropped)}")
        with zf.open(matches[0]) as src, destination.open("wb") as out:
            shutil.copyfileobj(src, out)


def provision(
    source: Source, *, verify_only: bool, local: Path | None = None
) -> tuple[str, bool]:
    """Return (digest, present). Downloads unless the file is already correct.

    `local` supplies an already-obtained artifact (the buffalo_l pack is 326 MB and the
    release endpoint is unreliable); it goes through the same extraction and digest checks
    as a download, so a local copy is not a way to skip verification.
    """
    target = MODELS_DIR / source.filename
    if target.is_file():
        digest = sha256_file(target)
        if source.sha256 and digest != source.sha256:
            raise SystemExit(
                f"{source.filename}: on-disk digest {digest[:12]} does not match the pin "
                f"{source.sha256[:12]}. Delete it and re-run, or update the pin deliberately."
            )
        print(f"  present, sha256 {digest}")
        return digest, True
    if verify_only:
        print("  absent")
        return "", False

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp) / "artifact"
        if local is not None:
            print(f"  using local artifact {local}")
            shutil.copyfile(local, staged)
        else:
            download(source.url, staged)
        if source.archive_member is None:
            shutil.move(str(staged), target)
        else:
            extract_member(staged, source.archive_member, target, source.discards)

    digest = sha256_file(target)
    if source.sha256 and digest != source.sha256:
        target.unlink()
        raise SystemExit(
            f"{source.filename}: downloaded digest {digest} does not match the pin "
            f"{source.sha256}. Refusing to keep the file."
        )
    print(f"  stored {target.relative_to(REPO_ROOT)} sha256 {digest}")
    return digest, True


def write_lock(entries: list[dict[str, object]]) -> None:
    payload = {
        "version": 1,
        "note": (
            "Digests of the weight files in models/. Generated by tools/fetch_models.py. "
            "Startup verifies every present file against this lock and refuses on mismatch "
            "or on any unlisted .onnx (invariant 8). commercial_use=false requires "
            "allow_noncommercial_models, an audited operator setting (invariant 9, C7)."
        ),
        "models": entries,
    }
    LOCK_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {LOCK_PATH.relative_to(REPO_ROOT)} with {len(entries)} entries")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Provision model weights (build-time only)")
    parser.add_argument(
        "--allow-noncommercial",
        action="store_true",
        help="also fetch non-commercial weights (buffalo_l); they stay gated at runtime",
    )
    parser.add_argument(
        "--verify", action="store_true", help="check what is on disk; download nothing"
    )
    parser.add_argument(
        "--local",
        action="append",
        default=[],
        metavar="ID=PATH",
        help="use an already-downloaded artifact for a model id instead of fetching it; "
        "still extracted and digest-checked normally",
    )
    args = parser.parse_args(argv)

    locals_by_id: dict[str, Path] = {}
    for item in args.local:
        model_id, _, path = item.partition("=")
        if not path:
            parser.error(f"--local expects ID=PATH, got {item!r}")
        artifact = Path(path).expanduser().resolve()
        if not artifact.is_file():
            parser.error(f"--local artifact not found: {artifact}")
        locals_by_id[model_id] = artifact

    known_ids = {s.id for s in SOURCES}
    for model_id in locals_by_id:
        if model_id not in known_ids:
            parser.error(f"--local names unknown model {model_id!r}; known: {sorted(known_ids)}")

    entries: list[dict[str, object]] = []
    for source in SOURCES:
        if not source.commercial_use and not args.allow_noncommercial:
            print(f"{source.id}: skipped (non-commercial; pass --allow-noncommercial)")
            continue
        print(f"{source.id}:")
        digest, present = provision(
            source, verify_only=args.verify, local=locals_by_id.get(source.id)
        )
        if not present:
            continue
        entry: dict[str, object] = {
            "id": source.id,
            "name": source.name,
            "version": source.version,
            "kind": source.kind,
            "file": source.filename,
            "sha256": digest,
            "license": source.license,
            "commercial_use": source.commercial_use,
        }
        if source.dim is not None:
            entry["dim"] = source.dim
        entries.append(entry)

    if args.verify:
        print("\nverify only: models.lock not rewritten")
        return 0

    # Keep any already-listed model whose file is not being provisioned now, so a partial
    # run cannot silently drop an entry and make an existing file "unlisted" at startup.
    if LOCK_PATH.is_file():
        existing = json.loads(LOCK_PATH.read_text(encoding="utf-8")).get("models", [])
        known = {e["id"] for e in entries}
        for previous in existing:
            if previous["id"] not in known and (MODELS_DIR / previous["file"]).is_file():
                entries.append(previous)

    write_lock(entries)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Content-addressed object store for media originals and aligned crops (spec 6.1, 9).

Layout is `root/<sha256[0:2]>/<sha256[2:4]>/<sha256>`: two levels of 256-way sharding keep
any one directory small enough for a filesystem to list quickly, while the name itself is
the evidence that the bytes are what the database says they are.

Writes go to a temp file in the store and land with `os.replace`, so a crash can never
leave a truncated object under a name that claims its digest. Re-storing bytes that are
already present is a no-op that still returns the digest: that is how the same file
appearing in two cases shares one object on disk (spec 6.1).
"""

from __future__ import annotations

import hashlib
import io
import os
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

# 1 MiB: large enough that hashing a video is IO-bound, small enough to stay off the heap.
CHUNK_BYTES = 1 << 20

_TEMP_PREFIX = ".incoming-"


class ObjectNotFoundError(FileNotFoundError):
    """The store holds no object under the requested digest."""


def path_for(root: Path, sha256: str) -> Path:
    """Return the canonical path of an object, whether or not it exists yet."""
    if len(sha256) != 64 or not all(c in "0123456789abcdef" for c in sha256):
        raise ValueError(f"not a lowercase hex sha256: {sha256!r}")
    return root / sha256[:2] / sha256[2:4] / sha256


def relative_path_for(sha256: str) -> str:
    """Store-relative path recorded in `media.path`.

    Relative, not absolute, so a database moved with its store (or an export bundle) still
    resolves against whatever `media_dir` the reader is configured with.
    """
    return f"{sha256[:2]}/{sha256[2:4]}/{sha256}"


def resolve(root: Path, stored_path: str) -> Path:
    """Where `media.path` actually is. Store-relative normally; legacy imports are absolute.

    One reader for both the byte-serving path and the purge path: a file the API can serve
    and a file the purge cannot find would be an object nobody deletes.
    """
    path = Path(stored_path)
    return path if path.is_absolute() else root / path


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(src: Path) -> tuple[str, int]:
    """Hash a file without storing it. Returns (sha256, size_bytes)."""
    hasher = hashlib.sha256()
    size = 0
    with src.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            hasher.update(chunk)
            size += len(chunk)
    return hasher.hexdigest(), size


def store_bytes(root: Path, data: bytes) -> str:
    """Store bytes under their SHA-256. Returns the digest; existing objects are untouched."""
    digest = sha256_bytes(data)
    dest = path_for(root, digest)
    if dest.exists():
        return digest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = _new_temp(root)
    try:
        with tmp.open("wb") as out:
            out.write(data)
            out.flush()
            os.fsync(out.fileno())
        _land(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)
    return digest


def store_file(root: Path, src: Path) -> tuple[str, int]:
    """Copy a file into the store, hashing as it streams. Returns (sha256, size_bytes).

    One pass over the bytes: the digest is only known once the copy is complete, so the
    copy lands in a temp file and is renamed to its final name afterwards.
    """
    hasher = hashlib.sha256()
    size = 0
    tmp = _new_temp(root)
    try:
        with src.open("rb") as source, tmp.open("wb") as out:
            while chunk := source.read(CHUNK_BYTES):
                hasher.update(chunk)
                size += len(chunk)
                out.write(chunk)
            out.flush()
            os.fsync(out.fileno())
        digest = hasher.hexdigest()
        dest = path_for(root, digest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        _land(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)
    return digest, size


def exists(root: Path, sha256: str) -> bool:
    return path_for(root, sha256).exists()


def load_bytes(root: Path, sha256: str) -> bytes:
    path = path_for(root, sha256)
    try:
        return path.read_bytes()
    except FileNotFoundError as exc:
        raise ObjectNotFoundError(f"no object {sha256} under {root}") from exc


def store_crop(crops_dir: Path, crop: np.ndarray) -> str:
    """PNG-encode an aligned RGB crop and store it. Returns the PNG's SHA-256.

    The digest is of the PNG file, which is what `detections.crop_sha256` records, so the
    stored crop is byte-reproducible from that column alone.
    """
    return store_bytes(crops_dir, encode_png(crop))


def load_crop(crops_dir: Path, sha256: str) -> np.ndarray:
    """Load a stored crop as an (H, W, 3) uint8 RGB array."""
    with Image.open(io.BytesIO(load_bytes(crops_dir, sha256))) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def encode_png(image: np.ndarray) -> bytes:
    """Lossless PNG bytes for an (H, W, 3) uint8 RGB array."""
    if image.dtype != np.uint8:
        raise ValueError(f"crop must be uint8, got {image.dtype}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"crop must be (H, W, 3) RGB, got shape {image.shape}")
    buffer = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(image), mode="RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def _new_temp(root: Path) -> Path:
    """Create an empty temp file inside the store so `os.replace` stays on one filesystem."""
    root.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(dir=root, prefix=_TEMP_PREFIX)
    os.close(handle)
    return Path(name)


def _land(tmp: Path, dest: Path) -> None:
    """Atomically move `tmp` to `dest`, or drop it when `dest` already holds these bytes."""
    if dest.exists():
        return
    os.replace(tmp, dest)

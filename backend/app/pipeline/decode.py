"""Image decoding (spec 6.1, 6.2 step 1).

One convention for every pixel array in the pipeline: shape (H, W, 3), dtype uint8, RGB,
EXIF orientation already applied. Detection, quality and alignment all assume it, so the
transform happens exactly once, here.

The stored original is never rewritten. EXIF orientation is applied to the decoded array
only, so `media.sha256` keeps matching the bytes the operator handed us (spec 6.1).
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pillow_heif
from PIL import Image, ImageOps, UnidentifiedImageError

# HEIC/HEIF is what iPhones produce, so it is a first-class input, not an extra.
pillow_heif.register_heif_opener()  # type: ignore[attr-defined]

SUPPORTED_IMAGE_SUFFIXES: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}
)


class ImageDecodeError(ValueError):
    """An image could not be turned into a pixel array. Carries an operator-readable reason."""


class UnsupportedImageError(ImageDecodeError):
    """The file extension is not one of `SUPPORTED_IMAGE_SUFFIXES`."""


class CorruptImageError(ImageDecodeError):
    """The bytes are not a decodable image: truncated, empty, or not an image at all."""


def is_supported_image(name: str | Path) -> bool:
    return Path(name).suffix.lower() in SUPPORTED_IMAGE_SUFFIXES


def require_supported_image(name: str | Path) -> str:
    """Return the normalized suffix, or raise `UnsupportedImageError`."""
    suffix = Path(name).suffix.lower()
    if suffix not in SUPPORTED_IMAGE_SUFFIXES:
        raise UnsupportedImageError(
            f"unsupported image type {suffix or '(none)'!r}; "
            f"expected one of {', '.join(sorted(SUPPORTED_IMAGE_SUFFIXES))}"
        )
    return suffix


def decode_image(path: Path) -> np.ndarray:
    """Decode one image file to an (H, W, 3) uint8 RGB array with EXIF orientation applied."""
    return _decode(path, label=path.name)


def decode_bytes(data: bytes, *, label: str = "frame") -> np.ndarray:
    """Decode encoded image bytes with the same conventions as `decode_image`.

    The live match path receives a frame as bytes and never stores it, so it must not need
    a temp file to get pixels. Everything else — RGB, uint8, EXIF orientation, the shape
    check — is identical, because both paths feed the same detector.
    """
    if not data:
        raise CorruptImageError(f"{label}: empty image data")
    return _decode(io.BytesIO(data), label=label)


def _decode(source: Path | io.BytesIO, *, label: str) -> np.ndarray:
    try:
        with Image.open(source) as opened:
            oriented = ImageOps.exif_transpose(opened) or opened
            # `convert` forces the decode, so truncation surfaces here and not later as a
            # half-black frame handed to the detector.
            with oriented:
                rgb = oriented.convert("RGB")
            array = np.asarray(rgb, dtype=np.uint8)
    except (UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise CorruptImageError(f"{label}: not a decodable image ({exc})") from exc
    except OSError as exc:
        raise CorruptImageError(
            f"{label}: image data is truncated or unreadable ({exc})"
        ) from exc
    if array.ndim != 3 or array.shape[2] != 3 or array.shape[0] == 0 or array.shape[1] == 0:
        raise CorruptImageError(f"{label}: decoded to an unusable shape {array.shape}")
    return array


def probe_format(path: Path) -> str | None:
    """Pillow's format name for a stored object, or None when it is not an image.

    Content-addressed objects have no extension, so the served `Content-Type` comes from
    the bytes rather than from a filename we did not keep.
    """
    try:
        with Image.open(path) as opened:
            return opened.format
    except (UnidentifiedImageError, OSError):
        return None

"""Video decoding and sampling (spec 6.2 step 1).

Same pixel convention as `decode.py`: every frame leaves here as (H, W, 3) uint8 RGB, so
detection, quality and alignment cannot tell a video frame from a still.

Two decisions are load-bearing and the rest of the pipeline depends on both.

**Sampling is a fixed time grid, not "every Nth frame".** Targets sit at `k / sample_fps`
seconds and each sample is the first decoded frame at or after its target. A variable frame
rate — which is what a phone, a screen recording and most containers actually produce —
makes every frame-counting scheme drift against the clock, and the timestamps are what the
player overlay draws against.

**`frame_idx` is that grid's index `k`, not a decode ordinal.** `detections` is unique on
`(media_id, frame_idx, det_idx)`, which is what makes a resumed job idempotent (spec 6.2,
"Job resume"). A decode ordinal depends on where decoding started, so a job resumed after a
kill would renumber every remaining frame and insert a second copy of work already done. A
grid index is a function of the timestamp alone, so the same frame is the same slot whether
the job started at the beginning or seeked into the middle.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import av
import av.error
import numpy as np

# Containers PyAV decodes that an operator is likely to hand over. The suffix list exists
# because ingest has to decide `media.kind` before anything opens the file, and because
# probing every dropped file to find out whether it is video is a decode of untrusted bytes
# on the upload path. A container we do not list is refused at ingest, not half-processed.
SUPPORTED_VIDEO_SUFFIXES: frozenset[str] = frozenset(
    {".mp4", ".m4v", ".mov", ".avi", ".mkv", ".webm", ".mpg", ".mpeg", ".wmv"}
)


class VideoDecodeError(ValueError):
    """A video could not be opened or sampled. Carries an operator-readable reason."""


@dataclass(frozen=True, slots=True)
class VideoInfo:
    """What the container says about itself, for the `media` row and the player."""

    width: int
    height: int
    #: None when the container declares no duration (a stream, or a truncated file).
    duration_ms: int | None
    #: The container's average rate, recorded for display. Sampling ignores it.
    fps: float | None


@dataclass(frozen=True, slots=True)
class Sample:
    """One sampled frame: its grid slot, its true timestamp, and its pixels."""

    frame_idx: int
    t_ms: int
    image: np.ndarray


def is_supported_video(name: str | Path) -> bool:
    return Path(name).suffix.lower() in SUPPORTED_VIDEO_SUFFIXES


def probe(path: Path) -> VideoInfo:
    """Read dimensions, duration and average rate without decoding the whole file."""
    try:
        with av.open(str(path)) as container:
            if not container.streams.video:
                raise VideoDecodeError(f"{path.name}: no video stream")
            stream = container.streams.video[0]
            width = int(stream.codec_context.width)
            height = int(stream.codec_context.height)
            if width <= 0 or height <= 0:
                raise VideoDecodeError(f"{path.name}: video stream declares no frame size")
            rate = stream.average_rate
            duration = _duration_ms(container, stream)
    except av.error.FFmpegError as exc:
        raise VideoDecodeError(f"{path.name}: {exc}") from exc
    return VideoInfo(
        width=width,
        height=height,
        duration_ms=duration,
        fps=None if rate is None else float(rate),
    )


def iter_samples(path: Path, *, sample_fps: float, start_ms: int = 0) -> Iterator[Sample]:
    """Yield frames on the `sample_fps` grid, from `start_ms` onwards.

    `start_ms` is the resume point: the container is seeked to just before it and every
    frame earlier than it is dropped. Seeking lands on the nearest preceding keyframe, so
    the decoder still has the reference frames it needs to produce correct pixels — a seek
    straight to a P-frame yields the macroblock smear that would otherwise be embedded and
    matched as if it were a face.
    """
    if sample_fps <= 0.0:
        raise ValueError("sample_fps must be positive")
    step_ms = 1000.0 / sample_fps
    try:
        with av.open(str(path)) as container:
            if not container.streams.video:
                raise VideoDecodeError(f"{path.name}: no video stream")
            stream = container.streams.video[0]
            time_base = stream.time_base
            if time_base is None:
                raise VideoDecodeError(f"{path.name}: video stream declares no time base")
            # Frames are decoded in one thread per slice where the codec allows it; the
            # pipeline behind this is model-bound, so decode should not also be serial.
            stream.thread_type = "AUTO"
            if start_ms > 0:
                # `seek` measures in the stream's own time base when a stream is named,
                # not in microseconds: passing the wrong unit lands somewhere arbitrary,
                # usually the end of the file, and silently samples nothing.
                container.seek(
                    round(start_ms / 1000.0 / float(time_base)),
                    stream=stream,
                    backward=True,
                    any_frame=False,
                )
            # The first slot at or after the resume point, so a resumed job continues the
            # same grid rather than starting a new one from where it woke up.
            next_slot = int(start_ms // step_ms) if start_ms > 0 else 0
            for frame in container.decode(stream):
                # A frame with no presentation timestamp cannot be placed on the grid, and
                # a box drawn at a guessed time is worse than one frame not sampled.
                if frame.pts is None:
                    continue
                t_ms = round(float(frame.pts * time_base) * 1000.0)
                if t_ms < start_ms:
                    continue
                if t_ms + 0.5 < next_slot * step_ms:
                    continue
                yield Sample(
                    frame_idx=next_slot,
                    t_ms=t_ms,
                    image=frame.to_ndarray(format="rgb24"),
                )
                # Skip every slot this frame already satisfied, and never re-emit the one
                # it was taken for: at a sample rate above the file's own frame rate there
                # is no second frame to fill those slots, and emitting the same pixels
                # twice would double-count the face in them.
                next_slot = max(next_slot + 1, int(t_ms // step_ms) + 1)
    except av.error.FFmpegError as exc:
        raise VideoDecodeError(f"{path.name}: {exc}") from exc


def first_frame(path: Path) -> np.ndarray:
    """The first decodable frame, for a poster image. `VideoDecodeError` when there is none."""
    for sample in iter_samples(path, sample_fps=1.0):
        return sample.image
    raise VideoDecodeError(f"{path.name}: no decodable frames")


def _duration_ms(
    container: av.container.InputContainer, stream: av.video.VideoStream
) -> int | None:
    """Duration in milliseconds, preferring the stream's own clock over the container's."""
    if stream.duration is not None and stream.time_base is not None:
        return round(float(stream.duration * stream.time_base) * 1000.0)
    if container.duration is not None:
        # `container.duration` is in AV_TIME_BASE units, which is microseconds.
        return round(container.duration / 1000.0)
    return None

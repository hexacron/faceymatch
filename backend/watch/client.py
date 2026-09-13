"""The only thing in the helper that touches the network.

Standard library only: `urllib.request` plus a hand-rolled multipart encoder. No HTTP
dependency is added for four requests against loopback, and a stdlib client keeps invariant 1
auditable by reading one file. The base URL is checked against `LOOPBACK_HOSTS` at
construction, so the helper cannot be pointed at a remote instance even by environment
variable (invariants 1 and 11).
"""

from __future__ import annotations

import json
import secrets
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from watch.config import BASE_URL, LOOPBACK_HOSTS, REQUEST_TIMEOUT_S

JPEG_TYPE = "image/jpeg"


class BackendError(RuntimeError):
    """The backend refused, or is not running. Carries text fit for the panel."""


@dataclass(frozen=True, slots=True)
class Case:
    id: str
    name: str


@dataclass(frozen=True, slots=True)
class Candidate:
    person_id: str
    name: str
    rank: int
    score: float
    band: str
    best_template_id: str


@dataclass(frozen=True, slots=True)
class Face:
    x: float
    y: float
    w: float
    h: float
    det_score: float
    quality_passed: bool
    quality_reasons: tuple[str, ...]
    candidates: tuple[Candidate, ...]


@dataclass(frozen=True, slots=True)
class MatchResult:
    """`LiveMatchOut` from `backend/app/api/live.py`, as the helper sees it."""

    width: int
    height: int
    faces: tuple[Face, ...]
    identified: bool
    auto_accept_allowed: bool
    auto_accept_reason: str | None
    gallery_persons: int
    elapsed_ms: int
    timings: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class Health:
    version: str
    execution_provider: str
    gallery_calibrated: bool
    auto_accept_allowed: bool
    auto_accept_reason: str | None


class Client:
    """An HTTP client of the local backend, and nothing else."""

    def __init__(self, base_url: str = BASE_URL) -> None:
        parts = urlsplit(base_url)
        if parts.scheme != "http" or parts.hostname not in LOOPBACK_HOSTS:
            raise ValueError(
                f"the watch helper only talks to a loopback backend, not {base_url!r} "
                f"(one of {', '.join(sorted(LOOPBACK_HOSTS))} over http)"
            )
        self.base_url = base_url.rstrip("/")

    # ---------------------------------------------------------------- requests

    def cases(self) -> list[Case]:
        payload = self._get("/api/cases")
        items = _list(payload, "items")
        return [Case(id=_str(item, "id"), name=_str(item, "name")) for item in items]

    def health(self) -> Health:
        payload = self._get("/api/healthz")
        threshold_set = payload.get("threshold_set")
        auto_accept = _obj(payload, "auto_accept")
        return Health(
            version=_str(payload, "version"),
            execution_provider=_str(payload, "execution_provider"),
            gallery_calibrated=bool(
                isinstance(threshold_set, dict) and threshold_set.get("calibrated")
            ),
            auto_accept_allowed=bool(auto_accept.get("allowed")),
            auto_accept_reason=_opt_str(auto_accept, "reason"),
        )

    def match(self, jpeg: bytes, *, case_id: str, identify: bool) -> MatchResult:
        """One frame through tier 2. Nothing is persisted by this call (spec 6.10)."""
        fields = {"identify": "true" if identify else "false"}
        if case_id:
            fields["case_id"] = case_id
        payload = self._post_multipart(
            "/api/live/match", fields=fields, files={"frame": ("frame.jpg", jpeg)}
        )
        return _match_result(payload)

    def ingest(self, jpeg: bytes, *, case_id: str, filename: str, capture_mode: str) -> str:
        """Store the frame as evidence through tier 1 and return its media id."""
        payload = self._post_multipart(
            "/api/media",
            fields={
                "case_id": case_id,
                "acquisition": "screen_capture",
                "capture_mode": capture_mode,
            },
            files={"file": (filename, jpeg)},
        )
        return _str(payload, "media_id")

    # ------------------------------------------------------------------ wire

    # The URL is built from a base whose scheme and host `__init__` checked against
    # LOOPBACK_HOSTS, so every request below is http to loopback and nothing else: S310's
    # concern (a `file:` or custom scheme reaching urlopen) cannot arise.
    def _get(self, path: str) -> dict[str, Any]:
        request = urllib.request.Request(f"{self.base_url}{path}", method="GET")  # noqa: S310
        return self._send(request)

    def _post_multipart(
        self,
        path: str,
        *,
        fields: Mapping[str, str],
        files: Mapping[str, tuple[str, bytes]],
    ) -> dict[str, Any]:
        body, content_type = _encode_multipart(fields, files)
        request = urllib.request.Request(  # noqa: S310
            f"{self.base_url}{path}",
            data=body,
            headers={"Content-Type": content_type},
            method="POST",
        )
        return self._send(request)

    def _send(self, request: urllib.request.Request) -> dict[str, Any]:
        try:
            # The URL is built from a base checked against LOOPBACK_HOSTS in __init__, so
            # the scheme and host are known-good http loopback: S310 is satisfied there.
            with urllib.request.urlopen(  # noqa: S310
                request, timeout=REQUEST_TIMEOUT_S
            ) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            raise BackendError(_detail(exc)) from exc
        except urllib.error.URLError as exc:
            raise BackendError(
                f"the backend is not answering on {self.base_url}; "
                "start it with `uv run uvicorn app.main:app`"
            ) from exc
        except TimeoutError as exc:
            raise BackendError(
                f"the backend did not answer within {REQUEST_TIMEOUT_S:.0f}s"
            ) from exc
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise BackendError("the backend answered with something that is not JSON") from exc
        if not isinstance(parsed, dict):
            raise BackendError("the backend answered with an unexpected shape")
        return parsed


def _detail(exc: urllib.error.HTTPError) -> str:
    """FastAPI's own refusal, so the panel shows the reason and not a status code."""
    try:
        body = json.loads(exc.read())
    except (json.JSONDecodeError, OSError, ValueError):
        return f"the backend refused with HTTP {exc.code}"
    detail = body.get("detail") if isinstance(body, dict) else None
    if isinstance(detail, str) and detail:
        return detail
    return f"the backend refused with HTTP {exc.code}"


def _encode_multipart(
    fields: Mapping[str, str], files: Mapping[str, tuple[str, bytes]]
) -> tuple[bytes, str]:
    """`multipart/form-data`, built by hand so the helper needs no HTTP dependency."""
    boundary = f"----faceymatch{secrets.token_hex(16)}"
    marker = f"--{boundary}\r\n".encode()
    parts: list[bytes] = []
    for name, value in fields.items():
        parts.append(marker)
        parts.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        parts.append(value.encode())
        parts.append(b"\r\n")
    for name, (filename, blob) in files.items():
        parts.append(marker)
        parts.append(
            f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
            f"Content-Type: {JPEG_TYPE}\r\n\r\n".encode()
        )
        parts.append(blob)
        parts.append(b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


# ----------------------------------------------------------------- parsing
#
# Unknown keys are ignored — the response can grow without breaking the helper — but a
# missing required key is a BackendError rather than a KeyError three frames later.


def _match_result(payload: Mapping[str, Any]) -> MatchResult:
    timings = payload.get("timings")
    return MatchResult(
        width=_int(payload, "width"),
        height=_int(payload, "height"),
        faces=tuple(_face(face) for face in _list(payload, "faces")),
        identified=bool(payload.get("identified", False)),
        auto_accept_allowed=bool(payload.get("auto_accept_allowed", False)),
        auto_accept_reason=_opt_str(payload, "auto_accept_reason"),
        gallery_persons=_int(payload, "gallery_persons"),
        elapsed_ms=_int(payload, "elapsed_ms"),
        timings={k: float(v) for k, v in timings.items()} if isinstance(timings, dict) else {},
    )


def _face(payload: Mapping[str, Any]) -> Face:
    reasons = payload.get("quality_reasons")
    return Face(
        x=_float(payload, "x"),
        y=_float(payload, "y"),
        w=_float(payload, "w"),
        h=_float(payload, "h"),
        det_score=_float(payload, "det_score"),
        quality_passed=bool(payload.get("quality_passed", False)),
        quality_reasons=tuple(str(r) for r in reasons) if isinstance(reasons, list) else (),
        candidates=tuple(_candidate(c) for c in _list(payload, "candidates")),
    )


def _candidate(payload: Mapping[str, Any]) -> Candidate:
    return Candidate(
        person_id=_str(payload, "person_id"),
        name=_str(payload, "name"),
        rank=_int(payload, "rank"),
        score=_float(payload, "score"),
        band=_str(payload, "band"),
        best_template_id=_str(payload, "best_template_id"),
    )


def _missing(key: str) -> BackendError:
    return BackendError(f"the backend's answer is missing {key!r}; is this a faceymatch API?")


def _obj(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise _missing(key)
    return value


def _list(payload: Mapping[str, Any], key: str) -> Sequence[Mapping[str, Any]]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise _missing(key)
    return [item for item in value if isinstance(item, dict)]


def _str(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise _missing(key)
    return value


def _opt_str(payload: Mapping[str, Any], key: str) -> str | None:
    value = payload.get(key)
    return value if isinstance(value, str) else None


def _int(payload: Mapping[str, Any], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise _missing(key)
    return int(value)


def _float(payload: Mapping[str, Any], key: str) -> float:
    value = payload.get(key)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise _missing(key)
    return float(value)

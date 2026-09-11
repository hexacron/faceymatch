"""UUIDv7 (RFC 9562). Python 3.12 has no stdlib generator; 3.14 adds one.

Layout: 48-bit big-endian Unix milliseconds, 4-bit version, 12-bit counter, 2-bit variant,
62-bit randomness. The counter makes IDs strictly increasing within a millisecond, so
insertion order is preserved in the primary key index.
"""

from __future__ import annotations

import os
import threading
import time
from uuid import UUID

_LOCK = threading.Lock()
_last_ms = -1
_counter = 0

_MAX_COUNTER = 0xFFF


def uuid7() -> UUID:
    """Return a time-ordered UUIDv7."""
    global _last_ms, _counter

    with _LOCK:
        ms = time.time_ns() // 1_000_000
        if ms == _last_ms:
            _counter += 1
            if _counter > _MAX_COUNTER:
                # Exhausted the intra-millisecond counter: step into the next millisecond
                # rather than emit a non-monotonic value.
                ms += 1
                _last_ms = ms
                _counter = 0
        elif ms < _last_ms:
            # Clock moved backwards (NTP step). Keep monotonicity by holding the last value.
            ms = _last_ms
            _counter += 1
            if _counter > _MAX_COUNTER:
                ms += 1
                _last_ms = ms
                _counter = 0
        else:
            _last_ms = ms
            _counter = 0
        counter = _counter

    rand_b = int.from_bytes(os.urandom(8), "big") & ((1 << 62) - 1)
    value = (
        ((ms & 0xFFFF_FFFF_FFFF) << 80)
        | (0x7 << 76)
        | (counter << 64)
        | (0b10 << 62)
        | rand_b
    )
    return UUID(int=value)


def new_id() -> str:
    """Return a UUIDv7 as the canonical hyphenated string used in every table."""
    return str(uuid7())

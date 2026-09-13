"""Start the watch helper (spec 6.11).

    cd backend && uv run --extra watch python -m watch

The two things that can be wrong before a window is ever drawn — no capture backend for this
platform, and no backend answering on loopback — are checked first and reported as a
sentence, because a traceback from inside Qt tells the operator nothing they can act on. The
capture self-test runs on Start instead: it needs a target.
"""

from __future__ import annotations

import argparse
import sys

from watch.client import BackendError, Client
from watch.config import BASE_URL, DEFAULT_FPS
from watch.panel import Panel
from watch.sources import UnsupportedPlatformError, platform_backend

EXIT_UNAVAILABLE = 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="watch", description=__doc__)
    parser.add_argument("--url", default=BASE_URL, help=f"backend base URL (default {BASE_URL})")
    parser.add_argument(
        "--fps", type=float, default=DEFAULT_FPS, help=f"sample rate (default {DEFAULT_FPS:g})"
    )
    args = parser.parse_args(argv)

    try:
        backend = platform_backend()
    except UnsupportedPlatformError as exc:
        print(exc, file=sys.stderr)
        return EXIT_UNAVAILABLE

    try:
        client = Client(args.url)
        health = client.health()
    except (BackendError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return EXIT_UNAVAILABLE

    from PySide6.QtWidgets import QApplication

    app = QApplication([])
    panel = Panel(client, backend, fps=args.fps)
    panel.show()
    # `./run --watch` redirects this to data/logs/watch.log, where stdout is block-buffered:
    # the one line that proves the helper came up would not land until the process exited, and
    # a signal kill (Ctrl-C, run's cleanup) dropped it entirely. Flush it while it still means
    # something.
    print(
        f"faceymatch {health.version} on {health.execution_provider} at {client.base_url}",
        flush=True,
    )
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

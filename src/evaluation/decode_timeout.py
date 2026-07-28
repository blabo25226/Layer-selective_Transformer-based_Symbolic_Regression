"""Wall-clock guard for pathological single-problem symbolic decodes."""

from __future__ import annotations

import signal
import threading
from contextlib import contextmanager
from typing import Iterator


class DecodeTimeout(Exception):
    """Raised when one symbolic-regression decode exceeds its time budget."""


@contextmanager
def decode_time_limit(seconds: float) -> Iterator[None]:
    """Limit a decode on the POSIX main thread using ``SIGALRM``.

    Colab and the supported Linux GPU environment execute phase loops on the
    main thread. Unsupported platforms and worker threads run unguarded.
    """
    if (
        seconds <= 0
        or not hasattr(signal, "SIGALRM")
        or threading.current_thread() is not threading.main_thread()
    ):
        yield
        return

    def _handler(signum, frame):
        raise DecodeTimeout(f"decode exceeded {seconds:.0f}s")

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)

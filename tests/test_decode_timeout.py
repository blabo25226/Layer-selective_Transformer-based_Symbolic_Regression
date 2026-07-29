from __future__ import annotations

import signal
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation.decode_timeout import DecodeTimeout, decode_time_limit


def test_disabled_decode_timeout_is_noop() -> None:
    with decode_time_limit(0):
        pass


@pytest.mark.skipif(not hasattr(signal, "SIGALRM"), reason="SIGALRM requires POSIX")
def test_decode_timeout_interrupts_main_thread() -> None:
    with pytest.raises(DecodeTimeout, match="decode exceeded"):
        with decode_time_limit(0.01):
            time.sleep(0.1)

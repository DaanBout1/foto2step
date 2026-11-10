"""Tiny logger that writes to console and to an optional log file."""
from __future__ import annotations

import sys
import time

_fh = None
_t0 = time.time()


def set_logfile(path):
    global _fh
    _fh = open(path, "a", encoding="utf-8")


def log(msg: str):
    line = f"[{time.time() - _t0:7.1f}s] {msg}"
    print(line, flush=True, file=sys.stdout)
    if _fh:
        _fh.write(line + "\n")
        _fh.flush()

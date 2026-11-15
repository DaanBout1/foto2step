"""CPU budget: how many threads the heavy stages may use.

Laptops can overheat (or trip their power supply) when SIFT, matching and dense stereo run on every
core for hours. Fewer threads only costs time, never accuracy, so the default leaves half the cores
free. Must be applied before numpy/open3d are imported (OpenMP reads the env vars once)."""
from __future__ import annotations

import os

_N = None


def configure(cfg: dict) -> int:
    global _N
    cores = os.cpu_count() or 2
    want = int((cfg.get("compute") or {}).get("max_threads", 0) or 0)
    n = want if want > 0 else max(1, cores // 2)
    n = max(1, min(n, cores))
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[k] = str(n)
    _N = n
    return n


def n_threads() -> int:
    """Threads for external tools; -1 (= all cores, the tools' own default) if never configured."""
    return _N if _N else -1

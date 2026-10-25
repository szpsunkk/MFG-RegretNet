"""Exact, bounded memoization of Opacus' pure scalar RDP calculation.

This execution optimization preserves all floating-point inputs, the original
Opacus calculation on every cache miss, and the composition/search algorithms.
It deliberately does not round noise multipliers or approximate RDP values.
Forked calibration workers inherit the wrapper and own independent LRU caches.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from functools import lru_cache
from typing import Iterator

from opacus.accountants.analysis import rdp


_original_evaluate = None


def _evaluate_with_progress(candidate_loss: float):
    """Delegate one complete candidate and log only public calibration state."""
    if _original_evaluate is None:
        raise RuntimeError("calibration progress context is missing")
    started = time.monotonic()
    print(json.dumps({"event": "calibration_candidate_start", "pid": os.getpid(),
                      "candidate_loss": candidate_loss}), flush=True)
    result = _original_evaluate(candidate_loss)
    cache = rdp._compute_rdp.cache_info()
    print(json.dumps({"event": "calibration_candidate_complete", "pid": os.getpid(),
                      "candidate_loss": candidate_loss,
                      "elapsed_seconds": time.monotonic() - started,
                      "cache": cache._asdict()}), flush=True)
    return result


@contextmanager
def exact_rdp_cache(*, maxsize: int = 65536, calibration_module=None) -> Iterator[object]:
    """Cache identical (q, sigma, alpha) calls and always restore originals."""
    global _original_evaluate
    if maxsize <= 0:
        raise ValueError("RDP cache size must be positive")
    original_rdp = rdp._compute_rdp
    cached = lru_cache(maxsize=maxsize, typed=True)(original_rdp)
    original_evaluate = None
    previous_evaluate = _original_evaluate
    if calibration_module is not None:
        if previous_evaluate is not None:
            raise RuntimeError("nested calibration progress contexts are unsupported")
        original_evaluate = calibration_module._calibration_worker_evaluate
        _original_evaluate = original_evaluate
        calibration_module._calibration_worker_evaluate = _evaluate_with_progress
    rdp._compute_rdp = cached
    try:
        yield cached
    finally:
        rdp._compute_rdp = original_rdp
        if calibration_module is not None:
            calibration_module._calibration_worker_evaluate = original_evaluate
            _original_evaluate = previous_evaluate


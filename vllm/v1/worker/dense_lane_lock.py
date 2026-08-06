# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Launch-window mutex for the shared dense CUDA stream.

The gen model's prefill lane and the embed role launch onto ONE dense
stream from different executor threads. Interleaving their kernel LAUNCH
windows is unsafe (forward-context module state, FA3 TMA descriptor
staging) -- but full serialization of the work itself is not needed: the
stream's FIFO order provides launch-ahead once each window is enqueued
atomically. This mutex serializes only the launch windows.

Fairness: a plain Lock has no FIFO handoff, so the embed side could
re-acquire within while the prefill thread wakes. Prefill announces itself
around its acquire; the embed side defers while any prefill launcher waits,
bounding prefill's wait to one embed launch window.

Source protocol: tlebryk/vllm@3a3dec837 vllm/v1/engine/hb_embed_sidecar.py
lines 69-139. Stats read out via lock_stats() at drain end.
"""

from __future__ import annotations

import threading
import time

_LOCK = threading.Lock()
_PREFILL_WAITING = 0
_WAITING_GUARD = threading.Lock()

_STATS = {
    "prefill_wait_ms": 0.0,
    "prefill_waits": 0,
    "prefill_max_ms": 0.0,
    "embed_wait_ms": 0.0,
    "embed_waits": 0,
    "embed_max_ms": 0.0,
}
_STATS_GUARD = threading.Lock()

# Embed defer granularity while a prefill launcher waits.
_EMBED_DEFER_S = 0.0005


def _note_wait(side: str, waited_ms: float) -> None:
    if waited_ms < 1.0:
        return
    with _STATS_GUARD:
        _STATS[f"{side}_wait_ms"] += waited_ms
        _STATS[f"{side}_waits"] += 1
        if waited_ms > _STATS[f"{side}_max_ms"]:
            _STATS[f"{side}_max_ms"] = waited_ms


def prefill_launch_lock_acquire() -> None:
    global _PREFILL_WAITING
    with _WAITING_GUARD:
        _PREFILL_WAITING += 1
    start = time.perf_counter()
    _LOCK.acquire()
    with _WAITING_GUARD:
        _PREFILL_WAITING -= 1
    _note_wait("prefill", (time.perf_counter() - start) * 1e3)


def embed_launch_lock_acquire() -> None:
    start = time.perf_counter()
    while True:
        # Fairness: yield to any waiting prefill launcher first.
        while _PREFILL_WAITING > 0:
            time.sleep(_EMBED_DEFER_S)
        if _LOCK.acquire(timeout=_EMBED_DEFER_S):
            if _PREFILL_WAITING > 0:
                # A prefill arrived while we acquired; hand the lock over.
                _LOCK.release()
                time.sleep(_EMBED_DEFER_S)
                continue
            break
    _note_wait("embed", (time.perf_counter() - start) * 1e3)


def launch_lock_release() -> None:
    _LOCK.release()


def lock_stats() -> dict[str, float | int]:
    with _STATS_GUARD:
        return dict(_STATS)


def reset_lock_stats() -> None:
    with _STATS_GUARD:
        for key in _STATS:
            _STATS[key] = 0.0 if key.endswith("_ms") else 0

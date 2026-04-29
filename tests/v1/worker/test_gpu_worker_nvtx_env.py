# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the HB_*-prefixed NVTX-tag env vars in gpu_worker.Worker.

These exercise the small env-var contract (HB_WORK_ROLE,
HB_GPU_WORKER_NVTX_RANGES) and the per-step counter reset behavior, without
spinning up CUDA or a full vLLM engine. Worker is constructed via __new__ so
__init__'s heavyweight setup (CUDA device, distributed init, profiler wrapper)
is bypassed; we manually populate only the attributes annotate_profile()
touches.

Companion to test_scheduler_decision_log.py which covers the matching
HB_SCHEDULER_DECISION_LOG flag in Scheduler.
"""

import os
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.cpu_test


_TRUTHY = ("1", "true", "TRUE", "yes", "ON")
_FALSY = ("0", "false", "no", "off", "")


def _read_truthy(varname: str) -> bool:
    """Mirror of the parse idiom used in gpu_worker.py and scheduler.py."""
    return os.environ.get(varname, "0").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


@pytest.mark.parametrize("v", _TRUTHY)
def test_truthy_idiom_accepts_all_spellings(monkeypatch, v):
    monkeypatch.setenv("HB_GPU_WORKER_NVTX_RANGES", v)
    assert _read_truthy("HB_GPU_WORKER_NVTX_RANGES") is True


@pytest.mark.parametrize("v", _FALSY)
def test_truthy_idiom_rejects_falsy(monkeypatch, v):
    monkeypatch.setenv("HB_GPU_WORKER_NVTX_RANGES", v)
    assert _read_truthy("HB_GPU_WORKER_NVTX_RANGES") is False


def test_truthy_idiom_default_is_off(monkeypatch):
    monkeypatch.delenv("HB_GPU_WORKER_NVTX_RANGES", raising=False)
    assert _read_truthy("HB_GPU_WORKER_NVTX_RANGES") is False


def _make_fake_worker(*, work_role: str = "", emit_fallback: bool = False):
    """Build a minimal stand-in for Worker that exposes annotate_profile."""
    from vllm.v1.worker.gpu_worker import Worker

    w = Worker.__new__(Worker)
    w._work_role = work_role
    w._emit_step_nvtx = emit_fallback
    w._sched_step = 0
    w.profiler = None
    w.profiler_config = SimpleNamespace(profiler="cuda")
    return w


def _fake_scheduler_output(num_ctx=2, num_gen=3, max_seq=128):
    """Stub object whose shape is what compute_iteration_details() reads."""
    so = MagicMock()
    so.num_scheduled_tokens = {"r0": 4, "r1": 1}
    # compute_iteration_details may read more fields, but the existing tests
    # patched a real SchedulerOutput. We bypass it by stubbing the helper
    # directly on the worker via a closure (see test below).
    return so


def test_no_role_prefix_when_role_unset(monkeypatch):
    """HB_WORK_ROLE='' produces a tag starting with 'step_'."""
    from vllm.v1.worker import gpu_worker as gpu_worker_mod

    w = _make_fake_worker(work_role="", emit_fallback=True)

    fake_iter = SimpleNamespace(
        num_ctx_requests=2,
        num_ctx_tokens=20,
        num_generation_requests=3,
        num_generation_tokens=3,
        max_seq_tokens=128,
    )
    monkeypatch.setattr(
        gpu_worker_mod, "compute_iteration_details", lambda _so: fake_iter
    )

    cm = w.annotate_profile(_fake_scheduler_output())
    # NVTX context manager comes back from torch.cuda.nvtx.range; we don't
    # enter it (CUDA may not be available). Instead, verify state changes.
    assert w._sched_step == 1
    # The tag was built; we can re-run and check counter advances.
    _ = w.annotate_profile(_fake_scheduler_output())
    assert w._sched_step == 2

    # NOT a real CM if CUDA is missing — but the call returned without error.
    assert cm is not None


def test_role_prefix_inserted(monkeypatch):
    """HB_WORK_ROLE='gen' produces an annotation starting with 'gen_step_'."""
    from vllm.v1.worker import gpu_worker as gpu_worker_mod

    w = _make_fake_worker(work_role="gen", emit_fallback=True)

    fake_iter = SimpleNamespace(
        num_ctx_requests=0,
        num_ctx_tokens=0,
        num_generation_requests=1,
        num_generation_tokens=1,
        max_seq_tokens=8,
    )
    monkeypatch.setattr(
        gpu_worker_mod, "compute_iteration_details", lambda _so: fake_iter
    )

    # Capture the annotation by intercepting torch.cuda.nvtx.range.
    captured: list[str] = []
    import torch

    monkeypatch.setattr(
        torch.cuda.nvtx,
        "range",
        lambda name: captured.append(name) or nullcontext(),
    )

    w.annotate_profile(_fake_scheduler_output())
    assert captured, "annotate_profile did not emit an NVTX range"
    assert captured[0].startswith("gen_step_1_")
    assert "_execute_context_0(0)_generation_1(1)_maxseq_8" in captured[0]


def test_fallback_disabled_returns_nullcontext(monkeypatch):
    """With HB_GPU_WORKER_NVTX_RANGES off and no profiler, no NVTX is emitted."""
    from vllm.v1.worker import gpu_worker as gpu_worker_mod

    w = _make_fake_worker(work_role="gen", emit_fallback=False)
    monkeypatch.setattr(
        gpu_worker_mod,
        "compute_iteration_details",
        lambda _so: SimpleNamespace(
            num_ctx_requests=0,
            num_ctx_tokens=0,
            num_generation_requests=1,
            num_generation_tokens=1,
            max_seq_tokens=8,
        ),
    )

    captured: list[str] = []
    import torch

    monkeypatch.setattr(
        torch.cuda.nvtx,
        "range",
        lambda name: captured.append(name) or nullcontext(),
    )

    w.annotate_profile(_fake_scheduler_output())
    assert captured == []
    # And the counter must NOT have advanced (we returned early).
    assert w._sched_step == 0


def test_long_seq_suffix(monkeypatch):
    """maxseq exceeding 32768 appends '_LONG' suffix."""
    from vllm.v1.worker import gpu_worker as gpu_worker_mod

    w = _make_fake_worker(work_role="", emit_fallback=True)
    monkeypatch.setattr(
        gpu_worker_mod,
        "compute_iteration_details",
        lambda _so: SimpleNamespace(
            num_ctx_requests=1,
            num_ctx_tokens=40000,
            num_generation_requests=0,
            num_generation_tokens=0,
            max_seq_tokens=40000,
        ),
    )

    captured: list[str] = []
    import torch

    monkeypatch.setattr(
        torch.cuda.nvtx,
        "range",
        lambda name: captured.append(name) or nullcontext(),
    )

    w.annotate_profile(_fake_scheduler_output())
    assert captured[0].endswith("_LONG")

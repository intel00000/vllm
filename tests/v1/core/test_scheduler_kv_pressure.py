# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for the KV-pressure defensive gate.

The gate is a one-line predicate that the schedule() loop consults
before admitting fresh waiting-queue requests. We test the predicate
in isolation; the surrounding loop wiring is exercised by the
post-port end-to-end smoke.
"""

import pytest

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.worker.dual_model_helpers import DualModelConfig


# --------------------------------------------------------------------------
# Disabled paths -- gate must never fire
# --------------------------------------------------------------------------


def test_single_model_never_defers():
    """When dual_cfg is None the gate is a no-op regardless of usage."""
    for usage in (0.0, 0.5, 0.9, 0.99, 1.0):
        assert (
            Scheduler._kv_pressure_should_defer(dual_cfg=None, kv_usage=usage)
            is False
        )


def test_dual_cfg_without_threshold_never_defers():
    """When kv_pressure_skip_threshold is None, gate is off even at 100% usage."""
    cfg = DualModelConfig(embed_model="x")  # threshold defaults to None
    for usage in (0.0, 0.5, 0.9, 0.99, 1.0):
        assert (
            Scheduler._kv_pressure_should_defer(dual_cfg=cfg, kv_usage=usage)
            is False
        )


# --------------------------------------------------------------------------
# Normal cases -- gate fires at/above threshold, passes below
# --------------------------------------------------------------------------


def test_below_threshold_admits():
    cfg = DualModelConfig(embed_model="x", kv_pressure_skip_threshold=0.9)
    assert Scheduler._kv_pressure_should_defer(cfg, kv_usage=0.0) is False
    assert Scheduler._kv_pressure_should_defer(cfg, kv_usage=0.5) is False
    assert Scheduler._kv_pressure_should_defer(cfg, kv_usage=0.89) is False


def test_at_threshold_defers():
    """Boundary: usage exactly equal to threshold should defer (>=)."""
    cfg = DualModelConfig(embed_model="x", kv_pressure_skip_threshold=0.9)
    assert Scheduler._kv_pressure_should_defer(cfg, kv_usage=0.9) is True


def test_above_threshold_defers():
    cfg = DualModelConfig(embed_model="x", kv_pressure_skip_threshold=0.9)
    assert Scheduler._kv_pressure_should_defer(cfg, kv_usage=0.95) is True
    assert Scheduler._kv_pressure_should_defer(cfg, kv_usage=1.0) is True


# --------------------------------------------------------------------------
# Edge cases
# --------------------------------------------------------------------------


def test_threshold_zero_always_defers():
    """threshold=0.0 means "any usage at all blocks new admissions" --
    pathological config, but the predicate should still behave consistently."""
    cfg = DualModelConfig(embed_model="x", kv_pressure_skip_threshold=0.0)
    assert Scheduler._kv_pressure_should_defer(cfg, kv_usage=0.0) is True
    assert Scheduler._kv_pressure_should_defer(cfg, kv_usage=0.5) is True


def test_threshold_one_only_defers_at_full():
    """threshold=1.0 means "only defer when KV is literally full"."""
    cfg = DualModelConfig(embed_model="x", kv_pressure_skip_threshold=1.0)
    assert Scheduler._kv_pressure_should_defer(cfg, kv_usage=0.5) is False
    assert Scheduler._kv_pressure_should_defer(cfg, kv_usage=0.999) is False
    assert Scheduler._kv_pressure_should_defer(cfg, kv_usage=1.0) is True


def test_threshold_uses_ge_not_gt():
    """Regression: this is `>=` not `>`. Catches a future refactor that
    might accidentally tighten the comparison."""
    cfg = DualModelConfig(embed_model="x", kv_pressure_skip_threshold=0.75)
    # Just barely below should pass.
    assert Scheduler._kv_pressure_should_defer(cfg, kv_usage=0.74) is False
    # Exact match should defer (>= semantics).
    assert Scheduler._kv_pressure_should_defer(cfg, kv_usage=0.75) is True


@pytest.mark.parametrize(
    "threshold,usage,expected",
    [
        # Tight band around threshold=0.85
        (0.85, 0.84, False),
        (0.85, 0.849999, False),
        (0.85, 0.85, True),
        (0.85, 0.85001, True),
        # Different thresholds with the same usage
        (0.5, 0.7, True),
        (0.6, 0.7, True),
        (0.7, 0.7, True),
        (0.8, 0.7, False),
        (0.9, 0.7, False),
    ],
)
def test_threshold_parametric_sweep(threshold, usage, expected):
    cfg = DualModelConfig(embed_model="x", kv_pressure_skip_threshold=threshold)
    assert Scheduler._kv_pressure_should_defer(cfg, usage) is expected


def test_gate_is_independent_of_other_dual_cfg_knobs():
    """Other gate knobs being set should not affect the KV pressure
    predicate -- it only looks at threshold and usage."""
    cfg = DualModelConfig(
        embed_model="x",
        kv_pressure_skip_threshold=0.9,
        decode_running_reserve=16,
        max_embed_running_reqs=4,
        enforce_pair_dependency=True,
        enforce_no_double_prefill=True,
    )
    assert Scheduler._kv_pressure_should_defer(cfg, kv_usage=0.5) is False
    assert Scheduler._kv_pressure_should_defer(cfg, kv_usage=0.95) is True

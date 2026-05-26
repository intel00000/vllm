# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for the embed-waiting admission gate.

Covers DRR (decode_running_reserve), MER (max_embed_running_reqs), and
the phase-aware "release after decode waiting drains" knob. Bypasses
Scheduler.__init__ -- the gates only consult the four counters and the
two phase flags, all maintained by the stub.
"""

from types import SimpleNamespace

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.worker.dual_model_helpers import DualModelConfig


def _stub_scheduler(dual_cfg: DualModelConfig | None) -> Scheduler:
    sched = Scheduler.__new__(Scheduler)
    sched.dual_model_config = dual_cfg
    sched._num_running_decode_reqs = 0
    sched._num_running_embed_reqs = 0
    sched._num_waiting_decode_reqs = 0
    sched._num_waiting_embed_reqs = 0
    sched._embed_wait_drained_phase_started = False
    sched._embed_wait_drained_phase_released = False
    return sched


def _req(model_id: str) -> SimpleNamespace:
    return SimpleNamespace(model_id=model_id, request_id=f"r-{model_id}")


# --------------------------------------------------------------------------
# _embed_waiting_gate_enabled
# --------------------------------------------------------------------------


def test_gate_enabled_false_when_no_knobs():
    cfg = DualModelConfig(embed_model="x")
    assert Scheduler._embed_waiting_gate_enabled(cfg) is False


def test_gate_enabled_true_for_each_knob():
    for kw in [
        dict(max_embed_running_reqs=4),
        dict(embed_release_when_decode_waiting_drained=True),
        dict(embed_release_running_decode_threshold=10),
        dict(decode_running_reserve=16),
    ]:
        cfg = DualModelConfig(embed_model="x", **kw)
        assert Scheduler._embed_waiting_gate_enabled(cfg) is True, kw


# --------------------------------------------------------------------------
# _should_skip_embed_waiting_request
# --------------------------------------------------------------------------


def test_skip_returns_false_for_decode_requests():
    cfg = DualModelConfig(
        embed_model="x", max_embed_running_reqs=0, decode_running_reserve=16
    )
    sched = _stub_scheduler(cfg)
    sched._num_running_embed_reqs = 99
    # Gate ignores decode reqs entirely.
    assert sched._should_skip_embed_waiting_request(_req("decode")) is False


def test_skip_returns_false_when_single_model():
    sched = _stub_scheduler(dual_cfg=None)
    assert sched._should_skip_embed_waiting_request(_req("embed")) is False


def test_skip_returns_false_when_no_knobs():
    cfg = DualModelConfig(embed_model="x")  # gate disabled
    sched = _stub_scheduler(cfg)
    assert sched._should_skip_embed_waiting_request(_req("embed")) is False


def test_mer_caps_running_embed():
    cfg = DualModelConfig(embed_model="x", max_embed_running_reqs=4)
    sched = _stub_scheduler(cfg)
    sched._num_running_embed_reqs = 4
    assert sched._should_skip_embed_waiting_request(_req("embed")) is True


def test_mer_passes_when_below_cap():
    cfg = DualModelConfig(embed_model="x", max_embed_running_reqs=4)
    sched = _stub_scheduler(cfg)
    sched._num_running_embed_reqs = 3
    assert sched._should_skip_embed_waiting_request(_req("embed")) is False


def test_drr_holds_back_embed_when_decode_underprovisioned():
    cfg = DualModelConfig(embed_model="x", decode_running_reserve=16)
    sched = _stub_scheduler(cfg)
    sched._num_running_decode_reqs = 8  # below reserve
    sched._num_waiting_decode_reqs = 5  # decode still queued
    assert sched._should_skip_embed_waiting_request(_req("embed")) is True


def test_drr_releases_embed_when_decode_meets_reserve():
    cfg = DualModelConfig(embed_model="x", decode_running_reserve=16)
    sched = _stub_scheduler(cfg)
    sched._num_running_decode_reqs = 16
    sched._num_waiting_decode_reqs = 5
    assert sched._should_skip_embed_waiting_request(_req("embed")) is False


def test_drr_releases_embed_when_no_decode_in_waiting():
    cfg = DualModelConfig(embed_model="x", decode_running_reserve=16)
    sched = _stub_scheduler(cfg)
    sched._num_running_decode_reqs = 4
    sched._num_waiting_decode_reqs = 0  # no decode waiters
    # No decode queued -> no need to reserve, embed can run.
    assert sched._should_skip_embed_waiting_request(_req("embed")) is False


def test_release_threshold_blocks_when_decode_load_high():
    cfg = DualModelConfig(
        embed_model="x", embed_release_running_decode_threshold=100
    )
    sched = _stub_scheduler(cfg)
    sched._num_running_decode_reqs = 128  # above threshold
    assert sched._should_skip_embed_waiting_request(_req("embed")) is True


def test_release_threshold_admits_when_decode_load_drops():
    cfg = DualModelConfig(
        embed_model="x", embed_release_running_decode_threshold=100
    )
    sched = _stub_scheduler(cfg)
    sched._num_running_decode_reqs = 50
    assert sched._should_skip_embed_waiting_request(_req("embed")) is False


def test_drained_phase_blocks_then_releases():
    cfg = DualModelConfig(
        embed_model="x", embed_release_when_decode_waiting_drained=True
    )
    sched = _stub_scheduler(cfg)
    # Phase 1: decode is queued -> block + start phase.
    sched._num_waiting_decode_reqs = 3
    sched._num_running_decode_reqs = 5
    assert sched._should_skip_embed_waiting_request(_req("embed")) is True
    assert sched._embed_wait_drained_phase_started is True

    # Phase 2: decode drained, phase_started -> first call releases.
    sched._num_waiting_decode_reqs = 0
    assert sched._should_skip_embed_waiting_request(_req("embed")) is False
    assert sched._embed_wait_drained_phase_released is True


# --------------------------------------------------------------------------
# _is_embed_gate_blocked_without_state_change
# --------------------------------------------------------------------------


def test_peek_variant_does_not_mutate_phase_flags():
    cfg = DualModelConfig(
        embed_model="x", embed_release_when_decode_waiting_drained=True
    )
    sched = _stub_scheduler(cfg)
    sched._num_waiting_decode_reqs = 3
    sched._num_running_decode_reqs = 5
    # peek path returns blocked but does NOT flip phase_started
    assert (
        sched._is_embed_gate_blocked_without_state_change(_req("embed")) is True
    )
    assert sched._embed_wait_drained_phase_started is False


def test_peek_variant_agrees_with_skip_variant_on_steady_state():
    """When phase flags don't matter, peek and skip should agree."""
    cfg = DualModelConfig(embed_model="x", max_embed_running_reqs=4)
    sched = _stub_scheduler(cfg)
    sched._num_running_embed_reqs = 4
    assert (
        sched._is_embed_gate_blocked_without_state_change(_req("embed")) is True
    )
    assert sched._should_skip_embed_waiting_request(_req("embed")) is True

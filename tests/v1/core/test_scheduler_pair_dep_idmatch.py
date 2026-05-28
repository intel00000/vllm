# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Regression test: pair-dep gate's parent lookup keys on internal
request_id, not external_req_id.

The bug: vllm v0.21.0's InputProcessor.assign_request_id randomizes
request.request_id (appends 8 hex chars), and the scheduler keys
self.requests by that randomized id. A caller-supplied
parent_request_id MUST be the parent's *internal* id (the value
LLMEngine.add_request returned for the parent's add_request call).
If the caller passes the parent's external id, the gate's
self.requests.get(parent_id) returns None, the gate fail-opens, and
pair-dep silently becomes a no-op.

The symptom we saw: in scripts/profile_dual_model.py's paired path,
parent_request_id was set to f"embed-{i}" (the external id passed to
add_request). Every decode child fell through the pair-dep gate and
went straight into self.waiting. The NVTX trace showed dctx_3+ in
step_2 even though only 1 embed had been admitted in step_1, because
all decodes were freely admitted alongside the embeds.

These tests pin down the contract: pair-dep keys on internal
request_id.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import RequestStatus
from vllm.v1.worker.dual_model_helpers import DualModelConfig


def _stub_scheduler(pair_dep: bool = True) -> Scheduler:
    """Hand-assembled Scheduler with only the fields _should_gate_child_on_parent
    consults (dual_model_config, requests). Bypass __init__ since it
    needs a full kv_cache_config / structured_output_manager."""
    sched = Scheduler.__new__(Scheduler)
    sched.dual_model_config = DualModelConfig(
        embed_model="embed-model",
        enforce_pair_dependency=pair_dep,
    )
    sched.requests = {}
    return sched


def _fake_request(internal_id: str, finished: bool = False,
                  parent_request_id: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        request_id=internal_id,
        external_req_id=internal_id.rsplit("-", 1)[0] if "-" in internal_id else internal_id,
        parent_request_id=parent_request_id,
        status=(
            RequestStatus.FINISHED_STOPPED if finished
            else RequestStatus.WAITING
        ),
        is_finished=lambda f=finished: f,
    )


def test_gate_fires_when_child_points_at_parent_internal_id():
    """The good case: caller passed the parent's internal id."""
    sched = _stub_scheduler(pair_dep=True)
    parent = _fake_request("embed-0-abc12345", finished=False)
    sched.requests[parent.request_id] = parent
    child = _fake_request("decode-0-def67890",
                          parent_request_id=parent.request_id)
    assert sched._should_gate_child_on_parent(child) is True


def test_gate_fails_open_when_child_points_at_parent_external_id():
    """The bug we hit: caller passed the parent's external id. The
    gate's self.requests.get() returns None and the gate fails open."""
    sched = _stub_scheduler(pair_dep=True)
    parent = _fake_request("embed-0-abc12345", finished=False)
    sched.requests[parent.request_id] = parent
    child = _fake_request(
        "decode-0-def67890",
        parent_request_id="embed-0",  # ← external id; bug pattern
    )
    # The gate fails open silently (correct behavior — pair-dep is
    # advisory; it can't crash on unknown parents). The warning_once
    # logged from _should_gate_child_on_parent is what flags this in
    # practice.
    with patch("vllm.v1.core.sched.scheduler.logger.warning_once") as w:
        result = sched._should_gate_child_on_parent(child)
    assert result is False
    # The defensive log must fire so future callers notice.
    assert w.called, (
        "Pair-dep fail-open must warn so callers notice they passed "
        "the wrong id (external instead of internal)."
    )


def test_gate_off_short_circuits_no_warning():
    """When pair-dep is disabled, an unknown parent_id is fine and
    should NOT warn (the gate short-circuits before the lookup)."""
    sched = _stub_scheduler(pair_dep=False)
    child = _fake_request("decode-0-xx",
                          parent_request_id="embed-0")
    with patch("vllm.v1.core.sched.scheduler.logger.warning_once") as w:
        result = sched._should_gate_child_on_parent(child)
    assert result is False
    assert not w.called


def test_no_parent_request_id_short_circuits_no_warning():
    """Single-model requests have no parent_request_id; no warn."""
    sched = _stub_scheduler(pair_dep=True)
    child = _fake_request("gen-0-xx", parent_request_id=None)
    with patch("vllm.v1.core.sched.scheduler.logger.warning_once") as w:
        result = sched._should_gate_child_on_parent(child)
    assert result is False
    assert not w.called


def test_finished_parent_treats_as_satisfied():
    """If parent is in self.requests but already finished, the gate
    returns False (don't hold the child — parent's work is done)."""
    sched = _stub_scheduler(pair_dep=True)
    parent = _fake_request("embed-0-abc12345", finished=True)
    sched.requests[parent.request_id] = parent
    child = _fake_request("decode-0-def67890",
                          parent_request_id=parent.request_id)
    assert sched._should_gate_child_on_parent(child) is False

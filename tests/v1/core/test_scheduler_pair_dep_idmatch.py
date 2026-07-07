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


def test_finish_requests_gated_child_no_counter_underflow():
    """Regression: aborting a request that is still in _gated_children must
    NOT decrement _num_waiting_decode_reqs.

    Gated children are held in _gated_children (outside self.waiting) and
    were never counted via _mark_waiting_added_by_model. Before the fix,
    finish_requests routed them through _mark_waiting_removed_by_model,
    underflowing the counter (`assert _num_waiting_decode_reqs > 0` fired),
    crashing engine-core shutdown at scale when gated children remained.
    """
    from collections import defaultdict

    sched = _stub_scheduler(pair_dep=True)
    sched._num_waiting_decode_reqs = 0  # gated child was never counted
    sched._num_waiting_embed_reqs = 0
    sched._num_running_decode_reqs = 0
    sched._num_running_embed_reqs = 0
    sched._gated_children = {}
    sched._children_of_parent = defaultdict(set)
    sched.num_waiting_for_streaming_input = 0
    sched.finished_req_ids = set()
    sched.finished_req_ids_dict = None

    child = _fake_request("decode-0-def67890",
                          parent_request_id="embed-0-abc12345")
    child.model_id = "decode"       # -> _dual_model_request_kind == "decode"
    child.client_index = 0
    sched.requests = {child.request_id: child}
    sched._gated_children[child.request_id] = child
    sched._children_of_parent["embed-0-abc12345"].add(child.request_id)

    # _free_request is heavy (block manager, connectors) -> stub it; we only
    # assert the gate/counter bookkeeping.
    with patch.object(Scheduler, "_free_request", lambda self, r, **k: None):
        aborted = sched.finish_requests(None, RequestStatus.FINISHED_ABORTED)

    # No underflow, and the child is cleaned out of the gate structures.
    assert sched._num_waiting_decode_reqs == 0
    assert child.request_id not in sched._gated_children
    assert child.request_id not in sched._children_of_parent.get(
        "embed-0-abc12345", set()
    )
    assert (child.request_id, 0) in aborted

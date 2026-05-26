# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for RAG pair-dependency gating in Scheduler.

A child request with parent_request_id is held in _gated_children
until the parent finishes. Tests bypass Scheduler.__init__ so they can
exercise _should_gate_child_on_parent / _release_gated_children on a
hand-assembled stub.
"""

from collections import defaultdict
from types import SimpleNamespace

import vllm.v1.core.sched.scheduler as sched_module
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import RequestStatus
from vllm.v1.worker.dual_model_helpers import DualModelConfig


class _FakeRequest:
    """Minimal duck-type for Request that satisfies the gate / release
    code paths. Real Request needs SamplingParams, sometimes a tokenizer,
    etc. -- way too heavy for a unit test of a 2-method gate."""

    def __init__(
        self,
        request_id: str,
        *,
        parent_request_id: str | None = None,
        model_id: str = "decode",
        status: RequestStatus = RequestStatus.WAITING,
    ) -> None:
        self.request_id = request_id
        self.parent_request_id = parent_request_id
        self.model_id = model_id
        self.status = status

    def is_finished(self) -> bool:
        return RequestStatus.is_finished(self.status)


def _stub_scheduler(dual_cfg: DualModelConfig | None) -> Scheduler:
    sched = Scheduler.__new__(Scheduler)
    sched.dual_model_config = dual_cfg
    sched.requests = {}
    sched._gated_children = {}
    sched._children_of_parent = defaultdict(set)
    sched._decision_log_enabled = False
    sched._decisions = []
    sched._num_running_decode_reqs = 0
    sched._num_running_embed_reqs = 0
    sched._num_waiting_decode_reqs = 0
    sched._num_waiting_embed_reqs = 0
    sched.waiting = []
    sched.skipped_waiting = []
    return sched


def test_gate_returns_false_when_no_parent_id():
    cfg = DualModelConfig(embed_model="x", enforce_pair_dependency=True)
    sched = _stub_scheduler(cfg)
    assert not sched._should_gate_child_on_parent(_FakeRequest("c1"))


def test_gate_returns_false_when_pair_dep_disabled():
    cfg = DualModelConfig(embed_model="x", enforce_pair_dependency=False)
    sched = _stub_scheduler(cfg)
    parent = _FakeRequest("p1")
    sched.requests["p1"] = parent
    child = _FakeRequest("c1", parent_request_id="p1")
    assert not sched._should_gate_child_on_parent(child)


def test_gate_returns_false_when_single_model():
    sched = _stub_scheduler(dual_cfg=None)
    parent = _FakeRequest("p1")
    sched.requests["p1"] = parent
    child = _FakeRequest("c1", parent_request_id="p1")
    assert not sched._should_gate_child_on_parent(child)


def test_gate_returns_false_when_parent_unknown():
    cfg = DualModelConfig(embed_model="x", enforce_pair_dependency=True)
    sched = _stub_scheduler(cfg)
    child = _FakeRequest("c1", parent_request_id="unknown-p")
    assert not sched._should_gate_child_on_parent(child)


def test_gate_returns_false_when_parent_already_finished():
    cfg = DualModelConfig(embed_model="x", enforce_pair_dependency=True)
    sched = _stub_scheduler(cfg)
    parent = _FakeRequest("p1", status=RequestStatus.FINISHED_STOPPED)
    sched.requests["p1"] = parent
    child = _FakeRequest("c1", parent_request_id="p1")
    assert not sched._should_gate_child_on_parent(child)


def test_gate_returns_true_when_parent_in_flight():
    cfg = DualModelConfig(embed_model="x", enforce_pair_dependency=True)
    sched = _stub_scheduler(cfg)
    parent = _FakeRequest("p1", status=RequestStatus.WAITING)
    sched.requests["p1"] = parent
    child = _FakeRequest("c1", parent_request_id="p1")
    assert sched._should_gate_child_on_parent(child)


def test_release_normal_finish_enqueues_children(monkeypatch):
    cfg = DualModelConfig(embed_model="x", enforce_pair_dependency=True)
    sched = _stub_scheduler(cfg)
    enqueued: list[_FakeRequest] = []
    monkeypatch.setattr(
        sched, "_enqueue_waiting_request", lambda req: enqueued.append(req)
    )
    c1 = _FakeRequest("c1", parent_request_id="p1")
    c2 = _FakeRequest("c2", parent_request_id="p1")
    sched._gated_children["c1"] = c1
    sched._gated_children["c2"] = c2
    sched._children_of_parent["p1"] = {"c1", "c2"}

    sched._release_gated_children("p1", RequestStatus.FINISHED_STOPPED)

    assert {r.request_id for r in enqueued} == {"c1", "c2"}
    assert sched._gated_children == {}
    assert "p1" not in sched._children_of_parent


def test_release_aborted_parent_cascades_abort(monkeypatch):
    cfg = DualModelConfig(embed_model="x", enforce_pair_dependency=True)
    sched = _stub_scheduler(cfg)
    freed: list[_FakeRequest] = []
    monkeypatch.setattr(sched, "_free_request", lambda req: freed.append(req))
    enqueued: list[_FakeRequest] = []
    monkeypatch.setattr(
        sched, "_enqueue_waiting_request", lambda req: enqueued.append(req)
    )
    c1 = _FakeRequest("c1", parent_request_id="p1")
    sched._gated_children["c1"] = c1
    sched._children_of_parent["p1"] = {"c1"}

    sched._release_gated_children("p1", RequestStatus.FINISHED_ABORTED)

    assert freed == [c1]
    assert c1.status == RequestStatus.FINISHED_ABORTED
    assert enqueued == []
    assert sched._gated_children == {}


def test_release_with_no_children_is_noop():
    cfg = DualModelConfig(embed_model="x", enforce_pair_dependency=True)
    sched = _stub_scheduler(cfg)
    # No exception, no side effects.
    sched._release_gated_children("nonexistent-parent", RequestStatus.FINISHED_STOPPED)
    assert sched._gated_children == {}
    assert sched._children_of_parent == {}

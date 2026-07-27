# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for RAG pair-dependency gate and no-double-prefill admission gate.

Both behaviors are gated behind explicit flags on `DualModelConfig`:
  - enforce_pair_dependency  -> Patch B (dependency state machine)
  - enforce_no_double_prefill -> Patch C (admission gate)

Default-off configs reproduce legacy behavior exactly.
"""

import pytest

from vllm.platforms import current_platform
from vllm.pooling_params import PoolingParams
from vllm.v1.request import Request, RequestStatus
from vllm.v1.worker.dual_model_helpers import DualModelConfig

from .utils import create_requests, create_scheduler

pytestmark = pytest.mark.cpu_test


def _make_embed(rid: str, num_tokens: int = 8) -> Request:
    return Request(
        request_id=rid,
        prompt_token_ids=[7] * num_tokens,
        sampling_params=None,
        pooling_params=PoolingParams(task="embed"),
    )


# ---------------------------------------------------------------------------
# Patch B: pair dependency
# ---------------------------------------------------------------------------


def test_pair_dependency_disabled_by_default(monkeypatch):
    """No flag -> child with parent_request_id still lands in waiting."""
    monkeypatch.setattr(current_platform, "device_type", "cuda")
    scheduler = create_scheduler()
    scheduler.dual_model_config = DualModelConfig(embed_model="embed-model")

    parent = _make_embed("embed-0")
    child = create_requests(num_requests=1, num_tokens=8, req_ids=["decode-0"])[0]
    child.parent_request_id = "embed-0"

    scheduler.add_request(parent)
    scheduler.add_request(child)

    assert scheduler._gated_children == {}
    # Both went straight into the waiting queue.
    assert "embed-0" in scheduler.requests
    assert "decode-0" in scheduler.requests


def test_pair_dependency_holds_child_then_releases_on_parent_finish(monkeypatch):
    """With the flag on, child is held in _gated_children until parent finishes."""
    monkeypatch.setattr(current_platform, "device_type", "cuda")
    scheduler = create_scheduler()
    scheduler.dual_model_config = DualModelConfig(
        embed_model="embed-model", enforce_pair_dependency=True
    )

    parent = _make_embed("embed-0")
    child = create_requests(num_requests=1, num_tokens=8, req_ids=["decode-0"])[0]
    child.parent_request_id = "embed-0"

    scheduler.add_request(parent)
    scheduler.add_request(child)

    # Child is held back; parent is in the waiting queue.
    assert "decode-0" in scheduler._gated_children
    assert scheduler._children_of_parent["embed-0"] == {"decode-0"}
    assert "embed-0" in scheduler.requests
    assert "decode-0" in scheduler.requests  # still tracked

    # Mark the parent finished and free it. Child should release.
    parent.status = RequestStatus.FINISHED_LENGTH_CAPPED
    scheduler._free_request(parent)

    assert "decode-0" not in scheduler._gated_children
    assert "embed-0" not in scheduler._children_of_parent
    # Child is now in the normal waiting queue.
    assert "decode-0" in scheduler.requests


def test_pair_dependency_unknown_parent_falls_through(monkeypatch):
    """If parent_request_id is unknown, the gate is satisfied — legacy enqueue."""
    monkeypatch.setattr(current_platform, "device_type", "cuda")
    scheduler = create_scheduler()
    scheduler.dual_model_config = DualModelConfig(
        embed_model="embed-model", enforce_pair_dependency=True
    )

    child = create_requests(num_requests=1, num_tokens=8, req_ids=["decode-0"])[0]
    child.parent_request_id = "embed-DOESNOTEXIST"
    scheduler.add_request(child)

    assert scheduler._gated_children == {}
    assert "decode-0" in scheduler.requests


def test_pair_dependency_parent_abort_cascades_to_child(monkeypatch):
    """Aborted parent -> child is also aborted, not released to waiting."""
    monkeypatch.setattr(current_platform, "device_type", "cuda")
    scheduler = create_scheduler()
    scheduler.dual_model_config = DualModelConfig(
        embed_model="embed-model", enforce_pair_dependency=True
    )

    parent = _make_embed("embed-0")
    child = create_requests(num_requests=1, num_tokens=8, req_ids=["decode-0"])[0]
    child.parent_request_id = "embed-0"

    scheduler.add_request(parent)
    scheduler.add_request(child)
    assert "decode-0" in scheduler._gated_children

    scheduler.finish_requests("embed-0", RequestStatus.FINISHED_ABORTED)

    # Both freed; neither remains in tracking maps.
    assert "decode-0" not in scheduler._gated_children
    assert "embed-0" not in scheduler._children_of_parent
    assert "embed-0" not in scheduler.requests
    assert "decode-0" not in scheduler.requests


# ---------------------------------------------------------------------------
# Patch C: no-double-prefill admission gate
# ---------------------------------------------------------------------------


def test_ndp_disabled_admits_both_prefills_same_step(monkeypatch):
    """With the flag off, decode + embed prefill admit together (legacy)."""
    monkeypatch.setattr(current_platform, "device_type", "cuda")
    scheduler = create_scheduler(max_num_seqs=4, max_num_batched_tokens=64)
    scheduler.dual_model_config = DualModelConfig(embed_model="embed-model")

    decode = create_requests(num_requests=1, num_tokens=8, req_ids=["decode-0"])[0]
    embed = _make_embed("embed-0")
    scheduler.add_request(decode)
    scheduler.add_request(embed)

    output = scheduler.schedule()
    scheduled = set(output.num_scheduled_tokens.keys())
    assert "decode-0" in scheduled
    assert "embed-0" in scheduled


def test_ndp_enabled_defers_second_prefill_kind(monkeypatch):
    """With the flag on, embed-prefill defers when decode-prefill admits first."""
    monkeypatch.setenv("HB_SCHEDULER_DECISION_LOG", "1")
    monkeypatch.setattr(current_platform, "device_type", "cuda")
    scheduler = create_scheduler(max_num_seqs=4, max_num_batched_tokens=64)
    scheduler.dual_model_config = DualModelConfig(
        embed_model="embed-model", enforce_no_double_prefill=True
    )

    decode = create_requests(num_requests=1, num_tokens=8, req_ids=["decode-0"])[0]
    embed = _make_embed("embed-0")
    scheduler.add_request(decode)  # added first -> admitted first
    scheduler.add_request(embed)

    output = scheduler.schedule()
    scheduled = set(output.num_scheduled_tokens.keys())
    assert "decode-0" in scheduled
    # Embed should be deferred this step.
    assert "embed-0" not in scheduled

    # Decision log should record the deferral.
    reasons = [r for (rid, r, _) in scheduler._decisions if rid == "embed-0"]
    assert "DEFER_NO_DOUBLE_PREFILL" in reasons


def test_ndp_does_not_block_carryover_prefill(monkeypatch):
    """Refined NDP rule: a fresh embed-prefill is NOT blocked by a *carry-over*
    decode-prefill chunk that's already in `running` (chunked-prefill case).

    The earlier rule treated any decode-prefill bucket activity (running OR
    waiting) as a block trigger. With chunked prefill enabled, every step
    has a decode-prefill chunk in running, which would defer all embed
    admissions indefinitely. The refined rule fires only on *fresh*
    waiting-loop admissions, so carry-over chunks compose freely with new
    embed-prefill admissions.
    """
    monkeypatch.setenv("HB_SCHEDULER_DECISION_LOG", "1")
    monkeypatch.setattr(current_platform, "device_type", "cuda")
    scheduler = create_scheduler(
        max_num_seqs=4, max_num_batched_tokens=64,
        enable_chunked_prefill=True,
        long_prefill_token_threshold=4,
    )
    scheduler.dual_model_config = DualModelConfig(
        embed_model="embed-model", enforce_no_double_prefill=True
    )

    # Step 1: admit a long-ish decode prefill that chunked prefill will split.
    decode = create_requests(
        num_requests=1, num_tokens=16, req_ids=["decode-0"],
    )[0]
    scheduler.add_request(decode)
    out1 = scheduler.schedule()
    assert "decode-0" in out1.num_scheduled_tokens
    # Mid-prefill: still in `running` with computed_tokens < prompt_tokens.
    assert decode.num_computed_tokens < decode.num_prompt_tokens
    assert decode.num_output_tokens == 0  # still in prefill

    # Step 2: now add a fresh embed and schedule. The embed should be
    # admitted *despite* decode-0 being in running (carry-over). With the
    # OLD rule (running set the flag), the embed would be deferred.
    embed = _make_embed("embed-0", num_tokens=8)
    scheduler.add_request(embed)
    out2 = scheduler.schedule()
    scheduled2 = set(out2.num_scheduled_tokens.keys())
    assert "embed-0" in scheduled2, (
        "refined NDP rule should permit embed admission alongside a "
        "carry-over decode-prefill chunk"
    )
    # The decode-prefill chunk should still be progressing too.
    assert "decode-0" in scheduled2

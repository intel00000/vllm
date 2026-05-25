# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for the Scheduler's dual-model bookkeeping hooks.

A full Scheduler construction is heavy (requires kv_cache_config,
structured_output_manager, etc.), so these tests bypass __init__ and
exercise the helper methods directly on a hand-assembled stub. The
helpers only consult ``self.dual_model_config`` and the four counter
fields, which the stub provides.
"""

from types import SimpleNamespace

import pytest

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.worker.dual_model_helpers import DualModelConfig


def _stub_scheduler(dual_cfg: DualModelConfig | None) -> Scheduler:
    """Return a Scheduler with just the dual-model fields populated."""
    sched = Scheduler.__new__(Scheduler)
    sched.dual_model_config = dual_cfg
    sched._num_running_decode_reqs = 0
    sched._num_running_embed_reqs = 0
    sched._num_waiting_decode_reqs = 0
    sched._num_waiting_embed_reqs = 0
    return sched


def _req(model_id: str) -> SimpleNamespace:
    return SimpleNamespace(model_id=model_id, request_id=f"r-{model_id}")


def test_kind_returns_none_when_single_model():
    sched = _stub_scheduler(dual_cfg=None)
    assert sched._dual_model_request_kind(_req("decode")) is None
    assert sched._dual_model_request_kind(_req("embed")) is None


def test_kind_buckets_decode_and_embed_correctly():
    cfg = DualModelConfig(embed_model="some/embed-model")
    sched = _stub_scheduler(dual_cfg=cfg)
    assert sched._dual_model_request_kind(_req("decode")) == "decode"
    assert sched._dual_model_request_kind(_req("embed")) == "embed"
    # Unknown model_id (e.g., pooling task != "embed") falls through.
    assert sched._dual_model_request_kind(_req("classify")) is None


def test_counters_remain_zero_in_single_model_mode():
    sched = _stub_scheduler(dual_cfg=None)
    sched._mark_waiting_added_by_model(_req("decode"))
    sched._mark_running_added_by_model(_req("decode"))
    sched._mark_waiting_added_by_model(_req("embed"))
    sched._mark_running_added_by_model(_req("embed"))
    assert sched._num_running_decode_reqs == 0
    assert sched._num_running_embed_reqs == 0
    assert sched._num_waiting_decode_reqs == 0
    assert sched._num_waiting_embed_reqs == 0


def test_counters_increment_then_decrement_symmetrically():
    cfg = DualModelConfig(embed_model="some/embed-model")
    sched = _stub_scheduler(dual_cfg=cfg)

    r_dec, r_emb = _req("decode"), _req("embed")
    sched._mark_waiting_added_by_model(r_dec)
    sched._mark_waiting_added_by_model(r_emb)
    assert sched._num_waiting_decode_reqs == 1
    assert sched._num_waiting_embed_reqs == 1

    # Promote both to running.
    sched._mark_waiting_removed_by_model(r_dec)
    sched._mark_running_added_by_model(r_dec)
    sched._mark_waiting_removed_by_model(r_emb)
    sched._mark_running_added_by_model(r_emb)
    assert sched._num_waiting_decode_reqs == 0
    assert sched._num_waiting_embed_reqs == 0
    assert sched._num_running_decode_reqs == 1
    assert sched._num_running_embed_reqs == 1

    # Finish them.
    sched._mark_running_removed_by_model(r_dec)
    sched._mark_running_removed_by_model(r_emb)
    assert sched._num_running_decode_reqs == 0
    assert sched._num_running_embed_reqs == 0


def test_running_underflow_asserts():
    cfg = DualModelConfig(embed_model="some/embed-model")
    sched = _stub_scheduler(dual_cfg=cfg)
    with pytest.raises(AssertionError):
        sched._mark_running_removed_by_model(_req("decode"))


def test_waiting_underflow_asserts():
    cfg = DualModelConfig(embed_model="some/embed-model")
    sched = _stub_scheduler(dual_cfg=cfg)
    with pytest.raises(AssertionError):
        sched._mark_waiting_removed_by_model(_req("embed"))


def test_unknown_model_id_does_not_touch_counters():
    cfg = DualModelConfig(embed_model="some/embed-model")
    sched = _stub_scheduler(dual_cfg=cfg)
    sched._mark_waiting_added_by_model(_req("classify"))
    sched._mark_running_added_by_model(_req("classify"))
    assert sched._num_waiting_decode_reqs == 0
    assert sched._num_waiting_embed_reqs == 0
    assert sched._num_running_decode_reqs == 0
    assert sched._num_running_embed_reqs == 0

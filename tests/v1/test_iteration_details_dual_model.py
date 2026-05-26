# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for the dual-model extension of compute_iteration_details.

The original (single-model) behavior is covered by upstream tests; here
we cover:
  - the new max_seq_tokens field
  - dual-mode bucketing (decode_ctx_* vs embed_ctx_*) using
    req_id_to_model_id + scheduled_new_reqs
  - the single_runner_kind fallback for embed-only baselines
  - edge cases: empty scheduler output, unknown req_ids, new vs cached
    request mix
"""

from types import SimpleNamespace

from vllm.v1.core.sched.output import (
    CachedRequestData,
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.utils import IterationDetails, compute_iteration_details


def _make_new_req(req_id: str, model_id: str) -> NewRequestData:
    return NewRequestData(
        req_id=req_id,
        model_id=model_id,
        prompt_token_ids=[1, 2, 3],
        mm_features=[],
        sampling_params=None,
        pooling_params=None,
        block_ids=([0],),
        num_computed_tokens=0,
        lora_request=None,
    )


def _make_cached(
    req_ids: list[str],
    num_output_tokens: list[int],
) -> CachedRequestData:
    """Build a CachedRequestData that reports each req's num_output_tokens
    via the same path is_context_phase() consults."""
    return CachedRequestData(
        req_ids=req_ids,
        resumed_req_ids=set(),
        new_token_ids=[],
        all_token_ids={},
        new_block_ids=[None for _ in req_ids],
        num_computed_tokens=[1 for _ in req_ids],
        num_output_tokens=num_output_tokens,
    )


def _make_so(
    new_reqs: list[NewRequestData],
    num_scheduled_tokens: dict[str, int],
    cached: CachedRequestData | None = None,
) -> SchedulerOutput:
    return SchedulerOutput(
        scheduled_new_reqs=new_reqs,
        scheduled_cached_reqs=cached or CachedRequestData.make_empty(),
        num_scheduled_tokens=num_scheduled_tokens,
        total_num_scheduled_tokens=sum(num_scheduled_tokens.values()),
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[0],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )


# --------------------------------------------------------------------------
# Single-model legacy paths
# --------------------------------------------------------------------------


def test_empty_scheduler_output_returns_zeros():
    out = compute_iteration_details(_make_so([], {}))
    assert out == IterationDetails(
        num_ctx_requests=0,
        num_ctx_tokens=0,
        num_generation_requests=0,
        num_generation_tokens=0,
        max_seq_tokens=0,
        num_decode_ctx_requests=0,
        num_decode_ctx_tokens=0,
        num_embed_ctx_requests=0,
        num_embed_ctx_tokens=0,
    )


def test_single_model_mirrors_into_decode_ctx_by_default():
    """Without dual-mode info or pooling hint, ctx counts mirror into
    decode_ctx_* (legacy / generative path)."""
    so = _make_so(
        new_reqs=[_make_new_req("d1", "decode")],
        num_scheduled_tokens={"d1": 256},
    )
    out = compute_iteration_details(so)
    assert out.num_ctx_requests == 1
    assert out.num_ctx_tokens == 256
    assert out.num_decode_ctx_requests == 1
    assert out.num_decode_ctx_tokens == 256
    assert out.num_embed_ctx_requests == 0
    assert out.num_embed_ctx_tokens == 0
    assert out.max_seq_tokens == 256


def test_single_runner_kind_pooling_routes_to_embed_ctx():
    """Pass single_runner_kind='pooling' so an embed-only baseline run
    reports activity as ectx_* instead of dctx_*."""
    so = _make_so(
        new_reqs=[_make_new_req("e1", "embed")],
        num_scheduled_tokens={"e1": 64},
    )
    out = compute_iteration_details(so, single_runner_kind="pooling")
    assert out.num_decode_ctx_requests == 0
    assert out.num_embed_ctx_requests == 1
    assert out.num_embed_ctx_tokens == 64


def test_single_runner_kind_embed_alias_works():
    """Both 'pooling' and 'embed' route to ectx_*; everything else falls
    back to decode."""
    so = _make_so(
        new_reqs=[_make_new_req("e1", "embed")],
        num_scheduled_tokens={"e1": 32},
    )
    assert (
        compute_iteration_details(so, single_runner_kind="embed").num_embed_ctx_requests
        == 1
    )
    # Unknown kind falls through to decode.
    assert (
        compute_iteration_details(
            so, single_runner_kind="something_else"
        ).num_decode_ctx_requests
        == 1
    )


def test_max_seq_tokens_picks_the_largest():
    so = _make_so(
        new_reqs=[
            _make_new_req("d1", "decode"),
            _make_new_req("d2", "decode"),
            _make_new_req("d3", "decode"),
        ],
        num_scheduled_tokens={"d1": 100, "d2": 500, "d3": 250},
    )
    out = compute_iteration_details(so)
    assert out.max_seq_tokens == 500


# --------------------------------------------------------------------------
# Dual-model path
# --------------------------------------------------------------------------


def test_dual_mode_buckets_new_reqs_by_model_id():
    """In dual mode, scheduled_new_reqs go to decode_ctx vs embed_ctx
    based on their model_id (the per-step override path)."""
    so = _make_so(
        new_reqs=[
            _make_new_req("d1", "decode"),
            _make_new_req("e1", "embed"),
            _make_new_req("d2", "decode"),
        ],
        num_scheduled_tokens={"d1": 200, "e1": 50, "d2": 300},
    )
    out = compute_iteration_details(
        so,
        req_id_to_model_id={},
        embed_model_id="embed",
    )
    assert out.num_ctx_requests == 3
    assert out.num_decode_ctx_requests == 2
    assert out.num_decode_ctx_tokens == 500
    assert out.num_embed_ctx_requests == 1
    assert out.num_embed_ctx_tokens == 50


def test_dual_mode_uses_persistent_table_for_cached_context_reqs():
    """A cached request in context phase isn't in new_reqs but should
    still be classified via the persistent req_id_to_model_id table."""
    cached = _make_cached(req_ids=["e1"], num_output_tokens=[0])  # context phase
    so = _make_so(
        new_reqs=[],
        num_scheduled_tokens={"e1": 128},
        cached=cached,
    )
    out = compute_iteration_details(
        so,
        req_id_to_model_id={"e1": "embed"},
        embed_model_id="embed",
    )
    assert out.num_embed_ctx_requests == 1
    assert out.num_embed_ctx_tokens == 128
    assert out.num_decode_ctx_requests == 0


def test_dual_mode_decode_loop_reqs_count_as_generation():
    """A request with output tokens (decode loop) is NOT a context
    request -- counts toward num_generation_* regardless of bucket."""
    cached = _make_cached(req_ids=["d1"], num_output_tokens=[5])  # loop phase
    so = _make_so(
        new_reqs=[],
        num_scheduled_tokens={"d1": 1},
        cached=cached,
    )
    out = compute_iteration_details(
        so,
        req_id_to_model_id={"d1": "decode"},
        embed_model_id="embed",
    )
    assert out.num_generation_requests == 1
    assert out.num_generation_tokens == 1
    assert out.num_decode_ctx_requests == 0
    assert out.num_embed_ctx_requests == 0


def test_dual_mode_mixed_step():
    """Realistic mixed step: some new prefills + some carry-over decode
    loop reqs. Verify all four buckets get populated correctly."""
    new_reqs = [
        _make_new_req("d1_new", "decode"),  # fresh decode prefill
        _make_new_req("e1_new", "embed"),  # fresh embed prefill
    ]
    cached = _make_cached(
        req_ids=["d_loop_a", "d_loop_b"],
        num_output_tokens=[3, 7],  # both in loop phase
    )
    so = _make_so(
        new_reqs=new_reqs,
        num_scheduled_tokens={
            "d1_new": 500,
            "e1_new": 64,
            "d_loop_a": 1,
            "d_loop_b": 1,
        },
        cached=cached,
    )
    out = compute_iteration_details(
        so,
        req_id_to_model_id={"d_loop_a": "decode", "d_loop_b": "decode"},
        embed_model_id="embed",
    )
    assert out.num_ctx_requests == 2  # the two new prefills
    assert out.num_decode_ctx_requests == 1
    assert out.num_decode_ctx_tokens == 500
    assert out.num_embed_ctx_requests == 1
    assert out.num_embed_ctx_tokens == 64
    assert out.num_generation_requests == 2
    assert out.num_generation_tokens == 2
    assert out.max_seq_tokens == 500


def test_dual_mode_unknown_model_id_falls_back_to_decode_bucket():
    """If a req in context phase has no entry in req_id_to_model_id
    AND isn't in new_reqs, we still need to bucket it. The function
    defaults to decode (safer than embed)."""
    cached = _make_cached(req_ids=["mystery"], num_output_tokens=[0])
    so = _make_so(
        new_reqs=[],
        num_scheduled_tokens={"mystery": 128},
        cached=cached,
    )
    # Empty table, no model_id resolution
    out = compute_iteration_details(
        so,
        req_id_to_model_id={},
        embed_model_id="embed",
    )
    assert out.num_decode_ctx_requests == 1
    assert out.num_embed_ctx_requests == 0


def test_dual_mode_new_req_model_id_overrides_stale_table():
    """If new_reqs has model_id=embed but the persistent table has the
    same req_id mapped to 'decode' (stale), the new_req's model_id wins."""
    new = [_make_new_req("collision", "embed")]
    so = _make_so(
        new_reqs=new,
        num_scheduled_tokens={"collision": 50},
    )
    out = compute_iteration_details(
        so,
        req_id_to_model_id={"collision": "decode"},  # stale
        embed_model_id="embed",
    )
    assert out.num_embed_ctx_requests == 1
    assert out.num_decode_ctx_requests == 0


def test_dual_mode_requires_both_table_and_embed_id():
    """If req_id_to_model_id is provided but embed_model_id is None
    (or vice versa), dual_mode is False -- fall through to single-
    model mirroring."""
    so = _make_so(
        new_reqs=[_make_new_req("e1", "embed")],
        num_scheduled_tokens={"e1": 64},
    )
    # Missing embed_model_id -> single-model behavior.
    out = compute_iteration_details(so, req_id_to_model_id={})
    assert out.num_decode_ctx_requests == 1  # mirrored
    assert out.num_embed_ctx_requests == 0
    # Missing req_id_to_model_id -> same.
    out2 = compute_iteration_details(so, embed_model_id="embed")
    assert out2.num_decode_ctx_requests == 1
    assert out2.num_embed_ctx_requests == 0

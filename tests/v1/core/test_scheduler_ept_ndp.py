# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for EPT (per-step bucket budgets) and NDP (no-double-prefill).

The actual budget enforcement happens inside the schedule() loop using
loop-local state, so we can't unit-test the *integration* without a
full Scheduler + KV cache. Instead we cover the two pure-function
predicates that the loop delegates to:

    Scheduler._bucket_for_request -- classifies a request into
        {decode_prefill, decode_loop, embed_prefill}
    Scheduler._ndp_should_defer -- decides whether NDP blocks a fresh
        admission given the step's current FRESH_* flags

Together those two cover every interesting decision the loop body
makes; the surrounding loop code is mechanical and exercised by the
post-port end-to-end smoke.
"""

from types import SimpleNamespace

import pytest

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.worker.dual_model_helpers import DualModelConfig


# --------------------------------------------------------------------------
# Test fixtures
# --------------------------------------------------------------------------


def _req(model_id: str = "decode", num_output_tokens: int = 0) -> SimpleNamespace:
    return SimpleNamespace(model_id=model_id, num_output_tokens=num_output_tokens)


@pytest.fixture
def dual_cfg() -> DualModelConfig:
    return DualModelConfig(embed_model="some/embed-model")


# --------------------------------------------------------------------------
# Scheduler._bucket_for_request -- classifier
# --------------------------------------------------------------------------


def test_bucket_single_model_treats_all_as_decode():
    """Without a dual cfg, every request is decode (prefill or loop)."""
    assert Scheduler._bucket_for_request(None, _req("decode")) == "decode_prefill"
    assert (
        Scheduler._bucket_for_request(None, _req("decode", num_output_tokens=5))
        == "decode_loop"
    )
    # Even an "embed"-named request falls through to decode in single-model.
    assert Scheduler._bucket_for_request(None, _req("embed")) == "decode_prefill"


def test_bucket_decode_prefill_vs_loop(dual_cfg):
    decode_prefill = _req("decode", num_output_tokens=0)
    decode_loop = _req("decode", num_output_tokens=1)
    assert Scheduler._bucket_for_request(dual_cfg, decode_prefill) == "decode_prefill"
    assert Scheduler._bucket_for_request(dual_cfg, decode_loop) == "decode_loop"


def test_bucket_embed_always_classifies_as_embed_prefill(dual_cfg):
    # Embed reqs are one-shot prefill -- num_output_tokens is irrelevant.
    assert (
        Scheduler._bucket_for_request(dual_cfg, _req("embed", num_output_tokens=0))
        == "embed_prefill"
    )
    assert (
        Scheduler._bucket_for_request(dual_cfg, _req("embed", num_output_tokens=10))
        == "embed_prefill"
    )


def test_bucket_unknown_model_id_classified_as_decode(dual_cfg):
    """A request with neither decode nor embed model_id falls into the
    decode side -- safer default, since the embed side is the gated one."""
    unknown = _req("classify", num_output_tokens=0)
    assert Scheduler._bucket_for_request(dual_cfg, unknown) == "decode_prefill"


def test_bucket_decode_loop_at_boundary(dual_cfg):
    """Exactly num_output_tokens == 1 is the boundary -- belongs to loop."""
    assert (
        Scheduler._bucket_for_request(dual_cfg, _req("decode", num_output_tokens=1))
        == "decode_loop"
    )


def test_bucket_respects_custom_model_ids():
    cfg = DualModelConfig(
        embed_model="m",
        decode_model_id="gen",
        embed_model_id="vec",
    )
    assert Scheduler._bucket_for_request(cfg, _req("vec")) == "embed_prefill"
    assert Scheduler._bucket_for_request(cfg, _req("gen")) == "decode_prefill"
    # Custom decode_model_id doesn't change the classification path; the
    # function only checks embed match.
    assert Scheduler._bucket_for_request(cfg, _req("gen", 5)) == "decode_loop"


# --------------------------------------------------------------------------
# Scheduler._ndp_should_defer -- mutual-exclusion predicate
# --------------------------------------------------------------------------


def test_ndp_disabled_never_defers():
    """When the gate is off, no combination defers."""
    for bucket in ("decode_prefill", "decode_loop", "embed_prefill"):
        for dec, emb in ((True, True), (True, False), (False, True), (False, False)):
            assert (
                Scheduler._ndp_should_defer(
                    enforce_no_double_prefill=False,
                    request_bucket=bucket,
                    fresh_decode_prefill_admitted=dec,
                    fresh_embed_prefill_admitted=emb,
                )
                is False
            )


def test_ndp_no_flags_set_admits_anything():
    """First admission of the step always passes -- no flags set yet."""
    for bucket in ("decode_prefill", "decode_loop", "embed_prefill"):
        assert (
            Scheduler._ndp_should_defer(
                enforce_no_double_prefill=True,
                request_bucket=bucket,
                fresh_decode_prefill_admitted=False,
                fresh_embed_prefill_admitted=False,
            )
            is False
        )


def test_ndp_decode_prefill_blocks_embed_prefill():
    assert (
        Scheduler._ndp_should_defer(
            enforce_no_double_prefill=True,
            request_bucket="embed_prefill",
            fresh_decode_prefill_admitted=True,
            fresh_embed_prefill_admitted=False,
        )
        is True
    )


def test_ndp_embed_prefill_blocks_decode_prefill():
    assert (
        Scheduler._ndp_should_defer(
            enforce_no_double_prefill=True,
            request_bucket="decode_prefill",
            fresh_decode_prefill_admitted=False,
            fresh_embed_prefill_admitted=True,
        )
        is True
    )


def test_ndp_does_not_block_decode_loop():
    """decode_loop is the productive overlap partner for embed_prefill;
    it must NEVER be blocked by NDP regardless of FRESH_* flags."""
    for dec, emb in ((True, True), (True, False), (False, True), (False, False)):
        assert (
            Scheduler._ndp_should_defer(
                enforce_no_double_prefill=True,
                request_bucket="decode_loop",
                fresh_decode_prefill_admitted=dec,
                fresh_embed_prefill_admitted=emb,
            )
            is False
        ), f"decode_loop should never be NDP-blocked, but was with dec={dec}, emb={emb}"


def test_ndp_same_kind_does_not_block_itself():
    """A decode_prefill followed by another decode_prefill is fine
    (still one fresh decode prefill from NDP's POV). NDP is about the
    cross-model case."""
    assert (
        Scheduler._ndp_should_defer(
            enforce_no_double_prefill=True,
            request_bucket="decode_prefill",
            fresh_decode_prefill_admitted=True,
            fresh_embed_prefill_admitted=False,
        )
        is False
    )
    assert (
        Scheduler._ndp_should_defer(
            enforce_no_double_prefill=True,
            request_bucket="embed_prefill",
            fresh_decode_prefill_admitted=False,
            fresh_embed_prefill_admitted=True,
        )
        is False
    )


def test_ndp_both_flags_set_blocks_either_fresh_prefill():
    """If somehow both flags are set this step (defensive), the gate
    still blocks any new fresh prefill (but not decode_loop)."""
    for bucket in ("decode_prefill", "embed_prefill"):
        assert (
            Scheduler._ndp_should_defer(
                enforce_no_double_prefill=True,
                request_bucket=bucket,
                fresh_decode_prefill_admitted=True,
                fresh_embed_prefill_admitted=True,
            )
            is True
        ), f"bucket={bucket} should be blocked when both flags set"
    # decode_loop still passes through
    assert (
        Scheduler._ndp_should_defer(
            enforce_no_double_prefill=True,
            request_bucket="decode_loop",
            fresh_decode_prefill_admitted=True,
            fresh_embed_prefill_admitted=True,
        )
        is False
    )

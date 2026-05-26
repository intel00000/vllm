# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for wave-batching gate statics.

These exercise the pure-function predicates (`_compute_in_prefill_sets`
and `_wave_gate_blocks`) directly off the Scheduler class -- no
instance is needed since both helpers are @staticmethod. The full
schedule() integration is covered by W3 + slurm validation, not here.
"""

from types import SimpleNamespace

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.worker.dual_model_helpers import DualModelConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _req(
    request_id: str,
    model_id: str = "decode",
    num_computed_tokens: int = 0,
    num_prompt_tokens: int = 100,
    parent_request_id: str | None = None,
) -> SimpleNamespace:
    """Minimal stand-in for Request -- the gate helpers only touch
    request_id, model_id, num_computed_tokens, num_prompt_tokens,
    parent_request_id."""
    return SimpleNamespace(
        request_id=request_id,
        model_id=model_id,
        num_computed_tokens=num_computed_tokens,
        num_prompt_tokens=num_prompt_tokens,
        parent_request_id=parent_request_id,
    )


def _cfg(wave_batching: bool = True, wave_size: int | None = None) -> DualModelConfig:
    return DualModelConfig(
        embed_model="embed-model",
        wave_batching=wave_batching,
        wave_size=wave_size,
    )


# ---------------------------------------------------------------------------
# _compute_in_prefill_sets
# ---------------------------------------------------------------------------


def test_compute_sets_empty_when_dual_cfg_none():
    """Single-model mode: no embed bucket exists, both sets empty."""
    running = [
        _req("a", model_id="decode", num_computed_tokens=10, num_prompt_tokens=100),
        _req("b", model_id="embed", num_computed_tokens=5, num_prompt_tokens=20),
    ]
    gen_pf, embed_pf = Scheduler._compute_in_prefill_sets(running, None)
    assert gen_pf == set()
    assert embed_pf == set()


def test_compute_sets_buckets_gen_vs_embed_in_prefill():
    """Mix of running reqs: bucket by model_id, drop those past prefill."""
    cfg = _cfg()
    running = [
        # In prefill, gen side.
        _req("g1", model_id="decode", num_computed_tokens=10, num_prompt_tokens=100),
        _req("g2", model_id="decode", num_computed_tokens=99, num_prompt_tokens=100),
        # In prefill, embed side.
        _req("e1", model_id="embed", num_computed_tokens=0, num_prompt_tokens=50),
        # Gen past prefill (decode loop, fully primed).
        _req("g3", model_id="decode", num_computed_tokens=100, num_prompt_tokens=100),
        # Gen past prefill (output token already produced, computed > prompt).
        _req("g4", model_id="decode", num_computed_tokens=120, num_prompt_tokens=100),
        # Embed past prefill (one-shot done -- the helper still excludes it).
        _req("e2", model_id="embed", num_computed_tokens=50, num_prompt_tokens=50),
    ]
    gen_pf, embed_pf = Scheduler._compute_in_prefill_sets(running, cfg)
    assert gen_pf == {"g1", "g2"}
    assert embed_pf == {"e1"}


def test_compute_sets_empty_running_returns_empty_sets():
    gen_pf, embed_pf = Scheduler._compute_in_prefill_sets([], _cfg())
    assert gen_pf == set()
    assert embed_pf == set()


# ---------------------------------------------------------------------------
# _wave_gate_blocks
# ---------------------------------------------------------------------------


def test_wave_gate_disabled_when_dual_cfg_none():
    """Single-model -- gate must not fire."""
    req = _req("e1", model_id="embed")
    assert (
        Scheduler._wave_gate_blocks(
            req, None, {"g1"}, set(), "embed_prefill"
        )
        is None
    )


def test_wave_gate_disabled_when_wave_batching_false():
    """Gate is opt-in: wave_batching=False short-circuits to None."""
    cfg = _cfg(wave_batching=False)
    req = _req("e1", model_id="embed")
    assert (
        Scheduler._wave_gate_blocks(
            req, cfg, {"g1", "g2"}, {"e_prev"}, "embed_prefill"
        )
        is None
    )


def test_wave_gate_passes_when_no_prefill_active():
    """Empty prefill sets: nothing to exclude against, admit freely."""
    cfg = _cfg()
    req = _req("e1", model_id="embed")
    assert (
        Scheduler._wave_gate_blocks(req, cfg, set(), set(), "embed_prefill")
        is None
    )


def test_wave_gate_blocks_embed_when_gen_prefill_active():
    """Embed-prefill admission while gen-prefill running -- defer."""
    cfg = _cfg()
    req = _req("e1", model_id="embed")
    reason = Scheduler._wave_gate_blocks(
        req, cfg, {"g1"}, set(), "embed_prefill"
    )
    assert reason == "DEFER_WAVE_GEN_PREFILL_ACTIVE"


def test_wave_gate_blocks_decode_prefill_when_embed_prefill_active():
    """Decode-prefill admission while embed-prefill running -- defer."""
    cfg = _cfg()
    req = _req("g1", model_id="decode")
    reason = Scheduler._wave_gate_blocks(
        req, cfg, set(), {"e1"}, "decode_prefill"
    )
    assert reason == "DEFER_WAVE_EMBED_PREFILL_ACTIVE"


def test_wave_gate_never_blocks_decode_loop():
    """Decode-loop x embed-prefill is the productive overlap partner --
    never block it, regardless of either side's prefill state."""
    cfg = _cfg()
    req = _req("g1", model_id="decode")
    assert (
        Scheduler._wave_gate_blocks(
            req, cfg, {"g_other"}, {"e1", "e2"}, "decode_loop"
        )
        is None
    )


def test_wave_gate_skips_pair_dep_children():
    """Requests with parent_request_id are pair-dep'd; the wave-gate
    must short-circuit and let pair-dep own the ordering."""
    cfg = _cfg()
    req = _req("d_child", model_id="decode", parent_request_id="e_parent")
    # Would otherwise be DEFER_WAVE_EMBED_PREFILL_ACTIVE.
    assert (
        Scheduler._wave_gate_blocks(
            req, cfg, set(), {"e_parent"}, "decode_prefill"
        )
        is None
    )
    # Same for an embed child (hypothetical).
    embed_req = _req("e_child", model_id="embed", parent_request_id="g_parent")
    assert (
        Scheduler._wave_gate_blocks(
            embed_req, cfg, {"g_parent"}, set(), "embed_prefill"
        )
        is None
    )


def test_wave_size_caps_embed_concurrent_prefills():
    """wave_size=2 + embed_in_prefill={a,b} -> next embed admit defers."""
    cfg = _cfg(wave_size=2)
    req = _req("e_new", model_id="embed")
    reason = Scheduler._wave_gate_blocks(
        req, cfg, set(), {"a", "b"}, "embed_prefill"
    )
    assert reason == "DEFER_WAVE_SIZE_CAP"


def test_wave_size_allows_below_cap():
    """wave_size=2 + one embed prefill in flight + clean gen side: admit."""
    cfg = _cfg(wave_size=2)
    req = _req("e_new", model_id="embed")
    assert (
        Scheduler._wave_gate_blocks(
            req, cfg, set(), {"a"}, "embed_prefill"
        )
        is None
    )


def test_wave_size_caps_decode_prefill_side():
    """Symmetric: wave_size also caps decode_prefill concurrency."""
    cfg = _cfg(wave_size=3)
    req = _req("g_new", model_id="decode")
    reason = Scheduler._wave_gate_blocks(
        req, cfg, {"g1", "g2", "g3"}, set(), "decode_prefill"
    )
    assert reason == "DEFER_WAVE_SIZE_CAP"


def test_phase_exclusion_dominates_size_cap():
    """When both phase-exclusion and size-cap would fire, the phase
    reason wins (phase exclusion is checked first; size cap is the
    fallback diagnostic when phase is clear)."""
    cfg = _cfg(wave_size=2)
    req = _req("e_new", model_id="embed")
    # gen side has prefill active AND embed side already at cap.
    reason = Scheduler._wave_gate_blocks(
        req, cfg, {"g1"}, {"a", "b"}, "embed_prefill"
    )
    assert reason == "DEFER_WAVE_GEN_PREFILL_ACTIVE"

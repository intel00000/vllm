# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Signature-lock test for LLMEngine.add_request(parent_request_id=...).

scripts/profile_dual_model.py calls
    engine.add_request(decode_rid, prompt, params, parent_request_id=embed_rid)
to register a decode child whose parent is the embed request. If the
kwarg gets removed or renamed, the call raises TypeError at runtime and
the pair-dep admission gate never sees the parent_request_id. This test
catches that regression at the signature level.
"""

import inspect


def test_add_request_accepts_parent_request_id():
    from vllm.v1.engine.llm_engine import LLMEngine

    sig = inspect.signature(LLMEngine.add_request)
    assert "parent_request_id" in sig.parameters, (
        "LLMEngine.add_request must accept a 'parent_request_id' kwarg so "
        "scripts/profile_dual_model.py:_submit_dual_requests can wire it "
        "to the pair-dep admission gate."
    )
    param = sig.parameters["parent_request_id"]
    assert param.default is None, (
        "parent_request_id should default to None (single-model paths "
        "must not opt into pair-dep gating accidentally)."
    )

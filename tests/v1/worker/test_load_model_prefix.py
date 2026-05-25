# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Signature-lock test for GPUModelRunner.load_model(prefix=...).

DualModelRunner calls ``self.embed_runner.load_model(prefix="embed_model")``
to namespace the embed model so its layers don't collide with the decode
model in static_forward_context. If the V2 runner's load_model stops
accepting ``prefix``, the call silently no-ops and the two models clobber
each other -- this test catches that regression at the signature level.
The actual end-to-end behavior is exercised by the dual-model smoke run.
"""

import inspect


def test_v2_load_model_accepts_prefix():
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    sig = inspect.signature(GPUModelRunner.load_model)
    assert "prefix" in sig.parameters, (
        "GPUModelRunner.load_model must accept a 'prefix' kwarg so the "
        "embed model can be namespaced under EMBED_MODEL_PREFIX."
    )
    prefix_param = sig.parameters["prefix"]
    assert prefix_param.default == "", (
        "GPUModelRunner.load_model 'prefix' must default to empty string "
        "to preserve single-model behavior."
    )


def test_base_loader_accepts_prefix():
    from vllm.model_executor.model_loader.base_loader import BaseModelLoader

    sig = inspect.signature(BaseModelLoader.load_model)
    assert "prefix" in sig.parameters

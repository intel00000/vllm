# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm.v1.worker.dual_model_runner import DualModelRunner
from vllm.v1.worker.gpu_worker import _select_model_runner_class


def _vllm_config(additional_config: dict | None) -> SimpleNamespace:
    return SimpleNamespace(additional_config=additional_config)


def test_selector_returns_v1_by_default():
    cfg = _vllm_config(None)
    cls = _select_model_runner_class(cfg, use_v2_model_runner=False)
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    assert cls is GPUModelRunner


def test_selector_returns_v2_when_env_set():
    cfg = _vllm_config(None)
    cls = _select_model_runner_class(cfg, use_v2_model_runner=True)
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner as GPUModelRunnerV2

    assert cls is GPUModelRunnerV2


def test_selector_returns_dual_when_additional_config_set():
    cfg = _vllm_config({"dual_model": {"embed_model": "some/embed-model"}})
    cls = _select_model_runner_class(cfg, use_v2_model_runner=False)
    assert cls is DualModelRunner


def test_selector_prefers_dual_over_v2():
    cfg = _vllm_config({"dual_model": {"embed_model": "some/embed-model"}})
    cls = _select_model_runner_class(cfg, use_v2_model_runner=True)
    assert cls is DualModelRunner


def test_selector_ignores_unrelated_additional_config():
    cfg = _vllm_config({"some_other_feature": {"foo": "bar"}})
    cls = _select_model_runner_class(cfg, use_v2_model_runner=False)
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    assert cls is GPUModelRunner


def test_dual_model_worker_shim_subclasses_worker():
    from vllm.v1.worker.dual_model_gpu_worker import DualModelWorker
    from vllm.v1.worker.gpu_worker import Worker

    assert issubclass(DualModelWorker, Worker)

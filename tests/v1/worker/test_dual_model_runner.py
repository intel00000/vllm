# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, field

from vllm.v1.worker.dual_model_helpers import DualModelConfig
from vllm.v1.worker.dual_model_runner import DualModelRunner


@dataclass
class FakeModelConfig:
    model: str = "decode-model"
    tokenizer: str = "decode-model"
    runner: str = "generate"
    served_model_name: str = "decode-model"
    dtype: str = "bfloat16"
    max_model_len: int = 1024
    enforce_eager: bool = True
    model_weights: str = ""
    hf_config_path: str | None = "config.json"
    is_encoder_decoder: bool = False


@dataclass
class FakeSchedulerConfig:
    max_model_len: int = 1024
    is_encoder_decoder: bool = False
    runner_type: str = "generate"


@dataclass
class FakeCompilationConfig:
    static_forward_context: dict[str, object] = field(default_factory=dict, init=False)
    static_all_moe_layers: list[str] = field(default_factory=list, init=False)


@dataclass
class FakeVllmConfig:
    model_config: FakeModelConfig = field(default_factory=FakeModelConfig)
    scheduler_config: FakeSchedulerConfig = field(default_factory=FakeSchedulerConfig)
    compilation_config: FakeCompilationConfig = field(
        default_factory=FakeCompilationConfig
    )
    additional_config: dict = field(default_factory=dict)


def test_build_embed_vllm_config_detaches_compilation_state():
    runner = DualModelRunner.__new__(DualModelRunner)
    vllm_config = FakeVllmConfig(
        additional_config={"dual_model": {"embed_model": "embed-model"}}
    )
    vllm_config.compilation_config.static_forward_context["model.layers.0"] = object()
    vllm_config.compilation_config.static_all_moe_layers.append("moe.layer.0")

    embed_vllm_config = runner._build_embed_vllm_config(
        vllm_config,
        DualModelConfig(embed_model="embed-model"),
    )

    assert embed_vllm_config is not vllm_config
    assert embed_vllm_config.model_config.model == "embed-model"
    assert embed_vllm_config.model_config.runner == "pooling"
    assert embed_vllm_config.scheduler_config.runner_type == "pooling"
    assert embed_vllm_config.additional_config == {}

    assert embed_vllm_config.compilation_config is not vllm_config.compilation_config
    assert (
        embed_vllm_config.compilation_config.static_forward_context
        is not vllm_config.compilation_config.static_forward_context
    )
    assert (
        embed_vllm_config.compilation_config.static_all_moe_layers
        is not vllm_config.compilation_config.static_all_moe_layers
    )
    assert embed_vllm_config.compilation_config.static_forward_context == {}
    assert embed_vllm_config.compilation_config.static_all_moe_layers == []
    assert list(vllm_config.compilation_config.static_forward_context) == [
        "model.layers.0"
    ]
    assert vllm_config.compilation_config.static_all_moe_layers == ["moe.layer.0"]


def test_dual_model_runner_delegates_unknown_attrs_to_decode_runner():
    runner = DualModelRunner.__new__(DualModelRunner)
    runner.decode_runner = type("DecodeRunner", (), {"lora_config": "cfg"})()

    assert runner.lora_config == "cfg"

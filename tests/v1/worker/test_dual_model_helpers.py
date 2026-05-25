# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.v1.worker.dual_model_helpers import (
    DEFAULT_DECODE_MODEL_ID,
    DEFAULT_EMBED_MODEL_ID,
    DUAL_MODEL_CONFIG_KEY,
    DualModelConfig,
)


def test_constants():
    assert DUAL_MODEL_CONFIG_KEY == "dual_model"
    assert DEFAULT_DECODE_MODEL_ID == "decode"
    assert DEFAULT_EMBED_MODEL_ID == "embed"


def test_default_construct():
    cfg = DualModelConfig(embed_model="some/embed-model")
    assert cfg.embed_model == "some/embed-model"
    assert cfg.decode_model_id == DEFAULT_DECODE_MODEL_ID
    assert cfg.embed_model_id == DEFAULT_EMBED_MODEL_ID
    assert cfg.embed_tokenizer is None
    assert cfg.decode_running_reserve is None
    assert cfg.max_embed_running_reqs is None
    assert cfg.enforce_pair_dependency is False
    assert cfg.enforce_no_double_prefill is False
    assert cfg.embed_release_when_decode_waiting_drained is False
    assert cfg.kv_pressure_skip_threshold is None


def test_from_vllm_config_parses_populated_dict():
    vllm_config = SimpleNamespace(
        additional_config={
            DUAL_MODEL_CONFIG_KEY: {
                "embed_model": "some/embed-model",
                "decode_running_reserve": 16,
                "max_embed_running_reqs": 4,
                "max_embed_prefill_tokens_per_step": 2048,
                "enforce_pair_dependency": True,
                "enforce_no_double_prefill": True,
                "embed_release_when_decode_waiting_drained": True,
                "kv_pressure_skip_threshold": 0.9,
            }
        }
    )
    cfg = DualModelConfig.from_vllm_config(vllm_config)
    assert cfg is not None
    assert cfg.embed_model == "some/embed-model"
    assert cfg.decode_running_reserve == 16
    assert cfg.max_embed_running_reqs == 4
    assert cfg.max_embed_prefill_tokens_per_step == 2048
    assert cfg.enforce_pair_dependency is True
    assert cfg.enforce_no_double_prefill is True
    assert cfg.embed_release_when_decode_waiting_drained is True
    assert cfg.kv_pressure_skip_threshold == 0.9


def test_from_vllm_config_returns_none_when_additional_config_missing():
    vllm_config = SimpleNamespace()
    assert DualModelConfig.from_vllm_config(vllm_config) is None


def test_from_vllm_config_returns_none_when_additional_config_not_dict():
    vllm_config = SimpleNamespace(additional_config=None)
    assert DualModelConfig.from_vllm_config(vllm_config) is None


def test_from_vllm_config_returns_none_when_dual_model_key_missing():
    vllm_config = SimpleNamespace(additional_config={"something_else": {}})
    assert DualModelConfig.from_vllm_config(vllm_config) is None


def test_from_vllm_config_returns_none_when_dual_model_value_not_dict():
    vllm_config = SimpleNamespace(
        additional_config={DUAL_MODEL_CONFIG_KEY: "not-a-dict"}
    )
    assert DualModelConfig.from_vllm_config(vllm_config) is None


def test_from_vllm_config_raises_when_embed_model_missing():
    vllm_config = SimpleNamespace(
        additional_config={DUAL_MODEL_CONFIG_KEY: {"decode_running_reserve": 16}}
    )
    with pytest.raises(ValueError, match="embed_model"):
        DualModelConfig.from_vllm_config(vllm_config)


def test_from_vllm_config_raises_when_embed_model_empty_string():
    vllm_config = SimpleNamespace(
        additional_config={DUAL_MODEL_CONFIG_KEY: {"embed_model": ""}}
    )
    with pytest.raises(ValueError, match="embed_model"):
        DualModelConfig.from_vllm_config(vllm_config)


def test_frozen_dataclass():
    cfg = DualModelConfig(embed_model="some/embed-model")
    with pytest.raises((AttributeError, Exception)):
        cfg.embed_model = "other/embed-model"  # type: ignore[misc]

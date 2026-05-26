# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.v1.core.sched.output import (
    CachedRequestData,
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.dual_model_helpers import (
    DEFAULT_DECODE_MODEL_ID,
    DEFAULT_EMBED_MODEL_ID,
    DUAL_MODEL_CONFIG_KEY,
    DualModelConfig,
    SplitSchedulerOutputs,
    merge_model_runner_outputs,
    split_scheduler_output_by_model,
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
    assert cfg.wave_batching is False
    assert cfg.wave_size is None


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
                "embed_enable_chunked_prefill": False,
                "wave_batching": True,
                "wave_size": 128,
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
    assert cfg.embed_enable_chunked_prefill is False
    assert cfg.wave_batching is True
    assert cfg.wave_size == 128


def test_embed_enable_chunked_prefill_defaults_to_none():
    """None means 'inherit gen scheduler_config.enable_chunked_prefill'."""
    cfg = DualModelConfig(embed_model="some/embed-model")
    assert cfg.embed_enable_chunked_prefill is None


def test_wave_batching_omitted_keys_default_off():
    """When the raw cfg lacks wave_batching/wave_size, the gate is off."""
    vllm_config = SimpleNamespace(
        additional_config={
            DUAL_MODEL_CONFIG_KEY: {"embed_model": "some/embed-model"}
        }
    )
    cfg = DualModelConfig.from_vllm_config(vllm_config)
    assert cfg is not None
    assert cfg.wave_batching is False
    assert cfg.wave_size is None


def test_wave_batching_enabled_without_size():
    """wave_batching=True with no wave_size keeps wave_size None (unbounded)."""
    vllm_config = SimpleNamespace(
        additional_config={
            DUAL_MODEL_CONFIG_KEY: {
                "embed_model": "some/embed-model",
                "wave_batching": True,
            }
        }
    )
    cfg = DualModelConfig.from_vllm_config(vllm_config)
    assert cfg is not None
    assert cfg.wave_batching is True
    assert cfg.wave_size is None


def test_wave_batching_coerces_truthy_to_bool():
    """Non-bool truthy values (e.g. 1) are normalised to True."""
    vllm_config = SimpleNamespace(
        additional_config={
            DUAL_MODEL_CONFIG_KEY: {
                "embed_model": "some/embed-model",
                "wave_batching": 1,
            }
        }
    )
    cfg = DualModelConfig.from_vllm_config(vllm_config)
    assert cfg is not None
    assert cfg.wave_batching is True


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


# ---------------------------------------------------------------------------
# split_scheduler_output_by_model / merge_model_runner_outputs
# ---------------------------------------------------------------------------


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


def _make_scheduler_output(
    new_reqs: list[NewRequestData],
    num_scheduled_tokens: dict[str, int],
    finished_req_ids: set[str] | None = None,
) -> SchedulerOutput:
    return SchedulerOutput(
        scheduled_new_reqs=new_reqs,
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens=num_scheduled_tokens,
        total_num_scheduled_tokens=sum(num_scheduled_tokens.values()),
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[0],
        finished_req_ids=finished_req_ids or set(),
        free_encoder_mm_hashes=[],
    )


def test_split_routes_new_requests_by_model_id():
    new_reqs = [
        _make_new_req("d1", "decode"),
        _make_new_req("e1", "embed"),
        _make_new_req("d2", "decode"),
    ]
    sched = _make_scheduler_output(
        new_reqs=new_reqs,
        num_scheduled_tokens={"d1": 4, "e1": 3, "d2": 4},
    )
    req_id_to_model_id: dict[str, str] = {}

    split = split_scheduler_output_by_model(
        sched,
        req_id_to_model_id,
        decode_model_id="decode",
        embed_model_id="embed",
        decode_kv_group_indices=(0,),
        embed_kv_group_indices=(0,),
    )

    assert isinstance(split, SplitSchedulerOutputs)
    assert {r.req_id for r in split.decode.scheduled_new_reqs} == {"d1", "d2"}
    assert {r.req_id for r in split.embed.scheduled_new_reqs} == {"e1"}
    assert set(split.decode.num_scheduled_tokens.keys()) == {"d1", "d2"}
    assert set(split.embed.num_scheduled_tokens.keys()) == {"e1"}
    assert split.decode.total_num_scheduled_tokens == 8
    assert split.embed.total_num_scheduled_tokens == 3
    assert req_id_to_model_id == {"d1": "decode", "e1": "embed", "d2": "decode"}
    assert split.scheduled_req_order == ["d1", "e1", "d2"]


def test_split_preserves_finished_ids_and_drops_from_table():
    new_reqs = [_make_new_req("d1", "decode"), _make_new_req("e1", "embed")]
    sched = _make_scheduler_output(
        new_reqs=new_reqs,
        num_scheduled_tokens={"d1": 4, "e1": 3},
        finished_req_ids={"d1", "e1"},
    )
    req_id_to_model_id: dict[str, str] = {}
    split = split_scheduler_output_by_model(
        sched,
        req_id_to_model_id,
        decode_model_id="decode",
        embed_model_id="embed",
        decode_kv_group_indices=(0,),
        embed_kv_group_indices=(0,),
    )
    assert split.decode.finished_req_ids == {"d1"}
    assert split.embed.finished_req_ids == {"e1"}
    # After finished, mapping is cleared so KV churn isn't carried into the
    # next step.
    assert req_id_to_model_id == {}


def test_merge_outputs_preserves_order_and_routes_payloads():
    scheduled_req_order = ["d1", "e1", "d2"]
    decode_out = ModelRunnerOutput(
        req_ids=["d1", "d2"],
        req_id_to_index={"d1": 0, "d2": 1},
        sampled_token_ids=[[42], [43]],
    )
    embed_out = ModelRunnerOutput(
        req_ids=["e1"],
        req_id_to_index={"e1": 0},
        pooler_output=[None],
    )
    merged = merge_model_runner_outputs(scheduled_req_order, decode_out, embed_out)
    assert merged.req_ids == scheduled_req_order
    assert merged.req_id_to_index == {"d1": 0, "e1": 1, "d2": 2}
    assert merged.sampled_token_ids == [[42], [], [43]]
    assert merged.pooler_output == [None, None, None]


def test_merge_outputs_handles_empty_decode_side():
    scheduled_req_order = ["e1"]
    embed_out = ModelRunnerOutput(
        req_ids=["e1"],
        req_id_to_index={"e1": 0},
        pooler_output=[None],
    )
    merged = merge_model_runner_outputs(scheduled_req_order, None, embed_out)
    assert merged.req_ids == ["e1"]
    assert merged.sampled_token_ids == [[]]


def test_merge_outputs_handles_empty_embed_side():
    scheduled_req_order = ["d1"]
    decode_out = ModelRunnerOutput(
        req_ids=["d1"],
        req_id_to_index={"d1": 0},
        sampled_token_ids=[[7]],
    )
    merged = merge_model_runner_outputs(scheduled_req_order, decode_out, None)
    assert merged.req_ids == ["d1"]
    assert merged.sampled_token_ids == [[7]]

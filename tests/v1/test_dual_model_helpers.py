# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.dual_model_helpers import (
    merge_model_runner_outputs,
    split_scheduler_output_by_model,
)


def test_split_scheduler_output_by_model():
    scheduler_output = SchedulerOutput.make_empty()
    scheduler_output.num_common_prefix_blocks = [3, 4, 5]
    scheduler_output.scheduled_new_reqs = [
        NewRequestData(
            req_id="decode-1",
            model_id="decode",
            prompt_token_ids=[1],
            mm_features=[],
            sampling_params=None,
            pooling_params=None,
            block_ids=([10], [11], [12]),
            num_computed_tokens=0,
            lora_request=None,
            prefill_token_ids=[1],
        ),
        NewRequestData(
            req_id="embed-1",
            model_id="embed",
            prompt_token_ids=[2],
            mm_features=[],
            sampling_params=None,
            pooling_params=None,
            block_ids=([20], [21], [22]),
            num_computed_tokens=0,
            lora_request=None,
            prefill_token_ids=[2],
        ),
    ]
    scheduler_output.scheduled_cached_reqs = CachedRequestData(
        req_ids=["decode-2", "embed-2"],
        resumed_req_ids={"decode-2"},
        new_token_ids=[[11], [22]],
        all_token_ids={"decode-2": [1, 11], "embed-2": [2, 22]},
        new_block_ids=[(([30], [31], [32])), None],
        num_computed_tokens=[1, 2],
        num_output_tokens=[0, 0],
    )
    scheduler_output.num_scheduled_tokens = {
        "decode-1": 3,
        "embed-1": 4,
        "decode-2": 5,
        "embed-2": 6,
    }
    scheduler_output.total_num_scheduled_tokens = 18
    scheduler_output.finished_req_ids = {"embed-finished"}

    req_id_to_model_id = {
        "decode-2": "decode",
        "embed-2": "embed",
        "embed-finished": "embed",
    }
    split = split_scheduler_output_by_model(
        scheduler_output,
        req_id_to_model_id=req_id_to_model_id,
        decode_model_id="decode",
        embed_model_id="embed",
        decode_kv_group_indices=(0, 2),
        embed_kv_group_indices=(1,),
    )

    assert split.decode.scheduled_new_reqs[0].req_id == "decode-1"
    assert split.decode.scheduled_new_reqs[0].block_ids == ([10], [12])
    assert split.embed.scheduled_new_reqs[0].req_id == "embed-1"
    assert split.embed.scheduled_new_reqs[0].block_ids == ([21],)
    assert split.decode.scheduled_cached_reqs.req_ids == ["decode-2"]
    assert split.decode.scheduled_cached_reqs.new_block_ids == [([30], [32])]
    assert split.embed.scheduled_cached_reqs.req_ids == ["embed-2"]
    assert split.embed.scheduled_cached_reqs.new_block_ids == [None]
    assert split.decode.num_scheduled_tokens == {"decode-1": 3, "decode-2": 5}
    assert split.embed.num_scheduled_tokens == {"embed-1": 4, "embed-2": 6}
    assert split.decode.num_common_prefix_blocks == [3, 5]
    assert split.embed.num_common_prefix_blocks == [4]
    assert split.embed.finished_req_ids == {"embed-finished"}
    assert "embed-finished" not in req_id_to_model_id


def test_merge_model_runner_outputs():
    decode_output = ModelRunnerOutput(
        req_ids=["decode-1"],
        req_id_to_index={"decode-1": 0},
        sampled_token_ids=[[101, 102]],
    )
    embed_output = ModelRunnerOutput(
        req_ids=["embed-1"],
        req_id_to_index={"embed-1": 0},
        pooler_output=["embedding"],  # type: ignore[list-item]
    )
    merged = merge_model_runner_outputs(
        scheduled_req_order=["decode-1", "embed-1"],
        decode_output=decode_output,
        embed_output=embed_output,
    )

    assert merged.req_ids == ["decode-1", "embed-1"]
    assert merged.sampled_token_ids == [[101, 102], []]
    assert merged.pooler_output == [None, "embedding"]


def test_split_scheduler_output_handles_non_pp_cached_request_shape():
    scheduler_output = SchedulerOutput.make_empty()
    scheduler_output.scheduled_cached_reqs = CachedRequestData(
        req_ids=["decode-1", "embed-1"],
        resumed_req_ids=set(),
        new_token_ids=[],
        all_token_ids={},
        new_block_ids=[([10], [11]), ([20], [21])],
        num_computed_tokens=[2, 3],
        num_output_tokens=[1, 0],
    )
    scheduler_output.num_scheduled_tokens = {"decode-1": 1, "embed-1": 1}
    scheduler_output.total_num_scheduled_tokens = 2
    scheduler_output.num_common_prefix_blocks = [0, 0]

    split = split_scheduler_output_by_model(
        scheduler_output,
        req_id_to_model_id={"decode-1": "decode", "embed-1": "embed"},
        decode_model_id="decode",
        embed_model_id="embed",
        decode_kv_group_indices=(0,),
        embed_kv_group_indices=(1,),
    )

    assert split.decode.scheduled_cached_reqs.req_ids == ["decode-1"]
    assert split.decode.scheduled_cached_reqs.new_token_ids == []
    assert split.decode.scheduled_cached_reqs.new_block_ids == [([10],)]
    assert split.embed.scheduled_cached_reqs.req_ids == ["embed-1"]
    assert split.embed.scheduled_cached_reqs.new_token_ids == []
    assert split.embed.scheduled_cached_reqs.new_block_ids == [([21],)]


def test_split_scheduler_output_handles_empty_common_prefix_blocks():
    scheduler_output = SchedulerOutput.make_empty()
    scheduler_output.finished_req_ids = {"decode-1", "embed-1"}

    split = split_scheduler_output_by_model(
        scheduler_output,
        req_id_to_model_id={"decode-1": "decode", "embed-1": "embed"},
        decode_model_id="decode",
        embed_model_id="embed",
        decode_kv_group_indices=(0,),
        embed_kv_group_indices=(1,),
    )

    assert split.decode.num_common_prefix_blocks == []
    assert split.embed.num_common_prefix_blocks == []
    assert split.decode.finished_req_ids == {"decode-1"}
    assert split.embed.finished_req_ids == {"embed-1"}

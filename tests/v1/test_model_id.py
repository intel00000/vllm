# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.engine import EngineCoreRequest, infer_model_id
from vllm.v1.request import Request


def test_infer_model_id_defaults():
    assert infer_model_id(SamplingParams(max_tokens=8), None) == "decode"
    assert infer_model_id(None, PoolingParams(task="embed")) == "embed"
    assert infer_model_id(
        None,
        PoolingParams(task="score"),
        explicit_model_id="rerank",
    ) == "rerank"


def test_request_and_scheduler_payload_preserve_model_id():
    request = EngineCoreRequest(
        request_id="req-1",
        prompt_token_ids=[1, 2, 3],
        mm_features=None,
        sampling_params=None,
        pooling_params=PoolingParams(task="embed"),
        model_id="embed_worker",
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )
    runtime_request = Request.from_engine_core_request(request, block_hasher=None)
    new_req_data = NewRequestData.from_request(runtime_request, block_ids=([],))

    assert runtime_request.model_id == "embed_worker"
    assert new_req_data.model_id == "embed_worker"


def test_request_defaults_decode_model_id():
    request = EngineCoreRequest(
        request_id="req-2",
        prompt_token_ids=[1, 2],
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=4),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )
    runtime_request = Request.from_engine_core_request(request, block_hasher=None)
    new_req_data = NewRequestData.from_request(runtime_request, block_ids=([],))

    assert request.resolved_model_id == "decode"
    assert runtime_request.model_id == "decode"
    assert new_req_data.model_id == "decode"

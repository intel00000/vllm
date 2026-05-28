# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import concurrent.futures
import copy
import inspect
import os
from contextlib import contextmanager
from dataclasses import is_dataclass
from typing import Any

import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.core.sched.output import GrammarOutput
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput
from vllm.v1.worker.dual_model_helpers import (
    DualModelConfig,
    merge_model_runner_outputs,
    split_scheduler_output_by_model,
)
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

logger = init_logger(__name__)


@contextmanager
def _nvtx_range(name: str):
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


_ASYNC_OUTPUT_STREAM_ATTRS = ("async_output_copy_stream", "output_copy_stream")


@contextmanager
def _temporary_async_outputs(runner: GPUModelRunner, enabled: bool):
    """Lazy-init the async copy stream + prepare_inputs event on a runner.

    GPUModelRunner only creates these at __init__ when
    scheduler_config.async_scheduling is True, but DualModelRunner forces
    async scheduling off, so they're None here. Without this, the
    Async*ModelRunnerOutput constructors crash on `with torch.cuda.stream(None)`.

    The attribute name has changed between vllm versions
    (`async_output_copy_stream` vs `output_copy_stream`) — we patch whichever
    one the runner exposes.
    """
    old_value = runner.use_async_scheduling
    if enabled:
        for attr in _ASYNC_OUTPUT_STREAM_ATTRS:
            if hasattr(runner, attr) and getattr(runner, attr) is None:
                setattr(runner, attr, torch.cuda.Stream())
        if (
            hasattr(runner, "prepare_inputs_event")
            and getattr(runner, "prepare_inputs_event") is None
        ):
            runner.prepare_inputs_event = torch.Event()
        runner.use_async_scheduling = True
    try:
        yield
    finally:
        runner.use_async_scheduling = old_value


def _resolve_model_runner_output(
    output: ModelRunnerOutput | AsyncModelRunnerOutput | None,
    label: str,
) -> ModelRunnerOutput | None:
    if isinstance(output, AsyncModelRunnerOutput):
        with _nvtx_range(label):
            return output.get_output()
    return output


def _create_disjoint_green_streams(
    device: torch.device,
    device_index: int,
    decode_sms_target: int,
    embed_sms_target: int,
    total_sms: int,
) -> tuple[torch.cuda.Stream, torch.cuda.Stream, int, int, list[Any]]:
    """Create two CUDA Green Context streams with disjoint SM partitions.

    Uses ``flashinfer.green_ctx`` building blocks (``split_resource`` +
    ``create_green_ctx_streams``) under a *peel-then-remainder* policy so
    that the embed partition exactly fits whatever SMs are left after the
    decode peel. This avoids ``flashinfer.green_ctx.split_device_green_ctx_by_sm_count``,
    which rounds **each** requested count up independently to the 8-SM
    alignment — e.g. ``[40, 92]`` becomes ``[40, 96]`` and 40+96 > 132
    fails with ``CUDA_ERROR_INVALID_RESOURCE_CONFIGURATION``.

    PyTorch's bare ``torch.cuda.GreenContext.create(num_sms)`` does NOT
    coordinate disjoint allocation across calls — two consecutive calls
    can yield green contexts whose SM resources overlap, producing a
    deferred ``vectorized_gather_kernel`` index-out-of-bounds assert
    when used concurrently. The peel-then-remainder split below is
    guaranteed disjoint by the driver's split semantics.

    ``embed_sms_target`` is informational only — embed gets exactly what
    is left after decode is peeled.

    Returns:
        decode_stream:     torch.cuda.Stream on the decode SM partition.
        embed_stream:      torch.cuda.Stream on the embed SM partition.
        actual_decode_sms: SM count actually realized for decode.
        actual_embed_sms:  SM count realized for embed.
        green_ctx_handles: kept for API compatibility; flashinfer manages
            green-ctx lifetime via the returned streams, so this is empty.
    """
    try:
        from flashinfer.green_ctx import (
            create_green_ctx_streams,
            get_cudevice,
            get_device_resource,
            split_resource,
        )
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "flashinfer is required for dual-model green-context partitioning. "
            "Install with `pip install flashinfer-python`."
        ) from e

    cu_dev = get_cudevice(device)
    primary = get_device_resource(cu_dev)
    decode_list, remaining = split_resource(primary, 1, decode_sms_target)
    if not decode_list or decode_list[0].sm.smCount == 0:
        raise RuntimeError(
            f"Could not peel off {decode_sms_target} SMs for decode partition "
            f"from primary resource of {primary.sm.smCount} SMs"
        )
    decode_res = decode_list[0]
    if remaining.sm.smCount == 0:
        raise RuntimeError(
            f"No SMs remaining for embed partition after taking "
            f"{decode_res.sm.smCount} for decode"
        )

    streams = create_green_ctx_streams(cu_dev, [decode_res, remaining])
    decode_stream = streams[0]
    embed_stream = streams[1]
    actual_decode_sms = int(decode_res.sm.smCount)
    actual_embed_sms = int(remaining.sm.smCount)
    if actual_decode_sms + actual_embed_sms > total_sms:
        raise RuntimeError(
            f"Green-context partitions overlap: decode={actual_decode_sms} + "
            f"embed={actual_embed_sms} > total {total_sms}"
        )
    return (
        decode_stream,
        embed_stream,
        actual_decode_sms,
        actual_embed_sms,
        [],  # flashinfer manages green-ctx lifetime via the streams
    )


def _destroy_green_contexts(handles: list[Any]) -> None:
    """Cleanup hook — flashinfer manages green-ctx lifetime via the streams,
    so the handles list is empty under the new implementation. Kept for
    backwards-compatible call sites."""
    if not handles:
        return
    try:
        from cuda.bindings import driver as drv
    except ImportError:
        return
    for ctx in handles:
        try:
            drv.cuGreenCtxDestroy(ctx)
        except Exception:
            pass


class DualModelRunner:
    """Experimental wrapper that hosts one decode and one embed runner."""

    EMBED_MODEL_PREFIX = "embed_model"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        if not envs.VLLM_USE_V2_MODEL_RUNNER:
            raise ValueError("DualModelRunner currently requires V2 model runner.")
        if vllm_config.scheduler_config.async_scheduling:
            raise ValueError(
                "DualModelRunner currently does not support async scheduling."
            )
        if vllm_config.parallel_config.pipeline_parallel_size != 1:
            raise ValueError(
                "DualModelRunner currently requires pipeline_parallel_size=1."
            )
        if vllm_config.speculative_config is not None:
            raise ValueError(
                "DualModelRunner currently does not support speculative decoding."
            )
        # Full-GPU streams (used in solo phases or when SM partitioning is off).
        self._decode_stream_full = torch.cuda.default_stream(device)
        stream_mode = os.environ.get("VLLM_DUAL_MODEL_STREAM_MODE", "two_stream")
        if stream_mode == "default_stream":
            self._embed_stream_full = self._decode_stream_full
        elif stream_mode == "two_stream":
            self._embed_stream_full = torch.cuda.Stream(device)
        else:
            raise ValueError(
                "VLLM_DUAL_MODEL_STREAM_MODE must be 'two_stream' or "
                f"'default_stream', got {stream_mode!r}."
            )

        # Optional CUDA Green Context partitioning: when both
        # VLLM_DUAL_DECODE_SMS and VLLM_DUAL_EMBED_SMS are set, create
        # *disjoint* green contexts per role and route mixed-step kernels
        # through them. In solo steps (only decode or only embed has work)
        # we fall back to the full-GPU streams so we don't waste SMs.
        #
        # NOTE: We deliberately do NOT use ``torch.cuda.GreenContext.create``
        # here. PyTorch's wrapper calls ``cuGreenCtxCreate`` per-invocation
        # with no cross-call coordination, so two separate calls can return
        # green contexts that share underlying SMs. When both contexts are
        # active concurrently (mixed phase), kernels on different streams
        # end up scheduled on the same physical SMs, racing in shared
        # memory / L2 / register state. This manifests as a deferred
        # ``vectorized_gather_kernel`` index-out-of-bounds assert at full
        # scale — the kernel reads stale tensor data corrupted by the race.
        # Instead, we use the CUDA Driver API directly via ``cuda-python``
        # to perform a single split of the device's SM resource, then
        # combine sub-resources to build two *guaranteed-disjoint*
        # descriptors before creating each green context.
        self._decode_stream_partitioned = None
        self._embed_stream_partitioned = None
        self._partition_sm_decode = None
        self._partition_sm_embed = None
        self._green_ctx_handles: list[Any] = []  # for cleanup
        sm_decode_env = os.environ.get("VLLM_DUAL_DECODE_SMS", "").strip()
        sm_embed_env = os.environ.get("VLLM_DUAL_EMBED_SMS", "").strip()
        if sm_decode_env and sm_embed_env:
            sm_decode = int(sm_decode_env)
            sm_embed = int(sm_embed_env)
            total_sms = torch.cuda.get_device_properties(device).multi_processor_count
            if sm_decode <= 0 or sm_decode > total_sms:
                raise ValueError(
                    f"VLLM_DUAL_DECODE_SMS={sm_decode} out of range (1..{total_sms})"
                )
            if sm_embed <= 0 or sm_embed > total_sms:
                raise ValueError(
                    f"VLLM_DUAL_EMBED_SMS={sm_embed} out of range (1..{total_sms})"
                )
            device_index = device.index if device.index is not None else 0
            (
                self._decode_stream_partitioned,
                self._embed_stream_partitioned,
                actual_decode_sms,
                actual_embed_sms,
                self._green_ctx_handles,
            ) = _create_disjoint_green_streams(
                device, device_index, sm_decode, sm_embed, total_sms
            )
            self._partition_sm_decode = actual_decode_sms
            self._partition_sm_embed = actual_embed_sms
            logger.info(
                "DualModelRunner green-ctx partition: decode=%d/%d SMs (requested %d), "
                "embed=%d/%d SMs (requested %d) — disjoint via cuDevSmResourceSplitByCount",
                actual_decode_sms, total_sms, sm_decode,
                actual_embed_sms, total_sms, sm_embed,
            )

        # Initial stream binding: default to full streams. Mixed steps swap
        # to the partitioned streams inside execute_model when partitioning
        # is enabled.
        self.decode_stream = self._decode_stream_full
        self.embed_stream = self._embed_stream_full

        execute_order = os.environ.get("VLLM_DUAL_MODEL_EXECUTE_ORDER", "decode_first")
        if execute_order not in ("decode_first", "embed_first"):
            raise ValueError(
                "VLLM_DUAL_MODEL_EXECUTE_ORDER must be 'decode_first' or "
                f"'embed_first', got {execute_order!r}."
            )
        self.execute_order = execute_order
        self.async_outputs = os.environ.get(
            "VLLM_DUAL_MODEL_ASYNC_OUTPUTS", "0"
        ).lower() in ("1", "true", "yes", "on")

        dual_cfg = DualModelConfig.from_vllm_config(vllm_config)
        if dual_cfg is None:
            raise ValueError(
                "DualModelRunner requires additional_config['dual_model']."
            )

        self.vllm_config = vllm_config
        self.dual_cfg = dual_cfg
        self.device = device
        self.decode_runner = GPUModelRunner(vllm_config, device)
        self.embed_vllm_config = self._build_embed_vllm_config(vllm_config, dual_cfg)
        self.embed_runner = GPUModelRunner(self.embed_vllm_config, device)

        self.model_memory_usage = 0
        self.is_pooling_model = False
        self.model: nn.Module | None = None
        self.req_id_to_model_id: dict[str, str] = {}
        self.pending_embed_output: ModelRunnerOutput | None = None
        self.pending_embed_needs_pool = False
        self.pending_req_order: list[str] = []
        self.decode_kv_group_indices: tuple[int, ...] = ()
        self.embed_kv_group_indices: tuple[int, ...] = ()
        self.decode_kv_cache_spec: dict[str, Any] = {}
        self.embed_kv_cache_spec: dict[str, Any] = {}

        # Per-role single-worker thread pools. Each pool keeps exactly
        # ONE long-lived Python thread, so each role's CUDA work always
        # runs on the same thread. That matters because cuBLAS handles
        # are keyed by (thread, device): pinning the role to its own
        # thread lets cuBLAS bind its handle to the role's stream
        # (default OR green-context partition) once, and keep it bound
        # without context-switching state on every step. Without this,
        # alternating green-ctx streams on a single thread trips
        # CUBLAS_STATUS_EXECUTION_FAILED at scale.
        self._decode_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="dual_decode"
        )
        self._embed_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="dual_embed"
        )
        # Warm each thread: set device + issue a tiny matmul under the
        # role's CURRENT stream (full or partitioned). This forces
        # torch to lazy-init the per-thread cuBLAS handle now, bound
        # to the role's stream, before any real forward call.
        self._warm_role_thread(
            self._decode_executor, self.decode_stream, self.device
        )
        self._warm_role_thread(
            self._embed_executor, self.embed_stream, self.device
        )

    @staticmethod
    def _warm_role_thread(
        executor: concurrent.futures.ThreadPoolExecutor,
        stream: torch.cuda.Stream,
        device: torch.device,
    ) -> None:
        """Touch CUDA on the executor's worker thread so cuBLAS creates
        its per-thread handle now (bound to ``stream``)."""
        def _warm():
            torch.cuda.set_device(device)
            with torch.cuda.stream(stream):
                a = torch.zeros((2, 2), device=device, dtype=torch.bfloat16)
                b = torch.zeros((2, 2), device=device, dtype=torch.bfloat16)
                _ = torch.matmul(a, b)
                torch.cuda.current_stream().synchronize()
        executor.submit(_warm).result()

    def __del__(self):
        # Best-effort: shut down the per-role executors. Non-blocking
        # so we don't deadlock on tasks that might be in flight at
        # interpreter shutdown.
        for attr in ("_decode_executor", "_embed_executor"):
            executor = getattr(self, attr, None)
            if executor is not None:
                try:
                    executor.shutdown(wait=False)
                except Exception:
                    pass
        # Best-effort: free green-context handles. Streams created from
        # them are torch.cuda.ExternalStream (no automatic cleanup), so
        # the contexts must outlive any stream use; this is called only
        # when the runner itself goes away.
        handles = getattr(self, "_green_ctx_handles", None)
        if handles:
            _destroy_green_contexts(handles)

    def __getattr__(self, name: str) -> Any:
        decode_runner = self.__dict__.get("decode_runner")
        if decode_runner is not None:
            return getattr(decode_runner, name)
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")

    @staticmethod
    def _replace_dataclass(instance: Any, **updates: Any) -> Any:
        cls = type(instance)
        if not is_dataclass(cls):
            raise TypeError(f"{cls.__name__} is not a dataclass.")
        init_kwargs: dict[str, Any] = {}
        for name, parameter in inspect.signature(cls).parameters.items():
            if name == "self":
                continue
            if name in updates:
                init_kwargs[name] = updates.pop(name)
                continue
            if not hasattr(instance, name):
                if parameter.default is inspect._empty:
                    raise ValueError(
                        f"Missing required constructor argument {name!r} for "
                        f"{cls.__name__}."
                    )
                continue
            value = getattr(instance, name)
            # Pydantic dataclass quirk (v0.21.0+ vllm.config.ModelConfig):
            # InitVar fields like mm_tensor_ipc don't persist as
            # meaningful attributes on the instance -- they surface as
            # None. Copying None back through cls(**kwargs) then fails
            # the InitVar's Literal validation. Drop None copies from
            # the source; the new instance's own default path
            # (MultiModalConfig.mm_tensor_ipc = "direct_rpc") handles
            # them. The caller's `updates` dict overrides this -- it's
            # already applied above and never reaches this branch.
            if value is None:
                continue
            init_kwargs[name] = value
        init_kwargs.update(updates)
        return cls(**init_kwargs)

    def _build_embed_vllm_config(
        self,
        vllm_config: VllmConfig,
        dual_cfg: DualModelConfig,
    ) -> VllmConfig:
        # The embed runner must own an independent config tree so model-load
        # side effects such as static_forward_context registration do not alias
        # the decode runner's state.
        embed_vllm_config = copy.deepcopy(vllm_config)
        embed_model_config = self._replace_dataclass(
            embed_vllm_config.model_config,
            model=dual_cfg.embed_model,
            tokenizer=dual_cfg.embed_tokenizer or dual_cfg.embed_model,
            runner="pooling",
            served_model_name=dual_cfg.embed_model,
            dtype=dual_cfg.embed_dtype or embed_vllm_config.model_config.dtype,
            max_model_len=(
                dual_cfg.embed_max_model_len
                or embed_vllm_config.model_config.max_model_len
            ),
            enforce_eager=(
                dual_cfg.embed_enforce_eager
                if dual_cfg.embed_enforce_eager is not None
                else embed_vllm_config.model_config.enforce_eager
            ),
            model_weights="",
            hf_config_path=None,
        )
        scheduler_overrides: dict[str, Any] = dict(
            max_model_len=embed_model_config.max_model_len,
            is_encoder_decoder=embed_model_config.is_encoder_decoder,
            runner_type="pooling",
        )
        if dual_cfg.embed_enable_chunked_prefill is not None:
            scheduler_overrides["enable_chunked_prefill"] = (
                dual_cfg.embed_enable_chunked_prefill
            )
        embed_scheduler_config = self._replace_dataclass(
            embed_vllm_config.scheduler_config,
            **scheduler_overrides,
        )
        embed_vllm_config.model_config = embed_model_config
        embed_vllm_config.scheduler_config = embed_scheduler_config
        embed_vllm_config.additional_config = {}
        embed_vllm_config.compilation_config.static_forward_context.clear()
        embed_vllm_config.compilation_config.static_all_moe_layers.clear()
        return embed_vllm_config

    def update_max_model_len(self, max_model_len: int) -> None:
        self.decode_runner.update_max_model_len(max_model_len)
        self.embed_runner.update_max_model_len(max_model_len)

    @staticmethod
    def _project_kv_cache_config(
        kv_cache_config: KVCacheConfig,
        runner_kv_cache_spec: dict[str, Any],
    ) -> tuple[KVCacheConfig, tuple[int, ...]]:
        runner_layer_names = set(runner_kv_cache_spec)
        projected_groups: list[KVCacheGroupSpec] = []
        group_indices: list[int] = []
        for group_idx, group in enumerate(kv_cache_config.kv_cache_groups):
            runner_group_layers = [
                layer_name
                for layer_name in group.layer_names
                if layer_name in runner_layer_names
            ]
            if not runner_group_layers:
                continue
            group_spec = group.kv_cache_spec
            if isinstance(group_spec, UniformTypeKVCacheSpecs):
                group_spec = UniformTypeKVCacheSpecs(
                    block_size=group_spec.block_size,
                    kv_cache_specs={
                        layer_name: group_spec.kv_cache_specs[layer_name]
                        for layer_name in runner_group_layers
                    },
                )
            projected_groups.append(
                KVCacheGroupSpec(
                    layer_names=runner_group_layers,
                    kv_cache_spec=group_spec,
                )
            )
            group_indices.append(group_idx)

        projected_tensors = [
            KVCacheTensor(
                size=kv_cache_tensor.size,
                shared_by=[
                    layer_name
                    for layer_name in kv_cache_tensor.shared_by
                    if layer_name in runner_layer_names
                ],
            )
            for kv_cache_tensor in kv_cache_config.kv_cache_tensors
            if any(
                layer_name in runner_layer_names
                for layer_name in kv_cache_tensor.shared_by
            )
        ]
        return (
            KVCacheConfig(
                num_blocks=kv_cache_config.num_blocks,
                kv_cache_tensors=projected_tensors,
                kv_cache_groups=projected_groups,
            ),
            tuple(group_indices),
        )

    def get_supported_tasks(self) -> tuple[str, ...]:
        tasks = list(self.decode_runner.get_supported_tasks())
        for task in self.embed_runner.get_supported_tasks():
            if task not in tasks:
                tasks.append(task)
        return tuple(tasks)

    def load_model(self, *args, **kwargs) -> None:
        self.decode_runner.load_model(*args, **kwargs)
        self.embed_runner.load_model(
            *args,
            prefix=self.EMBED_MODEL_PREFIX,
            **kwargs,
        )
        self.decode_kv_cache_spec = self.decode_runner.get_kv_cache_spec()
        self.embed_kv_cache_spec = self.embed_runner.get_kv_cache_spec()
        overlap = set(self.decode_kv_cache_spec) & set(self.embed_kv_cache_spec)
        if overlap:
            raise ValueError(
                "Decode/embed KV layer names must be disjoint; overlapping names: "
                f"{sorted(overlap)[:4]}"
            )
        self.model_memory_usage = (
            self.decode_runner.model_memory_usage + self.embed_runner.model_memory_usage
        )
        self.model = self.decode_runner.model

    def get_model(self) -> nn.Module:
        assert self.model is not None
        return self.model

    def get_kv_cache_spec(self):
        return {
            **self.decode_kv_cache_spec,
            **self.embed_kv_cache_spec,
        }

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        decode_kv_cache_config, self.decode_kv_group_indices = (
            self._project_kv_cache_config(kv_cache_config, self.decode_kv_cache_spec)
        )
        embed_kv_cache_config, self.embed_kv_group_indices = (
            self._project_kv_cache_config(kv_cache_config, self.embed_kv_cache_spec)
        )
        self.decode_runner.initialize_kv_cache(decode_kv_cache_config)
        self.embed_runner.initialize_kv_cache(embed_kv_cache_config)

    def profile_run(self) -> None:
        self.decode_runner.profile_run()
        self.embed_runner.profile_run()

    def reset_mm_cache(self) -> None:
        self.decode_runner.reset_mm_cache()
        self.embed_runner.reset_mm_cache()

    def reset_encoder_cache(self) -> None:
        self.decode_runner.reset_encoder_cache()
        self.embed_runner.reset_encoder_cache()

    def profile_cudagraph_memory(self) -> int:
        return (
            self.decode_runner.profile_cudagraph_memory()
            + self.embed_runner.profile_cudagraph_memory()
        )

    def capture_model(self) -> int:
        # Dispatch each capture to the role's dedicated executor thread.
        # Critical for green-context: the per-thread cuBLAS handle is what
        # gets recorded into the captured graphs (graph_capture itself uses
        # an internal stream, but the cuBLAS handle is thread-local). Later
        # execute_model also runs on the role's executor thread (see T3),
        # so the capturing and replaying threads -- and their cuBLAS state --
        # match. Mismatched threads here would resurface as
        # CUBLAS_STATUS_EXECUTION_FAILED at first replay.
        decode_future = self._decode_executor.submit(
            self.decode_runner.capture_model
        )
        embed_future = self._embed_executor.submit(
            self.embed_runner.capture_model
        )
        return decode_future.result() + embed_future.result()

    def _dummy_run(self, *args, **kwargs):
        return self.decode_runner._dummy_run(*args, **kwargs)

    def _dummy_sampler_run(self, *args, **kwargs) -> None:
        self.decode_runner._dummy_sampler_run(*args, **kwargs)

    def _dummy_pooler_run(self, *args, **kwargs) -> None:
        self.embed_runner._dummy_pooler_run(*args, **kwargs)

    def execute_model(
        self,
        scheduler_output,
        intermediate_tensors=None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
    ):
        if dummy_run:
            self.decode_runner.execute_model(
                scheduler_output,
                intermediate_tensors=intermediate_tensors,
                dummy_run=True,
                skip_attn_for_dummy_run=skip_attn_for_dummy_run,
            )
            self.embed_runner.profile_run()
            return None

        split_outputs = split_scheduler_output_by_model(
            scheduler_output=scheduler_output,
            req_id_to_model_id=self.req_id_to_model_id,
            decode_model_id=self.dual_cfg.decode_model_id,
            embed_model_id=self.dual_cfg.embed_model_id,
            decode_kv_group_indices=self.decode_kv_group_indices,
            embed_kv_group_indices=self.embed_kv_group_indices,
        )
        self.pending_req_order = split_outputs.scheduled_req_order

        def run_decode() -> ModelRunnerOutput | None:
            with torch.cuda.stream(self.decode_stream):
                with _nvtx_range("decode_execute"):
                    return self.decode_runner.execute_model(
                        split_outputs.decode,
                        intermediate_tensors=intermediate_tensors,
                    )

        def run_embed() -> ModelRunnerOutput | None:
            with torch.cuda.stream(self.embed_stream):
                with _nvtx_range("embed_execute"):
                    return self.embed_runner.execute_model(
                        split_outputs.embed,
                    )

        has_decode_work = (
            split_outputs.decode.num_scheduled_tokens
            or split_outputs.decode.finished_req_ids
        )
        has_embed_work = (
            split_outputs.embed.num_scheduled_tokens
            or split_outputs.embed.finished_req_ids
        )

        # SM-partition stream selection: if green contexts are configured,
        # use the partitioned streams when both roles have work in this step
        # so kernels run on disjoint SM groups; otherwise use full-GPU streams.
        if self._decode_stream_partitioned is not None and (
            has_decode_work and has_embed_work
        ):
            self.decode_stream = self._decode_stream_partitioned
            self.embed_stream = self._embed_stream_partitioned
        else:
            self.decode_stream = self._decode_stream_full
            self.embed_stream = self._embed_stream_full

        decode_output = None
        embed_exec_output = None
        embed_output: ModelRunnerOutput | None = None
        # Dispatch each role to its dedicated executor so cuBLAS's
        # per-thread handle stays pinned to the role's stream. Both
        # tasks run concurrently on the GPU when both have work.
        # execute_order only affects which task is submitted first --
        # CPU-side ordering, not GPU scheduling.
        if self.execute_order == "embed_first":
            embed_future = (
                self._embed_executor.submit(run_embed) if has_embed_work else None
            )
            decode_future = (
                self._decode_executor.submit(run_decode) if has_decode_work else None
            )
        else:
            decode_future = (
                self._decode_executor.submit(run_decode) if has_decode_work else None
            )
            embed_future = (
                self._embed_executor.submit(run_embed) if has_embed_work else None
            )
        if decode_future is not None:
            decode_output = decode_future.result()
        if embed_future is not None:
            embed_exec_output = embed_future.result()

        decode_needs_sample = decode_output is None and bool(
            split_outputs.decode.num_scheduled_tokens
        )
        if decode_needs_sample:
            self.pending_embed_output = embed_exec_output
            self.pending_embed_needs_pool = (
                split_outputs.embed.num_scheduled_tokens
                or split_outputs.embed.finished_req_ids
            ) and embed_exec_output is None
            return None

        if has_embed_work:
            if embed_exec_output is None:
                # Pool() reads tensors produced by the embed forward on
                # _embed_executor's thread and issues its own CUDA work.
                # Run it on the same thread so cuBLAS / stream state stays
                # consistent.
                def _run_pool():
                    with torch.cuda.stream(self.embed_stream):
                        with _nvtx_range("embed_pool"):
                            with _temporary_async_outputs(
                                self.embed_runner, self.async_outputs
                            ):
                                return self.embed_runner.pool()
                embed_pool_output = self._embed_executor.submit(_run_pool).result()
                embed_output = _resolve_model_runner_output(
                    embed_pool_output,
                    "embed_output_get",
                )
            else:
                embed_output = embed_exec_output
            if embed_output is None:
                raise RuntimeError("Embed runner failed to produce pooling output.")

        self.pending_embed_output = None
        self.pending_embed_needs_pool = False
        with _nvtx_range("merge_outputs"):
            return merge_model_runner_outputs(
                scheduled_req_order=self.pending_req_order,
                decode_output=decode_output,
                embed_output=embed_output,
            )

    def sample_tokens(self, grammar_output: GrammarOutput | None):
        # Run sampling on the decode role's dedicated executor thread,
        # which keeps cuBLAS / stream state aligned with the preceding
        # run_decode forward. Pool likewise runs on the embed thread.
        # See the long comment at the top of execute_model for why the
        # per-role single-worker executor pinning matters under green
        # contexts.
        def _run_sample():
            with torch.cuda.stream(self.decode_stream):
                with _nvtx_range("decode_sample"):
                    with _temporary_async_outputs(
                        self.decode_runner, self.async_outputs
                    ):
                        return self.decode_runner.sample_tokens(grammar_output)
        decode_sample_future = self._decode_executor.submit(_run_sample)

        embed_pool_future = None
        if self.pending_embed_needs_pool:
            def _run_pool():
                with torch.cuda.stream(self.embed_stream):
                    with _nvtx_range("embed_pool"):
                        with _temporary_async_outputs(
                            self.embed_runner, self.async_outputs
                        ):
                            return self.embed_runner.pool()
            embed_pool_future = self._embed_executor.submit(_run_pool)

        decode_sample_output = decode_sample_future.result()
        embed_pool_output = (
            embed_pool_future.result() if embed_pool_future is not None else None
        )
        embed_output = self.pending_embed_output
        decode_output = _resolve_model_runner_output(
            decode_sample_output,
            "decode_output_get",
        )
        if self.pending_embed_needs_pool:
            embed_output = _resolve_model_runner_output(
                embed_pool_output,
                "embed_output_get",
            )
            if embed_output is None:
                raise RuntimeError("Embed runner failed to produce pooling output.")
        with _nvtx_range("merge_outputs"):
            merged = merge_model_runner_outputs(
                scheduled_req_order=self.pending_req_order,
                decode_output=decode_output,
                embed_output=embed_output,
            )
        self.pending_embed_output = None
        self.pending_embed_needs_pool = False
        self.pending_req_order = []
        return merged

    def take_draft_token_ids(self):
        return self.decode_runner.take_draft_token_ids()

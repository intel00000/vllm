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
from vllm.v1.worker.work_stream import WorkStream

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

        # WorkStreams are populated when the worker calls
        # ``set_work_streams``.  They are the single source of stream
        # identity for both roles -- ``execute_model`` / ``sample_tokens``
        # require them to be set.  ``Worker.init_device`` wires them
        # automatically after constructing the runner; direct callers
        # must call ``set_work_streams`` themselves before executing.
        self._decode_ws: WorkStream | None = None
        self._embed_ws: WorkStream | None = None
        # The WorkStream actually used THIS step per role: the partitioned ws
        # when both roles have work, else the role's full-SM fallback. Set at
        # the top of execute_model; _decode_ctx/_embed_ctx follow it.
        self._decode_ws_active: WorkStream | None = None
        self._embed_ws_active: WorkStream | None = None
        # A/B toggle: VLLM_DUAL_SOLO_FALLBACK=0 disables the solo-step full-SM
        # fallback (partitioned stream on every step = old regressed behavior),
        # so we can isolate the fix's effect. Default on.
        self._solo_full_fallback = (
            os.environ.get("VLLM_DUAL_SOLO_FALLBACK", "1") != "0")
        self.decode_stream: torch.cuda.Stream | None = None
        self.embed_stream: torch.cuda.Stream | None = None

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
        # Sequential baseline for the latency experiment: run one role per step,
        # GPU-serialized (never decode ∥ embed). When set, execute_model runs
        # the two role forwards back-to-back with a device-side wait between
        # them so their kernels do not co-reside. This is the "no overlap" arm
        # against which co-location (default) is compared. See
        # notes/ttft_latency_measurement_plan.md.
        self.serialize_models = os.environ.get(
            "VLLM_DUAL_MODEL_SERIALIZE", "0"
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

        # Per-role single-worker thread pools.  Each pool keeps exactly
        # ONE long-lived Python thread, so each role's CUDA work always
        # runs on the same thread.  That matters because cuBLAS handles
        # are keyed by (thread, device): pinning the role to its own
        # thread lets cuBLAS bind its handle to the role's stream once
        # and keep it bound without context-switching state on every
        # step.  Without this, alternating green-ctx streams on a
        # single thread trips CUBLAS_STATUS_EXECUTION_FAILED at scale.
        # Each executor is warmed under its role's WorkStream context
        # in :meth:`set_work_streams` (which the worker calls during
        # init_device).
        self._decode_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="dual_decode"
        )
        self._embed_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="dual_embed"
        )

    @staticmethod
    def _warm_role_thread(
        executor: concurrent.futures.ThreadPoolExecutor,
        work_stream: WorkStream,
        device: torch.device,
    ) -> None:
        """Touch CUDA on the executor's worker thread under
        ``work_stream.context()`` so cuBLAS creates its per-thread handle
        bound to the WorkStream's stream, and any backend-specific
        per-thread state (e.g. libsmctrl's TPC mask) gets installed on
        this executor's thread before the first real forward."""
        def _warm():
            torch.cuda.set_device(device)
            with work_stream.context():
                a = torch.zeros((2, 2), device=device, dtype=torch.bfloat16)
                b = torch.zeros((2, 2), device=device, dtype=torch.bfloat16)
                _ = torch.matmul(a, b)
                torch.cuda.current_stream().synchronize()
        executor.submit(_warm).result()

    def bind_main_stream(self, stream: torch.cuda.Stream) -> None:
        """Preempt the decode runner's ``main_stream`` cache.

        Called from ``Worker.init_device`` with the primary WorkStream's
        stream so the decode-side async D2H paths sync correctly.  The
        embed runner gets its own ``main_stream`` binding inside
        :meth:`set_work_streams`.
        """
        self.decode_runner.bind_main_stream(stream)

    def set_work_streams(
        self,
        decode_ws: WorkStream,
        embed_ws: WorkStream,
    ) -> None:
        """Wire decode + embed WorkStream(s) from the worker.

        Overrides the legacy full / partitioned stream-swap logic in
        ``execute_model``; from this point on, role work always runs in
        the corresponding WorkStream's context.  Re-warms each role
        executor thread under the new context so cuBLAS rebinds its
        per-thread handle to the WorkStream's stream (and any
        backend-specific per-thread state -- libsmctrl mask -- gets
        installed before the first real forward).
        """
        self._decode_ws = decode_ws
        self._embed_ws = embed_ws
        # Active ws defaults to the (partitioned) ws; execute_model swaps to the
        # full-SM fallback on solo steps.
        self._decode_ws_active = decode_ws
        self._embed_ws_active = embed_ws
        self.decode_stream = decode_ws.stream
        self.embed_stream = embed_ws.stream
        self.decode_runner.bind_main_stream(decode_ws.stream)
        self.embed_runner.bind_main_stream(embed_ws.stream)
        # Warm cuBLAS / per-thread state on each executor thread under
        # the role's WorkStream context -- both the partitioned ws AND its
        # full-SM fallback, since a role thread runs on either across steps.
        self._warm_role_thread(self._decode_executor, decode_ws, self.device)
        self._warm_role_thread(self._embed_executor, embed_ws, self.device)
        if decode_ws.full_fallback is not None:
            self._warm_role_thread(
                self._decode_executor, decode_ws.full_fallback, self.device)
        if embed_ws.full_fallback is not None:
            self._warm_role_thread(
                self._embed_executor, embed_ws.full_fallback, self.device)

    def _select_active_ws(self, has_decode_work: bool, has_embed_work: bool) -> None:
        """Pick each role's WorkStream for this step: the partitioned ws only
        when BOTH roles have work (so their kernels run on disjoint SM groups);
        otherwise the lone role runs on its full-SM fallback. No-op for
        backend=none (full_fallback is None -> stays on the full-SM ws).

        On a stream SWITCH this is where the original vllm raced
        (cudaErrorIllegalAddress). Two barriers make the swap safe:
          1. ``new.stream.wait_stream(old.stream)`` -- the new stream waits for
             all work already queued on the old stream (this role's KV-cache
             writes from prior steps) before its forward reads/writes KV, so a
             read can't run ahead of the previous write on the other stream.
          2. ``runner.bind_main_stream(new.stream)`` -- rebind the runner's
             cached ``main_stream`` so async-output D2H gates on the stream the
             forward actually ran on (a stale main_stream is exactly the
             illegal-address failure documented in bind_main_stream). Cheap
             (a dict assignment)."""
        both = bool(has_decode_work) and bool(has_embed_work)
        if not self._solo_full_fallback:
            both = True  # A/B off: force partitioned every step (old behavior)
        d, e = self._decode_ws, self._embed_ws
        new_d = (d if (both or d is None or d.full_fallback is None)
                 else d.full_fallback)
        new_e = (e if (both or e is None or e.full_fallback is None)
                 else e.full_fallback)
        prev_d, prev_e = self._decode_ws_active, self._embed_ws_active
        if (new_d is not None and prev_d is not None
                and new_d.stream is not prev_d.stream):
            new_d.stream.wait_stream(prev_d.stream)
            self.decode_runner.bind_main_stream(new_d.stream)
        if (new_e is not None and prev_e is not None
                and new_e.stream is not prev_e.stream):
            new_e.stream.wait_stream(prev_e.stream)
            self.embed_runner.bind_main_stream(new_e.stream)
        self._decode_ws_active = new_d
        self._embed_ws_active = new_e
        if new_d is not None:
            self.decode_stream = new_d.stream
        if new_e is not None:
            self.embed_stream = new_e.stream

    def _decode_ctx(self):
        """Context manager for decode-role CUDA work, routed through the
        decode WorkStream installed by ``set_work_streams``."""
        if self._decode_ws is None:
            raise RuntimeError(
                "DualModelRunner: set_work_streams() must be called "
                "before execute_model() / sample_tokens().  "
                "Worker.init_device wires this automatically; if you're "
                "constructing DualModelRunner directly, build "
                "WorkStreams via make_work_streams() and pass them in."
            )
        return (self._decode_ws_active or self._decode_ws).context()

    def _embed_ctx(self):
        if self._embed_ws is None:
            raise RuntimeError(
                "DualModelRunner: set_work_streams() must be called "
                "before execute_model() / sample_tokens()."
            )
        return (self._embed_ws_active or self._embed_ws).context()

    def __del__(self):
        # Best-effort: shut down the per-role executors. Non-blocking
        # so we don't deadlock on tasks that might be in flight at
        # interpreter shutdown.  WorkStream lifetime is managed by the
        # worker (Worker.shutdown -> WorkStream.close()).
        for attr in ("_decode_executor", "_embed_executor"):
            executor = getattr(self, attr, None)
            if executor is not None:
                try:
                    executor.shutdown(wait=False)
                except Exception:
                    pass

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

    @classmethod
    def _paired_layer_map(cls, kv_cache_config: KVCacheConfig) -> dict[str, str]:
        """From an overlaid config (tensors ``shared_by=[decode_layer,
        embed_layer]``) return ``{embed_layer: decode_layer}``. Empty when the
        KV overlay did not run (each tensor is shared by a single layer)."""
        mapping: dict[str, str] = {}
        for t in kv_cache_config.kv_cache_tensors:
            decode_layers = [
                ln for ln in t.shared_by if cls.EMBED_MODEL_PREFIX not in ln
            ]
            embed_layers = [
                ln for ln in t.shared_by if cls.EMBED_MODEL_PREFIX in ln
            ]
            if len(decode_layers) == 1 and len(embed_layers) == 1:
                mapping[embed_layers[0]] = decode_layers[0]
        return mapping

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        decode_kv_cache_config, self.decode_kv_group_indices = (
            self._project_kv_cache_config(kv_cache_config, self.decode_kv_cache_spec)
        )
        embed_kv_cache_config, self.embed_kv_group_indices = (
            self._project_kv_cache_config(kv_cache_config, self.embed_kv_cache_spec)
        )
        # When the KV overlay paired decode-layer-i with embed-layer-i onto
        # shared tensors, the projected decode/embed configs each claim the FULL
        # (doubled) num_blocks -- allocating both independently would need 2x the
        # memory. Instead allocate the decode buffers, then overlay the embed
        # model's layers onto the same physical buffers. Safe: the shared block
        # pool gives any physical block to exactly one model at a time.
        embed_to_decode_layer = self._paired_layer_map(kv_cache_config)
        self.decode_runner.initialize_kv_cache(decode_kv_cache_config)
        external_raw = None
        if embed_to_decode_layer:
            decode_raw = self.decode_runner.kv_cache_raw_tensors
            external_raw = {
                embed_layer: decode_raw[decode_layer]
                for embed_layer, decode_layer in embed_to_decode_layer.items()
                if decode_layer in decode_raw
            }
            if len(external_raw) != len(embed_to_decode_layer):
                # Incomplete pairing -> fall back to independent allocation.
                logger.warning(
                    "Dual-model KV overlay: only %d/%d embed layers mapped to "
                    "decode buffers; falling back to independent allocation.",
                    len(external_raw),
                    len(embed_to_decode_layer),
                )
                external_raw = None
        self.embed_runner.initialize_kv_cache(
            embed_kv_cache_config, external_raw_tensors=external_raw
        )

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

        # Each task records a CUDA event on its role's stream after the
        # forward is enqueued. The main thread waits on those events
        # before reading outputs -- see the comment in sample_tokens
        # for the full rationale (torch's caching allocator does NOT
        # track external green-ctx streams, so explicit event sync is
        # required to avoid stale reads / illegal memory accesses on
        # the main thread's subsequent operations).
        def run_decode() -> tuple[ModelRunnerOutput | None, torch.cuda.Event]:
            with self._decode_ctx():
                with _nvtx_range("decode_execute"):
                    out = self.decode_runner.execute_model(
                        split_outputs.decode,
                        intermediate_tensors=intermediate_tensors,
                    )
                ev = torch.cuda.Event()
                ev.record(self.decode_stream)
            return out, ev

        def run_embed() -> tuple[ModelRunnerOutput | None, torch.cuda.Event]:
            with self._embed_ctx():
                with _nvtx_range("embed_execute"):
                    out = self.embed_runner.execute_model(
                        split_outputs.embed,
                    )
                ev = torch.cuda.Event()
                ev.record(self.embed_stream)
            return out, ev

        has_decode_work = (
            split_outputs.decode.num_scheduled_tokens
            or split_outputs.decode.finished_req_ids
        )
        has_embed_work = (
            split_outputs.embed.num_scheduled_tokens
            or split_outputs.embed.finished_req_ids
        )

        # Choose each role's active stream (partitioned only when BOTH have
        # work; else full-SM fallback) BEFORE dispatch. run_decode/run_embed and
        # the embed pool below all route through _decode_ctx/_embed_ctx and
        # self.{decode,embed}_stream, which now follow the active selection.
        self._select_active_ws(bool(has_decode_work), bool(has_embed_work))

        decode_output = None
        embed_exec_output = None
        embed_output: ModelRunnerOutput | None = None
        # Dispatch each role to its dedicated executor so cuBLAS's
        # per-thread handle stays pinned to the role's stream. Both
        # tasks run concurrently on the GPU when both have work.
        # execute_order only affects which task is submitted first --
        # CPU-side ordering, not GPU scheduling.
        decode_event: torch.cuda.Event | None = None
        embed_event: torch.cuda.Event | None = None
        if self.serialize_models:
            # Sequential arm: run the two role forwards ONE AT A TIME, GPU-
            # serialized. .result() only waits for the executor thread to finish
            # enqueuing, so we synchronize the role's CUDA event before starting
            # the other -- otherwise the first role's kernels would still be on
            # the GPU when the second role's launch, defeating "no overlap".
            order = (("embed", "decode") if self.execute_order == "embed_first"
                     else ("decode", "embed"))
            for role in order:
                if role == "decode" and has_decode_work:
                    decode_output, decode_event = (
                        self._decode_executor.submit(run_decode).result()
                    )
                    if decode_event is not None:
                        decode_event.synchronize()
                elif role == "embed" and has_embed_work:
                    embed_exec_output, embed_event = (
                        self._embed_executor.submit(run_embed).result()
                    )
                    if embed_event is not None:
                        embed_event.synchronize()
        else:
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
                decode_output, decode_event = decode_future.result()
            if embed_future is not None:
                embed_exec_output, embed_event = embed_future.result()

        # Cross-stream sync: have the main thread's current stream wait
        # for both role streams before any subsequent host or
        # default-stream read of the outputs.
        _main_stream = torch.cuda.current_stream()
        if decode_event is not None:
            _main_stream.wait_event(decode_event)
        if embed_event is not None:
            _main_stream.wait_event(embed_event)

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
                # consistent. Record an event after the pool and have
                # the main stream wait on it before _resolve reads the
                # output (same rationale as the decode/embed forward
                # event sync above).
                def _run_pool():
                    with self._embed_ctx():
                        with _nvtx_range("embed_pool"):
                            with _temporary_async_outputs(
                                self.embed_runner, self.async_outputs
                            ):
                                out = self.embed_runner.pool()
                        ev = torch.cuda.Event()
                        ev.record(self.embed_stream)
                    return out, ev
                embed_pool_output, embed_pool_event = (
                    self._embed_executor.submit(_run_pool).result()
                )
                torch.cuda.current_stream().wait_event(embed_pool_event)
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
        # Each role's executor task records a CUDA event on its stream
        # after the work is enqueued, and returns (output, event). After
        # we collect both futures we explicitly wait for those events on
        # the current (main-thread) stream before reading the outputs.
        # This is the cross-stream sync the OLD heterobatch tree had at
        # commit 6f1d6321c -- torch's caching allocator does NOT track
        # external (green-ctx) streams, so without these waits the main
        # thread can read tensors that decode_stream/embed_stream have
        # not finished writing. Symptoms ranged from
        # vectorized_gather_kernel index-out-of-bounds (sampling reads
        # stale logits) to cudaErrorIllegalAddress inside the async D2H
        # copy of sampled token ids.
        def _run_sample():
            with self._decode_ctx():
                with _nvtx_range("decode_sample"):
                    with _temporary_async_outputs(
                        self.decode_runner, self.async_outputs
                    ):
                        out = self.decode_runner.sample_tokens(grammar_output)
                ev = torch.cuda.Event()
                ev.record(self.decode_stream)
            return out, ev
        decode_sample_future = self._decode_executor.submit(_run_sample)

        embed_pool_future = None
        if self.pending_embed_needs_pool:
            def _run_pool():
                with self._embed_ctx():
                    with _nvtx_range("embed_pool"):
                        with _temporary_async_outputs(
                            self.embed_runner, self.async_outputs
                        ):
                            out = self.embed_runner.pool()
                    ev = torch.cuda.Event()
                    ev.record(self.embed_stream)
                return out, ev
            embed_pool_future = self._embed_executor.submit(_run_pool)

        decode_sample_output, decode_sample_event = decode_sample_future.result()
        if embed_pool_future is not None:
            embed_pool_output, embed_pool_event = embed_pool_future.result()
        else:
            embed_pool_output, embed_pool_event = None, None

        # Make the current (main-thread default) stream wait for both
        # role streams to complete before we proceed to resolve / merge
        # outputs. Without this, downstream reads -- including the
        # GPUModelRunner's internal async D2H of sampled token ids --
        # may race against still-pending decode/embed work.
        _main_stream = torch.cuda.current_stream()
        _main_stream.wait_event(decode_sample_event)
        if embed_pool_event is not None:
            _main_stream.wait_event(embed_pool_event)

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

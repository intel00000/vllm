# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-lane execution state for the gen model.

- PRIVATE per lane: InputBuffers instance, block-table/slot-mapping forward
  scratch, execute_model_state handoff slot, main_stream identity, D2H copy
  stream + event, completion/staging/prep events.
- SHARED single-writer: RequestState, canonical BlockTables staged tensors,
  KV tensors, sampler tables, model weights. The CPU state-update +
  apply_staged_writes phase stays on one thread.

The decode lane deliberately owns the BASE runner's persistent buffers --
cudagraph capture bakes those addresses. Prefill gets fresh scratch and must
never touch the captured tensors (never call get_dummy_*).
"""

from __future__ import annotations

import os
import threading
from contextlib import nullcontext
from dataclasses import dataclass, field

import torch

from vllm.config import CompilationMode, CUDAGraphMode, VllmConfig
from vllm.logger import init_logger
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.buffer_utils import fence_uva_pools
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.attn_utils import set_lane_attn_builder_idx
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.vllm_flash_attn.flash_attn_interface import lane_sm_margin

logger = init_logger(__name__)

LANE_DECODE = "decode"
LANE_PREFILL = "prefill"
GEN_LANES = (LANE_DECODE, LANE_PREFILL)


class LaneAttentionScratch:
    """Private forward block-tables + slot-mappings for a non-decode lane.
    """

    def __init__(self, logical: BlockTables):
        self.input_block_tables = [
            torch.zeros_like(table.gpu) for table in logical.block_tables
        ]
        self.input_block_table_ptrs = logical._make_ptr_tensor(
            self.input_block_tables
        )
        self.slot_mappings = torch.zeros(
            logical.num_kv_cache_groups,
            logical.max_num_batched_tokens,
            dtype=torch.int64,
            device=logical.block_tables[0].gpu.device,
        )

    def prepare(
        self, logical: BlockTables, input_batch: InputBatch
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        block_tables = logical.gather_block_tables(
            input_batch.idx_mapping,
            num_reqs_padded=input_batch.num_reqs_after_padding,
            out=tuple(self.input_block_tables),
            out_ptrs=self.input_block_table_ptrs,
        )
        slot_mappings = logical.compute_slot_mappings(
            input_batch.idx_mapping,
            input_batch.query_start_loc,
            input_batch.positions,
            num_tokens_padded=input_batch.num_tokens_after_padding,
            out=self.slot_mappings,
        )
        return block_tables, slot_mappings


@dataclass
class LaneContext:
    """Everything one lane must own privately to coexist with another lane."""

    input_buffers: InputBuffers
    completion_event: torch.cuda.Event = field(default_factory=torch.cuda.Event)
    staging_event: torch.cuda.Event = field(default_factory=torch.cuda.Event)
    prep_event: torch.cuda.Event = field(default_factory=torch.cuda.Event)
    copy_stream: torch.cuda.Stream | None = None
    copy_event: torch.cuda.Event | None = None
    # Stream the lane's forward ran on, graph captured at execute time (S5c).
    main_stream: torch.cuda.Stream | None = None
    # Forward -> sample handoff; the base runner's single slot would be
    # clobbered by the other lane.
    execute_model_state: object | None = None
    # Prefill-only private forward tensors; built after initialize_kv_cache.
    attn_scratch: LaneAttentionScratch | None = None


class LaneModelRunner(GPUModelRunner):
    """V2 runner + per-lane execution contexts for the gen model.

    Decode keeps every base persistent buffer (graph addresses); prefill gets
    a private InputBuffers + LaneAttentionScratch + copy stream. Constructed
    by DualModelRunner for the gen role when lanes are enabled; without lanes
    the plain GPUModelRunner is used and this class never loads.
    """

    # S8: one attention-metadata builder per gen lane (decode=0, prefill=1),
    # via the ubatch multi-builder mechanism. Isolates persistent builder
    # state that the pre-forward lock release would otherwise share across
    # concurrently executing lanes (FlashInfer index buffers/wrappers/float
    # workspace; FA3 scheduler tensor).
    lane_attn_builder_count = 2

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        # v0.26's module-global WorkspaceManager keys its slot by the DBO
        # thread registry, which maps unregistered threads to slot 0, and two
        # lanes would silently alias MoE/DCP/sparse-attn workspace.
        hf_config = vllm_config.model_config.hf_config
        num_experts = getattr(hf_config, "num_experts", None) or getattr(
            hf_config, "num_local_experts", None
        )
        dcp_size = getattr(
            vllm_config.parallel_config, "decode_context_parallel_size", 1
        )
        if num_experts or (dcp_size and dcp_size > 1):
            raise ValueError(
                "LaneModelRunner does not support MoE or DCP models "
                "(shared WorkspaceManager slot across lanes)."
            )
        # Graph/compile exclusivity: captured graph buffers and the
        # compiled callable's runtime buffers are single-owner. With
        # HB_P2_DECODE_GRAPHS_ONLY=1 the decode lane owns both: prefill
        # tickets are forced eager at dispatch and (with HB_P2_COMPILE_DECODE)
        # bypass the compiled callable -- which also keeps prefill GEMMs on
        # UnquantizedLinearMethod.apply, where the cuBLASLt SM-cap hook fires.
        self._graphs_only = os.environ.get("HB_P2_DECODE_GRAPHS_ONLY") == "1"
        self._compile_decode = os.environ.get("HB_P2_COMPILE_DECODE") == "1"
        # S9a: both lanes execute the compiled artifact. The artifact is
        # re-entrant (inductor output allocates per call; DBO drives it from
        # two threads); the single-owner hazard was the dispatcher, which
        # this mode removes structurally: the bytecode hook (process-global
        # __code__ swap per forward) must be OFF, and wrapper.py pins the
        # process-global dynamo knobs once instead of per call.
        self._compile_both = os.environ.get("HB_P2_COMPILE_BOTH") == "1"
        if self._compile_both:
            import vllm.envs as envs_mod
            if envs_mod.VLLM_USE_BYTECODE_HOOK:
                raise ValueError(
                    "HB_P2_COMPILE_BOTH requires VLLM_USE_BYTECODE_HOOK=0 "
                    "(the bytecode hook swaps cls.forward.__code__ process-"
                    "wide per forward -- unsafe with two lane threads)."
                )
            if envs_mod.VLLM_USE_AOT_COMPILE:
                raise ValueError(
                    "HB_P2_COMPILE_BOTH is incompatible with "
                    "VLLM_USE_AOT_COMPILE (process-global partition wrapper)."
                )
            if vllm_config.compilation_config.cudagraph_copy_inputs:
                raise ValueError(
                    "HB_P2_COMPILE_BOTH requires cudagraph_copy_inputs=False "
                    "(shared static input buffers in the compiled path)."
                )
        if self._compile_decode and not self._graphs_only:
            raise ValueError(
                "HB_P2_COMPILE_DECODE requires HB_P2_DECODE_GRAPHS_ONLY=1."
            )
        cg_mode = vllm_config.compilation_config.cudagraph_mode
        if (
            cg_mode is not None
            and cg_mode != CUDAGraphMode.NONE
            and not self._graphs_only
        ):
            raise ValueError(
                "VLLM_DUAL_LANES with cudagraphs requires "
                f"HB_P2_DECODE_GRAPHS_ONLY=1 (got cudagraph_mode {cg_mode})."
            )
        comp_mode = vllm_config.compilation_config.mode
        if (
            comp_mode is not None
            and comp_mode != CompilationMode.NONE
            and not (self._compile_decode or self._compile_both)
        ):
            # Without COMPILE_BOTH the dispatcher is single-owner; decode
            # must own the callable exclusively (COMPILE_DECODE) or lanes
            # must stay uncompiled.
            raise ValueError(
                "VLLM_DUAL_LANES with compilation requires "
                "HB_P2_COMPILE_DECODE=1 or HB_P2_COMPILE_BOTH=1."
            )
        super().__init__(vllm_config, device)
        self.lane_contexts: dict[str, LaneContext] = {
            LANE_DECODE: LaneContext(
                input_buffers=self.input_buffers,
                copy_stream=self.output_copy_stream,
            ),
            LANE_PREFILL: LaneContext(
                input_buffers=InputBuffers(
                    max_num_reqs=self.max_num_reqs,
                    max_num_tokens=self.max_num_tokens,
                    device=self.device,
                ),
                copy_stream=torch.cuda.Stream(self.device),
                copy_event=torch.cuda.Event(),
            ),
        }
        # Serializes the shared-host state-update + prep phase across lanes
        # (base execute_model calls the acquire/release seams); re-taken for
        # sample_tokens' shared-mutation tail. Recreates the single-host-
        # thread invariant the slackserve design has by construction.
        self._gen_state_lock = threading.Lock()
        # Per-thread active lane: each lane's ticket runs execute+sample on
        # its own pinned executor thread.
        self._tl = threading.local()
        self.fa_sm_margin = {
            LANE_DECODE: int(os.environ.get("HB_P2_FA_SM_MARGIN_DECODE", "0") or 0),
            LANE_PREFILL: int(
                os.environ.get("HB_P2_FA_SM_MARGIN_PREFILL", "0") or 0
            ),
        }
        logger.info(
            "LaneModelRunner: %s lanes (decode = base buffers, prefill = "
            "private scratch)",
            list(self.lane_contexts),
        )

    def _tl_context(self) -> LaneContext | None:
        lane = getattr(self._tl, "lane", None)
        return self.lane_contexts.get(lane) if lane is not None else None

    def _lane_force_eager(self) -> bool:
        return (
            self._graphs_only
            and getattr(self._tl, "lane", None) == LANE_PREFILL
        )

    def _lane_skip_compiled(self) -> bool:
        # S9a: under COMPILE_BOTH the prefill lane enters the compiled
        # artifact too (it still never runs graphs -- _lane_force_eager
        # keeps its cudagraph_runtime_mode NONE).
        if self._compile_both:
            return False
        return (
            self._compile_decode
            and getattr(self._tl, "lane", None) == LANE_PREFILL
        )

    def _fa3_builders(self):
        builders = {}
        for groups in getattr(self, "attn_groups", []) or []:
            for group in groups:
                builder = group.get_metadata_builder(0)
                if getattr(builder, "use_full_cuda_graph", False):
                    builders[id(builder)] = builder
        return list(builders.values())

    def _host_prep_lock_acquire(self) -> None:
        self._gen_state_lock.acquire()
        self._tl.lock_held = True
        context = self._tl_context()
        if context is not None:
            # Redirect the prepare path onto this lane's buffers. Safe: only
            # ever mutated with the lock held; decode's swap is the identity.
            self.input_buffers = context.input_buffers
            # S8: select this lane's private attention-metadata builder
            # (decode=0, prefill=1). Backends with persistent builder state
            # (FlashInfer paged_kv_* buffers + wrappers + float workspace,
            # FA3 scheduler tensor) are lane-isolated by construction; the
            # forward reads only refs captured into the built metadata.
            set_lane_attn_builder_idx(
                0 if getattr(self._tl, "lane", None) == LANE_DECODE else 1
            )
            if not getattr(self, "_s8_isolation_checked", False):
                self._s8_isolation_checked = True
                self._assert_lane_builder_isolation()
        if self._lane_force_eager():
            # FA3's builder owns ONE persistent scheduler-metadata tensor when
            # full-graph support is on; an eager prefill metadata build would
            # overwrite it while a decode graph reads it. Flip the builders to
            # the fresh-allocation path for this lane's build. Lock-scoped:
            # the metadata build happens inside the locked prep, so decode
            # can never observe the flipped value during its own build.
            flipped = self._fa3_builders()
            for builder in flipped:
                builder.use_full_cuda_graph = False
            self._tl.fa3_flipped = flipped

    def _assert_lane_builder_isolation(self) -> None:
        """One-time S8 invariant: per-lane builders share NO persistent GPU
        state. Structural proof of memory safety -- token-level checks cannot
        distinguish a race from FlashInfer's composition-dependent numerics.
        """
        for groups in getattr(self, "attn_groups", []) or []:
            for group in groups:
                if len(group.metadata_builders) < 2:
                    continue
                b0, b1 = group.metadata_builders[0], group.metadata_builders[1]
                for attr in ("paged_kv_indptr", "paged_kv_indices",
                             "paged_kv_last_page_len"):
                    t0, t1 = getattr(b0, attr, None), getattr(b1, attr, None)
                    if t0 is not None and t1 is not None:
                        assert t0.gpu.data_ptr() != t1.gpu.data_ptr(), (
                            "S8 violation: shared builder buffer", attr)
                w0 = getattr(b0, "_workspace_buffer", None)
                w1 = getattr(b1, "_workspace_buffer", None)
                if w0 is not None and w1 is not None:
                    assert w0.data_ptr() != w1.data_ptr(), (
                        "S8 violation: shared float workspace")

    def _host_prep_lock_release(self) -> None:
        # Idempotent: called at the pre-forward seam on the happy path and
        # from execute_model's finally for early returns and exceptions.
        if not getattr(self._tl, "lock_held", False):
            return
        try:
            for builder in getattr(self._tl, "fa3_flipped", ()):
                builder.use_full_cuda_graph = True
            self._tl.fa3_flipped = ()
            if self._tl_context() is not None:
                # Every staged-transport consumer for this ticket is enqueued
                # on the current (lane) stream; arm buffer-reuse events.
                fence_uva_pools(torch.cuda.current_stream(self.device))
                self.input_buffers = self.lane_contexts[
                    LANE_DECODE
                ].input_buffers
        finally:
            self._tl.lock_held = False
            self._gen_state_lock.release()

    def _stash_execute_state(self, state, dummy_run: bool = False) -> None:
        context = self._tl_context()
        if context is None:
            self.execute_model_state = state
            return
        context.execute_model_state = state
        if dummy_run:
            # Profiling warmup reads the shared slot directly; the following
            # dummy sampler run consumes and clears it.
            self.execute_model_state = state

    def execute_model(self, scheduler_output, *args, **kwargs):
        lane = self.lane_for(scheduler_output)
        context = self.lane_contexts[lane]
        self._tl.lane = lane
        margin = self.fa_sm_margin.get(lane, 0)
        margin_ctx = lane_sm_margin(margin) if margin > 0 else nullcontext()
        try:
            with margin_ctx:
                result = super().execute_model(scheduler_output, *args, **kwargs)
        finally:
            self._host_prep_lock_release()
        context.main_stream = torch.cuda.current_stream(self.device)
        context.completion_event.record(context.main_stream)
        return result

    def sample_tokens(self, grammar_output):
        context = self._tl_context()
        if context is None:
            return super().sample_tokens(grammar_output)
        # Whole-body lock: sampling reads shared sampler tables and its
        # post_update mutates them.
        with self._gen_state_lock:
            prev_main = self.__dict__.get("main_stream")
            prev_copy = self.output_copy_stream
            if context.main_stream is not None:
                self.__dict__["main_stream"] = context.main_stream
                torch.cuda.current_stream(self.device).wait_event(
                    context.completion_event
                )
            if context.copy_stream is not None:
                self.output_copy_stream = context.copy_stream
            self.execute_model_state = context.execute_model_state
            context.execute_model_state = None
            try:
                result = super().sample_tokens(grammar_output)
            finally:
                if prev_main is not None:
                    self.__dict__["main_stream"] = prev_main
                self.output_copy_stream = prev_copy
            # Re-record so controller readiness queries cover the full
            # ticket (forward + sampling + postprocess).
            context.completion_event.record(
                torch.cuda.current_stream(self.device)
            )
            return result

    def prepare_attn(self, input_batch: InputBatch):
        context = self._tl_context()
        if context is None or context.attn_scratch is None:
            # Decode (or pre-KV-init dummy): base path, persistent buffers.
            return super().prepare_attn(input_batch)
        return context.attn_scratch.prepare(self.block_tables, input_batch)

    @staticmethod
    def lane_for(scheduler_output) -> str:
        lane = getattr(scheduler_output, "execution_lane", "default")
        return LANE_DECODE if lane == "default" else lane

    def initialize_kv_cache(self, *args, **kwargs):
        result = super().initialize_kv_cache(*args, **kwargs)
        for lane, context in self.lane_contexts.items():
            if lane != LANE_DECODE:
                context.attn_scratch = LaneAttentionScratch(self.block_tables)
        return result

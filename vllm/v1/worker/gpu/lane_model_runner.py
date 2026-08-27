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


def rebind_custom_ops_to_cuda(model: torch.nn.Module, spec: str) -> list[str]:
    """Rebind the CustomOp instances named in ``spec`` to forward_cuda."""
    from vllm.model_executor.custom_op import CustomOp

    names = {s.strip().lstrip("+") for s in spec.split(",") if s.strip()}
    rebound: list[str] = []
    for module in model.modules():
        if isinstance(module, CustomOp) and getattr(module, "name", None) in names:
            module._forward_method = module.forward_cuda
            rebound.append(type(module).__name__)
    if not rebound:
        raise RuntimeError(
            f"HB_P2_PREFILL_CUSTOM_OPS matched no modules: {sorted(names)}"
        )
    return rebound


class LaneAttentionScratch:
    """Private forward block-tables + slot-mappings for a non-decode lane."""

    def __init__(self, logical: BlockTables):
        self.input_block_tables = [
            torch.zeros_like(table.gpu) for table in logical.block_tables
        ]
        self.input_block_table_ptrs = logical._make_ptr_tensor(self.input_block_tables)
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
    # HB_P2_PREP_STREAM: dedicated stream for the ticket-head prep window
    # (staged-write scatters through input/attention preparation).
    prep_stream: torch.cuda.Stream | None = None


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
        # S9c-P1: sampling runs lock-free; only the shared tail
        # (postprocess_sampled) and a buffer fence take the lock.
        self._sample_narrow = os.environ.get("HB_P2_SAMPLE_NARROW") == "1"
        # S9c-P2: release the lock right after the input batch is built;
        # attention build + forward read only captured refs / per-lane or
        # row-disjoint state (see notes/s9c design, hazard table).
        self._prep_early_release = os.environ.get("HB_P2_PREP_EARLY_RELEASE") == "1"
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
        self._prefill_custom_ops = os.environ.get("HB_P2_PREFILL_CUSTOM_OPS", "")
        if self._prefill_custom_ops and self._compile_both:
            raise ValueError(
                "HB_P2_PREFILL_CUSTOM_OPS has no reachable dispatch path "
                "under HB_P2_COMPILE_BOTH=1: the prefill lane executes the "
                "compiled artifact, whose custom-op choices were baked in at "
                "trace time (inductor-fused under custom_ops 'none', opaque "
                "torch.ops._C calls when enabled) -- the CustomOp instances' "
                "_forward_method is never consulted by compiled code, so the "
                "rebind would silently change nothing. Use "
                "HB_P2_COMPILE_DECODE=1 (skip-compiled eager prefill) or an "
                "uncompiled-lanes config."
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
        # HB_P2_PREP_STREAM=1 (slackserve port, Theo d8224a4a5/52a25c079):
        # enqueue the ticket-head prep window -- staged-write scatters
        # through input/attention preparation -- on a dedicated PER-LANE
        # prep stream, and make the lane's execution stream wait on a
        # prep-done event before the forward. Prep kernels then never queue
        # behind a busy execution stream: the staged-buffer fence events
        # (fence_uva_pools) complete promptly, so one lane's ticket head no
        # longer serializes behind the other lane's in-flight forward. The
        # stream must be per lane: prep-time allocations (attention
        # metadata) free into the allocating stream's pool as soon as host
        # refs drop, and a shared prep stream would let one lane's next
        # prep reuse memory the other lane's forward is still reading.
        # Same-lane reuse is safe: the controller keeps one in-flight
        # ticket per lane and host-resolves its output (D2H copy event)
        # before dispatching the lane's next ticket.
        self._prep_stream_enabled = os.environ.get("HB_P2_PREP_STREAM") == "1"
        if self._prep_stream_enabled:
            for lane_context in self.lane_contexts.values():
                lane_context.prep_stream = torch.cuda.Stream(self.device)
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
            LANE_PREFILL: int(os.environ.get("HB_P2_FA_SM_MARGIN_PREFILL", "0") or 0),
        }
        logger.info(
            "LaneModelRunner: %s lanes (decode = base buffers, prefill = "
            "private scratch)",
            list(self.lane_contexts),
        )

    def _tl_context(self) -> LaneContext | None:
        lane = getattr(self._tl, "lane", None)
        return self.lane_contexts.get(lane) if lane is not None else None

    # S9c-P1: the sampling path's stream/state funnel attrs are routed to the
    # calling lane's context instead of swapped under the lock (same shape as
    # the S8 builder selection). Off-lane accesses (init, warmup, profiling)
    # fall through to the instance slots.

    @property
    def main_stream(self):
        ctx = self._tl_context() if hasattr(self, "_tl") else None
        if ctx is not None and ctx.main_stream is not None:
            return ctx.main_stream
        return self.__dict__.get("main_stream")

    @main_stream.setter
    def main_stream(self, value):
        ctx = self._tl_context() if hasattr(self, "_tl") else None
        if ctx is not None:
            ctx.main_stream = value
        else:
            self.__dict__["main_stream"] = value

    @property
    def output_copy_stream(self):
        ctx = self._tl_context() if hasattr(self, "_tl") else None
        if ctx is not None and ctx.copy_stream is not None:
            return ctx.copy_stream
        return self.__dict__.get("output_copy_stream")

    @output_copy_stream.setter
    def output_copy_stream(self, value):
        ctx = self._tl_context() if hasattr(self, "_tl") else None
        if ctx is not None:
            ctx.copy_stream = value
        else:
            self.__dict__["output_copy_stream"] = value

    @property
    def execute_model_state(self):
        ctx = self._tl_context() if hasattr(self, "_tl") else None
        if ctx is not None:
            return ctx.execute_model_state
        return self.__dict__.get("execute_model_state")

    @execute_model_state.setter
    def execute_model_state(self, value):
        ctx = self._tl_context() if hasattr(self, "_tl") else None
        if ctx is not None:
            ctx.execute_model_state = value
        else:
            self.__dict__["execute_model_state"] = value

    def _sampler_tail_ctx(self):
        # Narrow mode: postprocess_sampled mutates shared sampler state and
        # request rows -- the one part of sampling that still needs the lock.
        # Whole-body mode already holds it (non-reentrant): nullcontext.
        if self._sample_narrow and self._tl_context() is not None:
            return self._gen_state_lock
        return nullcontext()

    def _lane_force_eager(self) -> bool:
        return self._graphs_only and getattr(self._tl, "lane", None) == LANE_PREFILL

    def _lane_skip_compiled(self) -> bool:
        # S9a: under COMPILE_BOTH the prefill lane enters the compiled
        # artifact too (it still never runs graphs -- _lane_force_eager
        # keeps its cudagraph_runtime_mode NONE).
        if self._compile_both:
            return False
        return self._compile_decode and getattr(self._tl, "lane", None) == LANE_PREFILL

    def _apply_prefill_custom_ops(self) -> None:
        """HB_P2_PREFILL_CUSTOM_OPS: post-capture per-lane custom-op rebind.

        Intended use: serve with the listed ops fused (custom_ops
        ``["none"]``) so decode's FULL graphs are frozen over the
        inductor-fused stock kernels at capture time, then rebind the
        listed CustomOp instances of the gen model to forward_cuda. The
        rebind swaps the instance-level ``_forward_method``, which only
        EAGER module calls consult; compiled code baked its op choices in
        at trace time. Reachable dispatch paths after capture, per mode:

        - HB_P2_COMPILE_DECODE=1: the prefill lane bypasses the compiled
          callable (_lane_skip_compiled -> forward_context.skip_compiled ->
          original Python forward), so ONLY prefill picks up the custom
          kernels; decode keeps entering the compiled artifact (uncaptured
          sizes) or replaying its graphs. This is the intended mode --
          Theo's eager-prefill regime, translated.
        - graphs-only without compilation: both lanes' Python forwards are
          eager; decode normally replays FULL graphs, but a non-graph
          decode fallback would see the rebound ops too. (With
          CompilationMode NONE custom ops default to enabled anyway, so
          the rebind is normally a no-op in this mode.)
        - HB_P2_COMPILE_BOTH=1: NOT reachable for the compiled prefill --
          __init__ rejects the combination instead of silently changing
          nothing.
        """
        spec = self._prefill_custom_ops
        if not spec:
            return
        rebound = rebind_custom_ops_to_cuda(self.model, spec)
        logger.info(
            "HB_P2_PREFILL_CUSTOM_OPS: rebound %d modules to forward_cuda "
            "(%s); decode's captured graphs and compiled-artifact entries "
            "keep their capture-time kernels",
            len(rebound),
            sorted(set(rebound)),
        )

    def capture_model(self) -> int:
        captured = super().capture_model()
        self._apply_prefill_custom_ops()
        return captured

    def _fa3_builders(self):
        builders = {}
        for groups in getattr(self, "attn_groups", []) or []:
            for group in groups:
                builder = group.get_metadata_builder(0)
                if getattr(builder, "use_full_cuda_graph", False):
                    builders[id(builder)] = builder
        return list(builders.values())

    def _begin_prep(self, context: LaneContext) -> None:
        # HB_P2_PREP_STREAM: run this ticket's prep window on the lane's
        # prep stream. _end_prep (called from the pre-forward
        # _host_prep_lock_release, after the staged-buffer fence) records a
        # prep-done event and restores the execution stream, which then
        # waits on that event. Never used for dummy runs: capture/warmup/
        # profiling must see stock stream semantics.
        assert context.prep_stream is not None
        self._tl.prep_exec_stream = torch.cuda.current_stream(self.device)
        torch.cuda.set_stream(context.prep_stream)

    def _end_prep(self) -> None:
        """Return to the lane's execution stream, ordered after this prep."""
        exec_stream = getattr(self._tl, "prep_exec_stream", None)
        if exec_stream is None:
            return
        self._tl.prep_exec_stream = None
        context = self._tl_context()
        assert context is not None
        context.prep_event.record(torch.cuda.current_stream(self.device))
        torch.cuda.set_stream(exec_stream)
        exec_stream.wait_event(context.prep_event)

    def _host_prep_lock_acquire(self) -> None:
        self._gen_state_lock.acquire()
        self._tl.lock_held = True
        self._tl.fence_owed = True
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
                for attr in (
                    "paged_kv_indptr",
                    "paged_kv_indices",
                    "paged_kv_last_page_len",
                ):
                    t0, t1 = getattr(b0, attr, None), getattr(b1, attr, None)
                    if t0 is not None and t1 is not None:
                        assert t0.gpu.data_ptr() != t1.gpu.data_ptr(), (
                            "S8 violation: shared builder buffer",
                            attr,
                        )
                w0 = getattr(b0, "_workspace_buffer", None)
                w1 = getattr(b1, "_workspace_buffer", None)
                if w0 is not None and w1 is not None:
                    assert w0.data_ptr() != w1.data_ptr(), (
                        "S8 violation: shared float workspace"
                    )

    def _host_prep_early_release(self) -> None:
        # S9c-P2. Runs under the lock: restore the FA3 flags and the shared
        # input_buffers pointer BEFORE releasing (the other lane may acquire
        # immediately and install its own buffers -- a post-release swap-back
        # would clobber it). The staged-buffer fence stays owed to the
        # pre-forward seam: it must record after ALL consumer enqueues.
        if not self._prep_early_release:
            return
        if not getattr(self._tl, "lock_held", False):
            return
        for builder in getattr(self._tl, "fa3_flipped", ()):
            builder.use_full_cuda_graph = True
        self._tl.fa3_flipped = ()
        if self._tl_context() is not None:
            self.input_buffers = self.lane_contexts[LANE_DECODE].input_buffers
        self._tl.lock_held = False
        self._gen_state_lock.release()

    def _host_prep_lock_release(self) -> None:
        # Idempotent: called at the pre-forward seam on the happy path and
        # from execute_model's finally for early returns and exceptions.
        held = getattr(self._tl, "lock_held", False)
        fence_owed = getattr(self._tl, "fence_owed", False)
        if not held and not fence_owed:
            return
        try:
            for builder in getattr(self._tl, "fa3_flipped", ()):
                builder.use_full_cuda_graph = True
            self._tl.fa3_flipped = ()
            if self._tl_context() is not None:
                # Every staged-transport consumer for this ticket is enqueued
                # on the current (lane) stream; arm buffer-reuse events.
                fence_uva_pools(torch.cuda.current_stream(self.device))
                # HB_P2_PREP_STREAM: prep is fully enqueued and fenced on
                # the prep stream; hand the ticket back to the execution
                # stream (records the prep-done event it waits on). No-op
                # when the feature is off or prep never began (dummy runs).
                self._end_prep()
                if held:
                    self.input_buffers = self.lane_contexts[LANE_DECODE].input_buffers
        finally:
            self._tl.fence_owed = False
            if held:
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
        # Base signature: (scheduler_output, intermediate_tensors=None,
        # dummy_run=False, ...).
        dummy_run = kwargs.get("dummy_run", args[1] if len(args) > 1 else False)
        if self._prep_stream_enabled and not dummy_run:
            self._begin_prep(context)
        margin = self.fa_sm_margin.get(lane, 0)
        margin_ctx = lane_sm_margin(margin) if margin > 0 else nullcontext()
        try:
            with margin_ctx:
                result = super().execute_model(scheduler_output, *args, **kwargs)
        finally:
            self._host_prep_lock_release()
            # Safety net for exits that bypass the pre-forward seam (an
            # exception before the lock acquire): never leave this thread
            # on the prep stream. Idempotent -- the seam already cleared it
            # on the happy path and on locked early returns.
            self._end_prep()
        context.main_stream = torch.cuda.current_stream(self.device)
        context.completion_event.record(context.main_stream)
        return result

    def sample_tokens(self, grammar_output):
        context = self._tl_context()
        if context is None:
            return super().sample_tokens(grammar_output)
        if context.main_stream is not None:
            # Order this thread's sampling stream behind the lane's forward.
            torch.cuda.current_stream(self.device).wait_event(context.completion_event)
        if self._sample_narrow:
            # S9c-P1: lock-free body (streams/state routed per-lane by the
            # properties; postprocess takes the short tail lock in-base),
            # then fence sampling's staged-buffer consumers under the lock
            # so ring-slot rewrites keep waiting on them ([H2]).
            result = super().sample_tokens(grammar_output)
            with self._gen_state_lock:
                fence_uva_pools(torch.cuda.current_stream(self.device))
        else:
            with self._gen_state_lock:
                result = super().sample_tokens(grammar_output)
        # Re-record so controller readiness queries cover the full
        # ticket (forward + sampling + postprocess).
        context.completion_event.record(torch.cuda.current_stream(self.device))
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

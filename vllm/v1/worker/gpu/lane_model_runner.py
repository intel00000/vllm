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

from contextlib import contextmanager
from dataclasses import dataclass, field

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

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

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
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
        logger.info(
            "LaneModelRunner: %s lanes (decode = base buffers, prefill = "
            "private scratch)",
            list(self.lane_contexts),
        )

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

    @contextmanager
    def _use_input_buffers(self, context: LaneContext):
        """Scoped redirect of the prepare path onto the lane's buffers.

        Base prepare_inputs writes exclusively through self.input_buffers
        (verified at v0.26 -- no helper caches a reference), so an attribute
        swap is sufficient and keeps the base method byte-identical for the
        decode lane, whose context holds the base buffers anyway.
        """
        previous = self.input_buffers
        self.input_buffers = context.input_buffers
        try:
            yield
        finally:
            self.input_buffers = previous

    def _prepare_attn_for_lane(
        self, context: LaneContext, input_batch: InputBatch
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        if context.attn_scratch is None:
            # Decode (or pre-KV-init dummy): base path, persistent buffers.
            return self.prepare_attn(input_batch)
        return context.attn_scratch.prepare(self.block_tables, input_batch)

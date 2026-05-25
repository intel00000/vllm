# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


DUAL_MODEL_CONFIG_KEY = "dual_model"
DEFAULT_DECODE_MODEL_ID = "decode"
DEFAULT_EMBED_MODEL_ID = "embed"


@dataclass(frozen=True)
class DualModelConfig:
    embed_model: str
    decode_model_id: str = DEFAULT_DECODE_MODEL_ID
    embed_model_id: str = DEFAULT_EMBED_MODEL_ID
    embed_tokenizer: str | None = None
    embed_dtype: str | None = None
    embed_max_model_len: int | None = None
    embed_enforce_eager: bool | None = None
    decode_running_reserve: int | None = None
    max_embed_running_reqs: int | None = None
    embed_release_running_decode_threshold: int | None = None
    embed_release_when_decode_waiting_drained: bool = False
    # Per-step token caps split into three buckets. Each request's token cost
    # is counted against exactly one bucket per scheduler step:
    #   - decode_prefill: model_id == decode and num_output_tokens == 0
    #   - decode_loop:    model_id == decode and num_output_tokens > 0
    #   - embed_prefill:  model_id == embed (always prefill — pool is one step)
    # When a bucket cap is reached, additional requests of that bucket are
    # deferred to a later step even if the global max_num_batched_tokens has
    # room. None = no cap (legacy behaviour).
    max_decode_prefill_tokens_per_step: int | None = None
    max_decode_loop_tokens_per_step: int | None = None
    max_embed_prefill_tokens_per_step: int | None = None

    # RAG-style scheduling.
    #   enforce_pair_dependency: hold decode requests with a parent_request_id
    #     out of the waiting queue until the parent embed has finished.
    #   enforce_no_double_prefill: per-step admission gate — do not admit a
    #     new embed-prefill if a decode-prefill has already been admitted (or
    #     is carry-over running) this step, and vice versa.
    enforce_pair_dependency: bool = False
    enforce_no_double_prefill: bool = False

    # Defensive KV-pressure gate. When KV usage is at/above this threshold,
    # the scheduler skips NEW admissions from the waiting queue (running
    # requests carry over freely). Protects against the pair-dep release
    # cascade where many decode children get released into the waiting queue
    # at once and overcommit KV blocks in the same step. None = off (legacy).
    kv_pressure_skip_threshold: float | None = None

    @classmethod
    def from_vllm_config(cls, vllm_config: Any) -> "DualModelConfig | None":
        additional_config = getattr(vllm_config, "additional_config", None)
        if not isinstance(additional_config, dict):
            return None
        raw_cfg = additional_config.get(DUAL_MODEL_CONFIG_KEY)
        if not isinstance(raw_cfg, dict):
            return None

        embed_model = raw_cfg.get("embed_model")
        if not embed_model:
            raise ValueError(
                "additional_config['dual_model']['embed_model'] is required."
            )

        return cls(
            embed_model=embed_model,
            decode_model_id=raw_cfg.get(
                "decode_model_id", DEFAULT_DECODE_MODEL_ID
            ),
            embed_model_id=raw_cfg.get("embed_model_id", DEFAULT_EMBED_MODEL_ID),
            embed_tokenizer=raw_cfg.get("embed_tokenizer"),
            embed_dtype=raw_cfg.get("embed_dtype"),
            embed_max_model_len=raw_cfg.get("embed_max_model_len"),
            embed_enforce_eager=raw_cfg.get("embed_enforce_eager"),
            decode_running_reserve=raw_cfg.get("decode_running_reserve"),
            max_embed_running_reqs=raw_cfg.get("max_embed_running_reqs"),
            embed_release_running_decode_threshold=raw_cfg.get(
                "embed_release_running_decode_threshold"
            ),
            embed_release_when_decode_waiting_drained=bool(
                raw_cfg.get("embed_release_when_decode_waiting_drained", False)
            ),
            max_decode_prefill_tokens_per_step=raw_cfg.get(
                "max_decode_prefill_tokens_per_step"
            ),
            max_decode_loop_tokens_per_step=raw_cfg.get(
                "max_decode_loop_tokens_per_step"
            ),
            max_embed_prefill_tokens_per_step=raw_cfg.get(
                "max_embed_prefill_tokens_per_step"
            ),
            enforce_pair_dependency=bool(
                raw_cfg.get("enforce_pair_dependency", False)
            ),
            enforce_no_double_prefill=bool(
                raw_cfg.get("enforce_no_double_prefill", False)
            ),
            kv_pressure_skip_threshold=raw_cfg.get("kv_pressure_skip_threshold"),
        )

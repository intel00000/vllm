# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

import pytest

from vllm.v1.worker.dual_model_runner import DualModelRunner


def test_import_and_class_shape():
    assert DualModelRunner.EMBED_MODEL_PREFIX == "embed_model"
    # Sanity check the public surface so a future rename surfaces here.
    expected_methods = {
        "capture_model",
        "execute_model",
        "get_kv_cache_spec",
        "get_model",
        "get_supported_tasks",
        "initialize_kv_cache",
        "load_model",
        "profile_cudagraph_memory",
        "profile_run",
        "reset_encoder_cache",
        "reset_mm_cache",
        "sample_tokens",
        "take_draft_token_ids",
        "update_max_model_len",
    }
    actual_methods = {
        name for name in dir(DualModelRunner) if not name.startswith("_")
    }
    missing = expected_methods - actual_methods
    assert not missing, f"DualModelRunner is missing methods: {missing}"


@dataclass
class _ToyConfig:
    name: str
    value: int
    optional_flag: bool = False


def test_replace_dataclass_overrides_fields():
    original = _ToyConfig(name="orig", value=1)
    updated = DualModelRunner._replace_dataclass(
        original, name="new", optional_flag=True
    )
    assert isinstance(updated, _ToyConfig)
    assert updated.name == "new"
    assert updated.value == 1  # not in updates -> inherited
    assert updated.optional_flag is True
    # Original unchanged (frozen replacement semantics).
    assert original.name == "orig"
    assert original.optional_flag is False


def test_replace_dataclass_rejects_non_dataclass():
    class _Plain:
        def __init__(self, x: int) -> None:
            self.x = x

    with pytest.raises(TypeError, match="not a dataclass"):
        DualModelRunner._replace_dataclass(_Plain(1), x=2)

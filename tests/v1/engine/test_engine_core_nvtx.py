# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for the engine_core_nvtx_range context manager.

The context manager is opt-in via HB_ENGINE_CORE_NVTX_RANGES=1. When
unset it must be a complete no-op (no torch.cuda.nvtx imports, no
range_push/pop calls). When set it must push and pop exactly once.
"""

from unittest.mock import patch

import pytest


def test_disabled_by_default_is_noop(monkeypatch):
    """No env var -> nvtx.range_push must not be called."""
    monkeypatch.delenv("HB_ENGINE_CORE_NVTX_RANGES", raising=False)
    from vllm.v1.engine.core import engine_core_nvtx_range

    with patch("torch.cuda.nvtx.range_push") as push, patch(
        "torch.cuda.nvtx.range_pop"
    ) as pop:
        with engine_core_nvtx_range("test"):
            pass
        push.assert_not_called()
        pop.assert_not_called()


@pytest.mark.parametrize("falsy", ["0", "", "false", "no", "off", "FALSE", "NO"])
def test_falsy_values_disable(monkeypatch, falsy):
    """Various falsy spellings keep the gate off."""
    monkeypatch.setenv("HB_ENGINE_CORE_NVTX_RANGES", falsy)
    from vllm.v1.engine.core import engine_core_nvtx_range

    with patch("torch.cuda.nvtx.range_push") as push:
        with engine_core_nvtx_range("test"):
            pass
        push.assert_not_called()


@pytest.mark.parametrize("truthy", ["1", "true", "yes", "on", "TRUE", "Yes", "ON"])
def test_truthy_values_enable(monkeypatch, truthy):
    """Various truthy spellings enable the gate."""
    monkeypatch.setenv("HB_ENGINE_CORE_NVTX_RANGES", truthy)
    from vllm.v1.engine.core import engine_core_nvtx_range

    with patch("torch.cuda.nvtx.range_push") as push, patch(
        "torch.cuda.nvtx.range_pop"
    ) as pop:
        with engine_core_nvtx_range("test_range"):
            pass
        push.assert_called_once_with("test_range")
        pop.assert_called_once()


def test_pop_runs_on_exception(monkeypatch):
    """range_pop must run even if the wrapped block raises -- otherwise
    the NVTX stack would be unbalanced."""
    monkeypatch.setenv("HB_ENGINE_CORE_NVTX_RANGES", "1")
    from vllm.v1.engine.core import engine_core_nvtx_range

    with patch("torch.cuda.nvtx.range_push") as push, patch(
        "torch.cuda.nvtx.range_pop"
    ) as pop:
        with pytest.raises(RuntimeError, match="boom"):
            with engine_core_nvtx_range("test"):
                raise RuntimeError("boom")
        push.assert_called_once_with("test")
        pop.assert_called_once()


def test_passes_range_name_to_push(monkeypatch):
    """The label string is forwarded to nvtx.range_push verbatim."""
    monkeypatch.setenv("HB_ENGINE_CORE_NVTX_RANGES", "1")
    from vllm.v1.engine.core import engine_core_nvtx_range

    with patch("torch.cuda.nvtx.range_push") as push, patch(
        "torch.cuda.nvtx.range_pop"
    ):
        with engine_core_nvtx_range("engine_core: schedule"):
            pass
        push.assert_called_once_with("engine_core: schedule")

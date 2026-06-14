# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for vllm.v1.worker.work_stream.

Pure-Python tests for backend selection + env-var parsing.  Tests that
need actual CUDA streams are gated on ``torch.cuda.is_available()``.
"""

from __future__ import annotations

import contextlib
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.v1.worker import work_stream as ws
from vllm.v1.worker.work_stream import (
    BACKEND_GREEN_CTX,
    BACKEND_LIBSMCTRL,
    BACKEND_NONE,
    DefaultWorkStream,
    ENV_BACKEND,
    ENV_DECODE_SMS,
    ENV_EMBED_SMS,
    ENV_SINGLE_SMS,
    LibsmctrlWorkStream,
    WorkStream,
    _read_int_env,
    make_work_streams,
)


def _fake_vllm_config(dual: bool):
    cfg = MagicMock()
    cfg.additional_config = (
        {"dual_model": {"embed_model": "BAAI/bge-large-en-v1.5"}} if dual else None
    )
    return cfg


# ---------------------------------------------------------------------------
# Module surface
# ---------------------------------------------------------------------------


def test_backend_constants_match_documented_values():
    # If anyone renames these, sbatches and docs must change too.
    assert BACKEND_NONE == "none"
    assert BACKEND_GREEN_CTX == "green_ctx"
    assert BACKEND_LIBSMCTRL == "libsmctrl"


def test_workstream_is_abstract():
    with pytest.raises(TypeError):
        WorkStream()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# _read_int_env
# ---------------------------------------------------------------------------


def test_read_int_env_missing_raises(monkeypatch):
    monkeypatch.delenv("FOO_TEST", raising=False)
    with pytest.raises(ValueError, match="required"):
        _read_int_env("FOO_TEST")


def test_read_int_env_blank_raises(monkeypatch):
    monkeypatch.setenv("FOO_TEST", "   ")
    with pytest.raises(ValueError, match="required"):
        _read_int_env("FOO_TEST")


def test_read_int_env_non_int_raises(monkeypatch):
    monkeypatch.setenv("FOO_TEST", "abc")
    with pytest.raises(ValueError, match="not an integer"):
        _read_int_env("FOO_TEST")


def test_read_int_env_zero_raises(monkeypatch):
    monkeypatch.setenv("FOO_TEST", "0")
    with pytest.raises(ValueError, match="must be positive"):
        _read_int_env("FOO_TEST")


def test_read_int_env_valid(monkeypatch):
    monkeypatch.setenv("FOO_TEST", "42")
    assert _read_int_env("FOO_TEST") == 42


# ---------------------------------------------------------------------------
# make_work_streams: backend validation (no CUDA needed)
# ---------------------------------------------------------------------------


def test_make_work_streams_invalid_backend_raises(monkeypatch):
    monkeypatch.setenv(ENV_BACKEND, "bogus")
    with pytest.raises(ValueError, match="bogus"):
        make_work_streams(_fake_vllm_config(dual=False), torch.device("cuda:0"))


def test_make_work_streams_default_when_unset(monkeypatch):
    monkeypatch.delenv(ENV_BACKEND, raising=False)
    # Stub torch.cuda.default_stream so this works without a real GPU.
    sentinel = object()
    with patch.object(torch.cuda, "default_stream", return_value=sentinel):
        primary, aux = make_work_streams(
            _fake_vllm_config(dual=False), torch.device("cuda:0")
        )
    assert isinstance(primary, DefaultWorkStream)
    assert primary.stream is sentinel
    assert primary.backend == BACKEND_NONE
    assert aux is None


def test_make_work_streams_default_dual(monkeypatch):
    monkeypatch.setenv(ENV_BACKEND, BACKEND_NONE)
    sentinel = object()
    with patch.object(torch.cuda, "default_stream", return_value=sentinel):
        primary, aux = make_work_streams(
            _fake_vllm_config(dual=True), torch.device("cuda:0")
        )
    assert isinstance(primary, DefaultWorkStream)
    assert isinstance(aux, DefaultWorkStream)


def test_make_work_streams_green_ctx_dual_missing_sms_raises(monkeypatch):
    monkeypatch.setenv(ENV_BACKEND, BACKEND_GREEN_CTX)
    monkeypatch.delenv(ENV_DECODE_SMS, raising=False)
    monkeypatch.delenv(ENV_EMBED_SMS, raising=False)
    fake_props = MagicMock(multi_processor_count=132)
    with patch.object(
        torch.cuda, "get_device_properties", return_value=fake_props
    ):
        with pytest.raises(ValueError, match="DECODE_SMS"):
            make_work_streams(
                _fake_vllm_config(dual=True), torch.device("cuda:0")
            )


def test_make_work_streams_green_ctx_single_missing_sms_raises(monkeypatch):
    monkeypatch.setenv(ENV_BACKEND, BACKEND_GREEN_CTX)
    monkeypatch.delenv(ENV_SINGLE_SMS, raising=False)
    fake_props = MagicMock(multi_processor_count=132)
    with patch.object(
        torch.cuda, "get_device_properties", return_value=fake_props
    ):
        with pytest.raises(ValueError, match="WORK_STREAM_SMS"):
            make_work_streams(
                _fake_vllm_config(dual=False), torch.device("cuda:0")
            )


def test_make_work_streams_green_ctx_dual_overflow_raises(monkeypatch):
    monkeypatch.setenv(ENV_BACKEND, BACKEND_GREEN_CTX)
    monkeypatch.setenv(ENV_DECODE_SMS, "100")
    monkeypatch.setenv(ENV_EMBED_SMS, "100")
    fake_props = MagicMock(multi_processor_count=132)
    with patch.object(
        torch.cuda, "get_device_properties", return_value=fake_props
    ):
        with pytest.raises(ValueError, match="total_sms"):
            make_work_streams(
                _fake_vllm_config(dual=True), torch.device("cuda:0")
            )


# ---------------------------------------------------------------------------
# LibsmctrlWorkStream: rejection on Hopper
# ---------------------------------------------------------------------------


def test_libsmctrl_rejects_hopper():
    with patch.object(
        torch.cuda, "get_device_capability", return_value=(9, 0)
    ):
        with pytest.raises(RuntimeError, match="Ampere"):
            LibsmctrlWorkStream(torch.device("cuda:0"), 32, 132)


def test_libsmctrl_rejects_zero_tpc():
    # sm_count=1 -> tpc_count=0 -> rejected.
    with patch.object(
        torch.cuda, "get_device_capability", return_value=(8, 0)
    ):
        with pytest.raises(ValueError, match="0 TPCs"):
            LibsmctrlWorkStream(torch.device("cuda:0"), 1, 108)


def test_libsmctrl_rejects_oversized():
    with patch.object(
        torch.cuda, "get_device_capability", return_value=(8, 0)
    ):
        with pytest.raises(ValueError, match="out of range"):
            LibsmctrlWorkStream(torch.device("cuda:0"), 200, 108)


# ---------------------------------------------------------------------------
# CUDA-required tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_default_work_stream_context_is_nullcontext():
    d = torch.device("cuda:0")
    w = DefaultWorkStream(d)
    ctx = w.context()
    # We don't promise "is" identity, only that it's a no-op
    # equivalent to nullcontext (so the unpartitioned path pays zero
    # stream-context-manager overhead).
    assert isinstance(ctx, contextlib.nullcontext)
    # torch returns a fresh Stream wrapper each call; identity isn't
    # preserved, but the underlying cuda_stream pointer is.
    assert w.stream == torch.cuda.default_stream(d)
    assert w.sm_count is None
    w.close()  # should not raise


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_make_work_streams_none_returns_real_default_stream():
    d = torch.device("cuda:0")
    primary, aux = make_work_streams(_fake_vllm_config(dual=False), d)
    assert isinstance(primary, DefaultWorkStream)
    assert primary.stream == torch.cuda.default_stream(d)
    assert aux is None

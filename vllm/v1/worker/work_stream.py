# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Worker-owned CUDA stream abstraction with pluggable partitioning backends.

A ``WorkStream`` is a stable stream identity owned by ``Worker``.  Worker
entry methods (``capture_model`` / ``execute_model`` / ``sample_tokens``)
wrap their bodies in ``self._work_stream.context()`` so every CUDA op the
worker issues -- including the misc orchestration around the model runner
call (NVTX, async D2H setup, sampled-token copy events) -- runs on the
same stream identity.  Without this, those ops fall on the default stream
while the model_runner.execute_model body runs on a partitioned stream,
which causes cuBLAS handle bouncing and async-copy event-vs-stream
mismatches (manifests as ``CUBLAS_STATUS_EXECUTION_FAILED`` on H200,
``cudaErrorIllegalAddress`` on A100).

Three backends:
  * :class:`DefaultWorkStream` -- no partitioning, ``context()`` is a true
    no-op (returns ``contextlib.nullcontext()``) so the unpartitioned path
    pays zero overhead.
  * :class:`GreenCtxWorkStream` -- flashinfer green-context stream bound
    to a disjoint SM partition via ``cuDevSmResourceSplitByCount``.
  * :class:`LibsmctrlWorkStream` -- vanilla torch stream + libsmctrl_v2's
    persistent thread-local TPC mask.  Ampere only (Hopper QMD v4 port is
    not done); requires ``enforce_eager=True`` because the mask doesn't
    survive cudagraph capture.

Selection via env vars (see :func:`make_work_streams`):

==================================  =======================  =====================
Env var                             Single-model             Dual-model
==================================  =======================  =====================
VLLM_WORK_STREAM_BACKEND            applies to the one ws    same backend for both
VLLM_WORK_STREAM_SMS                sm count for the one ws  unused
VLLM_DUAL_DECODE_SMS                unused                   decode partition sms
VLLM_DUAL_EMBED_SMS                 unused                   embed partition sms
VLLM_LIBSMCTRL_LIB_PATH             path to libsmctrl_v2.so  same
==================================  =======================  =====================
"""

from __future__ import annotations

import contextlib
import ctypes
import os
from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any

import torch

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


# Public env var names -- referenced by sbatches and tests.
ENV_BACKEND = "VLLM_WORK_STREAM_BACKEND"
ENV_SINGLE_SMS = "VLLM_WORK_STREAM_SMS"
ENV_DECODE_SMS = "VLLM_DUAL_DECODE_SMS"
ENV_EMBED_SMS = "VLLM_DUAL_EMBED_SMS"
ENV_LIBSMCTRL_PATH = "VLLM_LIBSMCTRL_LIB_PATH"

_DEFAULT_LIBSMCTRL_PATH = "/n/home09/ttan1/rag_profile/lib/libsmctrl_v2.so"

# Backend identifiers accepted in VLLM_WORK_STREAM_BACKEND.
BACKEND_NONE = "none"
BACKEND_GREEN_CTX = "green_ctx"
BACKEND_LIBSMCTRL = "libsmctrl"
_VALID_BACKENDS = (BACKEND_NONE, BACKEND_GREEN_CTX, BACKEND_LIBSMCTRL)


class WorkStream(ABC):
    """A stable stream identity owned by the worker.

    Subclasses MUST set :attr:`stream` to the torch stream they wrap.
    ``context()`` MUST establish that stream as ``torch.cuda.current_stream``
    inside the with-block and MAY also install backend-specific per-thread
    state (e.g. libsmctrl mask).  ``close()`` releases backend-owned
    resources.
    """

    stream: torch.cuda.Stream
    backend: str  # one of BACKEND_*
    sm_count: int | None = None  # how many SMs this stream is restricted to

    @abstractmethod
    def context(self) -> AbstractContextManager:
        """Context manager that pins CUDA work to this WorkStream.

        For :class:`DefaultWorkStream` this is :class:`contextlib.nullcontext`
        so the zero-partition path pays no overhead.  For partitioning
        backends it sets the stream and any per-thread mask state.
        """

    @abstractmethod
    def close(self) -> None:
        """Free backend-owned resources (green-ctx handles, libsmctrl
        mask state, etc.).  Idempotent."""

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        sms = f", sm_count={self.sm_count}" if self.sm_count is not None else ""
        return f"{type(self).__name__}(backend={self.backend!r}{sms})"


class DefaultWorkStream(WorkStream):
    """No partitioning.  ``stream`` is the device default stream and
    ``context()`` is a true no-op.

    Used when ``VLLM_WORK_STREAM_BACKEND`` is unset or ``none``.  Callers
    can rely on ``context()`` being free; the unpartitioned path doesn't
    even take the stream-context-manager overhead.
    """

    backend = BACKEND_NONE

    def __init__(self, device: torch.device):
        self.stream = torch.cuda.default_stream(device)
        self.sm_count = None

    def context(self) -> AbstractContextManager:
        return contextlib.nullcontext()

    def close(self) -> None:
        pass


class GreenCtxWorkStream(WorkStream):
    """flashinfer green-context stream bound to a disjoint SM partition.

    Instances MUST be obtained via :meth:`create_pair` (dual-model) or
    :meth:`create_single` (single-model).  Direct construction takes the
    pre-built torch stream and the realized SM count.

    Lifetime: flashinfer manages the underlying green-ctx via the stream
    object, so :meth:`close` only drops the resource reference.
    """

    backend = BACKEND_GREEN_CTX

    def __init__(
        self,
        stream: torch.cuda.Stream,
        sm_count: int,
        device: torch.device,
        _resource_handle: Any = None,
    ):
        self.stream = stream
        self.sm_count = sm_count
        self._device = device
        self._resource = _resource_handle

    @classmethod
    def create_single(
        cls,
        device: torch.device,
        device_index: int,
        sm_count: int,
        total_sms: int,
    ) -> "GreenCtxWorkStream":
        """Build a single green-ctx stream carving ``sm_count`` SMs off the
        primary resource.  Used in single-model deployments where we still
        want SM-scoping (e.g. comparing scaling curves at fixed SM counts).
        """
        if sm_count <= 0 or sm_count > total_sms:
            raise ValueError(
                f"sm_count={sm_count} out of range (1..{total_sms})"
            )
        try:
            from flashinfer.green_ctx import (
                create_green_ctx_streams,
                get_cudevice,
                get_device_resource,
                split_resource,
            )
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "flashinfer is required for GreenCtxWorkStream. "
                "Install flashinfer-python."
            ) from e
        cu_dev = get_cudevice(device)
        primary = get_device_resource(cu_dev)
        peeled, _remainder = split_resource(primary, 1, sm_count)
        if not peeled or peeled[0].sm.smCount == 0:
            raise RuntimeError(
                f"Could not peel {sm_count} SMs from primary resource "
                f"({primary.sm.smCount} SMs available)"
            )
        res = peeled[0]
        streams = create_green_ctx_streams(cu_dev, [res])
        return cls(
            stream=streams[0],
            sm_count=int(res.sm.smCount),
            device=device,
            _resource_handle=res,
        )

    @classmethod
    def create_pair(
        cls,
        device: torch.device,
        device_index: int,
        decode_sms: int,
        embed_sms: int,
        total_sms: int,
    ) -> tuple["GreenCtxWorkStream", "GreenCtxWorkStream"]:
        """Build two disjoint green-ctx streams via peel-then-remainder.

        ``embed_sms`` is informational only -- embed gets exactly the
        remainder after decode is peeled.  This avoids
        ``flashinfer.green_ctx.split_device_green_ctx_by_sm_count`` which
        rounds each requested count up independently and can over-allocate
        (e.g. ``[40, 92]`` becomes ``[40, 96]`` and 40+96 > 132 fails with
        ``CUDA_ERROR_INVALID_RESOURCE_CONFIGURATION``).
        """
        if decode_sms <= 0 or decode_sms > total_sms:
            raise ValueError(
                f"decode_sms={decode_sms} out of range (1..{total_sms})"
            )
        if embed_sms <= 0 or embed_sms > total_sms:
            raise ValueError(
                f"embed_sms={embed_sms} out of range (1..{total_sms})"
            )
        try:
            from flashinfer.green_ctx import (
                create_green_ctx_streams,
                get_cudevice,
                get_device_resource,
                split_resource,
            )
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "flashinfer is required for GreenCtxWorkStream. "
                "Install flashinfer-python."
            ) from e
        cu_dev = get_cudevice(device)
        primary = get_device_resource(cu_dev)
        decode_list, remaining = split_resource(primary, 1, decode_sms)
        if not decode_list or decode_list[0].sm.smCount == 0:
            raise RuntimeError(
                f"Could not peel {decode_sms} SMs off primary "
                f"({primary.sm.smCount} available)"
            )
        decode_res = decode_list[0]
        if remaining.sm.smCount == 0:
            raise RuntimeError(
                f"No SMs left for embed after peeling {decode_res.sm.smCount} "
                f"for decode"
            )
        streams = create_green_ctx_streams(cu_dev, [decode_res, remaining])
        actual_decode_sms = int(decode_res.sm.smCount)
        actual_embed_sms = int(remaining.sm.smCount)
        if actual_decode_sms + actual_embed_sms > total_sms:
            raise RuntimeError(
                f"Green-ctx partitions overlap: decode={actual_decode_sms} "
                f"+ embed={actual_embed_sms} > total {total_sms}"
            )
        decode_ws = cls(
            stream=streams[0],
            sm_count=actual_decode_sms,
            device=device,
            _resource_handle=decode_res,
        )
        embed_ws = cls(
            stream=streams[1],
            sm_count=actual_embed_sms,
            device=device,
            _resource_handle=remaining,
        )
        return decode_ws, embed_ws

    def context(self) -> AbstractContextManager:
        return torch.cuda.stream(self.stream)

    def close(self) -> None:
        # flashinfer manages green-ctx lifetime via the stream, so we
        # just drop the resource reference.
        self._resource = None


class LibsmctrlWorkStream(WorkStream):
    """libsmctrl_v2 thread-local TPC mask + a vanilla torch stream.

    Partitioning is enforced by writing the SM mask into each kernel's
    QMD via libsmctrl's launch callback.  The mask is per-thread
    (``__thread`` in libsmctrl_v2.c), so the calling thread must be
    stable: in DualModelRunner each role lives on a single-worker
    ``ThreadPoolExecutor``, and the per-role warmup primes the mask on
    that thread before any model compute.

    Constraints:
      * Ampere only -- Hopper uses QMD v4 with the TPC field at offset
        +304 and libsmctrl currently writes to +84.  ``__init__`` raises
        on Hopper.
      * Requires ``enforce_eager=True`` -- the mask doesn't bake into
        CUDA graphs; replayed kernels run unmasked.  We assert this in
        the factory once we know about the model runner config.
    """

    backend = BACKEND_LIBSMCTRL

    # Cache the loaded .so so we don't pay re-load cost when we build
    # multiple LibsmctrlWorkStream instances (e.g. decode + embed).
    _lib_cache: dict[str, ctypes.CDLL] = {}

    def __init__(
        self,
        device: torch.device,
        sm_count: int,
        total_sms: int,
        tpc_offset: int = 0,
        lib_path: str | None = None,
    ):
        # Hopper (CC 9.x) is not supported until libsmctrl gets QMD v4.
        major, _minor = torch.cuda.get_device_capability(device)
        if major >= 9:
            raise RuntimeError(
                f"LibsmctrlWorkStream requires Ampere (CC 8.x); device has "
                f"CC {major}.x.  libsmctrl's launchCallback patches QMD bytes "
                f"at offset +84 but Hopper QMD v4 puts the TPC mask at +304 "
                f"-- not yet ported.  Use {BACKEND_GREEN_CTX} on Hopper."
            )
        if sm_count <= 0 or sm_count > total_sms:
            raise ValueError(
                f"sm_count={sm_count} out of range (1..{total_sms})"
            )
        # Ampere has 2 SMs per TPC.  We compute a TPC bitmask where set
        # bit = TPC disabled (libsmctrl convention).
        tpc_total = total_sms // 2
        tpc_count = sm_count // 2
        if tpc_count <= 0:
            raise ValueError(
                f"sm_count={sm_count} resolves to 0 TPCs (need >=2 SMs)"
            )
        if tpc_offset + tpc_count > tpc_total:
            raise ValueError(
                f"TPC range [{tpc_offset}, {tpc_offset + tpc_count}) does "
                f"not fit in {tpc_total} total TPCs"
            )

        path = lib_path or os.environ.get(
            ENV_LIBSMCTRL_PATH, _DEFAULT_LIBSMCTRL_PATH
        )
        if path not in self._lib_cache:
            try:
                self._lib_cache[path] = ctypes.CDLL(path)
            except OSError as e:  # pragma: no cover
                raise RuntimeError(
                    f"Could not load libsmctrl_v2 from {path!r}.  Override "
                    f"with ${ENV_LIBSMCTRL_PATH}."
                ) from e
        self._lib = self._lib_cache[path]
        self._lib.libsmctrl_set_thread_mask.argtypes = [ctypes.c_uint64]
        self._lib.libsmctrl_clear_thread_mask.argtypes = []
        self._lib.libsmctrl_set_global_mask.argtypes = [ctypes.c_uint64]
        # Touching the global mask subscribes the launch callback.
        self._lib.libsmctrl_set_global_mask(ctypes.c_uint64(0))

        enabled = ((1 << tpc_count) - 1) << tpc_offset
        self._mask = (~enabled) & ((1 << 64) - 1)
        self.stream = torch.cuda.Stream(device)
        self.sm_count = tpc_count * 2

    @classmethod
    def create_pair(
        cls,
        device: torch.device,
        decode_sms: int,
        embed_sms: int,
        total_sms: int,
        lib_path: str | None = None,
    ) -> tuple["LibsmctrlWorkStream", "LibsmctrlWorkStream"]:
        """Build disjoint decode + embed libsmctrl streams.  Decode owns
        TPCs ``[0, decode_tpcs)``; embed owns ``[decode_tpcs, decode_tpcs +
        embed_tpcs)``.  Raises if the two ranges overflow ``total_sms``.
        """
        decode_tpcs = decode_sms // 2
        decode_ws = cls(
            device, decode_sms, total_sms, tpc_offset=0, lib_path=lib_path
        )
        embed_ws = cls(
            device,
            embed_sms,
            total_sms,
            tpc_offset=decode_tpcs,
            lib_path=lib_path,
        )
        return decode_ws, embed_ws

    def context(self) -> AbstractContextManager:
        # Capture by closure -- libsmctrl mask is per-thread, so the
        # set / clear must happen on whichever thread enters this with-block.
        lib = self._lib
        mask = self._mask
        stream = self.stream

        @contextlib.contextmanager
        def _ctx():
            lib.libsmctrl_set_thread_mask(ctypes.c_uint64(mask))
            try:
                with torch.cuda.stream(stream):
                    yield
            finally:
                lib.libsmctrl_clear_thread_mask()

        return _ctx()

    def close(self) -> None:
        try:
            self._lib.libsmctrl_clear_thread_mask()
        except Exception:  # pragma: no cover - belt-and-suspenders
            pass


def _read_int_env(name: str) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        raise ValueError(f"${name} is required when backend is set")
    try:
        value = int(raw)
    except ValueError as e:
        raise ValueError(f"${name}={raw!r} is not an integer") from e
    if value <= 0:
        raise ValueError(f"${name}={value} must be positive")
    return value


def make_work_streams(
    vllm_config: "VllmConfig",
    device: torch.device,
) -> tuple[WorkStream, WorkStream | None]:
    """Build the worker's WorkStream(s) from env vars + ``vllm_config``.

    Returns ``(primary_ws, aux_ws)``.  ``aux_ws`` is ``None`` in
    single-model deployments and the embed-side WorkStream when running
    DualModelRunner.

    Resolution rules:
      * ``VLLM_WORK_STREAM_BACKEND`` (default ``none``) picks the backend.
      * Dual vs single is detected from ``DualModelConfig.from_vllm_config``.
      * Single + green_ctx / libsmctrl: needs ``VLLM_WORK_STREAM_SMS``.
      * Dual + green_ctx / libsmctrl: needs ``VLLM_DUAL_DECODE_SMS`` and
        ``VLLM_DUAL_EMBED_SMS``.
      * Backend ``none``: env-var SM knobs are ignored.
    """
    # Local import to dodge any circular dependency with worker modules.
    from vllm.v1.worker.dual_model_helpers import DualModelConfig

    backend = os.environ.get(ENV_BACKEND, BACKEND_NONE).lower()
    if backend not in _VALID_BACKENDS:
        raise ValueError(
            f"${ENV_BACKEND}={backend!r} must be one of {_VALID_BACKENDS}"
        )

    is_dual = DualModelConfig.from_vllm_config(vllm_config) is not None

    if backend == BACKEND_NONE:
        primary = DefaultWorkStream(device)
        aux = DefaultWorkStream(device) if is_dual else None
        logger.info(
            "WorkStream backend=none (dual_model=%s) -- using device default "
            "stream(s); context() is a no-op.",
            is_dual,
        )
        return primary, aux

    total_sms = torch.cuda.get_device_properties(device).multi_processor_count
    device_index = device.index if device.index is not None else 0

    if is_dual:
        decode_sms = _read_int_env(ENV_DECODE_SMS)
        embed_sms = _read_int_env(ENV_EMBED_SMS)
        if decode_sms + embed_sms > total_sms:
            raise ValueError(
                f"decode_sms ({decode_sms}) + embed_sms ({embed_sms}) > "
                f"total_sms ({total_sms})"
            )
        if backend == BACKEND_GREEN_CTX:
            decode_ws, embed_ws = GreenCtxWorkStream.create_pair(
                device, device_index, decode_sms, embed_sms, total_sms
            )
        else:  # libsmctrl
            decode_ws, embed_ws = LibsmctrlWorkStream.create_pair(
                device, decode_sms, embed_sms, total_sms
            )
        logger.info(
            "WorkStream backend=%s dual: decode=%d SMs (req %d) "
            "embed=%d SMs (req %d) total=%d",
            backend, decode_ws.sm_count, decode_sms,
            embed_ws.sm_count, embed_sms, total_sms,
        )
        return decode_ws, embed_ws

    # Single-model + partitioning.
    sm_count = _read_int_env(ENV_SINGLE_SMS)
    if backend == BACKEND_GREEN_CTX:
        primary = GreenCtxWorkStream.create_single(
            device, device_index, sm_count, total_sms
        )
    else:  # libsmctrl
        primary = LibsmctrlWorkStream(device, sm_count, total_sms)
    logger.info(
        "WorkStream backend=%s single: %d SMs (req %d) total=%d",
        backend, primary.sm_count, sm_count, total_sms,
    )
    return primary, None

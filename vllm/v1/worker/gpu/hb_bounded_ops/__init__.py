"""HB bounded elementwise ops (kernel-control program S2b).

JIT-compiles bounded_eltwise.cu at first use (in-tree fmha_sm100 precedent:
torch cpp_extension, per-arch cached under ~/.cache/torch_extensions). The
CTA budget makes the dense lane's elementwise launches narrow (grid-stride)
so they stop queue-blocking wide co-runners; measured on MIG: 8x SM count
holds ~0.98 of wave bandwidth with 2.6x decode-graph recovery.

Wired via HB_P2_PREFILL_CUSTOM_OPS + HB_VLLM_ELTWISE_CTA_BUDGET in
lane_model_runner.rebind_custom_ops_to_cuda; usable standalone via
bounded_silu_and_mul().
"""

import functools
import os

import torch


@functools.lru_cache(maxsize=1)
def _load():
    from torch.utils.cpp_extension import load

    cc = torch.cuda.get_device_capability()
    arch = f"{cc[0]}.{cc[1]}"
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", arch)
    return load(
        name=f"hb_bounded_ops_sm{cc[0]}{cc[1]}",
        sources=[os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "bounded_eltwise.cu")],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3",
                           f"-gencode=arch=compute_{cc[0]}{cc[1]},"
                           f"code=sm_{cc[0]}{cc[1]}"],
        verbose=os.environ.get("HB_BOUNDED_OPS_VERBOSE", "0") == "1",
    )


def bounded_silu_and_mul(x: torch.Tensor, cta_budget: int) -> torch.Tensor:
    """silu(x[..., :d]) * x[..., d:] under a CTA budget (0 = one CTA/token)."""
    d = x.shape[-1] // 2
    out = torch.empty(x.shape[:-1] + (d,), dtype=x.dtype, device=x.device)
    _load().silu_and_mul_bounded(out, x, cta_budget)
    return out

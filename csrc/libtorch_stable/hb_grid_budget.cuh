#pragma once
// HB kernel-control S2b: env-gated CTA budget for wave-enumerated
// elementwise kernels (grid = one CTA per row). With
// HB_VLLM_ELTWISE_CTA_BUDGET=N > 0, launches are capped at N CTAs and the
// kernels grid-stride over rows; 0/unset = stock width. Read ONCE per
// process (static) so cudagraph capture bakes a consistent grid; a single
// global cap self-selects the prefill side — decode-step launches
// (~batch CTAs, about one wave) pass under any sane cap untouched.
// Measured operating point: ~8x SM count holds ~0.98 of wave bandwidth
// while restoring 2.6x co-runner progress (kernel_control_program.md).

#include <cstdlib>

namespace vllm {

inline int hb_eltwise_cta_budget() {
  static const int budget = []() {
    const char* env = std::getenv("HB_VLLM_ELTWISE_CTA_BUDGET");
    if (env == nullptr) return 0;
    const int v = std::atoi(env);
    return v > 0 ? v : 0;
  }();
  return budget;
}

// Launch grid for a one-CTA-per-row kernel under the budget.
inline int64_t hb_bounded_grid(int64_t rows) {
  const int budget = hb_eltwise_cta_budget();
  return (budget > 0 && budget < rows) ? budget : rows;
}

}  // namespace vllm

/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */
#pragma once

#include <functional>

#include "stinkytofu/analysis/asm/GirFrameAnalysis.hpp"
#include "stinkytofu/transforms/asm/waitcnt/WaitPlan.hpp"

namespace stinkytofu {
class Function;

namespace waitcnt {

/// Compute a finite-frame wait plan directly on `(ST basic block, GIR frame)` states.
/// `covers` must be the same predicate the caller emits with: a wait planned for a block that is
/// never written is not merely wasted, it demands a barrier nothing was asked to place.
WaitInsertionPlan buildGirFrameWaitPlan(Function& function, const GirFrameAnalysis::Result& frames,
                                        const GirFrameHazardAnalysis::Result& hazards,
                                        const std::function<bool(const BasicBlock&)>& covers);

}  // namespace waitcnt
}  // namespace stinkytofu

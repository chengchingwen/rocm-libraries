/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */
#pragma once

#include "stinkytofu/analysis/asm/GirFrameAnalysis.hpp"
#include "stinkytofu/transforms/asm/waitcnt/WaitPlan.hpp"

namespace stinkytofu {
class Function;

namespace waitcnt {

/// Compute a finite-frame wait plan directly on `(ST basic block, GIR frame)` states.
WaitInsertionPlan buildGirFrameWaitPlan(Function& function, const GirFrameAnalysis::Result& frames,
                                        const GirFrameHazardAnalysis::Result& hazards);

}  // namespace waitcnt
}  // namespace stinkytofu

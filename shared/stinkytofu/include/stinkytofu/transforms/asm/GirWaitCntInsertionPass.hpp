/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */
#pragma once

#include <memory>

#include "stinkytofu/Export.hpp"

namespace stinkytofu {
class AnalysisManager;
class Function;
class Pass;
class PassContext;

/// Inserts dscnt/tensorcnt waits for frame hazards reconstructed on the scheduled ST CFG.
/// Must run after the generic waits so its ranks credit them. Returns true if it emitted any.
STINKYTOFU_EXPORT bool insertGirFrameWaits(Function& function, PassContext& passCtx,
                                           AnalysisManager& AM);

/// `insertGirFrameWaits` as a stand-alone pass; the gfx1250 pipeline runs it from
/// StinkyWaitCntInsertionPass (`WaitCntInsertionOptions::girFrameWaits`).
STINKYTOFU_EXPORT std::unique_ptr<Pass> createGirWaitCntInsertionPass();

}  // namespace stinkytofu

# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
Verification oracles for the lowering -- nothing here runs during kernel generation.

Each answers a question about a kernel INDEPENDENTLY of the code that built it, so a defect the
generator cannot see in its own terms shows up as a disagreement:

  asm_lane_eval          -- interpret the emitted address arithmetic lane by lane
  lds_region_check       -- the LDS addresses that interpretation produces, against the layout
  region_derivation_check -- which TDMSplit region each wave tile lands in, derived from geometry

The tests and the config sweeps drive these; `Lowering/` proper never imports them.
"""

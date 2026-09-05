# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
GIR -- the GEMM-dataflow IR (layer 2) of the UseLoopModel lowering.

Pure-Python (no rocisa import) so the graph + analyses + passes are unit-testable standalone.
"""

from .nodes import (Tile, Gen, Ref, Move, Mma, Mark, MARK_KINDS,
                    Pred, Bound, Goto, CondGoto, LoopBack, Trips, CondChain, Return, Block,
                    GenPhi, GenXfer, Program,
                    Region, PendingMark, BLOCK_ENTRY, BLOCK_EXIT, EARLIEST, MIDPOINT,
                    copy_unit, read_operand, first_shared_ref, covered_coords)
from .analysis import Analysis, AnalysisManager
from .analyses import (successors, Dominators, BackEdges, BackEdge, BackEdgeSet,
                       GenReaching, Reaching, SwapRegions,
                       DepDefuseAnalysis, DepDefuse,
                       RegBandAnalysis, RegBand,
                       GrIncrementRegions, LdsHazards, LdsHazardSet, Hazard,
                       FenceRegions)
from .passes import (Pass, TokensPass, CollectPendingMarksPass,
                     PlacementPass, ApplyMarksPass, ScaffoldMapPass,
                     pipeline, run_pipeline)
from .verify import (verify_gir, check_register_slots, check_rotation_waw,
                     check_block_scope_covered)
from .render import render_gir, gir_counts
from .emit_plan import EmitAction, plan_block, plan_program
from .verify_dataflow import (check_plan, check_region_coverage,
                              check_refill_splits_consumers,
                              check_source_coverage, check_register_dataflow, prologue_blocks,
                              check_address_keys)

__all__ = [
    "Tile", "Gen", "Ref", "Move", "Mma", "Mark", "MARK_KINDS",
    "Pred", "Bound", "Goto", "CondGoto", "CondChain", "Return", "Block", "GenPhi", "GenXfer", "Program",
    "Region", "PendingMark", "BLOCK_ENTRY", "BLOCK_EXIT",
    "Analysis", "AnalysisManager",
    "successors", "Dominators", "BackEdges", "BackEdge", "BackEdgeSet",
    "GenReaching", "Reaching", "SwapRegions",
    "DepDefuseAnalysis", "DepDefuse",
    "RegBandAnalysis", "RegBand",
    "GrIncrementRegions", "LdsHazards", "LdsHazardSet", "Hazard", "FenceRegions",
    "Pass", "TokensPass", "CollectPendingMarksPass",
    "PlacementPass", "ApplyMarksPass", "ScaffoldMapPass",
    "pipeline", "run_pipeline",
    "verify_gir", "check_register_slots", "check_rotation_waw", "check_block_scope_covered",
    "check_plan", "check_region_coverage", "check_refill_splits_consumers", "check_source_coverage",
    "check_register_dataflow", "prologue_blocks", "check_address_keys",
    "render_gir", "gir_counts",
    "EmitAction", "plan_block", "plan_program",
]

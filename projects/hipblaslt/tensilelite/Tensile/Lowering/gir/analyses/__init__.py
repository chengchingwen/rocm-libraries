# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
GIR analyses -- one class per module, each an `Analysis` subclass.

 cfg successors (helper), Dominators, BackEdges (+ BackEdge / BackEdgeSet)
"""

from .cfg import successors, reachable, Dominators, BackEdges, BackEdge, BackEdgeSet
from .lds_buffers import LdsBufferIds, LdsBufferIdSet, Buffer
from .frame_map import FrameMap, FrameMapping, Frame
from .swap_regions import SwapRegions
from .dep_defuse import DepDefuseAnalysis, DepDefuse
from .reg_band import RegBandAnalysis, RegBand
from .gr_increment import GrIncrementRegions
from .gl2_prefetch import Gl2PrefetchRegions
from .region_increment import RegionIncrementRegions, walk_violations, region_of
from .fence_regions import FenceRegions
from .frame_hazards import FrameHazards, FrameHazardSet, Hazard, SharedTouch, RAW, WAR, WAW
from .reg_hazards import RegHazards, RegHazardSet, RegTouch
from .short_path import (ShortPathFold, FoldVerdict, Break,
                         FOLD, SPLIT, UNSOUND, NA)
from .loop_shape import (LoopShape, Loop as LoopShapeInfo, Linear,
                         reduction_coverage_violations, PRE, POST)
from .barrier_uniformity import BarrierUniformity, BarrierViolation, PROC_SCOPES

__all__ = [
    "successors", "reachable", "Dominators", "BackEdges", "BackEdge", "BackEdgeSet",
    "LdsBufferIds", "LdsBufferIdSet", "Buffer", "FrameMap", "FrameMapping", "Frame", "SwapRegions",
    "DepDefuseAnalysis", "DepDefuse",
    "RegBandAnalysis", "RegBand",
    "GrIncrementRegions", "Gl2PrefetchRegions",
    "RegionIncrementRegions", "walk_violations", "region_of",
    "FrameHazards", "FrameHazardSet", "Hazard", "SharedTouch", "Unresolved",
    "RAW", "WAR", "WAW", "FenceRegions",
    "RegHazards", "RegHazardSet", "RegTouch",
    "ShortPathFold", "FoldVerdict", "Break", "FOLD", "SPLIT", "UNSOUND", "NA",
    "LoopShape", "LoopShapeInfo", "Linear", "reduction_coverage_violations", "PRE", "POST",
    "BarrierUniformity", "BarrierViolation", "PROC_SCOPES",
]

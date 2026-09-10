# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
GIR passes. Each pass is its own class in its own module; the ordered pipeline
is data in pipeline.py (the "backend" file).

 base Pass / StructuralPass base classes (the CFG-edit contract)
"""

from .base import Pass, StructuralPass
from .fold_short_path import FoldShortPathPass, folded
from .hoist_copies import HoistCopiesPass
from .tokens import TokensPass
from .collect_pending import CollectPendingMarksPass
from .placement import PlacementPass
from .apply_marks import ApplyMarksPass
from .record_unemitted import RecordUnemittedPass
from .scaffold_map import ScaffoldMapPass
from .early_exit import EarlyExitPass
from .split_prefetch_guard import SplitPrefetchGuardPass
from .scaffold_shape import ScaffoldShapePass
from .pipeline import pipeline, run_pipeline

__all__ = [
    "Pass", "StructuralPass", "FoldShortPathPass", "folded", "HoistCopiesPass", "TokensPass",
    "CollectPendingMarksPass",
    "PlacementPass", "ApplyMarksPass", "ScaffoldMapPass", "EarlyExitPass", "SplitPrefetchGuardPass", "ScaffoldShapePass",
    "RecordUnemittedPass",
    "pipeline", "run_pipeline",
]

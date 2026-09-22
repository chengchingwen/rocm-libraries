# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
ScaffoldShapePass -- the ONE CFG-shaping stage: fold, early-exit, split the prefetch guard, label.
"""

from __future__ import annotations

from .base import StructuralPass
from .fold_short_path import FoldShortPathPass
from .early_exit import EarlyExitPass
from .split_prefetch_guard import SplitPrefetchGuardPass
from .scaffold_map import ScaffoldMapPass


class ScaffoldShapePass(StructuralPass):
    """See module docstring.  Runs the CFG-shaping passes in order, as one stage.

    The split runs before the labelling so its new guard is named from the same table as every
    other terminator.
    """

    def __init__(self, steps=None):
        self._steps = list(steps) if steps is not None else [
            FoldShortPathPass(), EarlyExitPass(), SplitPrefetchGuardPass(), ScaffoldMapPass()]

    @property
    def name(self):
        return "ScaffoldShapePass(%s)" % ", ".join(s.name for s in self._steps)

    def run(self, prog, am):
        invalidated = []
        for step in self._steps:
            invalidated.extend(step.run(prog, am) or ())
            am.invalidate()
        return tuple(invalidated)

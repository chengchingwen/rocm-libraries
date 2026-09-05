# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
The GIR backend: pass ORDER as data, and the driver that runs it.

Everything that only READS the program is an Analysis (under `gir/analyses/`) or a check in
`gir/verify.py`; only the passes listed here may change it.
"""

from __future__ import annotations

from ..analysis import AnalysisManager
from ..verify import verify_gir
from .scaffold_shape import ScaffoldShapePass
from .hoist_copies import HoistCopiesPass
from .tokens import TokensPass
from .collect_pending import CollectPendingMarksPass
from .placement import PlacementPass
from .apply_marks import ApplyMarksPass
from .record_unemitted import RecordUnemittedPass


def pipeline():
    """The backend pass order.

    ScaffoldShapePass runs first and is the ONLY stage allowed to change the CFG: it folds the
    `T < M` arm, adds that arm's early-exit edges into the drain chain, and labels the terminators.
    Everything after it edits block bodies or program state only.
    """
    return [
        ScaffoldShapePass(),
        HoistCopiesPass(),
        TokensPass(),
        CollectPendingMarksPass(),
        PlacementPass(),
        ApplyMarksPass(),
        RecordUnemittedPass(),
    ]


def run_pipeline(prog, passes=None, verify=True):
    """Run `passes` (default: pipeline()) over `prog`, then verify.  Returns `prog`."""
    passes = passes if passes is not None else pipeline()
    am = AnalysisManager()
    for p in passes:
        p.run(prog, am)
        am.invalidate()               # coarse: a mutating pass bumped prog.version anyway
    if verify:
        verify_gir(prog)
    return prog

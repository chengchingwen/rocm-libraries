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
from .entrance_frames import EntranceFramePhisPass
from .hoist_copies import HoistCopiesPass
from .tokens import TokensPass
from .collect_pending import CollectPendingMarksPass
from .placement import PlacementPass
from .apply_marks import ApplyMarksPass
from .record_unemitted import RecordUnemittedPass
from .wait_counts import WaitCntPass


def pipeline(waitcnt_mode="GIR"):
    """The backend pass order.

    ScaffoldShapePass is the ONLY stage allowed to change the CFG: it merges the correct short-arm
    register order into the prologue/drain shape, proves equivalence, adds early-exit edges, and
    labels the terminators.

    ``GIR`` materializes the existing numeric waits. ``StinkyTofu`` leaves waits out of GIR so
    the post-schedule ST frame pipeline can own them.  The internal default remains GIR for direct
    analysis/test callers; KernelWriter always passes the solution's explicit mode.
    """
    passes = [
        ScaffoldShapePass(),
        EntranceFramePhisPass(),
        HoistCopiesPass(reads=False),
        CollectPendingMarksPass(),
        TokensPass(),
        PlacementPass(),
        ApplyMarksPass(),
        RecordUnemittedPass(),
    ]
    if waitcnt_mode == "GIR":
        passes.append(WaitCntPass())
    elif waitcnt_mode != "StinkyTofu":
        raise ValueError("unknown LoopModel wait-count mode %r" % (waitcnt_mode,))
    return passes


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

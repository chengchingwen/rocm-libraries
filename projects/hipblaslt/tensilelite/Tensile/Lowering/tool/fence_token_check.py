# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
Is every cross-agent hazard ordered -- a RAW by a fence NAMING it, a WAR/WAW by any barrier?

Separation is re-derived from the hazard edges and the fences actually present in the final body,
never from the cover that placed them.
"""

from __future__ import annotations

from ..gir.analysis import AnalysisManager
from ..gir.analyses.frame_hazards import FrameHazards, RAW
from ..gir.analyses.dep_tokens import DependenceTokens
from ..gir.analyses.fence_regions import fences_of as fences_in, separated, separating
from ..gir.nodes import Move


def unseparated_edges(prog, am=None, drop=None):
    """The cross-agent edges no fence orders.  Non-empty is a correctness failure.

    `drop` is `(block, index, token)` to remove from that fence first -- the minimality probe."""
    am = am or AnalysisManager()
    hz, tokens = am.get(FrameHazards(), prog), am.get(DependenceTokens(), prog)
    fences = fences_in(prog)
    if drop is not None:
        blk, idx, tok = drop
        if idx in fences.get(blk, {}):
            fences[blk][idx] = fences[blk][idx] - {tok}
    n_of = {lab: len(b.body) + 1 for lab, b in prog.blocks.items()}

    def loose(h):
        # Only a RAW is stamped; a WAR/WAW just needs a barrier between its ends.
        if h.kind != RAW:
            return not separating(h, fences, n_of.get)
        pins = set(tokens.tokens_for(h.consumer.ref))
        return bool(pins) and not separated(h, fences, n_of.get, pins)

    return [h for h in hz.needing_fence() if loose(h)]


def widened_stamps(prog, am=None):
    """`[(operand, block, index, tokens)]` -- refs naming several buffers.

    A singleton stamp is exact; a wider one is a MAY-set, legitimate only where the ref leaves a
    region axis unpinned or rides the loop-carried ring, and worth listing either way."""
    am = am or AnalysisManager()
    tokens = am.get(DependenceTokens(), prog)
    out = []
    for lab, blk in prog.blocks.items():
        for i, inst in enumerate(blk.body):
            if not isinstance(inst, Move):
                continue
            for ref in list(inst.srcs) + list(inst.dsts):
                ids = tokens.tokens_for(ref)
                if ref.tile.space == "shared" and len(ids) > 1:
                    out.append((ref.tile.operand, lab, i, ids))
    return out

# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
BufferLiveness -- which LDS buffers span each program point, as a fixpoint over the CFG.
"""

from __future__ import annotations

from .cfg import successors


def _predecessors(succ):
    preds = {lab: [] for lab in succ}
    for lab, ss in succ.items():
        for s in ss:
            preds[s].append(lab)
    return preds


class BufferLiveness:
    """Live ranges of LDS BUFFERS -- the objects that have a lifetime.

    A token id is a logical name, free to choose, so it cannot be the subject of a dataflow; the
    `Buffer` a Ref resolves to can.  `written` carries buffers some path has filled on the way in,
    `read` those some path still consumes on the way out, and a buffer spans a point when both hold.
    Union lattices over the real CFG, so a loop's wrap-around arrives on the back edge.
    """

    def __init__(self, prog, writes, reads):
        self._prog = prog
        succ = successors(prog)
        self._written = self._solve(_predecessors(succ), writes, forward=True)
        self._read = self._solve(succ, reads, forward=False)

    def _transfer(self, lab, value, gen, forward):
        """Per-slot values through one block; returns (at, out)."""
        n = len(self._prog.blocks[lab].body)
        at, v, g = {}, set(value), gen.get(lab, {})
        steps = range(n + 1) if forward else range(n, -1, -1)
        for s in steps:
            at[s] = frozenset(v)
            v |= g.get(s if forward else s - 1, frozenset())
        return at, frozenset(v)

    def _solve(self, edges, gen, forward):
        out = {lab: frozenset() for lab in self._prog.blocks}
        at = {}
        changed = True
        while changed:
            changed = False
            for lab in self._prog.blocks:
                incoming = frozenset().union(*(out[e] for e in edges[lab])) \
                    if edges[lab] else frozenset()
                slots, leaving = self._transfer(lab, incoming, gen, forward)
                at[lab] = slots
                if leaving != out[lab]:
                    out[lab] = leaving
                    changed = True
        return at

    def across(self, block, slot):
        """The buffers whose live range spans `(block, slot)`."""
        return self._written[block][slot] & self._read[block][slot]

    def live_anywhere(self):
        """Every buffer that is live at some point -- the ranges a token numbering must separate."""
        return frozenset().union(*(frozenset().union(*w.values()) if w else frozenset()
                                   for w in self._written.values())) if self._written \
            else frozenset()

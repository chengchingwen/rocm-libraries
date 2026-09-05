# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
GenReaching -- the concrete generation each Move/Mma access uses.
"""

from __future__ import annotations

from ..nodes import Move, Mma
from ..analysis import Analysis
from .cfg import BackEdges


class Reaching:
    """Query API over the reaching-generation result.  The internal map is identity-keyed on the
    exact Ref/xfer objects in `prog`; `prog` is retained so those identities remain valid."""

    def __init__(self, prog, by_ref, by_xfer, relative_blocks=()):
        self._prog = prog
        self._by_ref = by_ref        # id(ref) -> generation int
        self._by_xfer = by_xfer      # (block_label, gen_id) -> next-trip entry generation
        self._relative = frozenset(relative_blocks)

    def of(self, ref):
        """Concrete generation `ref` uses, or None if `ref` carries no generation.

        CAUTION on the base: for a block reported by `is_relative`, the value is on an ARBITRARY
        base (see `_entry_of`) -- it is meaningful only RELATIVE to other values in that same
        block's frame, never as an absolute buffer identity, and never comparable across blocks.
        A consumer that needs an absolute generation must check `is_relative` and refuse."""
        return self._by_ref.get(id(ref))

    def is_relative(self, block_label) -> bool:
        """True when `block_label`'s entry generations could not be MERGED from its predecessors
        and were assigned an arbitrary base.  `of()` values inside such a block are relative."""
        return block_label in self._relative

    def xfer(self, block_label, gen):
        """Next-trip entry generation for the back-edge leaving `block_label` for `gen`."""
        return self._by_xfer.get((block_label, gen.id))


def _resolve_phi_entry(blk):
    """Per-Gen value on `blk`'s entry, at ANALYSIS granularity."""
    return {phi.gen: phi.entry_val for phi in blk.phis}


# Base assigned when a block's entry generations cannot be merged from its predecessors.  It is
# ARBITRARY -- reported through `Reaching.is_relative` rather than passed off as a fact.
RELATIVE_BASE = 0


def _entry_of(blk, entries, relative=()):
    """(per-Gen entry map, is_relative) for `blk` -- a real MERGE over its known predecessors.

    A header carries the phis, so its entry is declared.  Otherwise we merge: every known
    predecessor that AGREES on a Gen's value contributes it; a Gen the predecessors disagree on --
    or that no predecessor supplies -- gets `RELATIVE_BASE` and the block is flagged RELATIVE.
    """
    if blk.phis:
        return _resolve_phi_entry(blk), False
    known = [entries[p] for p in blk.preds if p in entries]
    if not known:
        # the entry block: no incoming generations by definition (its refs are absolute).
        return {}, bool(blk.preds)
    # inherited: any contributing predecessor on an arbitrary base makes this one arbitrary too.
    merged, relative = {}, any(p in relative for p in blk.preds if p in entries)
    for gen in {g for m in known for g in m}:
        vals = [m[gen] for m in known if gen in m]
        if len(vals) == len(known) and len(set(vals)) == 1:
            merged[gen] = vals[0]                    # every pred agrees -- a real merged value
        else:
            merged[gen] = RELATIVE_BASE              # no compile-time merge exists
            relative = True
    return merged, relative


class GenReaching(Analysis):
    """Resolve, for every access, which concrete buffer generation it reads or writes."""
    def run(self, prog, am):
        back = am.get(BackEdges(), prog)
        by_ref, by_xfer = {}, {}
        entries, relative = {}, set()
        for blk in prog.walk_rpo():
            entry, is_rel = _entry_of(blk, entries, relative)
            entries[blk.label] = entry
            if is_rel:
                relative.add(blk.label)
            for inst in blk.body:
                refs = ()
                if isinstance(inst, (Move, Mma)):
                    refs = tuple(inst.srcs) + tuple(inst.dsts)
                for ref in refs:
                    if getattr(ref, "gen", None) is not None:
                        base = entry.get(ref.gen, 0)
                        by_ref[id(ref)] = (base + ref.gdelta) % max(1, ref.gen.ring)
                    elif getattr(ref, "abs_gen", None) is not None:
                        by_ref[id(ref)] = ref.abs_gen
            for be in back.leaving(blk.label):
                for xf in be.xfers:
                    base = entry.get(xf.gen, 0)
                    by_xfer[(blk.label, xf.gen.id)] = (base + xf.adv) % max(1, xf.ring)
        return Reaching(prog, by_ref, by_xfer, relative)

# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
RegHazards -- the VGPR rotation hazard edges, the register-side twin of LdsHazards.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..nodes import Move, Mma
from ..analysis import Analysis
from .lds_hazards import Hazard, RAW, WAR, WAW, _hazard_kind, _reachable


#: Tile axes that name WHICH register, as opposed to which position within the rotation.
def _free_coord(ref):
    """Per tile axis that survives the rotation, the value this ref pins, or None for any.

    `Ref.covers` is the reference's own declaration that ONE instruction fills several coordinates
    -- a folded read -- so a covered axis is a MAY-set over its whole span, the same treatment
    `lds_hazards._regions_of` gives it.  Pinning it to the written coordinate loses every edge to
    the consumers of the others."""
    spanned = {a for a, _f in (getattr(ref, "covers", ()) or ())}
    return tuple((a, None if a in spanned else v)
                 for a, v in (getattr(ref.tile, "coord", ()) or ())
                 if not str(a).startswith("K"))


@dataclass(frozen=True)
class RegTouch:
    """One register touch, with the fields `Hazard` reads.  Mirrors `SharedTouch`."""
    block:    str
    pos:      int
    inst:     object
    ref:      object
    is_write: bool
    operand:  str
    ring_pos: tuple         # (group, slot) -- which rotation position
    coord:    tuple         # per free axis, the pinned value or None for any


def _accesses(blk):
    """Every ROTATING register touch in `blk`, in program order.

    A ref with no slot is not on a ring -- the accumulator -- so it has no rotation hazard and
    its ordering is program order plus the hardware's in-order issue."""
    for pos, inst in enumerate(blk.body):
        if not isinstance(inst, (Move, Mma)):
            continue
        for ref, is_write in ([(r, False) for r in inst.srcs] + [(r, True) for r in inst.dsts]):
            if ref.tile.space != "register" or ref.slot is None:
                continue
            yield RegTouch(block=blk.label, pos=pos, inst=inst, ref=ref, is_write=is_write,
                           operand=ref.tile.operand, ring_pos=(ref.group, ref.slot),
                           coord=_free_coord(ref))


def _same_register(a, b):
    """MAY these two touch one register?  An unpinned axis matches every value of it."""
    if a.operand != b.operand or a.ring_pos != b.ring_pos:
        return False
    other = dict(b.coord)
    return all(axis not in other or v is None or other[axis] is None or v == other[axis]
               for axis, v in a.coord)


class RegHazardSet:
    """Query API over the register hazard result.  Mirrors `LdsHazardSet` minus the fence arm:
    a register is wave-private, so no edge here is ever cross-agent."""

    def __init__(self, edges):
        self._edges = tuple(edges)

    def __iter__(self):
        return iter(self._edges)

    def __len__(self):
        return len(self._edges)

    def edges(self, *, kind=None, block=None):
        out = self._edges
        if kind is not None:
            out = [h for h in out if h.kind == kind]
        if block is not None:
            out = [h for h in out if block in (h.producer.block, h.consumer.block)]
        return tuple(out)


class RegHazards(Analysis):
    """See module docstring.  Pure; returns a `RegHazardSet`."""

    def run(self, prog, am):
        by_block = {blk.label: list(_accesses(blk)) for blk in prog.blocks.values()}
        rch = _reachable(prog)
        edges = []
        for label, accesses in by_block.items():
            repeats = label in rch.get(label, ())
            for i, a in enumerate(accesses):
                for b in accesses[i + 1:]:
                    self._pair(a, b, edges, repeats)
        self._cross_block(by_block, rch, edges)
        return RegHazardSet(edges)

    @staticmethod
    def _ring(touch):
        return max(1, touch.ref.reg_ring or 1)

    def _pair(self, a, b, edges, repeats):
        """Both directions on one register: the vacating read then the refill, and the refill then
        the next trip's read of it."""
        if not _same_register(a, b) or _hazard_kind(a, b) is None:
            return
        both = ((a, b), (b, a))
        for producer, consumer in (both if repeats else both[:1]):
            kind = _hazard_kind(producer, consumer)
            if kind is None:
                continue
            edges.append(Hazard(kind=kind, producer=producer, consumer=consumer,
                                ring=self._ring(producer), distance=0, cross_agent=False))

    def _cross_block(self, by_block, rch, edges):
        """A prologue fill and the steady read of it: real, and with no trip distance to speak of."""
        for pb, pas in by_block.items():
            for cb, cas in by_block.items():
                if pb == cb or cb not in rch.get(pb, ()):
                    continue
                for p in pas:
                    for c in cas:
                        kind = _hazard_kind(p, c)
                        if kind is None or not _same_register(p, c):
                            continue
                        edges.append(Hazard(kind=kind, producer=p, consumer=c,
                                            ring=self._ring(p), distance=None,
                                            cross_agent=False, cross_block=True))

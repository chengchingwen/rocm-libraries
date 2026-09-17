# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
HoistCopiesPass -- issue each movement at its earliest hazard-legal slot.
"""

from __future__ import annotations

from ..nodes import Move
from ..analyses.frame_hazards import FrameHazards
from ..analyses.reg_hazards import RegHazards
from .base import Pass


def _is_copy(inst):
    """A Move that FILLS shared storage -- the instruction whose issue point we are choosing."""
    return isinstance(inst, Move) and any(r.tile.space == "shared" for r in inst.dsts)


def _is_read(inst):
    """A Move that fills REGISTERS from shared: the other half of the pipeline."""
    return isinstance(inst, Move) and any(r.tile.space == "register" for r in inst.dsts)


def _earliest(inst, label, *hazard_sets):
    """The first index `inst` may occupy in its block, over every hazard set that binds it.

    Only a same-trip edge already in program order constrains it: that is a reader of the storage
    this movement overwrites, and it must stay behind that reader.  An edge at a trip distance is
    discharged by the trip boundary, not by this movement's position."""
    return max((h.producer.pos + 1 for hazards in hazard_sets for h in hazards
                if h.consumer.inst is inst and h.consumer.block == label
                and not h.cross_block and h.same_trip and h.in_program_order),
               default=0)


def hoisted(blk, lds, reg=(), group=True, reads=False):
    """`blk.body` with the movements moved up, or None if none moves.

    `group` sends the COPIES to one point, the latest of their individual earliests, so they keep
    sharing a barrier; letting each chase its own floor straddles another operand's last read and
    buys a second proc-scoped sync per trip.  `reads` lifts the reads too, which lets a fence sit
    past them instead of draining LDS at the trip top."""
    body = blk.body
    at = {i: _earliest(inst, blk.label, lds)
          for i, inst in enumerate(body) if _is_copy(inst)}
    if group and at:
        slot = max(at.values())
        at = {i: slot for i in at} if slot < min(at) else {}
    classes = [sorted(at)]
    if reads:
        reads_at = {i: _earliest(inst, blk.label, lds, reg)
                    for i, inst in enumerate(body) if _is_read(inst)}
        at.update(reads_at)
        classes.append(sorted(reads_at))
    # INTRA-CLASS ORDER IS NOT FREE: two copies to one ring must land in their original sequence,
    # and no token downstream can express an order that was already wrong.  A member that does NOT
    # move keeps index `i`, and a mover landing on slot `i` is emitted BEFORE it -- so the floor it
    # leaves behind is `i + 1`, not `i`.
    for indices in classes:
        floor = -1
        for i in indices:
            at[i] = max(at[i], floor)
            floor = i + 1 if at[i] >= i else at[i]
    at = {i: s for i, s in at.items() if s < i}
    if not at:
        return None
    # Indices are read off the ORIGINAL body: every producer precedes the movement it blocks, so
    # lifting the movements out does not shift any of them.  Sorted, so two landing on one slot
    # keep their original order.
    pending = {}
    for i in sorted(at):
        pending.setdefault(at[i], []).append(body[i])
    out = []
    for i, inst in enumerate(body):
        out.extend(pending.pop(i, ()))
        if i not in at:
            out.append(inst)
    for s in sorted(pending):
        out.extend(pending[s])
    return out


class HoistCopiesPass(Pass):
    """See module docstring.  Runs before the tokens so every later analysis sees this order."""

    def __init__(self, group=True, reads=False):
        self.group = group
        self.reads = reads

    def run(self, prog, am):
        lds, reg = am.get(FrameHazards(), prog), am.get(RegHazards(), prog)
        changed = False
        for blk in prog.blocks.values():
            order = hoisted(blk, lds, reg, self.group, self.reads)
            if order is not None:
                blk.body[:] = order
                changed = True
        if changed:
            prog.bump()
        return ("body",) if changed else ()

# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
HoistCopiesPass -- issue each movement at its earliest hazard-legal slot.
"""

from __future__ import annotations

from ..nodes import Move
from ..analyses.frame_hazards import SchedulingFrameHazards
from ..analyses.fence_regions import FenceRegions
from ..analyses.reg_hazards import RegHazards
from .base import Pass


def _is_copy(inst):
    """A Move that FILLS shared storage -- the instruction whose issue point we are choosing."""
    return isinstance(inst, Move) and any(r.tile.space == "shared" for r in inst.dsts)


def _is_read(inst):
    """A Move that fills REGISTERS from shared: the other half of the pipeline."""
    return isinstance(inst, Move) and any(r.tile.space == "register" for r in inst.dsts)


def _read_generation(inst):
    """The shared source generation relation this read consumes."""
    ref = next((ref for ref in getattr(inst, "srcs", ())
                if ref.tile.space == "shared"), None)
    if ref is None:
        return None
    absolute = getattr(ref, "abs_gen", None)
    return ("abs", int(absolute)) if absolute is not None \
        else ("rel", int(getattr(ref, "gdelta", 0) or 0))


def _is_split_copy(inst):
    """A shared fill pinned to a split-region coordinate."""
    return _is_copy(inst) and any(ref.tile.coord for ref in inst.dsts
                                  if ref.tile.space == "shared")


def _earliest(inst, label, *hazard_sets):
    """The first index `inst` may occupy in its block, over every hazard set that binds it.

    Only a same-trip edge already in program order constrains it: that is a reader of the storage
    this movement overwrites, and it must stay behind that reader.  An edge at a trip distance is
    discharged by the trip boundary, not by this movement's position."""
    return max((h.producer.pos + 1 for hazards in hazard_sets for h in hazards
                if h.consumer.inst is inst and h.consumer.block == label
                and not h.cross_block and h.same_trip and h.in_program_order),
               default=0)


def _raw_fence_profile(prog, am):
    """Per RAW edge, semantic work preceding its planned fence."""
    out = {}
    for pending in am.get(FenceRegions(), prog):
        raw = [hazard for hazard in pending.edges if hazard.kind == "RAW"]
        if not raw:
            continue
        body = prog.block(pending.region.block).body
        before = pending.region.before
        slot = body.index(before) if before in body else len(body)
        work = sum(not _is_read(inst) for inst in body[:slot])
        for hazard in raw:
            key = (id(hazard.producer.inst), id(hazard.consumer.inst),
                   hazard.kind, int(hazard.gap), bool(hazard.cross_agent))
            out[key] = max(work, out.get(key, -1))
    return out


def _moves_raw_fence_earlier(before, after):
    return (set(before) != set(after)
            or any(work < before[key] for key, work in after.items()))


def hoisted(blk, lds, reg=(), group=True, reads=False, copies=True):
    """`blk.body` with the movements moved up, or None if none moves.

    `group` sends unsplit copies to one point, the latest of their individual earliests, so they
    keep sharing a barrier. A region-pinned TDM copy keeps its own hazard floor so split movement
    follows the WMMA traversal instead of being batched across regions. `reads` lifts register
    reads too."""
    body = blk.body
    at = ({i: _earliest(inst, blk.label, lds)
           for i, inst in enumerate(body) if _is_copy(inst)}
          if copies else {})
    if group and at:
        grouped = {i: slot for i, slot in at.items() if not _is_split_copy(body[i])}
        if grouped:
            slot = max(grouped.values())
            if slot < min(grouped):
                at.update({i: slot for i in grouped})
            else:
                for i in grouped:
                    at.pop(i)
    classes = [sorted(at)]
    if reads:
        read_indices = [i for i, inst in enumerate(body) if _is_read(inst)]
        first_generation = _read_generation(body[read_indices[0]]) if read_indices else None
        reads_at = {i: _earliest(inst, blk.label, lds, reg)
                    for i, inst in enumerate(body)
                    if _is_read(inst) and _read_generation(inst) == first_generation}
        at.update(reads_at)
        classes.append(sorted(reads_at))
    # COPY ORDER IS NOT FREE, INCLUDING A COPY THAT DOES NOT MOVE.  Split copies have individual
    # floors while an unsplit movement may stay put; without putting that stationary copy into the
    # ordering scan, a later split copy can hoist across it and reverse one peel generation
    # (MX,A0,A1 -> A0,A1,MX). No downstream token can recover FIFO order after that reversal.
    copy_movers = set(classes[0])
    floor = -1
    for i, inst in enumerate(body):
        if not _is_copy(inst):
            continue
        if i not in copy_movers:
            floor = i + 1
            continue
        at[i] = max(at[i], floor)
        floor = i + 1 if at[i] >= i else at[i]

    # Reads are one independent movement class. Their relative order is protected the same way;
    # unlike copies, non-moving reads do not define tensor issue FIFO order.
    for indices in classes[1:]:
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
        changed = False

        def apply(*, copies, reads):
            nonlocal changed
            lds, reg = am.get(SchedulingFrameHazards(), prog), am.get(RegHazards(), prog)
            moved = False
            for blk in prog.blocks.values():
                order = hoisted(blk, lds, reg, self.group, reads, copies)
                if order is not None:
                    blk.body[:] = order
                    moved = True
            if moved:
                prog.bump()
                am.invalidate()
                changed = True
            return moved

        if self.reads:
            original = {label: list(block.body) for label, block in prog.blocks.items()}
            profile = _raw_fence_profile(prog, am)
            if apply(copies=False, reads=True):
                moved_profile = _raw_fence_profile(prog, am)
                if _moves_raw_fence_earlier(profile, moved_profile):
                    for label, body in original.items():
                        prog.block(label).body[:] = body
                    prog.bump()
                    am.invalidate()
                    changed = False
        apply(copies=True, reads=False)
        return ("body",) if changed else ()

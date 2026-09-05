# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
FenceRegions -- the MINIMAL set of proc-scoped fences covering the cross-wave LDS hazards.
"""

from __future__ import annotations

from ..nodes import Mark
from ..nodes import Region, PendingMark, BLOCK_ENTRY, BLOCK_EXIT
from ..analysis import Analysis
from .dep_tokens import DependenceTokens
from .lds_hazards import LdsHazards


def _admissible_slots(hazard, n_slots, block=None):
    """The slots of `block` at which one fence discharges `hazard`.

    `block` defaults to the (single) block both ends live in.  For a CROSS-BLOCK hazard it must be
    given, because the two ends offer different windows and only one of them is being covered."""
    if hazard.cross_block:
        if block == hazard.consumer.block:
            return frozenset(s for s in range(n_slots) if s <= hazard.consumer.pos)
        return frozenset()
    a, b = hazard.producer.pos, hazard.consumer.pos
    if hazard.same_trip and hazard.in_program_order:
        return frozenset(s for s in range(n_slots) if a < s <= b)
    # A wrap edge separates at any slot on the way round the loop: after the copy, or before the
    # read in the consuming trip.
    return frozenset(s for s in range(n_slots) if s > a or s <= b)


def _cover(hazards, n_slots, block):
    """Where each hazard's fence goes: the LATEST slot that still discharges it.

    The objective is the emitted waitcnt VALUE, not the barrier count. A wait retires the queue
    down to the producer it names, so every load issued between the fence and that producer stays
    in flight -- the later the fence, the more of them, and the larger the count the backend can
    derive. Placing a fence early is what forces the drain.

    Returns {slot: [hazards it discharges]}. Raises if some hazard admits no slot at all -- the
    emitted order would make the edge undischargeable, which is a defect in the schedule.
    """
    admissible = []
    for h in hazards:
        slots = _admissible_slots(h, n_slots, block)
        if not slots:
            raise RuntimeError(
                f"no slot discharges the {h.kind} edge {h.producer.operand} "
                f"pos {h.producer.pos} -> {h.consumer.pos} (distance {h.distance}): the emitted "
                f"order leaves it undischargeable, so sigma is illegal ")
        admissible.append(slots)

    chosen, remaining = {}, set(range(len(hazards)))
    while remaining:
        # One fence serves as many edges as it can -- a fence per edge measures far worse on a
        # large body -- but among the slots that tie, the LATEST, because every load issued between
        # the fence and the producer it names is depth the backend can keep.
        best = max(range(n_slots),
                   key=lambda s: (sum(1 for i in remaining if s in admissible[i]), s))
        covered = [i for i in remaining if best in admissible[i]]
        chosen[best] = [hazards[i] for i in covered]
        remaining -= set(covered)
    return chosen


def _spanning(tokens, covered):
    """The token ids a fence must name: both ends of the edges the COVER charged to it.

    Not every edge whose window it splits. The cover already gives each edge a fence, so an edge
    charged elsewhere is held there; naming it here only deepens this fence's wait."""
    return frozenset(t for h in covered
                     for e in (h.producer, h.consumer) for t in _whole_movement(tokens, e))


def _whole_movement(tokens, end):
    """Every token the endpoint's INSTRUCTION touches, not just the one ref of it.

    A Phi-fused copy is one movement that completes as one event, so a fence guarding it must name
    every member: naming one leaves the others in flight when the barrier releases the waves that
    read them."""
    inst = getattr(end, "inst", None)
    refs = ([r for r in list(inst.srcs) + list(inst.dsts) if r.tile.space == "shared"]
            if inst is not None else [end.ref])
    return frozenset(t for r in (refs or [end.ref]) for t in tokens.tokens_for(r))


def _region_at(body, slot):
    """A Region PINNED to one slot: after `body[slot-1]`, before `body[slot]`.

    The cover has already chosen the point, so unlike a swap this Mark has no freedom left -- the
    window is degenerate on purpose and PlacementPass resolves it exactly."""
    return Region(block=None,
                  after=body[slot - 1] if slot > 0 else BLOCK_ENTRY,
                  before=body[slot] if slot < len(body) else BLOCK_EXIT)


class FenceRegions(Analysis):
    """See module docstring.  Pure; returns `[PendingMark]` of `Mark('fence', ...)`."""

    def run(self, prog, am):
        hazards = am.get(LdsHazards(), prog)
        # the OBLIGATION token, not the buffer id: a fence names what it must order, and an access
        # that stands in no hazard on a buffer never puts that buffer on a fence.
        tokens = am.get(DependenceTokens(), prog)
        # A hazard is fenced in its CONSUMER's block.  Not "somewhere on the path": which other
        # block happens to carry a fence is a property of the block layout, so discharging against
        # it makes the emitted sync depend on how the CFG is cut rather than on the dependency.
        placed = []
        for blk in prog.walk_rpo():
            mine = [h for h in hazards.needing_fence()
                    if (h.consumer.block == blk.label
                        and (h.cross_block or h.producer.block == blk.label))]
            if not mine:
                continue
            n_slots = len(blk.body) + 1
            # Covered PER OPERAND.  One cover over the whole block lands A's and B's edges on a
            # single fence, which then names both operands' buffers and can only drain. Splitting
            # finer than this (per storage) measures worse: more fences, each still naming a copy.
            groups = {}
            for h in mine:
                groups.setdefault(
                    frozenset((h.producer.operand, h.consumer.operand)), []).append(h)
            for _key, edges in sorted(groups.items(), key=lambda kv: sorted(kv[0])):
                for slot, covered in sorted(_cover(edges, n_slots, blk.label).items()):
                    placed.append((blk, slot, covered, _spanning(tokens, covered)))

        pending = []
        for blk, slot, covered, spanning in placed:
            region = _region_at(blk.body, slot)
            pending.append(PendingMark(
                mark=Mark("fence", {
                    # the buffers this fence orders -- a MUST-set on a fence ("this fence
                    # orders all of these"), which is the reading barriers already carry.
                    "buffers": tuple(sorted({h.producer.operand for h in covered}
                                            | {h.consumer.operand for h in covered})),
                    # The token ids, which is what the backend needs -- `buffers` names
                    # operands and is only the readable tag.
                    "tokens": tuple(sorted(spanning)),
                    "scope": "block",
                    "kinds": tuple(sorted({h.kind for h in covered})),
                    "edges": len(covered),
                }),
                region=Region(block=blk.label, after=region.after, before=region.before,
                              policy=region.policy)))
        return pending

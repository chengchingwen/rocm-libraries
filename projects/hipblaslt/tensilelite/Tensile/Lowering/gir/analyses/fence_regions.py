# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
FenceRegions -- WHERE the proc-scoped fences go.  Position only; `FenceTokensPass` names them.

A fence sits at ONE of its hazard's own two ends, never on a block between them: an intermediate
block lies on some path, and the producer can re-reach the consumer around it via the back edge.
"""

from __future__ import annotations

from ..nodes import Mark, Move
from ..nodes import Region, PendingMark, BLOCK_ENTRY, BLOCK_EXIT
from ..analysis import Analysis
from .frame_hazards import FrameHazards, RAW
from .frame_map import FrameMap


def _admissible_slots(hazard, n_slots, block=None):
    """The slots of `block` at which one fence discharges `hazard`.

    A fence discharges an edge only at one of the edge's OWN ends, so a block that is neither
    offers nothing -- including for a same-block edge asked about somewhere else."""
    if hazard.cross_block:
        if block == hazard.producer.block:
            return frozenset(s for s in range(n_slots) if s > hazard.producer.pos)
        if block == hazard.consumer.block:
            return frozenset(s for s in range(n_slots) if s <= hazard.consumer.pos)
        return frozenset()
    if block is not None and block != hazard.producer.block:
        return frozenset()
    a, b = hazard.producer.pos, hazard.consumer.pos
    if hazard.same_trip and hazard.in_program_order:
        return frozenset(s for s in range(n_slots) if a < s <= b)
    return frozenset(s for s in range(n_slots) if s > a or s <= b)


def _fills_shared(end) -> bool:
    """Does this endpoint's instruction fill shared storage from global?"""
    inst = getattr(end, "inst", None)
    return (isinstance(inst, Move)
            and any(r.tile.space == "global" for r in inst.srcs)
            and any(r.tile.space == "shared" for r in inst.dsts))


def _drives_tensorcnt(hazard) -> bool:
    """Does discharging this edge cost a MEMORY WAIT rather than just a barrier?

    Its producer fills shared storage from global, so the fence covering it names a TDM token and
    lowers to `s_wait_tensorcnt`: everything between that load and the fence is latency it hides."""
    return _fills_shared(hazard.producer)


def _fenced_in(hazard, repeating):
    """Both ends of a cross-block edge: the consumer's names what it reads, and a one-shot
    producer needs its own because control leaves it before the downstream fence runs."""
    if not hazard.cross_block:
        return frozenset((hazard.consumer.block,))
    if hazard.producer.block in repeating:
        return frozenset((hazard.consumer.block,))
    return frozenset((hazard.producer.block, hazard.consumer.block))


def _unshared(slot, covered, n_slots, block, taken):
    """`slot`, or the latest other slot legal for every covered edge and not already fenced."""
    if slot not in taken:
        return slot
    free = set.intersection(*(set(_admissible_slots(h, n_slots, block)) for h in covered)) - taken
    return max(free, default=slot)


def _cover(hazards, n_slots, block):
    """The slots to fence, so that every hazard has a fence between its ends.

    Returns {slot: [hazards it discharges]}.  Raises if some hazard admits no slot at all -- that
    means the emitted order itself makes the edge undischargeable (an illegal sigma), a defect in
    the schedule rather than something to model over with a fence somewhere harmless.
    """
    admissible = []
    for h in hazards:
        slots = _admissible_slots(h, n_slots, block)
        if not slots:
            raise RuntimeError(
                f"no slot discharges the {h.kind} edge {h.producer.operand} "
                f"pos {h.producer.pos} -> {h.consumer.pos} (gap {h.gap}): the emitted "
                f"order leaves it undischargeable, so sigma is illegal ")
        admissible.append(slots)

    chosen, remaining = {}, set(range(len(hazards)))
    # RAW FIRST: only a RAW stamps the barrier, so its slot is the one that costs a wait and it is
    # placed as LATE as its window allows -- the loads issued after it stay in flight.  WAR and WAW
    # need a barrier but no stamp, so they RIDE a RAW barrier whenever one is legal for them;
    # StinkyTofu makes every barrier a waitcnt point, so riding one orders them for free.
    for stamps in (True, False):
        tier = {i for i in remaining if (hazards[i].kind == RAW) is stamps}
        while tier:
            if stamps:
                best = max(s for i in tier for s in admissible[i])
            else:
                ride = [s for s in chosen if any(s in admissible[i] for i in tier)]
                best = (max(ride) if ride else
                        max(range(n_slots),
                            key=lambda s: (sum(1 for i in tier if s in admissible[i]), -s)))
            covered = [i for i in remaining if best in admissible[i]]
            chosen.setdefault(best, []).extend(hazards[i] for i in covered)
            remaining -= set(covered)
            tier -= set(covered)
    return chosen


def _legal(edges, n_slots, block):
    """The slots that discharge EVERY edge -- the window one barrier would have to sit in."""
    return set.intersection(*(set(_admissible_slots(h, n_slots, block)) for h in edges))


def _one_barrier(covers, n_slots, block):
    """The per-buffer covers merged wherever ONE slot discharges several of them.

    The cover stays per buffer; what merges is the BARRIER, of which a block needs one per slot."""
    merged = []
    for edges in covers:
        legal = _legal(edges, n_slots, block)
        for i, (have, kept) in enumerate(merged):
            if have & legal:
                merged[i] = (have & legal, kept + list(edges))
                break
        else:
            merged.append((legal, list(edges)))
    return [edges for _slots, edges in merged]


def _cover_slot(edges, n_slots, block):
    """Where a merged barrier goes: the latest slot that still discharges every edge."""
    return max(_legal(edges, n_slots, block))


def _windows(prog, covered, only=None):
    """Every Region that discharges ALL of `covered`, in EVERY block that offers one.

    A wrap has two disjoint halves -- the end of the producing trip and the top of the consuming
    one -- and a cross-block edge offers a window at each end, so the freedom spans blocks and the
    naming layer is the one that spends it."""
    out = []
    for lab, blk in prog.blocks.items():
        if only is not None and lab != only:
            continue
        n_slots = len(blk.body) + 1
        legal = set.intersection(*(set(_admissible_slots(h, n_slots, lab)) for h in covered))
        # A barrier belongs INSIDE the body: at the boundary it sits between the last node and the
        # branch, where the scheduler has nothing left to pick.  Kept only if it is the sole slot.
        legal = (legal - {n_slots - 1}) or legal
        for run in _runs(sorted(legal)):
            out.append(Region(block=lab,
                              after=blk.body[run[0] - 1] if run[0] > 0 else BLOCK_ENTRY,
                              before=blk.body[run[-1]] if run[-1] < len(blk.body) else BLOCK_EXIT))
    return out


def _runs(slots):
    """`slots` split into maximal contiguous runs -- a wrap's two halves are two runs."""
    out = []
    for s in slots:
        if out and s == out[-1][-1] + 1:
            out[-1].append(s)
        else:
            out.append([s])
    return out


def fences_of(prog):
    """`{block: {index: frozenset(token ids)}}` for every fence Mark in the body."""
    out = {}
    for lab, blk in prog.blocks.items():
        for i, node in enumerate(blk.body):
            if isinstance(node, Mark) and node.kind == "fence":
                out.setdefault(lab, {})[i] = frozenset(
                    int(t) for t in (node.at or {}).get("tokens", ()))
    return out


def separating(hazard, fences, n_slots_of):
    """The `(block, index)` fences that stand between this hazard's ends.

    THE one definition of separation, shared by the cover, the verifier and the oracle: a fence at
    one of the edge's OWN ends, in that end's admissible window."""
    return [(b, i) for b in (hazard.producer.block, hazard.consumer.block)
            for i in fences.get(b, {})
            if i in _admissible_slots(hazard, n_slots_of(b), b)]


def separated(hazard, fences, n_slots_of, pins):
    """Is the hazard ordered by a separating fence that pins both its ends?

    A fence naming NOTHING is a full drain -- the same as naming every token -- so it pins this
    edge like any other.  Conservative, never wrong, and the fallback when no name covers."""
    return any(not fences[b][i] or (fences[b][i] & pins)
               for b, i in separating(hazard, fences, n_slots_of))


def _repeating(fm):
    """The blocks that can execute more than once -- their frame node reaches itself."""
    return frozenset(b for b, f in fm.nodes() if (b, f) in fm.reaches((b, f)))


class FenceRegions(Analysis):
    """See module docstring.  Pure; returns `[PendingMark]` of `Mark('fence', ...)`.

    `merge` collapses the tier-1 RAW covers that share a legal slot onto ONE barrier.  It is only
    meaningful for tier 1: a WAR/WAW already rides a RAW barrier rather than opening its own."""

    def __init__(self, merge=True):
        self.merge = merge

    @property
    def cache_key(self):
        return "%s(merge=%s)" % (type(self).__name__, self.merge)

    def run(self, prog, am):
        hazards = am.get(FrameHazards(), prog)
        fm = am.get(FrameMap(), prog)
        repeating = _repeating(fm)

        placed = []
        for blk in prog.walk_rpo():
            mine = [h for h in hazards.needing_fence()
                    if blk.label in _fenced_in(h, repeating)]
            if not mine:
                continue
            n_slots = len(blk.body) + 1
            # COVER PER BUFFER -- `(operand, region, ring)`, the one numbering -- not per operand.
            # Grouping A's two regions onto one fence makes it name both, and then every load defs
            # one of its names and the wait can never leave anything in flight.
            frame = fm.entrances(blk.label)[0]
            groups = {}
            for h in mine:
                key = frozenset(fm.touches(h.producer.ref, frame)
                                | fm.touches(h.consumer.ref, frame))
                groups.setdefault(key or frozenset((h.producer.operand,)), []).append(h)
            covers = [covered
                      for _key, edges in sorted(groups.items(),
                                                key=lambda kv: sorted(map(str, kv[0])))
                      for _slot, covered in sorted(_cover(edges, n_slots, blk.label).items())]
            taken = set()
            for covered in (_one_barrier(covers, n_slots, blk.label) if self.merge else covers):
                slot = _cover_slot(covered, n_slots, blk.label)
                if any(_drives_tensorcnt(h) for h in covered):
                    slot = _unshared(slot, covered, n_slots, blk.label, taken)
                    taken.add(slot)
                placed.append((blk, slot, covered))

        self._assert_covered(prog, hazards, placed)
        return [PendingMark(
            mark=Mark("fence", {
                # the operands this fence orders -- the readable tag; `FenceTokensPass` adds the ids
                "buffers": tuple(sorted({h.producer.operand for h in covered}
                                        | {h.consumer.operand for h in covered})),
                "scope": "block",
                "kinds": tuple(sorted({h.kind for h in covered})),
                "edges": len(covered),
            }),
            region=_windows(prog, covered, blk.label)[0],
            options=tuple(_windows(prog, covered, blk.label)),
            edges=tuple(covered))
            for blk, _slot, covered in placed]

    @staticmethod
    def _assert_covered(prog, hazards, placed):
        """Every cross-wave edge has a fence standing between its two ends."""
        fences = {}
        for blk, slot, _c in placed:
            fences.setdefault(blk.label, {})[slot] = frozenset()
        n_of = {lab: len(b.body) + 1 for lab, b in prog.blocks.items()}
        for h in hazards.needing_fence():
            if not separating(h, fences, n_of.get):
                raise RuntimeError(
                    f"G-FENCE: the cross-wave {h.kind} edge {h.producer.block}"
                    f"[{h.producer.pos}] -> {h.consumer.block}[{h.consumer.pos}] is separated by "
                    f"no fence at either of its ends -- the cover is wrong, not this edge")

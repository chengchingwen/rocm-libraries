# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
FenceRegions -- WHERE the proc-scoped fences go.  Position only; `FenceTokensPass` names them.

A fence sits at ONE of its hazard's own two ends, never on a block between them: an intermediate
block lies on some path, and the producer can re-reach the consumer around it via the back edge.
"""

from __future__ import annotations

from ..nodes import Mark, Region, PendingMark, BLOCK_ENTRY, BLOCK_EXIT
from ..analysis import Analysis
from .frame_hazards import FrameHazards, RAW
from .frame_map import FrameMap
from .wait_common import FANOUT_META, counter_for, issued


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


def _fenced_in(hazard, _repeating):
    """The consumer block: every wave waits on its own queue, then one barrier publishes."""
    if not hazard.cross_block:
        return frozenset((hazard.consumer.block,))
    return frozenset((hazard.consumer.block,))


def _clusters(deadlines, body, counter, counts):
    """Deadlines of one producer that can share a fence for free, latest run first.

    A cross-agent hazard's wait is computed at the position of the fence owning its relation
    (`potential_waits.anchor_site`), and one fence emits ONE wait -- the minimum of what its
    hazards require.  Sharing is therefore free only when the residuals are equal: same producer
    and counter gives the same rank at any position, and an issue-free span means moving the
    anchor down that span costs that rank nothing."""
    out = []
    for slot in sorted(deadlines, reverse=True):
        if out and issued(body, slot, max(out[-1]), counter, counts) == 0:
            out[-1].append(slot)
        else:
            out.append([slot])
    return out


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


def _at_slot(prog, block, slot):
    """A concrete Region pinned to one already-selected fence slot."""
    body = prog.block(block).body
    return Region(block=block,
                  after=body[slot - 1] if slot > 0 else BLOCK_ENTRY,
                  before=body[slot] if slot < len(body) else BLOCK_EXIT)


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

    A RAW's wait rides the fence that owns it, so each one wants its own LATEST slot; fences merge
    only across an issue-free span of one producer, where the residual is provably unchanged.
    WAR/WAW stamp nothing, so they ride any fence already standing in their window."""

    def run(self, prog, am):
        hazards = am.get(FrameHazards(), prog)
        repeating = _repeating(am.get(FrameMap(), prog))
        counts = (prog.meta or {}).get(FANOUT_META) or {}
        n_slots = {lab: len(blk.body) + 1 for lab, blk in prog.blocks.items()}

        def home(hazard):
            return next(iter(_fenced_in(hazard, repeating)))

        def window(hazard, label):
            slots = _admissible_slots(hazard, n_slots[label], label)
            if not slots:
                raise RuntimeError(
                    f"no slot discharges the {hazard.kind} edge {hazard.producer.operand} "
                    f"pos {hazard.producer.pos} -> {hazard.consumer.pos} (gap {hazard.gap}): "
                    f"the emitted order leaves it undischargeable, so sigma is illegal ")
            return slots

        edges = list(hazards.needing_fence())
        for hazard in edges:                  # raises early on an illegal sigma
            window(hazard, home(hazard))

        # EVERY edge at its own latest slot, grouped by the producer whose rank its wait carries.
        # A WAR stamps too -- its producer is a shared READ, so `counter_for` gives it dscnt and
        # `anchor_site` computes that wait at its owning fence exactly as for a RAW's tensorcnt.
        deadlines = {}
        for hazard in edges:
            label = home(hazard)
            deadlines.setdefault(label, {}).setdefault(
                (id(hazard.producer.inst), counter_for(hazard, prog)), {}).setdefault(
                    max(window(hazard, label)), []).append(hazard)

        chosen = {}
        for label, by_producer in deadlines.items():
            body, out, claimed = prog.block(label).body, {}, {}
            for producer, by_slot in by_producer.items():
                for cluster in _clusters(by_slot, body, producer[1], counts):
                    anchor = min(cluster)
                    # Merging is free only within one producer.  Landing on a slot another
                    # producer already claims makes the fence emit the MINIMUM of two different
                    # ranks, so leave this cluster spread rather than drag one of them down.
                    shared = (all(anchor in window(h, label)
                                  for slot in cluster for h in by_slot[slot])
                              and claimed.get(anchor, producer) == producer)
                    for slot in cluster:
                        at = anchor if shared else slot
                        claimed.setdefault(at, producer)
                        out.setdefault(at, []).extend(by_slot[slot])
            chosen[label] = out

        # The contract's `rel` records name ONE owning fence per edge; each edge was appended to
        # exactly one, so the partition holds by construction.
        placed = [(prog.block(label), slot, covered)
                  for label, by_slot in sorted(chosen.items())
                  for slot, covered in sorted(by_slot.items())]

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
            region=_at_slot(prog, blk.label, slot),
            options=tuple(_windows(prog, covered, blk.label)),
            edges=tuple(covered))
            for blk, slot, covered in placed]

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

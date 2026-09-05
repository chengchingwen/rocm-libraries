# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
LdsHazards -- the LDS hazard edges, derived from the SSA.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..nodes import Move
from ..analysis import Analysis
from .cfg import successors
from .gen_reaching import GenReaching


#: A hazard is an ordered pair of accesses to one location, at least one a write.
RAW = "RAW"        # write -> read   (residency)
WAR = "WAR"        # read  -> write  (rotation anti-dep)
WAW = "WAW"        # write -> write  (rotation output-dep -- the phase machine's blind spot:
                   # its `_conflicts` fires only read<->write, so it cannot see this at all)


@dataclass(frozen=True)
class SharedTouch:
    """One shared-space touch by one instruction."""
    block:    str
    pos:      int          # index in the block body -- program order within the block
    inst:     object
    ref:      object
    is_write: bool
    operand:  str
    regions:  tuple = ()   # per region axis, the SET of region values this access MAY touch

    @property
    def gdelta(self):
        return getattr(self.ref, "gdelta", 0)

    @property
    def storage(self):
        """What LDS storage this access names: `(operand, regions)`.
 A buffer is `(region, generation)`, so the storage name is the operand AND its regions.  Under
 TDMSplit one operand owns `split` storage-DISJOINT halves, and pairing on the operand
 name alone reports hazards between half 0 and half 1 that cannot exist."""
        return (self.operand, self.regions)


@dataclass(frozen=True)
class Hazard:
    """One ordered pair that can touch the same buffer, with the trip distance between them."""
    kind:        str        # RAW | WAR | WAW
    producer:    SharedTouch
    consumer:    SharedTouch
    ring:        int        # S -- the buffer count this pair rotates through
    distance:    object     # trips from producer to consumer MOD ring, or None across blocks
    cross_agent: bool
    cross_block: bool = False   # producer and consumer are in DIFFERENT blocks

    @property
    def same_trip(self) -> bool:
        """Does an instance of this pair occur within ONE trip?"""
        return self.distance == 0

    @property
    def in_program_order(self) -> bool:
        """Is the producer emitted before the consumer within the block?

 Only meaningful for a `same_trip` pair, and it is what tells a same-wave `S=delta` in-place
 refill from a violation of it: the steady body puts copies last precisely so that the
 vacating read precedes the refill."""
        return (self.producer.block == self.consumer.block
                and self.producer.pos < self.consumer.pos)


@dataclass(frozen=True)
class Unresolved:
    """A pair whose frame could not be established -- reported, never silently dropped."""
    a:      SharedTouch
    b:      SharedTouch
    reason: str


class LdsHazardSet:
    """Query API over the hazard result."""

    def __init__(self, edges, unresolved):
        self._edges = tuple(edges)
        self._unresolved = tuple(unresolved)

    def __iter__(self):
        return iter(self._edges)

    def __len__(self):
        return len(self._edges)

    def edges(self, *, kind=None, block=None, cross_agent=None):
        """Hazards, optionally filtered.  `block` matches either endpoint's block."""
        out = self._edges
        if kind is not None:
            out = [h for h in out if h.kind == kind]
        if block is not None:
            out = [h for h in out if block in (h.producer.block, h.consumer.block)]
        if cross_agent is not None:
            out = [h for h in out if h.cross_agent == cross_agent]
        return tuple(out)

    def needing_fence(self):
        """The edges a proc-scoped fence must cover -- the cross-wave ones.

 A same-wave edge is discharged by program order plus the completion counter, so fencing it
 would be pure cost. This is the ONLY place the fence/no-fence decision is made."""
        return self.edges(cross_agent=True)

    def unresolved(self):
        return self._unresolved


def _ring_depth(ref) -> int:
    gen = getattr(ref, "gen", None)
    return max(1, gen.ring) if gen is not None else 1


def split_region_modes(prog):
    """`region_axes`, with the axes of movements that have ONE region REMOVED."""
    rms = dict(prog.meta.get("region_axes", {}) or {})
    counts = {}
    for unit, n in (prog.meta.get("unit_regions", {}) or {}).items():
        for op in unit:
            counts[op] = max(1, int(n))
    # The OPERAND's own count wins where it is known: a member riding a longer member's walk has
    # one buffer, so its access names it whole rather than may-aliasing every region.
    for op, n in (prog.meta.get("operand_regions", {}) or {}).items():
        counts[op] = min(counts.get(op, int(n)), max(1, int(n)))
    return {op: (() if counts.get(op, 2) <= 1 else axes) for op, axes in rms.items()}


def _regions_of(ref, operand, region_axes, extents, spans=None):
    """Per region axis of `operand`, the SET of region values this access MAY touch.

    A coord that PINS the axis names one region.  A coord that does not mention it touches EVERY
    region, so the unknown is modelled as the full set and stays a MAY-alias -- being conservative
    costs a fence, being optimistic LOSES one.
    """
    rms = region_axes.get(operand) or ()
    if not rms:
        return ()                                   # unsplit: one region, nothing to distinguish
    coord = dict(getattr(ref.tile, "coord", ()) or ())
    # A MERGED MOVEMENT NAMES EVERY BUFFER IT COVERS.  `Ref.covers` is the reference's own
    # declaration that ONE instruction fills several coordinates.
    spanned = {a for a, _f in (getattr(ref, "covers", ()) or ())}
    span = (spans or {}).get(operand) or {}
    out = []
    for rm in rms:
        # THE AXIS COUNTS STEPS, THE OPERAND COUNTS REGIONS: divide by the steps one of its own
        # regions spans, so a coarser split names one region across several values of the walk.
        step = max(1, int(span.get(rm, 1)))
        v = coord.get(rm)
        n = max(1, extents.get(rm, 1)) // step
        out.append(frozenset((int(v) // step,)) if (v is not None and rm not in spanned)
                   else frozenset(range(max(1, n))))
    return tuple(out)


def _disjoint_storage(a, b) -> bool:
    """Do these two accesses provably touch DIFFERENT LDS storage?

    Different operand: different tensor, different storage.  Same operand: disjoint iff some region
    axis has non-overlapping possible values.  Sets, not equality, so a `bothHalves` access is
    disjoint from nothing."""
    if a.operand != b.operand:
        return True
    return any(not (ra & rb) for ra, rb in zip(a.regions, b.regions))


def _shared_accesses(blk, region_axes=None, extents=None, spans=None):
    """Every shared-space touch in `blk`, in program order.

    A Move's shared DST is a write (a copy filling a buffer); its shared SRC is a read (a ds_read
    draining one).  An instruction may hold several shared Refs -- a Phi-fused cooperative copy carries
    one dst Ref per member -- and each is its own access on its own ring."""
    for pos, inst in enumerate(blk.body):
        if not isinstance(inst, Move):
            continue
        for ref, is_write in ([(r, False) for r in inst.srcs] + [(r, True) for r in inst.dsts]):
            if ref.tile.space != "shared":
                continue
            yield SharedTouch(block=blk.label, pos=pos, inst=inst, ref=ref,
                         is_write=is_write, operand=ref.tile.operand,
                         regions=_regions_of(ref, ref.tile.operand,
                                             region_axes or {}, extents or {}, spans or {}))


def _reachable(prog):
    """{label: set(labels reachable from it, excluding itself unless via a cycle)}.

    Reachability, not dominance: a cross-block hazard exists if SOME execution runs the producer
    before the consumer.  drain0 is reachable from steady even though steady does not dominate it,
    because the `T < M` short path enters drain0 straight from the prologue."""
    succ = successors(prog)
    out = {}
    for start in prog.blocks:
        seen, stack = set(), list(succ.get(start, ()))
        while stack:
            lab = stack.pop()
            if lab in seen:
                continue
            seen.add(lab)
            stack.extend(succ.get(lab, ()))
        out[start] = seen
    return out


def _hazard_kind(producer: SharedTouch, consumer: SharedTouch):
    if producer.is_write and consumer.is_write:
        return WAW
    if producer.is_write:
        return RAW
    if consumer.is_write:
        return WAR
    return None                                  # read -> read is not a hazard


class LdsHazards(Analysis):
    """See module docstring.  Pure; returns an `LdsHazardSet`."""

    def run(self, prog, am):
        reach = am.get(GenReaching(), prog)
        distributed = prog.meta.get("agent_distributed", {}) or {}
        edges, unresolved = [], []

        region_axes = split_region_modes(prog)
        extents = prog.meta.get("axis_extents", {}) or {}
        spans = prog.meta.get("region_span", {}) or {}
        by_block = {blk.label: list(_shared_accesses(blk, region_axes, extents, spans))
                    for blk in prog.blocks.values()}
        # A block that cannot reach itself runs once, so the mirror direction -- "the trip
        # that reuses the slot" -- has no trip to happen on.
        rch = _reachable(prog)
        for lbl, accesses in by_block.items():
            repeats = lbl in rch.get(lbl, ())
            for i, a in enumerate(accesses):
                for b in accesses[i + 1:]:
                    self._pair(a, b, reach, distributed, edges, unresolved, repeats)
        self._cross_block(prog, by_block, reach, distributed, edges, unresolved)
        return LdsHazardSet(edges, unresolved)

    def _cross_block(self, prog, by_block, reach, distributed, edges, unresolved):
        """Hazards whose two ends live in DIFFERENT blocks."""
        reachable = _reachable(prog)
        for pb, pas in by_block.items():
            for cb, cas in by_block.items():
                if pb == cb or cb not in reachable.get(pb, ()):
                    continue
                for p in pas:
                    for c in cas:
                        kind = _hazard_kind(p, c)
                        if kind is None or _disjoint_storage(p, c):
                            continue
                        if self._provably_disjoint(p, c, reach):
                            continue
                        cross = bool(distributed.get(p.operand) or distributed.get(c.operand))
                        edges.append(Hazard(kind=kind, producer=p, consumer=c,
                                            ring=_ring_depth(p.ref), distance=None,
                                            cross_agent=cross, cross_block=True))

    @staticmethod
    def _provably_disjoint(p: SharedTouch, c: SharedTouch, reach) -> bool:
        """Only a peel<->peel pair in absolute frames can be shown not to alias across blocks."""
        if getattr(p.ref, "gen", None) is not None or getattr(c.ref, "gen", None) is not None:
            return False                             # loop-carried: every generation, eventually
        if reach.is_relative(p.block) or reach.is_relative(c.block):
            return False
        gp, gc = reach.of(p.ref), reach.of(c.ref)
        return gp is not None and gc is not None and gp != gc

    def _pair(self, a: SharedTouch, b: SharedTouch, reach, distributed, edges, unresolved,
              repeats=True):
        """Emit the hazard(s) between two same-block accesses, or record why we cannot.

        `repeats` says whether the block can execute again; it gates the MIRROR direction, whose
        whole premise is a later trip reusing the slot.
        """
        if _hazard_kind(a, b) is None:                      # read -> read
            return
        if _disjoint_storage(a, b):
            # DIFFERENT LDS STORAGE -- checked FIRST, so it covers every branch below.
            return
        gen_a, gen_b = getattr(a.ref, "gen", None), getattr(b.ref, "gen", None)

        if gen_a is not None and gen_b is not None:
            if gen_a.id != gen_b.id:
                return                               # distinct rings address disjoint storage
            ring = max(1, gen_a.ring)
            self._emit(a, b, ring, (a.gdelta - b.gdelta) % ring, distributed, edges,
                       repeats)
            return

        if gen_a is None and gen_b is None:
            # Both peel-pinned.  A peel instance executes once, so the only possible distance is 0
            # and they collide iff they name the same BUFFER.  A buffer is (region, generation).
            ga, gb = reach.of(a.ref), reach.of(b.ref)
            if ga is None or gb is None:
                unresolved.append(Unresolved(a, b, "peel access carries no resolvable generation"))
                return
            if reach.is_relative(a.block):
                unresolved.append(Unresolved(a, b, f"block {a.block!r} generation frame is relative"))
                return
            if ga == gb:
                self._emit(a, b, 1, 0, distributed, edges, repeats)
            return

        unresolved.append(Unresolved(a, b, "mixed loop-carried and peel-pinned access in one block"))

    def _emit(self, a: SharedTouch, b: SharedTouch, ring, distance, distributed, edges,
              repeats=True):
        """Append the hazard for the ordered pair (a->b), and its mirror where one exists.
 Both directions are real and they are DIFFERENT obligations at different distances: on one
 buffer a copy->read is the residency (RAW) while the read->copy of the trip that reuses the
 slot is the rotation WAR.  Modelling only one direction misses the other obligation entirely."""
        both = ((a, b, distance), (b, a, (-distance) % ring))
        for producer, consumer, d in (both if repeats else both[:1]):
            kind = _hazard_kind(producer, consumer)
            if kind is None:
                continue
            cross = bool(distributed.get(producer.operand) or distributed.get(consumer.operand))
            edges.append(Hazard(kind=kind, producer=producer, consumer=consumer,
                                ring=ring, distance=d, cross_agent=cross))

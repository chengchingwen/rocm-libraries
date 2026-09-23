# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
FrameMap -- which storage each access works on, under which rotation phase.

A FRAME is a phase vector, one entry per Gen; a block entered several ways runs under several.  So
the frame set is a forward dataflow over SETS of phase vectors whose meet is UNION -- nothing is
collapsed onto an invented base.
"""

from __future__ import annotations

from ..nodes import Move, Mma
from ..analysis import Analysis
from .cfg import BackEdges, successors
from .lds_buffers import LdsBufferIds


class Frame:
    """An immutable rotation phase vector: `{gen id: phase}`, absent entries reading 0."""

    __slots__ = ("phases",)

    def __init__(self, phases=()):
        self.phases = tuple(sorted(phases.items() if isinstance(phases, dict) else phases))

    def of(self, gen_id) -> int:
        for g, p in self.phases:
            if g == gen_id:
                return p
        return 0

    def as_dict(self) -> dict:
        return dict(self.phases)

    def __eq__(self, other):
        return isinstance(other, Frame) and self.phases == other.phases

    def __hash__(self):
        return hash(self.phases)

    def __lt__(self, other):
        return self.phases < other.phases

    def __repr__(self):
        """Ring ids, not operand names -- `FrameMapping.render` is the readable one."""
        return ("entry" if not self.phases
                else "phase(" + ", ".join(f"ring{g}={p}" for g, p in self.phases) + ")")


def _rings(prog):
    """Every Gen the program rotates through: `{gen id: depth}`."""
    out = {}
    for blk in prog.blocks.values():
        for phi in blk.phis:
            out[phi.gen.id] = max(1, phi.gen.ring)
        for xf in blk.xfers:
            out[xf.gen.id] = max(1, xf.ring)
        for inst in blk.body:
            for ref in tuple(getattr(inst, "srcs", ())) + tuple(getattr(inst, "dsts", ())):
                gen = getattr(ref, "gen", None)
                if gen is not None:
                    out[gen.id] = max(1, gen.ring)
    return out


def _advance(frame, src, dst, is_back):
    """`frame` along `src -> dst`.

    A generation rotates when its loop takes another TRIP, so a block's transfers apply on the
    back edge that closes the trip -- the edge `BackEdgeSet.leaving` already keys them to.  An
    edge out of the latch that leaves the loop is not another trip: applying them there handed the
    drain a phase one trip ahead, and with ring 2 its two reads aliased onto one buffer."""
    phases = frame.as_dict()
    if is_back:
        for xf in src.xfers:
            phases[xf.gen.id] = (phases.get(xf.gen.id, 0) + xf.adv) % max(1, xf.ring)
    if not is_back:
        for phi in dst.phis:
            incoming = dict(phi.incomings)
            if incoming:
                if src.label not in incoming:
                    raise RuntimeError(
                        "FrameMap: phi for Gen %d in %s has no input from %s"
                        % (phi.gen.id, dst.label, src.label)
                    )
                value = incoming[src.label]
                if value is None:
                    continue
            else:
                value = phi.entry_val
            phases[phi.gen.id] = int(value) % max(1, phi.gen.ring)
    return Frame(phases)


class FrameMapping:
    """Query API: the frames a block runs under, and the storage an access touches in each."""

    def __init__(self, frames, entrances, edges, ref_block, storage, unresolved, ring_names=(),
                 gen_by_region=()):
        self._frames = {b: tuple(sorted(fs)) for b, fs in frames.items()}
        self._entrances = {b: _classes(fs) for b, fs in entrances.items() if fs}
        self._cut = {}
        self._edges = dict(edges)            # node -> tuple[node]
        self._ref_block = dict(ref_block)    # id(ref) -> block label
        self._storage = storage
        self._unresolved = tuple(unresolved)
        self._names = dict(ring_names)       # gen id -> the operand that rotates through it
        self._gen_by_region = dict(gen_by_region)   # (operand, region) -> gen id
        self._reach = _closure(self._edges)
        self._touch_cache = {}
        self._frame_touch_cache = {}
        self._workgroup_touch_cache = {}

    def render(self, frame):
        """`frame` with each ring named by its operand: `A=0 B=1`, or `entry` for the empty one."""
        if not frame.phases:
            return "entry"
        return " ".join("%s=%d" % (self._names.get(g, "ring%d" % g), p) for g, p in frame.phases)

    def frames(self, block):
        """EVERY phase this block is entered on -- not a representative."""
        return self._frames.get(block, ())

    def cut(self, block):
        """The one frame this block's accesses are NAMED in.

        Propagated forward from the predecessor that carries generations, so both ends of every
        cross-block edge are named in frames one step apart -- which is what the edge itself is."""
        return self._cut.get(block, self.entrances(block)[0])

    def entrances(self, block):
        """One frame per DISTINCT entrance picture.

        A back edge is not an entrance -- it is this block one rotation on.  Neither are two
        forward edges that differ by a UNIFORM rotation of every ring: that is the same relation
        with the ids relabelled, which the token already survives.  Only a non-uniform difference
        is a second picture."""
        return self._entrances.get(block, (Frame(),))

    def block_of(self, ref):
        return self._ref_block.get(id(ref))

    def nodes(self):
        """Every `(block, frame)` the program can execute."""
        return tuple((b, f) for b in self._frames for f in self._frames[b])

    def reaches(self, node):
        """The nodes reachable from `node` in ONE OR MORE steps (so a loop reaches itself)."""
        return self._reach.get(node, frozenset())

    def successors(self, node):
        """The `(block, frame)` nodes one edge on from `node`."""
        return self._edges.get(node, ())

    def generation(self, ref, frame, coords=None):
        """The concrete rotation phase `ref` names in `frame`, or None if it carries none.

        `coords` names ONE region, and that region's own `Gen` sets the phase.  A Ref carries a
        single `gen` while `_loop_carried_gens` makes one per `(operand, region)`, so reading the
        Ref's alone pins every region to the leader's rotation -- a region whose reader and writer
        then sit on different generations has no rotation between them at all."""
        ring = max(1, self._storage.depth_of(ref.tile.operand))
        gen = getattr(ref, "gen", None)
        if gen is not None:
            gen_id = gen.id
            if coords is not None and len(coords) == 1:
                gen_id = self._gen_by_region.get((ref.tile.operand, int(coords[0])), gen_id)
            return (frame.of(gen_id) + int(ref.gdelta)) % ring
        abs_gen = getattr(ref, "abs_gen", None)
        return None if abs_gen is None else int(abs_gen) % ring

    def touches(self, ref, frame):
        """The CONCRETE storage ids `ref` works on in `frame` -- one per region combination."""
        key = (id(ref), frame)
        hit = self._touch_cache.get(key)
        if hit is None:
            phase = self.generation(ref, frame)
            hit = frozenset() if phase is None else self._storage.at(ref, phase)
            self._touch_cache[key] = hit
        return hit

    def frame_touches(self, ref, frame, is_write):
        """Concrete storage selected by this access's frame and agent-relative address."""
        key = (id(ref), frame, bool(is_write))
        hit = self._frame_touch_cache.get(key)
        if hit is None:
            hit = self._storage.frame_at(
                ref, lambda coords: self.generation(ref, frame, coords), is_write)
            self._frame_touch_cache[key] = hit
        return hit

    def workgroup_touches(self, ref, frame, is_write):
        """Concrete storage touched across every physical agent executing this access."""
        key = (id(ref), frame, bool(is_write))
        hit = self._workgroup_touch_cache.get(key)
        if hit is None:
            hit = self._storage.workgroup_frame_at(
                ref, lambda coords: self.generation(ref, frame, coords), is_write)
            self._workgroup_touch_cache[key] = hit
        return hit

    def may_touch(self, ref):
        """The union over every frame this ref's block runs under."""
        seen = [self.touches(ref, f) for f in self.frames(self.block_of(ref))]
        return frozenset().union(*seen) if seen else frozenset()

    @property
    def unresolved(self):
        """Refs carrying no generation at all -- a hole, reported rather than defaulted."""
        return self._unresolved


def _carries_generations(blk):
    """Does this block hold a loop-carried shared access?  A peel block names absolutely and so
    constrains nothing about which frame its successors are named in."""
    for inst in blk.body:
        for ref in tuple(getattr(inst, "srcs", ())) + tuple(getattr(inst, "dsts", ())):
            if ref.tile.space == "shared" and getattr(ref, "gen", None) is not None:
                return True
    return False


def _cut(prog, succ, back, base, order):
    """One naming frame per block: the entry's, advanced along each forward edge.

    Where several forward predecessors reach a block, the one that CARRIES GENERATIONS wins: a
    peel names absolutely, so it has no opinion, and letting it set the cut leaves every edge from
    the loop off by the rotation the exit performed."""
    cut, claimed = {prog.entry: base}, set()
    for lab in order:
        if lab not in cut:
            continue
        src = prog.blocks[lab]
        for s in succ.get(lab, ()):
            if back.is_back_edge(lab, s):
                continue
            better = _carries_generations(src)
            if s in cut and (s in claimed or not better):
                continue
            cut[s] = _advance(cut[lab], src, prog.blocks[s], False)
            if better:
                claimed.add(s)
    return cut


def _classes(frames):
    """`frames` with those differing by one uniform rotation of every ring collapsed to one."""
    keep = []
    for f in sorted(frames):
        shifts = {(p - q) for (g, p), (_h, q) in zip(f.phases, keep[0].phases)} if keep else None
        if keep and len(shifts) == 1:
            continue
        keep.append(f)
    return tuple(keep)


def _closure(edges):
    """Transitive closure over the frame graph, one BFS per node (the graph is tiny)."""
    out = {}
    for start in edges:
        seen, stack = set(), list(edges.get(start, ()))
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            stack.extend(edges.get(n, ()))
        out[start] = frozenset(seen)
    return out


class FrameMap(Analysis):
    """See module docstring.  Pure; returns a `FrameMapping`."""

    def run(self, prog, am):
        storage = am.get(LdsBufferIds(), prog)
        back = am.get(BackEdges(), prog)
        succ = successors(prog)

        # EVERY frame carries EVERY ring, so two frames are equal iff they denote the same phase.
        # A partial vector would make the entry frame and the all-zero loop frame distinct objects
        # for one phase, splitting the node set and inventing cross-frame edges between them.
        rings = _rings(prog)
        base = Frame({g: 0 for g in rings})

        order = [blk.label for blk in prog.walk_rpo()]
        frames = {lab: set() for lab in prog.blocks}
        entrances = {lab: set() for lab in prog.blocks}
        frames[prog.entry].add(base)
        entrances[prog.entry].add(base)

        work = [prog.entry]
        while work:
            lab = work.pop(0)
            src = prog.blocks[lab]
            for s in succ.get(lab, ()):
                is_back = back.is_back_edge(lab, s)
                new = {_advance(f, src, prog.blocks[s], is_back) for f in frames[lab]}
                if not is_back:
                    entrances[s] |= {_advance(f, src, prog.blocks[s], False)
                                     for f in entrances[lab]} or new
                fresh = new - frames[s]
                if not fresh:
                    continue
                frames[s] |= fresh
                if s not in work:
                    work.append(s)
            work.sort(key=lambda b: order.index(b) if b in order else len(order))

        edges = {}
        for lab, blk in prog.blocks.items():
            for f in frames[lab]:
                edges[(lab, f)] = tuple(
                    (s, _advance(f, blk, prog.blocks[s], back.is_back_edge(lab, s)))
                    for s in succ.get(lab, ()))

        ref_block, unresolved, names = {}, [], {}
        generation_regions = (prog.meta or {}).get("generation_regions", {}) or {}
        for blk in prog.blocks.values():
            for inst in blk.body:
                if not isinstance(inst, (Move, Mma)):
                    continue
                for ref in tuple(inst.srcs) + tuple(inst.dsts):
                    ref_block[id(ref)] = blk.label
                    gen = getattr(ref, "gen", None)
                    if gen is not None:
                        fact = generation_regions.get(gen.id, {})
                        operand = fact.get("operand", ref.tile.operand)
                        region = int(fact.get("region", 0))
                        names.setdefault(gen.id, f"{operand}/r{region}")
                    elif ref.tile.space == "shared" and getattr(ref, "abs_gen", None) is None:
                        unresolved.append((ref.tile.operand, blk.label))
        gen_by_region = {(str(fact.get("operand")), int(fact.get("region", 0))): int(gen_id)
                         for gen_id, fact in generation_regions.items()}
        out = FrameMapping(frames, entrances, edges, ref_block, storage, unresolved, names,
                           gen_by_region)
        out._cut = _cut(prog, succ, back, base, order)
        return out

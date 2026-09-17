# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
LdsBufferIds -- the name of every LDS buffer: `(operand, region, ring)`, densely numbered.

A buffer is a BYTE RANGE.  Under an S-deep rotation an operand owns S of them; which generation
occupies one is `FrameMap`'s question, layered on top.  Depends on nothing -- every input is tile
metadata already on the Ref -- and it is the one numbering the frames, the hazards and the tokens
all speak in.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..nodes import Move
from ..analysis import Analysis
from .token_liveness import BufferLiveness


@dataclass(frozen=True)
class Buffer:
    """One LDS byte range: an operand's region, at one point of its rotation."""
    operand: str
    region:  tuple      # one value per region axis; () when the tile is not split
    ring:    int        # which of the operand's rotating buffers, in [0, depth)

    def render(self):
        r = "".join(f"/r{v}" for v in self.region)
        return f"{self.operand}{r}#{self.ring}"


def buffer_key(b: Buffer):
    return (b.operand, b.region, b.ring)


def shared_refs(prog):
    """(block, inst, ref, is_write) for every shared-space touch, in program order."""
    for blk in prog.blocks.values():
        for inst in blk.body:
            if not isinstance(inst, Move):
                continue
            for ref, is_write in ([(r, False) for r in inst.srcs]
                                  + [(r, True) for r in inst.dsts]):
                if ref.tile.space == "shared":
                    yield blk, inst, ref, is_write


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


class Geometry:
    """The static region facts of one program, read once from `prog.meta`."""

    def __init__(self, prog):
        self.region_axes = split_region_modes(prog)
        self.extents = prog.meta.get("axis_extents", {}) or {}
        self.spans = prog.meta.get("region_span", {}) or {}
        self.agent_relative = frozenset(prog.meta.get("region_agent_relative", ()) or ())

    def regions_of(self, ref, is_write=True):
        """Per region axis, the SET of region values this access MAY touch.

        A coord PINS the axis or the access touches every region; a READ of an agent-relative
        operand pins nothing either, its region being a fact about the wave, not the coord."""
        operand = ref.tile.operand
        rms = self.region_axes.get(operand) or ()
        if not rms:
            return ()                               # unsplit: one region, nothing to distinguish
        coord = dict(getattr(ref.tile, "coord", ()) or ())
        # A MERGED MOVEMENT NAMES EVERY BUFFER IT COVERS -- `Ref.covers` is the reference's own
        # declaration that ONE instruction fills several coordinates.
        spanned = {a for a, _f in (getattr(ref, "covers", ()) or ())}
        span = self.spans.get(operand) or {}
        agent_rel = (not is_write) and operand in self.agent_relative
        out = []
        for rm in rms:
            # THE AXIS COUNTS STEPS, THE OPERAND COUNTS REGIONS: divide by the steps one of its own
            # regions spans, so a coarser split names one region across several values of the walk.
            step = max(1, int(span.get(rm, 1)))
            v = coord.get(rm)
            n = max(1, int(self.extents.get(rm, 1))) // step
            pinned = v is not None and rm not in spanned and not agent_rel
            out.append(frozenset((int(v) // step,)) if pinned
                       else frozenset(range(max(1, n))))
        return tuple(out)


class LdsBufferIdSet:
    """Query API over the numbering.  `ids_for(ref)` is every buffer a Ref may touch."""

    def __init__(self, ids, per_ref, rings, geometry, unresolved, writes, reads):
        self._ids = dict(ids)                 # Buffer -> int
        self._per_ref = dict(per_ref)         # id(ref) -> tuple[Buffer], the MAY-set over the ring
        self._depths = dict(rings)            # operand -> rotation depth
        self._unresolved = tuple(unresolved)
        self._writes = dict(writes)           # {block: {index: frozenset(Buffer)}}
        self._reads = dict(reads)
        self.geometry = geometry

    def __len__(self):
        return len(self._ids)

    @property
    def buffers(self):
        """{Buffer: id}, the whole numbering."""
        return dict(self._ids)

    @property
    def unresolved(self):
        """Refs carrying no generation at all -- a hole, reported rather than defaulted."""
        return self._unresolved

    def depth_of(self, operand) -> int:
        """How many buffers this operand rotates through."""
        return self._depths.get(operand, 1)

    def id_of(self, buf):
        return self._ids.get(buf)

    def buffers_for(self, ref):
        """Every Buffer `ref` may touch over the rotation -- `FrameMap` narrows it to one."""
        return self._per_ref.get(id(ref), ())

    def ids_for(self, ref):
        return tuple(self._ids[b] for b in self._per_ref.get(id(ref), ()))

    def at(self, ref, ring):
        """The ids `ref` touches when its rotation stands at `ring` -- one per region."""
        return frozenset(self._ids[b] for b in self._per_ref.get(id(ref), ()) if b.ring == ring)

    def liveness(self, prog):
        """Buffer live ranges over this program."""
        return BufferLiveness(prog, self._writes, self._reads)


def _depth_of_ref(ref) -> int:
    gen = getattr(ref, "gen", None)
    if gen is not None:
        return max(1, int(gen.ring))
    abs_gen = getattr(ref, "abs_gen", None)
    return 1 if abs_gen is None else max(1, int(abs_gen) + 1)


def _expand(operand, regions, depth):
    """The CROSS PRODUCT of the per-axis region sets, over the whole rotation."""
    combos = [()]
    for axis_values in regions:
        combos = [c + (v,) for c in combos for v in sorted(axis_values)]
    return [Buffer(operand, c, r) for c in combos for r in range(depth)]


class LdsBufferIds(Analysis):
    """See module docstring.  Pure; returns an `LdsBufferIdSet`.  Depends on no other analysis."""

    def run(self, prog, am):
        geometry = Geometry(prog)
        touches = list(shared_refs(prog))
        pos = {id(inst): i for blk in prog.blocks.values() for i, inst in enumerate(blk.body)}

        rings = {}
        for _blk, _inst, ref, _w in touches:
            op = ref.tile.operand
            rings[op] = max(rings.get(op, 1), _depth_of_ref(ref))

        per_ref, wanted, unresolved = {}, set(), []
        writes = {lab: {} for lab in prog.blocks}
        reads = {lab: {} for lab in prog.blocks}
        for blk, inst, ref, is_write in touches:
            op = ref.tile.operand
            if getattr(ref, "gen", None) is None and getattr(ref, "abs_gen", None) is None:
                unresolved.append((op, blk.label))
                continue
            bufs = tuple(sorted(_expand(op, geometry.regions_of(ref, is_write), rings[op]),
                                key=buffer_key))
            per_ref[id(ref)] = bufs
            wanted.update(bufs)
            side = writes if is_write else reads
            at = pos[id(inst)]
            side[blk.label][at] = side[blk.label].get(at, frozenset()) | frozenset(bufs)

        # DENSE and DETERMINISTIC: sorted key order, so the same Program always numbers the same.
        ids = {b: i for i, b in enumerate(sorted(wanted, key=buffer_key))}
        return LdsBufferIdSet(ids, per_ref, rings, geometry, unresolved, writes, reads)

# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
LdsBufferIds -- the name of every LDS buffer: `(operand, region, generation)`, densely numbered.

This is STORAGE IDENTITY, which is what `LdsHazards` reasons about.  It is NOT the memory token
the backend consumes -- `DependenceTokens` derives that from the hazard set.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..nodes import Move
from ..analysis import Analysis
from .gen_reaching import GenReaching
from .lds_hazards import _regions_of, split_region_modes
from .token_liveness import BufferLiveness


@dataclass(frozen=True)
class Buffer:
    """One logical LDS buffer -- the thing a token names."""
    operand:    str
    region:     tuple      # one value per region axis; () when the tile is not split
    generation: int

    def render(self):
        r = "".join(f"/r{v}" for v in self.region)
        return f"{self.operand}{r}#{self.generation}"


class LdsBufferIdSet:
    """Query API over the numbering.  `ids_for(ref)` is the storage a Ref may touch."""

    def __init__(self, ids, per_ref, unresolved, per_ref_bufs=None, writes=None, reads=None):
        self._ids = dict(ids)                 # Buffer -> int
        self._per_ref = dict(per_ref)         # id(ref) -> tuple[int]
        self._unresolved = tuple(unresolved)
        self._per_ref_bufs = dict(per_ref_bufs or {})
        self._writes = dict(writes or {})     # {block: {index: frozenset(Buffer)}}
        self._reads = dict(reads or {})

    def __len__(self):
        return len(self._ids)

    @property
    def buffers(self):
        """{Buffer: id}, the whole numbering."""
        return dict(self._ids)

    @property
    def unresolved(self):
        """Refs we could NOT name (no resolvable generation).  A non-empty list is a hole, not a
        default -- `verify_gir`'s G-TOKEN turns it into an error."""
        return self._unresolved

    def ids_for(self, ref):
        """The LDS buffer id(s) `ref` touches, or () if this ref was not assigned.

        A TUPLE, not an int: an access that does not pin its region axis may touch several buffers
        and must carry all of their ids, or the halves it does not name go un-ordered."""
        return self._per_ref.get(id(ref), ())

    def id_of(self, buf):
        return self._ids.get(buf)

    def buffers_for(self, ref):
        """The Buffers `ref` may touch -- `ids_for` before the numbering is applied."""
        return self._per_ref_bufs.get(id(ref), ())

    def liveness(self, prog):
        """Buffer live ranges over this program."""
        return BufferLiveness(prog, self._writes, self._reads)


def shared_refs(prog):
    """(block, inst, ref, is_write) for every shared-space touch -- the accesses a token names.

    Mirrors `lds_hazards._shared_accesses`; kept separate because that one builds the richer
    `SharedTouch` record the hazard pairing needs, while this walk only has to reach every Ref."""
    for blk in prog.blocks.values():
        for inst in blk.body:
            if not isinstance(inst, Move):
                continue
            for ref, is_write in ([(r, False) for r in inst.srcs]
                                  + [(r, True) for r in inst.dsts]):
                if ref.tile.space == "shared":
                    yield blk, inst, ref, is_write


class LdsBufferIds(Analysis):
    """See module docstring.  Pure; returns a `LdsBufferIdSet`."""

    def run(self, prog, am):
        reach = am.get(GenReaching(), prog)
        # THE COUNT, NOT THE TUPLE.  A/B: 2 regions x 2 buffers = 4 each; the scales
        # are unsplit, so 1 x 2 = 2 each; twelve, not sixteen.  See
        region_axes = split_region_modes(prog)
        extents = prog.meta.get("axis_extents", {}) or {}
        agent_rel = set(prog.meta.get("region_agent_relative", ()) or ())
        spans = prog.meta.get("region_span", {}) or {}

        wanted, per_ref_keys, unresolved = set(), {}, []
        pos = {id(inst): i for blk in prog.blocks.values() for i, inst in enumerate(blk.body)}
        writes = {lab: {} for lab in prog.blocks}
        reads = {lab: {} for lab in prog.blocks}
        for _blk, _inst, ref, _w in shared_refs(prog):
            gen = reach.of(ref)
            if gen is None:
                # No resolvable generation: we cannot say WHICH buffer, so we must not invent one.
                # Recorded rather than skipped -- a missing token breaks StinkyTofu's all-or-none
                unresolved.append((ref.tile.operand, getattr(ref, "gdelta", None)))
                continue
            operand = ref.tile.operand
            regions = _regions_of(ref, operand, region_axes, extents, spans)
            if not _w and operand in agent_rel and regions:
                regions = tuple(frozenset(range(max(1, extents.get(rm, 1))
                                                // max(1, (spans.get(operand) or {}).get(rm, 1))))
                                for rm in (region_axes.get(operand) or ()))
            # A relative frame keeps its RESOLVED generation: the token is a logical name, so only
            # this-buffer-vs-that-buffer has to hold, and `of()` already gives that in every frame.
            keys = tuple(sorted(_expand(operand, regions, int(gen)), key=_sort_key))
            per_ref_keys[id(ref)] = keys
            wanted.update(keys)
            side = writes if _w else reads
            at = pos[id(_inst)]
            side[_blk.label][at] = side[_blk.label].get(at, frozenset()) | frozenset(keys)

        # DENSE and DETERMINISTIC: sorted key order, so the same Program always numbers the same.
        ids = {buf: i for i, buf in enumerate(sorted(wanted, key=_sort_key))}
        per_ref = {rid: tuple(ids[k] for k in keys) for rid, keys in per_ref_keys.items()}
        return LdsBufferIdSet(ids, per_ref, unresolved, per_ref_keys, writes, reads)


def _sort_key(buf: Buffer):
    return (buf.operand, buf.region, buf.generation)


def _expand(operand, regions, generation):
    """The Buffers one access may touch -- the CROSS PRODUCT of its per-axis region sets.
 A pinned access yields one Buffer. An access that leaves an axis unpinned yields one per value
 of that axis, and carries them all: that is the `bothHalves` shape, and carrying only one of
 them is precisely the bug where a split tile's half-1 load goes un-waited."""
    combos = [()]
    for axis_values in regions:
        combos = [c + (v,) for c in combos for v in sorted(axis_values)]
    return [Buffer(operand=operand, region=c, generation=generation) for c in combos]

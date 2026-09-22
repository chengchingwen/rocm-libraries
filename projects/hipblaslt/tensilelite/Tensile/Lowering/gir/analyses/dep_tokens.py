# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
DependenceTokens -- tokens DERIVED to cover the hazard set, not read off the ids id.

`LdsBufferIds.Slot` names a byte range; a token names an OBLIGATION.  A hazard instance is resolved
to CONCRETE slots by `FrameMap`, so the two ends are related by construction and the classes close
only over what a frame actually forces -- never over both ends' whole may-sets.

The STAMP IS DERIVED IN ONE FRAME AND CHECKED IN THE OTHERS.  A token is a LOGICAL name and the
next frame is this one with the ring slots PERMUTED, so it relabels every stamp by a bijection and
"do these two ends share a name" has one answer everywhere.
"""

from __future__ import annotations

from ..analysis import Analysis
from .frame_hazards import FrameHazards, RAW
from .frame_map import FrameMap
from ..nodes import BLOCK_ENTRY, BLOCK_EXIT, Move, Region, region_slot
from .lds_buffers import LdsBufferIds, shared_refs, buffer_key


class DependenceTokenSet:
    """Query API mirroring `LdsBufferIdSet`, so emitters read tokens the same way."""

    def __init__(self, per_ref, obligations, storage_of):
        self._per_ref = dict(per_ref)           # id(ref) -> tuple[int]
        self._obligations = tuple(obligations)  # token -> the edges it discharges
        self._storage = dict(storage_of)        # token -> the slots it concerns, for rendering

    def __len__(self):
        return len(self._obligations)

    def tokens_for(self, ref):
        """The obligation token(s) `ref` stands in, or () if it stands in none."""
        return self._per_ref.get(id(ref), ())

    def storage_of(self, token):
        """The ids slots this token's obligations concern -- the readable tag, not the chain."""
        return self._storage.get(token, frozenset())

    def edges_of(self, token):
        return self._obligations[token]

    def naming(self, edges, block):
        """The tokens a fence in `block` must name: what THAT block's own stamps call the storage.

        The barrier exists FOR THE READS AFTER IT, so it names what THEY call the storage -- the
        consumer's stamp.  Naming the producer binds the wait to the newest execution of that copy,
        which is the one sitting in front of the barrier, and drains instead of waiting."""
        return frozenset(t for h in edges if h.kind == RAW
                         for t in self.tokens_for(h.consumer.ref))

    def order_naming(self, edges, block):
        """The tokens a fence ORDERS without waiting on -- the anti-dependences it carries.

        Both ends, because the rotation gives the vacating read and the refill different stamps in
        this block's frame and the barrier must sit between whichever one the schedule names."""
        return frozenset(t for h in edges if h.kind != RAW
                         for r in (h.producer.ref, h.consumer.ref)
                         for t in self.tokens_for(r))

    def choose(self, prog, options, edges, repeating, preds):
        """Which legal window this fence takes: the one whose WAIT leaves the most loads in flight.

        The wait a barrier emits is the number of loads issued since the last one that wrote a
        buffer it names, so the choice is scored by counting them, not by a rule of thumb."""
        if not options:
            return None
        named = self.naming(edges, options[0].block)
        best = None
        for r in options:
            body = prog.blocks[r.block].body
            lo, hi = region_slot(body, r.after, True), region_slot(body, r.before, False)
            for at in range(lo, hi + 1):
                score = (self._in_flight(prog, r.block, at, named, preds),
                         r.block not in repeating, -at)
                if best is None or score > best[0]:
                    best = (score, r.block, at)
        if best is None:
            return None                  # every window malformed: keep the region as stated
        # PINNED, not a window: this layer made the choice, so it does not hand the freedom on.
        _s, lab, at = best
        body = prog.blocks[lab].body
        return Region(block=lab,
                      after=body[at - 1] if at > 0 else BLOCK_ENTRY,
                      before=body[at] if at < len(body) else BLOCK_EXIT)

    def _in_flight(self, prog, block, pos, named, preds):
        """Loads still outstanding at `region`'s point -- the wait a fence there would emit.

        Walk back along the REAL predecessor paths, not inside one block: the load a loop-top fence
        waits on was issued on the previous trip.  A wait must be safe on every path, so the answer
        is the smallest count any path gives."""
        best, stack = None, [(block, pos, 0, 0)]
        while stack:
            lab, upto, seen, depth = stack.pop()
            found = False
            for inst in reversed(prog.blocks[lab].body[:upto]):
                if not _is_load(inst):
                    continue
                if {t for r in inst.dsts for t in self.tokens_for(r)} & named:
                    found = True
                    break
                seen += 1
            if found or depth >= _BACK_DEPTH or not preds.get(lab):
                best = seen if best is None else min(best, seen)
                continue
            for p in preds[lab]:
                stack.append((p, len(prog.blocks[p].body), seen, depth + 1))
        return best or 0

#: How many blocks back a wait may look for the load it names.
_BACK_DEPTH = 2


def _is_load(inst):
    """A Move that fills shared storage from global -- one TDM queue entry."""
    return (isinstance(inst, Move) and any(r.tile.space == "global" for r in inst.srcs)
            and any(r.tile.space == "shared" for r in inst.dsts))


class _Classes:
    """Union-find over ids ids: one class per set of slots some frame ties together."""

    def __init__(self, n):
        self._parent = list(range(n))

    def find(self, i):
        while self._parent[i] != i:
            self._parent[i] = self._parent[self._parent[i]]
            i = self._parent[i]
        return i

    def union(self, ids):
        members = sorted(ids)
        if not members:
            return
        root = self.find(members[0])
        for i in members[1:]:
            other = self.find(i)
            if other != root:
                self._parent[other] = root


class DependenceTokens(Analysis):
    """See module docstring.  Pure; returns a `DependenceTokenSet`."""

    def run(self, prog, am):
        ids = am.get(LdsBufferIds(), prog)
        fm = am.get(FrameMap(), prog)
        hz = am.get(FrameHazards(), prog)

        classes = _Classes(len(ids))
        edges_at = {}
        for fp, fc, h in hz.instances():
            # THE STORAGE BOTH ENDS TOUCH, each at its own frame -- the union ties every region a
            # multi-region read spans, so one edge concerning A's region 0 would name region 1 too.
            tied = fm.touches(h.producer.ref, fp) & fm.touches(h.consumer.ref, fc)
            if not tied:
                continue
            classes.union(tied)
            edges_at.setdefault(min(tied), []).append(h)

        # DENSE and DETERMINISTIC: classes numbered by their lowest slot, which is itself sorted.
        roots = sorted({classes.find(i) for i in range(len(ids))})
        token_of = {r: t for t, r in enumerate(roots)}
        members = {t: set() for t in range(len(roots))}
        for i in range(len(ids)):
            members[token_of[classes.find(i)]].add(i)

        obligations = [[] for _ in roots]
        seen = [set() for _ in roots]
        for anchor, edges in edges_at.items():
            t = token_of[classes.find(anchor)]
            for h in edges:
                if id(h) not in seen[t]:
                    seen[t].add(id(h))
                    obligations[t].append(h)

        by_id = {i: s for s, i in ids.buffers.items()}
        per_ref = {}
        for blk, _inst, ref, _w in shared_refs(prog):
            # THE FRAME PAIRS WITH ITS ENTRANCE, and the cut is the one every edge agrees on.
            here = fm.touches(ref, fm.cut(blk.label))
            per_ref[id(ref)] = tuple(sorted({token_of[classes.find(i)] for i in here}))
        tokens = DependenceTokenSet(
            per_ref, [tuple(o) for o in obligations],
            {t: frozenset(sorted((by_id[i] for i in mine), key=buffer_key))
             for t, mine in members.items()})
        return tokens

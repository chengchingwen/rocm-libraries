# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
DependenceTokens -- tokens DERIVED to cover the hazard set, not read off the buffer id.

The id (`LdsBufferIds.Buffer`) names storage and is what `LdsHazards` reasons about. A token
names an OBLIGATION: one per group of hazards over the same storage, carried only by the accesses
that actually stand in one. An access with no hazard on a buffer never names it.
"""

from __future__ import annotations

from ..analysis import Analysis
from .lds_hazards import LdsHazards
from .lds_buffers import LdsBufferIds, shared_refs


class DependenceTokenSet:
    """Query API mirroring `LdsBufferIdSet`, so emitters read tokens the same way."""

    def __init__(self, per_ref, obligations, ids):
        self._per_ref = dict(per_ref)          # id(ref) -> tuple[int]
        self._obligations = tuple(obligations)  # token -> the edges it discharges
        self._ids = dict(ids)                  # token -> the storage it concerns, for rendering

    def __len__(self):
        return len(self._obligations)

    def tokens_for(self, ref):
        """The obligation token(s) `ref` stands in, or () if it stands in none."""
        return self._per_ref.get(id(ref), ())

    def storage_of(self, token):
        """The buffer ids this token's obligations concern -- the readable tag, not the chain."""
        return self._ids.get(token, frozenset())

    def edges_of(self, token):
        return self._obligations[token]


def _class_key(ids, hazard):
    """One token per obligation: the ORDERED pair of the two ends' storage, plus the kind.

    A pair, never a union -- the two ends keep their own names, so a fence naming this token orders
    this edge and not every other edge that happens to touch one of the same buffers."""
    p, c = hazard.producer, hazard.consumer
    return (tuple(sorted(ids.ids_for(p.ref))),
            tuple(sorted(ids.ids_for(c.ref))), hazard.kind)


class DependenceTokens(Analysis):
    """See module docstring.  Pure; returns a `DependenceTokenSet`."""

    def run(self, prog, am):
        hz = am.get(LdsHazards(), prog)
        ids = am.get(LdsBufferIds(), prog)

        classes = {}
        for h in hz:
            classes.setdefault(_class_key(ids, h), []).append(h)
        # An access in no hazard still needs a name -- StinkyTofu's token rule is all-or-none per
        # block -- so it gets a PRIVATE class of no obligations. Its own storage, nothing else's:
        # standing in no hazard is exactly the statement that nothing has to be ordered against it.
        free = {}
        for _blk, _inst, ref, _w in shared_refs(prog):
            own = frozenset(ids.ids_for(ref))
            if own:
                free[id(ref)] = ("free", tuple(sorted(own)))

        # DENSE and DETERMINISTIC: sorted key order, so the same Program always numbers the same.
        per_ref = {}
        for key, edges in classes.items():
            for h in edges:
                for end in (h.producer, h.consumer):
                    per_ref.setdefault(id(end.ref), set()).add(key)
        for rid, key in free.items():
            if rid not in per_ref:
                classes.setdefault(key, [])
                per_ref[rid] = {key}
        order = sorted(classes, key=repr)
        index = {k: t for t, k in enumerate(order)}
        per_ref = {r: {index[k] for k in ks} for r, ks in per_ref.items()}
        return DependenceTokenSet({k: tuple(sorted(v)) for k, v in per_ref.items()},
                                  [tuple(classes[k]) for k in order],
                                  dict(enumerate(order)))

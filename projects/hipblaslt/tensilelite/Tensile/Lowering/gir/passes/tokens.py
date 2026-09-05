# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
TokensPass -- stamp each LDS Move with its completion token.
"""

from __future__ import annotations

from ..nodes import Move
from ..analyses.gen_reaching import GenReaching
from ..analyses.dep_defuse import DepDefuseAnalysis
from ..analyses.dep_tokens import DependenceTokens
from ..analyses.region_increment import region_of
from .base import Pass


def _storage_ids(inst, tokens):
    """The union of obligation tokens over every shared Ref this Move touches, sorted."""
    out = set()
    for r in list(inst.srcs) + list(inst.dsts):
        if r.tile.space == "shared":
            out.update(tokens.tokens_for(r))
    return tuple(sorted(out))


def _lds_ref(inst, side):
    refs = inst.srcs if side == "src" else inst.dsts
    for r in refs:
        if r.tile.space == "shared":
            return r
    return None


def _unit_key(prog, operand):
    """The CANONICAL key of the completion unit that produces `operand` -- the same key
    `Theta.movement_units()` uses, looked up rather than re-derived.
    """
    for key in prog.meta.get("unit_regions", {}) or {}:
        if operand in key:
            return key
    return (operand,)


def _completion_unit(prog, operand):
    """The completion unit AS IT APPEARS IN THE TOKEN: the group tuple when fusion merges several
    movements into one cooperative instruction (one completion for the whole group),
    else the bare operand name -- so an unfused kernel's token stays ('lds', 'A', gen), unchanged."""
    key = _unit_key(prog, operand)
    return key if len(key) > 1 else key[0]


def _region_key(prog, ref):
    """The region coordinates that identify WHICH movement instance `ref` touches, under the
 per-region pi -- else ``.
    """
    if not prog.meta.get("per_region_completion"):
        return ()
    # A read can only await at the granularity its PRODUCER offers.  Key on the region ONLY when
    # the completion unit that produces this value actually emits per-region movements: a COARSE
    if (prog.meta.get("unit_regions", {}) or {}).get(_unit_key(prog, ref.tile.operand), 1) <= 1:
        return ()
    # ONE DERIVATION of "which region does this ref name", shared with the walk.  A member with
    # fewer regions than its unit -- an unsplit scale riding a split parent -- has none, so its
    # token stays region-free and matches the single write it gets.
    return region_of(prog, ref)


def _buffer_generation(ref, gen):
    """The buffer generation this access names -- the COMPLETION token, not the storage name.

    Which storage a generation aliases is `LdsBufferIds`' question, not this one."""
    return (int(gen),)


class TokensPass(Pass):
    def run(self, prog, am):
        reach = am.get(GenReaching(), prog)
        tokens = am.get(DependenceTokens(), prog)
        am.get(DepDefuseAnalysis(), prog)          # available for L3 WAR-forwarding; consistency
        changed = False
        for blk in prog.blocks.values():
            for inst in blk.body:
                if not isinstance(inst, Move):
                    continue
                # a read consumes a shared SRC; a copy fills a shared DST
                ref = _lds_ref(inst, "src") or _lds_ref(inst, "dst")
                if ref is None:
                    continue
                gen = reach.of(ref)
                if gen is None:
                    continue
                inst.token = (("lds", _completion_unit(prog, ref.tile.operand),
                               _buffer_generation(ref, gen)) + _region_key(prog, ref))
                #...and the BACKEND storage name(s) alongside it. Stamped here, next to
                # the completion token, because the two are read together and drifting them apart
                inst.token_ids = _storage_ids(inst, tokens)
                changed = True
        if changed:
            prog.bump()
        return ("tokens",) if changed else ()

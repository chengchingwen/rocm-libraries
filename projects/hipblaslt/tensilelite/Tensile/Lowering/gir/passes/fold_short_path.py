# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
FoldShortPathPass -- the MERGE pass: coverage the `T < M` arm into the shared prologue/drain
path when that is provably legal, keep it as its own path when it is not, and raise when neither
shape works.
"""

from __future__ import annotations

from ..nodes import CondGoto, Goto, LoopBack, Mark
from ..nodes import terminator_targets
from ..analyses.short_path import ShortPathFold, FOLD, SPLIT, UNSOUND, NA, VACUOUS
from .base import StructuralPass


def folded(prog) -> bool:
    """Did the `T < M` arm get folded into the shared prologue/drain path?

    The question a consumer must ASK rather than assume.  True also when there was no arm to begin
    with (M == 0): the shared shape is then trivially what the Program has."""
    sl = prog.meta.get("short_loop")
    return sl is None or sl.get("verdict") == "folded"


class FoldShortPathPass(StructuralPass):
    """See module docstring.  A STRUCTURAL pass: it deletes blocks and retargets terminators, and
    never touches a body.  See `StructuralPass` for the contract it is held to."""

    def run(self, prog, am):
        v = am.get(ShortPathFold(), prog)
        sl = prog.meta.get("short_loop")
        if v.verdict == NA:
            return ()

        if v.verdict == UNSOUND:
            # EVERY unserved consumer, not just the first: which OPERANDS are unserved is the whole
            # diagnosis: A and B can be served while the MX scales are not.
            worst = "".join("      %r\n" % (u,) for u in (v.unbound or ())[:8])
            raise RuntimeError(
                f"the `T < M` path cannot be emitted in EITHER shape: {v.reason}.\n"
                f"  unserved consumers ({len(v.unbound or ())}):\n{worst}"
                f"  shape: short_loop={sl!r} M={prog.meta.get('M')!r} "
                f"inner_order={tuple(prog.meta.get('inner_axis_order') or ())} "
                f"mma_inputs={tuple(prog.meta.get('mma_inputs') or ())}\n"
                f"Folding would hide it behind a path that happens to work; emitting the arm would "
                f"ship known-broken code.  Fix the arm (it is built by `emit.build_ir`, not here) -- "
                f" for the two ways this has happened.")

        if sl is None:                       # blocks exist but nothing recorded them: a lowering bug
            raise RuntimeError(
                "short blocks are present but `meta['short_loop']` is missing -- the arm's verdict "
                "must be carried, not inferred from block names")

        record = dict(sl, fold=v.verdict, reason=v.reason, obligations=v.obligations,
                      extras=v.extras)

        if v.verdict == SPLIT:
            return self._keep_split(prog, record, v)
        return self._fold_arm(prog, record, v)

    def _keep_split(self, prog, record, v):
        """SPLIT: the two shapes disagree, so the arm stays as the lowering emitted it."""
        dropped = 0
        for lab in v.short_blocks:
            blk = prog.blocks[lab]
            blk.model_only = True          # already set by the lowering; restated for a
            dropped += sum(1 for n in blk.body
                           if isinstance(n, Mark) and n.kind != "phase_boundary")
        record["verdict"] = "kept"
        record["model_only"] = {
            "blocks": tuple(v.short_blocks),
            "marks_dropped": dropped,
            "reason": "the T < M arm is scaffold-owned (TensileLite's toPGR1 path); GIR keeps "
                      "it for the analyses but no backend emits it",
        }
        prog.meta["short_loop"] = record
        prog.bump()
        return ()

    def _fold_arm(self, prog, record, v):
        """FOLD/VACUOUS: rewire every edge into the short arm, then drop its blocks."""
        assert v.verdict in (FOLD, VACUOUS), v.verdict
        short = set(v.short_blocks)
        first_drain = v.folded_blocks[1] if len(v.folded_blocks) > 1 else "end"

        # THE FOLDED EDGE INHERITS THE ARM'S FRAME.  Rewiring `-> short0` into `-> drain0`
        # does not merely change a target: the two land at DIFFERENT positions on the summation
        arm_base = prog.blocks[v.short_blocks[0]].chunk_base if v.short_blocks else 0

        def _rewire(src_lab):
            # `first_drain` is "end" when the coverage leaves no drain block to land in; there is then
            # no frame to state.
            if first_drain in prog.blocks:
                prog.blocks[first_drain].path_chunk_base[src_lab] = arm_base

        for lab, blk in prog.blocks.items():
            if lab in short:
                continue
            t = blk.term
            if isinstance(t, CondGoto) and t.f_target in short:
                _rewire(lab)
                blk.term = CondGoto(t.pred, t.t_target, first_drain)
                blk.succs = (t.t_target, first_drain)
            elif isinstance(t, Goto) and t.target in short:
                _rewire(lab)
                blk.term = Goto(first_drain)
                blk.succs = (first_drain,)
            elif isinstance(t, LoopBack) and t.exit_target in short:
                _rewire(lab)
                blk.term = LoopBack(t.trips, t.body, first_drain, t.label)
                blk.succs = (t.body, first_drain)

        for lab in short:
            del prog.blocks[lab]

        # preds are a DECLARED fact checked by G-TERM, so recompute them from the rewritten
        # terminators rather than patching drain0 by name.
        computed = {lab: set() for lab in prog.blocks}
        for lab, blk in prog.blocks.items():
            for tgt in terminator_targets(blk.term):
                if tgt in prog.blocks:
                    computed[tgt].add(lab)
        for lab, blk in prog.blocks.items():
            if blk.preds:
                blk.preds = tuple(sorted(computed[lab]))

        record["verdict"] = "folded"
        # the arm is GONE, so the model-only declaration goes with it: a record naming deleted
        # blocks would leave G-EMIT checking a count against nothing.
        record.pop("model_only", None)
        prog.meta["short_loop"] = record
        prog.bump()
        return ("ShortPathFold",)



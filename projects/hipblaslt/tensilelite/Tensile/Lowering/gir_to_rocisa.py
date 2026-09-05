# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
gir_to_rocisa -- the layer 2 -> 3 lowering, the THIN rocisa adapter.
"""

from __future__ import annotations

from rocisa.code import Module

from .gir_tag import gir_tag, sync_comment
from .gir.emit_plan import plan_block
from .gir.coverage import plan_coverage
from .gir.nodes import Move
from .gir.analysis import AnalysisManager
from .gir.analyses.reg_band import RegBandAnalysis
from .leaves import LeafEmitters
from ..LoopModel.adapter import fuse_groups
from rocisa.instruction import SBarrier


def _register_depth(prog, band=None):
    """`{operand: W}` -- the register ring width the leaves use for their `m = u % W` FALLBACK, a
    GIR fact (`Ref.reg_ring`, validated by RegBand, B3).

    The fallback only runs when GIR did not supply a per-act slot, which for a GIR-planned wmma or
    read it always does; W is therefore a legacy path, not the authority.
    """
    per_op = {}
    items = band.items() if band is not None else {}
    if not items:
        items = {}
        for blk in prog.blocks.values():
            for inst in blk.body:
                if isinstance(inst, Move):
                    for ref in inst.dsts:
                        if ref.tile.space == "register" and ref.reg_ring:
                            items[(ref.tile.operand, ref.group)] = ref.reg_ring
    for (op, _group), w in items.items():
        if op in per_op and per_op[op] != w:
            raise NotImplementedError(
                f"operand {op!r} has register groups with differing rotation widths "
                f"{sorted({per_op[op], w})}; the leaf's fallback takes ONE scalar W per operand, "
                f"so no single value is correct.  Per-(operand, group) widths are already computed "
                f"by RegBand -- consuming them is.")
        per_op[op] = int(w)
    return per_op


class GirToRocisa:
    """Emit a finalized GIR block into a rocisa Module via the L3 `LeafEmitters`.

    `writer` is the active KernelWriterAssembly.  Construction builds the loop-order-invariant
    leaf contexts ONCE, then `emit_block` realizes any phase's plan.
    """

    def __init__(self, writer, kernel, tPA, tPB, prog):
        self.writer = writer
        self.kernel = kernel
        self.tPA, self.tPB = tPA, tPB
        # L3 OWNS the leaf emitters (they ARE the layer-2->3 realization).  The register ring width
        # W is a GIR fact passed as a real field -- no reaching into a retired walker's internals.
        self._M = int(prog.meta.get("M", 0))     # peel depth: prologue fills chunks 0..M-1
        self._memberRegions = dict(prog.meta.get("unit_member_regions", {}) or {})
        band = AnalysisManager().get(RegBandAnalysis(), prog)
        self._leaf = LeafEmitters(writer, reg_depth=_register_depth(prog, band))
        self._ctxMfma = self._leaf.buildMfmaContext(kernel, tPA, tPB)
        self._ctxRd = {"A": self._leaf.buildLdsReadContext(kernel, tPA),
                       "B": self._leaf.buildLdsReadContext(kernel, tPB)}
        self._tp = {"A": tPA, "B": tPB}
        # MICROSCALING SCALE TENSORS are ordinary operands here: theta gives `MXSA`/`MXSB` their own
        # paths, so GIR emits read acts for them exactly as it does for A and B, and they
        for _parent in (tPA, tPB):
            _mx = _parent.get("MX")
            if _mx is not None:
                _tc = _mx["tensorChar"]
                self._tp[_tc] = _mx
                self._ctxRd[_tc] = self._leaf.buildMxScaleReadContext(kernel, _mx)

    # -- scaffold consumed BY TAG (handed in by the fork per stage) --------------------------
    def emit_block(self, prog, phase, *, tpByOperand=None, internalPointerSwap=False) -> Module:
        """Realize the finalized GIR block `phase` into a rocisa Module."""
        tpByOperand = tpByOperand or self._tp
        w = self._leaf                      # reg_depth already set at construction (a GIR fact)
        out = Module("GIR %s body" % phase)

        acts = plan_block(prog, phase)
        cplan = plan_coverage(
            acts, lambda op: (prog.meta.get("read_quantum", {}) or {}).get(op),
            # whether theta FOLDED this operand's read axis, so a lone leader act is a
            # complete carrier group (it carries its span) rather than a missing one.
            folded_of=lambda op: bool((prog.meta.get("read_fold", {}) or {}).get(op))
                              or bool((prog.meta.get("coverage_axes", {}) or {}).get(op)))
        for _v in cplan.violations:
            raise RuntimeError(
                "GIR %s: %s\n  theta's merge and the acts in this block disagree; the emitted "
                "instruction would write a register whose act lives elsewhere." % (phase, _v))

        for _i, act in enumerate(acts):
            kind, at = act.kind, act.at
            if kind == "wmma":
                self._tag(out, phase, "wmma", "C[m=%s,n=%s] += A[m,k=%s]*B[n,k=%s]  regA=X%s regB=X%s"
                          % (at["idx0"], at["idx1"], at["u"], at["u"],
                             at.get("bufA"), at.get("bufB")))
                out.add(w.emitWmmaTile(self.kernel, self.tPA, self.tPB, self._ctxMfma,
                                       idx0=at["idx0"], idx1=at["idx1"], u=at["u"],
                                       bufA=at.get("bufA"), bufB=at.get("bufB"),
                                       bufMXA=at.get("bufMXA"), bufMXB=at.get("bufMXB")))
            elif kind == "read":
                tc = at["tc"]
                # `tile` is the WITHIN-REGION index and `tile_flat` the register index; they differ
                # only for a region-split operand, and passing one for both is the defect.
                self._tag(out, phase, "read", "%s[tile=%s,k=%s%s] shared->reg X%s%s"
                          % (tc, at.get("tile_flat", at["tile"]), at["k"],
                             "" if at.get("regions", 1) <= 1 else ",r%s" % at.get("region", 0),
                             at["reg_buf"], self._tok(at.get("token"))))
                _qd = cplan.decision(_i)
                _tok = (getattr(_qd, "tokens", ()) or self._token_ids(at)) if _qd else self._token_ids(at)
                out.add(w.emitLdsReadTile(self.kernel, self._tp[tc], self._ctxRd[tc],
                                          tileIdx=at["tile"], kIdx=at["k"],
                                          bufferIdx=at["reg_buf"],
                                          memToken=_tok,
                                          region=at.get("region", 0),
                                          regTileIdx=at.get("tile_flat", at["tile"]),
                                          kFlat=at.get("k_flat", at["k"]),
                                          quantum=_qd))
            elif kind == "swap":
                # A read swap names an operand, a copy swap names the Phi movement -- see
                # `gir/refs.copy_unit` and the act table in `gir/emit_plan`.
                who = at["operand"] if at["hop"] == "read" else at["unit"]
                self._tag(out, phase, "swap", "%s %s-hop buffer %s->%s"
                          % (self._who(who), at["hop"], at.get("gen_from"), at.get("gen_to")))
                self._emit_swap(out, at["hop"], who, tpByOperand, internalPointerSwap)
            elif kind == "gl2_prefetch":
                #: GIR chose the POINT; the instructions are the scaffold's own, unchanged.
                # Increment BEFORE issue -- the pair is ordered internally (the clamp zeroes
                _gl2tag = "gl2 depth=%s (cache warm, no consumer)" % at.get("depth")
                self._append_tag(self.writer.codes.gl2PrefetchIncrement, _gl2tag + " inc")
                self._append_tag(self.writer.codes.gl2Prefetch, _gl2tag + " issue")
                out.add(self.writer.codes.gl2PrefetchIncrement)
                out.add(self.writer.codes.gl2Prefetch)
            elif kind == "gr_inc":
                self._tag(out, phase, "gr_inc", "%s advance %s chunk(s) -> chunk %s"
                          % (self._who(at["unit"]), at.get("chunks", 1), at.get("to_chunk")))
                self._emit_gr_inc(out, at["unit"], tpByOperand, at.get("chunks", 1),
                                  at.get("to_chunk"), phase=phase)
            elif kind == "copy":
                self._tag(out, phase, "copy", "%s global->shared chunk gen=%s%s%s"
                          % (self._who(at["unit"]), at.get("gen", 0),
                             "" if at.get("region") is None else " region=%s" % at["region"],
                             self._tok(at.get("token"))))
                self._emit_copy_masked(out, at, tpByOperand)
            elif kind == "desc_enable":
                self._emit_desc_enable(out, tuple(at["unit"]), at["member"],
                                       bool(at["enabled"]), tpByOperand)
            elif kind == "region_inc":
                self._emit_region_inc_act(out, phase, at, tpByOperand)
            elif kind == "fence":
                self._emit_fence(out, phase, at)
            elif kind == "gsu_guard":
                pass          # R4: realized via the scaffold-anchor path
        return out

    @staticmethod
    def _tok(token):
        """`  tok=(kind,operand,gen)` for a completion token, or '' when the act carries none."""
        return "" if token is None else "  tok=%s" % (tuple(token),)

    @staticmethod
    def _who(unit):
        """The act's subject for a comment: `A` for an operand, `A+B` for a fused Phi movement.

        A fused movement's tag must NAME both members -- the whole point of the tag is that a reader
        of the `.s` can tell which acts GIR emitted and what each covers, and "copy A" over an
        instruction that also moves B is worse than no tag."""
        return "+".join(unit) if isinstance(unit, tuple) else unit

    @staticmethod
    def _append_tag(code, tag):
        """APPEND `tag` to every instruction comment in `code`, preserving what is already there."""
        mark = gir_tag(tag)
        for item in code.flatitems():
            c = getattr(item, "comment", None)
            if c is None:
                continue
            item.comment = "  ".join(part for part in (c, mark) if part)

    def _tag(self, out, phase, kind, detail):
        """Stamp a GIR act so a finding in the `.s` names the node that emitted it."""
        out.addComment0(gir_tag("%s %s %s" % (phase, kind, detail)))

    def _emit_fence(self, out, phase, at):
        """Realize a memory-ordering barrier: GIR decides where and what it orders, L3 the instruction.

        `_syncThreads` issues waitcnt + barrier, so it has the strength the anti-dependence demands."""
        buffers = at.get("buffers", ())
        tokens = self._fence_tokens(at)
        # The token set is printed: it is what the waitcnt derives from, so a barrier that does not
        # name it cannot be checked against the accesses it stands between.
        sync = sync_comment(tokens)
        self._tag(out, phase, "fence", "%s scope=%s covers %s edge(s) %s %s"
                  % ("+".join(buffers) or "?", at.get("scope"), at.get("edges"),
                     ",".join(at.get("kinds", ())), sync))
        code = self.writer._syncThreads(
            self.kernel, "GIR fence: order %s (%s) %s"
            % ("+".join(buffers) or "LDS", ",".join(at.get("kinds", ())), sync),
            memoryToken=tokens)
        if code is not None and code.count():
            self._append_tag(code, "fence %s" % ("+".join(buffers) or "LDS"))
            self._register_fence(code)
            out.add(code)

    def _register_fence(self, code):
        """Record the `SBarrier` objects in `code` as GIR-owned."""
        stack = [code]
        while stack:
            node = stack.pop()
            for item in node.items():
                if isinstance(item, SBarrier):
                    self.writer.states.girOwnedBarriers.append(item)
                elif hasattr(item, "items"):
                    stack.append(item)

    def _fence_tokens(self, at):
        """The token ids this fence stands between -- carried on the Mark by `FenceRegions`."""
        ids = tuple(at.get("tokens") or ())
        if not ids:
            raise RuntimeError(
                f"GIR fence over {at.get('buffers')} carries no token ids -- a fence with no "
                f"tokens is a barrier that orders nothing ( G-TOKEN should have caught this)")
        return sorted(int(t) for t in ids)

    def _fused_tp(self, unit, tpByOperand, what):
        """`(tP_owner, tP_peer)` for a Phi-fused movement, after checking the scaffold can realize it.

        The scaffold's fused form is ONE aliased descriptor: `tdm{tcA}Group0` serves both operands,
        each wave's copy of those SGPRs addressing its own, and wave parity selecting which
        (`KernelWriterAssembly.isTdmWaveSeparated`).
        """
        members = tuple(unit)
        # The groups the scaffold aliases, READ FROM THE ONE TABLE (`LoopModel/fuse.py`) that theta
        _present = ["A", "B"]
        _mxA, _mxB = self.tPA.get("MX"), self.tPB.get("MX")
        if _mxA is not None and _mxB is not None:
            _present += [_mxA["tensorChar"], _mxB["tensorChar"]]
        pairs = [tuple(g) for g in fuse_groups(self.kernel["TDMFuse"], _present)]
        if members not in pairs or not self.writer.isTdmWaveSeparated(self.kernel):
            raise NotImplementedError(
                f"GIR emitted a fused {what} for movement {members} "
                f"(isTdmWaveSeparated={self.writer.isTdmWaveSeparated(self.kernel)}).  The scaffold "
                f"aliases one descriptor per PAIR at NumWaves>1, and this kernel's pairs are "
                f"{pairs}.  A group outside them (a `paired` MX/data cross, or any fuse at single "
                f"wave) has no emitter; see task.")
        # From `tpByOperand`, not `self._tp`: the fork may hand in a different map (MX/meta), and
        # reading the constructor's copy would ignore it for fused movements only -- a discrepancy
        missing = [m for m in members if tpByOperand.get(m) is None]
        if missing:
            raise NotImplementedError(
                f"fused {what} for {members} but no tensor parameters for {missing}")
        return tpByOperand[members[0]], tpByOperand[members[1]]

    def _emit_swap(self, out, hop, who, tpByOperand, internalPointerSwap):
        """Realize ONE buffer-pointer swap via the scaffold's primitive.

        read hop -> `localReadSwapOffsets` (v_xor LocalReadAddr), per OP-CLASS: A and B read from
        their own LDS regions through their own address registers, so `who` is an operand name and
        multi-wave changes nothing here.
        """
        if hop == "read":
            tP = tpByOperand.get(who)
            if tP is None:
                return
            code = self.writer.localReadSwapOffsets(self.kernel, internalPointerSwap, tP)
        elif len(who) > 1:
            tP, _peer = self._fused_tp(who, tpByOperand, "descriptor swap")
            code = self.writer.tdmSwapLdsOffset(self.kernel, tP)
        else:
            tP = tpByOperand.get(who[0])
            if tP is None:
                return
            code = self.writer.tdmSwapLdsOffset(self.kernel, tP)
        if code is not None and code.count():
            self._append_tag(code, "swap %s %s-hop" % (self._who(who), hop))
            out.add(code)

    def _emit_gr_inc(self, out, unit, tpByOperand, chunks=1, to_chunk=None,
                     phase=None):
        """Realize ONE global-read increment for a Phi movement, via `writer.tdmIncrementGir`
        (s_add tdm+=inc, with the StaggerU WrapU cselect).  GIR OWNS the placement (a gr_increment
        Mark from the dataflow); the magnitude is L3's.

        `chunks` is how many summation chunks GIR's dataflow says the address must advance here.
        """
        if chunks != 1:
            raise NotImplementedError(
                f"gr_increment for {self._who(unit)!r} advances {chunks} summation chunks, but "
                f"tdmIncrementGir emits exactly one DepthU stride; a multi-chunk advance needs a "
                f"scaled increment on the scaffold side.")
        if len(unit) > 1:
            tP, tPFused = self._fused_tp(unit, tpByOperand, "global-read increment")
        else:
            tP, tPFused = tpByOperand.get(unit[0]), None
            if tP is None:
                return
        # `prefetchIndex` shifts the StaggerU WRAP COMPARE: `tdmIncrementAB` adds it to the loop
        # counter before deciding whether this increment is the one that must wrap the staggered
        pfi = self._wrap_lead(to_chunk, self._M) if to_chunk is not None else 0
        # `tdmIncrementGir`, not `tdmIncrementAB`: the GIR path owns the wrap decision, and the
        # shared function reconstructs it from PGR + phase.  Same emitted instructions today.
        code = self.writer.tdmIncrementGir(self.kernel, tP, wrapLead=pfi, tPFused=tPFused)
        if code is not None and code.count():
            self._append_tag(code, "gr_inc %s ->chunk%s pf=%d"
                                   % (self._who(unit), to_chunk, pfi))
            out.add(code)

    _STAGGER_PF_CONST = 2

    @classmethod
    def _wrap_lead(cls, to_chunk, M):
        """The shift that puts the wrap compare in the frame THIS advance runs in."""
        return cls._STAGGER_PF_CONST - min(to_chunk, M + 1)

    def _emit_region_inc_act(self, out, phase, at, tpByOperand):
        """Step the TDM descriptor one storage region, in the direction GIR chose."""
        self._tag(out, phase, "region_inc", "%s descriptor region %s->%s"
                  % (self._who(at["unit"]), at.get("from_region"), at.get("to_region")))
        # L3 DOES NOT GAIN THE 2-D CAPABILITY: GIR carries a per-axis step vector, and a step
        # moving SEVERAL region axes at once has no realization here.
        axis_steps = at.get("axis_steps") or {}
        if len(axis_steps) > 1:
            raise NotImplementedError(
                "region_increment for %s moves several region axes at once (%s): the TDM "
                "descriptor walks one axis per step, so a grid step needs one add per "
                "non-zero axis with its own stride.  GIR expresses it; L3 does not "
                "implement it.  TDMSplitA/B in {0,1,2} cannot produce this shape."
                % (self._who(at["unit"]), axis_steps))
        self._emit_region_inc(out, at["unit"], tpByOperand, int(at["steps"]))

    def _emit_copy_masked(self, out, at, tpByOperand):
        """The copy, with the descriptor group's non-members nulled across it.

        A SOLO movement still issues on the SHARED wave-separated descriptor, so a member this
        movement does not carry would move too unless it is nulled for the span of the load.
        """
        unit = tuple(at["unit"])
        absent = self._absent_from_unit(unit, tpByOperand)
        for member in absent:
            self._emit_desc_enable(out, unit, member, False, tpByOperand)
        self._emit_copy(out, at["unit"], at.get("gen", 0), tpByOperand,
                        memToken=self._token_ids(at),
                        regions=at.get("regions", 1), region=at.get("region"))
        for member in absent:
            self._emit_desc_enable(out, unit, member, True, tpByOperand)

    def _absent_from_unit(self, unit, tpByOperand):
        """Members of this movement's DESCRIPTOR group that the movement does not carry."""
        tc = next((m for m in unit if tpByOperand.get(m) is not None), None)
        if tc is None:
            return ()
        members, _ranges = self.writer.tdmFuseGroupOf(self.kernel, tc)
        return tuple(m for m in (members or ()) if m not in unit)

    def _emit_desc_enable(self, out, unit, member, enabled, tpByOperand):
        """Null (count=0) or restore the descriptor for the waves carrying `member`."""
        tP = tpByOperand.get(member)
        if tP is None:
            return
        code = self.writer.tdmDescriptorEnableGir(self.kernel, tP, enabled)
        if code is not None and code.count():
            self._append_tag(code, "desc_%s %s in %s"
                             % ("enable" if enabled else "null", member, self._who(unit)))
            out.add(code)

    def _walkTp(self, unit, tpByOperand, fallback):
        """The tP of the member that owns the region walk; a shorter member shares the descriptor
 but not its stride or its per-region load."""
        per = self._memberRegions.get(tuple(unit), {})
        if per and len(set(per.values())) > 1:
            return tpByOperand.get(max(per, key=lambda m: per[m]), fallback)
        return fallback

    def _emit_region_inc(self, out, unit, tpByOperand, steps):
        """Realize a descriptor walk of `steps` storage regions, via `tdmRegionIncrementGir`."""
        if len(unit) > 1:
            tP, _peer = self._fused_tp(unit, tpByOperand, "region_inc")
            # The walk belongs to the member that HAS the regions; a shorter member shares the
            # descriptor but not the stride.
            tP = self._walkTp(unit, tpByOperand, tP)
        else:
            tP = tpByOperand.get(unit[0])
            if tP is None:
                return
        back = steps < 0
        for _ in range(abs(steps)):
            code = self.writer.tdmRegionIncrementGir(self.kernel, tP, back=back)
            if code is not None and code.count():
                self._append_tag(code, "region_inc %s %s"
                                 % (self._who(unit), "back" if back else "fwd"))
                out.add(code)

    def _emit_copy(self, out, unit, gen, tpByOperand, memToken=None, regions=1, region=None):
        """Realize ONE global->shared movement, for the buffer generation GIR's Move names.

        One call per Move -- so the prologue's M peel fills each emit, which a single pre-built
        module per operand could not do.  Everything else about the load (addressing, vector width,
        DTL/DTV variants) stays in the scaffold's `globalReadDo`, reached through the leaf.
        """
        if len(unit) > 1:
            tP, _peer = self._fused_tp(unit, tpByOperand, "copy")
            tP = self._walkTp(unit, tpByOperand, tP)
        else:
            tP = tpByOperand.get(unit[0])
            if tP is None:
                return
        if regions > 1 and region is None:
            raise NotImplementedError(
                f"copy of {self._who(unit)!r} is split into {regions} storage regions but its "
                f"coordinate names NO region axis, so the walk cannot be derived. The"
                f"region count and the coordinate enumeration disagree; check that the axis "
                f"survived `translate._keep` and `aRegions`/`bRegions` (both gate on extent > 1) "
                f"while `Operand.split` still reports {regions}.")
        code = self._leaf.emitCopyTile(self.kernel, tP, bufIdx=int(gen),
                                       memToken=memToken, regions=regions,
                                       region=int(region or 0))
        if code is not None and code.count():
            self._append_tag(code, "copy %s gen=%s" % (self._who(unit), gen))
            out.add(code)

    @staticmethod
    def _token_ids(at):
        """The memory-token id(s) this act touches -- GIR's own numbering."""
        ids = at.get("token_ids") or ()
        return [int(t) for t in ids] if ids else None


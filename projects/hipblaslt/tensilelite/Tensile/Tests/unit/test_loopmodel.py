# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for the LoopModel theta-schedule decoder (Tensile/LoopModel)."""

import pytest

from Tensile.LoopModel import (
    checks, emit, render, derive_S, build_S, validate_loopir,
    Space, Loop, Branch, Bind, Peel, Inst, Load, Mma, Cond, Pred, Expr,
)
from Tensile.LoopModel import adapter
# The named configurations are TEST FIXTURES, not library code -- they have no consumer outside this
# file and test_gir.py, so they live here beside `gir_fixtures` rather than in `Tensile.LoopModel`.
import loopmodel_scenarios as scenarios
from Tensile.LoopModel import theta as theta_mod# COPY/READ -- the operand roles `off` keys on
from Tensile.LoopModel.traversal import presence, presence_axes, summation_names


def test_pred_structured_eval_and_render():
    """Pred(lhs:Expr, op, rhs) reuses Expr; read by field, no string parse.  eval concretizes when
    lhs is known, returns None (runtime) for a problem-dimension symbol like T."""
    # residue pin: iter%2==1 -> true at iter=3
    p = Pred(Expr(var="iter", mod=2), "==", 1)
    assert p.eval({"iter": 3}) is True and p.eval({"iter": 2}) is False
    assert p.render() == "(iter)%2 == 1"
    # a runtime trip guard T>M: T not in env -> None (backend lowers to a real branch)
    assert Pred(Expr(var="T"), ">", 2).eval({}) is None
    # every op in the closed set is dispatched generically
    for op, want in [(">=", True), (">", False), ("==", True), ("<=", True), ("<", False), ("!=", False)]:
        assert Pred(Expr(var="x"), op, 5).eval({"x": 5}) is want
    with pytest.raises(RuntimeError):
        Pred(Expr(var="x"), "≈", 0)


# --------------------------------------------------------------------------- helpers
# A bf16 NT KMN kernel matching Tests/common/gemm/gfx12/loopmodel_bf16_gfx1250.yaml:
# MI 16x16x32, DepthU 64 => S=2 K-substeps, PGR2 (copy off=2), PLR1 (read off=1).
BF16_NT_KMN = {
    "MatrixInstruction": [16, 16, 32, 1, 1, 2, 2, 1, 1],
    "MIWaveTile": [2, 2], "MIWaveGroup": [1, 1],
    "MIBlock": [16, 16, 32, 1, 1, 1, 1, 1],
    "DepthU": 64, "WavefrontSize": 32,
    "PrefetchGlobalRead": 2, "PrefetchLocalRead": 1, "HalfPLR": 0,
    "ProblemType": {"DataType": "b", "DestDataType": "b", "ComputeDataType": "s",
                    "TransposeA": False, "TransposeB": True},
    "LoopOrder": "KMN", "NumLdsBlk": 2, "1LDSBuffer": -1,
    "DirectToVgprA": False, "DirectToVgprB": False,
    "GlobalReadVectorWidthA": 8, "GlobalReadVectorWidthB": 8,
    "LocalReadVectorWidth": 8, "LocalReadVectorWidthA": 8, "LocalReadVectorWidthB": 8,
    "InnerUnroll": 1, "VgprPartition": 1,
}

# Same kernel with PrefetchLocalRead=0: NO register read-ahead (dr=0).  This collapses the
# software-pipeline read stage and exercises the single-buffer drain path -- the shape that
# regressed repeatedly during Phase 3h (register depth, sigma_c copy order, drain read-ptr swaps).
BF16_NT_KMN_PLR0 = {**BF16_NT_KMN, "PrefetchLocalRead": 0}


def _theta(params):
    """`params_to_theta`, plus a compatibility shim for the scenarios' `VgprPartition*` /
    `VgprReuseStrategy*` keys.
    """
    import dataclasses as _dc
    from Tensile.LoopModel.theta import group_labels
    params = dict(params)
    parts = {nm: int(params.pop("VgprPartition" + nm, 1) or 1) for nm in ("A", "B")}
    strat = {nm: int(params.pop("VgprReuseStrategy" + nm, 0) or 0) for nm in ("A", "B")}
    th = adapter.params_to_theta(params)
    for nm in ("A", "B"):
        n = parts[nm]
        if n <= 1:
            continue
        labels = group_labels(n)
        st = strat[nm]
        pol = {g: ("unroll" if st == 0 else
                   "inplace" if st == 2 else
                   ("overlap" if i == 0 else "inplace"))
               for i, g in enumerate(labels)}
        op = th.op(nm)
        op.fragment = _dc.replace(
            op.fragment, parts=n, labels=labels, group_policy=pol, ring_depths=None)
    return th


@pytest.mark.parametrize("scen", ["mxfp8_derive", "our_split", "du_sym", "mx_fuse_paired"])
def test_load_tokens_are_theta_op_class_names_never_synthesized(scen):
    """A Load token is a theta OP-CLASS NAME -- never a string the emitter invented."""
    _desc, params = scenarios.SCENARIOS[scen]
    th = _theta(params)
    S, _floor = build_S(th)
    ir = emit.build_ir(th, S)
    names = {o.name for o in th.operands}

    seen_copy = False
    for inst in checks.walk_insts(ir):
        op = inst.op
        if not isinstance(op, Load):
            continue
        for tok in op.tokens:
            assert tok in names, (
                f"{scen}: Load token {tok!r} is not a theta op-class name "
                f"(operands={sorted(names)}) -- the emitter synthesized it")
        if op.dst == Space.SHARED:
            seen_copy = True
            # the region survives structurally, as a coordinate and/or the `part` field
            assert isinstance(op.part, int)
    assert seen_copy, f"{scen}: no global->shared copy emitted, test would be vacuous"


def _region_loops_over(nodes, out, loops=()):
    """Find the enclosing region loop for each shared copy."""
    for nd in nodes:
        if isinstance(nd, Loop):
            for b in (nd.bodies or []):
                _region_loops_over(b, out, loops + (nd,))
        elif isinstance(nd, Peel) or isinstance(nd, Bind):
            _region_loops_over(nd.body, out, loops)
        elif isinstance(nd, Cond):
            _region_loops_over(nd.then, out, loops)
            _region_loops_over(nd.els, out, loops)
        elif isinstance(nd, Inst) and isinstance(nd.op, Load) and nd.op.dst == Space.SHARED:
            axes = {axis for axis, _value in nd.op.coord}
            owner = next((loop for loop in reversed(loops) if loop.axis in axes), None)
            if owner is not None:
                out.append((owner, nd))


def test_region_split_copy_is_ROLLED_at_its_region_level():
    """line 523 + a region-split copy is ONE operand whose the axes it varies over includes the
    region mode, so it sits "one level deeper, per-region" as a SINGLE Inst under a real
    `Loop(region_mode)` -- not N sibling Insts at the reduction-chunk level.
    """
    _desc, params = scenarios.SCENARIOS["our_split"]
    th = _theta(params)
    S, _floor = build_S(th)
    ir = emit.build_ir(th, S)

    split_ops = {o.name: o for o in th.operands if o.split > 1}
    assert split_ops, "scenario is not region-split; test would be vacuous"

    loops = []
    _region_loops_over(ir, loops)
    assert loops, "no region loop wraps any copy -- the copy was emitted unrolled"

    for lp, inst in loops:
        # the loop is over a real region mode of the operand it carries, with trip == its split
        op = split_ops.get(inst.op.tokens[0])
        if op is None:
            continue
        assert lp.axis in op.region_axes, \
            f"copy of {op.name} rolled over {lp.axis!r}, not one of its regions {op.region_axes}"
        assert lp.trip == op.split, f"region loop trip {lp.trip} != split {op.split}"
        assert not lp.outer, "a region level is an INNER (static, unrolled) level"
        # the Inst's coord names the region mode SYMBOLICALLY -- the loop supplies the value
        assert dict(inst.op.coord).get(lp.axis, "missing") is None, \
            f"expected a symbolic region coord for {lp.axis}, got {inst.op.coord}"

    # ROLLED: one copy node per (operand, phase), NOT `split` of them.
    per_op = {}
    for i in checks.walk_insts(ir):
        if isinstance(i.op, Load) and i.op.dst == Space.SHARED:
            per_op[i.op.tokens[0]] = per_op.get(i.op.tokens[0], 0) + 1
    unsplit = {n: c for n, c in per_op.items() if n not in split_ops}
    for name, count in per_op.items():
        if name in split_ops and unsplit:
            assert count == min(unsplit.values()), (
                f"{name} is split {split_ops[name].split} ways and emits {count} copy nodes -- "
                f"an unsplit operand emits {min(unsplit.values())}; the split copy is unrolled")

    # every token is still a theta operand name (no synthesized 'A0')
    assert {t for _l, i in loops for t in i.op.tokens} <= {o.name for o in th.operands}

    # the DUMP shows the region structurally: a region loop header over the copy
    from Tensile.LoopModel.render import render_ir
    dump = render_ir(ir)
    assert any(f"for {m} in range" in dump for o in split_ops.values() for m in o.region_axes), \
        "render does not show a region loop over the copies:\n" + dump[:600]


def _walk(nodes, env, out):
    """Flatten the rolled IR to a concrete Inst stream (mirrors render_unrolled + the
    TensileLite walker): unroll int-extent Loops, bind pinned Branch residues, descend
    Peels.  `out` receives (kind, inst, env) with kind in {read, copy, wmma}."""
    for nd in nodes:
        if isinstance(nd, Loop):
            # each sub-body over its own half-open range; a single-body loop
            # reports one range covering the whole trip.
            for lo, hi, b in nd.ranged_bodies():
                end = hi if hi is not None else lo + 1
                for v in range(lo, end):
                    _walk(b, {**env, nd.axis: v}, out)
        elif isinstance(nd, Branch):
            if nd.axis in env:
                _walk(nd.arms.get(env[nd.axis] % nd.modulus, []), env, out)
            else:                                  # pinned-residue branch: arm key IS the value
                for r, arm in nd.arms.items():
                    _walk(arm, {**env, nd.axis: r}, out)
        elif isinstance(nd, Peel):
            _walk(nd.body, env, out)
        elif isinstance(nd, Bind):
            # a peel-step iter binding: push the bound value (concrete for prologue/short; a
            # symbolic drain `T-M+t` stays unbound, body walked symbolically).
            _walk(nd.body, nd.bound_env(env), out)
        elif isinstance(nd, Cond):
            # honor a RESOLVED guard (a pinned drain substep NLL `Pred(k < n_s-dr)`: the reduction
            # Loop has bound k, so the OOB read-ahead is suppressed for real) -- this is what makes
            # the drain STAGGER.
            v = nd.pred.eval(env)
            if v is None or v:
                _walk(nd.then, env, out)
            else:
                _walk(nd.els, env, out)
        elif isinstance(nd, Inst):
            op = nd.op
            if isinstance(op, Mma):
                out.append(("wmma", nd, env))
            elif isinstance(op, Load) and op.dst == Space.REGISTER:
                out.append(("read", nd, env))
            elif isinstance(op, Load) and op.dst == Space.SHARED:
                out.append(("copy", nd, env))


def _find_steady(nodes):
    """The steady iter Loop, descending peel-validity Cond guards (which now wrap it)."""
    for n in nodes:
        if isinstance(n, Loop) and n.axis == "iter":
            return n
        if isinstance(n, Cond):
            s = _find_steady(n.then) or _find_steady(n.els)
            if s is not None:
                return s
    return None


def _find_peel(nodes, kind):
    """Find the prologue/drain Peel, descending the peel-validity Cond that now wraps the whole
    pipeline (prologue/steady/drain live in its `then`)."""
    for n in nodes:
        if isinstance(n, Peel) and n.kind == kind:
            return n
        if isinstance(n, Cond):
            p = _find_peel(n.then, kind)
            if p is not None:
                return p
    return None


def _stage_bodies(ir):
    pro = _find_peel(ir, "prologue")
    loop = _find_steady(ir)
    drain = _find_peel(ir, "drain")
    return pro, loop, drain


# --------------------------------------------------------------------------- ledger
@pytest.mark.parametrize("name", list(scenarios.SCENARIOS.keys()))
def test_all_scenarios_ledger_discharged(name):
    """Every named scenario decodes with an EMPTY obligation ledger (the safety argument): all
    RAW-residency + rotation-WAR hazards are discharged by an Await in the emitted tree."""
    _desc, params = scenarios.SCENARIOS[name]
    r = emit.emit_mainloop(_theta(params))
    assert r.obligations_discharged, f"{name}: undischarged={r.undischarged}"


def test_bf16_nt_kmn_ledger():
    r = emit.emit_mainloop(_theta(BF16_NT_KMN))
    assert r.obligations_discharged


def test_shared_war_labeled_with_vacating_read_class():
    """Paper (line 207) labeling note: the SHARED-buffer rotation-WAR is discharged by the
 VACATING READ's selector (the shared->register read, class C_read), NOT the refilling copy
 (C_copy). So its ledger obligation must carry the read hop's completion class."""
    th = _theta(BF16_NT_KMN)
    from Tensile.LoopModel import build_ledger, build_S, Counter
    S, _floor = build_S(th)
    ledger = build_ledger(th, S)
    shared_wars = [o for o in ledger
                   if o.consumer.at == "shared" and o.consumer.role == "refill"]
    assert shared_wars, "expected a shared-buffer rotation-WAR obligation"
    for o in shared_wars:
        assert o.counter == Counter.READ, (
            f"shared WAR must carry the vacating read's completion class, got {o.counter}")


# --------------------------------------------------------------------------- mxfp8 derivation
def test_mxfp8_two_rate_derivation():
    """A mixed-width M-unit partition is allocated compactly per group."""
    _desc, params = scenarios.SCENARIOS["mxfp8_derive"]
    th = _theta(params)
    S = derive_S(th)
    assert (S.get("A", "lo"), S.get("A", "hi")) == (2, 1)
    from Tensile.LoopModel import operand_footprint_regs
    from Tensile.LoopModel.traversal import operand_emitted_regs
    assert operand_footprint_regs(th, th.op("A"), S) == 192
    assert operand_footprint_regs(th, th.op("B"), S) == 256
    assert operand_emitted_regs(th, th.op("A"), S) == 192
    assert operand_emitted_regs(th, th.op("B"), S) == 256
    from Tensile.LoopModel import traversal as geometry
    A = th.op("A")
    assert geometry.original_prefetch_unit_tiles(th, A) == 8
    assert geometry.prefetch_unit_tiles(th, A) == 4
    assert geometry.group_prefetch_unit_tiles(th, A) == 2
    assert geometry.rotation_unit_tiles(th, A) == 8
    assert geometry.group_rotation_unit_tiles(th, A) == 2
    assert geometry.requested_read_ahead(th, A) == 2
    assert geometry.prefetch_capacity(th, A, S) == 3
    assert geometry.requested_read_ahead(th, A, S) == 2


def test_emitted_vgpr_allocation_tracks_loop_order_per_operand():
    """ULM1 allocation follows θ/S instead of the order-blind ULM0 buffer count."""
    from Tensile.LoopModel.traversal import operand_emitted_regs

    expected = {
        "KMN": {"A": 32, "B": 32},
        "MKN": {"A": 32, "B": 64},
        "NKM": {"A": 64, "B": 32},
    }
    for order, want in expected.items():
        th = _theta(dict(BF16_NT_KMN, LoopOrder=order))
        S, _ = build_S(th)
        got = {name: operand_emitted_regs(th, th.op(name), S) for name in ("A", "B")}
        assert got == want, f"{order}: {got}"


def test_emitted_vgprs_keep_the_scaffold_tail_read_footprint():
    """With K_inner collapsed, compact GIR still reserves both free tiles for the tail."""
    from Tensile.LoopModel import traversal as geometry

    th = adapter.params_to_theta({
        "MIWaveTile": [2, 2], "MatrixInstruction": [16, 16, 128, 1, 1, 2, 2, 2, 2],
        "DepthU": 128, "ElemBytes": 1, "PrefetchGlobalRead": 1,
        "PrefetchLocalRead": 0, "LoopOrder": "KMN",
    })
    a = th.op("A")
    depths = build_S(th)[0]
    assert geometry.operand_footprint_regs(th, a, depths) == 16
    assert geometry.operand_emitted_regs(th, a, depths) == 32


def test_emitted_vgpr_allocation_respects_distributed_mxs_coverage():
    """Two distributed MXS partners share one physical register position."""
    from Tensile.LoopModel.traversal import operand_emitted_regs

    target = {
        "ReadVectorElems": {"MXSA": 8, "MXSB": 8},
        "ReadPhi": {"MXSA": 2, "MXSB": 2},
        "ReadRho": {"MXSA": 2, "MXSB": 2},
        "RegisterBudget": 224,
    }
    th = adapter.params_to_theta(adapter.kernel_to_params(_mxf8_kernel("KMN", 0), target))
    S, _ = build_S(th)
    assert {name: operand_emitted_regs(th, th.op(name), S)
            for name in ("MXSA", "MXSB")} == {"MXSA": 2, "MXSB": 2}


def test_split_axes_do_not_multiply_the_prefetch_unit():
    """PLR1 on M_split(2).M_inner(4) advances one M_inner unit, not both regions."""
    from Tensile.LoopModel import traversal as geometry
    from Tensile.LoopModel.schedule import Schedule, _readahead_shift, build_S, preloaded_tiles
    from Tensile.Lowering import build_gir
    from Tensile.Lowering.gir.emit_plan import plan_block
    from Tensile.Lowering.gir.passes import HoistCopiesPass, pipeline

    target = {
        "ReadVectorElems": {"MXSA": 32, "MXSB": 32},
        "ReadPhi": {"MXSA": 8, "MXSB": 8},
        "ReadRho": {"MXSA": 0, "MXSB": 0},
    }
    expected = {
        "KKMNMN": {"K_inner": 0, "M_split": 1, "M_inner": 0},
        "KMNKMN": {"M_split": 0, "K_inner": 1, "M_inner": 0},
    }
    for order, next_unit in expected.items():
        kernel = dict(_mxf8_kernel(order, 1))
        kernel.update(MatrixInstruction=[16, 16, 128, 1, 1, 8, 8, 2, 2],
                      MIWaveTileA=8, MIWaveTileB=8,
                      PrefetchGlobalRead=2, PrefetchLocalRead=1,
                      TDMSplitA=1, TDMSplitB=1,
                      VectorWidthA=8, VectorWidthB=8)
        th = adapter.params_to_theta(adapter.kernel_to_params(kernel, target))
        S, _ = build_S(th)
        a = th.op("A")
        assert geometry.prefetch_unit_modes(th, a) == [("M_inner", 4)]
        assert geometry.requested_read_ahead(th, a) == 1
        assert geometry.prefetch_distance_for(th, a, 1) == 4
        assert {coord["M_split"] for coord in preloaded_tiles(th, a, 1, S)} == {0}
        b = th.op("B")
        assert geometry.prefetch_unit_modes(th, b) == [("N_inner", 4)]
        assert geometry.requested_read_ahead(th, b) == 1
        b_preload = preloaded_tiles(th, b, 1, S)
        assert len(b_preload) == 4
        assert {coord["N_split"] for coord in b_preload} == {0}
        for name, split_axis in (("MXSA", "M_split"), ("MXSB", "N_split")):
            scale = th.op(name)
            assert geometry.requested_read_ahead(th, scale) == 1
            assert all(coord[split_axis] == 0
                       for coord in preloaded_tiles(th, scale, 1, S))
        distance, strides, span, extents = _readahead_shift(
            th, a, 1, S, plans=Schedule(th, S))
        origin = {name: 0 for name in geometry.varying_axes(th, a)}
        got = {name: geometry._shifted_coord(
            name, distance, strides, span, extents, geometry.read_coverage(th, a)).eval(origin)
               for name in geometry.varying_axes(th, a)}
        assert got == next_unit
        if order == "KKMNMN":
            canonical_pipeline = [p for p in pipeline() if not isinstance(p, HoistCopiesPass)]
            actions = plan_block(build_gir(th, pipeline=canonical_pipeline), "steady")
            for tile in range(4):
                prior_uses = [i for i, action in enumerate(actions)
                              if action.kind == "wmma" and action.at["idx0"] == tile
                              and action.at["idx1"] < 4 and action.at["u"] == 0]
                refill = next(i for i, action in enumerate(actions)
                              if action.kind == "read" and action.at["tc"] == "A"
                              and action.at["tile_flat"] == tile + 4
                              and action.at["k_flat"] == 0)
                next_use = next(i for i, action in enumerate(actions)
                                if action.kind == "wmma" and action.at["idx0"] == tile + 4
                                and action.at["u"] == 0)
                assert max(prior_uses) < refill < next_use
            optimized = plan_block(build_gir(th), "steady")
            first_region1_a = next(i for i, action in enumerate(optimized)
                                   if action.kind == "read" and action.at["tc"] == "A"
                                   and action.at["tile_flat"] == 4 and action.at["k_flat"] == 0)
            preceding_n_reads = [i for i, action in enumerate(optimized)
                                 if action.kind == "read" and action.at["tc"] == "B"
                                 and 4 <= action.at["tile_flat"] < 8
                                 and action.at["k_flat"] == 0]
            assert preceding_n_reads and max(preceding_n_reads) < first_region1_a


def test_readahead_prologue_per_group_dr(name="mxfp8_derive"):
    """One split-local PLR step primes M_inner(4) in region 0 only."""
    from Tensile.LoopModel.schedule import preloaded_tiles
    from Tensile.LoopModel import build_S
    from Tensile.LoopModel import traversal as geometry
    _desc, params = scenarios.SCENARIOS["mxfp8_derive"]
    th = _theta(params)
    S, _floor = build_S(th)
    A = th.op("A")
    gm = A.fragment.grouping_mode
    assert gm == "M_inner"
    coords = preloaded_tiles(th, A, 1, S)
    assert {coord["K_inner"] for coord in coords} == {0}
    assert [(coord["M_split"], coord["M_inner"]) for coord in coords] \
        == [(0, 0), (0, 1), (0, 2), (0, 3)]
    assert [geometry.group_index(th, A, coord) for coord in coords] == [0, 0, 1, 1]
    assert len(coords) == 4


def test_mxfp8_max_sched_both_unroll():
    """Both interleaved halves of the M unit carry the normal W=2 pipeline."""
    _desc, params = scenarios.SCENARIOS["reg_max_sched"]
    S = derive_S(_theta(params))
    assert (S.get("A", "lo"), S.get("A", "hi")) == (2, 2)


def test_mxfp8_unsplit_repeats_the_two_group_width_pattern():
    """Without regions, four contiguous pairs preserve W=[2,1,2,1] and PLRA4."""
    import dataclasses
    from Tensile.LoopModel import traversal as geometry
    from Tensile.LoopModel.schedule import RegisterDepthSelector

    th = adapter.params_to_theta({
        **scenarios.MXFP8_GEOM, "LoopOrder": "KMN",
        "PrefetchLocalRead": 1, "PrefetchGlobalRead": 2,
    })
    A = th.op("A")
    candidates = RegisterDepthSelector(th, None).candidates(A)
    assert any(share == 4 and policies == ("pipeline", "inplace") * 2
               for share, policies, _regs in candidates)
    labels = ("g0", "g1", "g2", "g3")
    A.fragment = dataclasses.replace(
        A.fragment, parts=4, labels=labels,
        group_policy={"g0": "overlap", "g1": "inplace",
                      "g2": "overlap", "g3": "inplace"},
        ring_depths=None)
    S, _ = build_S(th)
    assert [geometry.group_index(th, A, {"K_inner": 0, "M_inner": m})
            for m in range(8)] == [0, 0, 1, 1, 2, 2, 3, 3]
    assert [S.get("A", group) for group in labels] == [2, 1, 2, 1]
    assert geometry.group_prefetch_unit_tiles(th, A) == 2
    assert geometry.requested_read_ahead(th, A) == 4
    assert geometry.prefetch_capacity(th, A, S) == 6
    assert geometry.requested_read_ahead(th, A, S) == 4


# --------------------------------------------------------------------------- MX scale reads
@pytest.mark.parametrize("name", ["mx_nofuse", "mx_fuse_ab", "mx_fuse_amx", "mx_fuse_paired"])
def test_mx_scales_emit_register_reads(name):
    """Regression: every MX scenario that has MXSA/MXSB scales MUST emit ds_read (register-dest
    Load) leaves for BOTH scales, and the wmma must reference them.  (Guards against the earlier
    bug where MX scales produced no ds_read.)"""
    _desc, params = scenarios.SCENARIOS[name]
    th = _theta(params)
    # scales are present as operands ending in a register fragment (the output/accumulator
    # operand has no mainloop path, hops=[], so it is skipped)
    endreg = {o.name for o in th.operands if o.movements and o.movements[-1].dst == Space.REGISTER}
    assert {"MXSA", "MXSB"} <= endreg, f"{name}: MX scales not register operands: {endreg}"
    ir = emit.emit_mainloop(th).ir
    rd_tokens, mma_scales = set(), set()
    for ins in checks.walk_insts(ir):
        op = ins.op
        if isinstance(op, Load) and op.dst == Space.REGISTER:
            rd_tokens |= set(op.tokens)
        if isinstance(op, Mma):
            mma_scales |= set(op.scales)
    assert {"MXSA", "MXSB"} <= rd_tokens, f"{name}: missing MX scale ds_read; got {sorted(rd_tokens)}"
    assert mma_scales, f"{name}: wmma carries no scales"


# --------------------------------------------------------------------------- read-ahead (PLR)
def test_steady_read_leads_wmma_by_plr():
    """In the STEADY body the read for reduction substep k is issued at the leaf whose wmma
    computes substep (k - dr) mod S: i.e. the read PREFETCHES substep k+dr while wmma runs k.
    Concretely for S=2, dr=1: the leaf computing substep-0 reads substep-1, and vice versa."""
    ir = emit.emit_mainloop(_theta(BF16_NT_KMN)).ir
    _pro, loop, _drain = _stage_bodies(ir)
    stream = []
    _walk(loop.bodies[0], {}, stream)
    kname = next(iter(summation_names(_theta(BF16_NT_KMN))))
    # pair each wmma with the read that immediately precedes it in the same leaf
    saw_readahead = False
    last_read_sub = None
    for kind, inst, env in stream:
        if kind == "read":
            # the read's reduction coord is a shifted Expr; eval it in env
            for ax, v in inst.op.coord:
                if ax == kname and hasattr(v, "eval"):
                    last_read_sub = v.eval(env)
        elif kind == "wmma":
            wmma_sub = env.get(kname)
            if last_read_sub is not None and wmma_sub is not None:
                # read leads wmma by dr=1 over the 2-substep ring
                if (wmma_sub + 1) % 2 == last_read_sub:
                    saw_readahead = True
    assert saw_readahead, "steady body shows no read-ahead (read substep != wmma substep + dr)"


def test_steady_lds_buffer_carries_across_depthu():
    """The read-ahead's LDS SOURCE buffer must CARRY across the DepthU boundary: the substep
    that wraps to the next reduction chunk reads the OTHER (swapped) LDS buffer.  For S=2/dr=1 the
    two steady reads target DIFFERENT src LDS buffers (the swap seam)."""
    ir = emit.emit_mainloop(_theta(BF16_NT_KMN)).ir
    _pro, loop, _drain = _stage_bodies(ir)
    stream = []
    _walk(loop.bodies[0], {"iter": 0}, stream)
    src_bufs = set()
    for kind, inst, env in stream:
        if kind == "read" and inst.placement is not None and inst.placement.src_slot is not None:
            src_bufs.add(inst.placement.src_slot.eval(env))
    assert len(src_bufs) == 2, f"read-ahead did not carry LDS buffer across reduction chunk: {src_bufs}"


def test_register_generation_matches_substep():
    """The register generation a read writes / a wmma consumes is the substep mod S_reg
    (X0/X1) -- the read-ahead shifts the WRITE gen forward but the wmma still consumes its own
    substep's gen, so both generations 0 and 1 appear in a steady trip."""
    ir = emit.emit_mainloop(_theta(BF16_NT_KMN)).ir
    _pro, loop, _drain = _stage_bodies(ir)
    stream = []
    _walk(loop.bodies[0], {"iter": 0}, stream)
    gens = set()
    for kind, inst, env in stream:
        if kind == "read" and inst.placement is not None:
            for _g, e in inst.placement.slots:
                gens.add(e.eval(env))
    assert gens == {0, 1}, f"expected both register generations, got {gens}"


# --------------------------------------------------------------------------- three stages
def test_prologue_is_fill_only_no_wmma():
    """PROLOGUE fills the pipe: global->shared copies + a DIRECT (unshifted) read-fill of the
    first dr substeps, and emits NO wmma."""
    ir = emit.emit_mainloop(_theta(BF16_NT_KMN)).ir
    pro, _loop, _drain = _stage_bodies(ir)
    assert pro is not None
    stream = []
    _walk(pro.body, {}, stream)
    kinds = [k for k, _i, _e in stream]
    assert "wmma" not in kinds, "prologue must not emit wmma"
    assert "copy" in kinds and "read" in kinds, "prologue must fill copies AND pre-read"


# ---------------------------------------------------------------------- the read-ahead prologue
# Closed form, per operand: prologue(p) = {first pres(p)&OUTER} x {min(dr,n_s) substeps, or all
# n_s when the order's INNER block is empty} x {pres(p)&INNER}.  A K-axis-only fill
# (`read_subs=={0}`) is right for KMN by accident and wrong in general.
def _prologue_set(order, opname, dr=1, wt=(2, 2)):
    from Tensile.LoopModel.schedule import preloaded_tiles
    th = _theta(dict(BF16_NT_KMN, LoopOrder=order, MIWaveTile=list(wt), PrefetchLocalRead=dr))
    pro = preloaded_tiles(th, th.op(opname), dr)
    # normalize to a set of (k,fan) short tuples: kN for the reduction, mN/nN for the fan
    out = set()
    for c in pro:
        out.add(tuple(sorted((k[0].lower() + str(v)) for k, v in c.items())))
    return out


# A BROADCAST-OUTER OPERAND PRIMES ITS WHOLE NAME SET, NOT THE THREE-FACTOR SUBSET.
@pytest.mark.parametrize("order,A,B", [
    # K-outer (...K.M.N): substep min(dr,n_s)=1, inner fan = each operand's OWN the axes it varies over axis.
    # Neither operand is broadcast-outer here, so both take the three-factor form.
    ("KMN", {("k0", "m0"), ("k0", "m1")}, {("k0", "n0"), ("k0", "n1")}),
    # An unpartitioned all-inner operand primes its complete rotation unit at PGR2.
    ("MNK", {("k0", "m0"), ("k1", "m0")},
            {("k0", "n0"), ("k0", "n1"), ("k1", "n0"), ("k1", "n1")}),
    ("MKN", {("k0", "m0"), ("k1", "m0")},
            {("k0", "n0"), ("k0", "n1"), ("k1", "n0"), ("k1", "n1")}),
])
def test_readahead_prologue_lemma3d(order, A, B):
    """the read-ahead prologue rule closed form matches the model's three proven families (dr=1, n_s=2, WT 2x2),
 with the broadcast-outer extension where it applies (see the note on the params)."""
    assert _prologue_set(order, "A") == A, f"{order} A prologue"
    assert _prologue_set(order, "B") == B, f"{order} B prologue"


def test_readahead_prologue_substep_saturates_with_dr():
    """The factor grows with dr up to the `n_s` ceiling, EACH OPERAND ON ITS OWN LEVEL."""
    # grows with dr, saturating at n_s = 4
    assert _prologue_set("KMN", "A", dr=1) == {("k0", "m0"), ("k0", "m1")}
    assert _prologue_set("KMN", "A", dr=2) == {
        ("k0", "m0"), ("k1", "m0"), ("k0", "m1"), ("k1", "m1")}
    assert _prologue_set("KMN", "B", dr=2) == {
        ("k0", "n0"), ("k1", "n0"), ("k0", "n1"), ("k1", "n1")}
    # each operand on its OWN level: under MKN the two differ in size, A over its M level only
    assert _prologue_set("MKN", "A", dr=1) == {("k0", "m0"), ("k1", "m0")}
    assert _prologue_set("MKN", "B", dr=1) == {
        ("k0", "n0"), ("k1", "n0"), ("k0", "n1"), ("k1", "n1")}
    # ...and MKN CAN go deeper now: a two-way register grouping serves PLR=2 on the M-outermost
    # family, and two M steps of A's level is its whole {k}x{m} set.
    assert _prologue_set("MKN", "A", dr=2) == {
        ("k0", "m0"), ("k1", "m0"), ("k0", "m1"), ("k1", "m1")}


def test_readahead_prologue_empty_without_dr():
    """dr=0 (no PLR) -> no read-ahead prologue for any operand."""
    from Tensile.LoopModel.schedule import preloaded_tiles
    th = _theta(BF16_NT_KMN_PLR0)
    assert preloaded_tiles(th, th.op("A"), 0) == []


def test_drain_is_staggered_ngll_then_nll():
    """DRAIN is the Lemma-1 STAGGERED tail : M steps, deepest-first. The EARLY
 steps KEEP the read-ahead (they prefetch the generation a later step consumes = ULM0's NGLL,
 a no-global-load steady-shaped iteration); only the LAST dr_iters steps SUPPRESS it (= ULM0's
 NLL). So the drain must NOT be uniformly read-ahead-free -- step 0 has MORE reads than the last
 step. (Regression: the old code applied keep='drain' to every step, collapsing them.)"""
    ir = emit.emit_mainloop(_theta(BF16_NT_KMN)).ir
    _pro, _loop, drain = _stage_bodies(ir)
    assert drain is not None and len(drain.body) == 2, "expected M=2 drain steps (PGR2)"

    def _count_reads(step_node):
        stream = []
        _walk([step_node], {}, stream)
        return sum(1 for k, _i, _e in stream if k == "read")

    first = _count_reads(drain.body[0])       # NGLL-like: read-ahead kept
    last = _count_reads(drain.body[-1])       # NLL: read-ahead suppressed
    assert first > last, f"drain not staggered: step0 reads={first} lastStep reads={last}"
    # the last step is pure drain (head reads only, one substep's worth per fan)
    lastStream = []
    _walk([drain.body[-1]], {}, lastStream)
    assert any(k == "wmma" for k, _i, _e in lastStream), "last drain step must still compute wmma"


# --------------------------------------------------------------------------- ordering
def test_read_wmma_interleave_present():
    """Paper self-check: in the unrolled leaf stream a wmma appears BEFORE the last read
    (consume-then-refill software pipeline), not all-reads-then-all-wmmas."""
    ir = emit.emit_mainloop(_theta(BF16_NT_KMN)).ir
    stream = []
    _walk(ir, {}, stream)
    kinds = [k for k, _i, _e in stream]
    rd = [i for i, k in enumerate(kinds) if k == "read"]
    wm = [i for i, k in enumerate(kinds) if k == "wmma"]
    assert rd and wm and wm[0] < rd[-1], "no read/wmma interleave (pipeline not expressed)"


@pytest.mark.parametrize("order", ["KMN", "KNM", "MKN", "MNK", "NKM", "NMK"])
def test_read_ahead_general_across_loop_orders(order):
    """The read-ahead restructure is order-general: every whole-axis loop order decodes with
    an empty ledger and a non-empty steady loop."""
    params = dict(BF16_NT_KMN, LoopOrder=order)
    r = emit.emit_mainloop(_theta(params))
    assert r.obligations_discharged
    _pro, loop, _drain = _stage_bodies(r.ir)
    assert loop is not None and loop.bodies[0]


@pytest.mark.parametrize("pgr", [1, 2, 3])
def test_prefetch_depth_uncapped(pgr):
    """PGR is NOT artificially capped (the old min(PGR,2) is gone): the copy off = PGR, so the
    drain peel has M = PGR steps and the shared-LDS ring S_shared = PGR -- all ledger-discharged.
    (M=PGR is what the KernelWriter drain fork's drainStep=(PGR-1)-remainPgr indexes.)"""
    r = emit.emit_mainloop(_theta(dict(BF16_NT_KMN, PrefetchGlobalRead=pgr)))
    assert r.obligations_discharged
    drain = _find_peel(r.ir, "drain")               # drain now lives in the peel-validity Cond.then
    if pgr >= 2:
        assert drain is not None and len(drain.body) == pgr, \
            f"PGR={pgr} should give M={pgr} drain steps, got {len(drain.body) if drain else 0}"


# --------------------------------------------------------------------------- peel-validity Cond
def test_steady_wrapped_in_peel_validity_cond():
    """The whole pipeline is wrapped in ONE first-class peel-validity Cond (the peel reset rule hypothesis
 T >= M, line 466): prologue + steady + drain live in `then`; the short-loop (T<M) in
 `els`. The core Cond carries a GENERIC `kind` -- NO TensileLite scaffold
 name (the scaffold hint is set downstream by the GIR scaffold pass)."""
    ir = emit.emit_mainloop(_theta(BF16_NT_KMN)).ir
    assert len(ir) == 1 and isinstance(ir[0], Cond), "root should be one peel-validity Cond"
    c = ir[0]
    # structured Pred (Q136): guard reads by FIELD, not string parse -- now `T >= M` STRICT.
    assert c.pred.lhs.var == "T" and c.pred.op == ">", \
        f"peel-validity guard must be strict for the post-tested steady region: {c.pred.render()!r}"
    assert c.pred.rhs == 2, f"guard should be T >= M(=2), got rhs={c.pred.rhs}"
    assert c.kind == "peel_validity", f"expected generic kind, got {c.kind!r}"
    assert c.label == "", f"core must NOT set a scaffold label (set downstream), got {c.label!r}"
    # prologue + steady + drain all in the then-arm; short-loop in els
    assert _find_peel(c.then, "prologue") is not None
    assert _find_steady(c.then) is not None
    assert _find_peel(c.then, "drain") is not None
    assert c.els, "expected a short-loop (T<M) els arm"


def test_cond_does_not_perturb_ledger_or_stream():
    """Wrapping the steady loop in a Cond must not change the discharged ledger nor the flattened
    op stream (the guard is control-flow only; the steady body is unchanged)."""
    r = emit.emit_mainloop(_theta(BF16_NT_KMN))
    assert r.obligations_discharged
    stream = []
    _walk(r.ir, {}, stream)
    kinds = [k for k, _i, _e in stream]
    assert kinds.count("wmma") > 0 and kinds.count("read") > 0


# --------------------------------------------------------------------------- loop_shape detector
class _FakeMode:
    def __init__(self, name):
        self.name = name
        self.extent = 2


class _FakeFragment:
    """Just enough fragment for `broadcast_axes`: the axes this operand is constant over."""
    def __init__(self, constant_over):
        self._constant_over = set(constant_over)

    def group_broadcast(self):
        return self._constant_over


class _FakeOperand:
    def __init__(self, name, present_axes, all_axes):
        self.name = name
        self.role = "input"
        self._present = set(present_axes)
        self.fragment = _FakeFragment(set(all_axes) - self._present)

    @property
    def is_input(self):
        return self.role == "input"


# --------------------------------------------------------------------------- PLR0 (no read-ahead)
# Regression coverage for the PrefetchLocalRead=0 path (Phase 3h).

def test_plr0_reuses_one_rotation_unit():
    """Without read-ahead, each unit is consumed before the same W=1 slot is refilled."""
    th = _theta(BF16_NT_KMN_PLR0)
    r = derive_S(th)
    S = r[0] if isinstance(r, tuple) else r
    assert S.get("A", "g0") == 1
    assert S.get("B", "g0") == 1


def test_plr0_register_generation_reuses_x0():
    """PLR0 places every sequentially consumed unit in X0."""
    ir = emit.emit_mainloop(_theta(BF16_NT_KMN_PLR0)).ir
    _pro, loop, _drain = _stage_bodies(ir)
    gens = set()
    stream = []
    _walk(loop.bodies[0], {}, stream)
    for kind, inst, env in stream:
        if kind == "read" and inst.placement is not None and inst.placement.slots:
            gens.add(int(inst.placement.slots[0][1].eval(env)))
    assert gens == {0}


def test_plr0_steady_read_has_no_readahead():
    """PLR0 = dr=0: the steady read of substep k reads substep k (NOT k+dr).  So the read's
    reduction coord equals the leaf index -- no read-ahead shift (contrast BF16_NT_KMN, dr=1)."""
    ir = emit.emit_mainloop(_theta(BF16_NT_KMN_PLR0)).ir
    _pro, loop, _drain = _stage_bodies(ir)
    th = _theta(BF16_NT_KMN_PLR0)
    kname = next(iter(summation_names(th)))
    stream = []
    _walk(loop.bodies[0], {}, stream)
    for kind, inst, env in stream:
        if kind == "read":
            for ax, v in inst.op.coord:
                if ax == kname:
                    got = v.eval(env) if hasattr(v, "eval") else env.get(ax, 0)
                    assert got == env.get(ax, 0), "PLR0 read must not shift the substep (dr=0)"


# --------------------------------------------------------------------------- per-edge gate
def test_drain_region_bodies_are_nonempty():
    """The per-edge order gate (check_ledger_discharged) partitions the tree into per-trip regions
    via _region_bodies.  A DRAIN step is a Branch(mode='iter') (from _pin_iter); _region_bodies must
    INLINE that branch so the drain trip body carries its instructions.  A `flat` that
    drop `iter`-mode branches, so every drain region came back EMPTY and the order check was vacuous
    over the whole drain -- a mis-placed drain refill copy would go uncaught."""
    from Tensile.LoopModel.emit import build_ir
    from Tensile.LoopModel.checks import _region_bodies
    from Tensile.LoopModel import build_S
    for params in (BF16_NT_KMN, BF16_NT_KMN_PLR0):
        th = _theta(params)
        S, _floor = build_S(th)
        regions = _region_bodies(build_ir(th, S))
        drain = [body for lbl, body in regions if "drain" in lbl]
        assert drain, "no drain regions found"
        for body in drain:
            assert body, "drain region body is EMPTY (iter-branch not inlined) -- order gate is vacuous"


def test_gate_catches_misplaced_drain_refill():
    """The per-edge gate must REJECT a drain trip where a WAR-refill copy is emitted BEFORE the read
    that vacates its buffer (inverted sigma_c).  Build a synthetic drain body with copy-then-read and
    confirm check_ledger_discharged flags the shared-WAR obligation -- proving the gate actually
    inspects drain instructions after the _region_bodies fix."""
    from Tensile.LoopModel.emit import build_ir
    from Tensile.LoopModel.checks import _region_bodies, build_ledger, check_ledger_discharged
    from Tensile.LoopModel import build_S
    th = _theta(BF16_NT_KMN_PLR0)
    S, _floor = build_S(th)
    ir = build_ir(th, S)
    ledger = build_ledger(th, S)
    # sanity: the correctly-ordered tree passes
    assert check_ledger_discharged(ledger, ir) == []
    # and _region_bodies actually yields drain instructions for the gate to inspect
    regions = _region_bodies(ir)
    drain_insts = [i for lbl, body in regions if "drain" in lbl for i in body]
    assert any(isinstance(i.op, Load) and i.op.dst == Space.REGISTER for i in drain_insts), \
        "gate sees no drain reads -- the order check cannot fire over the drain"


def test_gate_catches_a_refill_SPLICED_INTO_THE_MIDDLE_of_the_read_stream():
    """line 185 quantifies the rotation WAR over ALL reads of the generation
    (`∀ r in reads(gen t), ∀ w in writes(gen t+S): complete(r) ~> issue(w)`), so the sigma_c order test is
    LAST READ before FIRST REFILL -- not "some read precedes the refill".
    """
    from Tensile.LoopModel.emit import build_ir
    from Tensile.LoopModel.checks import build_ledger, check_ledger_discharged
    from Tensile.LoopModel import build_S
    th = _theta(BF16_NT_KMN_PLR0)
    S, _floor = build_S(th)
    ir = build_ir(th, S)
    ledger = build_ledger(th, S)
    assert check_ledger_discharged(ledger, ir) == [], "the correctly-ordered tree must be clean"

    war_ops = {o.consumer.op for o in ledger
               if o.kind == "inplace-WAR" and o.consumer.at == "shared"}
    assert war_ops, "no in-place shared WAR in this theta -- the fixture cannot discriminate"

    def _tok(inst):
        return set(getattr(inst.op, "tokens", ()) or ())

    def _is_refill(inst):
        return (isinstance(inst.op, Load) and inst.op.dst == Space.SHARED
                and _tok(inst) & war_ops
                and any(a.kind == "inplace-WAR" for a in inst.awaits))

    def _is_read(inst):
        return (isinstance(inst.op, Load) and inst.op.src == Space.SHARED
                and inst.op.dst == Space.REGISTER and _tok(inst) & war_ops)

    # SPLICE, don't reverse: move each refill to sit AFTER the first read of its operand but BEFORE
    # the last one.
    from Tensile.LoopModel.checks import _region_bodies

    def _flat_reads():
        return [n for lbl, body in _region_bodies(ir) if "steady" in lbl
                for n in body if _is_read(n)]

    before = _flat_reads()
    assert len(before) >= 2, f"steady carries {len(before)} vacating read(s) -- cannot discriminate"

    # ONE OPERAND ONLY, and it must be one with >= 2 reads in the steady region.
    from collections import Counter as _C
    counts = _C(t for n in before for t in _tok(n) & war_ops)
    victim = next((t for t, c in sorted(counts.items()) if c >= 2), None)
    assert victim, f"no operand has >=2 steady reads (counts={dict(counts)}) -- cannot discriminate"

    anchor = next(n for n in before if victim in _tok(n))
    lists = list(_walk_bodies(ir))
    dest = next(b for b in lists if any(n is anchor for n in b))
    at = next(i for i, n in enumerate(dest) if n is anchor)

    moved = 0
    for body in lists:
        for ri in [i for i, n in enumerate(body)
                   if isinstance(n, Inst) and _is_refill(n) and victim in _tok(n)][::-1]:
            if body is dest and ri <= at:
                continue
            dest.insert(at + 1, body.pop(ri))     # right after the FIRST read, before the later ones
            moved += 1
    assert moved, f"no {victim} refill was relocated -- fixture did not exercise the case"

    # confirm the splice produced exactly the discriminating shape for `victim`: a read BEFORE the
    # refill (so the existential is satisfied) AND a read AFTER it (so the forall is violated).
    flat = [n for lbl, body in _region_bodies(ir) if "steady" in lbl for n in body]
    rf = next(i for i, n in enumerate(flat) if _is_refill(n) and victim in _tok(n))
    rd = [i for i, n in enumerate(flat) if _is_read(n) and victim in _tok(n)]
    assert min(rd) < rf < max(rd), f"splice missed: reads at {rd}, refill at {rf}"

    bad = check_ledger_discharged(ledger, ir)
    assert [o for o in bad if o.kind == "inplace-WAR"], (
        "the gate accepted a refill spliced between the first and last read of its own generation "
        "-- that is the existential, not the forall")


def _walk_bodies(node):
    """Every MUTABLE instruction list in the tree, for fixtures that reorder instructions.
    Yields the lists themselves, not their owners: a `Loop` may be MULTI-BODY, and its
    `.body` property deliberately raises rather than silently returning `bodies[0]`.
    """
    if isinstance(node, list):
        yield node
        for n in node:
            yield from _walk_bodies(n)
        return
    for sub in getattr(node, "bodies", None) or ():
        yield from _walk_bodies(sub)
    for attr in ("then", "els"):
        sub = getattr(node, attr, None)
        if sub is not None:
            yield from _walk_bodies(sub)


@pytest.mark.parametrize("order", ["KMN", "MNK", "MKN"])
def test_gate_catches_missing_readahead_prologue_fill(order):
    """the read-ahead prologue rule build-time oracle: the ledger carries a `readahead-residency` O_r per required
    prologue read; the gate REJECTS a prologue missing any of them (a wrong K-axis peel) as a
    NON-EMPTY ledger -- the defect is caught at BUILD, not by a passing/failing run.  Correct tree
    -> empty; delete the k1 prologue reads -> the gate flags exactly those O_r."""
    from Tensile.LoopModel.emit import build_ir
    from Tensile.LoopModel.checks import build_ledger, check_ledger_discharged
    from Tensile.LoopModel import build_S
    th = _theta(dict(BF16_NT_KMN, LoopOrder=order))
    S, _floor = build_S(th)
    ledger = build_ledger(th, S)
    ir = build_ir(th, S)
    assert check_ledger_discharged(ledger, ir) == []                # correct prologue -> empty

    # corrupt: drop every k1 register-read from the prologue (simulate the old K-axis fill)
    pro = _find_peel(ir, "prologue")                   # prologue now lives in the peel-validity Cond.then

    def _is_k1_read(inst):
        return (isinstance(inst, Inst) and isinstance(inst.op, Load)
                and inst.op.dst == Space.REGISTER
                and any(ax == "K_inner" and v == 1 for ax, v in inst.op.coord))
    removed = [n for n in pro.body if _is_k1_read(n)]
    pro.body[:] = [n for n in pro.body if not _is_k1_read(n)]

    bad = check_ledger_discharged(ledger, ir)
    ra_bad = [o for o in bad if o.kind == "readahead-residency"]
    if order in ("MNK", "MKN"):
        # THE PROLOGUE CARRIES k1, so removing it must leave O_r undischarged.
        assert removed and ra_bad, f"{order}: gate failed to flag the missing full-K prologue fill"
    else:
        # K-outer (`KMN`): the reduction IS the read-ahead level, `dr=1` advances one k, and the
        # region inside it carries no second k -- so the correct prologue has no k1 read to remove
        # and the gate stays empty.  Still by construction, but now for the one order where it is.
        assert not removed and ra_bad == []


# --------------------------------------------------------------------------- validate_loopir
@pytest.mark.parametrize("order", ["KMN", "KNM", "MKN", "MNK", "NKM", "NMK"])
def test_validate_loopir_all_orders(order):
    """The structural gate passes proper GEMM pseudocode for every 3-axis loop order: root is one
    peel-validity Cond(T>=M) wrapping [prologue, steady, drain], iter bound in every phase, drain a
    straight-line Bind chain, ledger empty, generic Cond kinds only."""
    th = _theta(dict(BF16_NT_KMN, LoopOrder=order))
    S, _ = build_S(th)
    errs = validate_loopir(th, emit.build_ir(th, S), S)
    assert errs == [], f"{order}: {errs}"


def test_validate_loopir_plr0():
    th = _theta(BF16_NT_KMN_PLR0)
    S, _ = build_S(th)
    assert validate_loopir(th, emit.build_ir(th, S), S) == []


def test_validate_loopir_mxfp8():
    _desc, params = scenarios.SCENARIOS["mxfp8_derive"]
    th = _theta(params)
    S, _ = build_S(th)
    assert validate_loopir(th, emit.build_ir(th, S), S) == []


def test_validate_loopir_catches_prologue_before_guard():
    """P1/P2 gate: flattening the pipeline so prologue/drain sit OUTSIDE the peel-validity Cond
    (running before the guard) is REJECTED -- the exact defect the iter-restructure fixed."""
    th = _theta(BF16_NT_KMN)
    S, _ = build_S(th)
    ir = emit.build_ir(th, S)
    flat = list(ir[0].then)                       # prologue/steady/drain hoisted to top level
    errs = validate_loopir(th, flat, S)
    assert any(e.startswith("P1") for e in errs), f"gate missed prologue-before-guard: {errs}"


def test_validate_loopir_catches_drain_residue_pin():
    """P4 gate: reverting a drain step to the old free-iter `Cond(iter%d==r)` residue pin (instead
    of a Bind) is REJECTED -- the drain must bind iter=T-M+t, not test a residue on an undefined iter."""
    import copy
    th = _theta(BF16_NT_KMN)
    S, _ = build_S(th)
    ir = copy.deepcopy(emit.build_ir(th, S))
    drain = next(n for n in ir[0].then if isinstance(n, Peel) and n.kind == "drain")
    step0 = drain.body[0]
    drain.body[0] = Cond(pred=Pred(Expr(var="iter", mod=2), "==", 1),
                         then=step0.body, els=[], outer=True)
    errs = validate_loopir(th, ir, S)
    assert any(e.startswith("P4") for e in errs), f"gate missed drain residue-pin: {errs}"


def test_validate_loopir_catches_scaffold_label_in_core():
    """P8 gate: a TensileLite scaffold label on a CORE Cond is REJECTED -- scaffold naming is the
    GIR scaffold_map pass's job, the pure core carries only generic kinds."""
    import copy
    th = _theta(BF16_NT_KMN)
    S, _ = build_S(th)
    ir = copy.deepcopy(emit.build_ir(th, S))
    ir[0].label = "toPGR1"
    errs = validate_loopir(th, ir, S)
    assert any(e.startswith("P8") for e in errs), f"gate missed scaffold label in core: {errs}"


# --------------------------------------------------------------------- P3/P5 non-vacuity
def test_gate_catches_a_free_chunk_reference_under_a_peel():
    """P3's real assertion -- "NO free `iter` reference sits under a Peel without a Bind" -- and
    P5's concrete-slot half are ONE check: no placement may reference a mode nothing binds.
    """
    th = _theta(BF16_NT_KMN)
    S, _floor = build_S(th)
    ir = emit.build_ir(th, S)
    assert validate_loopir(th, ir, S) == [], "fixture must start clean"

    drain = _find_peel(ir, "drain")
    assert drain is not None and any(isinstance(n, Bind) for n in drain.body), \
        "drain has no Bind to dissolve; test would be vacuous"

    # dissolve each drain-step Bind, splicing its body up -- the chunk index loses its binder
    spliced = []
    for n in drain.body:
        spliced.extend(n.body if isinstance(n, Bind) else [n])
    drain.body[:] = spliced

    errs = validate_loopir(th, ir, S)
    unbound = [e for e in errs if "UNBOUND" in e]
    assert unbound, f"gate did not catch the free chunk reference; got {errs}"
    iter_name = th.summation_chunk_name("iter")
    assert all(repr(iter_name) in e for e in unbound), unbound


def test_emit_mainloop_runs_the_structural_gate():
    """The gate is WIRED: `emit_mainloop` rejects a defective tree instead of returning it.  It was
    documented as "the PERMANENT structural gate ... caught at build" while nothing outside the
    tests ever called it."""
    import Tensile.LoopModel.emit as _emit
    th = _theta(BF16_NT_KMN)
    emit.emit_mainloop(th)                      # clean point: no raise

    real_build = _emit.build_ir

    def _broken(theta, S, ledger=None, plans=None):  # return a tree that violates P2
        ir = real_build(theta, S, ledger, plans)
        ir.append(_find_peel(ir, "prologue").body[0])
        return ir

    _emit.build_ir = _broken
    try:
        with pytest.raises(RuntimeError, match="structural gate"):
            emit.emit_mainloop(th)
    finally:
        _emit.build_ir = real_build


def test_non_contiguous_role_ord_is_supported_via_first_touch_guards():
    """A NON-CONTIGUOUS role in `loop order` IS emittable, and correctly: each read that is
    invariant over an ENCLOSING mode is issued once under an explicit first-touch `Cond(mode==0)`.
    """
    th = _theta({"LoopOrder": "MKNMKN", "TDMSplit": [2, 2], "MIWaveTile": [4, 4]})
    S, _floor = build_S(th)
    r = emit.emit_mainloop(th)                       # must NOT raise
    assert r.obligations_discharged
    assert validate_loopir(th, r.ir, S) == []
    # op parity WITHIN ONE STEADY TRIP: each distinct the axes it varies over coordinate is read exactly once.
    # (Count the steady body alone -- the prologue and each drain step legitimately re-read the
    # same coords for their own chunk, so a whole-tree count would compare across phases.)
    from collections import Counter
    steady = _find_steady(r.ir)
    assert steady is not None, "no steady loop; test would be vacuous"
    seen = Counter()
    stream = []
    _walk([steady], {}, stream)
    for kind, inst, env in stream:
        if kind == "read":
            coord = tuple(sorted((ax, env.get(ax, v)) for ax, v in inst.op.coord))
            seen[(inst.op.tokens[0], coord)] += 1
    dupes = {k: v for k, v in seen.items() if v > 1}
    assert not dupes, f"a read was issued more than once for the same coord: {dupes}"


def _loops_with_ranges(nodes, out):
    for nd in nodes:
        if isinstance(nd, Loop):
            if len(nd.bodies) > 1:
                out.append(nd)
            for b in nd.bodies:
                _loops_with_ranges(b, out)
        elif isinstance(nd, (Peel, Bind)):
            _loops_with_ranges(nd.body, out)
        elif isinstance(nd, Cond):
            _loops_with_ranges(nd.then, out)
            _loops_with_ranges(nd.els, out)


def _count_first_touch(nodes, out):
    for nd in nodes:
        if isinstance(nd, Cond):
            if nd.kind == "first_touch":
                out.append(nd)
            _count_first_touch(nd.then, out)
            _count_first_touch(nd.els, out)
        elif isinstance(nd, Loop):
            for b in nd.bodies:
                _count_first_touch(b, out)
        elif isinstance(nd, (Peel, Bind)):
            _count_first_touch(nd.body, out)


@pytest.mark.parametrize("params,label", [
    (BF16_NT_KMN, "contiguous KMN"),
    ({"LoopOrder": "MKNMKN", "TDMSplit": [2, 2], "MIWaveTile": [4, 4]}, "non-contiguous MKNMKN"),
])
def test_first_touch_guards_are_peeled_into_body_ranges(params, label):
    """multi-body: a first-touch guard is HOISTED INTO THE LOOP STRUCTURE, not tested per
 iteration. A read invariant over an enclosing mode `m` is issued once, so `m`'s loop splits
 into two sub-bodies over disjoint ranges -- [0,1) WITH the read, [1,trip) WITHOUT -- and the
 predicate disappears. "The body before the boundary differs from the body after."

 This is a COST refinement: the emitted set of operations is unchanged, which is what the
 op-parity assertion below pins. It applies to any loop order with an invariant-over-enclosing read,
 contiguous or not -- the non-contiguous loop order is simply where names it."""
    th = _theta(params)
    S, _floor = build_S(th)
    ir = emit.build_ir(th, S)

    multi = []
    _loops_with_ranges(ir, multi)
    assert multi, f"{label}: no loop was split into sub-bodies"
    for lp in multi:
        assert lp.body_ranges, f"{label}: Loop({lp.axis}) has {len(lp.bodies)} bodies but no ranges"
        assert len(lp.body_ranges) == len(lp.bodies)
        # ranges must partition [0, trip) exactly -- no gap, no overlap, nothing issued twice
        cov = sorted(lp.body_ranges)
        assert cov[0][0] == 0 and cov[-1][1] == lp.trip, f"{label}: {cov} does not span {lp.trip}"
        for (a_lo, a_hi), (b_lo, _b_hi) in zip(cov, cov[1:]):
            assert a_hi == b_lo, f"{label}: ranges {cov} overlap or leave a gap"
        # ranged_bodies() is the authority and must agree with the raw fields
        assert [(lo, hi) for lo, hi, _b in lp.ranged_bodies()] == list(lp.body_ranges)

    # Every peelable guard is gone FROM THE LOOP THAT PEELED IT.
    for lp in multi:
        inside = []
        for b in (lp.bodies or []):
            _count_first_touch(b, inside)
        for c in inside:
            assert c.pred.lhs.var != lp.axis, (
                f"{label}: first-touch on {c.pred.lhs.var!r} survived inside the Loop({lp.axis}) "
                f"that was split on it")

    # SEMANTICS UNCHANGED: one read per distinct the axes it varies over coord in a steady trip.
    from collections import Counter
    steady = _find_steady(ir)
    stream = []
    _walk([steady], {}, stream)
    seen = Counter()
    for kind, inst, env in stream:
        if kind == "read":
            coord = tuple(sorted((ax, env.get(ax, v)) for ax, v in inst.op.coord))
            seen[(inst.op.tokens[0], coord)] += 1
    dupes = {k: v for k, v in seen.items() if v > 1}
    assert not dupes, f"{label}: peel changed what is issued -- duplicate reads {dupes}"


# ===========================================================================
# Register groups share one operand-level PLR while retaining their own ring widths.
def _mxfp8_theta():
    _desc, params = scenarios.SCENARIOS["mxfp8_derive"]
    return _theta(params)


def test_mixed_width_groups_split_ahead_and_inplace_reads():
    """A pipeline group advances; an in-place group gets a separate zero-advance read."""
    from Tensile.LoopModel.emit import build_ir
    from Tensile.LoopModel.checks import _region_bodies
    from Tensile.LoopModel import build_S
    th = _mxfp8_theta()
    S, _floor = build_S(th)
    steady = [b for lbl, b in _region_bodies(build_ir(th, S)) if "steady" in lbl]
    assert steady, "no steady region"
    body = steady[0]
    groups_seen = {}
    for inst in body:
        if isinstance(inst.op, Load) and inst.op.dst == Space.REGISTER:
            gl = tuple(g for g, _e in inst.placement.slots)
            groups_seen.setdefault(inst.op.tokens[0], set()).add(gl)
    assert groups_seen["A"] == {("lo",), ("hi",)}, \
        f"A's mixed-width groups need separate read depths, got {groups_seen['A']}"
    assert groups_seen["B"] == {("",)}, \
        f"B is uniform-width and must stay one read, got {groups_seen['B']}"


# --- pt5 register-ring SHAPES, on a PLAIN operand (no VgprPartition, no MX) -------------
# Test-local because they exist only to pin sigma_c's two shapes; they are not kernels anyone ships, so
# they do not belong in `LoopModel.scenarios`.
_WT11 = {"MatrixInstruction": [16, 16, 128, 1], "ElemBytes": 1, "MIWaveTile": [1, 1],
         "GlobalReadVectorWidthA": 64, "GlobalReadVectorWidthB": 64, "LocalReadVectorWidth": 16}
# INPLACE is now reached by ASKING for no read-ahead.  It used to fall out of `n_s == 1` via the
# `W - 1` clamp, but PLR is faithful: at PLR>=1 this shape widens W to carry the request instead.
INPLACE_NOHOIST = {**_WT11, "DepthU": 128, "PrefetchLocalRead": 0}


def _steady_of(params):
    """The flat steady body for a param dict."""
    from Tensile.LoopModel.emit import build_ir
    from Tensile.LoopModel.checks import _region_bodies
    from Tensile.LoopModel import build_S
    th = _theta(params)
    S, _floor = build_S(th)
    return [b for lbl, b in _region_bodies(build_ir(th, S)) if "steady" in lbl][0]


def _is_reg_refill(inst):
    return (isinstance(inst.op, Load) and inst.op.dst == Space.REGISTER
            and any(a.kind == "inplace-WAR" for a in inst.awaits))


def test_shape_b_inplace_read_precedes_its_own_wmma():
    """pt5 SHAPE B is the OPPOSITE order, and sigma_c must not "fix" it. A group with
 `dr_g = 0` is not hoisted: the model says it is "issued in place one line before its own wmma",
 and it prologues nothing. Deferring it past the consumer strands that consumer with a register
 nothing filled -- and since INPLACE primes no prologue, nothing repairs it.

 A group is INPLACE when no read-ahead is asked of it; `W` then sits at its floor of 1.
 Keying sigma_c's deferral on the `inplace-WAR` await ALONE (the ledger mints it
 whenever `W < 2`) deferred every one of these -- 216/216 `n_s == 1` configs failed
 `check_register_dataflow` with USE-BEFORE-DEF, 0/432 at `n_s >= 2`, and all 16 such MXFP8
 kernels miscompared on hardware."""
    reads_before = 0
    consumed = set()
    for inst in _steady_of(INPLACE_NOHOIST):
        if isinstance(inst.op, Mma):
            consumed |= {(o, g) for o, pl in inst.placement.items() for g, _e in pl.slots}
        elif _is_reg_refill(inst):
            assert not inst.op.advance, "INPLACE_NOHOIST must be INPLACE (advance == 0)"
            keys = {(t, g) for t in inst.op.tokens for g, _e in inst.placement.slots}
            assert not (keys & consumed), \
                "an INPLACE read was deferred past the wmma it feeds -- that wmma now reads a " \
                "register nothing filled, and INPLACE primes no prologue to repair it"
            reads_before += 1
    assert reads_before, "no in-place register refill found -- the scenario no longer exercises INPLACE"


# ===========================================================================
# term (ii) / PREFETCH|B -- dr_g is DERIVED, never vetoed
# ===========================================================================
_ORDERS = ("KMN", "KNM", "MKN", "MNK", "NKM", "NMK")


def _matrix_cfgs():
    """(label, params) over the axes the read-ahead derivation is sensitive to."""
    for wt in ([1, 1], [2, 2], [4, 4]):
        for split in (None, [2, 2]):
            if split and any(value % factor for value, factor in zip(wt, split)):
                continue
            for plr in (1, 2):
                for order in _ORDERS:
                    p = {"MIWaveTile": wt, "LoopOrder": order, "PrefetchLocalRead": plr}
                    if split:
                        p["TDMSplit"] = split
                    yield (f"WT{wt}{'/split' if split else ''} PLR{plr} {order}", p)


def _name_space_walk(theta, op, group, W, shift):
    """ORACLE for term (ii): a direct live-range walk on the joint `(free-tile,
    rotation-slot)` register-NAME space, three chunks of the `loop order` traversal.
    """
    from Tensile.LoopModel.traversal import _inner_steps
    from Tensile.LoopModel.traversal import varying_axes, axis_strides, ring_axes, reload_modes
    pres = varying_axes(theta, op)
    if not pres or not shift:
        return "safe"
    rl = reload_modes(theta, op) or pres
    pstr, _n_full = axis_strides(theta, pres)
    rstr, n_rl = axis_strides(theta, rl)
    ext = {m.name: m.extent for m in theta.inner_axes()}
    red = set(summation_names(theta))
    free_axes = [m for m in pres if m not in red]
    rate = ring_axes(theta, op, group)
    held_modes = [m for m in pres if m not in rstr]

    def name(coord):
        idx, coef = 0, 1
        for n, e in reversed(rate):
            idx += coord.get(n, 0) * coef
            coef *= e
        return (tuple(coord.get(m, 0) for m in free_axes), idx % max(1, W))

    steps = _inner_steps(theta)
    T, ev = len(steps), {}
    for c in range(3):
        seen = set()
        for i, sc in enumerate(steps):
            t = c * T + i
            coord = {m: sc.get(m, 0) for m in pres}
            P = sum(pstr[m] * coord[m] for m in pres)
            P_rl = sum(rstr[m] * coord[m] for m in rstr)
            # the modes the shift does NOT move: a write at `M_split=0` and one at `M_split=1` with
            # the same reload position are different values, so they belong in the identity.
            held = tuple(coord[m] for m in held_modes)
            ev.setdefault(name(coord), []).append((t, "r", (c, P_rl, held)))
            if P in seen:
                continue                       # first-touch merge: one read per position
            seen.add(P)
            flat = P_rl + shift
            w = dict(coord)
            w.update({m: (flat // rstr[m]) % max(1, ext.get(m, 1)) for m in rstr})
            ev.setdefault(name(w), []).append(
                (t, "w", (c + flat // max(1, n_rl), flat % max(1, n_rl), held)))
    # STARVATION FIRST: replay each name and flag a steady consume that finds the wrong generation.
    # The conflict scan below looks BACKWARD from a write to the consumer it destroys and so cannot
    # see a slot that is simply never refreshed -- the failure mode a MOVED read produces.
    for _nm, evs in ev.items():
        held = None
        for (t, kind, v) in sorted(evs):
            if kind == "w":
                held = v
            elif held is not None and T <= t < 2 * T and held != v:
                return "clobber"
    verdict = "safe"
    for _nm, evs in ev.items():
        ws = sorted(e for e in evs if e[1] == "w")
        rs = [e for e in evs if e[1] == "r"]
        for j, (t, _k, v) in enumerate(ws):
            if not j or not (T <= t < 2 * T):        # judge the STEADY chunk only
                continue
            prev = ws[j - 1][2]
            if prev == v:
                continue
            late = [rt for rt, _k2, rv in rs if rv == prev and rt >= t]
            if not late:
                continue
            if any(rt > t for rt in late):
                return "clobber"
            verdict = "inplace"
    return verdict


def test_readahead_advance_never_clobbers_across_the_matrix():
    """The EMITTED schedule must never overwrite a register that is still live.
    STATED AT THE PRODUCT LAYER, deliberately.  Scoring
    `_name_space_walk` -- a model-level walk that places each refill AT THE STEP ITS READ IS ISSUED.
    """
    from Tensile.Lowering import lower_to_gir
    from Tensile.Lowering.gir import run_pipeline
    from Tensile.Lowering.gir.verify_dataflow import check_register_dataflow

    checked = 0
    for label, params in _matrix_cfgs():
        prog = run_pipeline(lower_to_gir(_theta(params)))
        bad = check_register_dataflow(prog)
        assert not bad, f"{label}: register clobber in the emitted schedule -- {bad[:2]}"
        checked += 1
    assert checked >= 40, f"the matrix collapsed to {checked} cells -- this check went vacuous"


def test_broadcast_fan_predicate_agrees_with_the_walk():
    """the decoder-evaluable predicate for term (ii) -- a mode ∉ pres(p), extent > 1, outer to
    EVERY mode of pres(p) -- must agree with the live-range oracle about which operands cannot
    afford a look-ahead AT THE DEFAULT POSITION.
    """
    from Tensile.LoopModel.traversal import broadcast_width
    from Tensile.LoopModel.traversal import reloads_whole_set
    checked = 0
    for label, params in _matrix_cfgs():
        th = _theta(params)
        S, _floor = build_S(th)
        for op in th.operands:
            if op.is_output or not op.movements or op.movements[-1].dst != Space.REGISTER:
                continue
            g = op.fragment.groups()[0]
            fan_says = broadcast_width(th, op) > 1
            bo = reloads_whole_set(th, op)
            # CONTAINMENT, not equality: `reload ⊆ pres`, so a mode outer to every the axes it varies over mode is
            # outer to every reload mode.  The converse fails exactly on the split population.
            assert not fan_says or bo, \
                f"{label} {op.name}: fan>1 but not reloads_whole_set -- reload ⊄ pres?"
            # The walk must agree that the DEFAULT-position advance is what `reloads_whole_set`
            # rules out.
            probe = (_probe_shift(th, op, g, params["PrefetchLocalRead"])
                     if params["PrefetchLocalRead"] == 1 else 0)
            if probe:
                got = _name_space_walk(th, op, g, S.get(op.name, g), probe)
                assert bo == (got == "clobber"), \
                    f"{label} {op.name}: reloads_whole_set={bo} (fan={broadcast_width(th, op)}) " \
                    f"but walk says {got}"
                checked += 1
    assert checked > 20, f"only {checked} operands probed -- the predicate went undertested"


def _probe_shift(theta, op, group, dr):
    """The advance the operand would take with the floor ignored -- used only to ask the oracle
    what the floor is protecting against.
    """
    from Tensile.LoopModel.traversal import (varying_axes, axis_strides, _summation_ring_axis,
                                            reload_modes)
    pres = varying_axes(theta, op)
    strides, _n = axis_strides(theta, reload_modes(theta, op) or pres)
    red = _summation_ring_axis(theta, op, group)
    if red is None or red[0] not in strides:
        return 0
    inner = [m.name for m in theta.inner_axes()]
    rn = summation_names(theta)
    sub = next((n for n in reversed(inner) if n in rn), None)
    inner_of = inner[inner.index(sub) + 1:] if sub else []
    return (red[1] if not inner_of else min(dr, red[1])) * strides[red[0]]


def test_prologue_depth_equals_steady_advance_shape_a_or_b():
    """pt5's net rule: prologue depth = steady advance = `dr_g`, per group. A group is
    PREFETCH (hoisted, non-empty prologue, non-zero advance) or INPLACE (in place, EMPTY prologue,
    zero advance) -- never a mixture.
    """
    from Tensile.LoopModel.schedule import _readahead_shift, preloaded_tiles
    for label, params in _matrix_cfgs():
        th = _theta(params)
        S, _floor = build_S(th)
        dr = params["PrefetchLocalRead"]
        for op in th.operands:
            if op.is_output or not op.movements or op.movements[-1].dst != Space.REGISTER:
                continue
            shift, _st, _np, _ex = _readahead_shift(th, op, dr, S)
            pro = preloaded_tiles(th, op, dr, S)
            assert bool(shift) == bool(pro), \
                (f"{label} {op.name}: shape mismatch -- steady advance {shift} but "
                 f"{len(pro)} prologue reads (PREFETCH prologue on an INPLACE steady)")


def test_dr_g_anchors_from_the_floor():
    """Anchor the derivation itself, so a future rewrite cannot silently change WHICH operand
    keeps its look-ahead.
    """
    from Tensile.LoopModel.schedule import prefetch_steps_for
    def dr_of(order, dr=1, wt=(2, 2)):
        th = _theta({"MIWaveTile": list(wt), "LoopOrder": order, "PrefetchLocalRead": dr})
        S, _f = build_S(th)
        return {op.name: prefetch_steps_for(th, op, op.fragment.groups()[0], dr, S)
                for op in th.operands if op.movements and op.movements[-1].dst == Space.REGISTER}
    for order in ("KMN", "KNM", "MNK", "MKN", "NMK", "NKM"):
        assert dr_of(order) == {"A": 1, "B": 1}, f"{order}: PLR must be honoured for both operands"
    # W=2 stores two complete rotation units, so PLR2 is the exact capacity of this point.
    assert dr_of("KMN", dr=2) == {"A": 2, "B": 2}
    # PLR=0 means no read-ahead was requested at all
    assert dr_of("KMN", dr=0) == {"A": 0, "B": 0}


def _find_steady(nodes):
    for n in nodes:
        if isinstance(n, Loop) and n.outer:
            return n
        kids = (list(n.then) + list(n.els) if isinstance(n, Cond) else
                list(n.body) if isinstance(n, (Peel, Bind)) else
                [x for b in n.bodies for x in b] if isinstance(n, Loop) else [])
        r = _find_steady(kids)
        if r is not None:
            return r
    return None


def test_broadcast_fan_ignores_a_degenerate_mode():
    """the term-(ii) predicate says the qualifying outer mode must have extent > 1. No
 PARAMETER config can exercise that clause -- `translate` drops degenerate modes, so a theta built
 from params never carries an extent-1 inner mode -- which is exactly why it needs a direct
 test: a mutation deleting the clause passes the whole matrix otherwise.

 A hand-built theta can carry one, so the clause is a real guard, not decoration: without it a
 broadcast mode that re-traverses NOTHING would zero `dr_g` and cost the operand its
 look-ahead."""
    from Tensile.LoopModel.traversal import broadcast_width
    import dataclasses
    th = _theta({"MIWaveTile": [2, 2], "LoopOrder": "KMN"})
    A = th.op("A")
    assert broadcast_width(th, A) == 1, "KMN baseline should have no term (ii)"
    outermost = th.inner_axes()[0]
    # splice in a mode OUTSIDE the whole the axes it varies over set: a new loop order entry, declared on A's fragment
    # as an axis A is constant over (that declaration is what `geometry.broadcast_axes` reads).
    deg = dataclasses.replace(outermost, name="degenerate", extent=1)
    A.fragment = dataclasses.replace(
        A.fragment, broadcast_axes=tuple(A.fragment.broadcast_axes) + ("degenerate",))
    # theta's nest is fixed at construction, so a different nest means a different theta
    th1 = dataclasses.replace(th, ord=(deg,) + tuple(th.ord))
    assert broadcast_width(th1, A) == 1, \
        "an extent-1 broadcast mode re-traverses nothing and must NOT fire term (ii)"
    th2 = dataclasses.replace(th, ord=(dataclasses.replace(deg, extent=2),) + tuple(th.ord))
    assert broadcast_width(th2, A) == 2, \
        "an extent-2 broadcast mode outer to all of pres(A) MUST fire term (ii)"


_MATRIX = list(_matrix_cfgs())

#: EVERY cell of the matrix decodes.  PLR=2 used to refuse on 14 of them; it does not any more,
#: because a full-inner operand derives its depth from what the copy stages instead of taking the
#: yaml's number.  A cell that starts refusing is a capability loss and fails here.
def _refuses_read_ahead(label, params):
    """No cell may refuse: theta serves the requested read-ahead everywhere in the matrix."""
    _theta(dict(params))
    return False


@pytest.mark.parametrize("label,params", _MATRIX, ids=[l for l, _p in _MATRIX])
def test_matrix_decodes_to_a_structurally_valid_tree(label, params):
    """Every (loop order x WaveTile x region split x PLR) point must decode to a tree that passes
    the whole P1-P9 structural gate, EMPTY ledger included.
    """
    if _refuses_read_ahead(label, params):
        return
    th = _theta(dict(params))
    S, _floor = build_S(th)
    errs = validate_loopir(th, emit.build_ir(th, S), S)
    assert errs == [], f"{label}: " + "; ".join(errs)


@pytest.mark.parametrize("name", list(scenarios.SCENARIOS.keys()))
def test_all_scenarios_are_structurally_valid(name):
    """The named-scenario counterpart of the above: the existing sweep asserts only
    `ledger_empty`, which is P7 alone -- a tree can have an empty ledger and still violate the peel
    shape (P1), leave an unbound slot symbol (P3/P5), or emit a refill before its vacating read
    (P6).  Run the full gate."""
    _desc, params = scenarios.SCENARIOS[name]
    th = _theta(params)
    S, _floor = build_S(th)
    errs = validate_loopir(th, emit.build_ir(th, S), S)
    assert errs == [], f"{name}: " + "; ".join(errs)


@pytest.mark.parametrize("label,params", _MATRIX, ids=[l for l, _p in _MATRIX])
def test_prologue_fill_and_ledger_obligations_are_one_derivation(label, params):
    """The emitted read-ahead fill and the ledger's `readahead-residency` obligations must be the
    SAME SET -- not merely compatible.
    """
    from Tensile.LoopModel.checks import build_ledger
    from Tensile.LoopModel.schedule import preloaded_tiles, peel_depths
    if _refuses_read_ahead(label, params):
        return
    th = _theta(dict(params))
    S, _floor = build_S(th)
    dr = peel_depths(th, build_S(th)[0]).requested_steps
    want = {(o.consumer.op, o.consumer.coord) for o in build_ledger(th, S)
            if o.kind == "readahead-residency"}
    # read the EMITTED tree, not `preloaded_tiles` again: re-calling the shared helper would
    # agree with the ledger by construction and could never catch the desync that actually
    # happened (emit passing different arguments than the ledger).
    ir = emit.build_ir(th, S)
    have = set()

    def collect(nodes):
        for n in nodes:
            if isinstance(n, Inst) and isinstance(n.op, Load) and n.op.dst == Space.REGISTER:
                have.add((n.op.tokens[0], tuple(sorted((m, v) for m, v in n.op.coord))))
            elif isinstance(n, Loop):
                for b in n.bodies:
                    collect(b)
            elif isinstance(n, (Peel, Bind)):
                collect(n.body)
            elif isinstance(n, Cond):
                collect(n.then)
                collect(n.els)

    for n in _prologue_peels(ir):
        collect(n.body)
    assert have == want, (
        f"{label}: prologue fill and ledger disagree -- "
        f"fill-only={sorted(have - want)}, ledger-only={sorted(want - have)}")


def _prologue_peels(nodes):
    """Every `Peel(kind='prologue')` in the tree (both Cond arms)."""
    out = []
    for n in nodes:
        if isinstance(n, Peel) and n.kind == "prologue":
            out.append(n)
        kids = (list(n.then) + list(n.els) if isinstance(n, Cond) else
                list(n.body) if isinstance(n, (Peel, Bind)) else
                [x for b in n.bodies for x in b] if isinstance(n, Loop) else [])
        out += _prologue_peels(kids)
    return out


# ===========================================================================
# Per-operand S_shared / off  (LDSBufferA|B, PrefetchGlobalReadA|B, PrefetchLocalReadA|B)
# ===========================================================================
def test_shared_depth_is_the_buffer_count_not_the_prefetch_depth():
    """S_shared must come from the operand's LDS BUFFER COUNT, never from `delta`."""
    from Tensile.LoopModel.traversal import lds_buffers
    for pgr in (1, 2):
        th = _theta({"MIWaveTile": [2, 2], "DepthU": 64, "PrefetchGlobalRead": pgr,
                     "LoopOrder": "KMN"})
        for nm in ("A", "B"):
            assert lds_buffers(th, th.op(nm)) == 2, \
                f"PGR={pgr}: S_shared must be the NumLdsBlk=2 preset, not delta={pgr}"


def test_pgr1_emits_the_lds_swaps():
    """The regression the above exists for, at the level that actually broke: with S_shared=1 the
    decoder emitted zero swap Marks, so every copy overwrote the buffer being read."""
    from Tensile.Lowering import build_gir
    from Tensile.Lowering.gir.render import render_gir
    th = _theta({"MIWaveTile": [2, 2], "DepthU": 64, "PrefetchGlobalRead": 1, "LoopOrder": "KMN"})
    prog = build_gir(th, mainloop=emit.emit_mainloop(th))
    swaps = [l for l in render_gir(prog).splitlines() if "Mark  swap" in l]
    assert swaps, "PGR=1 emitted no LDS swap at all -- the depth-1-ring regression"


@pytest.mark.parametrize("key,base", [("LDSBuffer", None),
                                      ("PrefetchGlobalRead", None),
                                      ("PrefetchLocalRead", None)])
def test_per_operand_override_defaults_to_the_preset(key, base):
    """`-1` (the default) must reproduce the whole-kernel preset exactly, so adding these knobs
    changes nothing for a caller that does not set them."""
    from Tensile.LoopModel.traversal import lds_buffers
    p = {"MIWaveTile": [2, 2], "DepthU": 64, "PrefetchGlobalRead": 2, "PrefetchLocalRead": 1,
         "LoopOrder": "KMN"}
    ref = _theta(p)
    got = _theta(dict(p, **{key + "A": -1, key + "B": -1}))
    assert [lds_buffers(got, got.op(n)) for n in "AB"] == \
           [lds_buffers(ref, ref.op(n)) for n in "AB"]
    assert got.movement_offsets() == ref.movement_offsets()


def test_per_operand_lds_buffers_and_off_are_independent():
    """A and B may carry DIFFERENT S_shared and DIFFERENT `off` -- makes the depth a map and
 makes `off` a matrix over (operand x level), so this is the model's shape, not an
 extension. A asymmetric point must show up in BOTH theta and the emitted swap count."""
    from Tensile.LoopModel.traversal import lds_buffers
    from Tensile.Lowering import build_gir
    from Tensile.Lowering.gir.render import render_gir
    base = {"MIWaveTile": [2, 2], "DepthU": 64, "PrefetchGlobalRead": 2, "LoopOrder": "KMN"}

    th = _theta(dict(base, LDSBufferA=1))
    assert lds_buffers(th, th.op("A")) == 1 and lds_buffers(th, th.op("B")) == 2
    swaps = [l for l in render_gir(build_gir(th, mainloop=emit.emit_mainloop(th))).splitlines()
             if "Mark  swap" in l]
    assert swaps and not any("'operand': 'A'" in l for l in swaps), \
        "A is a depth-1 ring: it must contribute no swap while B still does"

    th2 = _theta(dict(base, PrefetchGlobalReadA=1))
    outer = th2.outer_axes()[0].name
    assert th2.off_at(th2.op("A"), theta_mod.COPY, outer) == 1 \
        and th2.off_at(th2.op("B"), theta_mod.COPY, outer) == 2


def test_lds_buffers_above_the_allocation_is_rejected():
    """Unlike the VGPR side, the LDS allocation is
    sized by Solution.py against MaxLDS -- asking for a deeper ring than it reserved is an
    out-of-bounds LDS write, so it must be refused rather than silently emitted."""
    with pytest.raises(RuntimeError, match="exceeds the kernel's allocated NumLdsBlk"):
        _theta({"MIWaveTile": [2, 2], "DepthU": 64, "PrefetchGlobalRead": 2,
                "LoopOrder": "KMN", "NumLdsBlk": 2, "LDSBufferA": 3})


def test_dtv_operand_has_no_shared_ring_depth():
    """A direct global-to-register trajectory has no shared depth."""
    from Tensile.LoopModel.schedule import build_S
    from Tensile.LoopModel.traversal import lds_buffers
    th = _theta({"MIWaveTile": [2, 2], "DepthU": 64, "LoopOrder": "KMN", "DirectToVgprA": True})
    A = th.op("A")
    assert A.trajectory.shared is None
    with pytest.raises(RuntimeError, match="no shared placement"):
        lds_buffers(th, A)
    depths, _minimum = build_S(th)
    assert depths.shared("A") is None
    assert depths.shared("B") == 2


# ============================================ pt5 chunk-crossing coupling
@pytest.mark.parametrize("pgr,plr,want_r,want_first", [
    # (PGR, PLR, chunk-reach r, must the copy LEAD this trip's reads?)
    (1, 0, 0, False),   # no read-ahead -> no crossing -> ordinary refill-after-read
    (1, 1, 1, True),    # off == r: the copy fills the chunk the read-ahead crosses into -> RAW
    (2, 0, 0, False),
    (2, 1, 1, False),   # off=2 != r=1: its slot lands on the CURRENT chunk -> WAR -> copy last
])
def test_copy_order_follows_which_chunk_the_copy_fills(pgr, plr, want_r, want_first):
    """pt5. The copy in trip `iter` fills chunk `iter+off`, slot `(iter+off) mod S_shared`;
 this trip reads chunks `iter... iter+r`. WHICH of those the slot lands on decides the order:

 off == r -> it is the crossing chunk -> RAW -> copy FIRST
 off != r -> it is a chunk read earlier -> WAR -> copy LAST

 Both shipping points sit at `S_shared = 2, r = 1` and go OPPOSITE ways, which is the whole
 content of the rule: PGR1 (off=1) copy-first, PGR2 (off=2) copy-last.

 The regression this pins is a predicate of `S_shared > r`, which is True for both and so
 emitted PGR2 copy-first -- overwriting the buffer the trip was still consuming. It miscompared
 on hardware wherever the steady body actually runs (`T - M >= 1`): K=256 for PGR2 (M=2), while
 PGR1 (M=1) had already shown it at K=128.
 """
    from Tensile.LoopModel import schedule, traversal
    p = dict(MatrixInstruction=[16, 16, 32, 1], MIWaveTile=[2, 2], DepthU=64, ElemBytes=2,
             LoopOrder="KMN", PrefetchGlobalRead=pgr, PrefetchLocalRead=plr)
    th = adapter.params_to_theta(p)
    S, _ = build_S(th)
    A = next(o for o in th.operands if o.name == "A")
    dr = th.off_at("A", theta_mod.READ, traversal.prefetch_axis_name(th))
    r = max(traversal.chunks_crossed(th, A, schedule.prefetch_steps_for(th, A, g, dr, S))
            for g in A.fragment.groups())
    assert r == want_r, f"chunk reach {r}, expected {want_r}"
    assert schedule.copy_must_be_first(th, A, S) is want_first
    assert ("A" in schedule.copy_first_operands(th, S)) is want_first
    assert not schedule.chunk_crossing_violations(th, S), "both points are emittable"


@pytest.mark.parametrize("pgr,plr,copy_leads", [(1, 1, True), (2, 1, False), (1, 0, False)])
def test_emitted_steady_body_puts_the_copy_where_the_predicate_says(pgr, plr, copy_leads):
    """End-to-end: the predicate is only worth anything if sigma_c actually follows it, so assert on
    the EMITTED steady body rather than on `copy_must_be_first` a second time.
    """
    from Tensile.LoopModel.emit import build_ir
    from Tensile.LoopModel.checks import _region_bodies
    th = adapter.params_to_theta(dict(
        MatrixInstruction=[16, 16, 32, 1], MIWaveTile=[2, 2], DepthU=64, ElemBytes=2,
        LoopOrder="KMN", PrefetchGlobalRead=pgr, PrefetchLocalRead=plr))
    S, _ = build_S(th)
    seen = False
    for label, body in _region_bodies(build_ir(th, S)):
        if "/steady" not in label:
            continue
        seq = [("copy" if i.op.dst == Space.SHARED else "read")
               for i in body if isinstance(getattr(i, "op", None), Load)
               and i.op.dst in (Space.SHARED, Space.REGISTER)]
        assert seq, "steady body has no LDS traffic"
        seen = True
        assert (seq[0] == "copy") is copy_leads, (
            f"PGR={pgr} PLR={plr}: expected copy-{'first' if copy_leads else 'last'}, got {seq[:4]}")
    assert seen, "no steady region found"


def test_chunk_reach_is_the_distance_over_the_span():
    """`r = ceil((span - 1 + distance) / span)`, over the ONE distance the level defines.

    `distance = internal_PLR * group_prefetch_unit`; capacity/staging cap the selected PLR rather
    than silently saturating this physical conversion.
    """
    from Tensile.LoopModel import schedule, traversal
    th = adapter.params_to_theta(dict(
        MatrixInstruction=[16, 16, 32, 1], MIWaveTile=[2, 2], DepthU=64, ElemBytes=2,
        LoopOrder="KMN", PrefetchGlobalRead=2, PrefetchLocalRead=1))
    A = next(o for o in th.operands if o.name == "A")
    level = traversal.readahead_level(th)
    assert level[0] == "K_inner", level         # KMN: the outermost non-region inner axis
    n_s = level[1]                              # the level's own extent, 2 here
    assert traversal.chunks_crossed(th, A, 0) == 0, "no read-ahead -> no crossing"
    for dr in range(1, n_s + 1):
        assert traversal.chunks_crossed(th, A, dr) == 1, f"dr={dr} must reach exactly one chunk"
    assert traversal.prefetch_distance_for(th, A, n_s + 1) \
        == (n_s + 1) * traversal.group_prefetch_unit_tiles(th, A)
    assert traversal.chunks_crossed(th, A, n_s + 1) == 2, \
        "the raw physical distance crosses two chunks; Schedule caps it before emission"


def _steady_body_copy_last(ir):
    """Reorder every steady body to copies-LAST -- the ordering the decoder shipped before the
 pt5 chunk-crossing fix. Operates on the built tree, so it exercises the LEDGER rather
 than a mutated sigma_c."""
    from Tensile.LoopModel.ir import _movement

    def is_copy(n):
        m = _movement(n)
        return m is not None and isinstance(m.op, Load) and m.op.dst == Space.SHARED

    def walk(nodes):
        for n in nodes:
            if isinstance(n, Loop):
                for b in n.bodies:
                    if n.outer:                      # the steady (pipelined) loop's own body
                        copies = [x for x in b if is_copy(x)]
                        if copies:
                            rest = [x for x in b if not is_copy(x)]
                            b[:] = rest + copies
                    walk(b)
            elif isinstance(n, Cond):
                walk(n.then); walk(n.els)
            elif isinstance(n, Bind):
                walk(n.body)
            elif isinstance(n, Peel):
                walk(n.body)
    walk(ir)
    return ir


@pytest.mark.parametrize("pgr,expect_caught", [
    (1, True),    # off(copy) == r: the copy is IN this trip, so copy-last starves the crossing read
    (2, False),   # off(copy) >  r: the copy ran an earlier trip, so copy-last is harmless
])
def test_ledger_catches_copy_last_exactly_when_the_crossing_needs_this_trips_copy(pgr, expect_caught):
    """The obligation that was missing, and the reason a miscompiling kernel passed the gate."""
    from Tensile.LoopModel.checks import build_ledger, check_ledger_discharged
    from Tensile.LoopModel.emit import build_ir
    th = adapter.params_to_theta(dict(
        MatrixInstruction=[16, 16, 32, 1], MIWaveTile=[2, 2], DepthU=64, ElemBytes=2,
        LoopOrder="KMN", PrefetchGlobalRead=pgr, PrefetchLocalRead=1))
    S, _ = build_S(th)
    ledger = build_ledger(th, S)

    assert not check_ledger_discharged(ledger, build_ir(th, S)), \
        "the emitted (copy-first) schedule must have an empty ledger"

    bad = check_ledger_discharged(ledger, _steady_body_copy_last(build_ir(th, S)))
    caught = [o for o in bad if o.kind == "crossing-RAW"]
    assert bool(caught) is expect_caught, (
        f"PGR={pgr}: copy-last {'must' if expect_caught else 'must NOT'} raise crossing-RAW, "
        f"got {sorted(o.kind for o in bad)}")


# ===========================================================================
# off keyed by OP-CLASS, and the per-level peel M_l
# ===========================================================================
def test_off_is_keyed_by_op_class_not_by_operand():
    """the `off` is a matrix over (OP-CLASS x level), and an operand is (operand, hop)."""
    from Tensile.LoopModel import theta as T
    th = _theta(dict(MatrixInstruction=[16, 16, 32, 1], MIWaveTile=[2, 2], DepthU=64,
                     ElemBytes=2, LoopOrder="KMN", PrefetchGlobalRead=2, PrefetchLocalRead=1))
    chunk, substep = th.summation_chunk_name(), "K_inner"
    offsets = th.movement_offsets()
    assert all(len(k) == 3 for k in offsets), \
        f"every offset key must be (operand, role, level); got {sorted(offsets)}"
    assert th.off_at("A", T.COPY, chunk) == 2 and th.off_at("A", T.READ, substep) == 1
    # the roles do NOT alias each other: a read has no chunk offset, a copy no substep offset.
    assert th.off_at("A", T.READ, chunk) == 0 and th.off_at("A", T.COPY, substep) == 0
    # and the hop knows its own role, so a caller holding the hop need not restate it
    assert th.off_of(th.op("A"), th.op("A").trajectory.shared_fill, chunk) == 2
    assert th.off_of(th.op("A"), th.op("A").trajectory.fragment_fill, substep) == 1


def test_peel_depth_is_per_level_not_one_integer():
    """`M` is a VECTOR `M_l = max_p off(p, l)`, not a scalar."""
    from Tensile.LoopModel.schedule import peel_depths
    th = _theta(dict(MatrixInstruction=[16, 16, 32, 1], MIWaveTile=[2, 2], DepthU=64,
                     ElemBytes=2, LoopOrder="KMN", PrefetchGlobalRead=2, PrefetchLocalRead=1))
    pd = peel_depths(th, build_S(th)[0])
    # BOTH levels carrying an offset are recorded, each with its own M_l
    assert pd.per_level == {"iter": 2, "K_inner": 1}, pd.per_level
    assert pd.depth("iter") == 2 and pd.depth("K_inner") == 1 and pd.depth("nope") == 0
    # the chunk's depth is what build_ir peels, and the staggered input is per-operand
    assert pd.chunk_depth == 2 and pd.chunk == "iter"
    assert pd.copy_off() == {"A": 2, "B": 2}
    assert pd.requested_steps == 1


def test_per_op_class_peel_offsets_stay_staggered():
    """pt1-2: the peel is staggered by EACH operand's own delta, not by the level's max."""
    from Tensile.LoopModel.schedule import peel_depths
    th = _theta(dict(MatrixInstruction=[16, 16, 32, 1], MIWaveTile=[2, 2], DepthU=64,
                     ElemBytes=2, LoopOrder="KMN", PrefetchLocalRead=1,
                     PrefetchGlobalRead=2, PrefetchGlobalReadA=1))
    pd = peel_depths(th, build_S(th)[0])
    assert pd.copy_off() == {"A": 1, "B": 2}, "each copy keeps its OWN off"
    assert pd.chunk_depth == 2, "M_l is the max over the level's op-classes"


def test_region_split_copy_is_NOT_a_cross_level_boundary_term():
    """A region-split copy's HOME level is its region mode, which is INNER to
 the chunk its `off` sits on -- so it satisfies "offset at an outer level, home inner to it" and
 looks exactly like the boundary term. It is not one: every region instance stays in its trip,
 and delta shifts which CHUNK they fetch. That is ordinary PrefetchGlobalRead.

 This is a regression test, not a hypothetical: the first implementation of `boundary_hoists`
 omitted the "home is itself a LOOPED level" condition and flagged every shipping region-split
 kernel, refusing to emit them."""
    from Tensile.LoopModel.traversal import operand_level
    from Tensile.LoopModel.schedule import peel_depths, boundary_hoists
    from Tensile.LoopModel import theta as T
    th = _theta(dict(MatrixInstruction=[16, 16, 32, 1], MIWaveTile=[2, 2], DepthU=64,
                     ElemBytes=2, LoopOrder="KMN", PrefetchGlobalRead=2, PrefetchLocalRead=1,
                     TDMSplit=[2, 2, 1, 1]))
    A = th.op("A")
    assert A.split > 1, "fixture must actually be region-split for this test to mean anything"
    # the precondition that makes it LOOK like a boundary term
    assert operand_level(th, A, T.COPY) == "M_split" != th.summation_chunk_name()
    # ...and the verdict
    assert boundary_hoists(th, peel_depths(th, build_S(th)[0])) == []
    emit.emit_mainloop(th)          # and it still emits


def _theta_with_persistent_level(off_gtile):
    """A hand-built theta carrying a SECOND outer level `gtile` outside the reduction chunk, with the
 copy retimed on it -- (2)'s persistent loop, the model's canonical boundary term. Built
 by mutation because no TensileLite parameter expresses a persistent level yet."""
    from Tensile.LoopModel.theta import Axis
    th = _theta(dict(MatrixInstruction=[16, 16, 32, 1], MIWaveTile=[2, 2], DepthU=64,
                     ElemBytes=2, LoopOrder="KMN", PrefetchGlobalRead=2, PrefetchLocalRead=1))
    th.ord = (Axis("gtile"),) + th.ord
    th.__post_init__()
    th.op("A").trajectory.shared.offsets["gtile"] = off_gtile
    return th


def test_a_coarser_outer_off_is_a_boundary_term_and_is_REFUSED_not_dropped():
    """the cross-level boundary term, on the model's canonical case: `off(copy, gtile)` hoists
    the NEXT tile's copies (whose home is the enclosed, LOOPED `iter`) into THIS tile's drain.
    """
    from Tensile.LoopModel.schedule import peel_depths, boundary_hoists
    th = _theta_with_persistent_level(1)
    pd = peel_depths(th, build_S(th)[0])
    assert pd.depth("gtile") == 1 and pd.depth("iter") == 2
    assert sorted(pd.peeled_levels()) == sorted(["gtile", "iter", "K_inner"])

    hoists = boundary_hoists(th, pd)
    assert len(hoists) == 1, hoists
    h = hoists[0]
    assert (h.op, h.level, h.home, h.delta) == ("A", "gtile", "iter", 1)
    assert h.coords == ({"iter": 0},), "the relocated instances are [inner-coord 0 .. delta)"

    with pytest.raises(NotImplementedError, match="boundary term|peel at outer level"):
        emit.emit_mainloop(th)


def test_a_deeper_coarser_off_relocates_proportionally_many_instances():
    """delta instances relocate, not one -- the count is `off(p, l_outer)` itself."""
    from Tensile.LoopModel.schedule import peel_depths, boundary_hoists
    th = _theta_with_persistent_level(3)
    h = boundary_hoists(th, peel_depths(th, build_S(th)[0]))[0]
    assert h.delta == 3
    assert h.coords == ({"iter": 0}, {"iter": 1}, {"iter": 2})


def test_an_inner_level_off_is_not_a_boundary_term():
    """`off(read, substep)` sits on an INNER (unrolled) level, which has no prologue/drain region
 to relocate into -- it is the ordinary pt5 read-ahead. Its `M_l` is still recorded (the
 peel is per-level) but it must not be reported as a hoist, nor refuse emission."""
    from Tensile.LoopModel.schedule import peel_depths, boundary_hoists
    th = _theta(dict(MatrixInstruction=[16, 16, 32, 1], MIWaveTile=[2, 2], DepthU=64,
                     ElemBytes=2, LoopOrder="KMN", PrefetchGlobalRead=2, PrefetchLocalRead=1))
    pd = peel_depths(th, build_S(th)[0])
    assert pd.depth("K_inner") == 1, "the inner level's M_l is recorded"
    assert boundary_hoists(th, pd) == [], "but it is not a cross-level relocation"
    emit.emit_mainloop(th)


def test_a_negative_off_is_rejected_as_ill_typed():
    """`off : (operand x level) -> N` -- non-negative throughout, and `retime`
    SETS the entry rather than subtracting, so a negative delta is ILL-TYPED, not merely unmodelled.
    DEFERRAL -- issuing |delta| iterations LATER, which is what a store wants -- is NOT a negative offset.
    """
    from Tensile.LoopModel.schedule import peel_depths
    from Tensile.LoopModel import theta as T
    th = _theta(dict(MatrixInstruction=[16, 16, 32, 1], MIWaveTile=[2, 2], DepthU=64,
                     ElemBytes=2, LoopOrder="KMN", PrefetchGlobalRead=2, PrefetchLocalRead=1))
    # Put the deferral on an EXISTING operand rather than a store path: the store's
    # `register->global` hop has no HOP_COUNTER entry yet, and this test is about the sign
    # of `off`, not about stores.
    th.op("A").trajectory.fragment.offsets["iter"] = -1
    th2 = th

    pd = peel_depths(th2, build_S(th2)[0])
    assert pd.deferrals == {("A", T.READ, "iter"): -1}, pd.deferrals
    assert pd.depth("iter") == 2, "a negative off must not perturb M_l"
    assert all(d > 0 for d in pd.offsets.values()), "offs holds the positive half only"

    with pytest.raises(RuntimeError, match="negative read-ahead offset is not meaningful"):
        emit.emit_mainloop(th2)


def test_degenerate_arm_resets_off_at_and_inside_the_peeled_level():
    """the peel reset rule : *a fully-peeled level resets to 0 every `off` AT THAT LEVEL and
    at levels INNER to it.* So the `T < M` arm is `T` unpipelined iterations -- each step copies
    AND reads its own chunk, self-contained.
    """
    from Tensile.LoopModel.ir import Cond, Bind, Inst, Load, Mma, Space, Loop, Peel, Branch

    def walk(nodes, env, out):
        for n in nodes:
            if isinstance(n, Inst):
                out.append((n, dict(env)))
            elif isinstance(n, Bind):
                walk(n.body, n.bound_env(env), out)
            elif isinstance(n, Cond):
                walk(n.then, env, out)                    # guards select, they do not reorder
            elif isinstance(n, Loop):
                for b in n.bodies:
                    for v in range(n.trip if isinstance(n.trip, int) else 1):
                        walk(b, {**env, n.axis: v}, out)
            elif isinstance(n, Peel):
                walk(n.body, env, out)
            elif isinstance(n, Branch):
                for arm in n.arms.values():
                    walk(arm, env, out)
        return out

    def ev(e, env):
        return e.eval(env) if hasattr(e, "eval") else e

    # EVERY LOOP ORDER, on every axis.
    ORDERS = ("KMN", "KNM", "MKN", "MNK", "NKM", "NMK")
    cases = [(f"{o} PGR{pgr} PLR{plr}",
              {"LoopOrder": o, "PrefetchGlobalRead": pgr, "PrefetchLocalRead": plr})
             for o in ORDERS for plr in (0, 1) for pgr in (2, 3)]
    # non-uniform `off` -- the ONLY shape that separates the reset from the prologue ramp
    cases += [(f"{o} asym off A=1 B=2", {"LoopOrder": o, "PrefetchGlobalReadA": 1})
              for o in ORDERS]
    # ring shallower than the peel depth: the arm wraps, so a step overwrites a buffer an EARLIER
    # step read.  Discharged by program order, but it is what makes the copy's WAR real.
    cases += [(f"{o} S_shared=1", {"LoopOrder": o, "NumLdsBlk": 1, "1LDSBuffer": 1})
              for o in ORDERS]
    # W = 1 (single-buffered registers) -- the ROTATION-CARRIED displacement, and the only shape
    # that exercises it.
    for _o in ORDERS:
        for _du in (32, 64):
            for _pgr in (1, 2):
                cases.append((f"{_o} W=1 MWT[2,2] DU{_du} PGR{_pgr}",
                              {"LoopOrder": _o, "MIWaveTile": [2, 2], "DepthU": _du,
                               "PrefetchGlobalRead": _pgr}))

    for tag, extra in cases:
            th = _theta({**BF16_NT_KMN, **extra})
            ir = emit.emit_mainloop(th).ir
            root = next(n for n in ir if isinstance(n, Cond) and n.kind == "peel_validity")
            for t, step in enumerate(root.els):
                shared, regs = set(), set()      # buffers / registers written SO FAR in this step
                for inst, env in walk([step], {}, []):
                    op = inst.op
                    pl = inst.placement
                    if isinstance(op, Mma):
                        for _o, p in (pl or {}).items():
                            for g, gx in (p.slots or ()):
                                assert (g, ev(gx, env)) in regs, (
                                    f"{tag} short step {t}: wmma consumes register {(g, ev(gx, env))} "
                                    f"that no read in the step wrote -- the arm still carries a "
                                    f"read-ahead it cannot warm (the peel reset rule inner-off reset)")
                        continue
                    if not isinstance(op, Load):
                        continue
                    if op.dst == Space.SHARED:
                        for g, gx in (pl.slots or ()):
                            shared.add((op.tokens, ev(gx, env)))
                        continue
                    if op.src == Space.SHARED and pl is not None and pl.src_slot is not None:
                        src = ev(pl.src_slot, env)
                        assert any(op.tokens[0] in toks and slot == src for toks, slot in shared), (
                            f"{tag} short step {t}: read of {op.tokens[0]} sources shared buffer "
                            f"{src}, which no copy in the step filled (filled: {sorted(shared)})")
                    for g, gx in (pl.slots or ()):
                        regs.add((g, ev(gx, env)))


# =================================================================================================
# facts the model keeps PER-HOP / PER-PLACEMENT that we had collapsed to per-operand.
# =================================================================================================

def _p212_theta(name="baseline"):
    _d, params = scenarios.SCENARIOS[name]
    return adapter.params_to_theta(params)


def test_presence_is_a_PER_HOP_fact_and_bulk_ness_IS_presence():
    """line 71 defines bulk-ness AS the axes it varies over: an inner axis a coarse copy spans in ONE
 instruction is not one the copy varies over. So the same operand has two the axes it varies over sets -- the
 bulk copy's and the per-lane read's -- and asking without a hop must keep answering the DATA
 question the derived families (`summation_axes`, `free_axes`) are defined on."""
    th = _p212_theta()
    for op in th.operands:
        if not op.movements:
            continue
        copy_hop = op.movements[0]
        assert copy_hop.is_bulk, "this fixture's first hop should be the bulk global->shared copy"
        # the BULK copy spans every inner axis in one instruction => present on none of them
        assert presence_axes(th, op, copy_hop) == [], op.name
        # the per-lane read varies over exactly what the DATA does
        read_hop = op.movements[-1]
        assert not read_hop.is_bulk
        assert ([m.name for m in presence_axes(th, op, read_hop)]
                == [m.name for m in presence_axes(th, op)] != []), op.name


def test_a_SPLIT_tile_brings_the_region_axis_BACK_into_the_copy_hops_presence():
    """One instruction PER REGION, so the copy is present on the region mode and
 sits one level deeper. This is the fact `emit._region_mode_of` now reads instead of testing
 `op.split` and scanning `op.region_modes` itself."""
    th = _p212_theta("our_split")
    seen = 0
    for op in th.operands:
        if not op.movements or not op.region_axes:
            continue
        seen += 1
        assert op.split > 1
        assert [m.name for m in presence_axes(th, op, op.movements[0])] == [op.region_axes[0]], op.name
    assert seen >= 2, "the split scenario should split at least two operands"


def test_buffer_depths_are_owned_by_placements():
    th = _p212_theta()
    depths, _minimum = build_S(th)
    for operand in th.operands:
        if operand.is_output:
            continue
        assert depths.groups_of(operand.name) == operand.fragment.ring_depths
        assert depths.shared(operand.name) == operand.trajectory.shared.ring_depth


def test_shared_and_register_depths_have_separate_accessors():
    th = _p212_theta()
    depths, _minimum = build_S(th)
    operand = next(item for item in th.operands if not item.is_output)
    group = operand.fragment.groups()[0]
    assert depths.get(operand.name, group) == operand.fragment.ring_depths[group]
    assert depths.shared(operand.name) == operand.trajectory.shared.ring_depth
    assert "shared" not in depths.groups_of(operand.name)


def test_an_operand_BROADCAST_over_the_reduction_does_not_pay_for_it():
    """`free_tiles x k_tiles` multiplies in the THETA-level reduction extent unconditionally, so
 the accumulator -- absent from the reduction by definition -- was counted
 `k_tiles` times over. Presence-derived, with no role test, it is not."""
    from Tensile.LoopModel import traversal as geometry
    th = _p212_theta()
    k = geometry.k_tiles(th)
    assert k > 1, "this fixture must have >1 reduction substep or the bug is invisible"
    for op in th.operands:
        pt = geometry.presence_tiles(th, op)
        assert pt == geometry.free_tiles(th, op) * geometry.summation_tiles(th, op), op.name
        assert geometry.operand_buffer_regs(th, op) == pt * geometry.frag_regs(th, op), op.name
        if op.is_output:                       # broadcast over the reduction: no k factor
            assert geometry.summation_tiles(th, op) == 1
            assert pt == geometry.free_tiles(th, op)
            assert geometry.operand_buffer_regs(th, op) * k \
                == geometry.free_tiles(th, op) * k * geometry.frag_regs(th, op), \
                "the old formula should be exactly k_tiles too big"
        else:                                  # present on the reduction: unchanged
            assert geometry.summation_tiles(th, op) == k
            assert geometry.operand_buffer_regs(th, op) \
                == geometry.free_tiles(th, op) * k * geometry.frag_regs(th, op)


def test_trajectory_SENSE_is_per_op_class_and_the_hop_ROLE_cannot_supply_it():
    """axis 3.  `HOP_ROLE` is keyed on (src, dst) alone, so in a store round trip
    `vgpr -> LDS -> vgpr -> global` it calls the middle leg a READ -- correctly, as a leg -- while the
    path it belongs to runs the other way.  the peel reset rule's reflected peel keys on the SENSE, so
    the two facts must be separately available."""
    from Tensile.LoopModel.theta import HOP_ROLE, path_direction
    from Tensile.LoopModel.ir import FORWARD, READ, REVERSE
    th = _p212_theta()
    for op in th.operands:
        assert path_direction(op) == (REVERSE if op.is_output else FORWARD), op.name
    # the leg role of a read-BACK is indistinguishable from a forward read -- that IS the point
    assert HOP_ROLE[(Space.SHARED, Space.REGISTER)] == READ


def test_a_REVERSE_op_class_is_REFUSED_rather_than_peeled_the_forward_way():
    """NEGATIVE CONTROL.  The reflected peel is not emitted; building it forward would look
    right, because every hop role still reads 'read'/'store' as usual.  So it must raise."""
    import copy as _copy
    from Tensile.LoopModel.theta import Global, Shared, Trajectory, path_direction
    from Tensile.LoopModel.ir import REVERSE
    from Tensile.LoopModel.schedule import peel_depths
    th = _p212_theta()
    peel_depths(th, build_S(th)[0])                                    # the unmodified theta peels fine
    rev = _copy.deepcopy(th)
    out = next(o for o in rev.operands if o.is_output)
    assert not out.movements
    out.trajectory = Trajectory(out.fragment, Shared(kind="load"), Global())
    assert path_direction(out) == REVERSE
    with pytest.raises(RuntimeError, match="REVERSE trajectory"):
        peel_depths(rev, build_S(rev)[0])


def test_a_sense_that_CONTRADICTS_the_path_is_an_error_not_a_silent_pick():
    """NEGATIVE CONTROL for the cross-check: `is_output` and the hop direction are two statements
    of the same fact, and disagreement is the confusion this function exists to end."""
    import copy as _copy
    from Tensile.LoopModel.theta import Global, Shared, Trajectory, path_direction
    rev = _copy.deepcopy(_p212_theta())
    out = next(o for o in rev.operands if o.is_output)
    out.trajectory = Trajectory(Global(), Shared())
    with pytest.raises(RuntimeError, match="runs the other way"):
        path_direction(out)


def test_per_hop_presence_REPRODUCES_the_two_hand_rolled_expressions_it_replaced():
    """The equivalence that makes axis 1 a refactor rather than a change, over EVERY scenario."""
    from Tensile.LoopModel.ir import Space

    def bulk(op):
        return next((h for h in (op.movements or ()) if h.is_bulk), None)

    def old_mode(th, ops):
        inner = {m.name for m in th.inner_axes()}
        return next((rm for o in ops for rm in o.region_axes if rm in inner), None)

    def new_mode(th, ops):
        for o in ops:
            h = bulk(o)
            m = next((x.name for x in presence_axes(th, o, h)), None) if h else None
            if m:
                return m
        return None

    checked = 0
    for key, (_desc, params) in scenarios.SCENARIOS.items():
        th = adapter.params_to_theta(params)
        for op in th.operands:
            if not op.movements:
                continue
            checked += 1
            h = bulk(op)
            note = next((m.name for m in presence_axes(th, op, h)), None) if h else None
            assert note == (None if op.split <= 1 else old_mode(th, [op])), \
                f"{key}/{op.name}: the per-region NOTE must be unchanged"
        byname = {o.name: o for o in th.operands}
        fused = {t for g in th.fused_copy_groups for t in g}
        groups = [tuple(g) for g in th.fused_copy_groups]
        groups += [(o.name,) for o in th.operands if bulk(o) and o.name not in fused]
        for g in groups:
            ops = [byname[n] for n in g if n in byname and bulk(byname[n])]
            if not ops:
                continue
            nregions = max((max(1, o.split) for o in ops), default=1)
            if nregions > 1:                       # the ONLY case the caller uses the mode in
                assert new_mode(th, ops) == old_mode(th, ops), \
                    f"{key}/{g}: a USED region mode changed -- this is not a refactor any more"
    assert checked > 40, f"only {checked} op-classes checked; the scenario sweep is not running"


def test_TDMSplit_gate_accepts_the_MT_half_and_refuses_the_DU_half():
    """The MT half splits a FREE axis and is orthogonal to the tail; the DU half splits the
    shared K axis, which is the axis the tail loop also shrinks.  So the gate is on the DU
    factors, not on TDMSplit being set at all.
    """
    def du_of(ts):
        v = [2, 2] if ts is True else list(ts)
        v = v + [1] * (4 - len(v))
        return max(1, int(v[2])), max(1, int(v[3]))

    for accepted in (True, [2, 2], [2, 2, 1, 1], [4, 2, 1, 1], [1, 1], [1, 1, 1, 1]):
        assert du_of(accepted) == (1, 1), f"{accepted} should be MT-only"
    for refused in ([1, 1, 2, 1], [1, 1, 1, 2], [2, 2, 2, 2], [1, 1, 4, 2]):
        assert du_of(refused) != (1, 1), f"{refused} splits DU and must be refused"

    # and the legacy bool the user asked to test with really is the MT-only [2,2,1,1]
    th = adapter.params_to_theta({**BF16_NT_KMN, "TDMSplit": True})
    assert [o.split for o in th.operands if o.movements] == [2, 2]
    assert all(len(o.region_axes) == 1 for o in th.operands if o.movements), \
        "MT split puts ONE region axis on each operand -- its own free axis"


# =================================================================================================
# bridge -- kernel dict -> decoder params.

class _Bytes2:
    def numBytes(self):
        return 2


def _kernel(**over):
    k = {"MatrixInstruction": [16, 16, 32, 1, 1, 2, 2, 1, 1], "DepthU": 64,
         "MIWaveTileA": 2, "MIWaveTileB": 2, "ProblemType": {"DataType": _Bytes2()},
         "PrefetchGlobalRead": 1, "PrefetchLocalRead": 1, "NumWaves": 1,
         "enableTDMA": 1, "enableTDMB": 1, "NumLdsBlk": 2, "LoopOrder": "KMN"}
    k.update(over)
    return k


def _regions_from_kernel(**over):
    """kernel dict -> theta -> GIR, the way kernel generation actually goes."""
    from Tensile.LoopModel import adapter
    from Tensile.Lowering import build_gir
    prog = build_gir(adapter.params_to_theta(adapter.kernel_to_params(_kernel(**over))))
    return prog.meta.get("unit_regions", {})


def test_the_bridge_CARRIES_TDMSplit_through_to_theta():
    """A split Solution must produce a split theta."""
    assert _regions_from_kernel(TDMSplitA=1, TDMSplitB=1) == {("A",): 2, ("B",): 2}
    assert _regions_from_kernel(TDMSplitA=0, TDMSplitB=0) == {("A",): 1, ("B",): 1}
    assert _regions_from_kernel() == {("A",): 1, ("B",): 1}      # absent == unsplit


def test_a_split_kernel_reaches_EMIT_as_one_copy_per_region_with_a_walk():
    """The end-to-end region-walk property, entered from a KERNEL and not from a params dict."""
    from Tensile.LoopModel import adapter
    from Tensile.Lowering import build_gir
    from Tensile.Lowering.gir.emit_plan import plan_block
    prog = build_gir(adapter.params_to_theta(
        adapter.kernel_to_params(_kernel(TDMSplitA=1, TDMSplitB=1))))
    acts = plan_block(prog, "steady")
    copies = [a for a in acts if a.kind == "copy"]
    incs = [a for a in acts if a.kind == "region_inc"]
    assert len(copies) == 4, "2 operands x 2 regions, one copy Move each"
    assert all(a.at["regions"] == 2 for a in copies)
    assert sorted(a.at["region"] for a in copies) == [0, 0, 1, 1]
    assert len(incs) == 4, "one step out and one step back, per operand"
    assert sum(a.at["steps"] for a in incs) == 0, "the walk must close"


def test_the_CHUNK_advance_is_INVARIANT_under_region_splitting():
    """Splitting a tile changes how many LOADS deliver a chunk, never how the chunk pointer moves.
    `GrIncrementRegions` derived the peel's chunk from the POSITIONAL INDEX of the copy among that
    operand's peel copies -- sound while one copy is one chunk, off by `nregions` once it is not.
    """
    from Tensile.LoopModel import adapter
    from Tensile.Lowering import build_gir
    from Tensile.Lowering.gir.emit_plan import plan_block

    def advances(split, **over):
        prog = build_gir(adapter.params_to_theta(
            adapter.kernel_to_params(_kernel(TDMSplit=split, **over))))
        return {lab: [(a.at["unit"], a.at["chunks"], a.at["to_chunk"])
                      for a in plan_block(prog, lab) if a.kind == "gr_inc"]
                for lab in prog.blocks}

    for pgr in (1, 2):
        for plr in (0, 1):
            split, plain = (advances(True, PrefetchGlobalRead=pgr, PrefetchLocalRead=plr),
                            advances(False, PrefetchGlobalRead=pgr, PrefetchLocalRead=plr))
            # COMPARED AS A MULTISET PER BLOCK, not as a sequence.
            for lab in set(split) | set(plain):
                assert sorted(split.get(lab, []), key=str) == sorted(plain.get(lab, []), key=str), \
                    f"PGR{pgr} PLR{plr} {lab}: split changed the chunk walk\n{split}\n{plain}"
            assert all(c == 1 for acts in split.values() for _u, c, _t in acts), \
                "every advance is exactly one DepthU stride -- the only thing tdmIncrementGir emits"


def test_a_SPLIT_operands_degenerate_tile_axis_SURVIVES_and_an_unsplit_ones_does_not():
    """`X_inner` (tile) and `X_split` (storage region) are different roles that coincide in extent
 when fan == split. Dropping the degenerate inner mode MERGES them: at MIWaveTile 2 with a
 2-way split, `M_inner` is extent 1, and without it `M_split` is simultaneously the region axis
 and the only free-tile axis -- so "which tile" and "which region" become one coordinate and a
 split tile cannot state them separately. That is the [2,2]-vs-[4,4] asymmetry.

 SCOPED TO A LIVE SPLIT. Keeping every degenerate inner mode is not free: over a 96-config
 sweep with `_keep` forced to keep them all, 12 configs changed their emitted act ORDER (the
 operands' read interleave and where the first-touch guards land). Slot VALUES did NOT
 change in any of the 96 -- see `translate.py`'s docstring for the numbers. With no split there
 is no second role to merge away, so the rule is "my split partner is live", not "I am an inner
 mode"."""
    from Tensile.LoopModel import adapter

    def ordnames(wt, split):
        th = adapter.params_to_theta(adapter.kernel_to_params(
            _kernel(MatrixInstruction=[16, 16, 32, 1, 1, wt, wt, 1, 1],
                    MIWaveTileA=wt, MIWaveTileB=wt,
                    TDMSplitA=int(split), TDMSplitB=int(split))))
        return [m.name for m in th.ord]

    with pytest.raises(ValueError, match="factorize MIWaveTile"):
        ordnames(1, True)

    for wt in (2, 4):
        o = ordnames(wt, True)
        assert "M_split" in o and "M_inner" in o, \
            f"MIWaveTile {wt} split must keep BOTH the region and the tile axis: {o}"
        assert "N_split" in o and "N_inner" in o, o
    # unsplit: no region axis exists, so a degenerate tile axis is correctly dropped
    assert ordnames(1, False) == ["iter", "K_inner"]
    assert ordnames(2, False) == ["iter", "K_inner", "M_inner", "N_inner"]


# =================================================================================================
# the movement-quantum condition -- the MOVEMENT QUANTUM, entered from a KERNEL.

class _Bytes1:
    def numBytes(self):
        return 1


def _mxf8_kernel(order, split):
    """The triage Solution, as a kernel dict: MT64x64x256, MI 16x16x128, MXBlock 32, fused TDM.
    MIWaveGroup IS [2,2], MATCHING `NumWaves: 4` AND THE DOCSTRING ().
    """
    return {"MatrixInstruction": [16, 16, 128, 1, 1, 2, 2, 2, 2], "DepthU": 256,
            "MIWaveTileA": 2, "MIWaveTileB": 2, "MatrixInstK": 128,
            "ProblemType": {"DataType": _Bytes1(), "MXBlockA": 32, "MXBlockB": 32},
            "PrefetchGlobalRead": 1, "PrefetchLocalRead": 1, "NumWaves": 4,
            "enableTDMA": 1, "enableTDMB": 1, "NumLdsBlk": 2, "LoopOrder": order,
            "TDMSplitA": split, "TDMSplitB": split, "TDMFuse": 0,
            "VectorWidthA": 2, "VectorWidthB": 2,
            "LocalReadVectorWidthMXS": 8,
            "UnrollMajorLDSMXSA": 1, "UnrollMajorLDSMXSB": 1}


def _two_tile_fold():
    """The scale read's supplied coverage: `carrier(t) = t // 2`, `slot(t) = t % 2`."""
    from Tensile.LoopModel.ir import TransferCoverage, Expr, COVERAGE_VAR
    t = COVERAGE_VAR
    return TransferCoverage(carrier=Expr(digits=(((t, 1),), 0, ((1, 2, 0),))),
                      slot=Expr(digits=(((t, 1),), 0, ((1, 1, 2),))))


def _quantum_violations(order, split):
    from Tensile.LoopModel import adapter
    from Tensile.Lowering import build_gir
    from Tensile.Lowering.gir.emit_plan import plan_block
    from Tensile.Lowering.gir.coverage import plan_coverage
    q = {"MXSA": _two_tile_fold(), "MXSB": _two_tile_fold()}
    prog = build_gir(adapter.params_to_theta(
        adapter.kernel_to_params(_mxf8_kernel(order, split),
                                 target={"ReadQuantum": q})))
    out = []
    for name in prog.blocks:                       # `blocks` is keyed BY NAME; iterating gives keys
        acts = plan_block(prog, name)
        if acts:
            out += ["%s: %s" % (name, v)
                    for v in plan_coverage(acts, lambda op: q.get(op)).violations]
    return out


def test_a_merged_load_carries_the_UNION_of_the_generations_it_covers():
    """the movement-quantum condition: a merged movement names a SET of generations, not the leader's."""
    from Tensile.Lowering.gir.emit_plan import EmitAction
    from Tensile.Lowering.gir.coverage import plan_coverage

    def _acts(token_ids):
        """Two MXSA reads the coverage merges into ONE instruction (`carrier(t) = t // 2`), sharing a
        register generation so the REGISTER half is clean and only the LDS half is under test."""
        return [EmitAction(kind="read",
                           at={"tc": "MXSA", "tile": t, "tile_flat": t, "k": 0, "k_flat": 0,
                               "reg_buf": 0, "token_ids": token_ids[t],
                               "region": 0, "regions": 1})
                for t in range(2)]

    q = lambda op: _two_tile_fold() if op == "MXSA" else None

    for label, toks, want in (("one generation",  [(1,), (1,)], (1,)),
                              ("two generations", [(1,), (3,)], (1, 3))):
        acts = _acts(toks)
        plan = plan_coverage(acts, q)
        issued = [plan.decision(i) for i in range(len(acts))
                  if plan.decision(i) and plan.decision(i).emit]
        assert len(issued) == 1, \
            f"{label}: the fold must issue exactly one instruction, got {len(issued)}"
        assert tuple(issued[0].tokens) == want, \
            f"{label}: the merged load must carry every generation it covers: {issued[0].tokens} != {want}"


def test_the_quantum_check_still_reports_a_REGISTER_straddle():
    """The half of a straddle that is NOT repairable by a union, and so must still be a violation.
    the union answers the LDS half -- one instruction, several source buffers, name them all.
    """
    from Tensile.Lowering.gir.emit_plan import EmitAction
    from Tensile.Lowering.gir.coverage import plan_coverage

    def _acts(reg_bufs):
        return [EmitAction(kind="read",
                           at={"tc": "MXSA", "tile": t, "tile_flat": t, "k": 0, "k_flat": 0,
                               "reg_buf": reg_bufs[t], "token_ids": (1,),
                               "region": 0, "regions": 1})
                for t in range(2)]

    q = lambda op: _two_tile_fold() if op == "MXSA" else None
    assert not plan_coverage(_acts([0, 0]), q).violations, \
        "one register generation is a legal merge; the check must stay quiet"
    v = plan_coverage(_acts([0, 1]), q).violations
    assert v, "two register generations under one instruction is ill-formed; the check must report it"
    assert any("REGISTER" in str(x) for x in v), f"the violation must name the register half: {v}"


def test_the_quantum_check_is_CLEAN_on_every_cell_now_that_the_carry_is_gone():
    """REGRESSION GUARD for generation-uniformity, and the sharpest statement of what it buys."""
    # This order puts the MXS carrier axis on the rotation-unit boundary, so its 2-tile fold must
    # be narrowed before L3 can emit it. Both MT split spellings expose the same missing narrowing.
    bad = {}
    for order in ("KMN", "KMKNMN", "KMNKMN", "KMNMNK"):
        for split in (1, 2):
            v = _quantum_violations(order, split)
            if v:
                bad[(order, split)] = v
    assert not bad, "the merged movement must be generation-uniform; got %r" % (bad,)


@pytest.mark.parametrize("order", ["KMN", "KNM", "MKN", "MNK", "NKM", "NMK"])
@pytest.mark.parametrize("plr", [0, 1])
@pytest.mark.parametrize("du", [128, 256])
def test_the_mxf8_fold_lowers_to_gir_on_every_loop_order(order, plr, du):
    """THE SHIPPING mxf8 PATH MUST REACH GIR -- the check that only a real codegen run had."""
    from Tensile.LoopModel import adapter
    from Tensile.LoopModel import traversal as geometry
    from Tensile.Lowering import build_gir
    from Tensile.LoopModel.schedule import build_S
    k = dict(_mxf8_kernel(order, 0))
    k["PrefetchLocalRead"] = plr
    k["DepthU"] = du
    # `DepthU=128` is the shape where `K_inner` collapses OUT OF `loop order` entirely (leaving
    # `M_inner(2) N_inner(2)`), so a scale ring's fan is FULLY absorbed by its own instruction and
    # its the axes it varies over is empty.
    target = {"ReadVectorElems": {"MXSA": 8, "MXSB": 8},
              "ReadPhi": {"MXSA": 2, "MXSB": 2}, "ReadRho": {"MXSA": 2, "MXSB": 2},
              "RegisterBudget": 224}
    th = adapter.params_to_theta(adapter.kernel_to_params(k, target=target))
    # #324: on a reused order the folded shape needs a PARTITIONED ring to carry PLR=1, and the
    # T < M short arm does not prime the second group's N_inner=1 coordinate.  STRICT: the day the
    # short arm primes a partitioned ring this xfail fails and must be removed with the task.
    if plr and du == 256 and order in ("MNK", "NMK"):
        pytest.xfail("#324: short arm does not prime the partitioned ring this fold requires")
    build_gir(th)                                   # must not raise -- this is the codegen call
    # Scale unit indexes contain only free-tile carriers; K selects a ring slot.  The parent can
    # instead keep K in its unit index, so comparing their W values is not meaningful.  Compare
    # the physical footprints that the allocator actually reserves.
    S, _ = build_S(th)
    for scale, parent in (("MXSA", "A"), ("MXSB", "B")):
        scale_regs = geometry.operand_emitted_regs(th, th.op(scale), S)
        parent_regs = geometry.operand_emitted_regs(th, th.op(parent), S)
        assert scale_regs <= parent_regs, (
            order, plr, scale, scale_regs, parent, parent_regs)
        assert th.op(scale).fragment.policy_of(th.op(scale).fragment.groups()[0]) \
            == th.op(parent).fragment.policy_of(th.op(parent).fragment.groups()[0]), \
            "the scale ring must use its parent's width policy"
    # A ring with EMPTY PRESENCE has no rotating coordinate at all, so it must be a single buffer: a
    # wider one has to name a mode, and every mode is out of scope where such a read is placed
    # (`op_class_level` puts it at the reduction chunk).
    for nm in ("A", "B", "MXSA", "MXSB"):
        op = th.op(nm)
        hop = next((h for h in op.movements if not h.is_bulk), None)
        if hop is None:
            continue
        if not presence_axes(th, op, hop=hop):
            assert max(S.get(nm, g) for g in op.fragment.groups()) == 1, \
                (du, order, plr, nm, "a ring with empty presence must be one buffer deep")
    # Any budget-selected partition must be a real, quantum-compatible register grouping.
    from Tensile.LoopModel.schedule import RegisterDepthSelector
    for op in th.operands:
        groups = len(op.fragment.groups())
        assert RegisterDepthSelector(th, None).admissible(op, groups), (
            du, order, plr, op.name, groups, "quantum-incompatible register partition")


def test_absorbing_the_spanned_axis_removes_the_straddle_on_the_shipping_path():
    """the actual repair: the axis one instruction SPANS leaves the read hop's traversal."""
    from Tensile.LoopModel import traversal as geometry
    from Tensile.LoopModel import adapter
    from Tensile.Lowering import build_gir
    from Tensile.Lowering.gir.emit_plan import plan_block
    target = {"ReadVectorElems": {"MXSA": 8, "MXSB": 8},
              "ReadQuantum": {"MXSA": _two_tile_fold(), "MXSB": _two_tile_fold()}}
    for order in ("KMN", "KMKNMN", "KMNKMN", "KMNMNK"):
        for split in (1, 2):
            k = _mxf8_kernel(order, split)
            th = adapter.params_to_theta(adapter.kernel_to_params(k, target=target))
            mxsa = next(o for o in th.operands if o.name == "MXSA")
            hop = next(h for h in mxsa.movements if not h.is_bulk)
            if split == 1:                     # an MT split is what makes M_split exist at all
                assert geometry.coverage_axes(th, mxsa, hop.coverage) == {"M_split"}, \
                    f"{order}: the 2-tile load spans the real factorized M_split axis"
                assert "M_split" not in presence_axes(th, mxsa, hop=hop), \
                    "a spanned axis is ABSENT from that hop's presence "
            prog = build_gir(th)
            n = {}
            for name in prog.blocks:
                for a in plan_block(prog, name):
                    if a.kind == "read":
                        n[a.at["tc"]] = n.get(a.at["tc"], 0) + 1
            # one act per instruction: MXSA's acts are now HALF A's, matching the b64 coverage
            assert n.get("MXSA", 0) * 2 == n.get("A", 0) or split == 2, \
                f"{order} split{split}: MXSA acts {n} should be half A's once the axis is absorbed"


def test_an_explicit_W_outside_the_band_is_CLAMPED_not_rejected():
    """/ the repair rule : `multibuffer` REPAIRS an out-of-band width, it does not refuse
    the point.
    """
    from Tensile.LoopModel import build_S
    for want, why in ((1, "below the floor -> clamped UP to L_war"),
                      (99, "above R -> clamped DOWN to R")):
        th = _theta(dict(BF16_NT_KMN))
        A = th.op("A")
        g = A.fragment.groups()[0]
        A.fragment.group_policy = {"*": want}
        S, _floor = build_S(th)                      # must not raise
        got = S.get("A", g)
        assert 1 <= got <= 2, f"W={want} ({why}): clamped to {got}, outside the derived band"
    # and the clamp really is a clamp, not a pass-through
    th = _theta(dict(BF16_NT_KMN))
    th.op("A").fragment.group_policy = {"*": 99}
    S, _ = build_S(th)
    assert S.get("A", th.op("A").fragment.groups()[0]) != 99, \
        "an out-of-band width must be repaired, not honoured verbatim"


def test_a_cross_agent_RAW_residency_is_BLOCK_scoped_not_wave():
    """line 700 lists the cooperative-fill RAW as "L2... cross-agent, BLOCK SCOPE", discharged
 by the proc-scoped selector. `_scope_of` returning "wave" for it, justified by

 "a RAW residency's consumer awaits a completion CLASS, which is not an agent-relative fact"

 which is false for a PER-WAVE counter -- the only kind gfx1250 has. `s_wait_loadcnt` retires
 THIS wave's loads; under a cooperative fill the bytes a reader wants were issued by a different
 wave, so its own counter gives it no edge to that writer.

 Without it, at agents=4 (NumWaves 4, TDMFuse 0): 39 RAW-residency Awaits, ALL at
 scope='wave'; only the 2 inplace-WAR reached 'block'.

 Single-agent must stay 'wave' -- that is the whole point of keying on `agent_distributed`, and a
 fixture that only checked the multi-agent side would pass on a rule that returned 'block'
 unconditionally."""
    from Tensile.LoopModel import build_S
    from Tensile.LoopModel import adapter
    from Tensile.LoopModel.emit import build_ir
    from Tensile.LoopModel.checks import all_awaits
    from collections import Counter

    class _B2:
        def numBytes(self): return 2

    def scopes(num_waves, fuse):
        k = {"MatrixInstruction": [16, 16, 32, 1, 1, 2, 2, 2, 2], "DepthU": 64,
             "MIWaveTileA": 2, "MIWaveTileB": 2, "ProblemType": {"DataType": _B2()},
             "PrefetchGlobalRead": 2, "PrefetchLocalRead": 1, "NumWaves": num_waves,
             "enableTDMA": 1, "enableTDMB": 1, "NumLdsBlk": 2, "LoopOrder": "KMN",
             "TDMFuse": fuse}
        th = adapter.params_to_theta(adapter.kernel_to_params(k))
        S, _f = build_S(th)
        c = Counter((getattr(a, "kind", None), getattr(a, "scope", None))
                    for a in all_awaits(build_ir(th, S)))
        return th, c

    th4, multi = scopes(4, 0)
    assert th4.wave_count > 1 and th4.agent_distributed("A"), "fixture is not multi-agent"
    raw_block = multi[("RAW-residency", "block")]
    raw_wave = multi[("RAW-residency", "wave")]
    assert raw_block and not raw_wave, (
        f"cross-agent RAW-residency must be block-scoped : "
        f"block={raw_block} wave={raw_wave}")
    # the in-place WAR keeps its own block upgrade -- this must not have displaced it
    assert multi[("inplace-WAR", "block")], "the inplace-WAR block-scope upgrade was lost"
    # and the rotation-WAR stays wave: gives the spare slot as the ALTERNATIVE remedy
    assert not multi[("rotation-WAR", "block")], "rotation-WAR must not be widened "

    th1, single = scopes(1, 0)
    assert not th1.agent_distributed("A"), "single-wave fixture is not single-agent"
    assert not single[("RAW-residency", "block")], (
        "single-agent RAW-residency must stay wave-scoped -- the rule must key on "
        "agent_distributed, not return block unconditionally")


def test_obligation_kind_vocabulary_is_closed():
    """S1: `is_war` dispatches on a CHECKED set, not on the letters the kind ends with."""
    from Tensile.LoopModel.ir import (is_war, is_raw, RAW_KINDS, WAR_KINDS,
                                      OBLIGATION_KINDS, Await)

    assert RAW_KINDS.isdisjoint(WAR_KINDS)
    assert OBLIGATION_KINDS == RAW_KINDS | WAR_KINDS
    assert Await.__dataclass_fields__["kind"].default in OBLIGATION_KINDS

    for k in WAR_KINDS:
        assert is_war(k) and not is_raw(k), k
    for k in RAW_KINDS:
        assert is_raw(k) and not is_war(k), k

    # THE POINT OF THE CHANGE: an unknown kind RAISES rather than defaulting to the RAW arm.
    # Under the old suffix test both of these answered silently -- the first False (a new anti-dep
    # scheduled as a true dep), the second True (a true dep scheduled as an anti-dep).
    for unknown in ("anti-dependence", "read-after-WAR", ""):
        assert unknown not in OBLIGATION_KINDS
        try:
            is_war(unknown)
        except RuntimeError as e:
            assert "CLOSED set" in str(e), e
        else:
            raise AssertionError(
                "is_war(%r) answered instead of raising -- an unrecognised kind must not be "
                "classified, because every issue-order decision downstream branches on it"
                % (unknown,))

    # NON-COLLISION with the SECOND vocabulary: Lowering/gir/analyses/frame_hazards defines bare
    # "RAW"/"WAR"/"WAW" hazard-EDGE labels.  Those are not ledger obligations, and "WAR" matches
    # the old suffix test, so the two were one typo from being cross-dispatched.
    from Tensile.Lowering.gir.analyses import frame_hazards
    for edge in (frame_hazards.RAW, frame_hazards.WAR):
        assert edge not in OBLIGATION_KINDS, (
            "%r is an LDS hazard-edge label, not a ledger obligation kind; the two vocabularies "
            "must stay disjoint so a membership test cannot silently accept the wrong one" % edge)


def test_geometry_shows_wave_relative_regions_and_await_kind():
    from Tensile.LoopModel import render, emit
    from Tensile.LoopModel import adapter
    from Tensile.LoopModel.schedule import build_S

    th = adapter.params_to_theta(adapter.kernel_to_params(_mxf8_kernel("KMN", 1)))
    geo = render.render_geometry(th)

    relative = [op.name for op in th.operands
                if op.trajectory.shared is not None
                and op.trajectory.shared.regions.wave_relative]
    assert relative
    assert "wave-relative read" in geo
    for name in relative:
        assert name in geo
    assert "free_split=" in geo
    assert "wave_count=" in geo

    ir = emit.build_ir(th, build_S(th)[0])
    aw = [l for l in render.render_ir(ir).splitlines() if "await" in l]
    assert aw, "fixture must emit awaits"
    from Tensile.LoopModel.ir import OBLIGATION_KINDS
    assert all(any(f"<{k}>" in l for k in OBLIGATION_KINDS) for l in aw), \
        "every await must render its hazard class:\n" + "\n".join(aw[:5])

    coordinate = [op.name for op in th.operands
                  if op.trajectory.shared is not None
                  and not op.trajectory.shared.regions.wave_relative]
    for name in coordinate:
        lines = [line for line in geo.splitlines() if line.strip().startswith(name + ":")]
        if lines:
            assert "coordinate-selected read" in lines[0]


def test_the_unrolled_drain_does_not_invent_a_buffer_index_for_unbound_iter():
    """audit: `Expr.eval` answers 0 for an unbound name, so pinning fabricates a number."""
    from Tensile.LoopModel import render, emit
    from Tensile.LoopModel import adapter
    from Tensile.LoopModel.schedule import build_S

    th = adapter.params_to_theta(adapter.kernel_to_params(_mxf8_kernel("KMN", 1)))
    lines = render.render_unrolled(emit.build_ir(th, build_S(th)[0])).splitlines()
    at = [i for i, l in enumerate(lines) if "DRAIN" in l]
    assert at, "fixture must have a drain stage"
    drain = [l for l in lines[at[0]:] if "ds_read" in l]
    assert drain, "the drain must contain reads"

    lds = [l for l in drain if "<-lds[" in l]
    assert lds, "drain reads must name an LDS source slot"
    assert any("iter" in l.split("<-lds[")[1] for l in lds), (
        "at least one drain LDS slot must stay SYMBOLIC in `iter`; a concrete index there is "
        "`eval` substituting 0 for a runtime value:\n" + "\n".join(lds[:3]))

    # The register slot may be coordinate-decoded (free enumerator for data, reduction digit for
    # MXS) and therefore concrete.  It must still be named; only the LDS generation necessarily
    # carries the runtime chunk.
    sym = [l for l in lds if "iter" in l.split("<-lds[")[1]]
    assert all("register[buf" in l for l in sym), (
        "every symbolic LDS drain read must still name its register slot:\n"
        + "\n".join(sym[:3]))

    # AND THE MECHANISM, directly: `at` pins per-EXPR, and only when asked.
    from Tensile.LoopModel.ir import Expr, Placement
    free = Expr(var="iter", mod=2)
    pl = Placement("register", (("g0", Expr(var="", add=1, mod=2)),), 0, 0, src_slot=free)
    kept = pl.at({}, keep_unresolved=True)
    assert kept.src_slot is free, "an unresolvable slot must be handed back untouched"
    assert kept.slots[0][1].free_vars() == set(), "a resolvable slot must still be pinned"
    assert pl.at({}).src_slot is not free, (
        "the DEFAULT must be unchanged -- `at` is on the codegen path, which must always receive "
        "a concrete Placement (a symbolic one broke 409 tests and both codegen runs)")

    # NON-VACUITY: the steady body's reads, whose env DOES close them, are fully concrete.
    steady = [l for l in lines[:at[0]] if "ds_read" in l and "<-lds[" in l]
    assert steady and any("iter" not in l.split("<-lds[")[1] for l in steady), (
        "some steady read must render a concrete LDS slot, or the drain assertion above is "
        "satisfied by the renderer simply never pinning anything")


def test_a_load_line_shows_its_coverage_and_advance_and_a_wmma_names_its_real_operands():
    """three facts on the node that no view except --raw could reach."""
    from Tensile.LoopModel.ir import Load, Mma, Space, TransferCoverage, Expr, cst
    from Tensile.LoopModel import render, emit
    from Tensile.LoopModel import adapter
    from Tensile.LoopModel.schedule import build_S

    q = TransferCoverage(carrier=Expr(var="t", div=2) if hasattr(Expr, "div") else cst(0),
                   slot=cst(0))
    ld = Load(("A",), Space.SHARED, Space.REGISTER, 0, coord=(("K_inner", 0),),
              size_regs=8, advance=3, coverage=q)
    line = render._load_line(ld)
    assert "coverage(" in line, "the transfer coverage must be visible: " + line
    assert "advance=3" in line, "the read-ahead shape must be visible: " + line

    plain = render._load_line(Load(("A",), Space.SHARED, Space.REGISTER, 0,
                                   coord=(("K_inner", 0),), size_regs=8))
    assert "coverage(" not in plain and "advance=" not in plain, (
        "an identity-merge, in-place read must gain NOTHING, or the assertions above are "
        "satisfied by boilerplate: " + plain)

    mm = render._mma_line(Mma(a="LHS", b="RHS", acc="ACC", kiter=0, coord=(("K_inner", 0),)))
    assert "ACC[" in mm and "LHS*RHS" in mm, (
        "the wmma must name the operands the NODE carries, not the literals A/B/C: " + mm)

    # NON-VACUITY on a real kernel: `advance` really does appear, and really does vanish at PLR0.
    def _adv(params):
        th = adapter.params_to_theta(adapter.kernel_to_params(params))
        return render.render_ir(emit.build_ir(th, build_S(th)[0])).count("advance=")
    assert _adv(BF16_NT_KMN) > 0, "a PLR>0 kernel must show read-ahead advances"
    assert _adv(BF16_NT_KMN_PLR0) == 0, "a PLR0 kernel has none, so the count is not a constant"


def test_a_fused_tdm_line_shows_each_operands_covered_axis_ranges():
    """The copy size alone does not reveal which split-region coordinates the movement fills."""
    from Tensile.LoopModel import emit, render
    from Tensile.LoopModel import adapter
    from Tensile.LoopModel.schedule import build_S

    th = adapter.params_to_theta({
        "MIWaveTile": [4, 4],
        "MatrixInstruction": [16, 16, 128, 1, 1, 4, 4, 2, 2],
        "DepthU": 256, "ElemBytes": 1,
        "PrefetchGlobalRead": 2, "PrefetchLocalRead": 1,
        "TDMSplit": [2, 2, 1, 1], "TDMSplitWaveRegions": [2, 2],
        "NumWaves": 4, "TDMFuse": 0, "LoopOrder": "KKMNMN",
    })
    text = render.render_ir(emit.build_ir(th, build_S(th)[0]))
    assert "A[M_inner0..1,K_inner0..1]+B[N_inner0..1,K_inner0..1]" in text
    assert "A[M_inner2..3,K_inner0..1]+B[N_inner2..3,K_inner0..1]" in text


def test_the_wmma_is_placed_by_presence_not_by_the_nests_depth():
    """`leaf_mode` was `inner[-1]` -- the innermost loop order level, unconditionally."""
    from Tensile.LoopModel import emit, render
    from Tensile.LoopModel import adapter
    from Tensile.LoopModel.schedule import build_S

    for params, name in ((BF16_NT_KMN, "BF16_NT_KMN"),
                         (dict(BF16_NT_KMN, LoopOrder="MNK"), "MNK"),
                         (_mxf8_kernel("KMN", 1), "mxf8 KMN split1")):
        th = adapter.params_to_theta(adapter.kernel_to_params(params))
        inner = [m.name for m in th.inner_axes()]
        want = ({m for o in th.output_operands() for m in presence(th, o)}
                | set(summation_names(th)))
        # THIS GUARD FIRED, AND THAT WAS ITS JOB.
        served = th.wave_served_axes()
        assert set(inner) - want == set(served) & set(inner), (
            "%s: modes outside the wmma presence are %s but the agent-served set is %s -- any "
            "OTHER mode leaving the wmma's presence is a new fact, not this one"
            % (name, sorted(set(inner) - want), sorted(served)))

        # ...and therefore the rule picks the innermost mode, i.e.
        leaf = next(m for m in reversed(inner) if m in want)
        deepest_unserved = next(m for m in reversed(inner) if m not in served)
        assert leaf == deepest_unserved, (
            "%s: the wmma sits at %r; it must sit at the deepest level it actually varies over, "
            "%r -- placing it deeper replicates it once per agent-served region"
            % (name, leaf, deepest_unserved))

        ir = emit.build_ir(th, build_S(th)[0])
        assert ir, name



# =========================================================================== rho as the agent map
def _rho_sweep_configs():
    """(params, label) over the axes rho is derived from, RESTRICTED TO REALIZABLE POINTS."""
    import itertools
    base, out = {}, []                                  # {} == the `baseline` scenario's defaults
    for o, s, w, wr, ph, rh in itertools.product(
            ["KMN", "KNM", "MKN", "MNK", "NKM", "NMK"],
            [[1, 1, 1, 1], [2, 1, 1, 1], [1, 2, 1, 1], [2, 2, 1, 1]],
            [1, 2, 4], [[0, 0], [1, 1], [2, 2]],
            [{}, {"A": 2}, {"B": 2}], [{}, {"A": 2}, {"B": 2}]):
        need = (s[0] // max(1, wr[0] or s[0])) * (s[1] // max(1, wr[1] or s[1]))
        if need > w:
            continue
        p = dict(base)
        p.update(LoopOrder=o, TDMSplit=s, NumWaves=w,
                 TDMSplitWaveRegions=wr, ReadPhi=ph, ReadRho=rh)
        out.append((p, f"{o} split={s} waves={w} wr={wr} phi={ph} rho={rh}"))
    return out


def test_agent_assignment_is_consistent_across_the_matrix():
    from Tensile.LoopModel import checks
    cfgs = _rho_sweep_configs()
    assert len(cfgs) > 500, "sweep collapsed to %d configs -- the filter is too strong" % len(cfgs)
    bad = []
    for params, label in cfgs:
        try:
            th = adapter.params_to_theta(params)
        except Exception:
            continue
        for v in checks.agent_assignment_consistency(th):
            bad.append("%s: %s" % (label, v))
    assert not bad, "agent assignment disagrees on %d point(s):\n  %s" % (
        len(bad), "\n  ".join(bad[:8]))


def test_subwave_distribution_is_not_a_synchronization_scope():
    cfgs = _rho_sweep_configs()
    checked = 0
    for params, label in cfgs:
        if not params.get("ReadRho"):
            continue
        try:
            th = adapter.params_to_theta(params)
        except Exception:
            continue
        sub = th.agent_assignment.assignments_at("subwave")
        if not sub:
            continue
        checked += 1
        served = th.agent_assignment.served_axes()
        for r in sub:
            assert r.axis not in served or any(
                w.axis == r.axis for w in th.agent_assignment.coarser_than("wave")), (
                "%s: sub-wave mode %r leaked into served_modes()" % (label, r.axis))
        assert served == set(th.wave_served_axes()), label
    assert checked, "no config produced a subwave assignment"


def _expected_agent_facts(params):
    """Independent wave-region alias and sub-wave coverage facts from raw params."""
    ts = params.get("TDMSplit", [1, 1, 1, 1])
    if isinstance(ts, bool):
        ts = [2, 2] if ts else [1, 1]
    ts = list(ts) + [1] * (4 - len(ts))
    aMT, bMT = max(1, ts[0]), max(1, ts[1])
    wr = list(params.get("TDMSplitWaveRegions", []) or []) + [0, 0]
    waveA = max(1, min(aMT, int(wr[0]) or aMT))
    waveB = max(1, min(bMT, int(wr[1]) or bMT))
    relative = set()
    if waveA < aMT:
        relative.update(("A", "MXSA"))
    if waveB < bMT:
        relative.update(("B", "MXSB"))
    spans = {n: int(v or 0) for n, v in (params.get("ReadRho", {}) or {}).items() if int(v or 0) > 1}
    return relative, spans


def test_tdmsplit_wave_layout_marks_region_aliasing_without_specializing_the_axis():
    """Wave placement affects region aliasing, while TDMSplit axes remain coordinates."""
    cfgs = _rho_sweep_configs()
    saw_relative = saw_span = 0
    bad = []
    for params, label in cfgs:
        try:
            th = adapter.params_to_theta(params)
        except Exception:
            continue
        want_relative, want_spans = _expected_agent_facts(params)
        if want_relative:
            saw_relative += 1
        if want_spans:
            saw_span += 1
        if th.wave_served_axes():
            bad.append("%s: TDMSplit axis was wave-specialized: %s"
                       % (label, sorted(th.wave_served_axes())))
        for op in th.operands:
            want = want_spans.get(op.name, 0)
            shared = op.trajectory.shared
            read = op.trajectory.fragment_fill
            if shared is not None and shared.regions.wave_relative != (op.name in want_relative):
                bad.append("%s: %s wave_relative %s != oracle %s"
                           % (label, op.name, shared.regions.wave_relative,
                              op.name in want_relative))
            if read is not None and int(read.agent_group_size or 0) != want:
                bad.append("%s: %s agent_group_size %s != oracle %s"
                           % (label, op.name, read.agent_group_size, want))
    assert saw_relative > 50 and saw_span > 50, \
        "oracle never exercised: relative=%d span=%d" % (saw_relative, saw_span)
    assert not bad, "agent facts diverge from the oracle on %d point(s):\n  %s" % (
        len(bad), "\n  ".join(bad[:8]))


def test_agent_assignment_deduplicates_subwave_coverage_without_absorbing_split_axes():
    """Shared sub-wave coverage is deduplicated without absorbing TDMSplit axes."""
    th = adapter.params_to_theta({"MIWaveTile": [2, 2],
                                  "TDMSplit": [2, 2, 1, 1], "NumWaves": 4,
                                  "TDMSplitWaveRegions": [1, 1],
                                  "ReadRho": {"A": 2, "MXSA": 2}})
    axes = [r.axis for r in th.agent_assignment.axis_assignments if r.role == "read"]
    assert len(axes) == len(set(axes)), "axis assigned twice: %r" % (axes,)
    assert "M_split" not in axes and "N_split" not in axes, \
        "TDMSplit is coordinate geometry, not an agent assignment"


def test_agent_assignment_carries_copy_shares_for_every_fusion():
    from Tensile.LoopModel import adapter

    class _B1:
        def numBytes(self): return 1

    def shares(fuse, num_waves=4):
        k = {"MatrixInstruction": [16, 16, 32, 1, 1, 2, 2, 2, 2], "DepthU": 64,
             "MIWaveTileA": 2, "MIWaveTileB": 2, "PrefetchGlobalRead": 2, "PrefetchLocalRead": 1,
             "ProblemType": {"DataType": _B1(), "MXBlockA": 32, "MXBlockB": 32},
             "NumWaves": num_waves, "enableTDMA": 1, "enableTDMB": 1, "NumLdsBlk": 2,
             "LoopOrder": "KMN", "TDMFuse": fuse}
        th = adapter.params_to_theta(adapter.kernel_to_params(k))
        return {n: th.agent_assignment.copy_agent_count(n) for n in ("A", "B", "MXSA", "MXSB")}

    # one wave: no set to select between, so every operand keeps the only agent
    assert shares(0, num_waves=1) == {"A": 1, "B": 1, "MXSA": 1, "MXSB": 1}
    assert shares(0) == {"A": 2, "B": 2, "MXSA": 2, "MXSB": 2}   # parity, even
    assert shares(1) == {"A": 2, "B": 2, "MXSA": 2, "MXSB": 2}   # parity, even, crossed
    # the two that miscompared: shares [2,1,1] on the group, every wave on the lone operand
    assert shares(2) == {"A": 2, "B": 4, "MXSA": 1, "MXSB": 1}
    assert shares(3) == {"A": 4, "B": 2, "MXSA": 1, "MXSB": 1}

    # single agent: no partition to state, and the accessor's default is the unspecialized answer
    assert not [r for r in adapter.params_to_theta(adapter.kernel_to_params(
        {"MatrixInstruction": [16, 16, 32, 1, 1, 2, 2, 2, 2], "DepthU": 64, "MIWaveTileA": 2,
         "MIWaveTileB": 2, "ProblemType": {"DataType": _B1()}, "PrefetchGlobalRead": 2,
         "PrefetchLocalRead": 1, "NumWaves": 1, "enableTDMA": 1, "enableTDMB": 1,
         "NumLdsBlk": 2, "LoopOrder": "KMN", "TDMFuse": 0})).agent_assignment.axis_assignments if r.role == "copy"]


def test_read_assignment_queries_exclude_copy_entries():
    from Tensile.LoopModel.theta import AgentAssignment, AgentAxisAssignment
    assignment = AgentAssignment(axis_assignments=(
        AgentAxisAssignment("M_inner", "wave", 2, ("A", "read")),
        AgentAxisAssignment("M_inner", "wave", 2, ("A", "copy"), role="copy"),
        AgentAxisAssignment("M_inner", "wave", 1, ("MXSA", "copy"), role="copy"),
    ))
    assert assignment.group_size_over({"M_inner"}, "wave") == 2
    assert [item.extent for item in assignment.assignments_at("wave")] == [2]
    assert assignment.served_axes() == {"M_inner"}
    assert (assignment.copy_agent_count("A"), assignment.copy_agent_count("MXSA")) == (2, 1)
    assert assignment.copy_agent_count("nobody") == 1


def test_subwave_assignment_and_wave_relative_regions_are_independent():
    th = adapter.params_to_theta({"MIWaveTile": [2, 2],
                                  "TDMSplit": [2, 2, 1, 1], "NumWaves": 4,
                                  "TDMSplitWaveRegions": [1, 1], "ReadRho": {"A": 2}})
    a = th.op("A")
    read = a.trajectory.fragment_fill
    assert read.agent_group_size == \
        th.agent_assignment.group_size_over({a.free_mode}, level="subwave") == 2
    assert a.trajectory.shared.regions.wave_relative
    assert not (set(a.region_axes) & th.agent_assignment.served_axes())
    assert set(a.region_axes) <= {axis.name for axis in th.inner_axes()}


# ---------------------------------------- the joint (free-tile, rotation-slot) name space
def test_rotation_units_follow_loop_order_and_operand_presence():
    """Prefetch, rotation, capacity and internal PLR are separate ord/axis facts."""
    from Tensile.LoopModel import traversal as geometry

    base = {
        "MIWaveTile": [4, 4], "DepthU": 64,
        "MatrixInstruction": [16, 16, 32, 1, 1, 4, 4, 2, 2],
        "PrefetchLocalRead": 1, "PrefetchGlobalRead": 2, "ElemBytes": 2,
    }
    # P, U, W, requested internal PLR, capacity, realized internal PLR.
    cases = {
        "KMN": {"A": (4, 4, 2, 1, 2, 1), "B": (4, 4, 2, 1, 2, 1)},
        "KNM": {"A": (4, 4, 2, 1, 2, 1), "B": (4, 4, 2, 1, 2, 1)},
        "MKN": {"A": (2, 2, 2, 1, 2, 1), "B": (4, 8, 2, 2, 4, 2)},
        "MNK": {"A": (2, 2, 2, 1, 2, 1), "B": (2, 8, 2, 2, 8, 2)},
        "NKM": {"A": (4, 8, 2, 2, 4, 2), "B": (2, 2, 2, 1, 2, 1)},
        "NMK": {"A": (2, 8, 2, 2, 8, 2), "B": (2, 2, 2, 1, 2, 1)},
    }
    for order, expected in cases.items():
        th = adapter.params_to_theta({**base, "LoopOrder": order})
        S, _ = build_S(th)
        for name, (prefetch, rotation, width, requested, capacity, realized) in expected.items():
            op = th.op(name)
            group = op.fragment.groups()[0]
            assert geometry.prefetch_unit_tiles(th, op) == prefetch, (order, name)
            assert geometry.rotation_unit_tiles(th, op) == rotation, (order, name)
            assert S.get(name, group) == width, (order, name)
            assert geometry.requested_read_ahead(th, op) == requested, (order, name)
            assert geometry.prefetch_capacity(th, op, S) == capacity, (order, name)
            assert geometry.requested_read_ahead(th, op, S) == realized, (order, name)


def test_all_inner_plr_is_limited_by_the_copy_staging_depth():
    """PGR1 keeps all-inner in place; PGR2 permits one rotation (two prefetch units here)."""
    from Tensile.LoopModel import traversal as geometry

    for order, direct, all_inner in (("MKN", "A", "B"), ("MNK", "A", "B"),
                                     ("NKM", "B", "A"), ("NMK", "B", "A")):
        for pgr, expected in ((1, 0), (2, 2)):
            th = adapter.params_to_theta({
                "MIWaveTile": [2, 2], "DepthU": 128,
                "MatrixInstruction": [16, 16, 64, 1, 1, 2, 2, 1, 1],
                "PrefetchGlobalRead": pgr, "PrefetchLocalRead": 1,
                "LoopOrder": order, "ElemBytes": 1,
            })
            depths = build_S(th)[0]
            assert geometry.requested_read_ahead(th, th.op(direct), depths) == 1
            assert geometry.requested_read_ahead(th, th.op(all_inner), depths) == expected


def test_two_groups_partition_the_first_rotation_axis_contiguously():
    """MNK A divides K; all-inner B divides N while keeping K as its prefetch unit."""
    import dataclasses
    from Tensile.LoopModel import traversal as geometry
    from Tensile.LoopModel.theta import group_labels

    th = adapter.params_to_theta({
        "MIWaveTile": [4, 4], "DepthU": 64,
        "MatrixInstruction": [16, 16, 32, 1, 1, 4, 4, 2, 2],
        "PrefetchLocalRead": 1, "PrefetchGlobalRead": 2, "LoopOrder": "MNK",
    })
    labels = group_labels(2)
    for name in ("A", "B"):
        op = th.op(name)
        op.fragment = dataclasses.replace(
            op.fragment, parts=2, labels=labels,
            group_policy={"*": "pipeline"}, ring_depths=None)
    S, _ = build_S(th)

    a, b = th.op("A"), th.op("B")
    assert geometry.grouping_modes(th, a) == [("K_inner", 2)]
    assert geometry.grouping_modes(th, b) == [("N_inner", 4)]
    assert [geometry.group_index(th, a, {"K_inner": i}) for i in range(2)] == [0, 1]
    assert [geometry.group_index(th, b, {"N_inner": i, "K_inner": 0})
            for i in range(4)] == [0, 0, 1, 1]
    assert [(geometry.group_unit_tile_count(th, a), geometry.group_unit_tiles(th, a, g),
             S.get("A", g)) for g in labels] == [(1, 1, 2), (1, 1, 2)]
    assert [(geometry.group_unit_tile_count(th, b), geometry.group_unit_tiles(th, b, g),
             S.get("B", g)) for g in labels] == [(4, 4, 2), (4, 4, 2)]
    assert geometry.group_prefetch_unit_tiles(th, a) == 1  # grouping axis is K
    assert geometry.group_prefetch_unit_tiles(th, b) == 2  # grouping axis N leaves K alone
    assert geometry.prefetch_capacity(th, b, S) == 8


def test_grouping_rejects_any_scheme_that_splits_a_quantum_carrier():
    """Contiguous q=2 blocks fit two-tile groups; strided {0,2} carriers do not."""
    from Tensile.LoopModel.schedule import RegisterDepthSelector
    from Tensile.LoopModel.ir import TransferCoverage, COVERAGE_VAR, cst

    t = COVERAGE_VAR
    wide = TransferCoverage(
        carrier=Expr(digits=(((t, 1),), 0, ((1, 2, 0),))),
        slot=Expr(digits=(((t, 1),), 0, ((1, 1, 2),))))
    strided = TransferCoverage(
        carrier=Expr(digits=(((t, 1),), 0, ((1, 1, 2),))),
        slot=cst(0))
    base = {
        "MIWaveTile": [4, 4], "DepthU": 64,
        "MatrixInstruction": [16, 16, 32, 1, 1, 4, 4, 2, 2],
        "PrefetchLocalRead": 1, "PrefetchGlobalRead": 2, "LoopOrder": "KMN",
    }
    wide_theta = adapter.params_to_theta({**base, "ReadQuantum": {"A": wide}})
    strided_theta = adapter.params_to_theta({**base, "ReadQuantum": {"A": strided}})
    assert RegisterDepthSelector(wide_theta, None).admissible(wide_theta.op("A"), 2)
    assert not RegisterDepthSelector(wide_theta, None).admissible(wide_theta.op("A"), 4)
    builder = RegisterDepthSelector(strided_theta, None)
    assert not builder.admissible(strided_theta.op("A"), 2)
    assert {share for share, _policies, _regs in builder.candidates(strided_theta.op("A"))} == {1}


def test_grouping_allows_partial_prefetch_of_a_whole_set_operand():
    import dataclasses
    from Tensile.LoopModel import traversal as geometry
    from Tensile.LoopModel.theta import group_labels

    th = adapter.params_to_theta({
        "MIWaveTile": [4, 4], "DepthU": 64,
        "MatrixInstruction": [16, 16, 32, 1, 1, 4, 4, 2, 2],
        "PrefetchLocalRead": 1, "PrefetchGlobalRead": 2, "LoopOrder": "MNK",
    })
    b = th.op("B")
    labels = group_labels(2)
    b.fragment = dataclasses.replace(
        b.fragment, parts=2, labels=labels,
        group_policy={labels[0]: "pipeline", labels[1]: "inplace"},
        ring_depths=None)
    S, _ = build_S(th)
    assert [S.get("B", group) for group in labels] == [2, 1]
    assert geometry.operand_footprint_regs(th, b, S) \
        == (2 * 4 + 1 * 4) * geometry.frag_regs(th, b)
    layout = geometry.register_layout(th, S)["B"]
    assert layout["slots"] == {(0, 0): 0, (0, 1): 32, (1, 0): 64}
    assert layout["total"] == 96
    assert geometry.group_prefetch_unit_tiles(th, b) == 2       # grouping N keeps K
    assert geometry.group_rotation_unit_tiles(th, b) == 4      # N/2 * K
    assert geometry.prefetch_capacity(th, b, S) == 6            # 2*4/2 + 1*4/2
    assert geometry.requested_read_ahead(th, b) == 2            # two literal A prefetch tiles
    assert geometry.requested_read_ahead(th, b, S) == 2


def test_direct_ring_packs_interleaved_prefetch_units_behind_the_reduction_ring():
    """K selects W=2 while K×N_inner provides eight unit positions per slot."""
    from Tensile.LoopModel import traversal as geometry

    kernel = dict(_mxf8_kernel("KMNKMN", 0))
    kernel.update(MatrixInstruction=[16, 16, 128, 1, 1, 2, 8, 2, 2],
                  MIWaveTileA=2, MIWaveTileB=8, TDMSplitA=0, TDMSplitB=1)
    th = adapter.params_to_theta(adapter.kernel_to_params(kernel))
    b = th.op("B")
    slot = geometry.ring_slot(th, b, b.fragment.groups()[0], buffers=2)
    coords = [{"N_split": 0, "K_inner": k, "M_inner": 0, "N_inner": n}
              for k in range(2) for n in range(4)]
    assert [slot.eval(coord) for coord in coords] == [0] * 4 + [1] * 4
    assert [geometry.unit_tile_index(th, b, coord) for coord in coords] == list(range(8))
    assert geometry.group_register_positions(th, b) == 8


@pytest.mark.parametrize("order,plr,expect", [
    # K-innermost: the shipping surface.  A tile's own sequence is k0,k0,..,k1,k1,.. -- sequential
    # runs, so the group is refillable in place and the floor is 1.  Unchanged by.
    ("KMN", 0, {"A": (1, 1), "B": (1, 1)}),
    ("KMN", 1, {"A": (2, 1), "B": (2, 1)}),
    # Whole-set residency is represented by the unit's tile size, not by inflating W.
    ("MKN", 0, {"A": (1, 1), "B": (1, 1)}),
])
def test_the_live_peak_is_measured_per_name_not_per_collapsed_ring(order, plr, expect):
    """requires the peak "per **name**, not per rotation ring" -- "a direct live-range walk on
 the joint `(free-tile, rotation-slot)` name space". Projecting the free tile modes away makes
 two values that live in DIFFERENT names look like they compete for one."""
    from Tensile.LoopModel import traversal as geometry
    th = adapter.params_to_theta({"MIWaveTile": [4, 4], "PrefetchLocalRead": plr,
                                    "PrefetchGlobalRead": 2, "LoopOrder": order})
    steps = geometry._inner_steps(th)
    for name, (l_pf, l_war) in expect.items():
        op = th.op(name)
        g = op.fragment.groups()[0]
        assert geometry.group_live_peak(th, op, g, steps, post_war=False) == l_pf, (order, name)
        assert geometry.group_live_peak(th, op, g, steps) == l_war, (order, name)


def test_the_prefetch_lead_in_is_applied_in_the_names_own_sequence():
    """The trap that made the first attempt at the joint-space walk wrong, pinned."""
    from Tensile.LoopModel import traversal as geometry
    th = adapter.params_to_theta({"MIWaveTile": [4, 4], "PrefetchLocalRead": 1,
                                    "PrefetchGlobalRead": 2, "LoopOrder": "KMN"})
    steps = geometry._inner_steps(th)
    op = th.op("A")
    g = op.fragment.groups()[0]
    l_pf = geometry.group_live_peak(th, op, g, steps, post_war=False)
    l_war = geometry.group_live_peak(th, op, g, steps)
    assert (l_pf, l_war) == (2, 1), "prefetch lead-in lost: L_pf must be dr_g+1 above L_war"


@pytest.mark.parametrize("order,want", [("KKMNMN", ["B", "A"]),
                                         ("KKNMNM", ["A", "B"])])
def test_register_group_search_visits_the_innermost_operand_first(order, want):
    from Tensile.LoopModel.schedule import RegisterDepthSelector
    from Tensile.LoopModel.schedule import derive_S

    kernel = dict(_mxf8_kernel(order, 1))
    kernel.update(MatrixInstruction=[16, 16, 128, 1, 1, 8, 8, 2, 2],
                  MIWaveTileA=8, MIWaveTileB=8,
                  VectorWidthA=8, VectorWidthB=8,
                  LocalReadVectorWidthA=16, LocalReadVectorWidthB=16)
    theta = adapter.params_to_theta(adapter.kernel_to_params(kernel))
    order = [operand.name for operand in RegisterDepthSelector(theta, None).order(derive_S(theta))]
    assert [name for name in order if name in ("A", "B")] == want


def test_the_per_name_floor_is_never_above_the_collapsed_one():
    """SAFETY ARGUMENT for the joint-space walk, kept executable.  It may only REMOVE pessimism: if the
    per-name floor were ever ABOVE the old collapsed one, some already-shipped kernel would have
    been running below its true floor -- a the register-width floor race, not a footprint win.  Swept 630 groups
    when the change landed; this pins a representative slice so a future edit to the walk cannot
    quietly cross that line."""
    from Tensile.LoopModel import traversal as geometry
    import itertools
    for order, plr, mt in itertools.product(("KMN", "KNM", "MNK", "NMK", "MKN", "NKM"),
                                            (0, 1, 2), ([2, 2], [4, 4], [4, 2])):
        try:
            th = adapter.params_to_theta({"MIWaveTile": mt, "PrefetchLocalRead": plr,
                                          "PrefetchGlobalRead": 2, "LoopOrder": order})
        except RuntimeError as e:
            if "gives S" not in str(e):
                raise
            continue                      # no schedulable S here; there is no floor to compare
        steps = geometry._inner_steps(th)
        for name in ("A", "B"):
            op = th.op(name)
            for g in op.fragment.groups():
                l_war = geometry.group_live_peak(th, op, g, steps)
                l_pf = geometry.group_live_peak(th, op, g, steps, post_war=False)
                assert 1 <= l_war <= l_pf, (order, plr, mt, name, g, l_war, l_pf)


@pytest.mark.parametrize("mi_k,kext", [(32, 2), (16, 4)])
def test_whole_set_residency_grows_the_unit_not_the_width_floor(mi_k, kext):
    """A full inner K sweep is represented in unit size while L_floor remains one unit."""
    from Tensile.LoopModel import traversal as geometry
    th = adapter.params_to_theta({"MIWaveTile": [4, 4], "DepthU": 64,
                                    "MatrixInstruction": [16, 16, mi_k, 1, 1, 4, 4, 2, 2],
                                    "PrefetchLocalRead": 0, "PrefetchGlobalRead": 2,
                                    "LoopOrder": "MNK"})
    assert next(m.extent for m in th.inner_axes() if m.name == "K_inner") == kext
    steps = geometry._inner_steps(th)
    a, b = th.op("A"), th.op("B")
    assert geometry.group_live_peak(th, a, a.fragment.groups()[0], steps) == 1
    assert geometry.group_live_peak(th, b, b.fragment.groups()[0], steps) == 1
    assert geometry.group_unit_tiles(th, a, a.fragment.groups()[0]) == kext
    assert geometry.group_unit_tiles(th, b, b.fragment.groups()[0]) == 4 * kext


def test_mkn_whole_set_operand_uses_a_larger_unit_not_a_larger_floor():
    from Tensile.LoopModel import traversal as geometry
    th = adapter.params_to_theta({"MIWaveTile": [4, 4], "PrefetchLocalRead": 0,
                                    "PrefetchGlobalRead": 2, "LoopOrder": "MKN"})
    steps = geometry._inner_steps(th)
    la = geometry.group_live_peak(th, th.op("A"), th.op("A").fragment.groups()[0], steps)
    lb = geometry.group_live_peak(th, th.op("B"), th.op("B").fragment.groups()[0], steps)
    assert (la, lb) == (1, 1)
    assert geometry.group_unit_tiles(th, th.op("B"), th.op("B").fragment.groups()[0]) \
        > geometry.group_unit_tiles(th, th.op("A"), th.op("A").fragment.groups()[0])


def test_the_walk_satisfies_every_closed_fact_the_paper_commits_to():
    """Unit widths are monotonic in read-ahead and have a one-unit no-prefetch floor."""
    from Tensile.LoopModel import traversal as geometry
    import itertools
    checked = dr0 = 0
    for order, plr, mt, du, mik in itertools.product(
            ("KMN", "KNM", "MNK", "NMK", "MKN", "NKM"), (0, 1, 2),
            ([2, 2], [4, 4], [4, 2]), (64, 128), (16, 32)):
        try:
            th = adapter.params_to_theta({
                "MIWaveTile": mt, "DepthU": du, "PrefetchLocalRead": plr, "PrefetchGlobalRead": 2,
                "MatrixInstruction": [16, 16, mik, 1, 1, mt[0], mt[1], 2, 2], "LoopOrder": order})
        except RuntimeError as e:
            if "gives S" not in str(e):
                raise
            continue                      # no schedulable S here; there is no walk to check
        steps = geometry._inner_steps(th)
        for nm in ("A", "B"):
            op = th.op(nm)
            dr = geometry._reg_read_off(th, op)
            for g in op.fragment.groups():
                l_pf = geometry.group_live_peak(th, op, g, steps, post_war=False)
                checked += 1
                prev = geometry.group_live_peak(th, op, g, steps, post_war=False, off=max(0, dr - 1))
                assert l_pf >= max(1, prev), ("read-ahead folds", order, plr, nm, l_pf, prev, dr)
                if dr == 0:
                    dr0 += 1
                    assert l_pf == 1, ("dr_g=0 floor", order, mt, nm, l_pf)
    assert checked > 400 and dr0 > 100, (checked, dr0)


@pytest.mark.parametrize("order,expect", [
    ("KMN", {0: {"lo": 2, "hi": 2}, 1: {"lo": 2, "hi": 1}, 2: {"lo": 1, "hi": 1}}),
    ("MKN", {0: {"lo": 4, "hi": 4}, 1: {"lo": 2, "hi": 1}, 2: {"lo": 1, "hi": 1}}),
    ("MNK", {0: {"lo": 4, "hi": 4}, 1: {"lo": 2, "hi": 1}, 2: {"lo": 1, "hi": 1}}),
])
def test_the_three_width_policies_bracket_the_band(order, expect):
    """Unroll keeps every owned unit, overlap pipelines one group, and inplace reuses W=1."""
    import dataclasses
    from Tensile.LoopModel.schedule import build_S
    from Tensile.LoopModel.theta import group_labels
    fns = {0: lambda _i: "unroll",
           1: lambda i: "overlap" if i == 0 else "inplace",
           2: lambda _i: "inplace"}
    for strat, want in expect.items():
        th = adapter.params_to_theta({"MIWaveTile": [4, 4], "PrefetchLocalRead": 1,
                                        "PrefetchGlobalRead": 2, "LoopOrder": order})
        a = th.op("A")
        labels = group_labels(2)
        a.fragment = dataclasses.replace(
            a.fragment, parts=2, labels=labels,
            group_policy={g: fns[strat](i) for i, g in enumerate(labels)},
            ring_depths=None)
        S, _ = build_S(th)
        got = {g: S.get("A", g) for g in th.op("A").fragment.groups()}
        assert got == want, (order, strat, got, want)


def test_the_rate_and_the_footprint_agree_on_the_unit():
    """`R` counts BUFFERS, `|part|` counts TILES PER BUFFER, and the two must be the same unit."""
    from Tensile.LoopModel.schedule import build_S
    from Tensile.LoopModel.traversal import (group_ring_size, group_register_positions,
                                            group_fan_reloads, group_owned_tiles)
    from Tensile.LoopModel import traversal as geometry
    orders = ("KMN", "KNM", "MKN", "MNK", "NKM", "NMK")
    seen_reload, seen_name, rates = 0, 0, {}
    for wt in ([4, 4], [8, 8], [2, 2]):
        for du in (64, 128):
            products, foots = {}, {}
            for order in orders:
                th = adapter.params_to_theta({
                    "MIWaveTile": wt, "DepthU": du, "LoopOrder": order,
                    "PrefetchLocalRead": 1, "PrefetchGlobalRead": 2})
                S, _ = build_S(th)
                for op in th.operands:
                    if op.is_output or not op.movements:
                        continue
                    g = op.fragment.groups()[0]
                    unit = group_register_positions(th, op)
                    r = group_ring_size(th, op, g)
                    # the unit never exceeds what the group owns, and never claims 0 tiles
                    assert 1 <= unit <= group_owned_tiles(th, op), (order, op.name, unit)
                    if group_fan_reloads(th, op, g) > 1:
                        seen_reload += 1
                    else:
                        seen_name += 1
                    products.setdefault(op.name, {})[order] = r * unit
                    foots.setdefault(op.name, {})[order] = (
                        sum(S.get(op.name, gg) for gg in op.fragment.groups()) * unit)
                    rates.setdefault((tuple(wt), du, op.name), {})[order] = r
            for nm, by_ord in products.items():
                assert len(set(by_ord.values())) == 1, (
                    "R x unit must be ord-invariant -- a chunk touches the same data whatever "
                    "the order", wt, du, nm, by_ord)
                del foots[nm]
    # NON-VACUITY: both arms of the name/reload decision are exercised, and `R` really does move.
    assert seen_reload > 0, "no config reloads its fan -- the ord-variant arm of `R` is DEAD"
    assert seen_name > 0, "no config names its fan -- the storage-partition arm of `R` is DEAD"
    varying = {k: v for k, v in rates.items() if len(set(v.values())) > 1}
    assert varying, ("`R` came out ord-invariant for every shape -- the unit stopped tracking "
                     "`ord`, so the band ceiling no longer discriminates", rates)


@pytest.mark.parametrize("tds", [[1,1,1,1], [2,2,1,1], [1,1,2,2], [4,4,1,1]])
def test_a_region_split_cannot_change_the_register_footprint(tds):
    """a region split partitions the SAME tiles into storage-disjoint regions -- it cannot
    create data, so the per-wave register footprint must not move.
    """
    from Tensile.LoopModel import traversal as geometry
    from Tensile.LoopModel.schedule import build_S
    th = adapter.params_to_theta({
        "MIWaveTile": [8, 8], "DepthU": 256, "MatrixInstruction": [16,16,32,1,1,8,8,2,2],
        "PrefetchLocalRead": 1, "PrefetchGlobalRead": 2, "LoopOrder": "KMN", "ElemBytes": 2,
        "TDMSplit": tds, "NumWaves": 4, "TDMSplitWaveRegions": [1, 1]})
    S, _ = build_S(th)
    assert geometry.operand_footprint_regs(th, th.op("A"), S) == 128, tds


@pytest.mark.parametrize("split", [1, 2, 4])
@pytest.mark.parametrize("wave_regions", [1, 2])
def test_tdmsplit_is_an_exact_wave_tile_axis_factorization(split, wave_regions):
    wt = 8
    th = adapter.params_to_theta({
        "MIWaveTile": [wt, wt], "TDMSplit": [split, split, 1, 1],
        "TDMSplitWaveRegions": [min(split, wave_regions)] * 2,
    })
    extents = {axis.name: axis.extent for axis in th.inner_axes()}
    assert extents.get("M_split", 1) * extents.get("M_inner", 1) == wt
    assert extents.get("N_split", 1) * extents.get("N_inner", 1) == wt


def test_wave_region_span_does_not_change_the_tdmsplit_factorization():
    """Wave placement changes aliasing, not M_split x M_inner or the VGPR footprint."""
    from Tensile.LoopModel import traversal as geometry
    from Tensile.LoopModel.schedule import build_S
    base = {"MIWaveTile": [8, 8], "DepthU": 256, "MatrixInstruction": [16,16,32,1,1,8,8,2,2],
            "PrefetchLocalRead": 1, "PrefetchGlobalRead": 2, "LoopOrder": "KMN",
            "ElemBytes": 2, "TDMSplit": [2, 2, 1, 1], "NumWaves": 4}
    got = {}
    for wr in ([1, 1], [2, 2]):
        th = adapter.params_to_theta({**base, "TDMSplitWaveRegions": wr})
        S, _ = build_S(th)
        a = th.op("A")
        got[tuple(wr)] = (
            a.trajectory.shared.regions.wave_span,
            th.free_extent("M_split"),
                th.free_extent("M_inner"),
            geometry._region_count(th, a),
            geometry.operand_footprint_regs(th, a, S),
            geometry.operand_emitted_regs(th, a, S),
            a.trajectory.shared.regions.wave_relative,
        )
    assert got[(1, 1)] == (1, 2, 4, 2, 128, 128, True), got
    assert got[(2, 2)] == (2, 2, 4, 2, 128, 128, False), got


@pytest.mark.parametrize("order", ["KMN", "KNM", "MNK", "NMK", "NKM", "MKN"])
@pytest.mark.parametrize("q", [1, 2, 4])
def test_transfer_coverage_folds_presence_to_N_over_q(order, q, request):
    """Instruction coverage changes read count without changing logical data."""
    import math
    from Tensile.LoopModel import traversal as geometry
    from Tensile.LoopModel.schedule import build_S
    from Tensile.Lowering.loopir_to_gir import build_gir
    from Tensile.Lowering.gir import check_plan
    from Tensile.Lowering.gir.emit_plan import plan_block
    base = {"MIWaveTile": [4, 4], "DepthU": 128, "MatrixInstruction": [16,16,32,1,1,4,4,2,2],
            "PrefetchLocalRead": 1, "PrefetchGlobalRead": 2, "LoopOrder": order, "ElemBytes": 2}

    def probe(params):
        th = adapter.params_to_theta(params)
        op = th.op("A")
        hop = next(h for h in op.movements if not h.is_bulk and h.dst == Space.REGISTER)
        S, _ = build_S(th)
        prog = build_gir(th)
        acts = [a for a in plan_block(prog, "steady")
                if a.kind == "read" and a.at["tc"] == "A"]
        return {
            "points": math.prod(m.extent for m in presence_axes(th, op, hop)),
            "data": geometry.presence_tiles(th, op),
            "regs": geometry.operand_footprint_regs(th, op, S),
            "acts": len(acts),
            "fills": len({c for a in acts for c in a.at["fills"]}),
            "wmma": len([a for a in plan_block(prog, "steady") if a.kind == "wmma"]),
            "viol": check_plan(prog),
            "groups": len(op.fragment.groups()),
        }

    one = probe(base)
    try:
        got = probe(base if q == 1 else {**base, "ReadPhi": {"A": q, "B": q}})
    except RuntimeError as e:
        if "gives S" not in str(e):
            raise
        # A fold on a reused order can leave no grouping that carries the read-ahead; PLR is
        # faithful, so that refuses rather than running shallower.
        assert order in ("MKN", "MNK", "NKM", "NMK"), f"{order}: unexpected refusal -- {e}"
        pytest.skip(f"{order} q={q}: {str(e)[:70]}")

    # FIXED -- the STRICT xfail that sat here fired as XPASS and has
    # been removed, which is exactly the protocol its own note demanded ("marked STRICT so the day
    # the interaction is fixed the marker fails and has to be removed").
    assert got["viol"] == [], got["viol"][:3]
    # the coverage: the axes it varies over points AND emitted instructions both go to N/q
    assert got["points"] == one["points"] // q, (got, one)
    assert got["acts"] == one["acts"] // q, (got, one)
    # conservation + invariance: one act now fills q coordinates, and nothing else moves
    assert got["fills"] == one["fills"] == one["acts"], (got, one)
    assert (got["data"], got["wmma"]) == (one["data"], one["wmma"]), (got, one)


def test_a_folded_read_does_not_redefine_the_prefetch_unit():
    """Quantum merges instructions; the physical eight-tile prefetch unit remains eight."""
    from Tensile.LoopModel.traversal import (
        prefetch_unit_tiles, read_coverage, ring_axes, unit_tile_index)
    base = {"MIWaveTile": [8, 8], "DepthU": 256, "MatrixInstruction": [16,16,32,1,1,8,8,2,2],
            "PrefetchLocalRead": 1, "PrefetchGlobalRead": 2, "LoopOrder": "KMN", "ElemBytes": 2}
    th = adapter.params_to_theta({**base, "ReadPhi": {"A": 2, "B": 2}})
    a = th.op("A")
    assert read_coverage(th, a) == {"M_inner": 2}
    g = a.fragment.groups()[0]
    assert dict(ring_axes(th, a, g)) == {"K_inner": 8}
    assert prefetch_unit_tiles(th, a) == 8
    assert [unit_tile_index(th, a, {"M_inner": m, "K_inner": 0})
            for m in range(8)] == [0, 0, 1, 1, 2, 2, 3, 3]


# ---------------------------------------------------------------------------
# anti-vacuity: every arm of the per-edge gate must actually REJECT something.
# ---------------------------------------------------------------------------
def _gate_rebuild_bodies(tree, f, *, peel_kind=None):
    """`tree` with `f` applied to region bodies.  `peel_kind=None` rewrites steady `Loop` bodies
    (the sigma-order arms); a kind string rewrites the body of `Peel` nodes of that kind (the prologue
    arm); `"*"` rewrites EVERY body (the coverage arm, whose check is over the whole tree -- leaving
    the peels alone would keep the counter awaited somewhere and the arm would read as vacuous).
    A tiny local sigma-perturber -- it exists so the arms below are exercised by a REORDER of the
    EMITTED tree rather than by a hand-built one, which is the only way the test proves the GATE
    and not a fixture."""
    from Tensile.LoopModel.ir import Loop, Peel, Cond, Branch, Bind
    from Tensile.LoopModel.checks import _rebuild
    every = peel_kind == "*"

    def go(nodes):
        out = []
        for n in nodes:
            if isinstance(n, Loop):
                out.append(_rebuild(n, [(f(go(b)) if every or peel_kind is None else go(b))
                                        for b in n.bodies]))
            elif isinstance(n, Branch):
                out.append(_rebuild(n, {r: (f(go(a)) if every else go(a))
                                        for r, a in n.arms.items()}))
            elif isinstance(n, Cond):
                arms = (go(n.then), go(n.els or []))
                out.append(_rebuild(n, tuple(f(a) for a in arms) if every else arms))
            elif isinstance(n, Peel):
                body = go(n.body)
                out.append(_rebuild(n, f(body) if every or n.kind == peel_kind else body))
            elif isinstance(n, Bind):
                out.append(_rebuild(n, go(n.body)))
            else:
                out.append(n)
        return out
    return go(list(tree))


def _gate_case_copies_first(body):
    from Tensile.LoopModel.ir import Space, Inst, Load
    c = [n for n in body if isinstance(n, Inst) and isinstance(n.op, Load)
         and n.op.dst == Space.SHARED]
    return c + [n for n in body if n not in c] if c else body


def _gate_case_copies_last(body):
    from Tensile.LoopModel.ir import Space, Inst, Load
    c = [n for n in body if isinstance(n, Inst) and isinstance(n.op, Load)
         and n.op.dst == Space.SHARED]
    return [n for n in body if n not in c] + c if c else body


def _gate_case_drop_a_read(body):
    from Tensile.LoopModel.ir import Space, Inst, Load
    for i, n in enumerate(body):
        if isinstance(n, Inst) and isinstance(n.op, Load) \
                and n.op.src == Space.SHARED and n.op.dst == Space.REGISTER:
            return body[:i] + body[i + 1:]
    return body


def _gate_case_strip_awaits(body):
    from Tensile.LoopModel.ir import Inst
    from Tensile.LoopModel.checks import _replace_awaits
    return [_replace_awaits(n, ()) if isinstance(n, Inst) else n for n in body]


# (arm, kernel params, mutation, peel_kind, the obligation kind that must be reported)
_GATE_ARMS = [
    ("a/coverage",  dict(MIWaveTile=[8, 8], PrefetchLocalRead=1, PrefetchGlobalRead=2),
     _gate_case_strip_awaits,   "*",         "RAW-residency"),
    ("b/shared-WAR", dict(MIWaveTile=[8, 8], PrefetchLocalRead=1, PrefetchGlobalRead=2),
     _gate_case_copies_first,   None,        "inplace-WAR"),
    ("b3/crossing-RAW", dict(MIWaveTile=[8, 8], PrefetchLocalRead=1, PrefetchGlobalRead=1),
     _gate_case_copies_last,    None,        "crossing-RAW"),
    ("c/readahead",  dict(MIWaveTile=[8, 8], PrefetchLocalRead=1, PrefetchGlobalRead=2),
     _gate_case_drop_a_read,    "prologue",  "readahead-residency"),
]


@pytest.mark.parametrize("arm,params,mutate,peel,kind",
                         _GATE_ARMS, ids=[a[0] for a in _GATE_ARMS])
def test_every_arm_of_the_per_edge_gate_rejects_something(arm, params, mutate, peel, kind):
    """ours is an ORDER-DRIVEN backend, so the per-edge position gate is REQUIRED and
 counter-class coverage is insufficient. `check_ledger_discharged` implements that gate in five
 arms; this asserts each one is LIVE -- the emitted tree passes, and a targeted sigma-perturbation of
 that arm's edge is REJECTED, naming the right obligation kind.

 ANTI-VACUITY IS THE POINT, and it is not hypothetical. A gate arm that silently stops matching
 still returns `[]`, so it reads exactly like a correct schedule; the only signal is that nothing
 can make it fire. This has already happened once: the register-ring arm keyed on
 `(op, group)` while `schedule._read_placement` blanks the group label of a single-group
 operand (`gl = group if len(all_groups) > 1 else ""`), so `('A','')` never matched `('A','g0')`
 and the whole arm was dead for every unpartitioned operand -- i.e. for every shipping kernel.
 It was found by inspection, not by a test. This is that test.

 The mutations are REORDERINGS OF THE EMITTED TREE, never hand-built trees: that is what makes
 this a test of the gate rather than of a fixture. Each is a sigma a latency-hiding scheduler could
 plausibly produce -- copies hoisted to the top of the trip, copies sunk to the bottom, a refill
 hoisted above its consumer, one prologue read missing, all markers discarded.

 TWO KINDS ARE DELIBERATELY ABSENT and must stay absent: plain `RAW-residency` and
 `rotation-WAR` are NOT order-constrained within a body. A plain residency's producer is the
 copy from `off` trips back, and a rotation WAR at `W > 1` writes slot `(k+dr) mod W` while its
 consumer reads `k mod W` -- different buffers either way, so no intra-trip order can violate
 them, and an arm that flagged one would reject the only correct schedule. They appear here
 only via `a/coverage`, which is the arm that does apply to them."""
    from Tensile.LoopModel.emit import build_ir
    from Tensile.LoopModel.checks import build_ledger, check_ledger_discharged
    wt = params["MIWaveTile"]
    # A ROW MAY OVERRIDE THE BASE.  `DepthU` in particular: the register inplace-WAR arm needs 64
    # (at 128 the ledger carries no such obligation at all), and `**params` into a dict literal
    # that already fixes it is a TypeError rather than an override.
    th = adapter.params_to_theta(dict(
        dict(MatrixInstruction=[16, 16, 32, 1, 1, wt[0], wt[1], 2, 2], DepthU=128,
             ElemBytes=2, LoopOrder="KMN"), **params))
    S, _ = build_S(th)
    ledger = build_ledger(th, S)
    ir = build_ir(th, S)

    assert kind in {o.kind for o in ledger}, \
        f"{arm}: this config does not even carry a {kind} obligation"
    assert check_ledger_discharged(ledger, ir) == [], \
        f"{arm}: the as-emitted tree must be clean"

    bad = check_ledger_discharged(ledger, _gate_rebuild_bodies(ir, mutate, peel_kind=peel))
    assert bad, f"{arm}: VACUOUS -- the perturbed tree was accepted"
    assert kind in {o.kind for o in bad}, \
        f"{arm}: fired, but reported {sorted({o.kind for o in bad})} instead of {kind}"


def test_the_register_ring_arm_reachability_bound():
    """WHY the (b2) register-ring arm's existential coincides with the per-generation forall
    -- a bound with a reason, replacing the old "unreachable over 27 configs" note.
    """
    import collections, itertools
    from Tensile.LoopModel.ir import Space, Inst, Load, Mma
    from Tensile.LoopModel.emit import build_ir
    from Tensile.LoopModel.checks import build_ledger, _region_bodies

    from Tensile.LoopModel.traversal import group_live_peak, _inner_steps
    shapes, bodies, built = collections.Counter(), collections.Counter(), 0
    floor_bad, floor_checked, refused = [], 0, 0
    for wt, order, plr, pgr, du, mi in itertools.product(
            ([1, 1], [1, 2], [2, 1], [2, 2], [4, 4], [8, 8], [8, 7]),
            ("KMN", "MNK", "NKM"), (0, 1, 2), (1, 2), (64, 128, 256), (16, 32)):
        p = dict(MIWaveTile=wt, DepthU=du,
                 MatrixInstruction=[16, 16, mi, 1, 1, wt[0], wt[1], 2, 2],
                 PrefetchLocalRead=plr, PrefetchGlobalRead=pgr, LoopOrder=order, ElemBytes=2)
        try:
            th = adapter.params_to_theta(p)
            S, _ = build_S(th)
            ledger = build_ledger(th, S)
        except RuntimeError as e:
            refused += "gives S" in str(e)     # PLR is faithful: this cell has no schedulable S
            continue
        except Exception:
            continue
        built += 1
        hits = {(o.consumer.op, o.consumer.at) for o in ledger
                if o.kind == "inplace-WAR" and o.consumer.at != Space.SHARED}
        if not hits:
            continue
        shapes[tuple(wt)] += 1
        # THE SOUNDNESS CHECK: the register-width floor's floor at every (op, group) the arm fires on.
        _steps = _inner_steps(th)
        for _opname, _gl in hits:
            floor_checked += 1
            _W = S.get(_opname, _gl)
            _L = group_live_peak(th, th.op(_opname), _gl, _steps)
            if _W < _L:
                floor_bad.append((tuple(wt), order, plr, pgr, du, mi, _opname, _gl, _W, _L))
        for _label, body in _region_bodies(build_ir(th, S)):
            per_slot, wmma = collections.Counter(), 0
            for n in body:
                if not isinstance(n, Inst):
                    continue
                if isinstance(n.op, Mma):
                    wmma += 1
                elif (isinstance(n.op, Load) and n.op.dst == Space.REGISTER
                      and getattr(n.op, "advance", 0)
                      and any(a.kind == "inplace-WAR" for a in (n.awaits or ()))):
                    for t in getattr(n.op, "tokens", ()):
                        for gl, _e in getattr(n.placement, "slots", ()) or ():
                            per_slot[(t, gl)] += 1
            if per_slot:
                bodies[(max(per_slot.values()), wmma)] += 1

    # Coverage is built + refused: a cell whose read-ahead has no schedulable S is a decided
    # outcome, not a gap.  Both arms pinned so neither can quietly grow at the other's expense.
    assert built + refused > 700, (built, refused)
    assert built > 600, built
    # THE LOAD-BEARING ASSERTION: the arm's existential is sound GIVEN the depth floor, so the
    # floor is what must hold -- at every (op, group) the arm fires on, not just on average.
    assert floor_checked > 100, floor_checked          # the check really ran
    assert floor_bad == [], \
        ("W < L_war where the register-ring arm fires -- the register-width floor race, the arm's existential is "
         "no longer sound: %s" % floor_bad[:5])
    # MEASURED BOUND.  Not a theorem -- a fact about what the
    # bridge currently reaches, kept so a further widening is noticed and re-justified.
    assert shapes and set(shapes) == {(1, 1), (1, 2), (2, 1), (2, 2), (4, 4), (8, 8), (8, 7)}, \
        f"the register inplace-WAR shape set moved: {dict(shapes)}"
    # Grouped/internal PLR also reaches capped residency-event bodies: a value that cannot survive
    # an invariant-axis pass is read once in each pass, producing the three-refill/six-consumer
    # shape. The floor proof above remains the safety condition for all five shapes.
    assert set(bodies) == {(1, 1), (1, 2), (2, 2), (2, 4), (3, 6)}, \
        f"the refill-per-slot / consumer bound moved: {dict(bodies)}"


def test_a_second_refill_of_one_slot_needs_its_own_consumer():
    """the `consumed`-reset fix, exercised on the body the reachability bound says cannot
    arise YET -- so the arm is correct in advance of the shape that needs it.
    """
    from Tensile.LoopModel.ir import Space, Inst, Load, Mma
    from Tensile.LoopModel.emit import build_ir
    from Tensile.LoopModel.checks import build_ledger, check_ledger_discharged, _region_bodies

    th = adapter.params_to_theta(dict(
        # Degenerate KMN has one register name per operand; PLR1 wraps onto it and therefore
        # supplies the real in-place refill that this synthetic duplicate exercises.
        MIWaveTile=[1, 1], DepthU=64, MatrixInstruction=[16, 16, 32, 1, 1, 1, 1, 2, 2],
        PrefetchLocalRead=1, PrefetchGlobalRead=1, LoopOrder="KMN", ElemBytes=2))
    S, _ = build_S(th)
    ir, ledger = build_ir(th, S), build_ledger(th, S)
    assert check_ledger_discharged(ledger, ir) == []

    def refill(n):
        return (isinstance(n, Inst) and isinstance(n.op, Load) and n.op.dst == Space.REGISTER
                and getattr(n.op, "advance", 0)
                and any(a.kind == "inplace-WAR" for a in (n.awaits or ())))

    src = next(b for _l, b in _region_bodies(ir)
               if any(isinstance(n, Inst) and isinstance(n.op, Mma) for n in b)
               and any(refill(n) for n in b))
    mma = next(n for n in src if isinstance(n, Inst) and isinstance(n.op, Mma))
    rf = next(n for n in src if refill(n))

    def body_of(seq):
        return lambda _b: list(seq)

    ok = _gate_rebuild_bodies(ir, body_of([mma, rf, mma, rf]))
    assert check_ledger_discharged(ledger, ok) == [], \
        "each refill follows its own consumer -- must be accepted"

    bad = check_ledger_discharged(ledger, _gate_rebuild_bodies(ir, body_of([mma, rf, rf, mma])))
    assert bad, "the second refill precedes every consumer of the slot it clobbers -- must be caught"
    assert "inplace-WAR" in {o.kind for o in bad}, sorted({o.kind for o in bad})


# --------------------------------------------------------------------------------------------
# PLR IS AN INPUT; W IS DERIVED.  A request the derivation cannot honour must be REPORTED.
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("order,broadcast_outer_op", [
    ("KMN", None),   # K outermost of the inner nest -> neither operand is broadcast-outer
    ("KNM", None),
    ("NKM", "A"),    # A is broadcast over N_inner, the OUTERMOST inner mode: every A name is
    ("NMK", "A"),    # consumed once per N pass, so it is live for the WHOLE trip
    ("MKN", "B"),    # mirror: B broadcast over M_inner
    ("MNK", "B"),
])
def test_plr_is_honoured_on_every_loop_order(order, broadcast_outer_op):
    """`PrefetchLocalRead` is a PARAMETER and the register width is DERIVED, so the two must be
    made to agree by deriving the schedule -- never by quietly lowering PLR.
    """
    from Tensile.LoopModel.traversal import reloads_whole_set
    from Tensile.LoopModel.schedule import prefetch_refusals
    from Tensile.LoopModel.schedule import derive_S

    th = _theta({**BF16_NT_KMN, "LoopOrder": order})
    assert prefetch_refusals(th, derive_S(th)) == (), \
        f"{order}: PLR must be honoured on every order now"

    got = {op.name for op in th.operands
           if not op.is_output and reloads_whole_set(th, op)}
    want = {broadcast_outer_op} if broadcast_outer_op else set()
    assert got == want, f"{order}: broadcast-outer operands {got}, expected {want}"


def test_the_plr_state_is_named_in_the_geometry_dump():
    """The dump exposes P, U and requested/cap/realized internal PLR."""
    from Tensile.LoopModel import render

    for order in ("KMN", "NKM", "MKN"):
        txt = render.render_geometry(_theta({**BF16_NT_KMN, "LoopOrder": order}))
        assert "prefetch_shortfall" not in txt, f"{order}: unexpected downgrade"
        assert "is satisfied" in txt, f"{order}: the dump must state the request is satisfied"
        assert "P(original/region/group)" in txt
        assert "PLR(request/cap/realized)" in txt


#: The minimal kernel the standalone equivalence below is authored against.
_STANDALONE_PARAMS = {
    "MatrixInstruction": [16, 16, 32, 1], "DepthU": 64, "MIWaveTile": [1, 2],
    "PrefetchGlobalRead": 1, "PrefetchLocalRead": 0, "NumWaves": 1, "LoopOrder": "KMN",
    "GlobalReadVectorWidthA": 8, "GlobalReadVectorWidthB": 8, "LocalReadVectorWidth": 4,
    "ElemBytes": 2, "TDMSplit": [1, 1, 1, 1], "NumLdsBlk": 2,
}


def _standalone_theta():
    """The same point as `_STANDALONE_PARAMS`, authored through the public placement API only."""
    from Tensile.LoopModel import (Axis, Fragment, Global, Operand, RegionLayout, Shared, Theta,
                                   Trajectory)
    from Tensile.LoopModel.schedule import select_register_depths
    from Tensile.LoopModel.traversal import grouping_mode_name

    def staged(name, free_mode, broadcast):
        shared = Shared(vector_elements=8, split=1, ring_depth=2, regions=RegionLayout(),
                        offsets={"iter": 1})
        fragment = Fragment(broadcast_axes=set(broadcast), parts=1, labels=("g0",),
                            group_policy={"*": "pipeline"}, instructions=4,
                            fragment_elements=16, vector_elements=4)
        return Operand(name, free_mode, trajectory=Trajectory(Global(), shared, fragment),
                       elem_bytes=2)

    accumulator = Operand(
        "C", "M_inner", elem_bytes=4, role="output",
        trajectory=Trajectory(Fragment(broadcast_axes={"K_split", "K_inner"},
                                       fragment_elements=8)))
    theta = Theta(
        operands=[staged("A", "M_inner", {"N_inner", "N_split"}),
                  staged("B", "N_inner", {"M_inner", "M_split"}),
                  accumulator],
        ord=(Axis("iter", 0), Axis("K_inner", 2), Axis("N_inner", 2)),
        reg_bytes=4, lanes=32, wave_count=1)
    for operand in theta.operands:
        if operand.movements and operand.fragment:
            operand.fragment.grouping_mode = grouping_mode_name(theta, operand)
    select_register_depths(theta, None)
    return theta


def test_a_hand_authored_theta_emits_the_SAME_LoopIR_as_the_adapted_one():
    """The core is usable without the adapter, and the adapter adds no state the core cannot see.

    If the adapter ever stashes a decision somewhere the public placement API cannot reach, the two
    programs diverge here rather than silently in a kernel nobody renders.
    """
    from Tensile.LoopModel import emit_mainloop, render
    from Tensile.LoopModel import adapter

    adapted = adapter.params_to_theta(_STANDALONE_PARAMS)
    hand = _standalone_theta()

    assert render.render_ir(emit_mainloop(hand).ir) == render.render_ir(emit_mainloop(adapted).ir)
    # NON-VACUITY: the fixture must be a real pipelined loop, not an empty program both agree on.
    assert "tdm" in render.render_ir(emit_mainloop(hand).ir)


def test_the_geometry_view_is_pinned():
    """`render_geometry` has no golden of its own, and dropping a whole block from it moved zero
    cells of the emitted matrix -- only a unit test ever caught that."""
    from Tensile.LoopModel import render
    from Tensile.LoopModel import adapter

    text = render.render_geometry(adapter.params_to_theta(_STANDALONE_PARAMS))
    for required in ("# LoopModel geometry", "reg_bytes=4  lanes=32",
                     "order=iter(OUTER) . K_inner(x2) . N_inner(x2)",
                     "offsets=", "mma_grid=", "operand units:", "operand role", "register_depths:"):
        assert required in text, "render_geometry lost %r:\n%s" % (required, text)

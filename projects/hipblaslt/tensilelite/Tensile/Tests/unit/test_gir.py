# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
Unit tests for GIR itself (Tensile/Lowering/gir + loopir_to_gir) -- the layer-2 IR, its analyses,
and its dataflow passes.  Covers:
  - the LoopIR -> GIR lowering (op-parity, CFG well-formedness, gen_reaching);
  - the swap dataflow analysis + apply pipeline (SwapRegions);
  - the R3 passes (reg_band/reg_gen, tokens, dep_defuse, gr_increment) + build-once cache.

GIR imports nothing from rocisa, so these run under a plain pytest (like test_loopmodel.py).
The GIR->rocisa emit half is tested separately in test_gir_to_rocisa.py.
"""

import pytest

from Tensile.LoopModel import (
    emit, render, theta, schedule, emit_mainloop, Space, Loop, Branch, Bind, Peel, Inst, Load, Mma, Cond,
)
from Tensile.LoopModel import adapter
from Tensile.Lowering import lower_to_gir, build_gir
from Tensile.Lowering.gir.nodes import LoopBack, Trips
from Tensile.Lowering.gir import (
    verify_gir, FrameMap, BackEdges, AnalysisManager, gir_counts, render_gir,
    SwapRegions, RegBandAnalysis, DepDefuseAnalysis, GrIncrementRegions,
    check_register_slots, TokensPass, run_pipeline, Move, Mark, BLOCK_EXIT,
)
from Tensile.Lowering.gir.emit_plan import plan_block
from Tensile.Lowering.gir.emit_plan import plan_program
from Tensile.Lowering.gir.frame_contract import (CONTRACT_MARKER, build_contract,
                                                 encode_frame_contract, parse_contract,
                                                 render_contract)

from gir_fixtures import (BF16_NT_KMN, BF16_NT_KMN_PLR0, BF16_NT_KMN_1LDS,
                          BF16_NT_KMN_FUSED_XAGENT,
                          BF16_NT_KMN_UNFUSED_MULTIAGENT, theta as _theta)
from Tensile.LoopModel.traversal import free_axes, presence, summation_axes


def test_frame_contract_has_dense_actions_and_no_gir_cfg_edges():
    prog = build_gir(_theta(BF16_NT_KMN))
    actions = [action for block in plan_program(prog).values() for action in block]
    assert [action.action_id for action in actions] == list(range(len(actions)))

    contract = encode_frame_contract(prog)
    assert contract.startswith(CONTRACT_MARKER + "\n")
    # The module text carries only what belongs to NO instruction: generations and phi edges.
    assert "\ngen " in contract
    assert "\nincoming " in contract
    assert "\ntransfer " in contract
    assert "\naction " not in contract
    assert "\naccess " not in contract
    # No GIR CFG topology may reach ST: it schedules physical blocks, not GIR ones.
    assert "hazard " not in contract
    assert "edge " not in contract
    assert "block " not in contract
    # The format must round-trip, or it is not the writeable format it claims to be.
    assert render_contract(parse_contract(contract)) == contract

    built = build_contract(prog)
    anchors = {action.id: action.anchor for action in built.actions.values()}
    for block_actions in plan_program(prog).values():
        if block_actions:
            assert all(anchors[action.action_id] == block_actions[0].action_id
                       for action in block_actions)
    # An access names ONE region, or the whole operand; a MAY-set is not expressible.
    assert all(access.region >= -1 for access in built.accesses)


class _Reaching:
    """`FrameMap` read at each block's naming frame -- what `of(ref)` used to mean."""

    def __init__(self, prog):
        self._fm = AnalysisManager().get(FrameMap(), prog)

    def of(self, ref):
        return self._fm.generation(ref, self._fm.cut(self._fm.block_of(ref)))


def _reaching(prog):
    return _Reaching(prog)


def _back_edges(prog):
    return AnalysisManager().get(BackEdges(), prog)


def _pending(prog, analysis):
    return AnalysisManager().get(analysis, prog)


def _swaps(prog, *, hop=None, block=None):
    out = []
    for pm in _pending(prog, SwapRegions()):
        at = pm.mark.at
        if hop is not None and at["hop"] != hop:
            continue
        if block is not None and pm.region.block != block:
            continue
        out.append(pm)
    return out


def _subject(mark):
    """The pointer a swap / gr_increment Mark acts on, as a TUPLE of operand names."""
    at = mark.at
    return (at["operand"],) if "operand" in at else tuple(at["unit"])


def _as_tuple(unit):
    """A pointer identity as a tuple, whichever shape the hop hands it in."""
    return unit if isinstance(unit, tuple) else (unit,)


# ======================================================================= LoopIR -> GIR lowering
def _loopir_stage_counts(ir):
    """Ground truth: read/copy/wmma counts per LoopIR stage via its own unrolled walk."""
    def walk(nodes, env, out):
        for nd in nodes:
            if isinstance(nd, Loop):
                # each sub-body over its OWN half-open range (multi-body: a peeled
                # first-touch guard gives [0,1) with the read, [1,trip) without).  Walking every
                # body over the whole trip would count the peeled read once per iteration.
                for lo, hi, b in nd.ranged_bodies():
                    end = hi if hi is not None else lo + 1
                    for v in range(lo, end):
                        walk(b, {**env, nd.axis: v}, out)
            elif isinstance(nd, Branch):
                if nd.axis in env:
                    walk(nd.arms.get(env[nd.axis] % nd.modulus, []), env, out)
                else:
                    for r, arm in nd.arms.items():
                        walk(arm, {**env, nd.axis: r}, out)
            elif isinstance(nd, Peel):
                walk(nd.body, env, out)
            elif isinstance(nd, Bind):
                walk(nd.body, nd.bound_env(env), out)       # push the peel-step iter binding
            elif isinstance(nd, Cond):
                v = nd.pred.eval(env)                       # honor resolved guards (first-touch / NLL)
                if v is None or v:
                    walk(nd.then, env, out)
                else:
                    walk(nd.els, env, out)
            elif isinstance(nd, Inst):
                op = nd.op
                if isinstance(op, Mma):
                    out["mma"] += 1
                elif isinstance(op, Load) and op.dst == Space.REGISTER:
                    out["read"] += 1
                elif isinstance(op, Load) and op.dst == Space.SHARED:
                    out["copy"] += 1

    def find_peel(nodes, kind):                             # descend the peel-validity Cond.then
        for n in nodes:
            if isinstance(n, Peel) and n.kind == kind:
                return n
            if isinstance(n, Cond):
                p = find_peel(n.then, kind)
                if p is not None:
                    return p
        return None

    pro = find_peel(ir, "prologue")
    drain = find_peel(ir, "drain")

    def find_steady(nodes):
        for n in nodes:
            if isinstance(n, Loop) and n.axis == "iter":
                return n
            if isinstance(n, Cond):
                s = find_steady(n.then)
                if s:
                    return s
        return None

    steady = find_steady(ir)
    res = {}
    for name, node in (("prologue", pro), ("steady", steady), ("drain", drain)):
        c = {"read": 0, "copy": 0, "mma": 0}
        if node is not None:
            walk([node], {}, c)
        res[name] = c
    return res


@pytest.mark.parametrize("params,name", [(BF16_NT_KMN, "PLR1"), (BF16_NT_KMN_PLR0, "PLR0")])
def test_gir_verifies(params, name):
    assert verify_gir(lower_to_gir(_theta(params))), f"{name}: verify_gir failed"


@pytest.mark.parametrize("params,name", [(BF16_NT_KMN, "PLR1"), (BF16_NT_KMN_PLR0, "PLR0")])
def test_gir_has_three_stages(params, name):
    prog = lower_to_gir(_theta(params))
    phases = [b.phase for b in prog.blocks.values()]
    assert "prologue" in phases and "steady" in phases
    assert any(p.startswith("drain") for p in phases), f"{name}: no drain"


@pytest.mark.parametrize("params,name", [(BF16_NT_KMN, "PLR1"), (BF16_NT_KMN_PLR0, "PLR0")])
def test_gir_op_parity_with_loopir(params, name):
    """GIR has the SAME read/copy/mma counts the LoopIR unrolled view emits, per stage."""
    th = _theta(params)
    want = _loopir_stage_counts(emit.build_ir(th, schedule.build_S(th)[0]))
    # `gir_counts` also reports `model_only` per block; project to the op counts this
    # parity check is about.  Keep the flag reachable below rather than dropping it silently.
    _ops = lambda c: {k: c[k] for k in ("read", "copy", "mma")}
    gc = gir_counts(lower_to_gir(th))
    zero = {"read": 0, "copy": 0, "mma": 0}
    got_pro = _ops(gc["prologue"]) if "prologue" in gc else dict(zero)
    got_steady = _ops(gc["steady"]) if "steady" in gc else dict(zero)
    got_drain = dict(zero)
    for ph, c in gc.items():
        if ph.startswith("drain"):
            for k in got_drain:
                got_drain[k] += c[k]
    # THE PARITY IS AGAINST THE LOOPIR, so a model-only block SHOULD be counted here -- it is an op
    # GIR holds.  Assert that no block is silently model-only without the caller being able to see
    # it, which is the fact this restores.
    assert all("model_only" in c for c in gc.values()), \
        "every block's counts must carry the model_only flag, or an op that reaches no .s file is " \
        "indistinguishable here from one that does"
    assert got_pro == want["prologue"], f"{name} prologue: {got_pro} != {want['prologue']}"
    assert got_steady == want["steady"], f"{name} steady: {got_steady} != {want['steady']}"
    assert got_drain == want["drain"], f"{name} drain: {got_drain} != {want['drain']}"


@pytest.mark.parametrize("order", ["KMN", "KNM", "MKN", "MNK", "NKM", "NMK"])
def test_gir_loop_order_general(order):
    """GIR is loop-order-general: every {K,M,N} order builds + verifies with the SAME steady op
    counts (8 read / 2 copy / 8 mma for this MI) -- only the program order differs.  `loop order` reflects
    the requested order (Phase 4 sub-goal 1)."""
    prog = lower_to_gir(_theta({**BF16_NT_KMN, "LoopOrder": order}))
    verify_gir(prog)
    gc = gir_counts(prog).get("steady", {})
    assert {k: gc[k] for k in ("read", "copy", "mma")} == {"read": 8, "copy": 2, "mma": 8}, \
        f"{order} steady counts {gc}"
    assert gc.get("model_only") is False, (
        "the steady block is emitted; `gir_counts` must say so, because a MODEL-ONLY block's ops "
        "are counted here but appear in no .s file")
    # loop order is the inner-mode word in the requested order (K/M/N first letters, in order)
    ord_letters = [m[0] for m in prog.meta["inner_axis_order"]]  # e.g. ['M','N','K'] for MNK
    assert ord_letters == list(order), f"{order}: ord {ord_letters} != {list(order)}"


def test_gir_steady_has_one_back_edge():
    headers = {e.header for e in _back_edges(lower_to_gir(_theta(BF16_NT_KMN)))}
    assert headers == {"steady"}, f"expected one steady back-edge, got {headers}"


def test_gir_render_smoke():
    txt = render_gir(lower_to_gir(_theta(BF16_NT_KMN)))
    assert "Block steady" in txt and "phi:" in txt


def test_theta_ir_text_uses_the_supplied_target_aware_theta_and_mainloop():
    from test_loopmodel import _mxf8_kernel
    from Tensile.LoopModel.render import render_theta

    target = {
        "ReadVectorElems": {"MXSA": 16, "MXSB": 16},
        "ReadPhi": {"MXSA": 4, "MXSB": 4},
        "ReadRho": {"MXSA": 0, "MXSB": 0},
    }
    theta = adapter.params_to_theta(adapter.kernel_to_params(
        _mxf8_kernel("KMN", 0), target))
    mainloop = emit_mainloop(theta)
    text = render_theta(theta, mainloop)
    assert "coverage(carrier=0+(tile)//4" in text
    plain = adapter.params_to_theta(adapter.kernel_to_params(_mxf8_kernel("KMN", 0)))
    assert text != render_theta(plain)


def test_gir_dump_uses_cfg_traversal_order():
    """Late-created join blocks render where control reaches them, not at dictionary tail."""
    txt = render_gir(build_gir(_theta(BF16_NT_KMN)))
    assert txt.index("Block prologue") < txt.index("Block prologue_join") \
        < txt.index("Block steady")


# ======================================================================= frame_map
def test_frame_map_steady_plr1():
    """PLR1: the steady read gen changes mid-body (delta=0 head, delta=1 read-ahead)."""
    prog = lower_to_gir(_theta(BF16_NT_KMN))
    reach = _reaching(prog)
    seen_deltas = set()
    for inst in prog.block("steady").body:
        if isinstance(inst, Move):
            for ref in tuple(inst.srcs) + tuple(inst.dsts):
                if getattr(ref, "gen", None) is not None:
                    seen_deltas.add(ref.gdelta)
                    assert reach.of(ref) is not None
    assert {0, 1} <= seen_deltas, "expected head (delta=0) and read-ahead (delta=1) refs under PLR1"


def test_frame_map_1lds_all_zero():
    """Single LDS buffer (ring 1): every generation is 0 -- the no-swap case."""
    prog = lower_to_gir(_theta(BF16_NT_KMN_1LDS))
    reach = _reaching(prog)
    for blk in prog.blocks.values():
        for inst in blk.body:
            if isinstance(inst, Move):
                for ref in tuple(inst.srcs) + tuple(inst.dsts):
                    g = reach.of(ref)
                    assert g in (None, 0), f"ring-1 generation must be 0, got {g}"


# ======================================================================= swap dataflow
def test_swap_all_are_swap_marks():
    for pm in _pending(lower_to_gir(_theta(BF16_NT_KMN)), SwapRegions()):
        assert isinstance(pm.mark, Mark) and pm.mark.kind == "swap"
        # The pointer key is HOP-DEPENDENT and that is the model, not sloppiness: a read swap moves
        # an operand's LocalReadAddr, a copy swap moves a Phi movement's descriptor.
        key = "operand" if pm.mark.at["hop"] == "read" else "unit"
        assert set(pm.mark.at) >= {"hop", "gen_from", "gen_to", key}
        if key == "unit":
            assert isinstance(pm.mark.at["unit"], tuple)


def test_swap_steady_read_plr1_midbody():
    """PLR1: steady read swaps once per operand (0->1) inside the loop body."""
    reads = _swaps(lower_to_gir(_theta(BF16_NT_KMN)), hop="read", block="steady")
    assert sorted(pm.mark.at["operand"] for pm in reads) == ["A", "B"]
    for pm in reads:
        assert (pm.mark.at["gen_from"], pm.mark.at["gen_to"]) == (0, 1)
        assert pm.region.before is not None


def test_swap_steady_copy_follows_its_own_operands_copy():
    """Steady copy-descriptor swap: one per operand, anchored AFTER the copy that just used the
    descriptor, at the block exit -- so the rotation for the next trip happens as early as it
    legally can and the whole body separates it from the `tensor_load` it feeds.
    """
    prog = lower_to_gir(_theta(BF16_NT_KMN))
    copies = _swaps(prog, hop="copy", block="steady")
    # Unfused: each operand is its own single-member Phi movement, so the units are ('A',) and ('B',).
    assert sorted(_subject(pm.mark) for pm in copies) == [("A",), ("B",)]
    for pm in copies:
        anchor = pm.region.after
        assert isinstance(anchor, Move), "copy swap must anchor on a copy Move"
        assert {d.tile.operand for d in anchor.dsts if d.tile.space == "shared"} \
               == set(_subject(pm.mark)), \
            "copy swap must anchor after the copy Move of its OWN movement"
        assert pm.region.before is BLOCK_EXIT, \
            "the per-trip rotation serves the NEXT trip, so its window runs to the block exit"


def test_prologue_hands_the_descriptor_off_at_its_exit():
    """The prologue rotates BETWEEN its fills, and once more at its exit to hand the descriptor to
    the steady body -- two per operand, both after a copy.
    """
    prog = lower_to_gir(_theta(BF16_NT_KMN))
    per_operand = {}
    for pm in _swaps(prog, hop="copy", block="prologue"):
        per_operand.setdefault(_subject(pm.mark), []).append(pm)
    assert sorted(per_operand) == [("A",), ("B",)], per_operand
    for operand, marks in per_operand.items():
        assert len(marks) == 2, f"{operand}: expected fill-to-fill + hand-off, got {len(marks)}"
        assert all(isinstance(pm.region.after, Move) for pm in marks), \
            "every prologue rotation follows the fill whose buffer it supersedes"
        assert sum(1 for pm in marks if pm.region.before is BLOCK_EXIT) == 1, \
            "exactly one of them is the exit hand-off to the steady body"


def test_swap_plr0_read_is_loop_bottom():
    """PLR0: no read-ahead -> the current gen is used through the body and the swapped gen is first
    used next trip, so the read swap's region ends at the trip bottom (BLOCK_EXIT)."""
    reads = _swaps(lower_to_gir(_theta(BF16_NT_KMN_PLR0)), hop="read", block="steady")
    assert sorted(pm.mark.at["operand"] for pm in reads) == ["A", "B"]
    for pm in reads:
        assert pm.region.before is BLOCK_EXIT, "PLR0 read swap must anchor at the trip bottom"


def test_swap_drain_boundary():
    """drain: the read pointer swaps across the drain chain (absolute generations, fact)."""
    drain_reads = [pm for pm in _swaps(lower_to_gir(_theta(BF16_NT_KMN)), hop="read")
                   if pm.region.block.startswith("drain")]
    assert drain_reads, "expected drain read swaps from absolute generations"
    for pm in drain_reads:
        assert pm.mark.at["gen_from"] != pm.mark.at["gen_to"]


def test_swap_no_drain_copy():
    """The drain copy-descriptor swap is scaffold-owned : no GIR copy swap in drain."""
    drain_copies = [pm for pm in _swaps(lower_to_gir(_theta(BF16_NT_KMN)), hop="copy")
                    if pm.region.block.startswith("drain")]
    assert drain_copies == [], f"drain copy swap must be scaffold-owned, got {drain_copies}"


def test_swap_ring1_none():
    assert _pending(lower_to_gir(_theta(BF16_NT_KMN_1LDS)), SwapRegions()) == []


# ======================================================================= reg_band / reg_gen
def test_reg_band_width_plr1():
    """PLR1: register rotation width W = 2 per operand group (from the LoopIR slot modulus, B3)."""
    band = AnalysisManager().get(RegBandAnalysis(), lower_to_gir(_theta(BF16_NT_KMN)))
    assert band.items()
    for (operand, group), w in band.items().items():
        assert w == 2, f"{operand} g{group}: W={w} (PLR1 expects 2)"


def test_reg_band_width_consistent_plr0():
    """Without prefetch, sequential units reuse one W=1 slot."""
    band = AnalysisManager().get(RegBandAnalysis(), lower_to_gir(_theta(BF16_NT_KMN_PLR0)))
    assert band.items()
    for (operand, group), w in band.items().items():
        assert w == 1, f"{operand} g{group}: W={w}"


def test_mnk_register_layout_uses_k_unit_for_a_and_full_nk_unit_for_b():
    params = {
        **BF16_NT_KMN,
        "MIWaveTile": [4, 4],
        "MatrixInstruction": [16, 16, 32, 1, 1, 4, 4, 2, 2],
        "LoopOrder": "MNK",
    }
    th = _theta(params)
    from Tensile.LoopModel import traversal as geometry
    depths = schedule.build_S(th)[0]
    assert geometry.requested_read_ahead(th, th.op("A"), depths) == 1
    assert geometry.requested_read_ahead(th, th.op("B"), depths) == 2
    prog = build_gir(th)
    layout = prog.meta["register_layout"]
    assert layout["A"]["total"] == 32
    assert layout["B"]["total"] == 128

    acts = plan_block(prog, "steady")
    b_reads = [act.at for act in acts if act.kind == "read" and act.at["tc"] == "B"]
    b_wmmas = [act.at for act in acts if act.kind == "wmma"]
    assert {at["reg_buf"] for at in b_reads} == {0, 1}
    assert {at["bufB"] for at in b_wmmas} == {0, 1}
    assert {at["unit_index"] for at in b_reads} == set(range(8))
    assert {at["unitB"] for at in b_wmmas} == set(range(8))


def test_unquantized_register_units_preserve_rotation_ord():
    """L3's compact bases use the same mixed-radix order as the LoopOrder rotation unit."""
    from Tensile.LoopModel import traversal as geometry

    base = {
        **BF16_NT_KMN, "MIWaveTile": [2, 2],
        "MatrixInstruction": [16, 16, 32, 1, 1, 2, 2, 2, 2],
    }
    mkn = _theta({**base, "LoopOrder": "MKN"})
    assert geometry.unit_tile_index(mkn, mkn.op("B"),
                                    {"K_inner": 0, "N_inner": 1}) == 1
    assert geometry.unit_tile_index(mkn, mkn.op("B"),
                                    {"K_inner": 1, "N_inner": 0}) == 2

    nkm = _theta({**base, "LoopOrder": "NKM"})
    assert geometry.unit_tile_index(nkm, nkm.op("A"),
                                    {"K_inner": 0, "M_inner": 1}) == 1
    assert geometry.unit_tile_index(nkm, nkm.op("A"),
                                    {"K_inner": 1, "M_inner": 0}) == 2


def test_all_inner_refills_use_static_reduction_phases_after_the_last_consumer():
    """A rolled body cannot alternate VGPR slots by `iter`; each phase is refilled in place."""
    from Tensile.LoopModel import traversal as geometry

    th = adapter.params_to_theta({
        "MatrixInstruction": [16, 16, 128, 1, 1, 2, 2, 1, 1],
        "MIWaveTile": [2, 2], "DepthU": 256, "ElemBytes": 1,
        "PrefetchGlobalRead": 2, "PrefetchLocalRead": 1, "LoopOrder": "MKN",
    })
    b = th.op("B")
    group = b.fragment.groups()[0]
    depths = schedule.build_S(th)[0]
    width = depths.get("B", group)
    want = geometry.requested_read_ahead(th, b, depths)
    assert want == 2
    distance = geometry.prefetch_distance_for(th, b, want)
    slot = geometry.ring_slot(th, b, group, width, advance=distance)
    assert slot.eval({"iter": 0, "M_inner": 0, "K_inner": 1, "N_inner": 0}) == 1
    assert slot.eval({"iter": 1, "M_inner": 0, "K_inner": 1, "N_inner": 0}) == 1

    actions = plan_block(build_gir(th), "steady")
    for read_at, read in ((i, action) for i, action in enumerate(actions)
                          if action.kind == "read" and action.at["tc"] == "B"):
        consumers = [i for i, action in enumerate(actions)
                     if action.kind == "wmma"
                     and action.at["unitB"] == read.at["unit_index"]
                     and action.at["bufB"] == read.at["reg_buf"]]
        assert consumers
        assert read_at > max(consumers)


def test_a_leading_split_axis_still_makes_the_other_operand_all_inner():
    """KMKNMN with live M_split classifies B against M, not the later prefetch axis K."""
    from Tensile.LoopModel import traversal as geometry
    from Tensile.Lowering.gir import check_plan

    th = adapter.params_to_theta({
        "MatrixInstruction": [16, 16, 128, 1, 1, 2, 2, 2, 2],
        "MIWaveTile": [2, 2], "DepthU": 256, "ElemBytes": 1,
        "PrefetchGlobalRead": 1, "PrefetchLocalRead": 0,
        "LoopOrder": "KMKNMN", "TDMSplit": [2, 1, 1, 1],
        "NumWaves": 4, "TDMFuse": 0, "VectorWidthA": 2, "VectorWidthB": 2,
    })
    assert geometry.global_outer_role(th) == "M"
    assert geometry.reloads_whole_set(th, th.op("B"))
    assert geometry.rotation_unit_tiles(th, th.op("B")) == 4
    assert check_plan(build_gir(th)) == []


def test_reduction_regions_follow_rotation_unit_membership():
    """An outer K_split reuses names; a K_split inside the rotation unit is a packed digit."""
    from Tensile.LoopModel import traversal as geometry

    th = adapter.params_to_theta({
        "MatrixInstruction": [16, 16, 128, 1, 1, 2, 2, 2, 2],
        "MIWaveTile": [2, 2], "DepthU": 256, "ElemBytes": 1,
        "PrefetchGlobalRead": 1, "PrefetchLocalRead": 0, "LoopOrder": "KMN",
        "TDMSplit": [1, 1, 2, 1], "NumWaves": 4, "TDMFuse": 0,
        "VectorWidthA": 2, "VectorWidthB": 2,
    })
    prog = build_gir(th)
    a = th.op("A")
    layout = prog.meta["register_layout"]["A"]
    assert geometry._region_count(th, a) == 1
    assert layout["total"] == 2 * geometry.frag_regs(th, a)
    units = {act.at["unit_index"] for act in plan_block(prog, "steady")
             if act.kind == "read" and act.at["tc"] == "A"}
    assert units == {0, 1}

    inner = adapter.params_to_theta({
        "MatrixInstruction": [16, 16, 32, 1, 1, 2, 2, 2, 2],
        "MIWaveTile": [2, 2], "DepthU": 128, "ElemBytes": 2,
        "PrefetchGlobalRead": 2, "PrefetchLocalRead": 0, "LoopOrder": "MNK",
        "TDMSplit": [1, 1, 2, 2], "NumWaves": 4,
    })
    inner_prog = build_gir(inner)
    inner_a = inner.op("A")
    assert geometry._region_count(inner, inner_a) == 2
    assert inner_prog.meta["register_layout"]["A"]["total"] \
        == 4 * geometry.frag_regs(inner, inner_a)
    assert {act.at["unit_index"] for act in plan_block(inner_prog, "steady")
            if act.kind == "read" and act.at["tc"] == "A"} == {0, 1, 2, 3}


@pytest.mark.parametrize("rho,total,unit_count,units", [
    (2, 2, 2, [(0, (0, 0)), (1, (1, 1))]),   # distributed: selector chooses the partner
    (0, 4, 4, [(0, (0, 1)), (2, (2, 3))]),   # contiguous: b64 writes adjacent VGPRs
])
def test_folded_mxs_reads_define_each_scalar_scale_lane(rho, total, unit_count, units):
    """Distributed partners alias one VGPR; contiguous lanes occupy adjacent VGPRs."""
    from test_loopmodel import _mxf8_kernel

    target = {
        "ReadVectorElems": {"MXSA": 8, "MXSB": 8},
        "ReadPhi": {"MXSA": 2, "MXSB": 2},
        "ReadRho": {"MXSA": rho, "MXSB": rho},
        "RegisterBudget": 224,
    }
    kernel = dict(_mxf8_kernel("MKN", 0))
    kernel["PrefetchLocalRead"] = 0
    th = adapter.params_to_theta(adapter.kernel_to_params(kernel, target))
    prog = build_gir(th)
    assert prog.meta["register_layout"]["MXSA"]["total"] == total
    reads = [act.at for act in plan_block(prog, "steady")
             if act.kind == "read" and act.at["tc"] == "MXSA"]
    assert [(at["unit_index"], at["unit_indexes"]) for at in reads] == units
    assert {src[3] for act in plan_block(prog, "steady") if act.kind == "wmma"
            for src in act.at["srcs"] if src[0] == "MXSA"} == set(range(unit_count))


def test_mx16_scale_units_advance_by_two_vgprs():
    """MXBlock16 has two VGPRs per logical unit; contiguous and distributed folds differ."""
    from test_loopmodel import _mxf8_kernel, _two_tile_fold
    from Tensile.LoopModel.traversal import frag_regs

    kernel = dict(_mxf8_kernel("KMN", 0))
    kernel["ProblemType"] = dict(kernel["ProblemType"], MXBlockA=16, MXBlockB=16)
    contiguous = _two_tile_fold()
    from Tensile.LoopModel.ir import TransferCoverage, Expr, COVERAGE_VAR
    t = COVERAGE_VAR
    distributed = TransferCoverage(carrier=Expr(digits=(((t, 1),), 0, ((1, 2, 0),))),
                              slot=Expr(digits=(((t, 1),), 0, ((1, 2, 1),))))
    th = adapter.params_to_theta(adapter.kernel_to_params(
        kernel, {"ReadQuantum": {"MXSA": contiguous, "MXSB": distributed}}))
    prog = build_gir(th)
    assert prog.meta["register_layout"]["MXSA"]["total"] == 8
    assert prog.meta["register_layout"]["MXSB"]["total"] == 4
    assert frag_regs(th, th.op("MXSA")) == 2


def test_partial_all_inner_prefetch_wraps_once_after_the_last_outer_consumer():
    """Non-wrapping B reads prime the next N region early; wrapping reads wait for M_split=1."""
    from collections import Counter
    from Tensile.Lowering.gir import check_plan
    from test_loopmodel import _mxf8_kernel

    target = {
        "ReadVectorElems": {"MXSA": 8, "MXSB": 8},
        "ReadPhi": {"MXSA": 2, "MXSB": 2},
        "ReadRho": {"MXSA": 0, "MXSB": 0},
        "RegisterBudget": 224,
    }
    kernel = dict(_mxf8_kernel("KMNKMN", 1))
    kernel.update(PrefetchGlobalRead=2, PrefetchLocalRead=1)
    prog = build_gir(adapter.params_to_theta(adapter.kernel_to_params(kernel, target)))
    assert check_plan(prog) == []
    fills = [fill for act in plan_block(prog, "steady")
             if act.kind == "read" and act.at["tc"] == "B"
             for fill in act.at["fills"]]
    assert set(Counter(fills).values()) == {1}


def test_capped_policy_search_rejects_an_unserved_pipeline_ring():
    """N.M.N reuses B across M; W=2 cannot retain four N_inner values, so B is in-place."""
    from Tensile.Lowering.gir import check_plan
    from test_loopmodel import _mxf8_kernel

    kernel = dict(_mxf8_kernel("NKMNKM", 0))
    kernel.update(MatrixInstruction=[16, 16, 128, 1, 1, 2, 8, 2, 2],
                  MIWaveTileA=2, MIWaveTileB=8,
                  PrefetchGlobalRead=2, PrefetchLocalRead=1,
                  TDMSplitA=1, TDMSplitB=1,
                  VectorWidthA=1, VectorWidthB=1)
    target = {
        "ReadVectorElems": {"MXSA": 4, "MXSB": 4},
        "ReadPhi": {"MXSA": 2, "MXSB": 2},
        "ReadRho": {"MXSA": 2, "MXSB": 2},
        "RegisterBudget": 480,
    }
    th = adapter.params_to_theta(adapter.kernel_to_params(kernel, target))
    b = th.op("B")
    assert [b.fragment.policy_of(group) for group in b.fragment.groups()] == ["inplace"]
    assert check_plan(build_gir(th)) == []


def test_mxs_reduction_coordinate_advances_the_scalar_unit_group():
    """On K.M4 with a two-M carrier, K1 advances past both K0 free carriers in W=1."""
    from Tensile.Lowering.gir import check_plan
    from test_loopmodel import _mxf8_kernel

    kernel = dict(_mxf8_kernel("KMN", 0))
    kernel.update(MatrixInstruction=[16, 16, 128, 1, 1, 4, 4, 2, 2],
                  MIWaveTileA=4, MIWaveTileB=4,
                  PrefetchGlobalRead=2, PrefetchLocalRead=1,
                  VectorWidthA=1, VectorWidthB=4)
    target = {
        "ReadVectorElems": {"MXSA": 4, "MXSB": 16},
        "ReadPhi": {"MXSA": 2, "MXSB": 4},
        "ReadRho": {"MXSA": 2, "MXSB": 0},
    }
    prog = build_gir(adapter.params_to_theta(adapter.kernel_to_params(kernel, target)))
    assert check_plan(prog) == []
    k1 = [act.at["unit_index"] for act in plan_block(prog, "steady")
          if act.kind == "read" and act.at["tc"] == "MXSA" and act.at["k_flat"] == 1]
    assert set(k1) == {2, 3}


def test_every_unit_index_fits_its_compact_group_slot():
    params = {
        **BF16_NT_KMN,
        "MIWaveTile": [4, 4],
        "MatrixInstruction": [16, 16, 32, 1, 1, 4, 4, 2, 2],
        "LoopOrder": "MNK",
    }
    prog = build_gir(_theta(params))
    layout = prog.meta["register_layout"]
    for block in prog.blocks:
        for act in plan_block(prog, block):
            if act.kind == "read":
                at = act.at
                base = layout[at["tc"]]["slots"][(at["group"], at["reg_buf"])]
                assert base + (at["unit_index"] + 1) * at["size_regs"] \
                    <= layout[at["tc"]]["total"]
            elif act.kind == "wmma":
                at = act.at
                for tc, group, slot, unit, size in (
                        ("A", at["groupA"], at["bufA"], at["unitA"], at["sizeA"]),
                        ("B", at["groupB"], at["bufB"], at["unitB"], at["sizeB"])):
                    base = layout[tc]["slots"][(group, slot)]
                    assert base + (unit + 1) * size <= layout[tc]["total"]


def test_group_reuse_floor_keeps_its_query_cache_key(monkeypatch):
    """Register-key construction must not overwrite the key used to cache the full query."""
    from Tensile.LoopModel import traversal as geometry

    th = _theta({**BF16_NT_KMN, "DepthU": 384, "MIWaveTile": [6, 6],
                 "MatrixInstruction": [16, 16, 32, 1], "PrefetchLocalRead": 0})
    if hasattr(th, "_group_reuse_floor_cache"):
        delattr(th, "_group_reuse_floor_cache")
    op = th.op("A")
    group = op.fragment.groups()[0]
    calls = 0
    inner_steps = geometry._inner_steps

    def counted(theta):
        nonlocal calls
        calls += 1
        return inner_steps(theta)

    monkeypatch.setattr(geometry, "_inner_steps", counted)
    first = geometry.group_reuse_floor(th, op, group)
    assert calls == 1
    assert geometry.group_reuse_floor(th, op, group) == first
    assert calls == 1


def test_register_slots_check_validates():
    check_register_slots(lower_to_gir(_theta(BF16_NT_KMN)), AnalysisManager())  # raises on defect


def test_register_slots_check_rejects_a_unit_past_the_compact_allocation():
    import dataclasses

    prog = lower_to_gir(_theta(BF16_NT_KMN))
    move = next(inst for block in prog.blocks.values() for inst in block.body
                if isinstance(inst, Move)
                and any(ref.tile.space == "register" for ref in inst.dsts))
    ref = next(ref for ref in move.dsts if ref.tile.space == "register")
    move.dsts = tuple(dataclasses.replace(item, unit_index=10**6, unit_indexes=(10**6,))
                      if item is ref else item
                      for item in move.dsts)
    with pytest.raises(RuntimeError, match="beyond compact allocation"):
        check_register_slots(prog, AnalysisManager())


# ======================================================================= tokens / dep_defuse
def test_tokens_stamped_from_generation():
    prog = lower_to_gir(_theta(BF16_NT_KMN))
    TokensPass().run(prog, AnalysisManager())
    lds = [i for i in prog.block("steady").body if isinstance(i, Move) and i.token is not None]
    assert lds, "expected stamped LDS moves in steady"
    for mv in lds:
        kind, operand, footprint = mv.token
        assert kind == "lds" and operand in ("A", "B")
        # the token now names the SET of generations this STATIC instruction can touch: a rolled
        # loop's pointer rotates, so a steady Move touches the whole ring (tokens._footprint).
        assert isinstance(footprint, tuple) and set(footprint) <= {0, 1} and footprint, mv.token


def test_every_token_names_ONE_buffer():
    """the COMPLETION token is a one-element MUST-set -- "this access touches this buffer"."""
    for params in (BF16_NT_KMN, BF16_NT_KMN_PLR0, BF16_NT_KMN_FUSED_XAGENT):
        prog = build_gir(_theta(params))
        for blk in prog.blocks.values():
            for inst in blk.body:
                tok = getattr(inst, "token", None)
                if tok is None:
                    continue
                assert len(tok[2]) == 1, f"{blk.label}: token {tok} names more than one buffer"


def test_dep_defuse_indexes_awaits():
    """DepDefuse indexes each Move's LoopIR awaits by hazard kind, no edge lost/double-counted."""
    prog = lower_to_gir(_theta(BF16_NT_KMN))
    dd = AnalysisManager().get(DepDefuseAnalysis(), prog)
    steady = prog.block("steady")
    assert [e for i in steady.body if isinstance(i, Move) for e in dd.raw(i)], "expected RAW edges"
    for i in steady.body:
        if isinstance(i, Move):
            assert len(dd.raw(i)) + len(dd.war(i)) == len(i.deps)


# ======================================================================= gr_increment
def test_gr_increment_follows_the_copy_that_used_the_descriptor():
    """One gr_increment per operand per CHUNK TRANSITION, anchored AFTER the copy that just consumed
    the descriptor.  The prologue carries M advances per operand, not M-1: M-1 between its own peel
    fills, plus the HAND-OFF at its exit.
    """
    prog = lower_to_gir(_theta(BF16_NT_KMN))
    pend = _pending(prog, GrIncrementRegions())
    assert pend, "expected gr_increment Marks"
    for pm in pend:
        anchor = pm.region.after
        assert isinstance(anchor, Move) and any(d.tile.space == "shared" for d in anchor.dsts), \
            "gr_inc must anchor AFTER a copy Move -- the one whose fetch it supersedes"
        assert set(pm.mark.at) == {"unit", "chunks", "to_chunk"}, \
            "gr_increment Mark carries the movement + chunk count + target chunk only -- no stride"
        assert pm.mark.at["chunks"] == 1
    M = prog.meta["peel_depth"]
    for operand in ("A", "B"):
        pro = [pm for pm in pend
               if pm.region.block == "prologue" and _subject(pm.mark) == (operand,)]
        assert len(pro) == M, (f"{operand}: {len(pro)} prologue advances, expected M = {M} "
                               f"(M-1 between peel fills + 1 hand-off)")
        assert sum(1 for pm in pro if pm.region.before is BLOCK_EXIT) == 1, \
            f"{operand}: exactly one prologue advance is the exit hand-off"


# ======================================================================= pipeline + build-once
@pytest.mark.parametrize("params,name", [(BF16_NT_KMN, "PLR1"), (BF16_NT_KMN_PLR0, "PLR0"),
                                         (BF16_NT_KMN_1LDS, "1LDS")])
def test_pipeline_applies_marks_and_verifies(params, name):
    """The full pipeline applies swap + gr_increment Marks and re-verifies; applied counts equal
    the analyses' PendingMark counts."""
    prog = lower_to_gir(_theta(params))
    n_swaps = len(_pending(prog, SwapRegions()))
    n_gr = len(_pending(prog, GrIncrementRegions()))
    run_pipeline(prog)
    swaps = sum(1 for b in prog.blocks.values() for i in b.body
                if isinstance(i, Mark) and i.kind == "swap")
    grs = sum(1 for b in prog.blocks.values() for i in b.body
              if isinstance(i, Mark) and i.kind == "gr_increment")
    assert swaps == n_swaps, f"{name}: {swaps} swap marks != {n_swaps}"
    assert grs == n_gr, f"{name}: {grs} gr marks != {n_gr}"
    assert prog.pending == [], f"{name}: pending should be cleared after apply"
    verify_gir(prog)


def test_build_once_produces_finalized_program():
    """build_gir builds theta->GIR and runs the pipeline ONCE -> a verified Program with swap +
    gr_increment + tokens all present."""
    theta = _theta(BF16_NT_KMN)
    prog = build_gir(theta, mainloop=emit_mainloop(theta))
    verify_gir(prog)
    has = lambda kind: any(isinstance(i, Mark) and i.kind == kind
                           for b in prog.blocks.values() for i in b.body)
    has_token = any(isinstance(i, Move) and i.token is not None
                    for b in prog.blocks.values() for i in b.body)
    assert has("swap") and has("gr_increment") and has_token


def test_build_once_mainloop_reused():
    """R-ONCE: build_gir accepts a pre-built mainloop and is self-contained."""
    theta = _theta(BF16_NT_KMN)
    ml = emit_mainloop(theta)
    assert list(build_gir(theta, mainloop=ml).blocks.keys()) == \
           list(build_gir(theta, mainloop=ml).blocks.keys())


# --------------------------------------------------------------------- per-region completion (pi)
def _split_theta(per_region):
    from Tensile.LoopModel import adapter
    import loopmodel_scenarios as scenarios
    _d, params = scenarios.SCENARIOS["our_split"]
    return adapter.params_to_theta({**params, "PerRegionCompletion": per_region})


def _lds_tokens(prog, block, operand, copies):
    """Tokens stamped on `operand`'s copy (or read) Moves in `block`."""
    out = set()
    for inst in prog.block(block).body:
        if not isinstance(inst, Move):
            continue
        is_copy = any(d.tile.space == "shared" for d in inst.dsts)
        if is_copy != copies:
            continue
        for r in list(inst.srcs) + list(inst.dsts):
            if r.tile.space == "shared" and r.tile.operand == operand and inst.token:
                out.add(inst.token)
    return out


def test_shared_completion_collapses_all_regions_onto_one_token():
    """DEFAULT pi (`per_region_completion=False`, what the scaffold implements today): every storage
    region of an operand shares ONE completion class, so a read must await them all -- no region can
    stay in flight.  The token keeps its 3-tuple shape, unchanged from before the toggle existed."""
    prog = run_pipeline(lower_to_gir(_split_theta(False)))
    toks = _lds_tokens(prog, "steady", "A", copies=True)
    assert len(toks) == 1, f"expected the split regions to share one token, got {sorted(toks)}"
    assert all(len(t) == 3 for t in toks), f"shared pi must keep the 3-tuple token: {sorted(toks)}"


def test_per_region_completion_gives_each_region_its_own_token():
    """PER-REGION pi: each region instance is its own completion class, so a read of region j pairs only with region j's copy and the other
 regions stay in flight. This is the whole point of the toggle."""
    th = _split_theta(True)
    prog = run_pipeline(lower_to_gir(th))
    a_op = th.op("A")
    assert a_op.split > 1, "fixture is not region-split; test would be vacuous"

    toks = _lds_tokens(prog, "steady", "A", copies=True)
    assert len(toks) == a_op.split, \
        f"expected {a_op.split} distinct region tokens, got {sorted(toks)}"
    assert all(len(t) > 3 for t in toks), \
        f"per-region token must carry the region coordinate: {sorted(toks)}"
    # the region component is what distinguishes them (same operand, same generation)
    assert len({t[:3] for t in toks}) == 1 and len({t[3:] for t in toks}) == a_op.split, \
        f"regions must differ ONLY in the region key: {sorted(toks)}"


def test_both_completion_settings_verify_and_discharge():
    """Both pi settings must produce a well-formed, empty-ledger program -- the toggle changes the
    completion-class granularity, never correctness."""
    from Tensile.LoopModel import build_S, validate_loopir
    for per_region in (False, True):
        th = _split_theta(per_region)
        r = emit_mainloop(th)
        S, _floor = build_S(th)
        assert r.obligations_discharged, f"per_region={per_region}: ledger not empty"
        assert validate_loopir(th, r.ir, S) == [], f"per_region={per_region}: validate failed"
        verify_gir(run_pipeline(lower_to_gir(th)))


# --------------------------------------------------------------------- TDMFuse (Phi) tokens
def _units(prog):
    """(produced, awaited) completion UNITS -- token[1] -- over the whole program."""
    produced, awaited = set(), set()
    for blk in prog.blocks.values():
        for i in blk.body:
            if isinstance(i, Move) and i.token:
                tgt = produced if any(d.tile.space == "shared" for d in i.dsts) else awaited
                tgt.add(i.token[1])
    return produced, awaited


@pytest.mark.parametrize("scen", sorted(__import__("loopmodel_scenarios").SCENARIOS))
def test_every_awaited_completion_unit_is_actually_produced(scen):
    """No read may await a completion unit that no copy in the program produces."""
    from Tensile.LoopModel import adapter
    import loopmodel_scenarios as scenarios
    _d, params = scenarios.SCENARIOS[scen]
    prog = run_pipeline(lower_to_gir(adapter.params_to_theta(params)))
    produced, awaited = _units(prog)
    assert not (awaited - produced), \
        f"{scen}: awaited but never produced: {sorted(awaited - produced, key=str)}"


def test_fused_group_is_the_completion_unit():
    """Under Phi the token names the GROUP, so every member's reads and the one cooperative copy
    agree on a single completion class."""
    from Tensile.LoopModel import adapter
    import loopmodel_scenarios as scenarios
    _d, params = scenarios.SCENARIOS["mx_fuse_ab"]
    th = adapter.params_to_theta(params)
    assert th.fused_copy_groups, "fixture has no fuse group; test would be vacuous"
    prog = run_pipeline(lower_to_gir(th))
    produced, _awaited = _units(prog)
    assert ("A", "B") in produced, f"fused group is not the completion unit: {sorted(produced, key=str)}"
    assert "A" not in produced and "B" not in produced, \
        f"a fused member still has its own completion unit: {sorted(produced, key=str)}"


# ============================================================== Phi movement identity
def _fused_theta(**over):
    """theta for the multi-wave TDM shape: A and B fused into one cooperative movement (TDMFuse=0)."""
    from Tensile.LoopModel import adapter
    p = dict(BF16_NT_KMN)
    p.update(over)
    p["TDMFuse"] = 0
    p.setdefault("NumWaves", 2)     # a fuse IS the wave selection, so it needs a wave to select
    return adapter.params_to_theta(p)


def test_copy_unit_names_every_member_of_the_fused_movement():
    """`copy_unit` must return ALL of a fused Move's shared destinations, not the first."""
    from Tensile.Lowering.gir.nodes import copy_unit
    prog = lower_to_gir(_fused_theta())
    seen = set()
    for blk in prog.blocks.values():
        for inst in blk.body:
            members, refs = copy_unit(inst)
            if members is None:
                continue
            seen.add(members)
            assert len(refs) == len(members), "one shared dst Ref per member"
    assert seen == {("A", "B")}, f"expected one fused movement, got {sorted(seen)}"

    # ... and the unfused program still reports singletons, so this is a generalization and not a
    # change of meaning for every kernel that already works.
    plain = lower_to_gir(_theta(BF16_NT_KMN))
    plain_units = {copy_unit(i)[0] for b in plain.blocks.values() for i in b.body
                   if copy_unit(i)[0] is not None}
    assert plain_units == {("A",), ("B",)}, plain_units


def test_copy_unit_rejects_members_that_rotate_differently():
    """The "one movement, one pointer" claim is CHECKED, not assumed."""
    import dataclasses
    import pytest as _pytest
    from Tensile.Lowering.gir.nodes import copy_unit
    prog = lower_to_gir(_fused_theta())
    victim = next(i for b in prog.blocks.values() for i in b.body
                  if copy_unit(i)[0] is not None and getattr(i.dsts[0], "gen", None) is not None)
    dsts = list(victim.dsts)
    dsts[1] = dataclasses.replace(dsts[1], gdelta=dsts[1].gdelta + 1)
    victim.dsts = tuple(dsts)
    with _pytest.raises(RuntimeError, match="rotate DIFFERENTLY"):
        copy_unit(victim)


@pytest.mark.parametrize("pgr", [1, 2])
@pytest.mark.parametrize("plr", [0, 1])
def test_fusion_halves_the_descriptor_acts_and_leaves_the_read_pointers_alone(pgr, plr):
    """The whole of multi-wave TDM, as a count invariant."""
    over = dict(PrefetchGlobalRead=pgr, PrefetchLocalRead=plr)
    def counts(theta):
        prog = build_gir(theta)          # FINALIZED: the Marks only exist after the pipeline runs
        n = {}
        for lab in prog.blocks:
            for a in plan_block(prog, lab):
                key = a.kind if a.kind != "swap" else "swap_" + a.at["hop"]
                n[key] = n.get(key, 0) + 1
        return n
    from Tensile.LoopModel import adapter
    plain = counts(adapter.params_to_theta(dict(BF16_NT_KMN, **over)))
    fused = counts(_fused_theta(**over))
    for k in ("copy", "gr_inc", "swap_copy"):
        assert fused[k] * 2 == plain[k], \
            f"{k}: fused {fused[k]} vs unfused {plain[k]} -- a fused movement is ONE instruction"
    assert fused["swap_read"] == plain["swap_read"], \
        "read pointers are per-operand and must NOT be fused"


def test_fused_acts_name_the_movement_and_the_analyses_agree_with_theta():
    """The `unit` on every copy-side act is the Phi group, and it is the SAME group theta's
    `movement_units()` reports.
    """
    from Tensile.Lowering.gir.passes.tokens import _unit_key
    theta = _fused_theta()
    assert [tuple(g) for g in theta.fused_copy_groups] == [("A", "B")], "fixture is not fused"
    prog = run_pipeline(lower_to_gir(theta))
    n = 0
    for lab in prog.blocks:
        for a in plan_block(prog, lab):
            if a.kind == "copy":
                assert a.at["unit"] == ("A", "B")
                assert a.at["token"][1] == ("A", "B"), "the completion class is the group too"
            elif a.kind == "gr_inc" or (a.kind == "swap" and a.at["hop"] == "copy"):
                assert a.at["unit"] == ("A", "B")
            elif a.kind == "swap":
                assert a.at["operand"] in ("A", "B"), "a read swap names an op-class"
                continue
            else:
                continue
            n += 1
    assert n, "no fused copy-side acts found"
    for member in ("A", "B"):
        assert _unit_key(prog, member) == ("A", "B"), \
            "TokensPass and copy_unit disagree about which movement carries this operand"


def test_one_hop_direct_to_register_lowers_to_a_wellformed_cfg():
    """A one-hop (direct-to-register) path stages nothing through shared, so there is no
 copy, no peel and NO prologue block -- the steady loop is the whole program. The steady block's preds were hardcoded to ("prologue","steady"), so G-TERM
 rejected exactly this kernel."""
    from Tensile.LoopModel import adapter
    import loopmodel_scenarios as scenarios
    _d, params = scenarios.SCENARIOS["dtv"]
    prog = lower_to_gir(adapter.params_to_theta(params))
    assert "prologue" not in prog.blocks, "dtv fixture unexpectedly has a prologue"
    verify_gir(prog)
    assert prog.block("steady").preds == ("steady",)


def test_heterogeneous_phi_region_paired_plus_whole_group():
    """the heterogeneous Phi: TDMFuse=0 with TDMSplit=[2,2] must give {A,B} paired BY REGION
 INDEX into two cooperative movements (A0/B0, A1/B1 -- "two fused movements with two
 completions") alongside {MXSA,MXSB} as ONE whole movement, "a per-region-paired group with a
 whole-operand (unsplit) group in the same Phi".

 Under the per-region pi this is also the case that caught a hole in the token key: the MX group
 is unsplit, so its copy has ONE completion covering every region, but MXSA's READS carry a
 region coord -- keying the reads by region made them await a token no copy stamped. A read can
 only await at the granularity its PRODUCER offers.
 """
    from Tensile.LoopModel import adapter
    import loopmodel_scenarios as scenarios
    _d, params = scenarios.SCENARIOS["mx_fuse_ab_split"]
    for per_region in (False, True):
        th = adapter.params_to_theta({**params, "PerRegionCompletion": per_region})
        assert [tuple(g) for g in th.fused_copy_groups] == [("A", "B"), ("MXSA", "MXSB")]
        units = {k: n for k, _m, n in th.movement_units()}
        assert units[("A", "B")] == 2, f"A/B must pair into 2 region movements: {units}"
        assert units[("MXSA", "MXSB")] == 1, f"unsplit MX group must move whole: {units}"

        prog = run_pipeline(lower_to_gir(th))
        produced, awaited = _units(prog)
        assert not (awaited - produced), \
            f"per_region={per_region}: awaited but never produced: {sorted(awaited-produced,key=str)}"

        # full-token check: the AB group is per-region under per_region pi, the MX group never is
        ptok = {t for blk in prog.blocks.values() for i in blk.body
                if isinstance(i, Move) and i.token and any(d.tile.space == "shared" for d in i.dsts)
                for t in [i.token]}
        ab = {t for t in ptok if t[1] == ("A", "B")}
        mx = {t for t in ptok if t[1] == ("MXSA", "MXSB")}
        assert all(len(t) == 3 for t in mx), f"unsplit group must not carry a region key: {mx}"
        if per_region:
            assert all(len(t) > 3 for t in ab), f"split group must carry a region key: {ab}"
            assert len({t[3:] for t in ab}) == 2, f"expected 2 region keys on A/B: {ab}"
        else:
            assert all(len(t) == 3 for t in ab), f"shared pi must keep 3-tuples: {ab}"


# ----------------------------------------------------------- the full 6-axis reorder space
_SIX_AXIS = {"TDMSplit": [2, 2, 2, 2], "MIWaveTile": [4, 4],
             "DepthU": 256, "MatrixInstruction": [16, 16, 64, 1]}


def _six_axis_words():
    from itertools import permutations
    return sorted({"".join(p) for p in permutations("KKMMNN")})


@pytest.mark.parametrize("plr", [0, 1])
def test_every_six_axis_reorder_emits_and_lowers(plr):
    """ALL 90 six-axis loop orders must decode, discharge, and lower -- including the 84 that are
    NON-CONTIGUOUS and so exercise the multi-body peel.
    """
    from Tensile.LoopModel import build_S, validate_loopir
    from Tensile.LoopModel import adapter
    words = _six_axis_words()
    assert len(words) == 90

    shapes = set()
    for w in words:
        for per_region in (False, True):
            th = adapter.params_to_theta(
                {**_SIX_AXIS, "LoopOrder": w, "PerRegionCompletion": per_region,
                 # PLR>0 USED TO BE UNREACHABLE HERE.
                 "PrefetchLocalRead": plr})
            S, _floor = build_S(th)
            r = emit_mainloop(th)
            assert r.obligations_discharged, f"{w} (per_region={per_region}): ledger not empty"
            assert validate_loopir(th, r.ir, S) == [], f"{w}: {validate_loopir(th, r.ir, S)}"

            # FULL PIPELINE, including the region walk.  Six live axes means each operand is
            # split on TWO region axes at once (`M_split` x `K_split`) -- the 2-D shape.
            prog = run_pipeline(lower_to_gir(th))
            verify_gir(prog)
            c = gir_counts(prog)["steady"]
            shapes.add((c["read"], c["copy"], c["mma"]))

    # ONE shape across every order: the reorder moved work, it did not add or lose any.
    assert len(shapes) == 1, f"loop order changed the work issued per steady trip: {sorted(shapes)}"
    reads, copies, mma = shapes.pop()
    assert mma == 2 ** 6, f"expected 2^6 wmma for six extent-2 axes, got {mma}"
    assert (reads, copies) == (32, 8), f"unexpected steady read/copy counts: {reads}/{copies}"


@pytest.mark.parametrize("plr", [0, 1, 2])
@pytest.mark.parametrize("order", ["KMN", "MNK", "MKN"])
def test_a_group_rotating_over_TWO_reduction_rate_modes_gets_a_SHIFTED_SLOT(plr, order):
    """The boundary `_SIX_AXIS` does not cover."""
    from Tensile.Lowering.gir import check_plan
    try:
        th = _theta({**BF16_NT_KMN, "DepthU": 128, "MatrixInstruction": [16, 16, 32, 1],
                     "LoopOrder": order, "PrefetchLocalRead": plr, "TDMSplit": [1, 1, 2, 2]})
    except RuntimeError as e:                     # no S carries this read-ahead here
        if "gives S" not in str(e):
            raise
        pytest.skip(f"{order} PLR{plr}: {str(e)[:70]}")
    assert {m.name: m.extent for m in summation_axes(th)} == {"K_split": 2, "K_inner": 2}
    assert check_plan(build_gir(th)) == []


def test_the_shifted_rate_slot_is_the_rate_index_OF_THE_SHIFTED_POSITION():
    """The identity the digit form implements, checked against an independent evaluation."""
    from Tensile.LoopModel import build_S
    from Tensile.LoopModel.traversal import _shifted_ring_slot, ring_axes, varying_axes, axis_strides
    from Tensile.LoopModel.schedule import _readahead_shift
    th = _theta({**BF16_NT_KMN, "DepthU": 128, "MatrixInstruction": [16, 16, 32, 1],
                 "PrefetchLocalRead": 1, "TDMSplit": [1, 1, 2, 2]})
    S, _ = build_S(th)
    op = next(o for o in th.operands if o.name == "A")
    grp = op.fragment.groups()[0]
    d = S.get(op.name, grp)
    shift, strides, n_pres, _ext = _readahead_shift(th, op, 1, S, None)
    rm = [(n, e) for n, e in ring_axes(th, op, grp) if n in strides]
    assert len(rm) > 1, "the probe must actually exercise the multi-mode path"
    e = _shifted_ring_slot(th, op, grp, shift, strides, d)
    pres = [m for m in varying_axes(th, op)]
    exts = {m.name: m.extent for m in th.inner_axes()}
    seen = 0
    for combo in __import__("itertools").product(*[range(exts[m]) for m in pres]):
        env = dict(zip(pres, combo))
        P = sum(strides[m] * env[m] for m in pres)
        radix, want = 1, 0
        for name, ext in reversed(rm):                    # innermost fastest, as `ring_slot`
            want += radix * (((P + shift) // strides[name]) % ext)
            radix *= ext
        assert e.eval(env) == want % d, (env, e.eval(env), want % d)
        seen += 1
    assert seen == __import__("math").prod(exts[m] for m in pres)


# ===========================================================================
# G0 -- the terminator predicate is STRUCTURED (read by field, not parsed).
# ===========================================================================
def test_terminator_pred_is_structured_not_a_string():
    """Every CondGoto predicate must expose `lhs` / `op` / `rhs:Bound` as fields.  Before G0 the
    predicate was a free-form `expr: str`, so a consumer wanting "which counter, against what
    bound?" -- exactly what a multi-exit header (G1) and the drain-multiplicity matrix (G3) need --
    had to parse `'iter < T - 2'` back apart."""
    from Tensile.Lowering.gir.nodes import CondGoto, Bound
    prog = lower_to_gir(_theta(BF16_NT_KMN))
    terms = {lab: b.term.pred for lab, b in prog.blocks.items()
             if isinstance(b.term, CondGoto)}
    assert terms, "no conditional terminators"
    for lab, p in terms.items():
        assert isinstance(p.lhs, str) and p.lhs, f"{lab}: lhs is not a counter symbol"
        assert isinstance(p.rhs, Bound), f"{lab}: rhs is not a structured Bound"
        assert not hasattr(p, "expr"), f"{lab}: the free-form `expr` string is back"
    # the peel-validity guard tests the trip symbol against the peel depth, by field
    g = terms["prologue"]
    assert (g.lhs, g.op, g.rhs.var) == ("T", ">", ""), f"guard shape {g!r}"
    assert g.rhs.const >= 1 and g.render() == f"T > {g.rhs.const}"
    # the steady loop no longer carries a PREDICATE at all: a counted loop states its TRIP COUNT ,
    # so there is nothing to test-against and nothing for a reader to decode.
    st = prog.block("steady").term
    assert isinstance(st, LoopBack), f"steady must carry a LoopBack, got {st!r}"
    assert (st.trips.var, st.trips.sub, st.trips.div) == ("T", g.rhs.const, 1), (
        f"loop must run T - {g.rhs.const} times; got {st.trips.render()}")
    assert st.trips.render() == f"T - {g.rhs.const}"
    # and the loop carries NO predicate to decode -- that is the point of.  There is no counter
    # start, no step direction and no test position for a reader to assume, so the two off-by-ones
    # those assumptions produced, and the guard strictness they forced, have nowhere to live.
    assert not hasattr(st, "pred"), "a counted loop must not carry a comparison"


def test_gir_pred_rejects_a_shape_bound_cannot_hold():
    """The LoopIR->GIR predicate translation is total over terminator shapes and RAISES otherwise.
    The old code rendered to a string, which could never fail -- and could never be read back."""
    from Tensile.LoopModel.ir import Pred as LMPred, Expr
    from Tensile.Lowering.loopir_to_gir import _gir_pred
    # a bare counter vs a constant is fine
    ok = _gir_pred(LMPred(Expr(var="iter"), "<", 4))
    assert (ok.lhs, ok.op, ok.rhs.const) == ("iter", "<", 4)
    # a MODULAR lhs (a residue pin, not a loop test) has no Bound representation
    with pytest.raises(NotImplementedError):
        _gir_pred(LMPred(Expr(var="iter", mod=2), "==", 1))
    # ...nor does a mixed-radix rhs
    with pytest.raises(NotImplementedError):
        _gir_pred(LMPred(Expr(var="iter"), "<", Expr(terms=(("a", 2), ("b", 1)))))


def test_scaffold_label_preserves_the_structured_predicate():
    """ScaffoldMapPass attaches an advisory label; it must not flatten the predicate back to a
    string while doing so (rebuilding `Pred(t.pred.expr, label)` loses it)."""
    from Tensile.Lowering.gir.nodes import CondGoto, Bound
    prog = build_gir(_theta(BF16_NT_KMN))
    for lab, blk in prog.blocks.items():
        if isinstance(blk.term, CondGoto):
            p = blk.term.pred
            assert isinstance(p.rhs, Bound), f"{lab}: label pass flattened the predicate"
            assert p.lhs and p.op, f"{lab}: predicate fields lost"


# ===========================================================================
# the short-loop (Cond.els) arm is dropped EXPLICITLY, not silently.
# ===========================================================================
def test_short_loop_arm_is_lowered_to_real_blocks_with_its_guard_chain():
    """The `T < M` arm becomes REAL blocks.  Summarized to {steps, guards} instead, its
    instructions thrown away on the unchecked claim that the scaffold rebuilds it from
    prologue+drain.  What the lowering owes: one block per peeled chunk, the peel-validity
    false edge landing on the first of them, and the `T > t` guard chain wired so step t-1 decides
    whether step t runs."""
    from Tensile.Lowering.gir.nodes import Bound, CondGoto, Goto
    prog = lower_to_gir(_theta(BF16_NT_KMN))
    M = prog.meta["peel_depth"]
    sl = prog.meta.get("short_loop")
    assert sl is not None and sl["steps"] == M
    assert sl["verdict"] == "unfolded", "the lowering emits both arms; the PASS decides the fold"
    assert sl["guards"][0] is None, "step 0 needs no guard (T > 0 is trivially true)"
    for t, g in enumerate(sl["guards"][1:], start=1):
        assert g is not None and (g.lhs, g.op) == ("T", ">"), f"step {t} guard shape {g!r}"
        assert g.rhs == Bound(const=t), f"step {t} must be guarded by T > {t}, got {g.rhs!r}"

    short = [b for b in prog.blocks.values() if str(b.phase).startswith("short")]
    assert len(short) == M, f"expected {M} short blocks, got {[b.phase for b in short]}"
    assert all(b.body for b in short), "a short block with no instructions is a dropped step"
    # the peel-validity guard's false arm enters the short chain, NOT the drain
    assert prog.block("prologue").term.f_target == "short0"
    assert "drain0" not in prog.block("prologue").succs
    # guard chain: short{t-1} decides step t; the last step leaves the region
    for t in range(M - 1):
        term = prog.block(f"short{t}").term
        assert isinstance(term, CondGoto) and term.t_target == f"short{t+1}"
        assert term.pred.rhs == Bound(const=t + 1)
    assert isinstance(prog.block(f"short{M-1}").term, Goto)


def test_short_loop_shape_mismatch_is_rejected():
    """The shape assertions survive the move from "justification for dropping" to "validation of
    what we lower": a short arm that is not M concrete guarded steps must raise at build time."""
    from Tensile.Lowering.loopir_to_gir import _short_steps
    from Tensile.LoopModel.ir import Cond, Pred, Expr
    from Tensile.LoopModel.emit import emit_mainloop
    th = _theta(BF16_NT_KMN)
    ir = emit_mainloop(th).ir
    els = list(ir[0].els)
    M = len(els)
    assert len(_short_steps(els, M)) == M                     # the real arm passes
    with pytest.raises(RuntimeError):                         # wrong step count
        _short_steps(els, M + 1)
    with pytest.raises(RuntimeError):                         # a step that is not a pinned Bind
        _short_steps([Cond(pred=Pred(Expr(var="T"), ">", 0), then=[], els=[],
                           kind="first_touch")] + els[1:], M)


# ===========================================================================
# G1 -- multi-exit loop headers (an ORDERED chain of guarded exits).
# ===========================================================================
def _chain_prog():
    """A minimal CFG whose header ends in a 2-arm CondChain: two typed early-exits into
    progressively-drained variants, else fall through to the steady loop."""
    from Tensile.Lowering.gir.nodes import (Program, Block, Pred, Bound, Goto, CondGoto,
                                            CondChain)
    prog = Program(entry="head")
    prog.add_block(Block(phase="prologue", succs=("d0", "d1", "steady"),
                         term=CondChain(((Pred("c", "<=", Bound(const=1)), "d0"),
                                         (Pred("c", "<=", Bound(const=2)), "d1")), "steady")))
    prog.blocks["head"] = prog.blocks.pop("prologue")
    prog.blocks["head"].phase = "prologue"
    prog.add_block(Block(phase="steady", loop=True, preds=("head", "steady"),
                         succs=("steady", "d0"),
                         term=CondGoto(Pred("c", "<", Bound(var="T")), "steady", "d0")))
    prog.add_block(Block(phase="drain0", preds=("head", "steady"), succs=("end",),
                         term=Goto("end")))
    prog.blocks["d0"] = prog.blocks.pop("drain0")
    prog.add_block(Block(phase="drain1", preds=("head",), succs=("end",), term=Goto("end")))
    prog.blocks["d1"] = prog.blocks.pop("drain1")
    return prog


def test_condchain_arms_are_ordinary_forward_edges():
    """Every guarded exit must appear as a normal CFG successor, IN ORDER (first arm wins), so
    dominance and back-edge detection need no special case for a multi-exit header."""
    from Tensile.Lowering.gir.analyses.cfg import successors, BackEdges
    from Tensile.Lowering.gir.analysis import AnalysisManager
    prog = _chain_prog()
    assert successors(prog)["head"] == ["d0", "d1", "steady"], "arm order not preserved"
    be = AnalysisManager().get(BackEdges(), prog)
    assert {e.header for e in be} == {"steady"}, "chain confused back-edge detection"


def test_condchain_verifies_and_renders():
    """verify_gir accepts the multi-exit shape: G-TERM resolves every arm target and agrees with
    the declared succs, and G-CFG still sees exactly one loop header."""
    from Tensile.Lowering.gir.render import render_gir
    prog = _chain_prog()
    assert verify_gir(prog) is True
    txt = render_gir(prog)
    assert "CondChain" in txt and "c <= 1" in txt and "c <= 2" in txt


def test_condchain_rejects_a_missing_arm_target():
    """A dangling arm target is a G-TERM violation like any other bad terminator target."""
    from Tensile.Lowering.gir.nodes import Pred, Bound, CondChain
    prog = _chain_prog()
    h = prog.blocks["head"]
    h.term = CondChain(((Pred("c", "<=", Bound(const=1)), "nope"),), "steady")
    h.succs = ("nope", "steady")
    with pytest.raises(RuntimeError):
        verify_gir(prog)


def test_condchain_is_well_formed_by_construction():
    """An empty chain or a malformed arm is rejected at construction -- a 0-arm chain is a Goto."""
    from Tensile.Lowering.gir.nodes import CondChain, Pred, Bound
    with pytest.raises(ValueError):
        CondChain((), "steady")
    with pytest.raises(ValueError):
        CondChain((("not a pred", "d0"),), "steady")


def test_scaffold_map_labels_every_chain_arm_by_its_target():
    """ScaffoldMapPass maps each generic exit to THE NAME TENSILELITE USES, which is the reason the
    pass exists -- so where the scaffold already has a name for a branch, that name wins over the
    generic role.  The deepest drain step is reached by TensileLite's `toPGR1` (its "PGR>=2 but only
    one loop" finalization: skip every NGLL and run the last stage); the other drain entries are its
    `NoGlobalLoadLoop_k`, keyed on the STEP they land on so the numbering matches
    `KernelWriter._lmDrainStep` rather than an arm ordinal."""
    from Tensile.Lowering.gir.passes.scaffold_map import ScaffoldMapPass
    from Tensile.Lowering.gir.analysis import AnalysisManager
    prog = _chain_prog()
    ScaffoldMapPass().run(prog, AnalysisManager())
    labels = [p.label for p, _t in prog.blocks["head"].term.arms]
    assert labels == ["NoGlobalLoadLoop_0", "toPGR1"], labels
    # the predicate itself survives the labelling, still structured
    p0 = prog.blocks["head"].term.arms[0][0]
    assert (p0.lhs, p0.op, p0.rhs.const) == ("c", "<=", 1)


# ===========================================================================
# G2 -- multi-block steady body (fallthrough chain, ONE shared back-edge).
# ===========================================================================
@pytest.mark.parametrize("n", [1, 2, 3])
def test_loop_copies_chain_shape(n):
    """`loop_copies=n` emits n fallthrough-chained steady blocks: only the LAST carries the
    back-edge, only the FIRST carries the phis, and the chain is still ONE loop."""
    from Tensile.Lowering.gir.analyses import BackEdges
    from Tensile.Lowering.gir.analysis import AnalysisManager
    from Tensile.Lowering.gir.nodes import Goto, CondGoto
    prog = lower_to_gir(_theta(BF16_NT_KMN), loop_copies=n)
    assert verify_gir(prog) is True
    labels = ["steady"] + [f"steady{i}" for i in range(1, n)]
    assert [l for l in prog.blocks if l.startswith("steady")] == labels
    for i, lab in enumerate(labels):
        blk = prog.blocks[lab]
        last = (i == n - 1)
        assert bool(blk.phis) == (i == 0), f"{lab}: phis belong to the header only"
        assert bool(blk.xfers) == last, f"{lab}: xfers belong to the last copy only"
        # the LAST copy carries the loop terminator; the others just fall through.  A counted
        # loop's terminator is a `LoopBack` stating the TRIP COUNT, not a comparison -- and
        # its divisor is the chunks one trip consumes, i.e. the number of copies.
        assert isinstance(blk.term, LoopBack if last else Goto), f"{lab}: wrong terminator"
        if last:
            assert blk.term.trips.div == n, (
                f"{lab}: a {n}-copy chain consumes {n} chunks per trip, so the count divides by "
                f"{n}; got {blk.term.trips.render()}")
    be = AnalysisManager().get(BackEdges(), prog)
    assert [(e.src, e.header) for e in be] == [(labels[-1], "steady")], "not one shared back-edge"


def test_loop_copies_residue_timeline_spans_the_chain():
    """The point of replicating the body: copy i sits i chunks further along the ONE residue
    timeline, so each copy's buffer index is a compile-time residue.  Copy 1 must be copy 0's
    sequence advanced by one generation on the ring -- not a restart at 0."""
    from Tensile.Lowering.gir.nodes import Move
    prog = lower_to_gir(_theta(BF16_NT_KMN), loop_copies=2)
    reach = _reaching(prog)

    def seq(lab):
        out = []
        for i in prog.blocks[lab].body:
            if isinstance(i, Move):
                for r in list(i.srcs) + list(i.dsts):
                    if getattr(r, "gen", None) is not None:
                        out.append(reach.of(r))
        return out

    ring = list(prog.blocks["steady"].phis)[0].gen.ring
    s0, s1 = seq("steady"), seq("steady1")
    assert s0 and len(s0) == len(s1)
    assert s1 == [(g + 1) % ring for g in s0], f"copy 1 is not one chunk ahead: {s0} vs {s1}"
    # and the back-edge advances by the whole chain, not by one
    xf = list(prog.blocks["steady1"].xfers)[0]
    assert xf.adv == 2, "back-edge must advance by the number of chunks a trip consumes"


def test_gen_entry_propagates_along_a_fallthrough_chain():
    """A copy block has no phis, so its entry value must be INHERITED from its single predecessor.
    Defaulting to 0 only looked right because the header's entry_val is 0; with a nonzero entry the
    chain would silently restart the timeline."""
    from Tensile.Lowering.gir.nodes import Move
    prog = lower_to_gir(_theta(BF16_NT_KMN), loop_copies=2)
    from dataclasses import replace
    blk = prog.blocks["steady"]                 # shift the whole timeline by one generation
    blk.phis = [replace(phi, entry_val=1) for phi in blk.phis]
    prog.bump()
    reach = _reaching(prog)
    first = next(reach.of(r) for i in prog.blocks["steady1"].body if isinstance(i, Move)
                 for r in list(i.srcs) + list(i.dsts) if getattr(r, "gen", None) is not None)
    assert first == 0, f"copy 1 ignored the header's entry value (got {first}, want (1+1)%2=0)"


# ===========================================================================
# G4 -- the Return terminator (a function-early-exit sink).
# ===========================================================================
def test_return_is_a_sink_with_no_successors():
    """A drain variant that finishes the kernel in place ends in `Return`.  It must contribute NO
    CFG edges -- distinct from `Goto('end')`, which is a real (filtered) edge a pass could retarget."""
    from Tensile.Lowering.gir.nodes import Program, Block, Goto, Return
    from Tensile.Lowering.gir.analyses.cfg import successors
    prog = Program(entry="prologue")
    prog.add_block(Block(phase="prologue", succs=("drain0",), term=Goto("drain0")))
    prog.add_block(Block(phase="drain0", preds=("prologue",), term=Return()))
    assert successors(prog)["drain0"] == [], "Return must have no successors"
    assert verify_gir(prog) is True, "Return is the one terminator allowed no targets"


def test_return_renders_and_survives_dominance():
    """Dominance / back-edge detection must simply skip a Return block rather than trip over a
    terminator with no targets."""
    from Tensile.Lowering.gir.nodes import Program, Block, Pred, Bound, CondGoto, Return
    from Tensile.Lowering.gir.analyses import BackEdges, Dominators
    from Tensile.Lowering.gir.analysis import AnalysisManager
    from Tensile.Lowering.gir.render import render_gir
    prog = Program(entry="steady")
    prog.add_block(Block(phase="steady", loop=True, preds=("steady",), succs=("steady", "drain0"),
                         term=CondGoto(Pred("c", "<", Bound(var="T")), "steady", "drain0")))
    prog.add_block(Block(phase="drain0", preds=("steady",), term=Return()))
    am = AnalysisManager()
    assert {e.header for e in am.get(BackEdges(), prog)} == {"steady"}
    assert "drain0" in am.get(Dominators(), prog)
    assert "term: Return" in render_gir(prog)


def test_missing_terminator_is_still_rejected():
    """`Return` legitimises a terminator with no TARGETS -- it must not legitimise a block with no
    TERMINATOR (the G-TERM check that catches a dropped tail)."""
    from Tensile.Lowering.gir.nodes import Program, Block, Goto
    prog = Program(entry="prologue")
    prog.add_block(Block(phase="prologue", succs=("drain0",), term=Goto("drain0")))
    prog.add_block(Block(phase="drain0", preds=("prologue",), term=None))
    with pytest.raises(RuntimeError):
        verify_gir(prog)


# ===========================================================================
# G3 (structural half) -- a tail loop is a legitimate SECOND loop region.
def _tail_prog(tail_phase="tail"):
    """steady (loop) -> drain0 -> tail (its own loop, own counter) -> end."""
    from Tensile.Lowering.gir.nodes import (Program, Block, Pred, Bound, Goto, CondGoto)
    prog = Program(entry="steady")
    prog.add_block(Block(phase="steady", loop=True, preds=("steady",), succs=("steady", "drain0"),
                         term=CondGoto(Pred("iter", "<", Bound(var="T")), "steady", "drain0")))
    prog.add_block(Block(phase="drain0", preds=("steady",), succs=("tail",), term=Goto("tail")))
    blk = Block(phase=tail_phase, loop=True, preds=("drain0", tail_phase),
                succs=(tail_phase, "end"),
                term=CondGoto(Pred("kt", "<", Bound(var="Ktail")), tail_phase, "end"))
    prog.add_block(blk)
    return prog


def test_tail_loop_is_an_accepted_second_loop_region():
    """The tail region has its OWN header, counter and back-edge.  G-CFG must count it separately
    from the reduction loop rather than rejecting the program for having two headers."""
    from Tensile.Lowering.gir.analyses import BackEdges
    from Tensile.Lowering.gir.analysis import AnalysisManager
    prog = _tail_prog()
    assert {e.header for e in AnalysisManager().get(BackEdges(), prog)} == {"steady", "tail"}
    assert verify_gir(prog) is True


def test_a_second_non_tail_loop_is_still_rejected():
    """The exemption is scoped to the tail PHASE -- a genuine second reduction loop must still
    fail, else G-CFG would have been weakened rather than made precise."""
    prog = _tail_prog(tail_phase="steady2")     # same shape, but not the tail region
    with pytest.raises(RuntimeError):
        verify_gir(prog)


def test_scaffold_map_labels_the_tail_back_edge_distinctly():
    """The tail back-edge is a different scaffold branch from the steady loop-exit compare."""
    from Tensile.Lowering.gir.passes.scaffold_map import ScaffoldMapPass
    from Tensile.Lowering.gir.analysis import AnalysisManager
    prog = _tail_prog()
    ScaffoldMapPass().run(prog, AnalysisManager())
    assert prog.blocks["steady"].term.pred.label == "LoopEndL"
    assert prog.blocks["tail"].term.pred.label == "TailLoopBegin"


def test_entry_block_exists_for_a_one_hop_kernel():
    """A direct-to-register (one-hop) kernel stages nothing through shared, so it has NO prologue
    block; the Program entry must name a block that exists.  With a single self-looping steady
    block a missing root happened to work out -- a multi-block loop body has no dominating entry at
    all and both copies then look like loop headers."""
    from Tensile.LoopModel import adapter
    import loopmodel_scenarios as scenarios
    _desc, params = scenarios.SCENARIOS["dtv"]
    th = adapter.params_to_theta(params)
    for n in (1, 2, 3):
        prog = lower_to_gir(th, loop_copies=n)
        assert prog.entry in prog.blocks, f"entry {prog.entry!r} is not a block"
        assert verify_gir(prog) is True


# ======================================================================= pointer def-use gate
def _peel_exit(prog):
    """The peel region's LAST block -- where the peel-validity guard lives.

    Not always the block named "prologue": a guarded prefetch generation splits the peel, and the
    guard that decides steady-vs-drain then sits on the join both arms reach.
    """
    from Tensile.Lowering.gir.verify_dataflow import prologue_blocks
    return prologue_blocks(prog)[-1]


def _simulate_pointer(prog, hop, unit, ring, trips, ncopies):
    """Walk the FINALIZED GIR as hardware would and return every place the physical pointer does
    not hold the generation the access names.
    """
    from Tensile.Lowering.gir.analyses.swap_regions import _hop_access
    from Tensile.Lowering.gir.verify_dataflow import prologue_blocks
    ptr, bad = 0, []
    # The prologue is a REGION, not a block: a guarded prefetch generation has its own, and a
    # walk that skips it misses that arm's swaps and reads every later generation off by one.
    path = [(lab, None) for lab in prologue_blocks(prog)]
    for v in range(trips):
        for c in range(ncopies):
            path.append((("steady" if c == 0 else f"steady{c}"), v * ncopies + c))
    path += [(f"drain{i}", trips * ncopies + i) for i in range(prog.meta.get("peel_depth", 0))]
    for lab, chunk in path:
        if lab not in prog.blocks:
            continue
        blk = prog.blocks[lab]
        rel = blk.gen_rel or 0
        for inst in blk.body:
            if isinstance(inst, Mark) and inst.kind == "swap" \
               and inst.at.get("hop") == hop and _subject(inst) == tuple(_as_tuple(unit)):
                ptr = (ptr + inst.at.get("steps", 1)) % ring
                continue
            acc = _hop_access(prog, inst, hop)
            if acc is None or acc[0] != unit:
                continue
            ref = acc[1]
            if getattr(ref, "gen", None) is not None:
                want = (chunk + (ref.gdelta - rel)) % ring
            elif getattr(ref, "abs_gen", None) is not None:
                want = ref.abs_gen % ring
            else:
                continue
            if ptr != want:
                bad.append((lab, chunk, want, ptr))
    return bad


def _pointer_hops(prog):
    """{(hop, unit): ring} for every physical pointer the program uses, `unit` as its hop names it."""
    from Tensile.Lowering.gir.analyses.swap_regions import _hop_access
    out = {}
    for blk in prog.blocks.values():
        for inst in blk.body:
            for hop in ("copy", "read"):
                acc = _hop_access(prog, inst, hop)
                if acc is not None:
                    g = getattr(acc[1], "gen", None)
                    ring = g.ring if g is not None else 1
                    out[(hop, acc[0])] = max(out.get((hop, acc[0]), 1), ring)
    return out


@pytest.mark.parametrize("order", ["KMN", "KNM", "MKN", "MNK", "NKM", "NMK"])
def test_pointer_never_reads_the_wrong_buffer_all_orders(order):
    """Every copy/read must find its physical pointer already on the generation its Ref names, on
    BOTH trip-count parities (the drain's buffer depends on it)."""
    from Tensile.LoopModel import adapter
    params = dict(BF16_NT_KMN, LoopOrder=order)
    prog = build_gir(adapter.params_to_theta(params))
    for (hop, operand), ring in _pointer_hops(prog).items():
        for trips in (2, 3, 4, 5):
            bad = _simulate_pointer(prog, hop, operand, ring, trips, 1)
            assert not bad, (f"{order} {hop}/{operand} trips={trips}: pointer holds the wrong "
                             f"generation at {bad[:4]}")


@pytest.mark.parametrize("params,label", [
    (BF16_NT_KMN, "PLR1"),
    (BF16_NT_KMN_PLR0, "PLR0"),
    (dict(BF16_NT_KMN, PrefetchGlobalRead=1), "PGR1"),
    (dict(BF16_NT_KMN, PrefetchGlobalRead=3), "PGR3"),
])
def test_pointer_never_reads_the_wrong_buffer_pipeline_depths(params, label):
    """PLR0 is the case a value-only analysis misses: no access inside a trip demands the rotation,
    so the whole rotation is the back-edge wrap and there is no in-body def site for it."""
    from Tensile.LoopModel import adapter
    prog = build_gir(adapter.params_to_theta(params))
    for (hop, operand), ring in _pointer_hops(prog).items():
        for trips in (2, 3, 4, 5):
            bad = _simulate_pointer(prog, hop, operand, ring, trips, 1)
            assert not bad, f"{label} {hop}/{operand} trips={trips}: wrong generation at {bad[:4]}"


@pytest.mark.parametrize("ncopies", [2, 3])
def test_pointer_correct_across_multi_block_loop_body(ncopies):
    """G2: the residue timeline spans the fallthrough chain, so the pointer must stay right across
    every copy of the body and across the single shared back-edge."""
    from Tensile.LoopModel import adapter
    from Tensile.Lowering.gir.passes import pipeline, run_pipeline as _run
    prog = lower_to_gir(adapter.params_to_theta(BF16_NT_KMN), loop_copies=ncopies)
    _run(prog, passes=pipeline())
    for (hop, operand), ring in _pointer_hops(prog).items():
        for trips in (2, 3):
            bad = _simulate_pointer(prog, hop, operand, ring, trips, ncopies)
            assert not bad, f"ncopies={ncopies} {hop}/{operand}: wrong generation at {bad[:4]}"


# ======================================================================= gr_increment dataflow
def _gr_increments(prog):
    """{block: [(unit, chunks)]} of gr_increment Marks actually in the finalized body.

    `unit` is the Phi movement's member tuple -- `('A',)` unfused."""
    out = {}
    for lab, blk in prog.blocks.items():
        got = [(_subject(i), i.at.get("chunks", 1)) for i in blk.body
               if isinstance(i, Mark) and i.kind == "gr_increment"]
        if got:
            out[lab] = got
    return out


@pytest.mark.parametrize("ncopies", [1, 2, 3])
def test_gr_increment_advances_one_chunk_per_copy(ncopies):
    """Each operand's global-read address must advance exactly once per reduction chunk fetched --
    M times across the prologue (M-1 between peel fills + the exit hand-off), then `adv` times per
    steady trip -- and each advance must sit AFTER the copy whose fetch it supersedes.
    """
    from Tensile.LoopModel import adapter
    from Tensile.Lowering.gir.passes import pipeline, run_pipeline as _run
    prog = lower_to_gir(adapter.params_to_theta(BF16_NT_KMN), loop_copies=ncopies)
    _run(prog, passes=pipeline())

    from Tensile.Lowering.gir.verify_dataflow import prologue_blocks
    incs = _gr_increments(prog)
    body = [lab for lab in prog.blocks if lab.startswith("steady")]
    # The peel is a REGION: a guarded prefetch generation owns its own block, and the advance that
    # supersedes its copies sits in the join both arms reach.
    peel = set(prologue_blocks(prog))
    assert set(body) <= set(incs), \
        f"every steady body block must advance the address, got {sorted(incs)}"
    # and NOTHING outside the peel + the steady body may advance it (the drain's global reads are
    # scaffold-owned; an increment there would double-advance).  This negative case had been
    # dropped when the scope widened to the prologue.
    assert set(incs) <= set(body) | peel, \
        f"gr_increment in a block that owns no copies: {sorted(set(incs) - set(body) - peel)}"

    adv = max((xf.adv for lab in body for xf in prog.blocks[lab].xfers), default=1)
    assert adv == ncopies
    for operand in ("A", "B"):
        per_trip = sum(c for lab in body for op, c in incs.get(lab, []) if operand in op)
        assert per_trip == adv, f"{operand}: {per_trip} chunks per trip, expected {adv}"
        pro = sum(c for lab in peel for op, c in incs.get(lab, []) if operand in op)
        assert pro == prog.meta["peel_depth"], (f"{operand}: {pro} prologue advances, expected "
                                       f"M = {prog.meta['M']} (peel fills + hand-off)")

    # each advance FOLLOWS its own operand's copy (the superseded version required the reverse; see
    # test_gr_increment_follows_the_copy_that_used_the_descriptor).  Across the peel REGION, since
    # the guarded generation's copies and the advance that supersedes them are in different blocks.
    def _ordered_advance_follows_copy(labels):
        seen_copy = set()
        for lab in labels:
            for inst in prog.blocks[lab].body:
                if isinstance(inst, Mark) and inst.kind == "gr_increment":
                    assert set(_subject(inst)) <= seen_copy, \
                        f"{lab}: advance of {_subject(inst)} with no preceding copy to supersede"
                elif isinstance(inst, Move):
                    for d in inst.dsts:
                        if d.tile.space == "shared":
                            seen_copy.add(d.tile.operand)

    _ordered_advance_follows_copy(prologue_blocks(prog))
    for lab in prog.blocks:
        if lab not in peel:
            _ordered_advance_follows_copy([lab])


def test_gr_increment_first_copy_has_no_advance():
    """The very first copy of each operand carries NO increment before it: the tile setup leaves
    the address on that chunk, so advancing first would skip chunk 0 entirely.
    """
    from Tensile.LoopModel import adapter
    prog = build_gir(adapter.params_to_theta(BF16_NT_KMN))
    from Tensile.Lowering.gir.analyses.gr_increment import _copy_of, _chunk_of
    seen_inc, npeel = set(), {}
    for lab, blk in prog.blocks.items():
        for inst in blk.body:
            if isinstance(inst, Mark) and inst.kind == "gr_increment":
                seen_inc.add(_subject(inst))
                continue
            acc = _copy_of(inst, prog)
            if acc is None:
                continue
            op = acc[0]
            k = npeel.get(op, 0)
            chunk = _chunk_of(blk, acc[1], peel_seq=k)
            if getattr(acc[1], "gen", None) is None:
                npeel[op] = k + 1
            if chunk == 0:                       # the first chunk this operand fetches
                assert op not in seen_inc, \
                    f"{op}: an advance precedes the copy of chunk 0 -- chunk 0 would be skipped"


# ===================================================================== placement POLICY
def _is_read(n):
    return (isinstance(n, Move) and any(s.tile.space == "shared" for s in n.srcs)
            and any(d.tile.space == "register" for d in n.dsts))


def _is_copy_of(n, unit):
    """Is `n` the copy Move of the Phi movement `unit` (a tuple of member operand names)?"""
    if not (isinstance(n, Move) and any(s.tile.space == "global" for s in n.srcs)):
        return False
    return tuple(d.tile.operand for d in n.dsts if d.tile.space == "shared") == tuple(unit)


def test_tdm_descriptor_changes_land_at_the_earliest_point_in_their_window():
    """Standing directive (lowering-design.md 7.1): ALL TDM descriptor changes -- `gr_increment` AND
    the copy-hop `swap` -- land as early as their window allows, i.e. immediately after the copy
    whose use of the descriptor they supersede, with no other Move in between.  Every cycle between
    the update and the `tensor_load` it feeds is overlap.

    Covers the PROLOGUE, not just the steady body, and that is the point: the steady windows here
    are ONE slot wide, so `earliest` and `midpoint` resolve to the same index and a steady-only test
    cannot tell the two policies apart.  The prologue windows are 4-5 slots wide.

    Asserts the RESOLVED position in the finalized body, not `pm.region.after` -- the Region is the
    window `ValuePlacementSolver` derived, and says nothing about which point inside it was chosen.
    """
    prog = build_gir(_theta(BF16_NT_KMN))
    checked = 0
    for block in ("prologue", "steady"):
        body = prog.blocks[block].body
        for idx, inst in enumerate(body):
            if not isinstance(inst, Mark):
                continue
            if inst.kind == "swap" and inst.at.get("hop") != "copy":
                continue
            if inst.kind not in ("gr_increment", "swap"):
                continue
            unit = _subject(inst)
            prior = [i for i, n in enumerate(body[:idx]) if _is_copy_of(n, unit)]
            assert prior, f"{block}/{inst.kind} {unit}: no preceding copy to supersede"
            between = [n for n in body[prior[-1] + 1:idx] if isinstance(n, Move)]
            assert not between, (
                f"{block}/{inst.kind} {unit} at {idx} is not at the earliest point: "
                f"{len(between)} Move(s) sit between it and the copy it supersedes")
            checked += 1
    assert checked >= 6, f"only {checked} TDM descriptor changes examined"


def test_lds_read_swap_lands_mid_window_not_at_its_earliest_point():
    """The LDS read pointer is bounded by two DIFFERENT resources -- the last read of `gen_from`
    below, the first read of `gen_to` above -- so it goes MIDWAY through the swapped usage rather
    than being slammed to an edge like a TDM descriptor change.  The two rules are deliberately
    different; this is the assertion that separates them.

    A weaker "reads on both sides" check holds at the earliest point too, so it cannot distinguish
    the midpoint policy from the descriptor one.  The real property is that at least one read sits
    strictly BETWEEN the swap and the bottom of its window.
    """
    prog = build_gir(_theta(BF16_NT_KMN))
    body = prog.blocks["steady"].body
    hits = [(i, n) for i, n in enumerate(body)
            if isinstance(n, Mark) and n.kind == "swap" and n.at.get("hop") == "read"]
    assert hits, "expected a read-hop swap in the steady body"
    interior = 0
    for idx, inst in hits:
        operand = inst.at["operand"]
        before = [i for i, n in enumerate(body[:idx])
                  if _is_read(n) and any(s.tile.operand == operand for s in n.srcs)]
        after = [i for i, n in enumerate(body[idx + 1:], start=idx + 1)
                 if _is_read(n) and any(s.tile.operand == operand for s in n.srcs)]
        assert before and after, f"read swap {operand} at {idx} sits outside its read run"
        if any(_is_read(n) for n in body[before[-1] + 1:idx]):
            interior += 1
    assert interior, ("every read swap sits at the earliest point of its window; the midpoint "
                      "policy is not in effect")


def test_pointer_oracle_reports_the_true_ring_for_every_hop():
    """Guard on the ORACLE, not the production code."""
    prog = build_gir(_theta(BF16_NT_KMN))
    rings = _pointer_hops(prog)
    assert rings, "expected pointers"
    for (hop, operand), ring in rings.items():
        assert ring >= 2, (
            f"{hop}/{operand} reported ring {ring}; with a double-buffered LDS every pointer "
            f"rotates, and ring 1 makes _simulate_pointer vacuous for it")


# ============================================================ bridge: the fuse is DERIVED
class _FakeDataType:
    def numBytes(self):
        return 2


def _kernel_dict(**over):
    """The minimum of a real Solution dict that `adapter.kernel_to_params` reads."""
    k = {
        "MatrixInstruction": [16, 16, 32, 1], "DepthU": 64,
        "MIWaveTileA": 2, "MIWaveTileB": 2,
        "ProblemType": {"DataType": _FakeDataType()},
        "enableTDMA": True, "enableTDMB": True, "NumWaves": 1, "TDMFuse": 0,
    }
    k.update(over)
    return k


@pytest.mark.parametrize("fuse", [0, 1])
def test_bridge_PASSES_THROUGH_the_fuse_instead_of_deriving_it(fuse):
    """`TDMFuse` is a Solution parameter; the bridge must READ it, not re-derive it."""
    from Tensile.LoopModel.adapter import kernel_to_params
    assert kernel_to_params(_kernel_dict(TDMFuse=fuse))["TDMFuse"] == fuse


def test_the_fuse_flag_reaches_theta_as_a_GROUP():
    """End to end: the flag becomes an actual Phi group, not just a number in a dict."""
    from Tensile.LoopModel.adapter import kernel_to_params
    from Tensile.LoopModel import adapter
    unfused = adapter.params_to_theta(kernel_to_params(_kernel_dict(TDMFuse=0)))
    fused   = adapter.params_to_theta(kernel_to_params(_kernel_dict(TDMFuse=0, NumWaves=2)))
    assert unfused.fused_copy_groups == []
    assert [tuple(g) for g in fused.fused_copy_groups] == [("A", "B")]


def test_a_fuse_needs_a_wave_to_select_on():
    """A fused set is ONE descriptor with the wave index choosing the member, so several waves
    moving both tensors is the whole precondition: with the mover off, or with one wave, no row
    fuses anything however it is written."""
    from Tensile.LoopModel.adapter import kernel_to_params
    from Tensile.LoopModel import adapter
    for over in (dict(TDMFuse=0, NumWaves=4, enableTDMA=False, enableTDMB=False),
                 dict(TDMFuse=0, NumWaves=1)):
        th = adapter.params_to_theta(kernel_to_params(_kernel_dict(**over)))
        assert th.fused_copy_groups == [], "%s must leave A and B as independent movements" % over


# =============================== Phi discharge: one movement, one completion (#169b) ============
def test_fused_reads_await_the_MOVEMENT_not_a_member():
    """Every consumer of a Phi-fused copy must await the MOVEMENT's completion."""
    from Tensile.LoopModel import adapter
    from Tensile.LoopModel.emit import emit_mainloop
    from Tensile.LoopModel.checks import walk_insts
    from Tensile.LoopModel.ir import Load
    from Tensile.LoopModel import Space
    th = _fused_theta()
    assert [tuple(g) for g in th.fused_copy_groups] == [("A", "B")], "fixture is not fused"
    seen = 0
    for inst in walk_insts(emit_mainloop(th).ir):
        op = inst.op
        if not (isinstance(op, Load) and op.dst == Space.REGISTER and op.src == Space.SHARED):
            continue
        for a in (inst.awaits or ()):
            if a.kind != "RAW-residency":
                continue
            assert str(a.dep).startswith("A+B@"), (
                f"a read of {op.tokens} awaits {a.dep!r}; under Phi the only completion is the "
                f"movement's, so it must await A+B")
            seen += 1
    assert seen, "no read RAW-residency awaits found -- the test would be vacuous"


def test_unfused_awaits_are_byte_identical_to_a_per_operand_ledger():
    """The Phi keying must be an IDENTITY on every unfused kernel."""
    from Tensile.LoopModel import adapter
    from Tensile.LoopModel.emit import emit_mainloop
    from Tensile.LoopModel.checks import walk_insts
    th = adapter.params_to_theta(BF16_NT_KMN)
    deps = {str(a.dep) for i in walk_insts(emit_mainloop(th).ir) for a in (i.awaits or ())}
    assert deps and not any("+" in d for d in deps), \
        f"unfused kernel grew a fused dep name: {sorted(d for d in deps if '+' in d)}"
    assert {"A@global:iter", "B@global:iter"} <= deps, sorted(deps)


def test_fused_inst_never_carries_the_same_await_twice():
    """`_copy_inst` unions the members' obligations rather than concatenating them."""
    from Tensile.LoopModel.emit import emit_mainloop
    from Tensile.LoopModel.checks import walk_insts
    for pgr in (1, 2):
        for plr in (0, 1):
            th = _fused_theta(PrefetchGlobalRead=pgr, PrefetchLocalRead=plr)
            for inst in walk_insts(emit_mainloop(th).ir):
                keys = [(str(a.dep), a.counter, a.kind) for a in (inst.awaits or ())]
                assert len(keys) == len(set(keys)), \
                    f"PGR{pgr}/PLR{plr}: duplicate awaits {keys}"
            # the steady fused copy still carries BOTH members' vacating-read WARs
            steady = [i for i in walk_insts(emit_mainloop(th).ir)
                      if any(a.kind.endswith("WAR") for a in (i.awaits or ()))]
            wars = {str(a.dep) for i in steady for a in i.awaits if a.kind.endswith("WAR")}
            if plr == 0 and pgr == 1:
                continue                       # no shared-ring WAR at this depth
            assert any("A:" in w for w in wars) and any("B:" in w for w in wars), \
                f"PGR{pgr}/PLR{plr}: dedup swallowed a member's WAR: {sorted(wars)}"


def test_one_obligation_is_discharged_exactly_once_in_the_prologue():
    """An obligation is ONE thing, so it gets ONE boundary -- not one per consumer."""
    from Tensile.LoopModel.emit import emit_mainloop
    from Tensile.LoopModel.render import render_stream

    def prologue(theta):
        return render_stream(emit_mainloop(theta).ir).split("for iter while")[0]

    fused = prologue(_fused_theta())
    n = fused.count("await   A+B@global")
    assert n == 1, f"fused prologue states its ONE boundary {n} times:\n{fused}"

    plain = prologue(_theta(BF16_NT_KMN))
    assert plain.count("await   A@global") == 1 and plain.count("await   B@global") == 1, \
        f"unfused must keep one boundary PER OBLIGATION (two of them):\n{plain}"


def test_a_refilling_copy_re_establishes_the_boundary():
    """The boundary must not survive into the next trip -- the steady body's FIRST read carries it."""
    from Tensile.LoopModel.emit import emit_mainloop
    from Tensile.LoopModel.render import render_stream
    checked = 0
    for pgr in (1, 2):
        for plr in (0, 1):
            th = _fused_theta(PrefetchGlobalRead=pgr, PrefetchLocalRead=plr)
            txt = render_stream(emit_mainloop(th).ir)
            steady = txt.split("for iter while", 1)[1]
            b = steady.find("await   A+B@global")
            r = steady.find("ds_read")
            assert b != -1, f"PGR{pgr}/PLR{plr}: steady body carries NO residency boundary"
            assert r == -1 or b < r, (
                f"PGR{pgr}/PLR{plr}: a ds_read precedes the residency boundary in the steady body "
                f"-- the body inherited a boundary from the prologue, which trip 1's copy has "
                f"already invalidated by trip 2")
            checked += 1
    assert checked == 4


# ===================================== await Mark: the discharge boundary in GIR ========
def _boundaries(params, phase="steady"):
    """[(kind, scope, unit)] for the await Marks in `phase`, in emitted order."""
    acts = plan_block(build_gir(_theta(params)), phase)
    return [(a.at.get("kind"), a.at.get("scope"), a.at.get("unit"))
            for a in acts if a.kind == "await"]


def test_cross_agent_is_the_AGENT_COUNT_not_the_fuse():
    """restated after the model was corrected: as soon as two agents cooperate on a shared
    buffer, its hazards are cross-agent and a barrier is needed.
    """
    one_plain  = _theta(BF16_NT_KMN)
    many_fused = _theta(BF16_NT_KMN_FUSED_XAGENT)
    many_plain = _theta(BF16_NT_KMN_UNFUSED_MULTIAGENT)

    assert many_fused.fused_copy_groups and not many_plain.fused_copy_groups, \
        "fixtures must separate the axes"
    assert not any(one_plain.agent_distributed(o.name) for o in one_plain.operands), \
        "one agent is never cross-agent"
    for th in (many_fused, many_plain):
        assert [o.name for o in th.operands if th.agent_distributed(o.name)] == ["A", "B"], \
            "2+ agents makes every SHARED buffer cross-agent, fuse or no fuse"
    # the accumulator is never staged in shared memory, so it has no cross-agent buffer
    assert not many_plain.agent_distributed("C")


def test_same_agent_kernels_emit_NO_war_boundary():
    """The whole change must be invisible to every kernel that is not cross-agent."""
    for params in (BF16_NT_KMN, BF16_NT_KMN_PLR0):
        scopes = {sc for _k, sc, _u in _boundaries(params)}
        assert scopes <= {"wave"}, f"{params.get('TDMFuse')}: unexpected non-wave boundary {scopes}"


# ---------------------------------------------------------------------------------------------
# FrameHazards -- the LDS hazard edges and their TRIP DISTANCE
# ---------------------------------------------------------------------------------------------

def _without_hoist():
    """The pipeline with HoistCopiesPass removed -- the copies-last order it is measured against."""
    from Tensile.Lowering.gir.passes import HoistCopiesPass, pipeline
    return [p for p in pipeline() if not isinstance(p, HoistCopiesPass)]


def _hazards(params, hoist=True):
    from Tensile.Lowering.gir import FrameHazards
    prog = build_gir(_theta(params), pipeline=None if hoist else _without_hoist())
    return AnalysisManager().get(FrameHazards(), prog)


def _steady(hz):
    return [h for h in hz if h.producer.block == "steady" and h.consumer.block == "steady"]


def _hoist_pipeline(group, reads=False):
    from Tensile.Lowering.gir.passes import HoistCopiesPass
    out = _without_hoist()
    return out[:1] + [HoistCopiesPass(group=group, reads=reads)] + out[1:]


def _in_flight(params, pl):
    """Per LDS buffer, instructions from the copy that fills it to the next trip's first read.

    The window the load has to land in, so the number the copy's issue point exists to grow."""
    from Tensile.Lowering.gir.analyses.lds_buffers import LdsBufferIds
    prog = build_gir(_theta(params), pipeline=pl)
    tokens = AnalysisManager().get(LdsBufferIds(), prog)
    body = prog.block("steady").body
    first = {}
    for i, node in enumerate(body):
        for ref in (node.srcs if isinstance(node, Move) else ()):
            if ref.tile.space == "shared":
                for t in tokens.ids_for(ref):
                    first.setdefault(t, i)
    return {t: (len(body) - i) + first[t]
            for i, node in enumerate(body) if isinstance(node, Move)
            for ref in node.dsts if ref.tile.space == "shared"
            for t in tokens.ids_for(ref) if t in first}


def _steady_shape(params, pl):
    """`(copy positions, fence positions)` in the steady body."""
    body = build_gir(_theta(params), pipeline=pl).block("steady").body
    return ([i for i, n in enumerate(body) if isinstance(n, Move)
             and any(r.tile.space == "shared" for r in n.dsts)],
            [i for i, n in enumerate(body) if isinstance(n, Mark) and n.kind == "fence"])


def test_the_copies_are_hoisted_as_a_GROUP_so_they_keep_sharing_one_barrier():
    """Each copy alone would go earlier still, but they then straddle another operand's last read
    and a second proc-scoped sync has to go between them.  A barrier per trip costs more than the
    slots one copy gains, so the group form is the default and this is what it buys."""
    # No mover: the default row would fuse A and B at two waves, and the group hoist needs two
    # separate copies to group.
    params = dict(BF16_NT_KMN_PLR0, NumWaves=2, TDMInst=0, PrefetchGlobalRead=2, DepthU=64)
    plain_c, plain_f = _steady_shape(params, _without_hoist())
    group_c, group_f = _steady_shape(params, _hoist_pipeline(True))
    solo_c, solo_f = _steady_shape(params, _hoist_pipeline(False))
    assert len(group_c) == len(plain_c) > 1, "the fixture must emit several separate copies"
    assert group_c < plain_c, "the group hoist must move the copies earlier"
    assert len(group_f) <= len(solo_f), "the per-copy form is what pays the extra barrier"


def test_split_tdm_copies_follow_their_individual_wmma_hazard_floors():
    """Grouping must not collapse region copies back to one loop-order-blind issue point."""
    from Tensile.Lowering.gir import check_plan

    base = {
        "MIWaveTile": [4, 4],
        "MatrixInstruction": [16, 16, 128, 1, 1, 4, 4, 2, 2],
        "DepthU": 256, "ElemBytes": 1,
        "PrefetchGlobalRead": 2, "PrefetchLocalRead": 1,
        "TDMSplit": [2, 2, 1, 1], "TDMSplitWaveRegions": [2, 2],
        "NumWaves": 4, "TDMFuse": 0,
    }

    def positions(prog):
        return [(i, dict(next(ref for ref in inst.dsts
                             if ref.tile.space == "shared").tile.coord))
                for i, inst in enumerate(prog.block("steady").body)
                if isinstance(inst, Move) and any(ref.tile.space == "shared"
                                                  for ref in inst.dsts)]

    for order in ("KKMNMN", "KMNKMN"):
        theta = adapter.params_to_theta({**base, "LoopOrder": order})
        grouped = build_gir(theta)
        individual = build_gir(theta, pipeline=_hoist_pipeline(False, reads=True))
        assert check_plan(grouped) == []
        assert positions(grouped) == positions(individual)
        assert positions(grouped)[0][0] < positions(grouped)[1][0]


def test_copy_hoist_never_crosses_a_stationary_copy():
    """A split copy may hoist to its own floor, but never ahead of an earlier stationary issue."""
    from Tensile.Lowering.gir.nodes import Block, Ref, Tile
    from Tensile.Lowering.gir.passes.hoist_copies import hoisted

    def copy(name, split=False):
        coord = (("M_split", 0),) if split else ()
        return Move(
            (Ref(Tile(name, "global", coord=coord)),),
            (Ref(Tile(name, "shared", coord=coord), abs_gen=0),))

    body = [
        copy("MX0"), copy("A0", split=True), copy("A1", split=True),
        copy("MX1"), copy("A2", split=True), copy("A3", split=True),
    ]
    block = Block("prologue", body=list(body))
    moved = hoisted(block, (), group=True)
    result = block.body if moved is None else moved
    assert result == body


def test_read_hoist_is_rejected_when_it_moves_a_raw_fence_earlier():
    """Future reads must not force an entry drain; copies still move to their recomputed WAR floor."""
    from Tensile.Lowering.gir.nodes import Mma as GirMma

    params = dict(BF16_NT_KMN_FUSED_XAGENT, PrefetchLocalRead=1)
    prog = build_gir(_theta(params))
    body = prog.block("steady").body
    first_mma = next(i for i, node in enumerate(body) if isinstance(node, GirMma))
    reads = [i for i, node in enumerate(body) if isinstance(node, Move)
             and any(ref.tile.space == "register" for ref in node.dsts)]
    tensor_waits = [node.at["tensorcnt"] for node in body
                    if isinstance(node, Mark) and node.kind == "waitcnt"
                    and "tensorcnt" in node.at]
    copies = [i for i, node in enumerate(body) if isinstance(node, Move)
              and any(ref.tile.space == "shared" for ref in node.dsts)]
    fences = [i for i, node in enumerate(body)
              if isinstance(node, Mark) and node.kind == "fence"]
    assert any(pos > first_mma for pos in reads), "all reads were pulled ahead of WMMA usage"
    assert tensor_waits and tensor_waits[0] > 0
    assert copies and all(any(fence < copy for fence in fences) for copy in copies)


_FOLDED = {"MIWaveTile": [8, 8], "DepthU": 256, "PrefetchLocalRead": 1, "PrefetchGlobalRead": 2,
           "MatrixInstruction": [16, 16, 32, 1, 1, 8, 8, 2, 2], "ElemBytes": 2, "LoopOrder": "KMN"}


def test_a_FOLDED_read_hazards_with_every_coordinate_it_covers():
    """One folded read fills several tile coordinates, so it may touch the register of any of them.
    Keying on the written coordinate alone loses the edges to the other consumers, and a reorder
    then walks the refill past a wmma that still needs the old value."""
    from Tensile.Lowering.gir.analyses.reg_hazards import RegHazards
    from Tensile.Lowering.gir.analyses.frame_hazards import WAR
    plain = AnalysisManager().get(RegHazards(), build_gir(_theta(_FOLDED)))
    folded = AnalysisManager().get(RegHazards(), build_gir(
        _theta(dict(_FOLDED, ReadPhi={"A": 2, "B": 2}))))
    war = [h for h in folded.edges(kind=WAR) if h.consumer.block == "steady"]
    assert war, "the folded fixture must still file a rotation WAR"
    assert any(None in dict(h.consumer.coord).values() for h in war), \
        "a covered axis must be unpinned, or its other coordinates file no edge"
    assert len(folded) >= len(plain) // 2 > 0, (len(plain), len(folded))


def test_the_read_hoist_keeps_the_plan_valid_on_a_FOLDED_read():
    """`check_plan` is the independent judge: it reads the emitted acts, not the hazard set."""
    from Tensile.Lowering.gir import check_plan
    for q in (1, 2, 4):
        params = _FOLDED if q == 1 else dict(_FOLDED, ReadPhi={"A": q, "B": q})
        prog = build_gir(_theta(params), pipeline=_hoist_pipeline(True, reads=True))
        assert check_plan(prog) == [], (q, check_plan(prog)[:2])


def test_read_hoist_preserves_hazards_when_split_coordinates_alias_one_register():
    """A split coordinate is logical; compact (group, slot, unit) is physical register identity."""
    from collections import Counter
    from test_loopmodel import _mxf8_kernel
    from Tensile.Lowering.gir import check_plan
    from Tensile.Lowering.gir.analyses.reg_hazards import RegHazards

    kernel = dict(_mxf8_kernel("KMNKMN", 1))
    kernel.update(MatrixInstruction=[16, 16, 128, 1, 1, 4, 4, 2, 2],
                  MIWaveTileA=4, MIWaveTileB=4, PrefetchGlobalRead=2,
                  VectorWidthA=4, VectorWidthB=4)
    target = {
        "ReadVectorElems": {"MXSA": 16, "MXSB": 16},
        "ReadPhi": {"MXSA": 4, "MXSB": 4},
        "ReadRho": {"MXSA": 0, "MXSB": 0},
    }
    theta = adapter.params_to_theta(adapter.kernel_to_params(kernel, target))
    canonical = build_gir(theta, pipeline=_without_hoist())
    optimized = build_gir(theta)
    assert check_plan(canonical) == []
    assert check_plan(optimized) == []

    def signature(prog):
        hazards = AnalysisManager().get(RegHazards(), prog)
        return Counter((h.kind, h.producer.block, h.consumer.block, h.producer.operand,
                        h.producer.ring_pos, h.consumer.ring_pos, h.gap, h.cross_block)
                       for h in hazards)

    assert signature(canonical) == signature(optimized)


def test_dscnt_ranks_register_fills_instead_of_defaulting_every_site_to_zero():
    """DSCNT must match a register producer; the LDS-only matcher made every rank empty."""
    prog = build_gir(_theta(BF16_NT_KMN))
    waits = [node.at for block in prog.blocks.values() for node in block.body
             if isinstance(node, Mark) and node.kind == "waitcnt" and "dscnt" in node.at]
    assert waits and any(wait["dscnt"] > 0 for wait in waits), waits
    assert not [node for node in prog.block("prologue_join").body
                if isinstance(node, Mark) and node.kind == "waitcnt"
                and "dscnt" in node.at]


def test_potential_waits_stamp_both_counters_and_anchor_cross_agent_waits_at_fences():
    """Hazards are stamped before ranking; a fence is an anchor, never a counter issue."""
    from Tensile.Lowering.gir.analyses import PotentialWaits, TENSORCNT, DSCNT
    from Tensile.Lowering.gir.passes import pipeline
    from Tensile.Lowering.gir.passes.wait_counts import WaitCntPass

    pl = [item for item in pipeline() if not isinstance(item, WaitCntPass)]
    prog = build_gir(_theta(BF16_NT_KMN_FUSED_XAGENT), pipeline=pl)
    waits = AnalysisManager().get(PotentialWaits(), prog)
    assert {potential.counter for potential in waits} == {TENSORCNT, DSCNT}

    cross_agent = [potential for potential in waits if potential.hazard.cross_agent]
    assert cross_agent
    fm = AnalysisManager().get(FrameMap(), prog)
    for potential in cross_agent:
        body = prog.block(potential.block).body
        assert potential.pos < len(body)
        fence = body[potential.pos]
        assert isinstance(fence, Mark) and fence.kind == "fence"
        assert any(
            relation["kind"] == potential.hazard.kind
            and relation["producer"]["identity"] == id(potential.hazard.producer.inst)
            and relation["consumer"]["identity"] == id(potential.hazard.consumer.inst)
            and relation["producer"]["frame"] == fm.render(potential.producer_frame)
            and relation["consumer"]["frame"] == fm.render(potential.consumer_frame)
            for relation in fence.at["relations"])

    counts = prog.meta["read_instructions"]
    for (block, pos, _frame), info in waits.issues().items():
        node = prog.block(block).body[pos]
        if info.counter == TENSORCNT:
            assert info.width == 1
        else:
            operand = node.dsts[0].tile.operand
            assert info.width == counts[operand]
    for block in prog.blocks.values():
        for pos, node in enumerate(block.body):
            if not (isinstance(node, Mark) and node.kind == "fence"):
                continue
            assert all(waits.issue(block.label, pos, frame) is None
                       for frame in fm.frames(block.label))


def test_retained_window_boundary_agrees_in_forward_and_backward_counter_models():
    """A wait keeps ages strictly below n; equality means the producer retired."""
    from types import SimpleNamespace

    from Tensile.Lowering.gir.nodes import Block, Program, Return
    from Tensile.Lowering.gir.analyses import TENSORCNT
    from Tensile.Lowering.gir.analyses.backward_wait_counts import potential_ranks
    from Tensile.Lowering.gir.analyses.counter_flow import simulate_counterflow
    from Tensile.Lowering.gir.analyses.potential_waits import (
        IssueInfo, IssueKey, PotentialWait, PotentialWaitSet)

    prog = Program(entry="entry")
    prog.add_block(Block("entry", body=[Mark("phase_boundary") for _ in range(4)],
                         term=Return()))
    fm = AnalysisManager().get(FrameMap(), prog)
    frame = fm.frames("entry")[0]
    producer = IssueKey(TENSORCNT, "entry", 0, frame, ("test",))
    touch = SimpleNamespace(block="entry", pos=0, operand="A")
    consumer = SimpleNamespace(block="entry", pos=4, operand="A")
    hazard = SimpleNamespace(kind="RAW", producer=touch, consumer=consumer,
                             gap=0, cross_agent=False)
    potential = PotentialWait("entry", 4, frame, TENSORCNT, producer,
                              frame, frame, hazard)
    issues = {
        ("entry", pos, frame): IssueInfo(
            TENSORCNT, 1, (producer,) if pos == 0 else ())
        for pos in range(4)
    }
    potentials = PotentialWaitSet((potential,), issues)

    retained = {("entry", 2, TENSORCNT): 2}
    retired = {("entry", 2, TENSORCNT): 1}
    nested_retained = {("entry", 2, TENSORCNT): 2, ("entry", 3, TENSORCNT): 3}
    nested_retired = {("entry", 2, TENSORCNT): 2, ("entry", 3, TENSORCNT): 2}
    assert potential_ranks(prog, fm, potentials, potential, retained) == (3,)
    assert potential_ranks(prog, fm, potentials, potential, retired) == ()
    assert potential_ranks(prog, fm, potentials, potential, nested_retained) == (3,)
    assert potential_ranks(prog, fm, potentials, potential, nested_retired) == ()

    trace, _ = simulate_counterflow(prog, fm, potentials, retained)
    assert not trace.safe and trace.ages_at(potential.site) == (3,)
    trace, _ = simulate_counterflow(
        prog, fm, potentials, {**retained, potential.site: 3})
    assert trace.safe
    assert simulate_counterflow(prog, fm, potentials, retired)[0].safe
    assert simulate_counterflow(prog, fm, potentials, nested_retired)[0].safe


def test_counterflow_merge_ignores_absent_paths_and_chooses_one_strongest_wait():
    from types import SimpleNamespace

    from Tensile.Lowering.gir.nodes import (
        Block, Bound, CondGoto, Goto, Pred, Program, Return)
    from Tensile.Lowering.gir.analyses import TENSORCNT
    from Tensile.Lowering.gir.analyses.backward_wait_counts import potential_ranks
    from Tensile.Lowering.gir.analyses.counter_flow import (
        derive_counterflow_plan, simulate_counterflow)
    from Tensile.Lowering.gir.analyses.potential_waits import (
        IssueInfo, IssueKey, PotentialWait, PotentialWaitSet)

    prog = Program(entry="entry")
    prog.add_block(Block("entry", term=CondGoto(
        Pred("T", "==", Bound(const=1)), "left", "right")))
    prog.add_block(Block("left", body=[Mark("phase_boundary"), Mark("phase_boundary")],
                         term=Goto("merge")))
    prog.add_block(Block("right", body=[Mark("phase_boundary") for _ in range(3)],
                         term=Goto("merge")))
    prog.add_block(Block("merge", term=Return()))
    fm = AnalysisManager().get(FrameMap(), prog)
    frame = fm.frames("merge")[0]

    left = IssueKey(TENSORCNT, "left", 0, frame, ("left",))
    right = IssueKey(TENSORCNT, "right", 0, frame, ("right",))

    def potential(key):
        producer = SimpleNamespace(block=key.block, pos=0, operand=key.identity[0])
        consumer = SimpleNamespace(block="merge", pos=0, operand=key.identity[0])
        hazard = SimpleNamespace(kind="RAW", producer=producer, consumer=consumer,
                                 gap=0, cross_agent=False)
        return PotentialWait("merge", 0, frame, TENSORCNT, key, frame, frame, hazard)

    waits = PotentialWaitSet(
        (potential(left), potential(right)),
        {
            ("left", 0, frame): IssueInfo(TENSORCNT, 1, (left,)),
            ("left", 1, frame): IssueInfo(TENSORCNT, 1),
            ("right", 0, frame): IssueInfo(TENSORCNT, 1, (right,)),
            ("right", 1, frame): IssueInfo(TENSORCNT, 1),
            ("right", 2, frame): IssueInfo(TENSORCNT, 1),
        })
    decisions, trace, _observed = derive_counterflow_plan(prog, fm, waits)
    site = ("merge", 0, TENSORCNT)
    assert decisions == {site: 1}
    assert trace.safe and trace.ages_at(site) == (1, 2)
    assert potential_ranks(prog, fm, waits, waits.for_site(site)[0]) == (1,)
    assert potential_ranks(prog, fm, waits, waits.for_site(site)[1]) == (2,)

    # Removing both producing issues proves absence on every path, so no wait is required.
    empty = PotentialWaitSet(tuple(waits), {})
    assert simulate_counterflow(prog, fm, empty, {})[0].safe


def test_counter_solvers_keep_exact_logical_ranks_above_backend_field_width():
    """GIR never clamps either solver's logical rank to an encoding width."""
    from types import SimpleNamespace

    from Tensile.Lowering.gir.nodes import Block, Program, Return
    from Tensile.Lowering.gir.analyses import BackwardWaitCounts, CounterFlow
    from Tensile.Lowering.gir.analyses import TENSORCNT
    from Tensile.Lowering.gir.analyses.backward_wait_counts import potential_ranks
    from Tensile.Lowering.gir.analyses.counter_flow import derive_counterflow_plan
    from Tensile.Lowering.gir.analyses.potential_waits import (
        IssueInfo, IssueKey, PotentialWait, PotentialWaitSet)
    from Tensile.Lowering.gir.passes import pipeline
    from Tensile.Lowering.gir.passes.wait_counts import WaitCntPass

    linear = Program(entry="entry")
    linear.add_block(Block("entry", body=[Mark("phase_boundary") for _ in range(81)],
                           term=Return()))
    linear_fm = AnalysisManager().get(FrameMap(), linear)
    frame = linear_fm.frames("entry")[0]
    producer = IssueKey(TENSORCNT, "entry", 0, frame, ("tensor",))
    touch = SimpleNamespace(block="entry", pos=0, operand="A")
    hazard = SimpleNamespace(
        kind="RAW", producer=touch,
        consumer=SimpleNamespace(block="entry", pos=81, operand="A"),
        gap=0, cross_agent=False)
    potential = PotentialWait("entry", 81, frame, TENSORCNT, producer,
                              frame, frame, hazard)
    tensor_waits = PotentialWaitSet(
        (potential,),
        {("entry", pos, frame): IssueInfo(
            TENSORCNT, 1, (producer,) if pos == 0 else ())
         for pos in range(81)})
    assert potential_ranks(linear, linear_fm, tensor_waits, potential) == (80,)
    assert derive_counterflow_plan(linear, linear_fm, tensor_waits)[0] == {
        ("entry", 81, TENSORCNT): 80}

    pl = [item for item in pipeline() if not isinstance(item, WaitCntPass)]
    prog = build_gir(_theta(BF16_NT_KMN), pipeline=pl)
    prog.meta["read_instructions"] = {"A": 80, "B": 80}
    prog.bump()
    am = AnalysisManager()
    forward = {(site.block, site.pos, site.counter): site.n
               for site in am.get(CounterFlow(), prog)}
    backward = {(site.block, site.pos, site.counter): site.n
                for site in am.get(BackwardWaitCounts(), prog)}
    assert forward == backward
    assert max(forward.values()) > 63


@pytest.mark.parametrize("order", ["KMN", "KNM", "MKN", "MNK", "NKM", "NMK"])
def test_forward_and_backward_wait_solvers_agree_on_every_loop_order(order):
    from Tensile.Lowering.gir.analyses import differential_wait_counts
    from Tensile.Lowering.gir.passes import pipeline
    from Tensile.Lowering.gir.passes.wait_counts import WaitCntPass

    pl = [item for item in pipeline() if not isinstance(item, WaitCntPass)]
    prog = build_gir(_theta({**BF16_NT_KMN, "LoopOrder": order}), pipeline=pl)
    assert differential_wait_counts(prog, AnalysisManager()) == ()


def test_backward_wait_rounds_keep_the_loop_carried_mxf8_tensor_site():
    """Removing later candidates must not make an earlier round erase a still-live producer."""
    from test_loopmodel import _mxf8_kernel

    from Tensile.Lowering.gir.analyses import CounterFlow, differential_wait_counts
    from Tensile.Lowering.gir.passes import pipeline
    from Tensile.Lowering.gir.passes.wait_counts import WaitCntPass

    kernel = dict(_mxf8_kernel("KMN", 0))
    kernel.update(PrefetchGlobalRead=2, LocalReadVectorWidthA=16,
                  LocalReadVectorWidthB=16)
    theta = adapter.params_to_theta(adapter.kernel_to_params(kernel))
    prog = build_gir(theta, pipeline=[item for item in pipeline()
                                     if not isinstance(item, WaitCntPass)])
    am = AnalysisManager()
    assert differential_wait_counts(prog, am) == ()
    waits = list(am.get(CounterFlow(), prog))
    assert any(site.block == "steady" and site.counter == "tensorcnt"
               and site.n == 2 and site.producer == "MXSA" for site in waits)


def test_loopir_orders_smaller_prefetch_ring_reads_and_their_copies_first():
    """MX has W=1 while A/B have W=2, so MX owns the first same-site issue slots."""
    from test_loopmodel import _mxf8_kernel

    from Tensile.Lowering.gir.nodes import copy_unit
    from Tensile.Lowering.gir.passes import ScaffoldShapePass
    from Tensile.Lowering.gir.passes.hoist_copies import HoistCopiesPass, _read_generation

    kernel = dict(_mxf8_kernel("MNK", 0))
    kernel.update(PrefetchGlobalRead=2, PrefetchLocalRead=1)
    target = {
        "ReadVectorElems": {"MXSA": 16, "MXSB": 16},
        "ReadPhi": {"MXSA": 4, "MXSB": 4},
        "ReadRho": {"MXSA": 0, "MXSB": 0},
    }
    theta = adapter.params_to_theta(adapter.kernel_to_params(kernel, target))
    prog = lower_to_gir(theta)
    prologue_reads = [
        (node.dsts[0].tile.operand, dict(node.dsts[0].tile.coord))
        for node in prog.block("prologue").body
        if isinstance(node, Move)
        and any(ref.tile.space == "register" for ref in node.dsts)]
    for k in (0, 1):
        same_use = [operand for operand, coord in prologue_reads
                    if coord.get("K_inner", 0) == k
                    and coord.get("M_inner", 0) == 0
                    and coord.get("N_inner", 0) == 0]
        assert same_use[:4] == ["MXSA", "MXSB", "A", "B"]
    prologue_copies = [copy_unit(node)[0] for node in prog.block("prologue").body
                       if copy_unit(node)[0] is not None]
    assert prologue_copies == [
        ("MXSA", "MXSB"), ("A", "B"), ("MXSA", "MXSB"), ("A", "B")]

    before = [copy_unit(node)[0] for node in prog.block("steady").body
              if copy_unit(node)[0] is not None]
    first_generation = next(_read_generation(node) for node in prog.block("steady").body
                            if _read_generation(node) is not None)
    reads_before = [node.dsts[0].tile.operand for node in prog.block("steady").body
                    if _read_generation(node) == first_generation]
    am = AnalysisManager()
    ScaffoldShapePass().run(prog, am)
    am.invalidate()
    HoistCopiesPass(reads=True).run(prog, am)
    after = [copy_unit(node)[0] for node in prog.block("steady").body
             if copy_unit(node)[0] is not None]
    reads_after = [node.dsts[0].tile.operand for node in prog.block("steady").body
                   if _read_generation(node) == first_generation]
    assert before == after
    assert reads_before == reads_after
    assert after[0] == ("MXSA", "MXSB")
    assert all(unit == ("A", "B") for unit in after[1:])


def test_waitcnt_insertion_is_idempotent_and_has_one_mark_per_counter_site():
    from Tensile.Lowering.gir.passes.wait_counts import WaitCntPass

    prog = build_gir(_theta(BF16_NT_KMN))
    before = {block.label: tuple((node.kind, repr(node.at))
                                 for node in block.body
                                 if isinstance(node, Mark) and node.kind == "waitcnt")
              for block in prog.blocks.values()}
    WaitCntPass().run(prog, AnalysisManager())
    after = {block.label: tuple((node.kind, repr(node.at))
                                for node in block.body
                                if isinstance(node, Mark) and node.kind == "waitcnt")
             for block in prog.blocks.values()}
    assert after == before
    for block in prog.blocks.values():
        waits = [(pos, next(key for key in ("tensorcnt", "dscnt") if key in node.at))
                 for pos, node in enumerate(block.body)
                 if isinstance(node, Mark) and node.kind == "waitcnt"]
        assert len(waits) == len(set(waits))


def test_a_too_weak_existing_wait_is_tightened_without_duplicate_exact_waits():
    from Tensile.Lowering.gir.analyses import WaitCounts
    from Tensile.Lowering.gir.passes.wait_counts import WaitCntPass
    from Tensile.Lowering.tool.frame_faithful_check import audit

    prog = build_gir(_theta(BF16_NT_KMN))
    victim = next(node for block in prog.blocks.values() for node in block.body
                  if isinstance(node, Mark) and node.kind == "waitcnt"
                  and node.at.get("dscnt", 0) > 0)
    expected = victim.at["dscnt"]
    victim.at["dscnt"] = expected + 1
    prog.bump()
    corrections = list(AnalysisManager().get(WaitCounts(), prog))
    assert any(site.counter == "dscnt" and site.n == expected for site in corrections)
    WaitCntPass().run(prog, AnalysisManager())
    assert not audit(prog)[0]


def test_both_counter_solvers_fail_closed_on_an_unresolved_shared_identity():
    from Tensile.Lowering.gir.nodes import Block, Program, Ref, Return, Tile
    from Tensile.Lowering.gir.analyses import BackwardWaitCounts, CounterFlow

    prog = Program(entry="entry")
    prog.add_block(Block(
        "entry",
        body=[Move(
            (Ref(Tile("A", "global")),),
            (Ref(Tile("A", "shared")),))],
        term=Return()))
    for analysis in (CounterFlow(), BackwardWaitCounts()):
        with pytest.raises(RuntimeError, match="unresolved shared frame identity"):
            AnalysisManager().get(analysis, prog)


def test_frame_faithful_counter_replay_catches_missing_and_extra_waits():
    import copy

    from Tensile.Lowering.tool.frame_faithful_check import audit

    prog = build_gir(_theta(BF16_NT_KMN))
    assert not audit(prog)[0]

    missing = copy.deepcopy(prog)
    removed = False
    for block in missing.blocks.values():
        for pos, node in enumerate(block.body):
            if isinstance(node, Mark) and node.kind == "waitcnt":
                block.body.pop(pos)
                removed = True
                break
        if removed:
            break
    missing.bump()
    assert audit(missing)[0]["W1_under_wait"] > 0

    extra = copy.deepcopy(prog)
    extra.block(extra.entry).body.insert(0, Mark("waitcnt", {"tensorcnt": 0}))
    extra.bump()
    assert audit(extra)[0]["W2_over_wait"] > 0


def test_later_loop_carried_dscnt_zero_does_not_tighten_earlier_waits():
    """A late next-trip drain stays at loop bottom; it cannot rewrite an earlier residual to zero."""
    from test_loopmodel import _mxf8_kernel

    kernel = dict(_mxf8_kernel("KKMNMN", 1))
    kernel.update(MatrixInstruction=[16, 16, 128, 1, 1, 4, 4, 2, 2],
                  MIWaveTileA=4, MIWaveTileB=4, PrefetchGlobalRead=2,
                  VectorWidthA=4, VectorWidthB=4)
    target = {
        "ReadVectorElems": {"MXSA": 16, "MXSB": 16},
        "ReadPhi": {"MXSA": 4, "MXSB": 4},
        "ReadRho": {"MXSA": 0, "MXSB": 0},
    }
    from Tensile.Lowering.gir.analyses import differential_wait_counts
    from Tensile.Lowering.gir.passes import pipeline
    from Tensile.Lowering.gir.passes.wait_counts import WaitCntPass

    theta = adapter.params_to_theta(adapter.kernel_to_params(kernel, target))
    prog = build_gir(theta, pipeline=[item for item in pipeline()
                                     if not isinstance(item, WaitCntPass)])
    assert differential_wait_counts(prog, AnalysisManager()) == ()
    WaitCntPass().run(prog, AnalysisManager())
    waits = [(pos, node.at["dscnt"]) for pos, node in enumerate(prog.block("steady").body)
             if isinstance(node, Mark) and node.kind == "waitcnt" and "dscnt" in node.at]
    positive = [pos for pos, value in waits if value > 0]
    later_drains = [pos for pos, value in waits if value == 0 and positive and pos > min(positive)]
    assert positive and later_drains, waits


def test_waits_and_fences_retain_frame_relative_ring_relations():
    """The dump must expose `token=(frame+gdelta)%ring`, not only legacy token unions."""
    prog = build_gir(_theta(dict(BF16_NT_KMN_FUSED_XAGENT, PrefetchLocalRead=1)))
    marks = [(block.phase, pos, node) for block in prog.blocks.values()
             for pos, node in enumerate(block.body)
             if isinstance(node, Mark) and node.kind in ("waitcnt", "fence")]
    waits = [(phase, pos, node) for phase, pos, node in marks if node.kind == "waitcnt"]
    assert waits and all(node.at.get("relations") for _phase, _pos, node in waits)
    assert all(not ({"tensorcnt", "dscnt"} <= set(node.at)) for _p, _i, node in waits)
    assert not any(phase.startswith("prologue") and "dscnt" in node.at
                   for phase, _pos, node in waits)
    assert any(phase in ("steady", "drain0", "drain1") and "dscnt" in node.at
               for phase, _pos, node in waits)

    join = [(pos, node) for phase, pos, node in marks if phase == "prologue_join"]
    wait_pos = [pos for pos, node in join if node.kind == "waitcnt"
                and "tensorcnt" in node.at]
    fence_pos = [pos for pos, node in join if node.kind == "fence"]
    assert wait_pos and fence_pos
    assert any(wait + 1 == fence for wait in wait_pos for fence in fence_pos)

    text = render_gir(prog)
    assert "gdelta=" in text and "tok=(frame+" in text and "advance=" in text
    assert "uniform rotation only; no divergent entrances" in text
    assert "legacy_tokens" not in text


@pytest.mark.parametrize("group", [True, False])
@pytest.mark.parametrize("reads", [True, False])
def test_every_flag_combination_emits_a_valid_plan(group, reads):
    """Both flags are shipping options, so both settings of each are covered, not just the default.
    `check_plan` judges the emitted acts and `unseparated_edges` the cross-wave order."""
    from Tensile.Lowering.gir import check_plan
    from Tensile.Lowering.tool.fence_token_check import unseparated_edges
    for params in (dict(BF16_NT_KMN, NumWaves=2, PrefetchLocalRead=1, PrefetchGlobalRead=2,
                        DepthU=64),
                   dict(_FOLDED, ReadPhi={"A": 2, "B": 2}),
                   dict(BF16_NT_KMN_FUSED_XAGENT, PrefetchGlobalRead=2, PrefetchLocalRead=1)):
        prog = build_gir(_theta(params), pipeline=_hoist_pipeline(group, reads))
        assert check_plan(prog) == [], (group, reads, check_plan(prog)[:2])
        assert unseparated_edges(prog) == [], (group, reads)


def test_hoisting_the_copies_grows_the_window_the_load_has_to_land_in():
    """The point of the issue point: in-flight time, not position."""
    params = dict(BF16_NT_KMN_PLR0, NumWaves=2, PrefetchGlobalRead=2, DepthU=64)
    plain = _in_flight(params, _without_hoist())
    group = _in_flight(params, _hoist_pipeline(True))
    assert plain and set(plain) == set(group)
    assert all(group[t] > plain[t] for t in plain), (plain, group)


def test_hoisting_reads_does_not_shorten_the_window_by_advancing_a_tensor_wait():
    """A read hoist is rejected when its RAW fence would force tensor completion earlier."""
    params = dict(BF16_NT_KMN, NumWaves=2, PrefetchLocalRead=1, PrefetchGlobalRead=2, DepthU=64)
    copies_only = _in_flight(params, _hoist_pipeline(True))
    with_reads = _in_flight(params, _hoist_pipeline(True, reads=True))
    assert with_reads == copies_only


def test_distance_separates_the_cell_that_FAILED_from_the_ones_that_did_not():
    """The one number the backend's phase machine cannot compute, and the whole bug in one assert.
    An access at offset `g` on a ring of `S`, at trip `v` of a loop advancing by 1, touches buffer
    `(base + v + g) mod S`, so producer and consumer collide at trip distance `(g_p - g_c) mod S`.
    """
    rolling = _steady(_hazards(dict(BF16_NT_KMN_PLR0, PrefetchGlobalRead=1)))
    inplace = _steady(_hazards(BF16_NT_KMN_1LDS))
    assert rolling and inplace
    assert {h.distance for h in rolling} == {1}, "PGR1/PLR0 must be purely loop-carried"
    assert not any(h.same_trip for h in rolling)
    # Ring 1 -- so the fixture must BE ring 1: the refill collides with the read in its own trip and
    # again on the next.  The residue `(gdelta_a - gdelta_b) % 1` could only ever say 0, so the wrap
    # edge was invisible.  On a ring of 2 the same pair is {0, 2} and says nothing about the wrap.
    assert {h.distance for h in inplace} == {0, 1}, "a 1-buffer ring refills what is being read"
    assert any(h.same_trip for h in inplace)


def test_the_loop_carried_RAW_and_WAR_are_both_found_at_the_failing_cell():
    """Both directions, because they are different obligations.  Copies last, so the read precedes
    the refill in the text; the edges themselves are the same either way."""
    from Tensile.Lowering.gir.analyses.frame_hazards import RAW, WAR
    hz = _steady(_hazards(dict(BF16_NT_KMN_PLR0, PrefetchGlobalRead=1), hoist=False))
    # No WAW: a read always sits between two writes of a buffer, so the RAW into it and the WAR
    # out of it already order them.
    assert {h.kind for h in hz} == {RAW, WAR}
    assert all(h.in_program_order for h in hz if h.kind == WAR)
    assert not any(h.in_program_order for h in hz if h.kind == RAW)


def test_HOISTED_the_same_edges_are_found_with_the_text_order_reversed():
    """HoistCopiesPass moves loop-carried movements to the front, so `in_program_order` flips on
    edges whose producer moved.  The hazard SET must not: distance comes from the gdeltas, not from
    position, and it is the distance that decides which trip discharges the edge."""
    from Tensile.Lowering.gir.analyses.frame_hazards import RAW, WAR
    from Tensile.Lowering.gir.analyses.fence_regions import _admissible_slots
    params = dict(BF16_NT_KMN_PLR0, PrefetchGlobalRead=1)
    plain = _steady(_hazards(params, hoist=False))
    hoisted = _steady(_hazards(params))
    assert {h.kind for h in hoisted} == {RAW, WAR}
    assert {(h.kind, h.distance) for h in hoisted} == {(h.kind, h.distance) for h in plain}
    assert not any(h.same_trip for h in hoisted), "nothing here is same-trip, so nothing pins order"
    # Nothing is same-trip, so every edge takes the wrap branch and the flip cannot move a fence:
    # the admissible window is the same whichever side of the body the producer was hoisted to.
    n_slots = max(max(h.producer.pos, h.consumer.pos) for h in hoisted) + 2
    for h in hoisted:
        a, b = h.producer.pos, h.consumer.pos
        assert _admissible_slots(h, n_slots, "steady") == frozenset(
            s for s in range(n_slots) if s > a or s <= b)


def test_a_same_agent_edge_needs_NO_fence_which_is_why_single_wave_needs_no_barrier_pass():
    """the reused-slot WAR is discharged by program-order-on-issue, which relates events on
    ONE agent -- so a same-agent edge needs the counter, not a barrier.
    """
    for params in (BF16_NT_KMN, BF16_NT_KMN_PLR0, dict(BF16_NT_KMN_PLR0, PrefetchGlobalRead=1)):
        hz = _hazards(params)
        assert len(hz) > 0, "hazards must still be FOUND -- they are simply same-agent"
        assert hz.needing_fence() == (), "single-wave must need no fence"


def test_cross_agent_marks_the_edges_a_fence_must_cover():
    """Under a Phi fuse across agents the buffer is refilled by the agents that wrote it and read by
    all of them, so program order establishes nothing between the two and every edge needs a
    proc-scoped fence.  (Phi, not rho: the fuse is what puts two agents on one buffer.)"""
    hz = _hazards(BF16_NT_KMN_FUSED_XAGENT)
    assert len(hz.needing_fence()) == len(hz) > 0
    assert all(h.cross_agent for h in hz)


def test_WAW_is_found_at_S_eq_1_the_phase_machines_blind_spot():
    """Two writes to one buffer with nothing between them is an output dependence, and the
    backend's `_conflicts` fires only on read-after-writing / write-after-reading -- so it can
    never see this.  At `1LDSBuffer` the ring is 1, so consecutive refills genuinely collide."""
    from Tensile.Lowering.gir.analyses.frame_hazards import WAW
    hz = _hazards(BF16_NT_KMN_1LDS)
    assert {h.ring for h in hz} == {1}
    assert [h for h in hz if h.kind == WAW], "no WAW found on a single-buffer ring"


def test_distinct_rings_do_not_alias():
    """A's and B's shared buffers rotate independently and address disjoint storage (the rotation anti-dependence rule's
    premise), so no hazard may pair them -- otherwise every fence would over-cover."""
    for params in (BF16_NT_KMN_PLR0, BF16_NT_KMN_FUSED_XAGENT):
        for h in _hazards(params):
            assert h.producer.operand == h.consumer.operand, \
                f"paired distinct rings {h.producer.operand} / {h.consumer.operand}"


def test_cross_block_hazards_are_FOUND_not_left_to_the_block_boundary():
    """The steady copy feeds the drain's reads, and no fence inside the steady body separates them."""
    hz = _hazards(BF16_NT_KMN_FUSED_XAGENT)
    xb = [h for h in hz if h.cross_block]
    assert xb, "no cross-block hazard found at all"
    assert any(h.producer.block == "steady" and h.consumer.block.startswith("drain")
               and h.kind == "RAW" for h in xb), "the steady-copy -> drain-read RAW is missing"
    # A cross-block pair spans a block boundary, so its instances are one or more frames apart --
    # a real gap, where the residue arithmetic could only report `None` (#396, #401).
    assert all(h.gap >= 1 for h in xb), "a cross-block pair spans at least one frame step"
    assert hz.unresolved() == (), f"unnamed shared accesses remain: {hz.unresolved()}"


def test_every_hazard_is_TOKEN_VISIBLE_or_FENCE_SEPARATED():
    """The invariant this rests on, asserted rather than assumed."""
    from Tensile.Lowering.gir import FrameHazards
    for params in (BF16_NT_KMN_FUSED_XAGENT,
                   dict(BF16_NT_KMN_FUSED_XAGENT, PrefetchGlobalRead=1),
                   dict(BF16_NT_KMN_FUSED_XAGENT, PrefetchLocalRead=1)):
        prog = build_gir(_theta(params))
        applied = [i for blk in prog.blocks.values() for i in blk.body
                   if getattr(i, "kind", None) == "fence"]
        for h in AnalysisManager().get(FrameHazards(), prog):
            pt, ct = h.producer.inst.token, h.consumer.inst.token
            assert pt is not None and ct is not None, "a hazard endpoint carries no token"
            if set(pt[2]) & set(ct[2]):
                continue                                   # visible to the backend as an alias
            assert h.cross_agent, (
                f"{h.kind} d={h.distance} carries disjoint tokens {pt}/{ct} and is NOT cross-agent, "
                f"so nothing orders it: no token overlap and no fence")
            # ...and a fence must actually be THERE.
            assert applied, (f"{h.kind} {h.producer.block}->{h.consumer.block} has disjoint tokens "
                             f"and no fence Mark reached the body")


def _fences(params, loop_copies=1):
    """`loop_copies` IS PART OF THE FIXTURE, not a detail."""
    from Tensile.Lowering.gir import FenceRegions
    prog = (lower_to_gir(_theta(params), loop_copies=loop_copies) if loop_copies > 1
            else build_gir(_theta(params)))
    return prog, AnalysisManager().get(FenceRegions(), prog)


def _slot_of(prog, pm):
    body = prog.block(pm.region.block).body
    return body.index(pm.region.before) if pm.region.before in body else len(body)


def _prefence_program(params):
    """Program state immediately before CollectPendingMarksPass commits fence sites."""
    from Tensile.Lowering.gir.passes import pipeline

    prog = lower_to_gir(_theta(params))
    am = AnalysisManager()
    for transform in pipeline()[:3]:
        transform.run(prog, am)
        am.invalidate()
    return prog, am


def _mab_prefence(vector_width_a):
    from test_loopmodel import _mxf8_kernel

    from Tensile.Lowering.gir.passes import pipeline

    kernel = dict(_mxf8_kernel("MKNMKN", 0))
    kernel.update(
        MatrixInstruction=[16, 16, 128, 1, 1, 4, 1, 4, 1],
        MIWaveTileA=4, MIWaveTileB=1, NumWaves=4,
        PrefetchGlobalRead=2, PrefetchLocalRead=2,
        TDMFuse=3, TDMSplitA=1, TDMSplitB=0,
        VectorWidthA=vector_width_a, VectorWidthB=1,
        LocalReadVectorWidthMXS=8,
    )
    theta = adapter.params_to_theta(adapter.kernel_to_params(kernel))
    assert [tuple(group) for group in theta.fused_copy_groups] == [("B", "MXSA", "MXSB")]

    prog = lower_to_gir(theta)
    am = AnalysisManager()
    for transform in pipeline()[:3]:
        transform.run(prog, am)
        am.invalidate()
    return prog, am


def _fence_slot(prog, pending):
    """Identity, not `.index` -- GIR nodes compare equal and `.index` returns the first twin."""
    body = prog.block(pending.region.block).body
    return next((i for i, node in enumerate(body) if node is pending.region.before), len(body))


def test_a_fused_movement_is_published_by_one_fence():
    """A Phi-fused movement is ONE instruction, so its components share a producer and a fence.

    That is the property the deleted completion-class key existed to guarantee; keying on the
    producer gives it directly."""
    from Tensile.Lowering.gir import FenceRegions

    prog, am = _prefence_program(dict(BF16_NT_KMN_FUSED_XAGENT, PrefetchLocalRead=1))
    fused = [item for item in am.get(FenceRegions(), prog)
             if item.region.block == "prologue_join"
             and {edge.producer.operand for edge in item.edges} >= {"A", "B"}]
    assert len(fused) == 1, "the fused prologue movement was published by several fences"
    raws = [edge for edge in fused[0].edges if edge.kind == "RAW"]
    assert raws and len({id(edge.producer.inst) for edge in raws}) == 1


def test_every_cross_wave_edge_has_exactly_one_owning_fence():
    """`potential_waits.anchor_site` raises on two relation owners, and StinkyTofu rejects it too,
    so the cover must PARTITION the edges rather than merely hit them."""
    from Tensile.Lowering.gir import FenceRegions
    from Tensile.Lowering.gir.analyses import FrameHazards

    prog, am = _prefence_program(dict(BF16_NT_KMN_FUSED_XAGENT, PrefetchLocalRead=1))
    owners = {}
    for pending in am.get(FenceRegions(), prog):
        for edge in pending.edges:
            owners.setdefault(id(edge), []).append(pending)
    assert owners, "no fence owns any edge"
    assert all(len(owning) == 1 for owning in owners.values())
    assert set(owners) == {id(h) for h in am.get(FrameHazards(), prog).needing_fence()}


@pytest.mark.parametrize("plr", [0, 1])
def test_a_raw_fence_sits_where_its_wait_is_largest(plr):
    """A RAW's wait is computed at the fence OWNING it, so that fence must sit at the latest slot
    its own RAW edges allow -- or be separated from it by a span issuing nothing on that counter,
    which leaves the residual identical."""
    from Tensile.Lowering.gir import FenceRegions
    from Tensile.Lowering.gir.analyses.fence_regions import _admissible_slots
    from Tensile.Lowering.gir.analyses.wait_common import FANOUT_META, counter_for, issued

    prog, am = _prefence_program(dict(BF16_NT_KMN_FUSED_XAGENT, PrefetchLocalRead=plr))
    counts = (prog.meta or {}).get(FANOUT_META) or {}
    checked = 0
    for pending in am.get(FenceRegions(), prog):
        label, slot = pending.region.block, _fence_slot(prog, pending)
        body = prog.block(label).body
        raws = [edge for edge in pending.edges if edge.kind == "RAW"]
        if not raws:
            continue
        checked += 1
        common = set.intersection(
            *(set(_admissible_slots(edge, len(body) + 1, label)) for edge in raws))
        for later in (s for s in common if s > slot):
            assert all(issued(body, slot, later, counter_for(edge, prog), counts) == 0
                       for edge in raws), (
                f"{label}@{slot} could sit at {later}, where its producers are deeper")
    assert checked, "fixture produced no RAW-bearing fence"


def test_a_war_fence_sits_where_its_wait_is_largest():
    """A WAR stamps too: its producer is a shared READ, so it carries a dscnt whose value is also
    computed at its owning fence.  Riding whatever barrier happens to stand latest in its window
    drains the reads sooner than the edge requires."""
    from Tensile.Lowering.gir import FenceRegions
    from Tensile.Lowering.gir.analyses.fence_regions import _admissible_slots
    from Tensile.Lowering.gir.analyses.wait_common import FANOUT_META, counter_for, issued

    prog, am = _prefence_program(dict(BF16_NT_KMN_FUSED_XAGENT, PrefetchLocalRead=1))
    counts = (prog.meta or {}).get(FANOUT_META) or {}
    checked = 0
    for pending in am.get(FenceRegions(), prog):
        label, slot = pending.region.block, _fence_slot(prog, pending)
        body = prog.block(label).body
        wars = [edge for edge in pending.edges if edge.kind != "RAW"]
        if not wars:
            continue
        checked += 1
        common = set.intersection(
            *(set(_admissible_slots(edge, len(body) + 1, label)) for edge in wars))
        for later in (s for s in common if s > slot):
            assert all(issued(body, slot, later, counter_for(edge, prog), counts) == 0
                       for edge in wars), (
                f"{label}@{slot} could sit at {later}, where its reads are deeper")
    assert checked, "fixture produced no WAR-bearing fence"


def test_a_same_agent_kernel_gets_NO_fences():
    """Nothing to stand between: program order plus the completion counter already discharge every
 same-agent edge. The hazards are still FOUND -- this is a placement decision, not a
 blind spot -- and the backend independently agrees by skipping its barrier pass at one wave."""
    for params in (BF16_NT_KMN, BF16_NT_KMN_PLR0, dict(BF16_NT_KMN_PLR0, PrefetchGlobalRead=1)):
        _prog, pend = _fences(params)
        assert pend == []


def test_every_cross_agent_edge_has_a_fence_on_EVERY_path_between_its_ends():
    """The specification the cover must meet, checked by enumerating paths rather than by asking
    the implementation what it thinks it covered.
    """
    from Tensile.Lowering.gir import FrameHazards
    from Tensile.Lowering.gir.analyses.cfg import successors
    from Tensile.Lowering.gir.analyses.fence_regions import _admissible_slots

    def paths(succ, src, dst, seen=()):
        if src == dst and seen:
            yield list(seen) + [src]
            return
        if src in seen:
            return
        for nxt in succ.get(src, ()):
            yield from paths(succ, nxt, dst, tuple(seen) + (src,))
        if src == dst:
            yield [src]

    # The last two entries carry `loop_copies=2`: the multi-block steady chain is the only shape in
    # which a candidate fence can be reachable from the producer solely along the BACK EDGE, which
    # is what an existential reachability test accepts and a separator test rejects.
    for params, ncopies in ((BF16_NT_KMN_FUSED_XAGENT, 1),
                            (dict(BF16_NT_KMN_FUSED_XAGENT, PrefetchGlobalRead=1), 1),
                            (dict(BF16_NT_KMN_FUSED_XAGENT, PrefetchLocalRead=1), 1),
                            (BF16_NT_KMN_FUSED_XAGENT, 2),
                            (dict(BF16_NT_KMN_FUSED_XAGENT, PrefetchLocalRead=1), 2)):
        prog, pend = _fences(params, loop_copies=ncopies)
        succ = successors(prog)
        placed = {}
        for pm in pend:
            placed.setdefault(pm.region.block, []).append(_slot_of(prog, pm))
        for h in AnalysisManager().get(FrameHazards(), prog).needing_fence():
            pb, cb = h.producer.block, h.consumer.block
            if not h.cross_block:
                n = len(prog.block(cb).body) + 1
                assert _admissible_slots(h, n, cb) & set(placed.get(cb, [])), \
                    f"{h.kind} {pb} {h.producer.pos}->{h.consumer.pos} (d={h.distance}) unfenced"
                continue
            routes = [p for p in paths(succ, pb, cb) if len(p) >= 2]
            assert routes, f"no path {pb}->{cb} but a cross-block hazard was reported"
            # ONLY the ends' own blocks separate (#416): an intermediate block's fence sits on
            # some path, and the producer can re-reach the consumer around it via the back edge.
            on_path = (any(sl > h.producer.pos for sl in placed.get(pb, []))
                       or any(sl <= h.consumer.pos for sl in placed.get(cb, [])))
            assert on_path, f"{h.kind} {pb}->{cb} is fenced only in an intermediate block"


def test_an_undischargeable_edge_is_an_ERROR_not_a_fence_somewhere_harmless():
    """If no slot separates a pair, the emitted order itself is illegal. Placing a fence anywhere would hide a schedule
 defect behind a barrier that discharges nothing."""
    import pytest as _pytest
    from Tensile.Lowering.gir.analyses.fence_regions import _admissible_slots

    class _Bad:
        kind, gap, same_trip, in_program_order = "WAR", 0, True, True
        cross_block = False
        class producer: operand, pos, block = "A", 5, "steady"
        class consumer: operand, pos, block = "A", 2, "steady"   # before producer: empty window
    assert not _admissible_slots(_Bad(), 10, "steady")

    # ...and an edge admitting no slot must reach the caller as an ERROR, not a fence somewhere
    # harmless.  Starving the predicate is how a real illegal sigma presents.
    import Tensile.Lowering.gir.analyses.fence_regions as _fr
    prog, am = _prefence_program(dict(BF16_NT_KMN_FUSED_XAGENT, PrefetchLocalRead=1))
    keep = _fr._admissible_slots
    try:
        _fr._admissible_slots = lambda *a, **k: frozenset()
        with _pytest.raises(RuntimeError, match="undischargeable"):
            _fr.FenceRegions().run(prog, am)
    finally:
        _fr._admissible_slots = keep


def test_short_path_fold_compares_VALUES_not_producer_identities():
    """The coverage check must ask "does every wmma receive the same data", not "is the same
    instruction its producer".  Three ways the identity reading was wrong, each with its own
    observable consequence, so each gets its own assertion.
    """
    from Tensile.Lowering.gir.analysis import AnalysisManager
    from Tensile.Lowering.gir.analyses import ShortPathFold
    from Tensile.Lowering.gir.analyses.short_path import FOLD, SPLIT
    from Tensile.Lowering.gir import check_register_slots

    ORDERS = ("KMN", "KNM", "MKN", "MNK", "NKM", "NMK")

    def verdict(params, order):
        prog = lower_to_gir(_theta(dict(params, LoopOrder=order)))
        am = AnalysisManager(); check_register_slots(prog, am); am.invalidate()
        return am.get(ShortPathFold(), prog)

    # (1) the register LOCATION must name the free tile, not just the rotation slot.  Without it
    #     A(k1,m0) and A(k1,m1) share a key and every read looks like it clobbers its sibling.
    for order in ORDERS:
        v = verdict(BF16_NT_KMN, order)
        expected = FOLD
        assert v.verdict == expected, (
            f"{order}: expected {expected}, got {v.verdict} with {len(v.breaks)} break(s): "
            f"{[b.cause for b in v.breaks][:4]}")

    # (3) a read's value must chain to the COPY that filled its buffer, not stop at the buffer's
    # generation.
    for order in ORDERS:
        v1 = verdict(dict(BF16_NT_KMN, NumLdsBlk=1, **{"1LDSBuffer": 1}), order)
        assert v1.verdict == SPLIT and v1.breaks, (
            f"{order}: S_shared=1 with M=2 must NOT fold -- one buffer cannot hold two peeled "
            f"chunks at once")

        # and the genuine ring-depth collision is still caught (M=3 > ring 2) -- the Phase-1
        # finding, now derived from the def-use trace rather than asserted as `S_shared >= M`.
        v3 = verdict(dict(BF16_NT_KMN, PrefetchGlobalRead=3, PrefetchLocalRead=0), order)
        assert v3.verdict == SPLIT and v3.breaks, f"{order}: M=3 over a 2-deep ring must not fold"


def test_fold_short_path_pass_always_rewrites_to_the_emitted_scaffold():
    """The analysis may diagnose SPLIT, but finalized GIR has only the physical scaffold shape."""
    for order in ("KMN", "KNM", "MKN", "MNK", "NKM", "NMK"):
        _assert_fold_and_split_shapes(order)


def _assert_fold_and_split_shapes(order):
    from Tensile.Lowering import build_gir
    from Tensile.Lowering.gir.nodes import CondGoto, CondChain
    from Tensile.Lowering.gir.passes import folded

    # FOLD (ring >= M): the arm is deleted and the peel-validity false edge returns to the drain --
    # exactly TensileLite's `prologue -> Cond -> loop -> else -> drain`.
    prog = build_gir(_theta(dict(BF16_NT_KMN, LoopOrder=order)))   # PGR2/PLR1, ring 2 >= M 2
    assert folded(prog), order
    assert not [b for b in prog.blocks if b.startswith("short")]
    term = prog.block(_peel_exit(prog)).term
    # The coverage's contract is "the peel-validity FALSE arm lands in the drain chain head", not a
    # node class: `ScaffoldShapePass` runs `EarlyExitPass` after the coverage, which turns the two-way
    # guard into the `CondChain` of typed `T < M` entries (G1) whose DEFAULT is that same head.
    short_entry = term.default if isinstance(term, CondChain) else term.f_target
    assert short_entry == "drain0", term
    # the coverage RESTORES the prologue->drain0 edge, and preds must be recomputed to match it
    assert _peel_exit(prog) in prog.block("drain0").preds
    assert prog.meta["short_loop"]["obligations"], "the fold's preconditions must be carried"

    from Tensile.Lowering.gir.verify import verify_gir
    assert verify_gir(prog)


def test_short_fold_merges_value_order_and_preserves_drain_tensor_hazard():
    """The emitted PGR1 short path is prologue->drain, never a model-only substitute."""
    from test_loopmodel import _mxf8_kernel

    from Tensile.Lowering.gir.analyses import CounterFlow
    from Tensile.Lowering.gir.passes import folded, pipeline
    from Tensile.Lowering.gir.passes.wait_counts import WaitCntPass

    kernel = dict(_mxf8_kernel("KMNKMN", 1))
    kernel.update(
        MatrixInstruction=[16, 16, 32, 1, 1, 4, 4, 1, 1],
        MIWaveTileA=4, MIWaveTileB=4,
        PrefetchGlobalRead=1, PrefetchLocalRead=1,
        TDMFuse=0, TDMSplitA=1, TDMSplitB=1,
        VectorWidthA=1, VectorWidthB=1,
        LocalReadVectorWidthA=8, LocalReadVectorWidthB=8,
    )
    prog = build_gir(
        adapter.params_to_theta(adapter.kernel_to_params(kernel)),
        pipeline=[item for item in pipeline() if not isinstance(item, WaitCntPass)])

    assert folded(prog)
    assert not any(label.startswith("short") for label in prog.blocks)
    assert "drain0" in prog.block("prologue").succs
    assert "prologue" in prog.block("drain0").preds
    assert any(site.block == "drain0" and site.counter == "tensorcnt"
               for site in AnalysisManager().get(CounterFlow(), prog))


def test_mxf8_carrier_pairings_drive_the_short_fold_order_merge():
    """A carrier refill is the next chunk even when its within-chunk coordinates are unchanged."""
    from test_loopmodel import _mxf8_kernel

    from Tensile.Lowering.gir.analyses import ShortPathFold
    from Tensile.Lowering.gir.analyses.short_path import OVERWRITTEN, SPLIT
    from Tensile.Lowering.gir.passes import folded
    from Tensile.Lowering.gir.verify_dataflow import check_register_dataflow

    kernel = dict(_mxf8_kernel("MKN", 0))
    kernel.update(
        MatrixInstruction=[16, 16, 128, 1, 1, 2, 8, 2, 2],
        MIWaveTileA=2, MIWaveTileB=8,
        PrefetchGlobalRead=2, PrefetchLocalRead=1,
        TDMFuse=0, TDMSplitA=1, TDMSplitB=0,
        VectorWidthA=1, VectorWidthB=1,
        LocalReadVectorWidthMXS=8,
    )
    target = {
        "ReadVectorElems": {"MXSA": 8, "MXSB": 8},
        "ReadPhi": {"MXSA": 2, "MXSB": 2},
        "ReadRho": {"MXSA": 2, "MXSB": 2},
        "RegisterBudget": None,
    }
    theta = adapter.params_to_theta(adapter.kernel_to_params(kernel, target))

    raw = lower_to_gir(theta)
    verdict = AnalysisManager().get(ShortPathFold(), raw)
    assert verdict.verdict == SPLIT
    assert len(verdict.breaks) == len(verdict.ordering) == 8
    assert {brk.cause for brk in verdict.breaks} == {OVERWRITTEN}
    assert {item.block for item in verdict.ordering} == {"drain0"}

    prog = build_gir(theta)
    assert folded(prog)
    assert not any(label.startswith("short") for label in prog.blocks)
    assert check_register_dataflow(prog) == []


# ===========================================================================
# Coverage: the configurations that have actually broken something.
def _fold_matrix_cases():
    import loopmodel_scenarios as scenarios
    cases = [(f"scenario:{name}", params) for name, (_d, params) in
             sorted(scenarios.SCENARIOS.items())]
    cases += [("fused-xagent", BF16_NT_KMN_FUSED_XAGENT),
              ("unfused-multiagent", BF16_NT_KMN_UNFUSED_MULTIAGENT),
              ("1LDS (ring 1)", BF16_NT_KMN_1LDS),
              ("PLR0", BF16_NT_KMN_PLR0)]
    # W = 1: single-buffered registers, where the body's only pipeline is positional
    cases += [(f"W=1 DU{du} PGR{pgr}",
               {"MIWaveTile": [2, 2], "DepthU": du, "PrefetchGlobalRead": pgr})
              for du in (32, 64) for pgr in (1, 2)]
    return cases


@pytest.mark.parametrize("label,params", _fold_matrix_cases())
@pytest.mark.parametrize("order", ["KMN", "KNM", "MKN", "MNK", "NKM", "NMK"])
def test_short_path_fold_is_decidable_for_every_known_hard_config(label, params, order):
    """No configuration may reach UNSOUND, and every one must survive the whole pipeline."""
    from Tensile.Lowering.gir.analyses import ShortPathFold
    from Tensile.Lowering.gir.analyses.short_path import UNSOUND
    from Tensile.Lowering.gir import check_register_slots
    from Tensile.Lowering.gir.passes import folded
    from Tensile.Lowering.gir.verify import verify_gir

    try:
        th = _theta(dict(params, LoopOrder=order))
    except (NotImplementedError, ValueError) as e:      # a theta this order cannot express: not ours
        pytest.skip(f"{label}/{order}: {type(e).__name__}: {str(e)[:80]}")
    except RuntimeError as e:                           # no S serves the requested read-ahead here
        if "gives S" not in str(e):
            raise
        pytest.skip(f"{label}/{order}: {str(e)[:80]}")

    prog = lower_to_gir(th)
    am = AnalysisManager(); check_register_slots(prog, am); am.invalidate()
    v = am.get(ShortPathFold(), prog)
    assert v.verdict != UNSOUND, (
        f"{label}/{order}: neither shape serves T<M -- {v.reason}\n"
        f"  first unserved consumer: {v.unbound[0] if v.unbound else None!r}")

    prog = build_gir(th)                                 # the real pipeline, coverage pass included
    assert verify_gir(prog)
    sl = prog.meta.get("short_loop")
    if sl is not None:
        n_short = len([b for b in prog.blocks if b.startswith("short")])
        assert folded(prog) and n_short == 0, (
            f"{label}/{order}: finalized GIR retained {n_short} unemitted short blocks")


# ===========================================================================
# The ledger-vs-GIR differential (was decoder_proto/ledger_vs_gir.py).
# ===========================================================================
ARRIVE, VACATE = "arrive", "vacate"


def _ledger_shared_edges(th, S):
    """The shared-space edges the LEDGER claims, as {(operand, kind)} -- derived from theta alone
    (paths, ring depths, read-ahead reach); it never looks at a CFG."""
    from Tensile.LoopModel.checks import build_ledger
    from Tensile.LoopModel.ir import Space

    def members(op):
        return tuple(op) if isinstance(op, tuple) else (op,)

    out = set()
    for o in build_ledger(th, S):
        if o.kind == "RAW-residency":
            if Space.SHARED in (o.producer.at, o.consumer.at):
                out |= {(m, ARRIVE) for m in members(o.producer.op)}
        elif o.kind == "crossing-RAW":
            out |= {(m, ARRIVE) for m in members(o.producer.op)}
        elif o.kind.endswith("WAR") and o.consumer.at == Space.SHARED:
            out |= {(m, VACATE) for m in members(o.consumer.op)}
    return out


def _gir_shared_edges(prog):
    """The same edges as GIR's LDS hazard analysis finds them -- derived from the CFG and the
    generation SSA, with ZERO references to theta.  RAW->arrive; WAR/WAW->vacate (the ledger does not
    split WAW out)."""
    from Tensile.Lowering.gir.analyses import FrameHazards
    hz = AnalysisManager().get(FrameHazards(), prog)
    out = set()
    for h in getattr(hz, "hazards", hz):
        kind = ARRIVE if h.kind == "RAW" else VACATE
        out.add((h.consumer.operand, kind))
        out.add((h.producer.operand, kind))
    return out


@pytest.mark.parametrize("label,params", [
    (f"{o} PGR{p} PLR{l}", dict(BF16_NT_KMN, LoopOrder=o, PrefetchGlobalRead=p,
                                PrefetchLocalRead=l))
    for o in ("KMN", "KNM", "MKN", "MNK", "NKM", "NMK") for p in (1, 2) for l in (0, 1)
] + [("fused multi-wave", BF16_NT_KMN_FUSED_XAGENT), ("1LDS", BF16_NT_KMN_1LDS)])
def test_ledger_and_gir_agree_on_the_shared_edges(label, params):
    """TWO INDEPENDENT DERIVATIONS of the same ordering facts must describe the same edges.
    The ledger derives them from theta; `FrameHazards` derives them from the CFG and the generation SSA.
    """
    from Tensile.LoopModel.schedule import build_S
    th = _theta(params)
    S, _ = build_S(th)
    led, gir = _ledger_shared_edges(th, S), _gir_shared_edges(build_gir(th))
    assert led == gir, (
        f"{label}\n  LEDGER-ONLY {sorted(led - gir)}  <- theta predicts an edge the CFG cannot see\n"
        f"  GIR-ONLY    {sorted(gir - led)}  <- CFG conflict theta did not predict")


def test_the_ledger_gir_differential_can_actually_fail():
    """The negative control for the test above, run every time.
    A differential that only ever passes is indistinguishable from one whose comparison is broken.
    Drop one edge from the ledger side and the comparison must notice.
    """
    from Tensile.LoopModel.schedule import build_S
    th = _theta(BF16_NT_KMN)
    S, _ = build_S(th)
    led, gir = _ledger_shared_edges(th, S), _gir_shared_edges(build_gir(th))
    assert led, "no shared edges at all -- the comparison would be vacuous"
    assert gir != (led - {sorted(led)[0]}), "dropping an edge went unnoticed: the check is dead"


# ===========================================================================
# LoopShape: the loop's test position is DERIVED, and the covering invariant is checked rather than
# assumed.
def _pre_tested_loop_prog():
    """A synthetic PRE-tested loop: a header that only tests, guarding a body that only works."""
    from Tensile.Lowering.gir.nodes import (Program, Block, Goto, LoopBack, Trips,
                                            Move, Ref, Tile, Gen, GenPhi, GenXfer)
    g = Gen(id=0, ring=2)
    work = Move(srcs=(Ref(Tile("A", "global", (), ()), size_bytes=4),),
                dsts=(Ref(Tile("A", "shared", (), ()), gen=g, gdelta=0, size_bytes=4),))
    prog = Program(entry="pre")
    prog.add_block(Block(phase="prologue", preds=(), succs=("head",), body=[],
                         term=Goto("head"), gen_rel=None, chunk_base=0))
    prog.blocks["pre"] = prog.blocks.pop("prologue")
    prog.add_block(Block(phase="head", loop=True, preds=("pre", "body"), succs=("body", "drain0"),
                         phis=[GenPhi(g, entry_val=0)], body=[],          # tests only, no work
                         term=LoopBack(Trips(var="T", sub=2), "body", "drain0"),
                         gen_rel=0, chunk_base=0))
    prog.add_block(Block(phase="body", preds=("head",), succs=("head",), body=[work],
                         xfers=[GenXfer(g, adv=1, ring=2)], term=Goto("head"),
                         gen_rel=0, chunk_base=0))
    prog.add_block(Block(phase="drain0", preds=("head",), succs=("end",), body=[work],
                         term=Goto("end"), gen_rel=-2, chunk_base=1))
    prog.meta["peel_depth"] = 2
    return prog


def test_loop_test_position_is_DERIVED_from_where_the_branch_sits():
    """Pre- vs post-test is not a declaration, it is a CFG fact: does the block carrying the
    exit test also do work?  The same predicate means a different trip count either way, which is
    why this must be derived and not assumed."""
    from Tensile.Lowering.gir.analyses import LoopShape, PRE, POST

    real = build_gir(_theta(BF16_NT_KMN))
    loops = AnalysisManager().get(LoopShape(), real)
    assert loops, "no loop found in a kernel that has one"
    steady = next(l for l in loops if l.exit_to.startswith("drain"))
    assert steady.position == POST, (
        "the steady block IS the body and carries the test on its terminator, so the work runs "
        "before the test -- post-test")
    # ...and the analysis must be able to say PRE, or the assertion above is vacuous
    synth = AnalysisManager().get(LoopShape(), _pre_tested_loop_prog())
    assert [l.position for l in synth] == [PRE], (
        f"a header that only tests must read as pre-test; got {[l.position for l in synth]}")
    # The two shapes reach equal trip counts from different bounds: the synthetic header
    # pre-tests `iter < T-2`, the real steady post-tests `iter < T-3`, both T-2 trips.
    assert synth[0].trips.render() == "T - 2" and steady.trips.render() == "T - 2"


def test_reduction_covering_invariant_holds_and_is_enforced_as_G_TRIP():
    """fixed, the invariant enforced."""
    from Tensile.Lowering.gir.analyses import LoopShape, reduction_coverage_violations

    for pgr in (1, 2, 3):
        prog = build_gir(_theta(dict(BF16_NT_KMN, PrefetchGlobalRead=pgr)))
        loops = AnalysisManager().get(LoopShape(), prog)
        v = reduction_coverage_violations(prog, loops)
        assert v == [], f"PGR{pgr}: steady and drain must cover [0, T) exactly; got {v}"

    # the synthetic PRE-tested loop over the same predicate covers exactly [0, T-M) -- proof that
    # the invariant is satisfiable and that the check is not simply always-failing.
    synth = _pre_tested_loop_prog()
    assert reduction_coverage_violations(synth, AnalysisManager().get(LoopShape(), synth)) == []


def test_multi_copy_loop_coverage_is_not_yet_EXPRESSIBLE():
    """PINS a hole in G-TRIP that is a limit of the predicate form, not a choice."""
    from Tensile.Lowering.gir.analyses import LoopShape, reduction_coverage_violations

    th = _theta(BF16_NT_KMN)
    for n in (2, 3):
        prog = lower_to_gir(th, loop_copies=n)
        loops = AnalysisManager().get(LoopShape(), prog)
        steady = next(l for l in loops if l.exit_to.startswith("drain"))
        assert steady.per_trip == n, "the back edge's GenXfer.adv states chunks-per-trip"
        # over-covers by exactly the factor n -- the defect this records
        assert steady.covers.coeff == n, (
            f"loop_copies={n} should cover n*(T-M); got {steady.covers.render()}")
        # ...and G-TRIP currently says nothing about it, by the scope note in loop_shape.py
        assert reduction_coverage_violations(prog, loops) == []

    # the single-copy case IS checked, so the exemption is narrow and not a blanket disable
    prog1 = lower_to_gir(th, loop_copies=1)
    loops1 = AnalysisManager().get(LoopShape(), prog1)
    assert next(l for l in loops1 if l.exit_to.startswith("drain")).per_trip == 1
    assert reduction_coverage_violations(prog1, loops1) == []


# ---------------------------------------------------------------------------------------------
# a component we cannot supply must be LOUD, never a silent omission.

def _fold_record(params):
    prog = build_gir(adapter.params_to_theta(dict(params)))
    return prog, (prog.meta.get("short_loop") or {})


def _verify_error():
    return RuntimeError


def test_a_FOLDED_arm_is_DELETED_not_flagged_and_the_shape_is_verified():
    """FOLD deletes.  `model_only` is for the arms that provably CANNOT coverage -- the opposite case."""
    prog, rec = _fold_record(dict(BF16_NT_KMN, PrefetchGlobalRead=2, PrefetchLocalRead=1))
    assert rec["verdict"] == "folded"
    assert not any(lab.startswith("short") for lab in prog.blocks), sorted(prog.blocks)
    assert rec.get("model_only") is None, "a deleted arm has nothing to flag"
    assert not any(b.model_only for b in prog.blocks.values())
    # the coverage rewired the peel-validity false edge to the drain, and `preds` is a DERIVED fact:
    # no terminator may still point at a deleted block, and no `preds` entry may name one.
    for lab, blk in prog.blocks.items():
        for tgt in blk.succs or ():
            assert tgt in prog.blocks or tgt == "end", (lab, tgt)
        for p in blk.preds or ():
            assert p in prog.blocks, (lab, p)
    verify_gir(prog)                       # G-TERM/G-CFG/G-TRIP/G-EMIT on the folded shape


def test_split_remains_an_analysis_diagnostic_not_a_finalized_shape():
    from Tensile.Lowering.gir.analyses import ShortPathFold
    from Tensile.Lowering.gir.analyses.short_path import SPLIT

    prog = lower_to_gir(_theta(dict(BF16_NT_KMN, PrefetchGlobalRead=3,
                                    PrefetchLocalRead=1)))
    assert AnalysisManager().get(ShortPathFold(), prog).verdict == SPLIT


@pytest.mark.parametrize("fixture,pgr", [
    (BF16_NT_KMN, 2),
    (BF16_NT_KMN_1LDS, 1),
    (BF16_NT_KMN_FUSED_XAGENT, 2),
    (BF16_NT_KMN_UNFUSED_MULTIAGENT, 2),
])
def test_every_short_arm_folds_without_dropping_marks(fixture, pgr):
    prog, rec = _fold_record(dict(fixture, PrefetchGlobalRead=pgr, PrefetchLocalRead=1))
    assert rec["verdict"] == "folded"
    assert not any(block.model_only for block in prog.blocks.values())


def test_finalized_gir_contains_no_model_only_short_blocks():
    for params in (BF16_NT_KMN, BF16_NT_KMN_FUSED_XAGENT):
        prog, _rec = _fold_record(params)
        assert not any(block.model_only for block in prog.blocks.values())


def test_G_EMIT_FIRES_when_a_block_is_flagged_with_no_recorded_reason():
    """NEGATIVE CONTROL: the flag alone is not a decision.  An unemitted block must say WHY."""
    prog, _rec = _fold_record(dict(BF16_NT_KMN, PrefetchGlobalRead=2, PrefetchLocalRead=1))
    next(iter(prog.blocks.values())).model_only = True      # flag, record nothing
    with pytest.raises(_verify_error(), match="model_only but nothing records WHY"):
        verify_gir(prog)


def test_RecordUnemittedPass_REFUSES_an_undeclared_flag_rather_than_inventing_a_reason():
    from Tensile.Lowering.gir.passes import RecordUnemittedPass
    prog, _ = _fold_record(dict(BF16_NT_KMN, PrefetchGlobalRead=2, PrefetchLocalRead=1))
    next(iter(prog.blocks.values())).model_only = True
    prog.meta.pop("short_loop", None)
    with pytest.raises(RuntimeError, match="no pass recorded why"):
        RecordUnemittedPass().run(prog, AnalysisManager())


# ---------------------------------------------------------------------------------------------
# the drain's generation frame.

def _fold_verdict(params):
    from Tensile.Lowering.gir.analyses.short_path import ShortPathFold
    prog = lower_to_gir(adapter.params_to_theta(dict(params)))
    return prog, AnalysisManager().get(ShortPathFold(), prog)


@pytest.mark.parametrize("order", ["KMN", "KNM", "MKN", "MNK", "NKM", "NMK"])
@pytest.mark.parametrize("pgr", [1, 2, 3])
def test_the_fold_verdict_is_exactly_ring_ge_M_at_every_loop_order(order, pgr):
    """The static all-inner phase ring folds at PGR1/2; PGR3 exceeds the two-deep ring."""
    expect = "split" if pgr == 3 else "fold"
    _prog, v = _fold_verdict(dict(BF16_NT_KMN, LoopOrder=order,
                                  PrefetchGlobalRead=pgr, PrefetchLocalRead=1))
    assert v.verdict == expect, [ (b.cause, b.location) for b in (v.breaks or ())[:3] ]


def test_a_ring_SHALLOWER_than_M_still_splits_and_for_the_RIGHT_reason():
    """1LDS gives ring 1, so even M=2 cannot coverage.  Guards against a fix that just folds everything:
    the break must be OVERWRITTEN (a real ring collision), never UNWRITTEN (a frame error)."""
    from Tensile.Lowering.gir.analyses.short_path import OVERWRITTEN
    _prog, v = _fold_verdict(dict(BF16_NT_KMN_1LDS, PrefetchGlobalRead=2, PrefetchLocalRead=0))
    assert v.verdict == "split"
    assert {b.cause for b in v.breaks} == {OVERWRITTEN}, {b.cause for b in v.breaks}


def test_the_drain_frame_base_is_the_PEEL_DEPTH_not_zero():
    """The frame itself, checked directly rather than only through the verdict."""
    from Tensile.Lowering.gir.analyses.short_path import _frame_base
    for pgr in (1, 2, 3):
        prog = lower_to_gir(adapter.params_to_theta(
            dict(BF16_NT_KMN, PrefetchGlobalRead=pgr, PrefetchLocalRead=1)))
        M = prog.meta["peel_depth"]
        drains = [b for l, b in prog.blocks.items() if l.startswith("drain")]
        assert drains and all(_frame_base(b) == M for b in drains), \
            (pgr, M, [(b.chunk_base, b.gen_rel, _frame_base(b)) for b in drains])
        # a block with no loop-relative frame contributes no offset
        assert _frame_base(prog.blocks["prologue"]) == 0


def test_the_OLD_base_zero_frame_FAILS_the_ring_ge_M_rule():
    """NEGATIVE CONTROL, and it reproduces the defect exactly: with the base forced back to 0, PGR1 stops
    folding even though ring 2 >= M 1, and it breaks on a generation NOTHING wrote."""
    from Tensile.Lowering.gir.analyses import short_path as SP
    from Tensile.Lowering.gir.analyses.short_path import ENTRY
    real = SP._frame_base
    try:
        SP._frame_base = lambda blk: 0
        _prog, v = _fold_verdict(dict(BF16_NT_KMN, PrefetchGlobalRead=1, PrefetchLocalRead=1))
    finally:
        SP._frame_base = real
    assert v.verdict == "split", "the negative control no longer reproduces the defect"
    # the signature: the folded path's READ chains back to ENTRY -- it read a shared generation
    # NOTHING on that path wrote, because the base-0 frame named generation 1 where the prologue
    # had written generation 0.  A ring-depth split (the real PGR3 one) chains to a real copy.
    assert any(b.folded[0] == "read" and b.folded[3] == ENTRY for b in v.breaks), \
        [b.folded for b in v.breaks[:3]]
    # and with the real frame the same config folds
    assert _fold_verdict(dict(BF16_NT_KMN, PrefetchGlobalRead=1,
                              PrefetchLocalRead=1))[1].verdict == "fold"


def test_BOTH_frames_unserved_is_a_FOLD_break_not_agreement():
    """The pairing ladder's first rung.  `f_src == s_src == ENTRY` says "neither frame wrote this",
    and the old ladder's `if f_src == s_src: continue` skipped it -- recording no Break at all, so
    `fold_breaks` came out empty and the verdict message reported the FOLDED PATH AS SOUND on the
    one input that proves it is not.

    Asserted against `pairing_cause` rather than a whole program because the classification is the
    unit under test: this is the rung a config has to reach, and no config in the standing matrix
    reaches it today (two attempts to find one failed) -- which is exactly why it went unnoticed.

    The second assertion is the mirror trap: both-unserved must be blamed on the FOLDED path.
    `ARM_UNWRITTEN` is filtered out of `fold_breaks` because it means "the arm is broken but folding
    rescues it", so classifying this case as `ARM_UNWRITTEN` would drop it right back out."""
    from Tensile.Lowering.gir.analyses.short_path import (
        pairing_cause, ENTRY, UNWRITTEN, OVERWRITTEN, ARM_UNWRITTEN)

    copy_a, copy_b = ("copy", "A", 0), ("copy", "A", 1)

    # THE FIX: both unserved is a break, and it is the FOLDED path's.
    assert pairing_cause(ENTRY, ENTRY) == UNWRITTEN
    assert pairing_cause(("<absent>",), ENTRY) == UNWRITTEN
    assert pairing_cause(None, ENTRY) == UNWRITTEN

    # and it must NOT be the cause that gets filtered out of `fold_breaks`
    assert pairing_cause(ENTRY, ENTRY) != ARM_UNWRITTEN

    # the other rungs are unchanged -- a real producer on both sides still agrees ...
    assert pairing_cause(copy_a, copy_a) is None
    # ... the arm alone unserved is still the ARM's defect ...
    assert pairing_cause(copy_a, ENTRY) == ARM_UNWRITTEN
    # ... the folded path alone unserved is still UNWRITTEN ...
    assert pairing_cause(ENTRY, copy_a) == UNWRITTEN
    assert pairing_cause(("<absent>",), copy_a) == UNWRITTEN
    # ... and two different real producers is still a ring-depth split.
    assert pairing_cause(copy_a, copy_b) == OVERWRITTEN


# ---------------------------------------------------------------------------------------------
# `chunk_base` is a PER-PATH fact, and the folded join proves it.

def test_the_folded_entry_edge_carries_the_ARMS_frame_not_the_long_paths():
    """The coverage rewires `-> short0` into `-> drain0`; the frame must come with it."""
    from Tensile.Lowering.gir.analyses.swap_regions import _edge_delta
    prog = build_gir(adapter.params_to_theta(
        dict(BF16_NT_KMN, PrefetchGlobalRead=2, PrefetchLocalRead=1)))
    assert (prog.meta["short_loop"]["verdict"] == "folded"
            and "drain0" in prog.blocks and prog.blocks["drain0"].chunk_base == 1)
    d0 = prog.blocks["drain0"]
    # the folded edge states its own base; the loop edge does not and keeps `chunk_base`
    assert d0.path_chunk_base == {_peel_exit(prog): 0}, d0.path_chunk_base
    assert _edge_delta(prog, _peel_exit(prog), "drain0", set()) == 0   # folded: peel and drain0
    assert _edge_delta(prog, "steady", "drain0", set()) == 1      # long: one chunk past the trip
    # `drain1` now also carries a per-edge frame, from the OTHER shaping step: `EarlyExitPass` gives
    # it the `T == 1` entry (prologue -> drain{M-T}), and on that edge it does chunk 0's work rather
    # than the long path's `M-1` chunks past the last trip.
    assert prog.blocks["drain1"].path_chunk_base == {_peel_exit(prog): 0}
    assert _edge_delta(prog, "drain0", "drain1", set()) == 1


@pytest.mark.parametrize(
    "pgr,target,phase,full_drain",
    [(1, "drain0", 1, True), (2, "drain0", 0, False), (2, "drain1", 1, True)],
)
def test_general_frame_phi_resolves_each_folded_drain_entrance(
        pgr, target, phase, full_drain):
    from Tensile.Lowering.gir.analyses import FrameHazards

    prog = build_gir(adapter.params_to_theta(
        dict(BF16_NT_KMN, PrefetchGlobalRead=pgr, PrefetchLocalRead=0)))
    source = _peel_exit(prog)
    fm = AnalysisManager().get(FrameMap(), prog)
    source_frame = fm.frames(source)[0]
    entered = [
        frame for block, frame in fm.successors((source, source_frame))
        if block == target
    ]
    assert len(entered) == 1
    assert {value for _gen, value in entered[0].phases} == {phase}

    phis = prog.block(target).phis
    assert phis and all(dict(phi.incomings)[source] == phase for phi in phis)
    hazards = AnalysisManager().get(FrameHazards(), prog)
    assert any(
        hazard.kind == "RAW"
        and hazard.producer.block == "prologue"
        and hazard.consumer.block == target
        for hazard in hazards
    )
    waits = [
        node.at.get("tensorcnt")
        for node in prog.block(target).body
        if isinstance(node, Mark)
        and node.kind == "waitcnt"
        and "tensorcnt" in node.at
    ]
    assert waits
    if full_drain:
        assert 0 in waits


def test_frame_contract_preserves_forwarding_phi_transfer_on_the_long_drain_path():
    """The steady latch advances its back edge but not an exit whose drain refs carry that step."""
    prog = build_gir(adapter.params_to_theta(
        dict(BF16_NT_KMN, PrefetchGlobalRead=1, PrefetchLocalRead=0)))
    plans = plan_program(prog)
    steady = plans["steady"][0].action_id
    drain = plans["drain0"][0].action_id
    transfers = {(edge.dst, edge.src, edge.gen, edge.value)
                 for edge in parse_contract(encode_frame_contract(prog)).edges if edge.relative}
    gens = {phi.gen.id for phi in prog.block("steady").phis}
    assert gens
    assert {(steady, steady, gen, 1) for gen in gens} <= transfers
    assert {(drain, steady, gen, 0) for gen in gens} <= transfers


@pytest.mark.parametrize("pgr,plr", [(1, 0), (1, 1), (2, 0), (2, 1)])
def test_a_FOLDED_configs_entry_edge_is_REAL_DATAFLOW_now(pgr, plr):
    """the deliverable: the folded entry is no longer excluded from the placement problem.
    `drain0` really does have two predecessors in the solved CFG, and it solves."""
    prog = build_gir(adapter.params_to_theta(
        dict(BF16_NT_KMN, PrefetchGlobalRead=pgr, PrefetchLocalRead=plr)))
    assert prog.meta["short_loop"]["verdict"] == "folded"
    assert prog.blocks["drain0"].preds == (_peel_exit(prog), "steady")
    from Tensile.Lowering.gir.analyses.swap_regions import _unemitted_edges
    assert _unemitted_edges(prog) == set(), "a folded program has no unemitted block to exclude"


def test_WITHOUT_the_per_path_frame_the_folded_join_is_UNSOLVABLE():
    """NEGATIVE CONTROL, and it reproduces the defect exactly: restore the per-block-only frame and the
    same config reports the join as an unsplittable critical edge."""
    from Tensile.Lowering.gir.analyses import swap_regions as SR
    th = adapter.params_to_theta(dict(BF16_NT_KMN, PrefetchGlobalRead=2, PrefetchLocalRead=1))
    real = SR._edge_delta

    def per_block_only(prog, src, dst, back_pairs):
        d = prog.blocks[dst].chunk_base - prog.blocks[src].chunk_base
        if (src, dst) in back_pairs:
            d += max((xf.adv for xf in prog.blocks[src].xfers), default=1)
        return d

    try:
        SR._edge_delta = per_block_only
        with pytest.raises(RuntimeError, match="drain0.*JOIN"):
            build_gir(th)
    finally:
        SR._edge_delta = real
    build_gir(th)                       # ...and with the real frame the same config solves


def test_a_KEPT_arm_is_the_ONLY_thing_excluded_from_the_placement_problem():
    """the exclusion is now the DECLARED `model_only` fact, not a CFG-shape pattern match.
    A kept arm is excluded (nothing emits it); a folded program excludes nothing."""
    from Tensile.Lowering.gir.analyses.swap_regions import _unemitted_edges
    # A KEPT arm is the PRE-FOLD program: `build_gir` folds every reachable configuration, so
    # building one here left nothing kept to exclude.
    kept = lower_to_gir(adapter.params_to_theta(
        dict(BF16_NT_KMN, PrefetchGlobalRead=3, PrefetchLocalRead=1)))
    assert kept.meta["short_loop"]["verdict"] == "unfolded"
    excluded = _unemitted_edges(kept)
    # EVERY edge into the arm -- its entry AND its internal chain -- since none of it is emitted
    assert excluded == {(_peel_exit(kept), "short0"), ("short0", "short1"),
                        ("short1", "short2")}, excluded
    # every excluded edge lands in a block that is declared model-only AND has a recorded reason
    for _src, dst in excluded:
        assert kept.blocks[dst].model_only
    # and nothing on the emitted path is excluded
    assert not any(kept.blocks[s_].model_only for s_, _d in excluded if s_ == _peel_exit(kept))
    assert kept.meta["short_loop"]["model_only"]["reason"]


def test_short_blocks_are_model_only_FROM_BIRTH_not_from_a_pass():
    """An analysis run standalone must see which edges lead nowhere.  Keying that on a flag
    `FoldShortPathPass` sets meant `GrIncrementRegions()` run before the pipeline saw an unflagged
    arm and reported conflicts on a path nothing emits."""
    from Tensile.Lowering.gir.analyses.swap_regions import _unemitted_edges
    raw = lower_to_gir(adapter.params_to_theta(
        dict(BF16_NT_KMN, PrefetchGlobalRead=3, PrefetchLocalRead=1)))
    assert all(raw.blocks[l].model_only for l in raw.blocks if l.startswith("short"))
    assert ("prologue", "short0") in _unemitted_edges(raw)
    assert all(raw.blocks[d].model_only for _s, d in _unemitted_edges(raw))
    assert raw.meta["short_loop"]["model_only"]["reason"]


def test_an_ambiguous_value_NOBODY_CONSUMES_is_not_a_conflict():
    """`want is BOTTOM` means no consumer on any path from here reads this register, so two
    predecessors leaving it at different values cannot be observed.
    """
    from Tensile.Lowering.gir.analyses.value_placement import BOTTOM
    from Tensile.Lowering.gir.analyses import value_placement as VP
    th = adapter.params_to_theta(dict(BF16_NT_KMN, PrefetchGlobalRead=2, PrefetchLocalRead=1))
    seen = []
    orig = VP.ValuePlacementSolver.solve

    def spy(self, on_conflict):
        antic = self.antic()
        ain, _body, _leave = self.avail(antic)
        for b in self._acc:
            if ain.get(b) is VP.TOP:
                seen.append((b, antic.get(b, BOTTOM)))
        return orig(self, on_conflict)

    try:
        VP.ValuePlacementSolver.solve = spy
        build_gir(th)                       # must NOT raise
    finally:
        VP.ValuePlacementSolver.solve = orig
    assert seen, "the folded entry should still produce an ambiguous arrival somewhere"
    assert all(want is BOTTOM for _b, want in seen), \
        f"every tolerated TOP must be one nothing demands: {seen}"


# =================================================================================================
# TDMSplit: LDS storage is (operand, REGION), not operand.
# =================================================================================================

def _split_prog(name):
    import loopmodel_scenarios as scenarios
    return lower_to_gir(adapter.params_to_theta(scenarios.SCENARIOS[name][1]))


def test_a_SPLIT_tiles_halves_are_storage_disjoint_and_raise_no_hazard():
    """`frame_hazards` paired on `p.operand != c.operand` -- pure name equality -- while `Tile.coord`
    carried the region all along.  A's two storage-disjoint LDS halves therefore looked like one
    region and collected hazards that cannot exist."""
    from Tensile.Lowering.gir.analyses.frame_hazards import FrameHazards, _disjoint_storage
    prog = _split_prog("our_split")
    assert prog.meta["region_axes"]["A"] == ("M_split",)
    hz = AnalysisManager().get(FrameHazards(), prog)
    assert len(hz) > 0, "the split kernel must still have real hazards"
    for e in hz:
        assert not _disjoint_storage(e.producer, e.consumer), \
            f"hazard between disjoint storage: {e.producer.storage} vs {e.consumer.storage}"
        if e.producer.operand == e.consumer.operand:
            # same tensor => the two ends must share at least one possible region on every axis
            for ra, rb in zip(e.producer.regions, e.consumer.regions):
                assert ra & rb, (e.producer.regions, e.consumer.regions)


def test_region_awareness_removes_hazards_ONLY_where_there_is_a_split():
    """The negative control that matters: an unsplit kernel must be bit-for-bit unaffected, or the
    change is not 'more precise', it is 'different'.

    The observable is CROSS-HALF PAIRING, not the edge count: under a killing dataflow, collapsing
    the halves onto one buffer merges two chains into one of the same length, so the totals match
    while the pairs differ."""
    from Tensile.Lowering.gir.analyses import lds_buffers as LS
    from Tensile.Lowering.gir.analyses.frame_hazards import FrameHazards, _disjoint_storage
    real = LS.Geometry.regions_of
    facts = {}
    try:
        for name in ("our_split", "baseline"):
            prog = _split_prog(name)
            # the "operand" arm forgets the region axis entirely, which is the pre-region model
            for tag, fn in (("region", real),
                            ("operand", lambda self, ref, is_write=True: ())):
                LS.Geometry.regions_of = fn
                hz = AnalysisManager().get(FrameHazards(), prog)
                LS.Geometry.regions_of = real
                # judge every pair by the REAL geometry, whichever arm produced it
                geo = LS.Geometry(prog)
                cross = sum(1 for h in hz
                            if h.producer.operand == h.consumer.operand
                            and any(not (ra & rb) for ra, rb in
                                    zip(geo.regions_of(h.producer.ref, h.producer.is_write),
                                        geo.regions_of(h.consumer.ref, h.consumer.is_write))))
                facts[(name, tag)] = (len(hz), cross)
    finally:
        LS.Geometry.regions_of = real
    assert facts[("baseline", "region")] == facts[("baseline", "operand")], \
        "an unsplit kernel must be unchanged"
    assert facts[("our_split", "region")][1] == 0, \
        f"region awareness must pair NO storage-disjoint halves: {facts[('our_split', 'region')]}"
    assert facts[("our_split", "operand")][1] > 0, \
        "the operand-only arm must show the phantom cross-half pairs this removes"


def test_an_access_that_does_NOT_pin_the_region_aliases_EVERY_region():
    """The scaffold's `bothHalves` case: a tile whose per-wave reads do not statically separate the
    halves.  Modelled as the full set, so it is disjoint from nothing -- conservative on purpose,
    because being optimistic here LOSES a fence."""
    from Tensile.Lowering.gir.analyses.frame_hazards import _disjoint_storage, SharedTouch
    from Tensile.Lowering.gir.analyses.lds_buffers import Geometry
    from Tensile.Lowering.gir.nodes import Program

    class _T:
        def __init__(self, coord, operand):
            self.coord, self.operand = coord, operand

    class _R:
        def __init__(self, coord, operand="A"):
            self.tile = _T(coord, operand)

    geo = Geometry(Program(meta={"region_axes": {"A": ("M_split",)},
                                 "axis_extents": {"M_split": 2},
                                 "operand_regions": {"A": 2}}))
    pinned0 = geo.regions_of(_R((("M_split", 0),)))
    pinned1 = geo.regions_of(_R((("M_split", 1),)))
    both = geo.regions_of(_R((("K_inner", 0),)))     # region axis absent
    assert pinned0 == (frozenset({0}),) and pinned1 == (frozenset({1}),)
    assert both == (frozenset({0, 1}),)

    def acc(regions, operand="A"):
        return SharedTouch(block="b", pos=0, inst=None, ref=None, is_write=True,
                      operand=operand, regions=regions)
    assert _disjoint_storage(acc(pinned0), acc(pinned1)), "pinned halves are disjoint"
    assert not _disjoint_storage(acc(pinned0), acc(both)), "bothHalves aliases half 0"
    assert not _disjoint_storage(acc(pinned1), acc(both)), "bothHalves aliases half 1"
    assert _disjoint_storage(acc(pinned0), acc(pinned0, operand="B")), "different tensors"
    # an unsplit operand has no region axis and so is never split apart
    assert geo.regions_of(_R((), operand="B")) == ()


# =================================================================================================
# GIR owns the LDS memory-token numbering.

def _tokens_of(params):
    from Tensile.Lowering.gir.analyses.lds_buffers import LdsBufferIds
    prog = build_gir(adapter.params_to_theta(dict(params)))
    return prog, AnalysisManager().get(LdsBufferIds(), prog)


def _dep_tokens_of(prog):
    """The obligation tokens -- what a stamp and a fence carry, as against the storage id."""
    from Tensile.Lowering.gir.analyses.dep_tokens import DependenceTokens
    return AnalysisManager().get(DependenceTokens(), prog)


def test_every_logical_LDS_buffer_gets_its_own_id():
    """The key is (unit, region, generation) -- each part earns its place."""
    import loopmodel_scenarios as scenarios
    _p, ts = _tokens_of(BF16_NT_KMN)
    names = {b.render(): i for b, i in ts.buffers.items()}
    # A and B are SEPARATE storage, so separate ids.  The scaffold put both on 0/1, which ordered
    # A's read after B's load for no reason.
    assert set(names) == {"A#0", "A#1", "B#0", "B#1"}, names
    assert len({names["A#0"], names["B#0"]}) == 2, "A and B must not share an id"

    # a region split makes storage-DISJOINT siblings -> one id each
    _p2, ts2 = _tokens_of(scenarios.SCENARIOS["our_split"][1])
    n2 = {b.render(): i for b, i in ts2.buffers.items()}
    assert set(n2) == {"A/r0#0", "A/r0#1", "A/r1#0", "A/r1#1",
                       "B/r0#0", "B/r0#1", "B/r1#0", "B/r1#1"}, n2
    assert len(set(n2.values())) == 8, "the eight buffers must be eight ids"

    # A Phi-fused movement is ONE COMPLETION but ONE INSTRUCTION WRITING TWO STORAGES.  Its members
    # therefore keep SEPARATE buffer names, and the fused copy DEFS BOTH -- the union, not one id.
    p3, ts3 = _tokens_of(dict(BF16_NT_KMN_FUSED_XAGENT, PrefetchGlobalRead=2, PrefetchLocalRead=1))
    assert set(b.render() for b in ts3.buffers) == {"A#0", "A#1", "B#0", "B#1"}
    fused = [i for b in p3.blocks.values() for i in b.body
             if isinstance(i, Move)
             and len([r for r in i.dsts if r.tile.space == "shared"]) > 1]
    assert fused, "this fixture must produce a fused copy with one dst Ref per member"
    dt3 = _dep_tokens_of(p3)
    # WHICH buffer of the ring is `FrameMap`'s answer -- `ids_for` is the may-set over the whole
    # rotation, so the frame-concrete pair comes from the stamp.
    for cp in fused:
        want, stamp = set(), set()
        for r in cp.dsts:
            if r.tile.space == "shared":
                want |= set(ts3.ids_for(r))
                stamp |= set(dt3.tokens_for(r))
        assert len(stamp) == 2, f"a fused copy fills TWO buffers: {stamp}"
        assert len(want) == 4, f"and may touch either end of each ring: {want}"
        assert set(cp.token_ids) == stamp, \
            f"the fused copy must def the UNION of its members' tokens: {cp.token_ids} vs {stamp}"
    # ...and each member's read stays on its own id
    reads = {}
    for b in p3.blocks.values():
        for i in b.body:
            if not isinstance(i, Move):
                continue
            srcs = [r for r in i.srcs if r.tile.space == "shared"]
            if len(srcs) == 1:
                reads.setdefault(srcs[0].tile.operand, set()).update(i.token_ids)
    assert reads.get("A") and reads.get("B")
    assert reads["A"].isdisjoint(reads["B"]), \
        f"a fused movement must not put its members' READS on one token: {reads}"


def test_the_ids_REACH_the_emit_plan():
    """Stamped by TokensPass next to the completion token, and carried on the act -- the emitter
    reads them instead of writing a mutable `states.<field>` and having the leaf read it back."""
    from Tensile.Lowering.gir.emit_plan import plan_block
    prog, _ts = _tokens_of(dict(BF16_NT_KMN, PrefetchGlobalRead=2, PrefetchLocalRead=1))
    dt = _dep_tokens_of(prog)
    acts = [a for a in plan_block(prog, "steady") if a.kind in ("read", "copy")]
    assert acts, "the steady block should plan reads and copies"
    for a in acts:
        ids = a.at.get("token_ids")
        assert ids, f"{a.kind} act carries no token ids: {a.at}"
        assert all(0 <= int(t) < len(dt) for t in ids), (ids, len(dt))
    # A's reads and B's reads land on different ids
    reads = {a.at["tc"]: a.at["token_ids"] for a in acts if a.kind == "read"}
    assert set(reads["A"]).isdisjoint(reads["B"]), reads


def test_every_fence_carries_frame_relative_storage_relations():
    """A fence owns per-frame ring relations; static dependency-token unions are not authoritative."""
    prog, _ts = _tokens_of(dict(BF16_NT_KMN_FUSED_XAGENT, PrefetchGlobalRead=2, PrefetchLocalRead=1))
    fences = [n.at for b in prog.blocks.values() for n in b.body
              if isinstance(n, Mark) and n.kind == "fence"]
    assert fences, "the cross-agent fixture must place fences"
    for at in fences:
        assert not at["tokens"] and not at["order_tokens"] and at["no_waitcnt"], at
        assert at["relations"], at
        for relation in at["relations"]:
            for endpoint in (relation["producer"], relation["consumer"]):
                assert endpoint["ring"] >= 1 and endpoint["token"] is not None
                assert "gdelta" in endpoint and "advance" in endpoint


def test_a_fence_names_every_buffer_LIVE_ACROSS_it_not_just_the_edges_that_placed_it():
    """Every cross-agent edge has a fence naming BOTH its ends, re-derived from the edges.

    Fewest-tokens is NOT also required, and asserting it was a defect: narrowing each fence to a
    hitting set over its edges keeps every edge separated and puts `s_wait_tensorcnt 0` back in the
    MAF loop, because the dropped names were inert while the kept ones were load-defined.
    """
    from Tensile.Lowering.tool.fence_token_check import unseparated_edges
    for params in (dict(BF16_NT_KMN_FUSED_XAGENT, PrefetchGlobalRead=2, PrefetchLocalRead=1),
                   dict(BF16_NT_KMN_FUSED_XAGENT, PrefetchGlobalRead=2, PrefetchLocalRead=1,
                        TDMSplit=[2, 2])):
        prog, _ts = _tokens_of(params)
        seen = [n for blk in prog.blocks.values() if blk.loop for n in blk.body
                if isinstance(n, Mark) and n.kind == "fence"]
        assert seen, "the cross-agent fixture must place a fence inside the loop"
        loose = unseparated_edges(prog)
        assert not loose, (
            f"{len(loose)} cross-agent edge(s) have no fence naming both ends, e.g. "
            f"{loose[0].kind} {loose[0].producer.block}[{loose[0].producer.pos}] -> "
            f"{loose[0].consumer.block}[{loose[0].consumer.pos}]")


@pytest.mark.parametrize("break_it,match", [
    ("unnamed", "the emitter reads the STAMP"),
])
def test_G_TOKEN_FIRES_on_an_unnamed_access(break_it, match):
    """NEGATIVE CONTROL.  An unnamed access breaks StinkyTofu's all-or-none rule for its whole
    basic block.  A fence with no ids is NOT one: it is the full drain.
    """
    prog, _ts = _tokens_of(dict(BF16_NT_KMN_FUSED_XAGENT, PrefetchGlobalRead=2,
                                PrefetchLocalRead=1))
    verify_gir(prog)                                    # clean before
    if break_it == "unnamed":
        for blk in prog.blocks.values():
            for n in blk.body:
                if isinstance(n, Move) and n.token_ids:
                    n.token_ids = ()
                    break
            else:
                continue
            break
    else:
        for blk in prog.blocks.values():
            for n in blk.body:
                if isinstance(n, Mark) and n.kind == "fence":
                    n.at["tokens"] = ()
                    break
            else:
                continue
            break
    with pytest.raises(RuntimeError, match=match):
        verify_gir(prog)


# =================================================================================================
# TDMSplit: GIR owns the REGION WALK.

def _walked(name):
    """A FULLY LOWERED split program -- `build_gir`, not `lower_to_gir`."""
    import loopmodel_scenarios as scenarios
    from Tensile.Lowering import build_gir
    return build_gir(adapter.params_to_theta(scenarios.SCENARIOS[name][1]))


def _walk(prog, label):
    """[('copy', unit, region)] / [('inc', unit, steps)] for one block, in program order."""
    from Tensile.Lowering.gir.analyses.region_increment import region_of, _flat, _n_regions
    from Tensile.Lowering.gir.nodes import copy_unit
    out = []
    for n in prog.blocks[label].body:
        if isinstance(n, Mark) and n.kind == "region_increment":
            out.append(("inc", n.at["unit"], n.at["steps"]))
        elif isinstance(n, Move):
            members, refs = copy_unit(n)
            if members is None or _n_regions(prog, members) <= 1:
                continue
            reg = region_of(prog, refs[0])
            out.append(("copy", members, _flat(reg, prog, refs[0].tile.operand) if reg else None))
    return out


def test_the_descriptor_STEPS_between_consecutive_regions_of_one_tile():
    """Between two copies of the same movement that fill DIFFERENT regions there must be a step."""
    prog = _walked("our_split")
    for label in ("prologue", "steady"):
        seq = _walk(prog, label)
        assert seq, f"{label} must contain split copies"
        prev = {}
        for kind, unit, val in seq:
            if kind != "copy":
                continue
            if unit in prev:
                assert prev[unit] != val, "two consecutive copies of one unit filled one region"
            prev[unit] = val
        # every adjacent copy pair of a unit is separated by exactly the step between their regions
        at = {}
        for kind, unit, val in seq:
            if kind == "inc":
                at[unit] = at.get(unit, 0) + val
            else:
                assert at.get(unit, 0) == val, \
                    f"{label}: copy of {unit} fills region {val} with the descriptor on " \
                    f"{at.get(unit, 0)}"


def test_the_walk_CLOSES_so_the_chunk_stride_stays_plain():
    """Net displacement per unit per block is zero."""
    prog = _walked("our_split")
    for label in prog.blocks:
        at = {}
        for kind, unit, val in _walk(prog, label):
            if kind == "inc":
                at[unit] = at.get(unit, 0) + val
        assert all(v == 0 for v in at.values()), f"{label} leaves the descriptor at {at}"


def test_an_UNSPLIT_movement_emits_no_walk_at_all():
    """The whole mechanism is behind `unit_regions > 1`, so a non-TDMSplit kernel is untouched --
    that is what makes the emission-neutrality sweep meaningful rather than tautological."""
    prog = _walked("baseline")
    marks = [n for blk in prog.blocks.values() for n in blk.body
             if isinstance(n, Mark) and n.kind == "region_increment"]
    assert marks == []


def test_each_region_is_loaded_ONCE_with_its_OWN_token():
    """`regions` copies per chunk per movement, and no two share a memory token."""
    from Tensile.Lowering.gir.emit_plan import plan_block
    prog = _walked("our_split")
    copies = [a for a in plan_block(prog, "steady") if a.kind == "copy"]
    assert copies and all(a.at["regions"] == 2 for a in copies)
    for unit in {a.at["unit"] for a in copies}:
        mine = [a for a in copies if a.at["unit"] == unit]
        assert sorted(a.at["region"] for a in mine) == [0, 1], \
            f"{unit} must load each of its 2 regions exactly once per chunk"
        toks = [a.at["token_ids"] for a in mine]
        assert len(set(toks)) == len(toks) and all(toks), \
            f"{unit}'s regions share a memory token: {toks}"


@pytest.mark.parametrize("break_it", ["drop", "duplicate", "hoist"])
def test_G_WALK_FIRES_when_the_walk_is_dropped_duplicated_or_HOISTED(break_it):
    """The three ways to get the walk wrong, and all three must be caught."""
    prog = _walked("our_split")
    verify_gir(prog)                                    # clean before
    blk = prog.blocks["steady"]
    incs = [n for n in blk.body if isinstance(n, Mark) and n.kind == "region_increment"]
    assert incs
    if break_it == "drop":
        blk.body.remove(incs[0])
    elif break_it == "duplicate":
        blk.body.insert(blk.body.index(incs[0]), incs[0])
    else:
        for n in incs:
            blk.body.remove(n)
        for i, n in enumerate(incs):
            blk.body.insert(i, n)
    with pytest.raises(RuntimeError, match="G-WALK"):
        verify_gir(prog)


# =================================================================================================
# the REGION IS AN ADDRESS FACT.

def _read_acts(name):
    import loopmodel_scenarios as scenarios
    from Tensile.Lowering import build_gir
    from Tensile.Lowering.gir.emit_plan import plan_block
    prog = build_gir(adapter.params_to_theta(scenarios.SCENARIOS[name][1]))
    return [a for a in plan_block(prog, "steady") if a.kind == "read"]


def test_a_split_read_carries_BOTH_indices_and_they_DIFFER():
    """The within-region index and the flat index are not the same number under a split."""
    acts = [a for a in _read_acts("our_split") if a.at["tc"] == "A"]
    assert acts and all(a.at["regions"] == 2 for a in acts)
    pairs = sorted({(a.at["region"], a.at["tile"], a.at["tile_flat"]) for a in acts})
    # MIWaveTile 2 split 2 => one tile per region, so within-region is 0 for BOTH regions while the
    # flat index still counts 0,1.  Collapsing them addresses region 1 at region 0's offset.
    assert pairs == [(0, 0, 0), (1, 0, 1)], pairs


def test_an_UNSPLIT_read_has_region_zero_and_the_two_indices_AGREE():
    """Emission neutrality, stated where it is enforced rather than assumed: with no split the
    region term is 0 and `tile == tile_flat`, so every downstream expression is the old one."""
    acts = _read_acts("baseline")
    assert acts
    for a in acts:
        assert a.at["regions"] == 1
        assert a.at["region"] == 0
        assert a.at["tile"] == a.at["tile_flat"]


@pytest.mark.parametrize("unrollMajor,expect", [(True, [0, 0]), (False, [0, 3072])])
def test_the_region_displacement_is_owed_only_by_the_TILE_MAJOR_layout(unrollMajor, expect):
    """The region term is a property of the LDS image, and the two layouts genuinely differ."""
    import types
    from Tensile.Lowering.lds_geometry import region_bytes
    ctx = types.SimpleNamespace(unrollMajor=unrollMajor, splitBoundaryBytes=3072, nsplit=2)
    assert [region_bytes(ctx, r) for r in (0, 1)] == expect


def test_an_UNSPLIT_read_owes_no_region_displacement_on_either_layout():
    """Emission neutrality at `nsplit == 1`: `splitBoundaryBytes` is 0 when unsplit, so the term
    vanishes on both layouts and every unsplit kernel emits byte-for-byte what it did before."""
    import types
    from Tensile.Lowering.lds_geometry import region_bytes
    for um in (True, False):
        ctx = types.SimpleNamespace(unrollMajor=um, splitBoundaryBytes=0, nsplit=1)
        assert region_bytes(ctx, 0) == 0 and region_bytes(ctx, 1) == 0


@pytest.mark.parametrize("vw,expect", [(1, [0, 2176]), (2, [0, 128])])
def test_the_unroll_major_tile_row_is_vector_group_plus_offset(vw, expect):
    """a wave takes `VW` ADJACENT tiles, then the next group starts `MIWaveGroupShape` rows
    later -- so `row(t) = (t // VW) * MIWaveGroupShape + (t % VW)`, and the unroll-major path had
    only the second term.
    """
    import types
    from Tensile.Lowering.lds_geometry import tile_row    # pure arithmetic; rocisa-free
    ctx = types.SimpleNamespace(unrollMajor=True, vectorWidth=vw, tile01=0,
                                MIWaveGroupShape=[16 * 1 * 1 * vw] * 2)   # MI_M*MIBM*MIWaveGroup*VW
    pad = lambda o: o + (o // 256) * 16
    got = [pad(tile_row(ctx, t) * 64 * 2) for t in (0, 1)]
    assert got == expect, f"VW={vw}: tile byte offsets {got}, scaffold emits {expect}"


@pytest.mark.parametrize("unrollMajor,nsplit,expect", [
    (False, 2, 16),    # tile-major split: the region's row is HALF the MacroTile
    (False, 1, 32),    # tile-major unsplit: the whole tile
    (True,  2, 32),    # unroll-major split: regions continue one array, row unchanged
    (True,  1, 32),
])
def test_the_region_row_length_shortens_only_for_a_PACKED_region(unrollMajor, nsplit, expect):
    """The row length feeds TWO consumers that must agree -- the per-read immediate offsets in
    `leaves.py` and the per-lane base register's `kOffset * strideUnroll` in
    `LraTileAssignment`.  They were independent spellings of `mt + ldsPad`, and shortening only
    the first is what left the two halves of one address in different frames: measured on
    `Cijk_Ailk_Bjlk MT32x32x64` the base emitted `kOffset *= 32` against immediates
    already built for 16, so every lane with a non-zero k component read the wrong row.

    Pinning the shared function is the point of the test; `mt=32, ldsPad=0` is that kernel."""
    from Tensile.Lowering.lds_geometry import region_row_elems
    assert region_row_elems(32, 0, unrollMajor, nsplit) == expect


@pytest.mark.parametrize("axis,unrollMajor,extent,expect", [
    # the DU diagonal: unroll-major LDS is `[free][unroll]`, so a DU split packs and the row that
    # shortens is the UNROLL one -- `_DepthU`, not `MacroTile`.  `Cijk_Alik_Bljk MT32x32x64`.
    ("DU", True,  64, 32),
    ("DU", False, 64, 64),   # tile-major: unroll is OUTER, regions continue -> row unchanged
    ("MT", True,  64, 64),   # unroll-major MT split: free is OUTER -> row unchanged
    ("MT", False, 32, 16),   # the already-measured NT cell, restated in the same call
])
def test_the_row_that_shortens_is_the_INNER_axis_of_the_image(axis, unrollMajor, extent, expect):
    """`region_row_elems` takes the extent from its CALLER, and that is the whole content of the
    DU half.  Naming the function `mt` end to end is right only while every
    caller was tile-major; the unroll-major image's row is `_DepthU`, and shortening `MacroTile`
    there divides a number that is not in the address.

    On TN + `TDMSplitA/B = 2` (MT32x32x64, bpe2) the descriptor is already
    correct -- `tile0 = 32`, `LdsSplitIncs = 2176` -- while every read still stepped a free tile
    by a whole `DepthU`, walking out of the region on every tile after the first."""
    from Tensile.Lowering.lds_geometry import region_row_elems
    from Tensile.Components.TDMSplit import AXIS_DU, AXIS_MT
    ax = AXIS_MT if axis == "MT" else AXIS_DU
    assert region_row_elems(extent, 0, unrollMajor, 2, ax) == expect


@pytest.mark.parametrize("u,expect", [(0, (0, 0)), (16, (0, 16)), (32, (1, 0)), (48, (1, 16))])
def test_a_packed_split_makes_the_inner_coordinate_PIECEWISE(u, expect):
    """`fold_inner_offset` is what turns one linear range into `region, in-region`.  There is no
    linear form: the region jump is a PADDED byte boundary (`tdmSplitLdsBoundary`), not an element
    stride, so an emitter that accumulates the inner coordinate has to divide somewhere.
    """
    from Tensile.Lowering.lds_geometry import fold_inner_offset
    from Tensile.Components.TDMSplit import AXIS_DU
    assert fold_inner_offset(u, 64, True, 2, AXIS_DU) == expect


def test_an_unpacked_split_folds_NOTHING():
    """Every non-diagonal cell must return the coordinate untouched, or the shortening and the
    coverage would double-count each other on the layouts where regions are a contiguous continuation."""
    from Tensile.Lowering.lds_geometry import fold_inner_offset
    from Tensile.Components.TDMSplit import AXIS_DU, AXIS_MT
    for axis, um in ((AXIS_MT, True), (AXIS_DU, False)):
        assert fold_inner_offset(48, 64, um, 2, axis) == (0, 48)
    assert fold_inner_offset(48, 64, True, 1, AXIS_DU) == (0, 48)     # unsplit


def test_the_scaffolds_DU_split_read_addresses_against_the_measured_kernel():
    """The whole read-side arithmetic composed, for `Cijk_Alik_Bljk MT32x32x64` TN
    (`TDMSplitA/B = 2`, VW=1, MIWaveGroup [1,1], LBSPP=256, LdsPad=8, bpe=2) -- the kernel that
    failed 192/192 on the scaffold arm and 1152/1152 under ULM on.
    """
    from Tensile.Lowering.lds_geometry import region_row_elems, fold_inner_offset
    from Tensile.Components.TDMSplit import AXIS_DU
    MT, DU, BPE, NSPLIT = 32, 64, 2, 2
    pad = lambda b: b + (b // 256) * 8 * BPE          # LdsBlockSizePerPad=256, LdsPad=8 elems
    boundary = pad((MT * DU * BPE) // NSPLIT)         # == tdmSplitLdsBoundary
    assert boundary == 2176
    row = region_row_elems(DU, 0, True, NSPLIT, AXIS_DU)
    assert row == 32

    def addr(vIdx, rIdx, lro):
        region, uInRow = fold_inner_offset(rIdx * 8 * 2 + lro, DU, True, NSPLIT, AXIS_DU)
        return pad((vIdx * 16 * row + uInRow) * BPE) + region * boundary   # 16 = MIWaveGroupShape

    got = [addr(v, r, lro) for lro in (0, 32) for v in (0, 1) for r in (0, 1)]
    assert got == [0, 32, 1088, 1120,  2176, 2208, 3264, 3296], got
    # and the regions are disjoint: nothing in region 1 collides with region 0's block
    assert max(got[:4]) < boundary <= min(got[4:])


def _read_plan(params, phase="steady"):
    """[(tc, tile, k, region)] for one block, in emitted order."""
    prog = build_gir(_theta(params))
    return [(a.at["tc"], a.at["tile"], a.at["k"], a.at["region"])
            for a in plan_block(prog, phase) if a.kind == "read"]


def test_the_read_region_is_projected_over_ALL_region_axes_free_OR_reduction():
    """/the region term was `free_modes & region_modes`, which is the same set as
    `region_axes` only while every region axis happens to be a FREE one.  That holds for the MT
    half (`M_split` / `N_split`) and fails for the DU half, whose axis is `K_split`: the
    intersection emptied and EVERY read came out `region = 0`.

    On TN + `TDMSplitA/B = 2`, `k=0` and `k=1` both come out labelled `r0` with k=1
    at the ordinary unsplit stride, while region 1 sat 2176 bytes away -- 100% failure, with a
    descriptor that was already right.

    The second half of the fix is that `k` becomes WITHIN-REGION.  A DU split moves the reduction
    displacement into the region term, so leaving `K_split` in `k` too would count it twice --
    once as `region * splitBoundary`, again as `k * substepStride`."""
    reads = _read_plan({**BF16_NT_KMN, "TDMSplit": [1, 1, 2, 2]})
    assert {r for _, _, _, r in reads} == {0, 1}, "the DU split's second region is unreachable"
    assert {k for _, _, k, _ in reads} == {0}, "K_split is still being counted inside k"
    # each (operand, tile, region) is read exactly once, and the pairs tile the operand
    for tc in ("A", "B"):
        got = sorted((t, r) for c, t, _, r in reads if c == tc)
        assert got == [(0, 0), (0, 1), (1, 0), (1, 1)], f"{tc}: {got}"


def test_the_MT_split_read_plan_is_UNCHANGED_by_the_all_axes_projection():
    """The regression guard for the fix above.  `M_split` / `N_split` ARE free modes, so widening
    the projection from `free & region` to `region` must be a no-op there -- including the
    `k` term, since subtracting the region modes from the reduction set removes nothing when the
    region axes were never reduction axes.  576/576 on is what this protects."""
    assert _read_plan({**BF16_NT_KMN, "TDMSplit": [2, 2, 1, 1]}) == [
        ("A", 0, 1, 0), ("B", 0, 1, 0), ("B", 0, 1, 1), ("A", 0, 1, 1),
        ("A", 0, 0, 0), ("B", 0, 0, 0), ("B", 0, 0, 1), ("A", 0, 0, 1)]


def test_an_unsplit_operand_keeps_every_read_in_region_zero():
    reads = _read_plan(BF16_NT_KMN)
    assert {r for _, _, _, r in reads} == {0}
    assert {k for _, _, k, _ in reads} == {0, 1}


def _per_operand(params, tc):
    return sorted((t, k, r) for c, t, k, r in _read_plan(params) if c == tc)


@pytest.mark.parametrize("split,cut,whole", [([1, 1, 2, 1], "A", "B"),
                                             ([1, 1, 1, 2], "B", "A")])
def test_an_ASYMMETRIC_DU_split_leaves_the_UNSPLIT_operand_reading_as_if_unsplit(split, cut, whole):
    """`K_split` is the SHARED reduction axis -- the only region axis two operands can both name --
    and `translate` gives it to both `region_axes` at the finest extent, the unsplit operand being
    present at every value.  Only the CUT operand actually has two storage regions, so only its
    reads owe a region term.

    Without the `operand_regions` gate the whole operand's reads come out
    `region in {0,1}` with `k` pinned to 0 -- a region displacement into a contiguous image, and the
    same unroll half read twice.  768/768 on both asymmetric DU cells while the symmetric ones
    passed 960/1152, which is the signature: it is the ASYMMETRY, not the DU axis."""
    params = {**BF16_NT_KMN, "TDMSplit": split}
    # the cut operand: region carries the reduction, k is within-region
    assert _per_operand(params, cut) == [(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 0, 1)]
    # the whole operand: EXACTLY its unsplit plan, region 0 throughout
    assert _per_operand(params, whole) == _per_operand(BF16_NT_KMN, whole)
    assert _per_operand(params, whole) == [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0)]


def test_an_ASYMMETRIC_MT_split_does_the_same_on_the_free_axis():
    """The MT half never showed this because `M_split` and `N_split` are per-operand by
    construction: an operand is never handed a region axis it does not own.  Pinned so the
    `operand_regions` gate is exercised on both halves."""
    params = {**BF16_NT_KMN, "TDMSplit": [2, 1, 1, 1]}
    assert _per_operand(params, "A") == [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1)]
    assert _per_operand(params, "B") == _per_operand(BF16_NT_KMN, "B")


@pytest.mark.parametrize("split,axes,span", [
    ([1, 1, 1, 1], {"A": (),                       "B": ()},                       {}),
    ([1, 1, 2, 2], {"A": ("K_split",),             "B": ("K_split",)},             {"A": 1, "B": 1}),
    ([1, 1, 2, 1], {"A": ("K_split",),             "B": ()},                       {"A": 1}),
    ([1, 1, 2, 4], {"A": ("K_split",),             "B": ("K_split",)},             {"A": 2, "B": 1}),
    ([1, 1, 2, 3], {"A": ("K_split",),             "B": ("K_split",)},             {"A": 3, "B": 2}),
    ([1, 1, 4, 6], {"A": ("K_split",),             "B": ("K_split",)},             {"A": 3, "B": 2}),
    ([2, 1, 1, 2], {"A": ("M_split",),             "B": ("K_split",)},             {"B": 1}),
    ([2, 2, 2, 2], {"A": ("M_split", "K_split"),   "B": ("N_split", "K_split")},   {"A": 1, "B": 1}),
])
def test_the_K_axis_is_ONE_shared_walk_and_the_split_is_the_operands_span(split, axes, span):
    """`K_inner = gcd(K/A_regions, K/B_regions)`, so there is ONE `K_split` both operands walk.

    An operand owns it as a REGION axis only if it splits K; how finely it splits is carried by
    `region_span` -- the axis values one of its own regions covers -- not by a second axis.
    """
    prog = build_gir(_theta({**BF16_NT_KMN, "DepthU": 384, "MIWaveTile": [6, 6],
                             "MatrixInstruction": [16, 16, 32, 1], "PrefetchLocalRead": 0,
                             "TDMSplit": split}))
    got = {o: prog.meta["region_axes"][o] for o in ("A", "B")}
    assert got == axes, got
    spans = {o: prog.meta["region_span"].get(o, {}).get("K_split")
             for o in ("A", "B") if "K_split" in got[o]}
    assert spans == span, spans


@pytest.mark.parametrize("split", [
    [1, 1, 1, 1], [2, 2, 1, 1], [3, 3, 1, 1], [2, 3, 1, 1], [6, 2, 1, 1],   # MT, incl. 3 and 6
    [1, 1, 2, 2], [1, 1, 3, 3], [1, 1, 2, 1], [1, 1, 2, 4], [1, 1, 4, 2],
    [1, 1, 2, 3], [1, 1, 3, 4], [1, 1, 4, 6],                               # COPRIME + gcd 2
    [2, 2, 2, 2], [2, 1, 1, 2], [3, 2, 2, 3], [4, 3, 3, 4],                 # both axes at once
])
def test_the_listed_region_axes_multiply_to_EXACTLY_the_operands_region_count(split):
    """The invariant, over arbitrary `[A_MT, B_MT, A_DU, B_DU]`: an operand's listed region axes
    divided by its own SPAN on each multiply to exactly `op.split`.  The axis counts the shared
    walk's steps; the span says how many of them one of this operand's regions covers.  Over-listing
    is what made a 2-region operand's reads name regions 0..3; under-listing leaves a region nothing
    addresses.  Both are one equation."""
    from math import prod
    wave_tile = [max(1, split[0]), max(1, split[1])]
    prog = build_gir(_theta({**BF16_NT_KMN, "DepthU": 384, "MIWaveTile": wave_tile,
                             "MatrixInstruction": [16, 16, 32, 1], "PrefetchLocalRead": 0,
                             "TDMSplit": split}))
    ext = prog.meta["axis_extents"]
    # and the count is the operand's OWN product of factors, not the peer's and not the lcm
    assert (prog.meta["operand_regions"]["A"],
            prog.meta["operand_regions"]["B"]) == (split[0] * split[2], split[1] * split[3])
    # A AND B ONLY, DELIBERATELY.
    for op in ("A", "B"):
        span = prog.meta["region_span"].get(op, {})
        listed = prod([ext[m] // max(1, span.get(m, 1))
                       for m in prog.meta["region_axes"][op]] or [1])
        assert listed == prog.meta["operand_regions"][op], (
            "%s: region_modes %s over span %s multiply to %d but the operand has %d regions"
            % (op, prog.meta["region_axes"][op], span, listed,
               prog.meta["operand_regions"][op]))


def test_the_transpose_path_takes_NO_vector_group_jump():
    """The two paths mean different things by `tileStrideElems`: unroll-major stores the ROW stride
    (`DepthU + pad`), the transpose path stores `MIWaveGroupShape` itself -- already a per-TILE
    stride.  Applying the row formula there would count the group jump twice, so unsplit the index
    passes through as the flat tile number."""
    import types
    from Tensile.Lowering.lds_geometry import tile_row
    ctx = types.SimpleNamespace(unrollMajor=False, vectorWidth=2, tile01=0,
                                MIWaveGroupShape=[32, 32], miWaveTileAxis=4, nsplit=1)
    assert [tile_row(ctx, t) for t in range(4)] == [0, 1, 2, 3]


@pytest.mark.parametrize("miWaveTile,nsplit,expect", [
    (2, 2, [0, 0]),                 # 1 tile per region -- the MT32x32 NT shape
    (4, 2, [0, 1, 0, 1]),           # 2 tiles per region
    (4, 1, [0, 1, 2, 3]),           # unsplit: untouched
])
def test_the_transpose_path_tile_index_is_taken_MODULO_the_region(miWaveTile, nsplit, expect):
    """A tile-major region is a separately packed block whose tiles are numbered from zero, so
    tile `t` reads at in-region index `t % tiles_per_region` and the block itself is reached by
    `region_bytes`.  Splitting the displacement this way is what keeps the two terms independent:
    the region picks the block, the index picks the tile inside it.

    On `MT32x32x64 NT`, MIWaveTile[2,2], nsplit 2: `tile=1` is the only tile of
    region 1, so its in-region index is 0 and its whole displacement is the 3072-byte block base
    -- not the 32-byte in-row step the global index would have given."""
    import types
    from Tensile.Lowering.lds_geometry import tile_row
    ctx = types.SimpleNamespace(unrollMajor=False, vectorWidth=1, tile01=0,
                                MIWaveGroupShape=[16, 16], miWaveTileAxis=miWaveTile, nsplit=nsplit)
    assert [tile_row(ctx, t) for t in range(miWaveTile)] == expect


# ===========================================================================
# verify_dataflow -- the semantic checker.  Catch these at the plan, not on hardware.
# ===========================================================================
_SPLITS = [None, [1, 1, 2, 2], [2, 2, 1, 1], [1, 1, 2, 1], [1, 1, 1, 2], [2, 1, 1, 1], [1, 2, 1, 1]]


def _sweep_params():
    for order in ("KMN", "KNM", "MKN", "NKM", "MNK", "NMK"):
        for split in _SPLITS:
            for pgr in (1, 2):
                for plr in (0, 1):
                    p = {**BF16_NT_KMN, "LoopOrder": order,
                         "PrefetchGlobalRead": pgr, "PrefetchLocalRead": plr}
                    if split:
                        p["TDMSplit"] = split
                    yield ("%s split=%s PGR%d PLR%d" % (order, split, pgr, plr), p)


def test_the_whole_order_x_split_x_prefetch_matrix_has_a_CLEAN_emit_plan():
    """The standing sweep -- 168 configurations, every one checked for the four defect shapes that
    cost a hardware build cycle each on..18.
    """
    from Tensile.Lowering.gir import check_plan
    bad = {}
    for label, params in _sweep_params():
        v = check_plan(build_gir(_theta(params)))
        if v:
            bad[label] = v
    assert not bad, "\n".join("%s\n    %s" % (k, "\n    ".join(v)) for k, v in bad.items())


def _mutated(params, fn, phase="steady"):
    """The real plan with `fn` applied to each act of `phase` -- how a test re-creates a defect
    without reverting the source that fixed it."""
    from Tensile.Lowering.gir import plan_program
    from Tensile.Lowering.gir.emit_plan import EmitAction
    prog = build_gir(_theta(params))
    plans = plan_program(prog)
    plans[phase] = [EmitAction(a.kind, fn(dict(a.at)) if a.kind == "read" else a.at)
                    for a in plans[phase]]
    return prog, plans


def test_the_checker_catches_A_REGION_THAT_IS_NEVER_READ():
    """/ the DU region term: `region` was projected over `free & region_modes`, which empties
    for `K_split`, so every read came out region 0 -- region 1 written by the copy and never read,
    region 0 read twice.  Reproduced by forcing the region back to 0."""
    from Tensile.Lowering.gir import check_plan
    prog, plans = _mutated({**BF16_NT_KMN, "TDMSplit": [1, 1, 2, 2]},
                           lambda at: {**at, "region": 0})
    v = check_plan(prog, plans)
    assert any("region" in x and "does not have" not in x for x in v), v
    assert any("nothing reads" in x for x in v), v


def test_the_checker_catches_A_REGION_THE_OPERAND_DOES_NOT_HAVE():
    """The asymmetric-DU defect: `K_split` is the SHARED reduction axis, so the UNSPLIT operand's
    reads were labelled `region in {0,1}` against a contiguous image.  Reproduced by relabelling
    the whole operand's reads."""
    from Tensile.Lowering.gir import check_plan
    prog, plans = _mutated({**BF16_NT_KMN, "TDMSplit": [1, 1, 2, 1]},
                           lambda at: {**at, "region": 1} if at["tc"] == "B" else at)
    assert any("does not have" in x for x in check_plan(prog, plans)), check_plan(prog, plans)


def test_the_checker_catches_TWO_GENERATIONS_SHARING_ONE_REGISTER():
    """The register-rate defect: `K_split` was excluded from the rate, W collapsed to 1, and both
    regions loaded into `ValuA_X0_I0+0` -- the second read clobbering the first before its wmma ran.
    Reproduced by forcing every read into buffer 0."""
    from Tensile.Lowering.gir import check_plan
    prog, plans = _mutated({**BF16_NT_KMN, "TDMSplit": [1, 1, 2, 2]},
                           lambda at: {**at, "reg_buf": 0})
    violations = check_plan(prog, plans)
    assert any(kind in x for x in violations
               for kind in ("OVERWRITE-BEFORE-USE", "USE-BEFORE-DEF")), violations


def test_the_checker_catches_A_PROLOGUE_THAT_PRIMES_TOO_FEW_SUBSTEPS():
    """The K-innermost peel defect: `prefetch_axis_mode` picked the degenerate `K_inner`, so the
    read-ahead prologue primed one substep where the steady body consumes two and the first
    `wmma u=1` read an unwritten generation.  Reproduced by dropping the peel's last read."""
    from Tensile.Lowering.gir import check_plan, plan_program
    from Tensile.Lowering.gir.verify_dataflow import prologue_blocks
    params = {**BF16_NT_KMN, "LoopOrder": "MNK", "TDMSplit": [1, 1, 2, 2],
              "PrefetchLocalRead": 1, "PrefetchGlobalRead": 2}
    prog = build_gir(_theta(params))
    plans = plan_program(prog)
    # The peel's reads are in whichever of its blocks the fill ended up in, not always the first.
    fill = next(lab for lab in reversed(prologue_blocks(prog))
                if any(a.kind == "read" for a in plans[lab]))
    reads = [i for i, a in enumerate(plans[fill]) if a.kind == "read"]
    assert len(reads) >= 2, "the probe needs a peel that primes more than one generation"
    plans[fill] = [a for i, a in enumerate(plans[fill]) if i != reads[-1]]
    assert any("USE-BEFORE-DEF" in x for x in check_plan(prog, plans)), check_plan(prog, plans)


@pytest.mark.parametrize("split", [[1, 1, 2, 4], [1, 1, 4, 2], [2, 2, 2, 2],
                                   [1, 1, 2, 3], [1, 1, 4, 6], [3, 2, 2, 3]])
def test_an_operand_on_TWO_region_axes_WALKS_with_a_per_axis_step(split):
    """an operand split on two axes at once -- `M_split x K_split`, or both K-chain links when
    the DU factors are unequal.
    """
    from Tensile.Lowering.gir.analyses.region_increment import walk_violations
    from Tensile.Lowering.gir import plan_program
    prog = build_gir(_theta({**BF16_NT_KMN, "DepthU": 384, "MIWaveTile": [6, 6],
                             "MatrixInstruction": [16, 16, 32, 1], "PrefetchLocalRead": 0,
                             "TDMSplit": split}))
    assert walk_violations(prog) == []
    for phase, acts in plan_program(prog).items():
        net = {}
        for a in acts:
            if a.kind != "region_inc":
                continue
            ax = a.at.get("axis_steps")
            assert ax, "%s: a region step with no per-axis vector: %s" % (phase, a.at)
            # THE TWO REPRESENTATIONS MUST DESCRIBE ONE STEP.
            rms = prog.meta["region_axes"][a.at["unit"][0]]
            ext = prog.meta["axis_extents"]
            w, weight = 1, {}
            for m in reversed(rms):
                weight[m] = w
                w *= ext[m]
            assert sum(d * weight[m] for m, d in ax.items()) == a.at["steps"], (
                "%s: axis_steps %s weighs %d but steps says %d"
                % (phase, ax, sum(d * weight[m] for m, d in ax.items()), a.at["steps"]))
            for m, d in ax.items():
                net.setdefault(a.at["unit"], {}).setdefault(m, 0)
                net[a.at["unit"]][m] += d
        for unit, per_axis in net.items():          # every axis returns to 0 within the block
            assert all(v == 0 for v in per_axis.values()), "%s %s: %s" % (phase, unit, per_axis)


def test_a_multi_axis_step_is_expressed_by_GIR_and_REFUSED_by_L3():
    """The plan's division of labour: GIR gains the 2-D capability, L3 does not.  A grid step has
    no realization in `tdmRegionIncrementGir`, which walks one region along one axis, so emitting
    `|steps|` single-axis walks for it would be silently wrong.  `TDMSplitA/B in {0,1,2}` cannot
    produce this shape, so nothing shippable reaches the refusal."""
    from Tensile.Lowering.gir import plan_program
    prog = build_gir(_theta({**BF16_NT_KMN, "DepthU": 384, "MIWaveTile": [6, 6],
                             "MatrixInstruction": [16, 16, 32, 1], "PrefetchLocalRead": 0,
                             "TDMSplit": [2, 2, 2, 2]}))
    multi = [a.at for acts in plan_program(prog).values() for a in acts
             if a.kind == "region_inc" and len(a.at.get("axis_steps") or {}) > 1]
    assert multi, "the 2x2 grid must produce at least one genuinely multi-axis step"


# ----------------------------------------------------------- the yaml's six-axis words
_YAML_SIX_AXIS_WORDS = ["KMKNMN", "KMNKMN", "KMNMNK"]


def _ord_of(word, splitA, splitB):
    """The loop order a REAL kernel produces, through the bridge -- not a hand-built theta."""
    from Tensile.LoopModel.adapter import kernel_to_params
    from Tensile.LoopModel import adapter
    k = dict(_KERNEL_6AX, LoopOrder=word, TDMSplitA=splitA, TDMSplitB=splitB)
    return tuple(m.name for m in adapter.params_to_theta(kernel_to_params(k)).inner_axes())


_KERNEL_6AX = {
    "MatrixInstruction": [16, 16, 32, 1, 1, 4, 4, 1, 1], "MIWaveTile": [4, 4],
    "MIWaveTileA": 4, "MIWaveTileB": 4, "MIWaveGroup": [1, 1], "DepthU": 64,
    "ProblemType": {"Sparse": 0}, "PrefetchGlobalRead": 2, "PrefetchLocalRead": 1,
    "GlobalReadVectorWidthA": 8, "GlobalReadVectorWidthB": 8,
    "VectorWidthA": 1, "VectorWidthB": 1, "LocalReadVectorWidth": 8,
    "DirectToVgprA": False, "DirectToVgprB": False, "InnerUnroll": 1,
    "NumLdsBlk": 2, "1LDSBuffer": 0, "TDMFuse": 0, "NumWaves": 1,
    "WaveSeparateGlobalReadA": 0, "WaveSeparateGlobalReadB": 0,
}


def test_short_fold_owns_drain_refill_order_for_aliased_physical_vgprs():
    """The raw split shape conflicts; folding merges the short arm's correct register order."""
    from Tensile.LoopModel import adapter
    from Tensile.LoopModel.adapter import kernel_to_params
    from Tensile.Lowering.gir.verify_dataflow import check_register_dataflow

    kernel = dict(
        _KERNEL_6AX,
        LoopOrder="KMNMNK",
        TDMSplitA=1,
        TDMSplitB=1,
        PrefetchGlobalRead=1,
        VectorWidthA=2,
        VectorWidthB=2,
    )
    theta = adapter.params_to_theta(kernel_to_params(kernel))
    raw = lower_to_gir(theta)
    assert any("drain0:" in violation and "WRONG SOURCE" in violation
               for violation in check_register_dataflow(raw))

    folded_prog = build_gir(theta)
    assert check_register_dataflow(folded_prog) == []


def _valid_loop_orders():
    """`ValidParameters["LoopOrder"]`, evaluated from source."""
    import pathlib as _pl
    from itertools import permutations
    src = (_pl.Path(__file__).parents[2] / "Common" / "ValidParameters.py").read_text()
    i = src.index('"LoopOrder":') + len('"LoopOrder":')
    depth, j = 0, i
    while j < len(src):
        c = src[j]
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        elif c == "," and depth == 0:
            break
        j += 1
    ns = {"_permutations": permutations}
    exec("VAL = " + src[i:j].strip(), ns)
    return ns["VAL"]


def test_EVERY_SPELLING_is_accepted_and_canonicalizes_to_90_traversals():
    """Two sets that must not be confused, and I confused them once already."""
    from Tensile.LoopModel.adapter import canonical_loop_order
    accepted = _valid_loop_orders()
    assert len(accepted) == len(set(accepted)) == 96, len(accepted)
    canon = {canonical_loop_order(w) for w in accepted}
    assert len(canon) == 90, sorted(canon)[:5]
    assert canonical_loop_order("KKMMNN") == "KMN"
    assert canonical_loop_order("MMNNKK") == "MNK"
    assert canonical_loop_order("KMKNMN") == "KMKNMN"      # interleaved: no 3-letter equivalent
    # the 6 collapsing pairs are exactly the doubled spellings
    assert {w for w in accepted if canonical_loop_order(w) != w} == {
        "KKMMNN", "KKNNMM", "MMKKNN", "MMNNKK", "NNKKMM", "NNMMKK"}


def test_every_one_of_the_90_orders_DECODES_and_gets_a_DISTINCT_name():
    """Both halves matter.  A word that does not decode is a kernel that fails to generate; two
    words sharing a kernel-name fragment dedupe and one silently never runs."""
    from Tensile.LoopModel import emit_mainloop
    from Tensile.LoopModel import adapter
    from Tensile.LoopModel.adapter import canonical_loop_order
    abbr, val = _naming_helpers()
    frag = {}
    for w in {canonical_loop_order(x) for x in _valid_loop_orders()}:
        th = adapter.params_to_theta({**BF16_NT_KMN, "LoopOrder": w, "TDMSplit": [2, 2, 1, 1],
                                        "MIWaveTile": [4, 4], "PrefetchLocalRead": 0})
        assert emit_mainloop(th).obligations_discharged, w
        frag[w] = abbr("LoopOrder") + val("LoopOrder", w)
    assert len(set(frag.values())) == 90, "kernel-name collision among the 90"


def test_the_yaml_six_axis_words_are_VALID_PARAMETERS():
    """A typo here is not a test failure, it is a silently skipped block: `ValidParameters` is what
    `Solution.py` checks the yaml against."""
    listed = set(_valid_loop_orders())
    for w in _YAML_SIX_AXIS_WORDS + ["KMN"]:
        assert w in listed, "%s missing from ValidParameters LoopOrder" % w


def test_the_yaml_six_axis_words_name_THREE_DISTINCT_non_contiguous_traversals():
    """Block 11 exists to reach the multi-body peel, so its words must (a) differ from each
 other and (b) each be non-contiguous. Two of the candidates I first picked (`KMNMNK` and
 `MNMNKK`) produce the SAME loop order at this config -- `K_split` drops, so words differing only in
 where the dropped mode sat are the same traversal -- which would have run a duplicate under
 three names. Chosen for WHERE THE REDUCTION SITS: between the splits, after them, innermost."""
    from Tensile.LoopModel import emit
    from Tensile.LoopModel import adapter
    from Tensile.LoopModel.adapter import kernel_to_params
    ords = {w: _ord_of(w, 1, 1) for w in _YAML_SIX_AXIS_WORDS}
    assert len(set(ords.values())) == 3, ords
    assert ords["KMKNMN"] == ("M_split", "K_inner", "N_split", "M_inner", "N_inner")
    assert ords["KMNKMN"] == ("M_split", "N_split", "K_inner", "M_inner", "N_inner")
    assert ords["KMNMNK"] == ("M_split", "N_split", "M_inner", "N_inner", "K_inner")
    for w in _YAML_SIX_AXIS_WORDS:                    # each needs the multi-body peel
        th = adapter.params_to_theta(kernel_to_params(
            dict(_KERNEL_6AX, LoopOrder=w, TDMSplitA=1, TDMSplitB=1)))


def test_a_six_axis_word_COLLAPSES_without_a_split_which_is_why_it_is_rejected():
    """The justification for `Solution.py`'s 6-letter gate, as a measurement rather than a claim."""
    # WHAT IT COLLAPSES TO is the 3-letter order its SECOND occurrences spell -- only the inner
    # modes survive -- so the duplicate is not always of `KMN`.  `KMNMNK` collapses to `MNK`, which
    # every block above already runs; I had asserted `KMN` here and the test caught it.
    def _inner_word(w):
        seen, out = set(), []
        for c in w:
            if c in seen:
                out.append(c)
            seen.add(c)
        return "".join(out)
    for w in _YAML_SIX_AXIS_WORDS:
        assert _ord_of(w, 0, 0) == _ord_of(_inner_word(w), 0, 0), \
            "%s collapsed to %s, not to its inner word %s" % (w, _ord_of(w, 0, 0), _inner_word(w))
        assert _ord_of(w, 0, 0) != _ord_of(w, 1, 1)      # the split is what makes the word real
    assert {_inner_word(w) for w in _YAML_SIX_AXIS_WORDS} == {"KMN", "MNK"}


def _naming_helpers():
    """The real `Naming.py` abbreviation functions, loaded from source."""
    import pathlib
    src = (pathlib.Path(__file__).parents[2] / "SolutionStructs" / "Naming.py").read_text().split("\n")

    def grab(name):
        i = next(k for k, l in enumerate(src) if l.startswith("def " + name))
        j = next((k for k in range(i + 1, len(src))
                  if src[k].startswith("def ") or src[k].startswith("@")), len(src))
        return "\n".join(src[i:j])

    ns = {"ProblemType": type("ProblemType", (), {})}
    for f in ("getParameterNameAbbreviation", "getPrimitiveParameterValueAbbreviation",
              "getParameterValueAbbreviation"):
        exec(compile(grab(f), "<naming>", "exec"), ns)
    return ns["getParameterNameAbbreviation"], ns["getParameterValueAbbreviation"]


def test_every_LoopOrder_VALUE_gets_a_DISTINCT_kernel_name_fragment():
    """A kernel-name collision is not a wrong answer, it is a MISSING kernel: two solutions with
    the same name dedupe and only one is generated, so the benchmark reports a matrix it never ran.
    """
    abbr, val = _naming_helpers()
    words = ["KMN", "KNM", "MKN", "MNK", "NKM", "NMK"] + _YAML_SIX_AXIS_WORDS
    frag = {w: abbr("LoopOrder") + val("LoopOrder", w) for w in words}
    assert len(set(frag.values())) == len(words), "kernel-name collision: %s" % frag
    assert frag["KMN"] == "LOKMN" and frag["KMKNMN"] == "LOKMKNMN"


def test_the_TDMSplit_and_TDMFuse_params_REACH_the_kernel_name():
    """A param absent from `RequiredParameters` does not appear in the name, so kernels differing
    only in it dedupe.  Pinned because the abbreviations are ALGORITHMIC (uppercase letters of the
    param name), so renaming a param silently renames its name fragment:
        TDMSplitA -> TDMSA,  TDMSplitB -> TDMSB,  TDMFuse -> TDMF,  LoopOrder -> LO
    matching the emitted `_TDMI3_TDMIM0_TDMLWS0_TDMSA0_TDMSB0_TIN0_` of a real kernel, whose
    default grouping is hidden so naming it would not rename every shipped kernel."""
    import pathlib, re
    abbr, val = _naming_helpers()
    assert (abbr("TDMSplitA"), abbr("TDMSplitB"), abbr("TDMFuse"), abbr("LoopOrder")) \
        == ("TDMSA", "TDMSB", "TDMF", "LO")
    assert [abbr("TDMSplitA") + val("TDMSplitA", v) for v in (0, 1, 2)] \
        == ["TDMSA0", "TDMSA1", "TDMSA2"]
    req = (pathlib.Path(__file__).parents[2] / "Common" / "RequiredParameters.py").read_text()
    for p in ("TDMSplitA", "TDMSplitB", "TDMFuse", "LoopOrder"):
        assert re.search(r"['\"]%s['\"]" % p, req), "%s missing from RequiredParameters" % p


@pytest.mark.parametrize("split,packed_layout", [([1, 1, 2, 2], True), ([1, 1, 2, 2], False),
                                                 ([2, 2, 1, 1], True), ([2, 2, 1, 1], False)])
def test_the_ADDRESS_KEYS_are_distinct_under_BOTH_layouts(split, packed_layout):
    """The check that would have caught the NT contiguous-DU defect (`check_address_keys`)."""
    from Tensile.Lowering.gir import check_address_keys
    prog = build_gir(_theta({**BF16_NT_KMN, "DepthU": 256, "MIWaveTile": [4, 4],
                             "MatrixInstruction": [16, 16, 32, 1], "PrefetchLocalRead": 0,
                             "TDMSplit": split}))
    assert check_address_keys(prog, lambda op: packed_layout) == []


def test_check_address_keys_CATCHES_the_contiguous_split_collapse():
    """The measured NT defect, reconstructed: a CONTIGUOUS split whose address is nevertheless
    built from `(region, within-region k)`.  Two sources then share one key, which is exactly the
    `r1` read repeating `r0`'s bytes."""
    from Tensile.Lowering.gir import check_address_keys, plan_program
    from Tensile.Lowering.gir.emit_plan import EmitAction
    prog = build_gir(_theta({**BF16_NT_KMN, "DepthU": 256, "MIWaveTile": [4, 4],
                             "MatrixInstruction": [16, 16, 32, 1], "PrefetchLocalRead": 0,
                             "TDMSplit": [1, 1, 2, 2]}))
    acts = plan_program(prog)["steady"]
    # CONTIGUOUS (no region term) but keeping the WITHIN-REGION coordinate: the pre-fix leaf.
    bad = check_address_keys(prog, lambda op: False, acts=acts, force_within=True)
    good = check_address_keys(prog, lambda op: False, acts=acts)
    assert good == [], good
    assert any("SAME address key" in v for v in bad), bad
    # the packed decomposition is injective on its own -- the defect is the DECOUPLING, and a
    # check that only modelled the two correct schemes could not have seen it
    assert check_address_keys(prog, lambda op: True, acts=acts) == []


# ============================================================ fp8 through the product layer
class _Bytes:
    """A DataType stand-in that answers only what `adapter.kernel_to_params` asks of it."""
    def __init__(self, n):
        self.n = n

    def numBytes(self):
        return self.n


def _fp8_kernel(**over):
    """A plain (unscaled) fp8 Solution dict, swept through `adapter.kernel_to_params` rather than
    handing theta an `ElemBytes` directly -- a dropped Solution key is invisible below the bridge."""
    k = {**BF16_NT_KMN, "MatrixInstruction": [16, 16, 128, 1, 1, 2, 2, 1, 1], "DepthU": 256,
         "LocalReadVectorWidth": 16, "LocalReadVectorWidthA": 16, "LocalReadVectorWidthB": 16,
         "ProblemType": {"DataType": _Bytes(1), "DestDataType": _Bytes(1),
                         "ComputeDataType": _Bytes(4), "TransposeA": True, "TransposeB": False}}
    k.update(over)
    return k


@pytest.mark.parametrize("pgr", [1, 2])
@pytest.mark.parametrize("plr", [0, 1])
@pytest.mark.parametrize("order", ["KMN", "KNM", "MKN", "MNK", "NKM", "NMK"])
def test_fp8_decodes_and_lowers_to_the_SAME_WORK_as_bf16(pgr, plr, order):
    """An 8-bit input changes ONE thing in theta: `ElemBytes`.  It is not a reduction-depth change --
    fp8's `MatrixInstruction` K is twice bf16's and its `DepthU` twice again, so the substep count
    `n_s` is the same 2 -- and the schedule is therefore the same schedule.
    """
    from Tensile.LoopModel.adapter import kernel_to_params
    from Tensile.Lowering.gir.nodes import Move, Mma
    from Tensile.Lowering.gir.verify_dataflow import check_plan
    from Tensile.Lowering.gir.analyses.region_increment import walk_violations

    def _counts(k):
        prog = build_gir(adapter.params_to_theta(kernel_to_params(k)))
        steady = [b for n, b in prog.blocks.items() if n.startswith("steady")]
        assert check_plan(prog) == []
        assert walk_violations(prog) == []
        return (sum(1 for b in steady for i in b.body if isinstance(i, Mma)),
                sum(1 for b in steady for i in b.body if isinstance(i, Move)))

    over = {"PrefetchGlobalRead": pgr, "PrefetchLocalRead": plr, "LoopOrder": order}
    fp8 = _fp8_kernel(**over)
    bf16 = {**BF16_NT_KMN, **over,
            "ProblemType": {**BF16_NT_KMN["ProblemType"], "DataType": _Bytes(2)}}
    assert kernel_to_params(fp8)["ElemBytes"] == 1
    assert kernel_to_params(bf16)["ElemBytes"] == 2
    assert _counts(fp8) == _counts(bf16)


@pytest.mark.parametrize("knob,want", [(0, [1, 1, 1, 1]), (1, [2, 2, 1, 1]), (2, [1, 1, 2, 2])])
def test_fp8_is_clean_under_TDMSplit_on_both_axes(knob, want):
    """The split is a COORDINATE fact and the element width is an ADDRESS fact, so they should be
    independent -- this is the check that they are, at the layer where a coupling would show.
    """
    from Tensile.LoopModel.adapter import kernel_to_params
    from Tensile.Lowering.gir.verify_dataflow import check_plan
    from Tensile.Lowering.gir.analyses.region_increment import walk_violations
    k = _fp8_kernel(TDMSplitA=knob, TDMSplitB=knob, MIWaveTile=[4, 4], PrefetchLocalRead=0,
                    ProblemType={**_fp8_kernel()["ProblemType"], "Sparse": 0})
    p = kernel_to_params(k)
    assert list(p["TDMSplit"]) == want
    prog = build_gir(adapter.params_to_theta(p))
    assert check_plan(prog) == []
    assert walk_violations(prog) == []


# ============================================================ MX scale operands reach L3
def _mx_kernel(**over):
    """The plain fp8 kernel plus microscaling, still through `adapter.kernel_to_params`."""
    k = _fp8_kernel(**over)
    k["ProblemType"] = {**k["ProblemType"], "MXBlockA": 32, "MXBlockB": 32, "Sparse": 0}
    return k


def test_the_bridge_CARRIES_MXBlock_so_theta_builds_the_scale_operands():
    from Tensile.LoopModel.adapter import kernel_to_params
    p = kernel_to_params(_mx_kernel())
    assert (p["MXBlockA"], p["MXBlockB"]) == (32, 32)
    names = [o.name for o in adapter.params_to_theta(p).operands]
    assert names == ["A", "B", "C", "MXSA", "MXSB"]
    # and it stays off for a kernel without scales -- the key is read, not defaulted on
    p0 = kernel_to_params(_fp8_kernel())
    assert (p0["MXBlockA"], p0["MXBlockB"]) == (0, 0)
    assert [o.name for o in adapter.params_to_theta(p0).operands] == ["A", "B", "C"]


@pytest.mark.parametrize("pgr", [1, 2, 3])
@pytest.mark.parametrize("plr", [0, 1])
def test_the_wmma_act_carries_the_SCALE_SLOTS_paired_by_presence(plr, pgr):
    """The scale slot must be the one the producing MXS read wrote, exactly as for A and B.
    The pairing is by FREE MODES, not by name: `free_modes[MXSA] == free_modes[A]` because a scale
    follows its parent's free axis.
    """
    from Tensile.LoopModel.adapter import kernel_to_params
    from Tensile.Lowering.gir import plan_program
    prog = build_gir(adapter.params_to_theta(
        kernel_to_params(_mx_kernel(PrefetchLocalRead=plr, PrefetchGlobalRead=pgr))))
    acts = plan_program(prog)["steady"]
    wmma = [a for a in acts if a.kind == "wmma"]
    reads = [a for a in acts if a.kind == "read"]
    assert wmma and reads
    assert {a.at["tc"] for a in reads} == {"A", "B", "MXSA", "MXSB"}
    for a in wmma:
        assert a.at["bufMXA"] is not None and a.at["bufMXB"] is not None
    # DEF-USE, per operand: the read of (tile, k) writes the slot the wmma at that (tile, k)
    # consumes.  Checked for the scales AND for their parents, so the two are held to the same
    # standard and a scale slot that silently tracked the wrong operand would show as a mismatch.
    for tc, parent_key, wmma_key, idx_key in (("MXSA", "bufA", "bufMXA", "idx0"),
                                              ("MXSB", "bufB", "bufMXB", "idx1"),
                                              ("A", "bufA", "bufA", "idx0"),
                                              ("B", "bufB", "bufB", "idx1")):
        defs = {(a.at["tile"], a.at["k"]): a.at["reg_buf"] for a in reads if a.at["tc"] == tc}
        for a in wmma:
            assert defs[(a.at[idx_key], a.at["u"])] == a.at[wmma_key], (tc, a.at)


def test_a_kernel_without_scales_carries_NO_slot_rather_than_a_zero():
    """`None`, not 0 -- the leaf switches on the kernel's own MXBlock, but a 0 here would read as
    'slot 0' and silently name a register on a kernel that has no scale ring at all."""
    from Tensile.LoopModel.adapter import kernel_to_params
    from Tensile.Lowering.gir import plan_program
    prog = build_gir(adapter.params_to_theta(kernel_to_params(_fp8_kernel())))
    for a in plan_program(prog)["steady"]:
        if a.kind == "wmma":
            assert a.at["bufMXA"] is None and a.at["bufMXB"] is None


def test_an_unpairable_extra_register_source_is_REFUSED():
    """A register src that is neither a matmul input nor a the axes it varies over-match for one has no faithful
    place on the wmma, so `_plan_mma` raises instead of dropping it.  Dropping it would emit a
    scaled wmma with an unscaled operand -- numerically wrong with nothing in the assembly naming
    the cause."""
    from Tensile.LoopModel.adapter import kernel_to_params
    from Tensile.Lowering.gir import plan_program
    prog = build_gir(adapter.params_to_theta(kernel_to_params(_mx_kernel())))
    # make MXSA's free modes match NEITHER input
    prog.meta["free_axes"] = {**prog.meta["free_axes"], "MXSA": ["K_inner"]}
    with pytest.raises(NotImplementedError) as e:
        plan_program(prog)
    assert "must follow exactly one parent" in str(e.value)


@pytest.mark.parametrize("pgr", [1, 2, 3])
@pytest.mark.parametrize("plr", [0, 1])
@pytest.mark.parametrize("nldsblk", [1, 2])
def test_the_scale_ring_has_ITS_PARENTSlds_buffers(plr, pgr, nldsblk):
    """A scale block is not a ring of its own: it lives in the SAME LDS allocation as its parent
    and rotates with the same buffer index (one swap moves A, B and their scales together).  So
    `S_shared(MXSA) == S_shared(A)` for every prefetch depth and buffer count.
    """
    from Tensile.LoopModel.adapter import kernel_to_params
    from Tensile.LoopModel.schedule import build_S
    th = adapter.params_to_theta(kernel_to_params(
        _mx_kernel(PrefetchLocalRead=plr, PrefetchGlobalRead=pgr, NumLdsBlk=nldsblk)))
    S, _ = build_S(th)
    depth = {o.name: S.shared(o.name) for o in th.operands if o.movements}
    assert depth["MXSA"] == depth["A"], depth
    assert depth["MXSB"] == depth["B"], depth
    assert depth["A"] == nldsblk, depth


@pytest.mark.parametrize("over,want", [
    # the preset chains: kernel NumLdsBlk -> LDSBuffer<parent> -> LDSBufferMXS<parent>
    ({}, {"A": 2, "MXSA": 2, "B": 2, "MXSB": 2}),
    ({"LDSBufferA": 1}, {"A": 1, "MXSA": 1, "B": 2, "MXSB": 2}),
    # ... and each level is still independently overridable
    ({"LDSBufferMXSA": 1}, {"A": 2, "MXSA": 1, "B": 2, "MXSB": 2}),
])
def test_the_scale_ring_is_ITS_OWN_knob_preset_to_the_parents(over, want):
    """`LDSBufferMXSA/B` are real parameters, not aliases: preset to the PARENT's resolved depth
    (which itself presets to the kernel's `NumLdsBlk`), and overridable so the depths can be
    searched apart -- the same treatment `LDSBufferA/B` get relative to `NumLdsBlk`.
    """
    from Tensile.LoopModel.schedule import build_S
    p = {"MatrixInstruction": [16, 16, 128, 1], "DepthU": 256, "MIWaveTile": [2, 2],
         "ElemBytes": 1, "MXBlockA": 32, "MXBlockB": 32, "NumLdsBlk": 2, **over}
    th = adapter.params_to_theta(p)
    S, _ = build_S(th)
    assert {o.name: S.shared(o.name) for o in th.operands if o.movements} == want


def test_the_scale_prefetch_depth_is_its_own_knob():
    """`off(copy)` per operand, so a scale may
 prefetch to a different depth than the operand it scales."""
    p = {"MatrixInstruction": [16, 16, 128, 1], "DepthU": 256, "MIWaveTile": [2, 2],
         "ElemBytes": 1, "MXBlockA": 32, "MXBlockB": 32, "PrefetchGlobalRead": 2}
    off = {k[0]: v for k, v in
           adapter.params_to_theta(p).movement_offsets().items() if k[1] == "copy"}
    assert off == {"A": 2, "B": 2, "MXSA": 2, "MXSB": 2}
    off = {k[0]: v for k, v in
           adapter.params_to_theta({**p, "PrefetchGlobalReadMXSA": 1}).movement_offsets().items()
           if k[1] == "copy"}
    assert off == {"A": 2, "B": 2, "MXSA": 1, "MXSB": 2}


def test_a_scale_ring_DEEPER_than_the_allocation_is_refused():
    """Same bound as A/B: `Solution.py` sizes the LDS block from `NumLdsBlk` against MaxLDS, so a
    deeper ring is an out-of-bounds write, not a tighter schedule."""
    p = {"MatrixInstruction": [16, 16, 128, 1], "DepthU": 256, "MIWaveTile": [2, 2],
         "ElemBytes": 1, "MXBlockA": 32, "MXBlockB": 32, "NumLdsBlk": 2, "LDSBufferMXSA": 4}
    with pytest.raises(RuntimeError) as e:
        adapter.params_to_theta(p)
    assert "LDSBufferMXSA=4 exceeds" in str(e.value)


@pytest.mark.parametrize("pgr", [1, 2, 3])
@pytest.mark.parametrize("plr", [0, 1])
@pytest.mark.parametrize("mxblock", [16, 32])
def test_an_MX_kernel_passes_the_LoopIR_STRUCTURAL_GATE(plr, pgr, mxblock):
    """`emit_mainloop` runs the structural gate (P1-P9) on every decode and RAISES on a
 defect, so this is the check that the scale operands do not break the sigma_c ordering or leave an
 obligation undischarged. `build_gir` reaches it, but only for the (PGR, PLR) it is handed --
 which is why the sweep is the point of this test rather than the single call."""
    from Tensile.LoopModel.adapter import kernel_to_params
    from Tensile.LoopModel.emit import emit_mainloop
    k = _mx_kernel(PrefetchLocalRead=plr, PrefetchGlobalRead=pgr)
    k["ProblemType"] = {**k["ProblemType"], "MXBlockA": mxblock, "MXBlockB": mxblock}
    emit_mainloop(adapter.params_to_theta(kernel_to_params(k)))     # raises on any defect


@pytest.mark.parametrize("plr", [0, 1])
@pytest.mark.parametrize("order", ["KMN", "KNM", "MKN", "MNK", "NKM", "NMK"])
def test_a_scale_read_is_NOT_scheduled_after_the_wmma_that_consumes_it(plr, order):
    """The scale's register ring must rotate as wide as its parent's, or the steady body emits the
    wmma BEFORE the read that defines its scale register.
    """
    from Tensile.LoopModel.adapter import kernel_to_params
    from Tensile.Lowering.gir import plan_program
    prog = build_gir(adapter.params_to_theta(kernel_to_params(
        _mx_kernel(PrefetchLocalRead=plr, LoopOrder=order,
                   MatrixInstruction=[16, 16, 128, 1, 1, 2, 2, 2, 2],
                   NumWaves=4, TDMFuse=0, VectorWidthA=2, VectorWidthB=2))))
    acts = plan_program(prog)["steady"]
    first_def = {}
    for i, a in enumerate(acts):
        if a.kind == "read":
            first_def.setdefault((a.at["tc"], a.at["tile"], a.at["reg_buf"]), i)
    for a in acts:
        if a.kind != "wmma":
            continue
        for parent, scale, idx, pk, sk in (("A", "MXSA", a.at["idx0"], "bufA", "bufMXA"),
                                           ("B", "MXSB", a.at["idx1"], "bufB", "bufMXB")):
            p_def = first_def.get((parent, idx, a.at[pk]))
            s_def = first_def.get((scale, idx, a.at[sk]))
            assert (p_def is None) == (s_def is None), (
                "%s and its scale %s disagree about which trip the wmma at (tile=%s, u=%s) reads "
                "from: parent def=%s, scale def=%s" % (parent, scale, idx, a.at["u"], p_def, s_def))


@pytest.mark.parametrize("plr", [0, 1, 2])
def test_the_scale_ring_matches_its_parents_width(plr):
    """`W(MXSA) == W(A)`: the scale ring rides with its parent, at the same derived width."""
    from Tensile.LoopModel.adapter import kernel_to_params
    from Tensile.LoopModel.schedule import build_S
    from Tensile.LoopModel.traversal import group_ring_size
    th = adapter.params_to_theta(kernel_to_params(
        _mx_kernel(PrefetchLocalRead=plr, MatrixInstruction=[16, 16, 128, 1, 1, 2, 2, 2, 2],
                   DepthU=512, NumWaves=4, TDMFuse=0, VectorWidthA=2, VectorWidthB=2)))
    S, _ = build_S(th)
    w = {o.name: max(S.get(o.name, g) for g in o.fragment.groups())
         for o in th.operands if o.movements}
    for scale, parent in (("MXSA", "A"), ("MXSB", "B")):
        # NEVER DEEPER; narrower is legitimate when the coverage absorbs the scale's fan.
        assert w[scale] <= w[parent], (scale, parent, w)
        assert th.op(scale).fragment.policy_of(th.op(scale).fragment.groups()[0]) \
            == th.op(parent).fragment.policy_of(th.op(parent).fragment.groups()[0]), \
            "the scale ring must use its parent's width policy, not a second default"
    # NON-VACUITY: the fixture must still reach a shape where `unroll` and `pipeline` DISAGREE,
    # otherwise the equality above holds for free and the regression could return unseen.
    R = max(group_ring_size(th, th.op("MXSA"), g) for g in th.op("MXSA").fragment.groups())
    if plr < 2:
        assert w["MXSA"] < R, ("PLR%d must be a shape where the old `unroll` default would "
                               "DIVERGE; if this fires the fixture stopped exercising it" % plr,
                               w, R)


# ===========================================================================================
# the movement coverage, SUPPLIED not derived
# ===========================================================================================
# The pairing follows from the ADDRESS LAYOUT, and theta is address-opaque, so the coverage is
# computed outside and handed in with the same standing as `S`.  These check the plumbing and the
# structural consequence, never a derivation -- every derivation is an address assumption.


def _q_kernel(word, split, quantum=None):
    """The triage block-6 shape, with an optionally SUPPLIED read coverage."""
    return _mx_kernel(MatrixInstruction=[16, 16, 128, 1, 1, 2, 2, 2, 2], DepthU=256,
                      NumWaves=4, PrefetchGlobalRead=1, PrefetchLocalRead=1,
                      TDMSplitA=split, TDMSplitB=split,
                      UnrollMajorLDSMXSA=1, UnrollMajorLDSMXSB=1,
                      LocalReadVectorWidthMXS=8, LoopOrder=word)


def _wide_load(factor):
    """A `TransferCoverage` for a wide load of `factor` tiles: carrier = t//f, slot = t%f."""
    from Tensile.LoopModel.ir import TransferCoverage, Expr, COVERAGE_VAR
    t = COVERAGE_VAR
    return TransferCoverage(carrier=Expr(digits=(((t, 1),), 0, ((1, factor, 0),))),
                      slot=Expr(digits=(((t, 1),), 0, ((1, 1, factor),))))


def _broadcast(factor):
    """A `TransferCoverage` for a sub-agent broadcast of `factor` partitions: carrier = (t//f)*f,
    slot = 0 -- partners share the register, which `regs` counts as 1."""
    from Tensile.LoopModel.ir import TransferCoverage, Expr, cst, COVERAGE_VAR
    t = COVERAGE_VAR
    return TransferCoverage(carrier=Expr(digits=(((t, 1),), 0, ((factor, factor, 0),))),
                      slot=cst(0))


def _theta_with_quantum(word, split, quantum=None):
    from Tensile.LoopModel.adapter import kernel_to_params
    tgt = {"ReadQuantum": quantum} if quantum else None
    return adapter.params_to_theta(kernel_to_params(_q_kernel(word, split), target=tgt))


def test_the_quantum_defaults_to_ONE_TILE_when_nothing_supplies_it():
    """No target facts -> no coverage -> one tile per instruction, which is always admissible.
    A standalone caller with no layout knowledge must get the narrow form and a correct kernel,
    never a guess. Codegen's `OutputLoopIR` uses its cached target-aware theta instead.
    """
    from Tensile.LoopModel.adapter import kernel_to_params
    p = kernel_to_params(_q_kernel("KMKNMN", 1))
    assert p["ReadQuantum"] == {}
    th = adapter.params_to_theta(p)
    for name in ("A", "B", "MXSA", "MXSB"):
        rd = [h for h in th.op(name).movements if h.dst == "register"][0]
        assert rd.coverage is None, name


def _mx_at(mt, order="KMN", split=0, quantum=None):
    """The MX acceptance shape at an arbitrary `MIWaveTile`, with an optional supplied coverage."""
    k = _mx_kernel(MatrixInstruction=[16, 16, 128, 1, 1, mt[0], mt[1], 2, 2], DepthU=256,
                   NumWaves=4, PrefetchGlobalRead=1, PrefetchLocalRead=1,
                   TDMSplitA=split, TDMSplitB=split, UnrollMajorLDSMXSA=1, UnrollMajorLDSMXSB=1,
                   LocalReadVectorWidthMXS=8, LoopOrder=order, MIWaveTile=mt)
    from Tensile.LoopModel.adapter import kernel_to_params
    tgt = {"ReadQuantum": {"MXSA": quantum, "MXSB": quantum}} if quantum else None
    return adapter.params_to_theta(kernel_to_params(k, target=tgt))


def test_a_PARTIAL_quantum_fold_leaves_the_GROUP_INDEX_present_not_the_whole_axis():
    """a `q`-wide coverage folds an axis of extent `N` into `N/q` carrier groups -- the
    intra-group coordinate leaves the axes it varies over, THE GROUP INDEX STAYS. Full absence is only `q = N`.
    """
    from Tensile.LoopModel import traversal as geometry
    # split=2 puts the whole N fan on ONE axis, so `N_inner` is the extent-8 tile axis the
    # `ds_load_b64` folds in pairs -- the `MIWaveTile [2,8]` + TileSpan shape that raised
    # INCOMPLETE CARRIER GROUP.
    th = _mx_at([2, 8], split=2, quantum=_wide_load(2))
    op = th.op("MXSB")
    rd = [h for h in op.movements if h.dst == "register"][0]
    ext = dict(geometry._coverage_candidate_axes(th, op))
    assert ext.get("N_inner") == 8, "non-vacuity: the axis must really be wider than the load"
    assert geometry.coverage_factors(th, op, rd.coverage).get("N_inner") == 2
    # 8 tiles / 2 per instruction = FOUR the axes it varies over points, which is the emitted ds_load count.
    assert ext["N_inner"] // 2 == 4
    # and the axis is NOT fully absorbed -- it still varies, so subtracting it would lose coordinates
    assert "N_inner" not in geometry.coverage_axes(th, op, rd.coverage)


@pytest.mark.parametrize("tpr,split", [(2, 0), (4, 0), (2, 2), (4, 2)])
def test_the_REGULARITY_CLAMP_refuses_to_fold_an_axis_the_load_does_not_divide(tpr, split):
    """"Regularity (`q | N`)": a carrier group is a uniform coverage of `N/q` equal groups, so an
    indivisible coverage has no clean partition and the decoder clamps -- "an odd extent under a two-tile
    coverage clamps to `q=1`, i.e. no coverage, narrower instructions", NEVER a ragged final group.
    """
    from Tensile.LoopModel import traversal as geometry
    th = _mx_at([8, 7], split=split, quantum=_wide_load(tpr))
    b, a = th.op("MXSB"), th.op("MXSA")
    rb = [h for h in b.movements if h.dst == "register"][0]
    ra = [h for h in a.movements if h.dst == "register"][0]
    cand_b = dict(geometry._coverage_candidate_axes(th, b))
    nb, tiles = cand_b["N_inner"], 1
    for _e in cand_b.values():
        tiles *= _e
    assert nb % 2, f"non-vacuity: the N tile axis must be ODD, got {nb}"
    assert tiles > tpr, \
        f"non-vacuity: the operand must need MORE THAN ONE instruction ({tiles} tiles vs {tpr}), " \
        "else this is the whole-span case the boundary test below owns"
    assert geometry.coverage_factors(th, b, rb.coverage) == {}, \
        "an odd axis has no divisor above 1 that a power-of-two load also divides"
    assert geometry.coverage_factors(th, a, ra.coverage).get("M_inner", 1) > 1, \
        "control: the even partner axis of the SAME kernel does fold"


def test_an_axis_NARROWER_than_the_load_is_spanned_WHOLE_not_clamped_away():
    """The boundary the regularity clamp must not swallow: the "the axis is fully absent only
    when one instruction spans it whole (`q = N`)".
    """
    from Tensile.LoopModel import traversal as geometry
    th = _mx_at([8, 7], split=2, quantum=_wide_load(8))
    op = th.op("MXSB")
    rd = [h for h in op.movements if h.dst == "register"][0]
    ext = dict(geometry._coverage_candidate_axes(th, op))
    assert ext["N_inner"] == 7 and 7 < 8, "non-vacuity: axis odd AND narrower than the load"
    assert geometry.coverage_factors(th, op, rd.coverage).get("N_inner") == 7, \
        "one 8-wide instruction spans all 7 tiles: one group, q >= N, fully absorbed"
    assert "N_inner" in geometry.coverage_axes(th, op, rd.coverage), "so it leaves presence entirely"


def test_coverage_axes_is_exactly_the_FULLY_SPANNED_view_of_coverage_factors():
    """The set-valued form every existing caller uses is now derived, not a second rule."""
    from Tensile.LoopModel import traversal as geometry
    saw_full = saw_partial = False
    for mt in ([2, 2], [2, 8], [4, 4], [8, 7]):
        for tpr in (2, 4, 8):
            th = _mx_at(mt, quantum=_wide_load(tpr))
            for nm in ("MXSA", "MXSB"):
                op = th.op(nm)
                rd = [h for h in op.movements if h.dst == "register"][0]
                ext = dict(geometry._coverage_candidate_axes(th, op))
                fac = geometry.coverage_factors(th, op, rd.coverage)
                assert geometry.coverage_axes(th, op, rd.coverage) == \
                    {n for n, f in fac.items() if f >= ext[n]}
                saw_full |= any(f >= ext[n] for n, f in fac.items())
                saw_partial |= any(f < ext[n] for n, f in fac.items())
    assert saw_full and saw_partial, "non-vacuity: the sweep must exercise BOTH sides of the view"


@pytest.mark.parametrize("word", ["KMN", "KMKNMN", "KMNKMN", "KMNMNK"])
def test_a_SUPPLIED_quantum_reaches_the_hop_unchanged_under_every_loop_order(word):
    """theta carries the pairing; it does not re-decide it per order.
    This is the load-bearing property of putting the coverage outside: the emitted grouping cannot
    silently differ between loop orders, because theta has no rule that could make it differ.
    """
    q = _wide_load(2)
    th = _theta_with_quantum(word, 1, {"MXSA": q})
    rd = [h for h in th.op("MXSA").movements if h.dst == "register"][0]
    assert rd.coverage is q, "theta carries the supplied map verbatim"
    from Tensile.LoopModel.schedule import resolve_coverage
    assert resolve_coverage(th) == {"MXSA": q}
    # and it evaluates the same regardless of order -- theta has no rule that could vary it
    assert [q.carrier_of(i) for i in range(4)] == [0, 0, 1, 1]
    assert [q.slot_of(i) for i in range(4)] == [0, 1, 0, 1]


def test_load_and_broadcast_differ_only_in_the_SLOT_RANGE_no_tag():
    """The reason the coverage is an EXPRESSION and not an `(axis, factor, kind)` schema."""
    wide, bcast = _wide_load(2), _broadcast(2)
    assert wide.group(0, 4) == (0, 1) and bcast.group(0, 4) == (0, 1), "same coordinates served"
    assert wide.regs(0, 4) == 2, "a wide load gives each tile its own register"
    assert bcast.regs(0, 4) == 1, "a broadcast maps partners onto one register"


def test_the_carrier_group_is_a_PREIMAGE_so_partners_need_not_be_adjacent():
    """A sub-agent partner sits a stride away, not next door, so "the axes it spans" cannot name
    the group while the carrier preimage always can."""
    from Tensile.LoopModel.ir import TransferCoverage, Expr, cst, COVERAGE_VAR
    t = COVERAGE_VAR
    # carrier = t % 2  ->  groups {0,2} and {1,3}: strided, not contiguous
    strided = TransferCoverage(carrier=Expr(digits=(((t, 1),), 0, ((1, 1, 2),))), slot=cst(0))
    assert strided.group(0, 4) == (0, 2)
    assert strided.group(1, 4) == (1, 3)


def test_the_quantum_reaches_the_emitted_span_end_to_end():
    """theta.Hop.coverage -> geometry.coverage_axes -> the read's the axes it varies over -> Ref.covers -> coords."""
    from Tensile.Lowering import lower_to_gir
    from Tensile.Lowering.gir.nodes import Move
    from Tensile.Lowering.gir.nodes import covered_coords
    q = _wide_load(2)
    th = _theta_with_quantum("KMN", 1, {"MXSA": q})
    prog = lower_to_gir(th)
    defs = [d for blk in prog.blocks.values() for i in blk.body if isinstance(i, Move)
            for d in i.dsts if d.tile.space == "register" and d.tile.operand == "MXSA"]
    assert defs, "no MXSA register def found"
    for d in defs:
        assert d.covers, "the absorbed axis must reach the def"
        (axis, ext), = d.covers
        assert ext == 2
        assert axis not in dict(d.tile.coord), \
            "an absorbed axis is ABSENT from the def's own coord -- that is what absorbing means"
        got = covered_coords(d)
        assert len(got) == 2, "one def, two locations"
        assert {dict(c)[axis] for c in got} == {0, 1}, "and they are the axis's two values"


# --- the sub-agent broadcast is N-way, and it is just an EXPRESSION -----------------------
@pytest.mark.parametrize("parts,tiles,want_groups", [
    (1, 4, [(0,), (1,), (2,), (3,)]),                       # no broadcast: every tile alone
    (2, 4, [(0, 1), (0, 1), (2, 3), (2, 3)]),               # half-wave: pairs share
    (4, 8, [(0, 1, 2, 3)] * 4 + [(4, 5, 6, 7)] * 4),        # quarter-wave: nothing is 2-specific
])
def test_the_sub_agent_broadcast_is_N_way_and_is_just_an_expression(parts, tiles, want_groups):
    """"sub-agent axes are ordinary modes" -- one loaded value serving N partitions."""
    from Tensile.LoopModel.ir import TransferCoverage, Expr, cst, COVERAGE_VAR
    t = COVERAGE_VAR
    q = TransferCoverage(carrier=Expr(digits=(((t, 1),), 0, ((parts, parts, 0),))), slot=cst(0))
    assert [q.group(i, tiles) for i in range(tiles)] == want_groups
    assert q.regs(0, tiles) == 1, "a broadcast writes ONE register however many partitions share it"
    # idempotent: a group's leader never re-folds onto a different instruction
    assert all(q.carrier_of(min(q.group(i, tiles))) == q.carrier_of(i) for i in range(tiles))


def test_a_REGION_difference_is_not_a_quantum_violation():
    """the movement-quantum condition fires on a CHUNK crossing, never on a region one."""
    from Tensile.LoopModel.adapter import kernel_to_params
    from Tensile.LoopModel.schedule import build_S
    from Tensile.LoopModel import schedule
    k = _mx_kernel(MatrixInstruction=[16, 16, 128, 1, 1, 2, 2, 2, 2], DepthU=256, NumWaves=4,
                   MIWaveTileA=2, MIWaveTileB=2, MIWaveTile=[2, 2],
                   PrefetchGlobalRead=1, PrefetchLocalRead=1, TDMSplitA=1, TDMSplitB=1,
                   LoopOrder="KMN")
    th = adapter.params_to_theta(kernel_to_params(k))
    S, _ = build_S(th)
    op = th.op("MXSA")
    pl = schedule._read_placement(th, op, S, prefetch_steps=0)
    assert pl.src_slot.mod == S.shared("MXSA"), (
        "the generation modulus must be the shared RING depth (chunks), not a region count")
    # and the operand really does carry a region axis of extent 2 at this shape, so the two are
    # genuinely different things here rather than coincidentally equal.
    assert any(m.name in (op.region_axes or ()) and m.extent > 1 for m in free_axes(th, op))


def test_L3_OBEYS_the_supplied_merge_and_invents_none_of_its_own():
    """The end of the group-leader rule.
    `leaves.emitLdsReadTile` must not decide on its own authority that an act need not be emitted --
    `tileIdx % ctx.tilePerRead` for the load coverage and a `ctx.tileSpanVW` branch for the half-wave.
    """
    from Tensile.Lowering.gir.coverage import plan_coverage
    from Tensile.Lowering.gir.emit_plan import EmitAction
    from Tensile.LoopModel.ir import TransferCoverage, Expr, cst, COVERAGE_VAR
    acts = [EmitAction("read", {"tc": "MXSA", "tile": i, "tile_flat": i, "k": 0, "region": 0})
            for i in range(4)]

    plain = plan_coverage(acts, lambda op: None)
    # NO MERGE SUPPLIED -> NO DECISION AT ALL, so the leaf keeps its own coverage.  A permissive
    # `Decision(emit=True)` here is what made the leaf skip that coverage and emit one `ds_load` per
    # tile: 16/16 of the mxf8 matrix failed, the 10 previously-green cells included.
    assert all(plain.decision(i) is None for i in range(4)), "silence, not permission"
    assert plain.merged == 0

    t = COVERAGE_VAR
    wide = TransferCoverage(carrier=Expr(digits=(((t, 1),), 0, ((1, 2, 0),))),
                      slot=Expr(digits=(((t, 1),), 0, ((1, 1, 2),))))
    merged = plan_coverage(acts, lambda op: wide, extent_of=lambda op: 4)
    assert [merged.decision(i).emit for i in range(4)] == [True, False, True, False]
    # carrier: tiles 0,1 ride instruction 0 and tiles 2,3 ride instruction 1 -- which is exactly
    # `regTileIdx // tilePerRead`, the base `leaves.emitLdsReadTile` computes for itself.
    assert [merged.decision(i).reg_slot for i in range(4)] == [0, 0, 1, 1]
    assert [wide.slot_of(i) for i in range(4)] == [0, 1, 0, 1], "slot is the position WITHIN"
    assert merged.violations == []


def test_an_INCOMPLETE_carrier_group_is_reported_not_silently_merged():
    """The model's enforcement point, and the only check this layer still makes."""
    from Tensile.Lowering.gir.coverage import plan_coverage
    from Tensile.Lowering.gir.emit_plan import EmitAction
    from Tensile.LoopModel.ir import TransferCoverage, Expr, COVERAGE_VAR
    t = COVERAGE_VAR
    wide = TransferCoverage(carrier=Expr(digits=(((t, 1),), 0, ((1, 2, 0),))),
                      slot=Expr(digits=(((t, 1),), 0, ((1, 1, 2),))))
    acts = [EmitAction("read", {"tc": "MXSA", "tile": 0, "tile_flat": 0, "k": 0, "region": 0})]
    pl = plan_coverage(acts, lambda op: wide, extent_of=lambda op: 4)
    assert len(pl.violations) == 1
    assert "serves 2 tiles but only 1" in pl.violations[0]


def test_transfer_coverage_is_derived_from_instruction_and_agent_group_sizes():
    from Tensile.LoopModel import traversal as geometry
    from Tensile.LoopModel.adapter import kernel_to_params
    from Tensile.LoopModel.theta import Fragment

    def _read(instances, agents):
        k = _mx_kernel(MatrixInstruction=[16, 16, 128, 1, 1, 2, 8, 2, 2], DepthU=256, NumWaves=4,
                       PrefetchGlobalRead=1, PrefetchLocalRead=1, TDMSplitA=2, TDMSplitB=2,
                       UnrollMajorLDSMXSA=1, UnrollMajorLDSMXSB=1, LocalReadVectorWidthMXS=8,
                       LoopOrder="NKM", MIWaveTile=[2, 8])
        th = adapter.params_to_theta(kernel_to_params(
            k, target={"ReadPhi": {"MXSB": instances}, "ReadRho": {"MXSB": agents}}))
        return th.op("MXSB").fragment

    contig = _read(2, 0)
    assert (contig.instances_per_instruction, contig.agent_group_size) == (2, 0)
    ref = geometry.derive_transfer_coverage(
        Fragment(instances_per_instruction=2, agent_group_size=0))
    assert [contig.coverage.carrier_of(i) for i in range(8)] == \
           [ref.carrier_of(i) for i in range(8)]
    assert [contig.coverage.carrier_of(i) for i in range(8)] == [0, 0, 1, 1, 2, 2, 3, 3]

    dist = _read(2, 2)
    assert (dist.instances_per_instruction, dist.agent_group_size) == (2, 2)
    assert [dist.coverage.carrier_of(i) for i in range(8)] == \
           [contig.coverage.carrier_of(i) for i in range(8)]
    assert len({dist.coverage.slot_of(i) for i in range(8)}) == 1, \
        "distributed partners share one register"
    assert len({contig.coverage.slot_of(i) for i in range(8)}) == 2, \
        "contiguous instances use separate registers"


def test_a_fold_THETA_CLAMPED_AWAY_is_not_grouped_by_the_supplied_map():
    """regularity: theta clamps `q` so `q | N` -- "an odd extent under a two-tile coverage clamps to
    `q=1`... never emitting a ragged final group". The supplied `TransferCoverage` is a TARGET fact and
    does NOT know that; its `carrier = t//2` happily cuts an extent-7 axis into {0,1}{2,3}{4,5}{6}.
    """
    from Tensile.Lowering.gir.coverage import plan_coverage
    from Tensile.Lowering.gir.emit_plan import EmitAction
    from Tensile.LoopModel.ir import TransferCoverage, Expr, COVERAGE_VAR
    t = COVERAGE_VAR
    wide = TransferCoverage(carrier=Expr(digits=(((t, 1),), 0, ((1, 2, 0),))),
                      slot=Expr(digits=(((t, 1),), 0, ((1, 1, 2),))))
    acts = [EmitAction("read", {"tc": "MXSB", "tile": i, "tile_flat": i, "k": 0, "region": 0})
            for i in range(7)]                      # an ODD axis: 7 tiles, no legal 2-coverage
    blind = plan_coverage(acts, lambda op: wide, extent_of=lambda op: 7)
    # NON-VACUITY: the raw map really does MERGE tiles the clamp forbids -- tile 1 is told to ride
    # tile 0's instruction (`emit=False`), which is the ragged 2-coverage of an odd axis.
    assert blind.decision(0) is not None and blind.decision(0).emit is True
    assert blind.decision(1) is not None and blind.decision(1).emit is False, \
        "the supplied map merges tiles 0 and 1 -- exactly the fold theta's regularity clamp refuses"

    clamped = plan_coverage(acts, lambda op: wide, extent_of=lambda op: 7,
                           folded_of=lambda op: False)          # theta accepted no coverage here
    assert clamped.violations == [], "theta clamped it away, so there is no group to be incomplete"
    # and, the part that matters for emission: nothing is told to skip its own load.
    assert all(clamped.decision(i) is None for i in range(7)), \
        "a clamped-away operand gets WHOLE, not a permissive Decision -- the leaf keeps its own fold"


def test_the_carrier_group_check_is_ASYMMETRIC_and_the_inference_is_why():
    """, CORRECTED. `plan_coverage`'s fallback `n = 1 + max(tile_flat in THIS group)` makes the
    incomplete-group check asymmetric: a group holding only the HIGH tile is reported, a group
    holding only the LOW tile is not, because n shrinks to fit and `want == len(idxs)`.
    """
    from Tensile.Lowering.gir.coverage import plan_coverage
    from Tensile.LoopModel.ir import TransferCoverage, Expr, COVERAGE_VAR

    t = COVERAGE_VAR
    two = TransferCoverage(carrier=Expr(digits=(((t, 1),), 0, ((1, 2, 0),))),
                     slot=Expr(digits=(((t, 1),), 0, ((1, 1, 2),))))

    class _Act:
        kind = "read"
        def __init__(self, tile):
            self.at = {"tc": "MXSA", "tile": tile, "tile_flat": tile, "k": 0, "k_flat": 0,
                       "region": 0, "token_ids": (4,)}

    def viol(tiles, extent=None):
        return plan_coverage([_Act(x) for x in tiles], lambda op: two,
                            extent_of=(lambda op: extent) if extent else None).violations

    # NON-VACUITY: the complete group is clean and the missing-LOW group fires, so the silence
    # below is a property of the input, not of a check that never speaks.
    assert not viol([0, 1]), "a complete 2-tile group must be clean"
    assert viol([1]), "a group holding only the HIGH tile must be reported"

    # THE ASYMMETRY, pinned: the mirror case is silent under the inference.
    assert not viol([0]), (
        "a group holding only the LOW tile is NOT reported -- n infers to 1 and want matches. "
        "This is the asymmetry this records; do not 'fix' it by supplying the operand tile count, "
        "which breaks every absorbed fold (measured: mxf8 exit 1, 56/444 kernels).")

    # AND THE CONSTRAINT ANY TIGHTENING MUST RESPECT: with the true tile count supplied, the
    # ABSORBED arrangement -- one act standing for two tiles -- is reported, which is wrong.
    assert viol([0], extent=2), (
        "supplying the operand tile count reports the absorbed fold; this is the trap, and it is "
        "why the naive fix was reverted")


def test_the_two_quantum_faults_are_LABELLED_APART():
    """(c). `plan_coverage` reports two different faults and they had one citation between
    them, so a block split read as the generation hazard. That conflation misled four
    separate analyses on.
    """
    from Tensile.Lowering.gir.coverage import plan_coverage
    from Tensile.LoopModel.ir import TransferCoverage, Expr, COVERAGE_VAR

    t = COVERAGE_VAR
    two = TransferCoverage(carrier=Expr(digits=(((t, 1),), 0, ((1, 2, 0),))),
                     slot=Expr(digits=(((t, 1),), 0, ((1, 1, 2),))))

    class _Act:
        kind = "read"
        def __init__(self, tile, buf=0):
            self.at = {"tc": "MXSA", "tile": tile, "tile_flat": tile, "k": 0, "k_flat": 0,
                       "region": 0, "token_ids": (4,), "reg_buf": buf, "group": 0}

    # INCOMPLETE GROUP: high tile alone. Must be labelled as such and must NOT claim.
    inc = plan_coverage([_Act(1)], lambda op: two).violations
    assert inc and "INCOMPLETE CARRIER GROUP" in inc[0], inc
    # It may NAME the straddle in order to disclaim it; what it must not do is wear the label,
    # which is what made a block split read as the hardware hazard.
    assert not inc[0].startswith("operand MXSA: GENERATION STRADDLE"), (
        "the incomplete-group fault must not be labelled a straddle -- borrowing that label is "
        "what made a block split read as the hardware hazard")
    assert "NOT a generation straddle" in inc[0], (
        "it should say explicitly which fault it is not; that disclaimer is the fix")

    # GENERATION STRADDLE: both tiles present but on different register buffers.
    st = plan_coverage([_Act(0, buf=0), _Act(1, buf=1)], lambda op: two).violations
    assert st, "two register generations in one carrier group must be reported"
    assert "GENERATION STRADDLE" in st[0] and "THIS is the hardware hazard" in st[0], st
    # and it must NOT be mistaken for the weaker fault
    assert "INCOMPLETE CARRIER GROUP" not in st[0], st

    # NON-VACUITY: the clean case says nothing, so neither assertion above is always-true.
    assert not plan_coverage([_Act(0), _Act(1)], lambda op: two).violations


def test_region_coverage_agent_relative_is_covered_by_the_agents():
    """an agent-relative region is covered by the AGENT SET, not by one agent's coordinates.
    `check_region_coverage` reads a single agent's plan.  While the region is a property of the
    READ COORDINATE, "every region the copy fills is named by some read" is the right question.
    """
    from Tensile.Lowering.gir import check_region_coverage

    class _P:
        def __init__(self, rel):
            self.meta = {"operand_regions": {"A": 2}, "wave_relative_regions": rel}

    class _Act:
        kind = "read"
        def __init__(self, region):
            self.at = {"tc": "A", "region": region}

    one = [_Act(0)]                      # one agent, naming only its own region

    # NOT agent-relative -> under-coverage is a real fault (this is the non-vacuity guard: the
    # exemption below is only meaningful because the same input DOES fault without it).
    v = check_region_coverage(_P(()), acts=one)
    assert v and "a region nothing reads" in v[0], v

    # Agent-relative -> covered by the agent set.
    assert not check_region_coverage(_P(("A",)), acts=one)

    # OVER-coverage is still a fault even when agent-relative.
    over = check_region_coverage(_P(("A",)), acts=[_Act(0), _Act(1), _Act(2)])
    assert over and "does not have" in over[0], over


def test_tdmsplit_regions_keep_independent_storage_under_wave_relative_addressing():
    """Wave-relative addressing does not merge independently generated LDS region tokens."""
    from Tensile.Lowering.gir.analysis import AnalysisManager
    from Tensile.Lowering.gir.analyses.lds_buffers import LdsBufferIds

    params = {
        **BF16_NT_KMN,
        "MIWaveTile": [4, 4],
        "TDMSplit": [2, 2, 1, 1],
        "NumWaves": 4,
    }

    def probe(wave_regions):
        prog = build_gir(_theta({**params, "TDMSplitWaveRegions": wave_regions}))
        storage = AnalysisManager().get(LdsBufferIds(), prog)
        aliases = {}
        for block in prog.blocks.values():
            for inst in block.body:
                if not isinstance(inst, Move):
                    continue
                for ref in inst.srcs:
                    if ref.tile.space != "shared" or ref.tile.operand != "A":
                        continue
                    split = dict(ref.tile.coord).get("M_split")
                    aliases.setdefault(split, set()).update(
                        buffer.region for buffer in storage.buffers_for(ref))
        return prog.meta["axis_extents"], aliases

    relative_axes, relative = probe([1, 1])
    direct_axes, direct = probe([2, 2])
    assert relative_axes == direct_axes
    assert relative_axes["M_split"] * relative_axes["M_inner"] == 4
    assert relative == {0: {(0,)}, 1: {(1,)}}
    assert direct == {0: {(0,)}, 1: {(1,)}}


def test_agent_relative_read_maps_to_one_exact_frame_region():
    """The logical split coordinate stays distinct while this agent's address selects region 0."""
    from Tensile.Lowering.gir.analyses.lds_buffers import LdsBufferIds

    prog = build_gir(_theta({
        **BF16_NT_KMN,
        "MIWaveTile": [2, 2],
        "MatrixInstruction": [16, 16, 32, 1, 1, 2, 2, 1, 1],
        "DepthU": 64,
        "PrefetchGlobalRead": 2,
        "PrefetchLocalRead": 0,
        "TDMSplit": [2, 1, 1, 1],
        "TDMSplitWaveRegions": [1, 1],
        "VectorWidthA": 2,
        "VectorWidthB": 1,
    }))
    storage = AnalysisManager().get(LdsBufferIds(), prog)
    ref = next(
        ref for block in prog.blocks.values() for inst in block.body
        if isinstance(inst, Move)
        for ref in inst.srcs
        if (ref.tile.space == "shared" and ref.tile.operand == "A"
            and dict(ref.tile.coord).get("M_split") == 1))
    by_id = {idx: buf for buf, idx in storage.buffers.items()}
    assert {buf.region for buf in storage.buffers_for(ref)} == {(1,)}
    assert {by_id[idx].region for idx in storage.frame_at(ref, 0, False)} == {(0,)}


def test_emit_plan_uses_region_displacement_only_for_coordinate_relative_reads():
    """Free-axis split region 1 is not a reduction mode, but still owns an LDS displacement."""
    params = {
        **BF16_NT_KMN,
        "PrefetchGlobalRead": 1,
        "PrefetchLocalRead": 0,
        "TDMSplit": [2, 1, 1, 1],
        "VectorWidthA": 1,
        "VectorWidthB": 1,
    }

    def region_one(wave_regions):
        prog = build_gir(_theta({**params, "TDMSplitWaveRegions": wave_regions}))
        return next(
            action for action in plan_program(prog)["steady"]
                if action.kind == "read" and action.at["tc"] == "A"
                and action.at["region"] == 1)

    coordinate_relative = region_one([2, 1])
    agent_relative = region_one([1, 1])
    assert coordinate_relative.at["address_region"] == 1
    assert agent_relative.at["address_region"] == 0


def test_agent_relative_frame_hazard_covers_the_workgroup_without_weakening_scheduling():
    """Completion sees every agent's region; scheduling keeps one exact representative agent."""
    from Tensile.Lowering.gir.analyses.frame_hazards import (
        FrameHazards, SchedulingFrameHazards, WAR)

    prog = build_gir(_theta({
        **BF16_NT_KMN,
        "MIWaveTile": [2, 2],
        "MatrixInstruction": [16, 16, 32, 1, 1, 2, 2, 1, 1],
        "DepthU": 64,
        "PrefetchGlobalRead": 2,
        "PrefetchLocalRead": 0,
        "TDMSplit": [2, 1, 1, 1],
        "TDMSplitWaveRegions": [1, 1],
        "VectorWidthA": 2,
        "VectorWidthB": 1,
    }))
    body = prog.block("steady").body
    reads = [
        (pos, node) for pos, node in enumerate(body)
        if isinstance(node, Move)
        and any(ref.tile.space == "shared" and ref.tile.operand == "A"
                for ref in node.srcs)]
    copy0 = next(
        (pos, node) for pos, node in enumerate(body)
        if isinstance(node, Move)
        and any(ref.tile.space == "shared" and ref.tile.operand == "A"
                and dict(ref.tile.coord).get("M_split") == 0
                for ref in node.dsts))
    assert copy0[0] > max(pos for pos, _node in reads)
    region1_read = next(
        node for _pos, node in reads
        if any(dict(ref.tile.coord).get("M_split") == 1 for ref in node.srcs))
    assert set(region1_read.token_ids).isdisjoint(copy0[1].token_ids)
    hazards = AnalysisManager().get(SchedulingFrameHazards(), prog)
    assert any(
        hazard.kind == WAR and hazard.gap == 0
        and hazard.producer.inst is region1_read
        and hazard.consumer.inst is copy0[1]
        and hazard.producer.regions == (frozenset({0}),)
        and hazard.consumer.regions == (frozenset({0}),)
        for hazard in hazards)
    assert any(
        hazard.kind == WAR
        and hazard.producer.inst is region1_read
        and hazard.consumer.inst is copy0[1]
        and hazard.producer.regions == (frozenset({0, 1}),)
        and hazard.consumer.regions == (frozenset({0}),)
        for hazard in AnalysisManager().get(FrameHazards(), prog))

    # ST receives the same complete footprint as separate exact ACCESS records, while
    # SchedulingFrameHazards intentionally remains the representative-agent projection above.
    read_action = next(
        action for actions in plan_program(prog).values() for action in actions
        if action.kind == "read" and action.source is region1_read)
    accesses = [access.region for access in build_contract(prog).accesses
                if access.action == read_action.action_id]
    assert set(accesses) == {0, 1}


def test_fp8_multiwave_drain_waits_for_both_physical_split_regions():
    """The GPU-failing VWA2/VWB2 shape must drain region 1 before its first logical-region-0 read."""
    from Tensile.LoopModel.adapter import kernel_to_params
    from Tensile.Lowering.gir.analyses.frame_hazards import FrameHazards, RAW

    kernel = _fp8_kernel(
        NumWaves=4, MIWaveGroup=[2, 2], MIWaveTile=[2, 2],
        MatrixInstruction=[16, 16, 128, 1, 1, 2, 2, 2, 2],
        PrefetchGlobalRead=1, PrefetchLocalRead=0, LoopOrder="KMKNMN",
        TDMFuse=0, TDMSplitA=1, TDMSplitB=1,
        VectorWidthA=2, VectorWidthB=2, TransposeLDS=1,
    )
    prog = build_gir(adapter.params_to_theta(kernel_to_params(kernel)))
    drain = prog.block("drain0")
    first_read = next(
        pos for pos, node in enumerate(drain.body)
        if isinstance(node, Move)
        and any(ref.tile.space == "shared" for ref in node.srcs)
    )
    prior_tensor_waits = [
        node.at["tensorcnt"] for node in drain.body[:first_read]
        if isinstance(node, Mark) and node.kind == "waitcnt"
        and "tensorcnt" in node.at
    ]
    assert 0 in prior_tensor_waits

    hazards = AnalysisManager().get(FrameHazards(), prog)
    producer_regions = {
        next(iter(hazard.producer.regions[0]))
        for hazard in hazards
        if hazard.kind == RAW
        and hazard.producer.block == "prologue"
        and hazard.consumer.block == "drain0"
        and hazard.consumer.pos == first_read
    }
    assert producer_regions == {0, 1}


def test_fp8_multiwave_plr1_drains_agent_relative_refills_before_loop_exit():
    """The two PLR1 GPU failures need the full-region wait on both steady-loop realizations."""
    from Tensile.LoopModel.adapter import kernel_to_params

    kernel = _fp8_kernel(
        NumWaves=4, MIWaveGroup=[2, 2], MIWaveTile=[2, 2],
        MatrixInstruction=[16, 16, 128, 1, 1, 2, 2, 2, 2],
        PrefetchGlobalRead=1, PrefetchLocalRead=1, LoopOrder="KMNMNK",
        TDMFuse=0, TDMSplitA=1, TDMSplitB=0,
        VectorWidthA=2, VectorWidthB=2, TransposeLDS=1,
    )
    prog = build_gir(adapter.params_to_theta(kernel_to_params(kernel)))
    tensor_waits = {
        label: [
            node.at["tensorcnt"] for node in block.body
            if isinstance(node, Mark) and node.kind == "waitcnt"
            and "tensorcnt" in node.at
        ]
        for label, block in prog.blocks.items()
    }
    assert 0 in tensor_waits["prologue"]
    assert 0 in tensor_waits["steady"]


def test_each_tdmsplit_storage_region_has_its_own_generation_phi():
    """Region 0 and region 1 are distinct LDS rings, not coordinates on one shared Gen."""
    from test_loopmodel import _mxf8_kernel

    kernel = dict(_mxf8_kernel("KKMNMN", 1))
    kernel.update(MatrixInstruction=[16, 16, 128, 1, 1, 4, 4, 2, 2],
                  MIWaveTileA=4, MIWaveTileB=4, PrefetchGlobalRead=2,
                  VectorWidthA=4, VectorWidthB=4)
    target = {
        "ReadVectorElems": {"MXSA": 16, "MXSB": 16},
        "ReadPhi": {"MXSA": 4, "MXSB": 4},
        "ReadRho": {"MXSA": 0, "MXSB": 0},
    }
    prog = build_gir(adapter.params_to_theta(adapter.kernel_to_params(kernel, target)))
    facts = prog.meta["generation_regions"]
    assert {(fact["operand"], fact["region"]) for fact in facts.values()} == {
        ("A", 0), ("A", 1), ("B", 0), ("B", 1), ("MXSA", 0), ("MXSB", 0),
    }
    assert len(prog.block("steady").phis) == 6

    for block in prog.blocks.values():
        for inst in block.body:
            if not isinstance(inst, Move):
                continue
            for ref in tuple(inst.srcs) + tuple(inst.dsts):
                if ref.tile.space != "shared" or ref.gen is None or ref.tile.operand not in ("A", "B"):
                    continue
                axis = "M_split" if ref.tile.operand == "A" else "N_split"
                assert facts[ref.gen.id]["region"] == dict(ref.tile.coord)[axis]
    # Every frame retains the exact split-region generation. Completion still covers the whole
    # workgroup, so the second join wait includes the other agents' physical split and is one
    # count stricter than the old representative-agent-only result.
    assert not [node for block in ("prologue", "prefetch_peel")
                for node in prog.block(block).body
                if isinstance(node, Mark) and node.kind == "waitcnt"
                and "tensorcnt" in node.at]
    join = prog.block("prologue_join").body
    tensor_waits = [(pos, node) for pos, node in enumerate(join)
                    if isinstance(node, Mark) and node.kind == "waitcnt"
                    and "tensorcnt" in node.at]
    fences = [pos for pos, node in enumerate(join)
              if isinstance(node, Mark) and node.kind == "fence"]
    assert [node.at["tensorcnt"] for _pos, node in tensor_waits] == [5, 3]
    assert all(pos + 1 in fences for pos, _node in tensor_waits)
    relation_regions = [
        relation["consumer"]["regions"]
        for _pos, node in tensor_waits for relation in node.at["relations"]
    ]
    assert ((0, 1),) in relation_regions
    assert all(regions in ((), ((0,),), ((1,),), ((0, 1),))
               for regions in relation_regions)

    text = render_gir(prog)
    assert ":A/r1 = phi(" in text and ":B/r1 = phi(" in text


def test_the_dump_shows_the_theta_facts_the_blocks_are_derived_from():
    """a dump that omits its premises makes a wrong premise unfalsifiable."""
    from Tensile.Lowering.gir.render import render_gir
    import test_loopmodel as TL
    from Tensile.LoopModel import adapter
    from Tensile.Lowering import build_gir

    prog = build_gir(adapter.params_to_theta(
        adapter.kernel_to_params(TL._mxf8_kernel("KMN", 1))))
    txt = render_gir(prog)

    assert prog.meta.get("wave_relative_regions"), "fixture must have an agent-relative operand"
    assert "wave-relative read reaches 2 regions" in txt
    # TDMSplit remains a real coordinate even when the physical region is wave-relative.
    reads = [l for l in txt.splitlines() if l.strip().startswith("read  A.shared")]
    assert reads, "fixture must emit A reads"
    assert any("M_split" in r for r in reads), (
        "TDMSplit disappeared from the read coordinate instead of remaining a factorized axis:\n"
        + "\n".join(reads[:3]))
    assert "2 storage regions" in txt and "M_split" in txt, txt[:400]
    assert "movement A+B" in txt, "the walk is per MOVEMENT, not per operand"
    assert "fused copies:" in txt and "A+B" in txt

    # The per-instruction facts: the obligation KIND (which decides issue order) and the LDS token
    # ids (which ARE the alias relation the backend orders on).
    assert "deps[" in txt and ("RAW-residency" in txt or "crossing-RAW" in txt), \
        "the ledger edge and its hazard class must be visible; without the class a correctly and " \
        "an incorrectly ordered body render identically"
    assert "dep=" in txt and "tok=" in txt, (
        "BOTH the dependence token the backend orders on and the semantic completion token must be "
        "visible, and spelled apart -- they answer different questions and one prefix for both hid it")

    # NON-VACUITY: an UNSPLIT single-wave kernel must NOT grow these lines, or the assertions above
    # are satisfied by boilerplate rather than by the fixture's actual shape.
    plain = build_gir(adapter.params_to_theta(
        adapter.kernel_to_params(TL._mxf8_kernel("KMN", 0))))
    ptxt = render_gir(plain)
    assert "storage regions" not in ptxt, "an unsplit operand must not claim regions"
    assert "AGENT-relative" not in ptxt or plain.meta.get("wave_relative_regions")


def test_an_mma_source_names_the_operands_own_coordinate_not_the_wmma_grid():
    """A broadcast operand's register Ref must name `pres(operand)`, not the full (K,M,N) coord."""
    import test_loopmodel as TL
    from Tensile.LoopModel import adapter
    from Tensile.Lowering import build_gir
    from Tensile.Lowering.gir.nodes import Mma

    params = adapter.kernel_to_params(TL._mxf8_kernel("KMN", 1))
    theta = adapter.params_to_theta(params)
    prog = build_gir(theta)
    pres = {op.name: set(presence(theta, op)) for op in theta.operands}

    seen_broadcast = False
    a_refs = {}
    for blk in prog.blocks.values():
        for inst in blk.body:
            if not isinstance(inst, Mma):
                continue
            full = {ax for ax, _ in inst.coord}
            for s in inst.srcs:
                axes = {ax for ax, _ in s.tile.coord}
                want = pres.get(s.tile.operand)
                if want is None:
                    continue
                assert axes <= want, (
                    "%s's wmma source names %s, which is outside pres(%s)=%s -- a register "
                    "location no read of %s ever writes"
                    % (s.tile.operand, sorted(axes - want), s.tile.operand,
                       sorted(want), s.tile.operand))
                if want < full:
                    seen_broadcast = True       # this operand really IS broadcast here
            for s in inst.srcs:
                if s.tile.operand == "A":
                    a_refs.setdefault(tuple(s.tile.coord), set()).add(
                        tuple(v for ax, v in inst.coord if ax.startswith("N")))

    assert seen_broadcast, (
        "fixture must contain an operand broadcast over some wmma axis, or the assertion above is "
        "vacuous")
    assert any(len(ns) > 1 for ns in a_refs.values()), (
        "one A source Ref must serve SEVERAL N positions -- that identity is the broadcast, and it "
        "is exactly what the unprojected coord destroyed")


def test_a_fused_movements_members_are_paired_by_index_and_that_is_CHECKED():
    """reading `refs[0]` is justified by a precondition, so the precondition is checked."""
    from Tensile.Lowering.gir.analyses import region_increment as RI
    import test_loopmodel as TL
    from Tensile.LoopModel import adapter
    from Tensile.Lowering import build_gir

    # The shipping mixed-axis fuse lowers cleanly: the members DO agree, on every cell.
    for order in ("KMN", "KMKNMN", "KMNKMN", "KMNMNK"):
        prog = build_gir(adapter.params_to_theta(
            adapter.kernel_to_params(TL._mxf8_kernel(order, 1))))
        marks = [n for blk in prog.blocks.values() for n in blk.body
                 if getattr(n, "kind", None) == "region_increment"]
        assert marks, f"{order}: fixture must emit region_increment Marks"
        for m in marks:
            ax = (m.at or {}).get("axis_steps") or {}
            assert len(ax) <= 1, (
                "L3 walks one axis per step (gir_to_rocisa.py:209); a Mark naming %s cannot be "
                "emitted" % sorted(ax))

    # NON-VACUITY: the check must actually REJECT disagreeing indices, or it is decoration.
    class _T:
        def __init__(self, operand, coord):
            self.operand, self.space, self.coord = operand, "shared", coord
    class _R:
        def __init__(self, operand, coord):
            self.tile = _T(operand, coord)
    meta = {"region_axes": {"A": ("M_split",), "B": ("N_split",)},
            # both members really are cut in two here -- owning the axis is not enough,
            # `region_of` also asks how many regions the operand's own tile occupies.
            "operand_regions": {"A": 2, "B": 2},
            "axis_extents": {"M_split": 2, "N_split": 2}}
    class _P: pass
    p = _P(); p.meta = meta
    assert RI._flat(RI.region_of(p, _R("A", (("M_split", 1),))), p, "A") == 1
    assert RI._flat(RI.region_of(p, _R("B", (("N_split", 0),))), p, "B") == 0, (
        "the two members would be at DIFFERENT indices here -- that is the shape the check in "
        "`_split_copies` refuses, and it must be constructible or the assertion is untestable")


def test_the_fixture_does_NOT_derive_MIWaveTile_from_MatrixInstruction():
    """THE TRAP A LARGE SWEEP FALLS INTO.  `MatrixInstruction[5:7]` are the wave-tile
    counts on a real Solution, but `adapter.kernel_to_params` reads `MIWaveTileA`/`MIWaveTile` and
    the fixture base pins `[2, 2]` -- so overriding ONLY `MatrixInstruction` leaves the shape at
    [2,2] while every log line says [8,7].

    A sweep of 1296 configs "across MIWaveTile (2,2)/(4,4)/(8,7)" runs entirely
    at [2,2] and reported ZERO reproductions of a defect that a real [2,8] kernel hits every time.
    That negative result was then used as evidence the trigger lay elsewhere.  It was evidence of
    nothing.  ALWAYS PASS `MIWaveTile=` EXPLICITLY."""
    from Tensile.LoopModel.adapter import kernel_to_params
    k = _mx_kernel(MatrixInstruction=[16, 16, 128, 1, 1, 8, 7, 2, 2])
    assert k["MIWaveTile"] == [2, 2], (
        "MatrixInstruction does NOT drive MIWaveTile here -- if this ever starts working, the "
        "warning above can go, but until then every sweep must pass MIWaveTile explicitly")
    k2 = _mx_kernel(MatrixInstruction=[16, 16, 128, 1, 1, 8, 7, 2, 2], MIWaveTile=[8, 7])
    ext = {m.name: m.extent for m in
           adapter.params_to_theta(kernel_to_params(k2)).inner_axes()}
    assert ext.get("M_inner") == 8 and ext.get("N_inner") == 7, ext


def test_the_acceptance_shape_has_an_ODD_tile_axis_so_a_2_TILE_FOLD_CANNOT_FACTOR_IT():
    """The ragged case is REAL and it is the SHIPPING shape, not a hypothetical."""
    from Tensile.LoopModel.adapter import kernel_to_params
    ext = {m.name: m.extent for m in adapter.params_to_theta(kernel_to_params(
        _mx_kernel(MatrixInstruction=[16, 16, 128, 1, 1, 8, 7, 2, 2],
                   MIWaveTile=[8, 7]))).inner_axes()}
    assert ext["M_inner"] % 2 == 0, "M factors under a 2-tile fold"
    assert ext["N_inner"] % 2 == 1, (
        "N does NOT -- this is the ragged case, and it is reachable at the acceptance shape")


@pytest.mark.parametrize("wt,expect", [([8, 7], 958), ([2, 8], 468)])
def test_the_register_footprint_of_the_real_wave_tile_shapes_is_PINNED(wt, expect):
    """THE GUARD THE SUITE DID NOT HAVE, and whose absence let a 2.4x vgpr blowup through."""
    from Tensile.LoopModel.adapter import kernel_to_params
    from Tensile.LoopModel import traversal as _geom
    th = adapter.params_to_theta(kernel_to_params(
        _mx_kernel(MatrixInstruction=[16, 16, 128, 1, 1, wt[0], wt[1], 2, 2], MIWaveTile=wt,
                   DepthU=256, NumWaves=4, PrefetchGlobalRead=2, PrefetchLocalRead=1,
                   UnrollMajorLDSMXSA=1, UnrollMajorLDSMXSB=1,
                   LocalReadVectorWidthMXS=8, LoopOrder="KMN")))
    total = sum(_geom.presence_tiles(th, o) * _geom.frag_regs(th, o)
                for o in th.operands)
    assert total == expect, (
        "the register footprint of MIWaveTile %s moved (%d -> %d).  If a mode-structure change is "
        "intended, update the number AND re-check the emitted vgpr count on a real build -- the "
        "last time this moved silently the acceptance kernel stopped generating." % (wt, expect, total))


# ============================================================================== PrefetchGL2
def _gl2_prog(depth, params=None):
    from Tensile.Lowering.gir.analyses import Gl2PrefetchRegions
    from Tensile.Lowering.loopir_to_gir import build_gir
    return build_gir(_theta(params or BF16_NT_KMN),
                     params={Gl2PrefetchRegions.PARAM: depth})


def _gl2_acts(prog, phase):
    return [i for i, a in enumerate(plan_block(prog, phase)) if a.kind == "gl2_prefetch"]


def test_gl2_is_INERT_when_the_feature_is_off():
    """The default must be byte-identical to a program built with no params at all."""
    from Tensile.Lowering.loopir_to_gir import build_gir
    plain = build_gir(_theta(BF16_NT_KMN))
    off = _gl2_prog(0)
    assert list(plain.blocks) == list(off.blocks)
    for phase in plain.blocks:
        a = [(x.kind, sorted(x.at)) for x in plan_block(plain, phase)]
        b = [(x.kind, sorted(x.at)) for x in plan_block(off, phase)]
        assert a == b, "PrefetchGL2=0 changed block %r" % phase
        assert not _gl2_acts(off, phase)


@pytest.mark.parametrize("depth", [1, 2])
def test_gl2_is_ONE_prefetch_per_STEADY_chunk_and_none_in_prologue_or_drain(depth):
    """RATE and PLACE, the two things that were not obvious.
    RATE -- one per steady block, i.e. one per reduction chunk, the SAME rate as `gr_increment`.
    """
    prog = _gl2_prog(depth)
    steady = [b for b in prog.blocks if b.startswith("steady")]
    assert steady, "fixture must have a steady block for this to mean anything"
    for b in steady:
        assert len(_gl2_acts(prog, b)) == 1, "expected exactly one gl2_prefetch in %r" % b
    for b in prog.blocks:
        if b not in steady:
            assert not _gl2_acts(prog, b), "%r must not prefetch (prologue is scaffold's, drain " \
                                           "is clamped)" % b


def test_gl2_lands_MIDPOINT_which_is_a_BANDWIDTH_choice_not_a_bound():
    """The Mark is strictly inside the body, not at either end."""
    from Tensile.Lowering.gir.analyses import Gl2PrefetchRegions
    from Tensile.Lowering.gir.passes import CollectPendingMarksPass, PlacementPass

    # MEASURE THE PASS'S OWN DECISION, on the body it actually bisected.
    prog = lower_to_gir(_theta(BF16_NT_KMN))
    prog.params[Gl2PrefetchRegions.PARAM] = 1
    am = AnalysisManager()
    CollectPendingMarksPass().run(prog, am)
    PlacementPass().run(prog, am)
    (pm,) = [p for p in prog.pending if p.mark.kind == "gl2_prefetch"]
    body = prog.block("steady").body
    assert pm.anchor == ("steady", body[len(body) // 2]), \
        "gl2_prefetch anchored at %r, not the midpoint of the %d-node steady body" \
        % (pm.anchor, len(body))

    # ...and STRICTLY INTERIOR in the finished emitted stream, which is the property the bandwidth
    # argument actually rests on: real work on both sides of it to overlap with.  That claim
    # survives both the expansion and the later insertions, so it is safe to state end to end.
    done = _gl2_prog(1)
    acts = plan_block(done, "steady")
    (act_at,) = _gl2_acts(done, "steady")
    assert 0 < act_at < len(acts) - 1, \
        "gl2_prefetch at %d of %d acts -- collapsed onto a block end, so nothing overlaps it" \
        % (act_at, len(acts))


def test_gl2_mark_names_NO_operand_and_NO_movement():
    """The absence IS the fact.  Every other Mark subject-names something it mutates (`unit` for a
    Phi movement, `operand` for an operand); GL2 runs off its own address VGPRs and its own
    increment SGPRs, never touching a TDM descriptor.  A `unit` key here would be a claim that some
    movement's pointer moved, and the region/token analyses would be entitled to believe it."""
    prog = _gl2_prog(2)
    acts = plan_block(prog, "steady")
    (at,) = _gl2_acts(prog, "steady")
    payload = acts[at].at
    assert set(payload) == {"depth", "block"}, payload
    assert payload["depth"] == 2 and payload["block"] == "steady"


def test_gl2_preset_is_VISIBLE_in_the_dump_and_marked_as_not_theta():
    """a fact that steers emission and is invisible in the dump is a fact nobody can falsify.
    `Program.params` is a SECOND input channel beside `meta`, so the dump must both show it and say
    it is not theta -- otherwise a reader hunts for a GL2 operand that does not exist."""
    text = render_gir(_gl2_prog(2))
    line = [l for l in text.splitlines() if "PrefetchGL2" in l]
    assert len(line) == 1, text[:400]
    assert "NOT theta" in line[0] and "MIDPOINT" in line[0]
    assert "PrefetchGL2" not in render_gir(_gl2_prog(0)), "off must print nothing"


# ==================================================== Await.scope is load-bearing
def test_a_block_scoped_obligation_on_a_NON_cross_agent_operand_FAILS_the_build():
    """NON-VACUITY for `verify.check_block_scope_covered`."""
    from Tensile.Lowering.gir.verify import check_block_scope_covered
    from Tensile.Lowering.gir.analysis import AnalysisManager
    from Tensile.Lowering.gir.nodes import Move

    prog = build_gir(_theta(BF16_NT_KMN))              # single wave -> no cross-agent edges
    am = AnalysisManager()
    check_block_scope_covered(prog, am)               # clean as built

    # find any Move carrying a dep, and re-scope ONE of them to "block"
    victim = next(m for b in prog.blocks.values() for m in b.body
                  if isinstance(m, Move) and getattr(m, "deps", ()))
    dep, counter, kind, _scope = victim.deps[0]
    victim.deps = ((dep, counter, kind, "block"),) + tuple(victim.deps[1:])

    with pytest.raises(RuntimeError, match="G-SCOPE"):
        check_block_scope_covered(prog, AnalysisManager())


def test_the_dep_tuple_carries_scope_so_it_survives_the_lowering():
    """`_deps_of` is the ONE place LoopIR awaits enter GIR, so a field dropped there is invisible
    everywhere below.  A dropped `scope` is invisible -- three fields went in, four come
    out -- and this pins the width and the vocabulary so a future edit cannot silently narrow it."""
    prog = build_gir(_theta(BF16_NT_KMN))
    deps = [d for b in prog.blocks.values() for i in b.body
            for d in (getattr(i, "deps", ()) or ())]
    assert deps, "fixture carries no deps -- the test would be vacuous"
    assert all(len(d) == 4 for d in deps), "every dep must be (dep, counter, kind, scope)"
    assert {d[3] for d in deps} <= {"wave", "block"}, {d[3] for d in deps}


# ------------------------------------------------------- VgprPartition (register groups)
def _part_theta(partsA=1, partsB=1, policy=None, **kw):
    """A theta whose A (and optionally B) register fragment is PARTITIONED, built by replacing the
    fragment directly rather than through a parameter.
    """
    import dataclasses as _dc
    from Tensile.LoopModel.theta import group_labels
    p = {"MIWaveTile": [4, 4], "PrefetchLocalRead": 1, "PrefetchGlobalRead": 2}
    p.update(kw)
    th = adapter.params_to_theta(p)
    pol = policy or (lambda i: "overlap" if i == 0 else "inplace")
    for nm, n in (("A", partsA), ("B", partsB)):
        if n <= 1:
            continue
        op = th.op(nm)
        labels = group_labels(n)
        op.fragment = _dc.replace(op.fragment, parts=n, labels=labels,
                                  group_policy={g: pol(i) for i, g in enumerate(labels)},
                                  ring_depths=None)
    return th


def test_mxfp8_split_groups_keep_the_region_based_register_names():
    """K0/K1 keep the region-based names while region-local PLR primes four reads."""
    import test_loopmodel as TL
    from Tensile.Lowering.gir import check_plan

    th = TL._mxfp8_theta()
    prog = build_gir(th)
    assert check_plan(prog) == []

    names = {0: [], 1: []}
    seen = {0: set(), 1: set()}
    for act in plan_block(prog, "steady"):
        if act.kind != "wmma":
            continue
        source = next(src for src in act.at["srcs"] if src[0] == "A")
        _op, group, slot, _unit, logical_tile = source
        k = act.at["u"]
        region = logical_tile // 4
        key = (region, group)
        if key in seen[k]:
            continue
        seen[k].add(key)
        names[k].append((group, slot * 2 + region))

    assert names[0] == [(0, 0), (1, 0), (0, 1), (1, 1)]
    assert names[1] == [(0, 2), (1, 0), (0, 3), (1, 1)]
    assert prog.meta["register_layout"]["A"]["total"] == 192

    prologue = [act for name in prog.blocks for act in plan_block(prog, name)
                if name.startswith("prologue") and act.kind == "read" and act.at["tc"] == "A"]
    assert len(prologue) == 4


def test_register_partition_reaches_compact_unit_indexed_names():
    """MNK A groups split each K unit; both groups recur across all M units."""
    prog = build_gir(_part_theta(partsA=2, LoopOrder="MNK"))
    used, filled, owned = set(), set(), {}
    for block in prog.blocks.values():
        for inst in block.body:
            refs = (inst.srcs if type(inst).__name__ == "Mma" else
                    inst.dsts if type(inst).__name__ == "Move" else ())
            for ref in refs:
                if ref.tile.operand != "A" or ref.tile.space != "register":
                    continue
                name = (ref.group, ref.slot, ref.unit_index)
                (used if type(inst).__name__ == "Mma" else filled).add(name)
                owned.setdefault(ref.group, set()).add(dict(ref.tile.coord).get("K_inner"))
    assert used == filled
    assert owned == {0: {0}, 1: {1}}
    assert len(used) == 3
    assert prog.meta["register_layout"]["A"]["total"] == 48


@pytest.mark.parametrize("order", ["KMN", "KNM", "MNK", "NMK", "MKN", "NKM"])
@pytest.mark.parametrize("kw", [dict(partsA=2), dict(partsB=2),
                                dict(partsA=2, partsB=2),
                                dict(partsA=2, MIWaveTile=[8, 4]),
                                dict(partsA=2, PrefetchLocalRead=0),
                                dict(partsA=2, policy=lambda _i: "unroll")])
def test_a_partitioned_plan_passes_its_own_semantic_check(kw, order):
    """`check_plan` is the gate codegen runs (KernelWriter ~3822).  It reported 0 violations before
    while the partition never reached GIR; the moment the groups became real it found
    19, and this pins that they are all resolved rather than re-hidden.
    """
    from Tensile.Lowering.gir import check_plan
    p = dict(kw)
    p["LoopOrder"] = order
    assert check_plan(build_gir(_part_theta(**p))) == []


# `test_the_register_partition_rejects_what_it_cannot_honour` and
# `test_the_three_strategies_are_three_distinct_policies` lived here.


def test_a_hoisted_refill_may_not_split_its_consumers():
    """The defect shape every COORDINATE-level check is blind to, and that the existing
    ordering test cannot see because it keys on the `inplace-WAR` await.
    """
    from Tensile.Lowering.gir import check_refill_splits_consumers

    class _Act:
        def __init__(self, kind, at):
            self.kind, self.at = kind, at

    def use(tile):
        return _Act("wmma", {"idx0": tile, "idx1": 0, "u": 0,
                             "srcs": (("A", None, 0, tile, tile),
                                      ("B", None, 0, 0, 0))})

    def refill(tile, adv=2):
        return _Act("read", {"tc": "A", "tile": tile, "tile_flat": tile, "k": 0, "k_flat": 0,
                             "reg_buf": 0, "group": None, "unit_index": tile, "advance": adv})

    # use / refill / use  -- the refill splits its consumer set: MUST fire.
    v = check_refill_splits_consumers(None, acts=[use(0), refill(0), use(0)])
    assert v and "REFILL-SPLITS-CONSUMERS" in v[0], v

    # refill / use / use  -- the refill is the producer for both: MUST NOT fire.
    assert not check_refill_splits_consumers(None, acts=[refill(0), use(0), use(0)])

    # an UNHOISTED read (advance == 0) is in place by construction and is never the defect.
    assert not check_refill_splits_consumers(None, acts=[use(0), refill(0, adv=0), use(0)])

    # a refill of a DIFFERENT register does not touch this consumer set.
    assert not check_refill_splits_consumers(None, acts=[use(0), refill(1), use(0)])


def test_every_reachable_config_is_free_of_refill_splitting():
    """The product-layer half: after the `_register_timeline` space fix, NO reachable
    configuration may split a consumer set.
    """
    from Tensile.Lowering.gir import check_refill_splits_consumers, check_plan
    from Tensile.LoopModel.adapter import kernel_to_params
    bad = {}
    for wg in ((1, 1), (2, 2)):
        for order in ("KMN", "KNM", "MKN", "NKM", "MNK", "NMK"):
            # ALL NINE split combinations, including the MIXED MT+DU ones.  A sweep over five of
            # them hid a real regression: `MNK 1/2` and `NMK 2/1` broke while the test stayed green.
            for sa in (0, 1, 2):
              for sb in (0, 1, 2):
                  for plr in (0, 1):
                      k = {**BF16_NT_KMN, "LoopOrder": order, "MIWaveGroup": list(wg),
                           "MatrixInstruction": [16, 16, 32, 1, 1, 2, 2, wg[0], wg[1]],
                           "PrefetchGlobalRead": 1, "PrefetchLocalRead": plr,
                           "MIWaveTileA": 2, "MIWaveTileB": 2, "TDMInst": 3,
                           "enableTDMA": True, "enableTDMB": True,
                           "TDMSplitA": sa, "TDMSplitB": sb,
                           "VectorWidthA": 1, "VectorWidthB": 1}
                      prog = build_gir(_theta(kernel_to_params(k)))
                      v = check_refill_splits_consumers(prog) + check_plan(prog)
                      if v:
                          bad["%s %d/%d plr%d w%d" % (order, sa, sb, plr,
                                                      wg[0] * wg[1])] = v[0]
    assert not bad, "refill splits a consumer set in %d config(s):\n  %s" % (
        len(bad), "\n  ".join("%s: %s" % kv for kv in sorted(bad.items())))


class _OneByte:
    def numBytes(self):
        return 1


def _mx_split_kernel(order, fuse, split_a, split_b):
    """The mxf8 shape whose Phi groups can MIX region counts: a scale is never split, so fusing one
    with a split data operand gives region 0 and region 1 different member sets."""
    return {"MatrixInstruction": [16, 16, 128, 1, 1, 2, 2, 2, 2], "DepthU": 256,
            "MIWaveTileA": 2, "MIWaveTileB": 2, "MatrixInstK": 128,
            "ProblemType": {"DataType": _OneByte(), "MXBlockA": 32, "MXBlockB": 32},
            "PrefetchGlobalRead": 2, "PrefetchLocalRead": 1, "NumWaves": 4,
            "enableTDMA": 1, "enableTDMB": 1, "NumLdsBlk": 2, "LoopOrder": order,
            "TDMSplitA": split_a, "TDMSplitB": split_b, "TDMFuse": fuse,
            "VectorWidthA": 2, "VectorWidthB": 2, "LocalReadVectorWidthMXS": 8,
            "UnrollMajorLDSMXSA": 1, "UnrollMajorLDSMXSB": 1}


def _walk_oracle(prog, groups, regions):
    """Replay the region walk against the descriptor derived from THETA's Phi groups.

    Independent of `unit_regions`, which is what `walk_violations` keys on -- when the walk was
    keyed on the movement instead of the descriptor both the analysis and its verifier agreed with
    each other and were wrong together.
    """
    from Tensile.Lowering.gir.nodes import copy_unit
    desc = lambda members: next((g for g in groups if set(members) & set(g)), tuple(members))
    out = []
    for label, blk in prog.blocks.items():
        at = {}
        for node in blk.body:
            if isinstance(node, Mark) and node.kind == "region_increment":
                key = desc(node.at["unit"])
                at[key] = at.get(key, 0) + int(node.at["steps"])
            elif isinstance(node, Move):
                members, refs = copy_unit(node)
                if members is None:
                    continue
                walker = max(refs, key=lambda r: regions.get(r.tile.operand, 1))
                if regions.get(walker.tile.operand, 1) <= 1:
                    continue
                coord = dict(walker.tile.coord or ())
                axes = (prog.meta.get("region_axes", {}) or {}).get(walker.tile.operand) or ()
                want = 0
                for axis in axes:
                    want = want * 2 + int(coord.get(axis) or 0)
                got = at.get(desc(members), 0)
                if got != want:
                    out.append("%s: copy of %s wants region %d, descriptor on %d"
                               % (label, members, want, got))
        out += ["%s: %s left %d regions from the chunk base" % (label, k, v)
                for k, v in sorted(at.items(), key=str) if v]
    return out


def _swap_oracle(prog, groups):
    """Between two copies on ONE descriptor, the buffer swaps must total the generation delta the
    two copies demand.  A mixed split draws a swap for `('A','B')` and another for `('A',)` -- two
    toggles of one XOR per chunk, which cancel, so the double buffer stops alternating."""
    from Tensile.Lowering.gir.nodes import copy_unit
    desc = lambda members: next((g for g in groups if set(members) & set(g)), tuple(members))
    out = []
    for label, blk in prog.blocks.items():
        swaps, last = {}, {}
        for node in blk.body:
            if isinstance(node, Mark) and node.kind == "swap" and node.at.get("hop") == "copy":
                key = desc(node.at["unit"])
                swaps[key] = swaps.get(key, 0) + int(node.at.get("steps", 1))
                continue
            if not isinstance(node, Move):
                continue
            members, refs = copy_unit(node)
            if members is None:
                continue
            ref = refs[0]
            gen = getattr(ref, "gen", None)
            if gen is None or getattr(ref, "gdelta", None) is None:
                continue
            key, ring = desc(members), max(1, gen.ring)
            want = (ref.gdelta - (blk.gen_rel or 0)) % ring
            if key in last:
                moved = swaps.get(key, 0) % ring
                if moved != (want - last[key]) % ring:
                    out.append("%s: %s moved %d buffer(s) between copies that want %d"
                               % (label, key, moved, (want - last[key]) % ring))
            last[key], swaps[key] = want, 0
    return out



@pytest.mark.unit
@pytest.mark.parametrize("order", ("KMN", "KNM", "MKN", "NKM", "MNK", "NMK"))
def test_mx_split_region_walk_reaches_every_region(order):
    """Every split copy loads the region its own descriptor has walked to, for every Phi grouping.

    The failing shape is an ASYMMETRIC split: region 0 moves the whole group and region 1 moves
    only the split member, so the two copies carry different member tuples for ONE descriptor.
    """
    from Tensile.LoopModel.adapter import kernel_to_params
    bad = {}
    for fuse in (0, 1, 2, 3):
        for split_a in (0, 1):
            for split_b in (0, 1):
                params = kernel_to_params(_mx_split_kernel(order, fuse, split_a, split_b))
                th = _theta(params)
                prog = build_gir(th, mainloop=emit_mainloop(th))
                inputs = [o.name for o in th.operands if o.role != "output"]
                groups = [tuple(g) for g in th.fused_copy_groups if all(m in inputs for m in g)]
                regions = {o.name: max(1, int(o.split)) for o in th.operands}
                v = _walk_oracle(prog, groups, regions) + _swap_oracle(prog, groups)
                if v:
                    bad["f%d %d/%d" % (fuse, split_a, split_b)] = v[0]
    assert not bad, "region walk broken in %d config(s):\n  %s" % (
        len(bad), "\n  ".join("%s: %s" % kv for kv in sorted(bad.items())))


def _split_pipeline():
    """The backend pipeline with the prefetch-guard split enabled."""
    from Tensile.Lowering.gir.passes import pipeline as default_pipeline
    from Tensile.Lowering.gir.passes.scaffold_shape import ScaffoldShapePass
    from Tensile.Lowering.gir.passes.fold_short_path import FoldShortPathPass
    from Tensile.Lowering.gir.passes.early_exit import EarlyExitPass
    from Tensile.Lowering.gir.passes.scaffold_map import ScaffoldMapPass
    from Tensile.Lowering.gir.passes.split_prefetch_guard import SplitPrefetchGuardPass
    ps = default_pipeline()
    ps[0] = ScaffoldShapePass([FoldShortPathPass(), EarlyExitPass(),
                               SplitPrefetchGuardPass(), ScaffoldMapPass()])
    return ps


def _pointer_state(prog, labels):
    """Net (chunk advance, copy-hop swap steps) per unit over an ordered block list."""
    chunk, swap = {}, {}
    for lab in labels:
        for n in prog.blocks[lab].body:
            if not isinstance(n, Mark):
                continue
            if n.kind == "gr_increment":
                u = tuple(n.at["unit"])
                chunk[u] = chunk.get(u, 0) + int(n.at.get("chunks", 1))
            elif n.kind == "swap" and n.at.get("hop") == "copy":
                u = tuple(n.at.get("unit") or ())
                swap[u] = swap.get(u, 0) + int(n.at.get("steps", 1))
    return chunk, swap


@pytest.mark.unit
@pytest.mark.parametrize("pgr", (1, 2))
def test_prefetch_guard_splits_only_a_peeled_generation(pgr):
    """PGR>=2 puts the last peeled generation in its own block; PGR=1 has nothing to guard."""
    from Tensile.LoopModel import adapter
    prog = build_gir(adapter.params_to_theta(dict(BF16_NT_KMN, PrefetchGlobalRead=pgr)),
                     pipeline=_split_pipeline())
    guard = prog.meta.get("prefetch_guard")
    if pgr < 2:
        assert guard is None, "nothing to guard at PGR=1, but %r was split" % (guard,)
        return
    assert guard, "PGR=2 peels two generations but none was guarded"
    blk = prog.blocks[guard["block"]]
    assert all(isinstance(i, Move)
               or (isinstance(i, Mark) and i.kind == "waitcnt") for i in blk.body), (
        "the guarded arm may hold only copies and their edge-local waits, got %s"
        % [i.kind for i in blk.body if isinstance(i, Mark) and i.kind != "waitcnt"])
    from Tensile.Lowering.gir.nodes import CondGoto
    term = prog.blocks[prog.entry].term
    assert isinstance(term, CondGoto) and term.f_target == guard["block"], term
    assert term.t_target == guard["skip"]
    assert prog.blocks[guard["skip"]].preds == (prog.entry,)
    assert prog.blocks[guard["skip"]].succs == (guard["join"],)
    assert prog.blocks[guard["join"]].preds == (guard["skip"], guard["block"])


@pytest.mark.unit
def test_prefetch_guard_arms_reach_the_join_alike():
    """Both arms leave the descriptor on the same chunk and the same buffer.

    An advance or a swap that lands inside the guarded arm is skipped on the short trip, so the
    join would inherit two different pointers -- the defect the split exists to make visible.
    """
    from Tensile.LoopModel import adapter
    prog = build_gir(adapter.params_to_theta(dict(BF16_NT_KMN, PrefetchGlobalRead=2)),
                     pipeline=_split_pipeline())
    guard = prog.meta["prefetch_guard"]
    skip = _pointer_state(prog, [prog.entry, guard["skip"]])
    through = _pointer_state(prog, [prog.entry, guard["block"]])
    assert skip == through, (
        "the guard's arms diverge before the join: skip=%r through=%r" % (skip, through))


def test_prefetch_guard_compensates_the_shallow_path_before_the_deep_join_wait():
    from test_loopmodel import _mxf8_kernel

    from Tensile.Lowering.gir.analyses import CounterFlow
    from Tensile.Lowering.gir.passes import pipeline
    from Tensile.Lowering.gir.passes.wait_counts import WaitCntPass

    kernel = dict(_mxf8_kernel("KKMNMN", 1))
    kernel.update(MatrixInstruction=[16, 16, 128, 1, 1, 4, 4, 2, 2],
                  MIWaveTileA=4, MIWaveTileB=4,
                  PrefetchGlobalRead=2, PrefetchLocalRead=1,
                  VectorWidthA=4, VectorWidthB=4)
    target = {
        "ReadVectorElems": {"MXSA": 16, "MXSB": 16},
        "ReadPhi": {"MXSA": 4, "MXSB": 4},
        "ReadRho": {"MXSA": 0, "MXSB": 0},
    }
    prog = build_gir(
        adapter.params_to_theta(adapter.kernel_to_params(kernel, target)),
        pipeline=[item for item in pipeline() if not isinstance(item, WaitCntPass)])
    guard = prog.meta["prefetch_guard"]
    waits = {(site.block, site.pos, site.counter): site.n
             for site in AnalysisManager().get(CounterFlow(), prog)}
    assert waits[(guard["skip"], 0, "tensorcnt")] == 0
    assert sorted(n for (block, _pos, counter), n in waits.items()
                  if block == guard["join"] and counter == "tensorcnt") == [3, 5]
    WaitCntPass().run(prog, AnalysisManager())
    assert [node.at["tensorcnt"] for node in prog.block(guard["skip"]).body
            if isinstance(node, Mark) and node.kind == "waitcnt"
            and "tensorcnt" in node.at] == [0]
    assert 5 in [node.at["tensorcnt"] for node in prog.block(guard["join"]).body
                 if isinstance(node, Mark) and node.kind == "waitcnt"
                 and "tensorcnt" in node.at]
    assert 3 in [node.at["tensorcnt"] for node in prog.block(guard["join"]).body
                 if isinstance(node, Mark) and node.kind == "waitcnt"
                 and "tensorcnt" in node.at]
    join = prog.block(guard["join"]).body
    tensor_wait_positions = [
        pos for pos, node in enumerate(join)
        if isinstance(node, Mark) and node.kind == "waitcnt" and "tensorcnt" in node.at
    ]
    assert all(pos + 1 < len(join)
               and isinstance(join[pos + 1], Mark) and join[pos + 1].kind == "fence"
               for pos in tensor_wait_positions)
    assert {operand for node in join
            if isinstance(node, Mark) and node.kind == "fence"
            for operand in node.at["buffers"]} == {"A", "B", "MXSA", "MXSB"}


def _racing_edges(prog, dep=None):
    """(reached, named) cross-wave RAW frame instances and their fence relations."""
    from Tensile.Lowering.gir.analyses.frame_hazards import FrameHazards, RAW
    hz = AnalysisManager().get(FrameHazards(), prog)
    fm = AnalysisManager().get(FrameMap(), prog)
    relations = [relation for block in prog.blocks.values() for node in block.body
                 if isinstance(node, Mark) and node.kind == "fence"
                 for relation in node.at.get("relations", ())]
    instances = [(fp, fc, hazard) for fp, fc, hazard in hz.instances()
                 if hazard.cross_agent and hazard.kind == RAW]
    named = sum(any(
        relation["kind"] == hazard.kind
        and relation["producer"].get("identity") == id(hazard.producer.inst)
        and relation["consumer"].get("identity") == id(hazard.consumer.inst)
        and relation["producer"]["frame"] == fm.render(fp)
        and relation["consumer"]["frame"] == fm.render(fc)
        for relation in relations) for fp, fc, hazard in instances)
    return len(instances), named


@pytest.mark.parametrize("pgr,plr", [(1, 0), (1, 1), (2, 0), (2, 1)])
def test_the_token_reuse_check_inspects_real_refills(pgr, plr):
    """G-TOKEN-REUSE is not vacuous: every config has racing refills, and all are named."""
    prog, _ts = _tokens_of(dict(BF16_NT_KMN_FUSED_XAGENT,
                                PrefetchGlobalRead=pgr, PrefetchLocalRead=plr))
    reached, named = _racing_edges(prog)
    assert reached, f"PGR{pgr}/PLR{plr}: the check reaches no edge -- it proves nothing"
    assert named == reached, f"PGR{pgr}/PLR{plr}: {reached - named} edge(s) unnamed"


def test_a_fence_that_drops_its_frame_relations_is_caught():
    """A barrier at the right position is insufficient if it names no frame hazard."""
    from dataclasses import replace
    prog, _ts = _tokens_of(dict(BF16_NT_KMN_FUSED_XAGENT,
                                PrefetchGlobalRead=2, PrefetchLocalRead=1))
    assert verify_gir(prog)
    reached, named = _racing_edges(prog)
    assert reached and named == reached, "fixture is not fully fenced -- the mutation proves nothing"
    for blk in prog.blocks.values():
        blk.body = [replace(n, at=dict(n.at, relations=()))
                    if isinstance(n, Mark) and n.kind == "fence" else n
                    for n in blk.body]
    with pytest.raises(RuntimeError, match="G-FRAME-FENCE"):
        verify_gir(prog)


#: Keys a pass writes only when its situation arises, so a single program need not carry them.
_CONDITIONAL_META = frozenset({
    "short_loop", "prefetch_guard", "prefetch_gl2", "guards", "steps", "blocks", "reason",
    "verdict", "model_only", "kinds_dropped", "marks_dropped", "agent_varying_symbols",
    "generation_regions",
})


def _meta_keys_read_from_source():
    """Every literal meta key the tree reads, as {key: [site, ...]}."""
    import pathlib
    import re
    root = pathlib.Path(__file__).resolve().parents[2]      # Tensile/
    patterns = (re.compile(r'meta(?:\s*or\s*\{\})?\s*\)?\s*\.get\(\s*["\']([A-Za-z_0-9]+)["\']'),
                re.compile(r'\.meta\[\s*["\']([A-Za-z_0-9]+)["\']\s*\]'))
    found = {}
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in str(path) or path.name.startswith("test_"):
            continue
        text = path.read_text()
        for pattern in patterns:
            for match in pattern.finditer(text):
                site = "%s:%d" % (path.name, text[:match.start()].count("\n") + 1)
                found.setdefault(match.group(1), set()).add(site)
    return found


def test_every_meta_key_a_consumer_READS_is_a_key_the_lowering_WRITES():
    """A renamed `Program.meta` key is silent: `.get()` returns None and the consumer does nothing.

    That is how the tail rewind vanished -- `meta['M']` became `meta['peel_depth']`, four readers
    were left behind, and the only symptom was 12 missing instructions in the emitted assembly.
    """
    written = set(_CONDITIONAL_META)
    for params in (BF16_NT_KMN, BF16_NT_KMN_PLR0,
                   {**BF16_NT_KMN, "TDMSplit": [2, 2, 1, 1]}):
        written |= set(build_gir(_theta(params)).meta)

    stale = {key: sorted(sites) for key, sites in _meta_keys_read_from_source().items()
             if key not in written}
    assert not stale, (
        "these meta keys are read but never written -- a rename left the reader behind:\n"
        + "\n".join("  %-24s %s" % (key, ", ".join(sites)) for key, sites in sorted(stale.items())))

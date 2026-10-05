# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""ULM parity sweep — the DYNAMIC half of `LoopModel/docs/PARITY_LEDGER.md`."""

import itertools

import pytest

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from gir_fixtures import mi_inputs
from Tensile.LoopModel import adapter
from Tensile.LoopModel.adapter import kernel_to_params
from Tensile.Lowering.loopir_to_gir import build_gir
from Tensile.Lowering.gir.verify_dataflow import check_plan
from Tensile.Lowering.gir.analyses.region_increment import walk_violations


class _Bytes:
    """A DataType stand-in answering only what the bridge asks."""
    def __init__(self, n): self.n = n
    def numBytes(self): return self.n


_BASE = {
    "MatrixInstruction": [16, 16, 32, 1, 1, 2, 2, 1, 1],
    "MIWaveTile": [2, 2], "MIWaveGroup": [1, 1],
    "MIBlock": [16, 16, 32, 1, 1, 1, 1, 1],
    "DepthU": 64, "WavefrontSize": 32,
    "PrefetchGlobalRead": 2, "PrefetchLocalRead": 1, "HalfPLR": 0,
    "ProblemType": {"DataType": _Bytes(2), "DestDataType": _Bytes(2),
                    "ComputeDataType": _Bytes(4), "TransposeA": False, "TransposeB": True},
    "LoopOrder": "KMN", "NumLdsBlk": 2, "1LDSBuffer": -1,
    "DirectToVgprA": False, "DirectToVgprB": False,
    "GlobalReadVectorWidthA": 8, "GlobalReadVectorWidthB": 8,
    "LocalReadVectorWidth": 8, "LocalReadVectorWidthA": 8, "LocalReadVectorWidthB": 8,
    "InnerUnroll": 1, "VgprPartition": 1,
    "enableTDMA": True, "enableTDMB": True, "NumWaves": 1, "TDMFuse": 0,
}

ORDERS3 = ["KMN", "KNM", "MKN", "MNK", "NKM", "NMK"]
ORDERS6 = ["KMKNMN", "KMNKMN", "KMNMNK", "NKMNKM", "NKNKMM"]

# PGR is deliberately capped at the REACHABLE range.  This is not a narrowed test: PGR>=3 cannot be
# generated under UseLoopModel at all, and `test_pgr3_is_unreachable_under_ulm` is what holds that
# claim honest rather than this cap.
REACHABLE_PGR = [1, 2]


def _mxf8(blk=None):
    o = {"MatrixInstruction": [16, 16, 128, 1, 1, 2, 2, 1, 1], "DepthU": 256,
         "LocalReadVectorWidth": 16, "LocalReadVectorWidthA": 16, "LocalReadVectorWidthB": 16,
         "ProblemType": {"DataType": _Bytes(1), "DestDataType": _Bytes(1),
                         "ComputeDataType": _Bytes(4), "TransposeA": True, "TransposeB": False}}
    if blk:
        o["ProblemType"] = {**o["ProblemType"], "MXBlockA": blk, "MXBlockB": blk, "Sparse": 0}
    return o


DTYPES = [({}, "bf16"), (_mxf8(), "fp8"), (_mxf8(32), "mxf8_32"), (_mxf8(16), "mxf8_16")]


def _kernel(dover, mi, pgr, plr, order, sa, sb, vw):
    nw = mi[7] * mi[8]
    out = {**_BASE, **dover, "MatrixInstruction": list(mi),
            "MIWaveTile": [mi[5], mi[6]], "MIWaveGroup": [mi[7], mi[8]],
            "MIWaveTileA": mi[5], "MIWaveTileB": mi[6],
            "PrefetchGlobalRead": pgr, "PrefetchLocalRead": plr, "LoopOrder": order,
            "TDMSplitA": sa, "TDMSplitB": sb, "VectorWidthA": vw, "VectorWidthB": vw,
            "NumLdsBlk": max(2, pgr), "NumWaves": nw, "TDMFuse": 0}
    # after the overrides: a cell that changes MatrixInstruction must not inherit a stale count
    return mi_inputs(out)


def _mi(dlabel, wt, wg):
    k = [16, 16, 128, 1, 1] if dlabel != "bf16" else [16, 16, 32, 1, 1]
    return tuple(k + [wt[0], wt[1], wg[0], wg[1]])


def _shape_configs():
    for dover, dlabel in DTYPES:
        for wt, wg, pgr, plr, sa, sb, vw in itertools.product(
                [(2, 2), (4, 4), (2, 8)], [(1, 1), (2, 2)], REACHABLE_PGR, [0, 1],
                [0, 1], [0, 1], [1, 2]):
            yield (dlabel, wt, wg, pgr, plr, sa, sb, vw), \
                _kernel(dover, _mi(dlabel, wt, wg), pgr, plr, "KMN", sa, sb, vw)


def _order_configs():
    for dover, dlabel in DTYPES:
        mi = _mi(dlabel, (2, 2), (2, 2))
        for order, pgr, sa, sb in itertools.product(
                ORDERS3 + ORDERS6, REACHABLE_PGR, [0, 1], [0, 1]):
            if order in ORDERS6 and not (sa or sb):
                continue                      # 6-axis words exist only under a split
            yield (dlabel, order, pgr, sa, sb), _kernel(dover, mi, pgr, 1, order, sa, sb, 1)


@pytest.mark.parametrize("gen,name", [(_shape_configs, "shape"), (_order_configs, "order")])
def test_no_reachable_config_is_refused_by_the_decode_and_lower_path(gen, name):
    """Every REACHABLE config must survive bridge -> theta -> GIR -> check_plan."""
    refused = []
    swept = 0
    for key, k in gen():
        swept += 1
        try:
            prog = build_gir(adapter.params_to_theta(kernel_to_params(k)))
            bad = check_plan(prog) or walk_violations(prog)
            if bad:
                refused.append((key, "SEMANTIC: %s" % (bad[0],)))
        except Exception as e:                       # noqa: BLE001 - any raise is the finding
            refused.append((key, "%s: %s" % (type(e).__name__, str(e).split("\n")[0][:120])))
    assert swept > 100, f"{name} sweep collapsed to {swept} configs — the generator is broken"
    assert not refused, (
        f"{len(refused)}/{swept} reachable configs refused by the ULM decode/lower path:\n  "
        + "\n  ".join(f"{k} -> {m}" for k, m in refused[:10]))


def test_pgr3_decode_is_structurally_supported():
    """The decoder itself carries PGR3; Solution policy may still reject a concrete kernel."""
    k = {**_BASE, "PrefetchGlobalRead": 3, "NumLdsBlk": 3}
    assert build_gir(adapter.params_to_theta(kernel_to_params(k)))


def test_pgr3_decode_plan_is_clean():
    k = {**_BASE, "PrefetchGlobalRead": 3, "NumLdsBlk": 3}
    assert not check_plan(build_gir(adapter.params_to_theta(kernel_to_params(k))))


# ---------------------------------------------------------------------------
# loop_copies: GIR unrolls the steady body into a chain, and the emitter walks
# it. A count the emitter is not told about would be built and dropped (#410).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("copies", [1, 2, 3])
def test_steady_chain_is_built_and_readable(copies):
    from Tensile.Lowering import steady_chain
    k = _kernel(_mxf8(), _mi("fp8", (2, 2), (2, 2)), 2, 1, "KMN", 0, 0, 1)
    k["SuppressNoLoadLoop"] = copies > 1
    prog = build_gir(adapter.params_to_theta(kernel_to_params(k)), loop_copies=copies)
    chain = steady_chain(prog)
    assert chain == ["steady"] + ["steady%d" % i for i in range(1, copies)]
    # Every physical copy decrements/tests the counter; only the last continuation is a back-edge.
    assert all(prog.blocks[c].body for c in chain)
    terms = [type(prog.blocks[c].term).__name__ for c in chain]
    assert terms == (["LoopBack"] if copies == 1 else ["LoopCopy"] * copies)
    assert not check_plan(prog)


@pytest.mark.parametrize("vg,va,copies", [(-1, -1, 1), (2, 4, 1), (2, 2, 1), (2, 3, 3)])
def test_loop_copies_are_read_off_the_rings_not_off_a_parameter(vg, va, copies):
    """A ring the trip closes needs one block; `VG=2, VA=3` -- 1.5 buffers -- needs three."""
    from Tensile.Components.LoopModel.Program import loopModelLoopCopies
    from Tensile.LoopModel.schedule import BufferDepths
    k = _kernel(_mxf8(), _mi("fp8", (4, 4), (1, 1)), 1, 0, "KMN", 0, 0, 1)
    k["ClusterLocalRead"] = 0
    for side in ("A", "B"):
        k["ClusterLocalRead" + side] = 0
        k["VgprGroup" + side] = vg if side == "A" else -1
        k["VgprAlloc" + side] = va if side == "A" else -1
    theta = adapter.params_to_theta(kernel_to_params(k))
    assert loopModelLoopCopies(theta, BufferDepths(theta, include_shared=False)) == copies


def test_three_copy_frame_contract_has_backedge_incoming_and_guard_requirements():
    from Tensile.LoopModel.schedule import BufferDepths
    from Tensile.LoopModel import traversal as geometry
    from Tensile.Lowering.gir.emit_plan import plan_program
    from Tensile.Lowering.gir.frame_contract import build_contract

    k = _half_plr_kernel("NKM", 1)
    k["SuppressNoLoadLoop"] = True
    theta = adapter.params_to_theta(kernel_to_params(k))
    copies = geometry.ring_trip_copies(
        theta, BufferDepths(theta, include_shared=False))
    prog = build_gir(theta, loop_copies=copies)
    assert [prog.blocks[label].gen_rel for label in ("steady", "steady1", "steady2")] == [0, 1, 2]

    plans = plan_program(prog)
    anchors = {label: actions[0].action_id for label, actions in plans.items() if actions}
    contract = build_contract(prog)
    backedge = [edge for edge in contract.edges
                if edge.relative
                and edge.src == anchors["steady2"] and edge.dst == anchors["steady"]]
    assert backedge
    assert all((contract.generations[edge.gen].ring,
                contract.generations[edge.gen].entry,
                contract.generations[edge.gen].advance) == (2, 0, 1)
               for edge in backedge)
    # PGR2's split prefetch guard must remain correlated with the loop entrance/backedge.
    assert contract.requires
    assert all(need.gen in contract.generations for need in contract.requires)


def test_the_covering_invariant_runs_on_an_unrolled_loop():
    """Per-copy exits preserve the remainder: an n-copy shell still covers exactly T-M."""
    from Tensile.Lowering.gir.analysis import AnalysisManager
    from Tensile.Lowering.gir.analyses.loop_shape import (
        Linear, LoopShape, reduction_coverage_violations)

    k = _kernel(_mxf8(), _mi("fp8", (2, 2), (2, 2)), 2, 1, "KMN", 0, 0, 1)
    k["SuppressNoLoadLoop"] = True
    th = adapter.params_to_theta(kernel_to_params(k))
    for copies in (1, 2, 3):
        prog = build_gir(th, loop_copies=copies)
        loops = AnalysisManager().get(LoopShape(), prog)
        steady = [lp for lp in loops if lp.header == "steady"]
        assert steady, f"no steady loop at loop_copies={copies}"
        lp = steady[0]
        assert lp.per_trip == 1
        assert lp.trips == Linear(1, 0)
        assert lp.covers == lp.trips
        assert not reduction_coverage_violations(prog, loops), copies


# ---------------------------------------------------------------------------
# HalfPLR shape: VG=2 halves one buffer and VA=3 allocates three halves.
# ---------------------------------------------------------------------------
def _half_plr_kernel(order, half_plr):
    k = _kernel(_mxf8(), _mi("fp8", (2, 2), (2, 2)), 2, 2, order, 0, 0, 1)
    k.update(HalfPLR=0, ClusterLocalRead=0,
             ClusterLocalReadA=0, ClusterLocalReadB=0)
    for side, bit in (("A", 1), ("B", 2)):
        stated = bool(half_plr & bit)
        k["VgprGroup" + side] = 2 if stated else -1
        k["VgprAlloc" + side] = 3 if stated else -1
    return k


@pytest.mark.parametrize("half_plr, halved", [(1, "A"), (2, "B"), (3, "AB")])
def test_halfplr_costs_one_and_a_half_buffers_on_the_named_operand(half_plr, halved):
    """THREE halves where PLR1 holds two whole buffers -- 1.5 blocks, so 3/4 of the pair.

    And only for the operand the bitmask names: HalfPLR=1 leaves B alone.
    """
    from Tensile.LoopModel.schedule import BufferDepths
    from Tensile.LoopModel import traversal as g

    def regs(k):
        th = adapter.params_to_theta(kernel_to_params(k))
        d = BufferDepths(th, include_shared=False)
        return th, {n: g.operand_emitted_regs(th, th.op(n), d) for n in ("A", "B")}

    _, plain = regs(_half_plr_kernel("KMN", 0))
    th, half = regs(_half_plr_kernel("KMN", half_plr))
    for name in ("A", "B"):
        if name in halved:
            assert half[name] * 4 == plain[name] * 3, (name, half, plain)   # 1.5 of 2 blocks
            assert th.op(name).fragment.vgpr_group == 2
            assert th.op(name).fragment.ring_depths == {"g0": 3}
        else:
            assert half[name] == plain[name], (name, half, plain)
            assert th.op(name).fragment.vgpr_group == 1


@pytest.mark.parametrize("order", ["NKM"])
@pytest.mark.parametrize("half_plr", [1, 2, 3])
def test_halfplr_is_clean_or_refused_never_broken(order, half_plr):
    """The YAML's canonical NKM cell lowers cleanly at a three-copy chain."""
    from Tensile.Lowering import steady_chain
    k = _half_plr_kernel(order, half_plr)
    k["SuppressNoLoadLoop"] = True
    prog = build_gir(adapter.params_to_theta(kernel_to_params(k)), loop_copies=3)
    assert len(steady_chain(prog)) == 3
    assert not check_plan(prog)


# ---------------------------------------------------------------------------
# SuppressNoLoadLoop: the scaffold folds the ramp-out into the loop and clamps the
# overrun loads, so the model says the same -- no drain, and the loop owns all T.
# ---------------------------------------------------------------------------
def _drain_shape(kernel, loop_copies=1):
    """(drain block labels, steady `covers`, meta peel/drain) for one kernel."""
    from Tensile.Lowering.gir import analysis as _an
    from Tensile.Lowering.gir.analyses.loop_shape import LoopShape, reduction_coverage_violations
    prog = build_gir(adapter.params_to_theta(kernel_to_params(kernel)), loop_copies=loop_copies)
    loops = _an.AnalysisManager().get(LoopShape(), prog)
    steady = next(lp for lp in loops if lp.header == "steady")
    return (sorted(l for l in prog.blocks if l.startswith("drain")),
            steady, prog.meta["peel_depth"], prog.meta["drain_steps"],
            reduction_coverage_violations(prog, loops), check_plan(prog))


@pytest.mark.parametrize("order", ORDERS3)
def test_default_keeps_the_drain(order):
    """CONTROL ARM. Without the flag nothing moves: M drain blocks, loop covers `T - M`."""
    k = _half_plr_kernel(order, 0)
    drains, steady, peel, steps, cov, plan = _drain_shape(k)
    assert len(drains) == peel == steps == 2
    assert (steady.covers.coeff, steady.covers.const) == (1, -2)
    assert steady.exit_to == "drain0"
    assert not cov and not plan


@pytest.mark.parametrize("order", ORDERS3)
def test_suppressed_no_load_loop_has_no_drain_and_the_loop_owns_every_chunk(order):
    """The prologue keeps its lead; the ramp-out goes; the loop's count grows by exactly M."""
    k = _half_plr_kernel(order, 0)
    k["SuppressNoLoadLoop"] = True
    drains, steady, peel, steps, cov, plan = _drain_shape(k)
    assert drains == []
    assert peel == 2 and steps == 0            # the lead survives, the ramp-out does not
    assert (steady.covers.coeff, steady.covers.const) == (1, 0)
    assert steady.exit_to == "end"
    assert not cov and not plan


@pytest.mark.parametrize("half_plr", [1, 2, 3])
def test_halfplr_lowers_drain_free_over_its_three_copies(half_plr):
    """VG2/VA3 uses per-copy exits, so a partial final unroll still covers all T chunks."""
    from Tensile.Lowering import steady_chain
    k = _half_plr_kernel("NKM", half_plr)
    k["SuppressNoLoadLoop"] = True
    prog = build_gir(adapter.params_to_theta(kernel_to_params(k)), loop_copies=3)
    assert len(steady_chain(prog)) == 3
    drains, steady, _peel, steps, cov, plan = _drain_shape(k, loop_copies=3)
    assert drains == [] and steps == 0
    assert (steady.covers.coeff, steady.covers.const) == (1, 0)
    assert steady.trips.div == 1               # copy exits preserve T % 3
    assert steady.per_trip == 1
    assert not cov and not plan


def test_a_drain_emitted_under_suppression_is_a_structural_error():
    """The P1 arm that keeps the two halves of the fix honest: the IR must not build both."""
    from Tensile.LoopModel.checks import _check_root_shape
    from Tensile.LoopModel.ir import Loop, Peel

    ir = [Peel(kind="prologue", axis="iter", k=2, body=[]),
          Loop(axis="iter", trip=None, outer=True, bodies=[[]]),
          Peel(kind="drain", axis="iter", k=2, body=[])]
    errs = []
    _check_root_shape(errs, ir, no_drain=True)
    assert errs and "suppresses the no-load loop" in errs[0]


# ---------------------------------------------------------------------------
# A region that LEADS the ring cannot also separate generations.  `ring_block_width`
# gives each region a share of the depth; when that share falls to one slot the
# enumerator keeps no digit, and a group shallower than its sibling then aliases its
# own generations under the deeper one's read-ahead.
# ---------------------------------------------------------------------------
def _split_half_plr_theta(order, half_plr, sa, sb, wt=(2, 2), clr=0, pgr=2):
    k = _kernel(_mxf8(), _mi("fp8", wt, (1, 1)), pgr, 2, order, sa, sb, 1)
    k.update(HalfPLR=0, ClusterLocalRead=clr,
             ClusterLocalReadA=clr, ClusterLocalReadB=clr)
    for side, bit in (("A", 1), ("B", 2)):
        stated = bool(half_plr & bit)
        k["VgprGroup" + side] = 2 if stated else -1
        k["VgprAlloc" + side] = 3 if stated else -1
    k["SuppressNoLoadLoop"] = True
    return adapter.params_to_theta(kernel_to_params(k))


def test_the_shallow_half_keeps_a_ring_digit_when_the_region_leads():
    """NON-VACUITY for the test below: the two groups really do get different widths.

    g1 is one slot per region, so without the override its share is 1 and the enumerator emits
    nothing -- the ring axis would vanish from the slot while g0 kept it.
    """
    from Tensile.LoopModel import traversal as g
    from Tensile.LoopModel.schedule import BufferDepths
    from Tensile.LoopModel.adapter import loop_order_of

    th = _split_half_plr_theta(loop_order_of(6, 1), 1, sa=1, sb=0)
    op, depths = th.op("A"), BufferDepths(th, include_shared=False)
    assert op.fragment.vgpr_group == 2
    assert op.fragment.ring_depths == {"g0": 3}
    assert g.regions_lead_the_ring(th, op, "g0")
    slot = g.ring_slot(th, op, "g0", g.group_ring_depth(th, op, "g0", depths))
    assert slot is not None and slot.digits[2]


@pytest.mark.parametrize("wio, sb", [(1, 0), (2, 0), (5, 0), (5, 1), (6, 0), (6, 1)])
@pytest.mark.parametrize("half_plr", [2, 3])
def test_shallow_half_buffer_repeats_reads_over_the_broadcast_pass(wio, sb, half_plr):
    """VA3 rereads B for each M pass instead of retaining four live values in three slots."""
    from Tensile.LoopModel.adapter import loop_order_of
    from Tensile.LoopModel.schedule import BufferDepths
    from Tensile.LoopModel import traversal as geometry
    from Tensile.Lowering.gir.emit_plan import plan_program

    theta = _split_half_plr_theta(loop_order_of(wio, 1), half_plr, 1, sb)
    copies = geometry.ring_trip_copies(theta, BufferDepths(theta, include_shared=False))
    prog = build_gir(theta, loop_copies=copies)
    assert not check_plan(prog)

    issues = {
        tuple(action.at.get("issue") or ())
        for actions in plan_program(prog).values()
        for action in actions
        if action.kind == "read" and action.at["tc"] == "B"
    }
    assert issues == {(("M_split", 0),), (("M_split", 1),)}


@pytest.mark.parametrize("wio", [1, 2])
@pytest.mark.parametrize("half_plr", [1, 2, 3])
def test_tdmsplit_consuming_vg_still_builds_the_three_copy_ring(wio, half_plr):
    """A split can spend VG's axis digit without erasing the stated VG/VA shape.

    These are the twelve PGR1/PGR2 YAML kernels that previously built one steady
    block, then returned to the wrong half-buffer on hardware.
    """
    from Tensile.LoopModel.adapter import loop_order_of
    from Tensile.LoopModel.schedule import BufferDepths
    from Tensile.LoopModel import traversal as geometry
    from Tensile.Lowering import steady_chain
    from Tensile.Lowering.gir.emit_plan import plan_program
    from Tensile.Lowering.gir.frame_contract import build_contract

    theta = _split_half_plr_theta(loop_order_of(wio, 1), half_plr, 1, 1)
    stated = [op for op in theta.operands
              if op.fragment is not None and geometry.vgpr_group(theta, op) > 1]
    assert stated and all(geometry.vgpr_subdivided(theta, op) for op in stated)
    assert any(geometry.vgpr_group_axis(theta, op) is None for op in stated)
    assert all(geometry.vgpr_ring_carries(theta, op, op.fragment.groups()[0])
               for op in stated)

    copies = geometry.ring_trip_copies(
        theta, BufferDepths(theta, include_shared=False))
    prog = build_gir(theta, loop_copies=copies)
    assert copies == 3
    assert steady_chain(prog) == ["steady", "steady1", "steady2"]
    assert not check_plan(prog)

    plans = plan_program(prog)
    anchors = {label: actions[0].action_id
               for label, actions in plans.items() if actions}
    contract = build_contract(prog)
    assert any(edge.relative
               and edge.src == anchors["steady2"]
               and edge.dst == anchors["steady"]
               for edge in contract.edges)
    assert contract.requires


def test_a_loopmodel_refusal_names_the_solution_exactly_once():
    """A bare refusal costs a bisect to place, and saying the name twice is noise."""
    from Tensile.Components.LoopModel.Theta import loopModelTheta, namingSolution

    class _Writer:
        class states:
            kernelName = "Cijk_Alik_Bljk_F8SS_NAMED"

    k = _kernel(_mxf8(), _mi("fp8", (4, 4), (1, 1)), 1, 2, "KMN", 0, 0, 1)
    k["ClusterLocalRead"] = 0
    for side in ("A", "B"):
        k["ClusterLocalRead" + side] = 0
        k["VgprGroup" + side] = 2 if side == "A" else -1
        k["VgprAlloc" + side] = -1
    with pytest.raises((ValueError, RuntimeError)) as refusal:
        loopModelTheta(_Writer(), k)
    assert str(refusal.value).count("solution:") == 1
    assert "Cijk_Alik_Bljk_F8SS_NAMED" in str(refusal.value)

    writer = _Writer()
    with pytest.raises(ValueError) as nested:
        with namingSolution(writer):
            with namingSolution(writer):
                raise ValueError("inner failure")
    assert str(nested.value).count("solution:") == 1


def test_gir_semantic_checker_failure_is_always_a_value_error(monkeypatch):
    """An explicit VG/VA shape does not turn a checker defect into a tuning rejection."""
    import importlib
    from types import SimpleNamespace

    program = importlib.import_module("Tensile.Components.LoopModel.Program")
    writer = SimpleNamespace(states=SimpleNamespace(kernelName="Cijk_CHECKER_VALUE_ERROR"))
    kernel = {
        "PrefetchGL2": 0,
        "VgprGroupA": 2,
        "VgprAllocA": 3,
        "VgprGroupB": -1,
        "VgprAllocB": -1,
    }
    fake_prog = SimpleNamespace()

    monkeypatch.setattr(program, "loopModelTheta", lambda *_args: (object(), object()))
    monkeypatch.setattr(program, "emit_mainloop", lambda *_args: object())
    monkeypatch.setattr(program, "loopModelLoopCopies", lambda *_args: 1)
    monkeypatch.setattr(program, "gir_pipeline", lambda: ())
    monkeypatch.setattr(program, "build_gir", lambda *_args, **_kwargs: fake_prog)
    monkeypatch.setattr(program, "check_plan", lambda _prog: ["deliberate checker finding"])

    with pytest.raises(ValueError, match="semantic check"):
        program.loopModelGirProgram(writer, kernel)

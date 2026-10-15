# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""ULM parity sweep — the DYNAMIC half of `LoopModel/docs/PARITY_LEDGER.md`."""

import inspect
import itertools

import pytest

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from gir_fixtures import calculate_registers, mi_inputs
from Tensile.LoopModel import adapter
from Tensile.LoopModel.adapter import kernel_to_params as _kernel_to_params
from Tensile.LoopModel.decoder import decode_theta
from Tensile.Lowering.loopir_to_gir import lower_to_gir
from Tensile.Lowering.gir.passes import pipeline as gir_pipeline
from Tensile.Lowering.gir.passes import run_pipeline
from Tensile.Lowering.gir.verify_dataflow import check_plan
from Tensile.Lowering.gir.analyses.region_increment import walk_violations


def kernel_to_params(kernel):
    """Run the same register calculation as Solution before adapting the kernel."""

    return _kernel_to_params(calculate_registers(kernel))


def build_gir(theta, loop_copies=None):
    decoded = decode_theta(theta)
    program = lower_to_gir(decoded.program, decoded.profile)
    run_pipeline(program, passes=gir_pipeline())
    if loop_copies is not None:
        from Tensile.Lowering import steady_chain
        assert len(steady_chain(program)) == max(1, int(loop_copies))
    return program


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


def test_local_read_pair_auto_side_inherits_scalar():
    from Tensile.SolutionStructs.Solution import localReadPair

    state = {"PrefetchLocalRead": 3, "PrefetchLocalReadA": -1,
             "PrefetchLocalReadB": 5}
    assert localReadPair(state, "PrefetchLocalRead") == (True, 3, 5)


def test_k_axis_read_ahead_cap_counts_vg_buffers():
    from Tensile.Components.LoopModel import Registers
    from Tensile.Components.LoopModel.Order import loopOrder

    def cap(kernel, side):
        kernel = dict(kernel)
        kernel["PrefetchLocalRead"] = 999
        kernel["PrefetchLocalReadA"] = -1
        kernel["PrefetchLocalReadB"] = -1
        kernel["UseLoopModel"] = True
        Registers.calculate(kernel)
        return kernel["PrefetchLocalRead" + side]

    k = _kernel(_mxf8(), _mi("fp8", (2, 2), (1, 1)), 2, 1, "KMN", 0, 0, 1)
    k.update(LoopIters=2, WmmaInnerOrder=1, WmmaOuterOrder=3, VgprGroupA=1)
    k["LoopOrder"] = loopOrder(1, 3, 0, 0)
    vg1 = cap(k, "A")
    k["VgprGroupA"] = 2
    assert cap(k, "A") == 2 * vg1

    k = _kernel(_mxf8(), _mi("fp8", (8, 8), (2, 2)),
                2, 8, loopOrder(6, 2, 1, 1), 1, 1, 4)
    k.update(LoopIters=2, WmmaInnerOrder=6, WmmaOuterOrder=2, VgprGroupB=2)
    assert cap(k, "B") == 8


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


@pytest.mark.parametrize(
    "order,pgr,split_a,split_b",
    (
        ("NMK", 1, 0, 0),
        ("MKN", 2, 1, 0),
        ("MKN", 2, 0, 2),
        ("KMNMKN", 2, 0, 2),
    ),
)
def test_folded_mxsb_refill_follows_its_last_broadcast_consumer(
    order, pgr, split_a, split_b
):
    """One folded scale read must not refill its slot between two M consumers."""
    k = _kernel(_mxf8(32), _mi("fp8", (2, 2), (2, 2)),
                pgr, 1, order, split_a, split_b, 2)
    k.update(
        ClusterLocalReadA=1,
        ClusterLocalReadB=1,
        VgprGroupA=1,
        VgprGroupB=1,
        VgprAllocA=2,
        VgprAllocB=2,
        ISA=[12, 5, 0],
        MXScaleFormat="InMemorySwizzle",
        LocalReadVectorWidthMXS=16,
        MatrixInstM=16,
        MatrixInstN=16,
    )
    target = {
        "ReadVectorElems": {"MXSA": 16, "MXSB": 16},
        "ReadPhi": {"MXSA": 2, "MXSB": 2},
        "ReadRho": {"MXSA": 0, "MXSB": 0},
    }
    theta = adapter.params_to_theta(
        adapter.kernel_to_params(calculate_registers(k), target),
        validate_register_depths=False,
    )

    assert check_plan(build_gir(theta)) == []


@pytest.mark.parametrize(
    "order,split_a,split_b,side",
    (
        ("KMN", 1, 0, "A"),
        ("KMN", 0, 1, "B"),
        ("MKN", 2, 0, "A"),
        ("MKN", 0, 1, "B"),
    ),
)
def test_clustered_split_region_keeps_the_original_physical_unit(
    order, split_a, split_b, side
):
    """A region becomes a ring digit; the next enumerator replaces it inside the CLR unit."""
    from Tensile.LoopModel import traversal as geometry
    from Tensile.LoopModel.schedule import BufferDepths

    k = _kernel(_mxf8(32), _mi("fp8", (2, 2), (2, 2)),
                1, 0, order, split_a, split_b, 2)
    k.update(ClusterLocalRead=1, ClusterLocalReadA=1, ClusterLocalReadB=1,
             SuppressNoLoadLoop=True)
    params = kernel_to_params(k)
    theta = adapter.params_to_theta(params)
    operand = theta.op(side)
    depths = BufferDepths(theta, include_shared=False)

    assert params["VgprAlloc" + side] == 2
    assert geometry.group_original_prefetch_unit_positions(theta, operand) == 2
    assert geometry.group_register_positions(theta, operand) == 2
    assert geometry.group_ring_depth(theta, operand, "g0", depths) == 2
    assert check_plan(build_gir(theta)) == []


def test_verifier_rejects_a_clustered_region_that_shrinks_the_physical_unit():
    k = _kernel(_mxf8(32), _mi("fp8", (2, 2), (2, 2)),
                1, 0, "KMN", 0, 1, 2)
    k.update(ClusterLocalRead=1, ClusterLocalReadA=1, ClusterLocalReadB=1,
             SuppressNoLoadLoop=True)
    program = build_gir(adapter.params_to_theta(kernel_to_params(k)))
    program.meta["register_unit_positions"]["B"] = 1

    assert any("CLUSTERED-REGION-SHRINKS-UNIT" in finding
               for finding in check_plan(program))


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
def test_loop_copy_count_is_not_a_lowering_input():
    assert "loop_copies" not in inspect.signature(lower_to_gir).parameters


@pytest.mark.parametrize("vg,va,copies", [(-1, -1, 1), (2, 4, 1), (2, 2, 1), (2, 3, 3)])
def test_loop_copies_are_read_off_the_rings_not_off_a_parameter(vg, va, copies):
    """A ring the trip closes needs one block; `VG=2, VA=3` -- 1.5 buffers -- needs three."""
    from Tensile.LoopModel.decoder import decode_theta
    from Tensile.LoopModel.loopir import Let, walk
    k = _kernel(_mxf8(), _mi("fp8", (4, 4), (1, 1)), 1, 0, "KMN", 0, 0, 1)
    k["ClusterLocalRead"] = 0
    for side in ("A", "B"):
        k["ClusterLocalRead" + side] = 0
        k["VgprGroup" + side] = vg if side == "A" else -1
        k["VgprAlloc" + side] = va if side == "A" else -1
    theta = adapter.params_to_theta(kernel_to_params(k))
    program = decode_theta(theta).program
    steps = tuple(
        node for node in walk(program)
        if isinstance(node, Let) and node.axis.name == "_steady_copy"
    )
    assert (len(steps) or 1) == copies


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
    # Every unrolled-copy transition advances the buffer frame by one.
    for source, destination in (("steady", "steady1"), ("steady1", "steady2")):
        incoming = [edge for edge in contract.edges
                    if edge.relative and edge.src == anchors[source]
                    and edge.dst == anchors[destination]]
        assert incoming
        assert all(edge.value == 1 for edge in incoming)

    # A separate control generation makes the copy chain one-way. Relative incoming alone is not
    # enough: without these singleton requirements ST can revisit a later copy through copy 0.
    copy_gens = {gen_id for gen_id, generation in contract.generations.items()
                 if generation.ring == 4}
    assert len(copy_gens) == 1
    copy_gen = next(iter(copy_gens))
    expected = {
        (anchors["prologue_join"], anchors["steady"]): ((0,), 0),
        (anchors["steady"], anchors["steady1"]): ((0, 1), 2),
        (anchors["steady1"], anchors["steady2"]): ((2,), 3),
        (anchors["steady2"], anchors["steady"]): ((3,), 1),
    }
    for key, (required, incoming_value) in expected.items():
        source, destination = key
        assert any(need.gen == copy_gen and need.src == source and need.dst == destination
                   and need.values == required for need in contract.requires)
        assert any(edge.gen == copy_gen and edge.src == source and edge.dst == destination
                   and not edge.relative and edge.value == incoming_value
                   for edge in contract.edges)

    # The shadow/skip arm owns its own full drain. Only the peeled prefetch arm enters the
    # modeled steady pipeline, otherwise its one-time rank-0 RAW forces tensorcnt(0) at the
    # recurring loop header.
    prefetch_gens = {gen_id for gen_id, generation in contract.generations.items()
                     if generation.ring == 3}
    assert len(prefetch_gens) == 1
    prefetch_gen = next(iter(prefetch_gens))
    assert any(need.gen == prefetch_gen
               and need.src == anchors["prologue_join"]
               and need.dst == anchors["steady"]
               and need.values == (2,)
               for need in contract.requires)


@pytest.mark.parametrize("vg,va,plr", [(1, 6, 4), (2, 12, 8)])
def test_deep_b_ring_keeps_full_plr_and_a_complete_three_copy_frame(vg, va, plr):
    """A 1.5-ring refills only slots whose old logical value has finished.

    The VA6 and VA12 spellings are the same shape at different VG granularity.  Both need three
    steady copies, no drain join, and a frame contract covering every copy edge.
    """
    from Tensile.LoopModel import traversal as geometry
    from Tensile.Components.LoopModel.Order import loopOrder
    from Tensile.LoopModel.schedule import BufferDepths, Schedule
    from Tensile.Lowering import steady_chain
    from Tensile.Lowering.gir.emit_plan import plan_program
    from Tensile.Lowering.gir.frame_contract import build_contract

    k = _kernel(_mxf8(), _mi("fp8", (8, 8), (2, 2)),
                2, plr, loopOrder(6, 2, 1, 1), 1, 1, 4)
    k.update(LoopIters=2, WmmaInnerOrder=6, WmmaOuterOrder=2,
             PrefetchLocalReadA=1, PrefetchLocalReadB=plr,
             ClusterLocalRead=0, ClusterLocalReadA=1, ClusterLocalReadB=0,
             VgprGroupB=vg, VgprAllocB=va, SuppressNoLoadLoop=True)
    theta = adapter.params_to_theta(kernel_to_params(k))
    depths = BufferDepths(theta, include_shared=False)
    operand = theta.op("B")
    plans = Schedule(theta, depths)
    assert geometry.requested_read_ahead(theta, operand, depths) == plr
    assert all(plans.plan(operand, group).prefetch_steps == plr
               for group in operand.fragment.groups())

    copies = geometry.ring_trip_copies(theta, depths)
    assert copies == 3
    prog = build_gir(theta, loop_copies=copies)
    assert not check_plan(prog)
    chain = steady_chain(prog)
    assert chain == ["steady", "steady1", "steady2"]
    assert not any(label.startswith("drain") for label in prog.blocks)
    assert [(prog.blocks[label].gen_rel, prog.blocks[label].chunk_base)
            for label in chain] == [(0, 0), (1, 1), (2, 2)]

    emitted = plan_program(prog)
    anchors = {label: emitted[label][0].action_id for label in chain}
    contract = build_contract(prog)
    backedge = [edge for edge in contract.edges
                if edge.relative
                and edge.src == anchors["steady2"] and edge.dst == anchors["steady"]]
    incoming = [edge for edge in contract.edges
                if not edge.relative and edge.dst == anchors["steady"]
                and edge.gen in {item.gen for item in backedge}]
    assert backedge and len(incoming) == len(backedge)
    required_edges = {(need.src, need.dst) for need in contract.requires}
    assert {(anchors["steady"], anchors["steady1"]),
            (anchors["steady1"], anchors["steady2"]),
            (anchors["steady2"], anchors["steady"])} <= required_edges


def test_the_covering_invariant_uses_the_derived_steady_period():
    """The calculated steady period covers exactly T."""
    from Tensile.Lowering.gir.analysis import AnalysisManager
    from Tensile.Lowering.gir.analyses.loop_shape import (
        Linear, LoopShape, reduction_coverage_violations)

    k = _kernel(_mxf8(), _mi("fp8", (2, 2), (2, 2)), 2, 1, "KMN", 0, 0, 1)
    k["SuppressNoLoadLoop"] = True
    th = adapter.params_to_theta(kernel_to_params(k))
    prog = build_gir(th)
    loops = AnalysisManager().get(LoopShape(), prog)
    steady = [lp for lp in loops if lp.header == "steady"]
    assert steady
    lp = steady[0]
    assert lp.per_trip == 1
    assert lp.trips == Linear(1, 0)
    assert lp.covers == lp.trips
    assert not reduction_coverage_violations(prog, loops)


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
    """VG2/VA3 is three halves: 1.5 PU, or 3/4 of a VG1/VA2 allocation.

    And only for the operand the bitmask names: HalfPLR=1 leaves B alone.
    """
    from Tensile.LoopModel.schedule import BufferDepths
    from Tensile.LoopModel import traversal as g

    def regs(k):
        th = adapter.params_to_theta(kernel_to_params(k))
        d = BufferDepths(th, include_shared=False)
        return th, {n: g.operand_emitted_regs(th, th.op(n), d) for n in ("A", "B")}

    half_kernel = _half_plr_kernel("KMN", half_plr)
    th, half = regs(half_kernel)
    for name in ("A", "B"):
        if name in halved:
            full_kernel = dict(half_kernel)
            full_kernel["VgprGroup" + name] = 1
            full_kernel["VgprAlloc" + name] = 2
            _, full = regs(full_kernel)
            assert half[name] * 4 == full[name] * 3, (name, half, full)
            assert th.op(name).fragment.vgpr_group == 2
            assert th.op(name).fragment.ring_depths == {"g0": 3}
        else:
            assert th.op(name).fragment.vgpr_group == 1


def test_clr_and_explicit_va_precedence_is_calculated_in_registers():
    k = _half_plr_kernel("KMN", 1)

    # CLR0: explicit VA replaces PLR+1.
    theta = adapter.params_to_theta(kernel_to_params(k))
    assert theta.op("A").fragment.ring_depths == {"g0": 3}

    # CLR0 with VA=-1: derive PLR2+1.
    k["VgprAllocA"] = -1
    theta = adapter.params_to_theta(kernel_to_params(k))
    assert theta.op("A").fragment.ring_depths == {"g0": 3}

    # CLR0 with VG=-1: explicit VA alone overrides PLR+1 at the default VG1.
    k["VgprGroupA"] = -1
    k["VgprAllocA"] = 5
    theta = adapter.params_to_theta(kernel_to_params(k))
    assert theta.op("A").fragment.vgpr_group == 1
    assert theta.op("A").fragment.ring_depths == {"g0": 5}

    # Explicit VA remains authoritative under CLR1.
    k["VgprGroupA"] = 2
    k["VgprAllocA"] = 3
    k.update(ClusterLocalRead=1, ClusterLocalReadA=1, ClusterLocalReadB=1)
    theta = adapter.params_to_theta(kernel_to_params(k))
    assert theta.op("A").fragment.ring_depths == {"g0": 3}


def test_all_inner_forced_clr1_retains_a_large_enough_explicit_va():
    from Tensile.LoopModel import traversal as geometry
    from Tensile.LoopModel.schedule import BufferDepths, Schedule

    def assert_prefetch_is_kept(theta):
        depths = BufferDepths(theta, include_shared=False)
        operand = theta.op("A")
        requested = geometry.requested_read_ahead(theta, operand, depths)
        assert requested > 0
        assert Schedule(theta, depths).plan(operand, "g0").prefetch_steps == requested

    k = _half_plr_kernel("NKM", 1)
    # One all-inner buffer is a complete rotation unit; VG2 therefore needs two such halves.
    theta = adapter.params_to_theta(kernel_to_params(k))
    assert theta.op("A").fragment.ring_depths == {"g0": 3}
    assert_prefetch_is_kept(theta)

    k["VgprAllocA"] = 1
    k["ClusterLocalReadA"] = 1
    with pytest.raises(RuntimeError, match="smaller.*rotation"):
        adapter.params_to_theta(kernel_to_params(k))


def test_forced_all_inner_rejects_explicit_va_below_required_rotation():
    """Registers owns the floor; adapter rejects an explicit smaller allocation."""
    from Tensile.Components.LoopModel.Order import loopOrder

    k = _kernel(_mxf8(), _mi("fp8", (2, 2), (1, 1)),
                2, 2, loopOrder(5, 0, 1, 0), 1, 0, 2)
    k.update(WmmaInnerOrder=5, WmmaOuterOrder=0,
             ClusterLocalRead=0, ClusterLocalReadA=1, ClusterLocalReadB=0,
             VgprGroupA=2, VgprAllocA=3,
             VgprGroupB=-1, VgprAllocB=-1,
             SuppressNoLoadLoop=True, SourceSwap=False)
    with pytest.raises(RuntimeError, match="VgprAllocA=3.*smaller"):
        adapter.params_to_theta(kernel_to_params(k))


def test_f8maf2_clra_allocates_the_complete_resident_set():
    """CLR1 A needs four prefetch-unit buffers, not two region slots."""
    from Tensile.Components.LoopModel.Order import loopOrder
    from Tensile.LoopModel import traversal as geometry
    from Tensile.LoopModel.schedule import BufferDepths

    k = _kernel(_mxf8(), _mi("fp8", (8, 8), (2, 2)),
                2, 4, loopOrder(6, 2, 1, 1), 1, 1, 4)
    k.update(LoopIters=2, WmmaInnerOrder=6, WmmaOuterOrder=2,
             PrefetchLocalReadA=1, PrefetchLocalReadB=4,
             ClusterLocalRead=0, ClusterLocalReadA=1, ClusterLocalReadB=0,
             VgprGroupA=-1, VgprAllocA=-1,
             VgprGroupB=1, VgprAllocB=6,
             SuppressNoLoadLoop=True, SourceSwap=True)

    params = kernel_to_params(k)
    theta = adapter.params_to_theta(params)
    depths = BufferDepths(theta, include_shared=False)

    assert (params["VgprAllocA"], params["VgprAllocB"]) == (4, 6)
    assert geometry.register_layout(theta, depths)["A"]["total"] == 256
    assert geometry.register_layout(theta, depths)["B"]["total"] == 192


def test_f8maf2_vab3_forced_clrb_keeps_three_copy_ring():
    """Regression for the WIO2/WOO1 F8 kernel that was numerically wrong in FFM."""
    from Tensile.LoopModel import traversal as geometry
    from Tensile.Components.LoopModel.Order import loopOrder
    from Tensile.LoopModel.schedule import BufferDepths
    from Tensile.Lowering import steady_chain

    k = _kernel(_mxf8(), _mi("fp8", (8, 8), (2, 2)),
                2, 2, loopOrder(2, 1, 1, 1), 1, 1, 4)
    k.update(LoopIters=2, WmmaInnerOrder=2, WmmaOuterOrder=1,
             PrefetchLocalReadA=2, PrefetchLocalReadB=2,
             ClusterLocalRead=0, ClusterLocalReadA=0, ClusterLocalReadB=1,
             VgprGroupA=1, VgprAllocA=4,
             VgprGroupB=1, VgprAllocB=3,
             SuppressNoLoadLoop=True, SourceSwap=True)

    theta = adapter.params_to_theta(kernel_to_params(k))
    depths = BufferDepths(theta, include_shared=False)
    b = theta.op("B")
    assert b.fragment.ring_depths == {"g0": 3}
    assert geometry.group_ring_depth(theta, b, "g0", depths) == 3
    assert geometry.register_layout(theta, depths)["B"]["total"] == 192

    copies = geometry.ring_trip_copies(theta, depths)
    program = build_gir(theta, loop_copies=copies)
    assert copies == 3
    assert steady_chain(program) == ["steady", "steady1", "steady2"]
    assert not check_plan(program)

    from Tensile.LoopModel.loopir import render
    text = render(decode_theta(theta).program)
    assert "case 0,1,2" not in text
    assert "over {0, 1, 2}" in text
    lines = text.splitlines()
    assert not any(
        line.lstrip().startswith("if ")
        and following.lstrip().startswith("if ")
        and len(following) - len(following.lstrip())
            > len(line) - len(line.lstrip())
        for line, following in zip(lines, lines[1:])
    )


@pytest.mark.parametrize("wio,sa,sb,arm", [
    (1, 1, 0, 1), (2, 1, 0, 1), (4, 0, 1, 2), (6, 1, 0, 1),
])
def test_split_provided_vg_edge_orders_keep_prefetch_and_sources(wio, sa, sb, arm):
    from Tensile.LoopModel import traversal as geometry
    from Tensile.Components.LoopModel.Order import loopOrder
    from Tensile.LoopModel.schedule import BufferDepths, Schedule

    k = _kernel(_mxf8(), _mi("fp8", (2, 2), (1, 1)),
                2, 2, loopOrder(wio, 0, sa, sb), sa, sb, 2)
    k.update(WmmaInnerOrder=wio, WmmaOuterOrder=0,
             ClusterLocalRead=0, ClusterLocalReadA=0, ClusterLocalReadB=0,
             SuppressNoLoadLoop=True, SourceSwap=False)
    for side, bit in (("A", 1), ("B", 2)):
        k["VgprGroup" + side] = 2 if arm & bit else -1
        k["VgprAlloc" + side] = 3 if arm & bit else -1
    theta = adapter.params_to_theta(kernel_to_params(k))
    depths = BufferDepths(theta, include_shared=False)
    plans = Schedule(theta, depths)
    for side, bit in (("A", 1), ("B", 2)):
        if arm & bit:
            operand = theta.op(side)
            requested = geometry.requested_read_ahead(theta, operand, depths)
            assert requested > 0
            assert plans.plan(operand, "g0").prefetch_steps > 0
    copies = geometry.ring_trip_copies(theta, depths)
    assert not check_plan(build_gir(theta, loop_copies=copies))


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


# ---------------------------------------------------------------------------
# A region is an ordinary digit of the PU offset.  A short ring may defer a refill, but it never
# manufactures another read on an invariant consumer pass.
# ---------------------------------------------------------------------------
def _split_half_plr_theta(order, half_plr, sa, sb, wt=(2, 2), clr=0, pgr=2):
    k = _kernel(_mxf8(), _mi("fp8", wt, (1, 1)), pgr, 2, order, sa, sb, 1)
    k.update(HalfPLR=0, ClusterLocalRead=clr,
             ClusterLocalReadA=clr, ClusterLocalReadB=clr)
    for side, bit in (("A", 1), ("B", 2)):
        stated = bool(half_plr & bit)
        walks = ("M", "K") if side == "A" else ("N", "K")
        pu_axis = next(axis for axis in reversed(order) if axis in walks)
        split = sa if side == "A" else sb
        split_axis = (("M" if side == "A" else "N") if split == 1
                      else "K" if split == 2 else None)
        # TDMSplit has already supplied the half when it cuts this operand's PU.
        vg = 1 if split_axis == pu_axis else 2
        k["VgprGroup" + side] = vg if stated else -1
        k["VgprAlloc" + side] = 3 if stated else -1
    k["SuppressNoLoadLoop"] = True
    return adapter.params_to_theta(kernel_to_params(k))


def test_the_shallow_half_keeps_a_ring_digit_when_the_region_leads():
    """A TDMSplit-provided half is VG1/VA3 and still retains a physical ring digit."""
    from Tensile.LoopModel import traversal as g
    from Tensile.LoopModel.schedule import BufferDepths
    from Tensile.Components.LoopModel.Order import loopOrder

    th = _split_half_plr_theta(loopOrder(1, 1, 1, 0), 1, sa=1, sb=0)
    op, depths = th.op("A"), BufferDepths(th, include_shared=False)
    assert op.fragment.vgpr_group == 1
    assert op.fragment.ring_depths == {"g0": 3}
    assert g.split_unit_divisor(th, op) == 2
    slot = g.ring_slot(th, op, "g0", g.group_ring_depth(th, op, "g0", depths))
    assert slot is not None
    assert "M_split" in {axis.name for axis in slot.free_axes()}


@pytest.mark.parametrize("wio, sb", [(1, 0), (2, 0), (5, 0), (5, 1), (6, 0), (6, 1)])
@pytest.mark.parametrize("half_plr", [2, 3])
def test_shallow_half_buffer_splits_broadcast_pass_for_refills(wio, sb, half_plr):
    """VA3 splits the M pass so each target slot is refilled only after its final consumer."""
    from Tensile.Components.LoopModel.Order import loopOrder
    from Tensile.LoopModel.schedule import BufferDepths
    from Tensile.LoopModel import traversal as geometry
    from Tensile.Lowering.gir.emit_plan import plan_program

    if half_plr == 3 and wio == 5:
        with pytest.raises(RuntimeError, match="VgprAllocA=3.*smaller"):
            _split_half_plr_theta(loopOrder(wio, 1, 1, sb), half_plr, 1, sb)
        return
    theta = _split_half_plr_theta(loopOrder(wio, 1, 1, sb), half_plr, 1, sb)
    depths = BufferDepths(theta, include_shared=False)
    copies = geometry.ring_trip_copies(theta, depths)
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
    """A split that halves the PU uses VG1 and still builds the HalfPLR ring.

    These are the twelve PGR1/PGR2 YAML kernels that previously built one steady
    block, then returned to the wrong half-buffer on hardware.
    """
    from Tensile.Components.LoopModel.Order import loopOrder
    from Tensile.LoopModel.schedule import BufferDepths
    from Tensile.LoopModel import traversal as geometry
    from Tensile.Lowering import steady_chain
    from Tensile.Lowering.gir.emit_plan import plan_program
    from Tensile.Lowering.gir.frame_contract import build_contract

    theta = _split_half_plr_theta(loopOrder(wio, 1, 1, 1), half_plr, 1, 1)
    shaped = [theta.op(side) for side, bit in (("A", 1), ("B", 2)) if half_plr & bit]
    assert shaped
    assert all(geometry.vgpr_group(theta, op) == 1 for op in shaped)
    assert all(geometry.split_unit_divisor(theta, op) == 2 for op in shaped)
    depths = BufferDepths(theta, include_shared=False)
    assert all(geometry.vgpr_ring_carries(
        theta, op, op.fragment.groups()[0],
        geometry.group_ring_depth(theta, op, op.fragment.groups()[0], depths))
        for op in shaped)

    copies = geometry.ring_trip_copies(theta, depths)
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


def test_ungrouped_peer_defers_a_reload_that_would_clobber_its_broadcast_reuse():
    """VG1 uses the same register timeline as an explicit ring when placing read-ahead.

    With M_split outside B's walk, B's four region slots are consumed once in each M pass.
    Loading the next generation between those passes overwrites values the second pass still
    needs.  PLR remains requested and LoopIR splits the pass at the refill boundary.
    """
    from Tensile.Components.LoopModel.Order import loopOrder
    from Tensile.LoopModel.schedule import BufferDepths, Schedule
    from Tensile.LoopModel import traversal as geometry

    k = _kernel(_mxf8(), _mi("fp8", (2, 2), (1, 1)),
                1, 2, loopOrder(1, 1, 1, 1), 1, 1, 1)
    k.update(HalfPLR=0, ClusterLocalRead=0,
             ClusterLocalReadA=0, ClusterLocalReadB=0,
             PrefetchLocalReadA=2, PrefetchLocalReadB=1,
             SuppressNoLoadLoop=True,
             VgprGroupA=2, VgprAllocA=3,
             VgprGroupB=-1, VgprAllocB=-1)
    theta = adapter.params_to_theta(kernel_to_params(k))
    depths = BufferDepths(theta, include_shared=False)
    b = theta.op("B")
    group = b.fragment.groups()[0]
    buffers = geometry.group_ring_depth(theta, b, group, depths)

    assert geometry.requested_read_ahead(theta, b, depths) == 1
    assert geometry.register_reuse_verdict(theta, b, group, buffers, 1) \
        == geometry.CLOBBER
    schedule = Schedule(theta, depths)
    assert schedule.want(b) == 1
    assert schedule.steps(b, group, 1) == 1
    prog = build_gir(theta, loop_copies=geometry.ring_trip_copies(theta, depths))
    assert not check_plan(prog)


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
        k["VgprGroup" + side] = 3 if side == "A" else -1
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
    fake_prog = SimpleNamespace(params={})

    monkeypatch.setattr(
        program, "loopModelTheta",
        lambda *_args: (SimpleNamespace(operands=[]), object()))
    monkeypatch.setattr(
        program,
        "decode_theta",
        lambda *_args, **_kwargs: SimpleNamespace(
            program=object(),
            profile=SimpleNamespace(schedule=object()),
        ),
    )
    monkeypatch.setattr(program, "validate_loopir", lambda *_args: [])
    monkeypatch.setattr(program, "gir_pipeline", lambda: ())
    monkeypatch.setattr(
        program, "lower_to_gir", lambda *_args, **_kwargs: fake_prog
    )
    monkeypatch.setattr(program, "run_pipeline", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(program, "check_plan", lambda _prog: ["deliberate checker finding"])

    with pytest.raises(ValueError, match="semantic check"):
        program.loopModelGirProgram(writer, kernel)


def test_loopmodel_ir_verify_false_skips_all_ir_checks(monkeypatch):
    import importlib
    from types import SimpleNamespace

    program = importlib.import_module("Tensile.Components.LoopModel.Program")
    writer = SimpleNamespace(states=SimpleNamespace(kernelName="Cijk_NO_IR_VERIFY"))
    kernel = {
        "PrefetchGL2": 0,
        "VgprGroupA": -1,
        "VgprAllocA": -1,
        "VgprGroupB": -1,
        "VgprAllocB": -1,
    }
    fake_prog = SimpleNamespace(params={})
    decoded = {}
    lowered = {}
    pipelined = {}

    monkeypatch.setitem(
        program.globalParameters,
        "LoopModelIRVerify",
        False,
    )
    monkeypatch.setitem(program.globalParameters, "OutputLoopIR", False)
    monkeypatch.setattr(
        program,
        "loopModelTheta",
        lambda *_args: (SimpleNamespace(operands=[]), object()),
    )
    def decode(*_args, **kwargs):
        decoded.update(kwargs)
        return SimpleNamespace(
            program=object(),
            profile=SimpleNamespace(schedule=object()),
        )

    monkeypatch.setattr(program, "decode_theta", decode)
    monkeypatch.setattr(
        program,
        "validate_loopir",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("LoopIR verification ran")
        ),
    )
    monkeypatch.setattr(program, "gir_pipeline", lambda: ())

    def lower(*_args, **kwargs):
        lowered.update(kwargs)
        return fake_prog

    def run(_prog, **kwargs):
        pipelined.update(kwargs)

    monkeypatch.setattr(program, "lower_to_gir", lower)
    monkeypatch.setattr(program, "run_pipeline", run)
    monkeypatch.setattr(
        program,
        "check_plan",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("GIR verification ran")
        ),
    )
    assert program.loopModelGirProgram(writer, kernel) is fake_prog
    assert decoded["verify"] is False
    assert lowered["include_dependencies"] is False
    assert lowered["verify"] is False
    assert pipelined["verify"] is False

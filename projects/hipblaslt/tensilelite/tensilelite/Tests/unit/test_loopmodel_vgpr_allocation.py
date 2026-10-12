# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit

_LEAVES = Path(__file__).resolve().parents[2] / "Lowering" / "leaves.py"
_KERNEL_WRITER_ASSEMBLY = Path(__file__).resolve().parents[2] / "KernelWriterAssembly.py"
_LOOPMODEL_REGISTERS = (Path(__file__).resolve().parents[2]
                        / "Components" / "LoopModel" / "Registers.py")


def _function(path, name, namespace=None):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    method = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    function = ast.FunctionDef(
        name=method.name,
        args=method.args,
        body=method.body,
        decorator_list=[],
        returns=method.returns,
        type_comment=method.type_comment,
    )
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = dict(namespace or {})
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[method.name]


def _apply_method():
    return _function(_LOOPMODEL_REGISTERS, "applyLoopModelValuRegs")


def _writer(counts):
    return SimpleNamespace(
        states=SimpleNamespace(
            a=SimpleNamespace(numVgprValu=11),
            b=SimpleNamespace(numVgprValu=12),
            mxsa=SimpleNamespace(numVgprValu=13),
            mxsb=SimpleNamespace(numVgprValu=14),
            loopModelValuVgprs=dict(counts),
        ),
    )


def _kernel(use_loop_model=True, mx=True):
    return {
        "UseLoopModel": use_loop_model,
        "ProblemType": {
            "MXBlockA": 32 if mx else 0,
            "MXBlockB": 32 if mx else 0,
        },
    }


def test_ulm1_overwrites_scaffold_value_ring_sizes():
    writer = _writer({"A": 32, "B": 48, "MXSA": 8, "MXSB": 16})

    _apply_method()(writer, _kernel())

    assert writer.states.a.numVgprValu == 32
    assert writer.states.b.numVgprValu == 48
    assert writer.states.mxsa.numVgprValu == 8
    assert writer.states.mxsb.numVgprValu == 16


def test_ulm0_keeps_scaffold_value_ring_sizes():
    writer = _writer({"A": 32, "B": 48})

    _apply_method()(writer, _kernel(use_loop_model=False, mx=False))

    assert writer.states.a.numVgprValu == 11
    assert writer.states.b.numVgprValu == 12


def test_ulm1_requires_a_count_for_every_present_operand():
    writer = _writer({"A": 32})

    with pytest.raises(RuntimeError, match="B"):
        _apply_method()(writer, _kernel(mx=False))


def _buffer_counts(**counts):
    return SimpleNamespace(states=SimpleNamespace(**counts))


def test_ulm1_set_table_uses_each_operands_rotation_width():
    # vgprAllocationImplClassic stores one depth per operand.  The .set table
    # reads those fields; it does not take the max across operands.
    depth = _function(_LOOPMODEL_REGISTERS, "operandNumVgprBuffer")
    writer = _buffer_counts(numVgprBuffer=7, numVgprBufferA=2, numVgprBufferB=3,
                            numVgprBufferMXSA=0, numVgprBufferMXSB=0)
    count = _function(
        _KERNEL_WRITER_ASSEMBLY,
        "valuBufferCount",
        {"self": writer, "kernel": {"UseLoopModel": True}, "numBi": 7,
         "operandNumVgprBuffer": depth},
    )

    assert count("A") == 2
    assert count("B") == 3

    legacy = _buffer_counts(numVgprBuffer=7, numVgprBufferA=0, numVgprBufferB=0,
                            numVgprBufferMXSA=0, numVgprBufferMXSB=0)
    legacy_count = _function(
        _KERNEL_WRITER_ASSEMBLY,
        "valuBufferCount",
        {"self": legacy, "kernel": {"UseLoopModel": False}, "numBi": 7,
         "operandNumVgprBuffer": depth},
    )
    assert legacy_count("A") == 7


def _valu_alloc(kernel, tc, bpe):
    from Tensile.Components.LoopModel import Order, Registers
    from Tensile.Components.LoopModel.Registers import operandValuPuVaVg

    kernel = dict(kernel)
    kernel.setdefault("UseLoopModel", True)
    if "LoopOrder" not in kernel:
        kernel.setdefault("WmmaOuterOrder", 0)
    kernel.setdefault("PrefetchLocalRead", 0)
    kernel.setdefault("ClusterLocalRead", 0)
    kernel.setdefault(
        "LoopIters",
        int(kernel["DepthU"]) // int(kernel["MatrixInstK"]),
    )
    kernel.setdefault(
        "MIWaveTile",
        [kernel["MIWaveTileA"], kernel["MIWaveTileB"]],
    )
    kernel.setdefault(
        "MatrixInstruction",
        [
            16,
            16,
            kernel["MatrixInstK"],
            1,
            1,
            kernel["MIWaveTileA"],
            kernel["MIWaveTileB"],
            1,
            1,
        ],
    )
    Order.calculate(kernel)
    Registers.calculate(kernel)
    from Tensile.LoopModel import adapter
    theta = adapter.params_to_theta(
        adapter.kernel_to_params(kernel),
        validate_register_depths=False,
    )
    from Tensile.LoopModel.schedule import BufferDepths
    depths = BufferDepths(theta, include_shared=False)
    writer = SimpleNamespace(states=SimpleNamespace(
        bpr=4, packDTVA=False, convDTVA=False, packDTVB=False, convDTVB=False))
    writer._loopModelThetaCache = (id(kernel), (theta, depths))
    return operandValuPuVaVg(writer, kernel, tc, bpe)


def _valu_size(kernel, tc, bpe):
    """`(perBuffer, va)` -- the two terms most tests assert on."""
    alloc = _valu_alloc(kernel, tc, bpe)
    return alloc.perBuffer, alloc.va


def _nmk_kernel():
    # WIO5 = NMK, so both operands' prefetch unit is K.  DepthU 256 / MI K 128 = 2.
    return {
        "WmmaInnerOrder": 5,
        "DepthU": 256,
        "MatrixInstK": 128,
        "MIWaveTileA": 8,
        "MIWaveTileB": 8,
        "MIInputPerThreadA": 64,
        "MIInputPerThreadB": 64,
        "VgprGroupA": 1,
        "VgprGroupB": 1,
        "VgprAllocA": 8,
        "VgprAllocB": 2,
        "ClusterLocalReadA": 1,
        "ClusterLocalReadB": 0,
        "PrefetchLocalReadA": 2,
        "PrefetchLocalReadB": 1,
        "TDMSplitA": 0,
        "TDMSplitB": 0,
        "DirectToVgprA": False,
        "DirectToVgprB": False,
        "ProblemType": {"Sparse": 0},
    }


def test_nmk_va8_is_one_prefetch_unit_times_va():
    # One K-tile is 64 elems * 1 byte / 4 = 16 regs.  PU is 2, VG is 1, VA is 8.
    # 32 * 8 = 256.  The wave-tile block (8 * 16) times VA8 is 1024 and is not used.
    per, va = _valu_size(_nmk_kernel(), "A", 1)

    assert (per, va) == (32, 8)
    assert per * va == 2 * 8 // 1 * 16


def test_nmk_b_clr0_uses_stated_va_not_the_outer_loop():
    per, va = _valu_size(_nmk_kernel(), "B", 1)

    assert (per, va) == (32, 2)
    assert per * va == 64


def test_kmn_a_prefetch_unit_is_m_and_clr0_va_is_plr_plus_one():
    kernel = _nmk_kernel()
    kernel.update(WmmaInnerOrder=1, VgprAllocA=-1, ClusterLocalReadA=0, PrefetchLocalReadA=1)

    per, va = _valu_size(kernel, "A", 1)

    assert (per, va) == (128, 2)
    assert per * va == 8 * 2 // 1 * 16


def test_clr1_without_stated_va_uses_calculated_rotation_depth():
    kernel = _nmk_kernel()
    kernel["VgprAllocA"] = -1

    per, va = _valu_size(kernel, "A", 1)

    assert va == 8
    assert per * va == 256


def test_nkm_clr1_counts_the_split_half_as_an_outer_loop():
    # WIO6 = NKM, so A's prefetch unit is M.  TDMSplit MT halves it to 4 tiles,
    # and the other half is one of the loops outside the unit, beside K.  CLR
    # on with no stated VA allocates the complete resident set in prefetch-unit
    # buffers.
    kernel = _nmk_kernel()
    kernel.update(WmmaInnerOrder=6, TDMSplitA=1, VgprAllocA=-1, VgprGroupA=-1,
                  ClusterLocalReadA=1, PrefetchLocalReadA=1)

    per, va = _valu_size(kernel, "A", 1)

    assert (per, va) == (64, 4)
    assert per * va == 256


def test_mkn_tdmsplit_mt_halves_b_prefetch_unit_and_keeps_vg1():
    # MKN: B's prefetch unit is N.  TDMSplit MT already halves it, so HalfPLR's
    # shape is VG1 VA3 PLR2 rather than VG2.
    kernel = _nmk_kernel()
    kernel.update(WmmaInnerOrder=4, TDMSplitB=1, VgprGroupB=1, VgprAllocB=3,
                  ClusterLocalReadB=0, PrefetchLocalReadB=2)

    per, va = _valu_size(kernel, "B", 1)

    assert (per, va) == (64, 3)
    assert per * va == (8 // 2) * 3 // 1 * 16


def test_lowering_folds_the_layout_address_into_the_allocation():
    # The GIR layout address is a rotation block per ring slot.  When the ring
    # repeats the resident set over several prefetch generations it walks past
    # PU * VA / VG, and the generation it repeats is the buffer it reuses.
    fold = _function(_LEAVES, "allocated_valu_offset")

    # NMK VA8: slot stride 128, 8 slots.  The allocation is 8 * 32 = 256, so
    # slot 2 unit 0 is the next generation of buffer 0.
    assert fold(256, 7 * 128 + 7 * 16) == 240
    assert fold(256, 2 * 128 + 0 * 16) == 0
    # NKM: slot stride 128, 2 slots, allocation 4 * 64.  Nothing to fold, and
    # the two M_split halves keep their own registers.
    assert fold(256, 0 * 128 + 7 * 16) == 112
    assert fold(256, 1 * 128 + 7 * 16) == 240
    # A zero span is the non-UseLoopModel path.
    assert fold(0, 7 * 128 + 7 * 16) == 7 * 128 + 7 * 16


def test_lowering_spells_physical_valu_addresses_through_the_x_ring():
    fold = _function(_LEAVES, "allocated_valu_offset")
    symbol = _function(
        _LEAVES,
        "allocated_valu_symbol",
        {"allocated_valu_offset": fold},
    )

    assert symbol("A", 192, 64, 1, 0) == "ValuA_X0_I0"
    assert symbol("A", 192, 64, 1, 64 + 16) == "ValuA_X1_I0+16"
    assert symbol("A", 192, 64, 1, 128 + 48) == "ValuA_X2_I0+48"
    assert symbol("A", 256, 32, 1, 2 * 128) == "ValuA_X0_I0"


def test_tdmsplit_calculates_the_remaining_vg_divisor():
    # The IR dump states PU * VA / VG, so the allocation reports the terms and
    # not only their product.  MKN splits B's prefetch unit from 8 tiles to 4.
    kernel = _nmk_kernel()
    kernel.update(WmmaInnerOrder=4, TDMSplitB=1, VgprGroupB=2, VgprAllocB=3,
                  ClusterLocalReadB=0, PrefetchLocalReadB=2)

    alloc = _valu_alloc(kernel, "B", 1)

    assert (alloc.puTiles, alloc.puRegs, alloc.vg, alloc.va) == (4, 64, 1, 3)
    assert alloc.perBuffer == alloc.puRegs // alloc.vg
    assert alloc.total == alloc.perBuffer * alloc.va == 192


def test_halfplr_shape_is_vg2_va3():
    kernel = _nmk_kernel()
    kernel.update(WmmaInnerOrder=1, VgprGroupA=2, VgprAllocA=3, ClusterLocalReadA=0,
                  PrefetchLocalReadA=2)

    per, va = _valu_size(kernel, "A", 1)

    assert (per, va) == (64, 3)
    assert per * va == 8 * 3 // 2 * 16


def test_degenerate_free_axis_uses_the_same_k_unit_for_model_and_allocation():
    kernel = {
        "WmmaInnerOrder": 6,
        "WmmaOuterOrder": 2,
        "DepthU": 256,
        "MatrixInstK": 128,
        "MIWaveTileA": 1,
        "MIWaveTileB": 4,
        "MIInputPerThreadA": 64,
        "MIInputPerThreadB": 64,
        "MIInputPerThreadMXSA": 4,
        "MIInputPerThreadMXSB": 4,
        "PrefetchLocalRead": 1,
        "ClusterLocalRead": 1,
        "TDMSplitA": 0,
        "TDMSplitB": 1,
        "DirectToVgprA": False,
        "DirectToVgprB": False,
        "ProblemType": {"Sparse": 0, "MXBlockA": 32, "MXBlockB": 32},
    }

    assert _valu_alloc(kernel, "A", 1).perBuffer == 32
    assert _valu_alloc(kernel, "MXSA", None).total == 2


def test_split_mxs_allocation_uses_its_model_footprint():
    kernel = {
        "LoopOrder": "KMKNMN",
        "DepthU": 256,
        "MatrixInstK": 128,
        "MIWaveTileA": 2,
        "MIWaveTileB": 2,
        "MIInputPerThreadA": 64,
        "MIInputPerThreadB": 64,
        "MIInputPerThreadMXSA": 4,
        "MIInputPerThreadMXSB": 4,
        "PrefetchLocalRead": 1,
        "ClusterLocalRead": 0,
        "TDMSplitA": 1,
        "TDMSplitB": 0,
        "DirectToVgprA": False,
        "DirectToVgprB": False,
        "ProblemType": {"Sparse": 0, "MXBlockA": 32, "MXBlockB": 32},
    }

    # M_split advances the ring offset; it does not duplicate the physical PU.
    assert _valu_alloc(kernel, "MXSA", None).total == 4


def test_mab3_distributed_mxsb_uses_covered_unit_and_reduction_ring():
    kernel = {
        "UseLoopModel": True,
        "WmmaInnerOrder": 6,
        "WmmaOuterOrder": 2,
        "DepthU": 256,
        "MatrixInstK": 128,
        "MatrixInstM": 16,
        "MatrixInstN": 16,
        "MatrixInstruction": [16, 16, 128, 1, 1, 1, 4, 1, 4],
        "MIWaveTile": [1, 4],
        "MIWaveTileA": 1,
        "MIWaveTileB": 4,
        "MIInputPerThreadA": 64,
        "MIInputPerThreadB": 64,
        "MIInputPerThreadMXSA": 4,
        "MIInputPerThreadMXSB": 4,
        "PrefetchGlobalRead": 2,
        "PrefetchLocalRead": 1,
        "PrefetchLocalReadA": 1,
        "PrefetchLocalReadB": 1,
        "ClusterLocalRead": 1,
        "ClusterLocalReadA": 1,
        "ClusterLocalReadB": 1,
        "VgprGroupA": 1,
        "VgprGroupB": 1,
        "VgprAllocA": 1,
        "VgprAllocB": 4,
        "TDMSplitA": 0,
        "TDMSplitB": 1,
        "TDMFuse": 2,
        "VectorWidthA": 1,
        "VectorWidthB": 1,
        "VectorWidthMXSA": 1,
        "VectorWidthMXSB": 1,
        "LocalReadVectorWidthMXS": 16,
        "ISA": [12, 5, 0],
        "MXScaleFormat": "InMemorySwizzle",
        "WavefrontSize": 32,
        "NumWaves": 4,
        "DirectToVgprA": False,
        "DirectToVgprB": False,
        "ProblemType": {"Sparse": 0, "MXBlockA": 32, "MXBlockB": 32},
    }

    alloc = _valu_alloc(kernel, "MXSB", None)
    assert (alloc.perBuffer, alloc.va, alloc.total) == (2, 4, 8)


def _f8_split_register_kernel(va_b=3):
    return {
        "UseLoopModel": True,
        "MatrixInstruction": [16, 16, 128, 1, 1, 8, 8, 2, 2],
        "MatrixInstK": 128,
        "DepthU": 256,
        "LoopIters": 2,
        "MIWaveTile": [8, 8],
        "MIWaveTileA": 8,
        "MIWaveTileB": 8,
        "MIInputPerThreadA": 64,
        "MIInputPerThreadB": 64,
        "WmmaInnerOrder": 2,
        "WmmaOuterOrder": 1,
        "PrefetchGlobalRead": 2,
        "PrefetchLocalRead": 2,
        "ClusterLocalRead": 0,
        "ClusterLocalReadA": 0,
        "ClusterLocalReadB": 1,
        "VgprGroupA": 1,
        "VgprGroupB": 1,
        "VgprAllocA": 4,
        "VgprAllocB": va_b,
        "TDMSplitA": 1,
        "TDMSplitB": 1,
        "NumWaves": 4,
        "ElemBytes": 1,
        "ProblemType": {"Sparse": 0},
    }


def _f8_split_nkm_register_kernel():
    """The VA3/VA3 NKM shape whose split regions used to alias live inputs."""
    kernel = _f8_split_register_kernel()
    kernel.update(
        MatrixInstruction=[16, 16, 128, 1, 1, 8, 10, 2, 2],
        MIWaveTile=[8, 10],
        MIWaveTileB=10,
        WmmaInnerOrder=6,
        WmmaOuterOrder=2,
        PrefetchLocalRead=3,
        PrefetchLocalReadA=2,
        PrefetchLocalReadB=3,
        ClusterLocalReadA=0,
        ClusterLocalReadB=0,
        VgprAllocA=3,
        VgprAllocB=3,
    )
    return kernel


def test_calculated_f8_split_ring_matches_allocation():
    from Tensile.Components.LoopModel import Order, Registers
    from Tensile.LoopModel import adapter
    from Tensile.LoopModel import traversal as geometry
    from Tensile.LoopModel.schedule import BufferDepths

    kernel = _f8_split_register_kernel()
    Order.calculate(kernel)
    Registers.calculate(kernel)
    theta = adapter.params_to_theta(kernel)
    depths = BufferDepths(theta, include_shared=False)

    assert kernel["VgprAllocA"] == geometry.group_ring_depth(
        theta, theta.op("A"), "g0", depths) == 4
    assert kernel["VgprAllocB"] == geometry.group_ring_depth(
        theta, theta.op("B"), "g0", depths) == 3


def test_split_regions_are_flattened_offsets_with_loop_split_refills():
    from Tensile.Components.LoopModel import Order, Registers
    from Tensile.LoopModel import adapter
    from Tensile.LoopModel import traversal as geometry
    from Tensile.LoopModel.decoder import decode_theta
    from Tensile.LoopModel.schedule import BufferDepths, Schedule, prefetch_refusals
    from Tensile.Lowering import lower_to_gir
    from Tensile.Lowering.gir.emit_plan import plan_block
    from Tensile.Lowering.gir.passes import pipeline, run_pipeline
    from Tensile.Lowering.gir.verify_dataflow import check_plan

    kernel = _f8_split_nkm_register_kernel()
    Order.calculate(kernel)
    Registers.calculate(kernel)
    theta = adapter.params_to_theta(kernel)
    depths = BufferDepths(theta, include_shared=False)
    schedule = Schedule(theta, depths)

    expected = {
        "A": (2, 4),
        "B": (3, 10),
    }
    for name, (plr, period) in expected.items():
        operand = theta.op(name)
        group = operand.fragment.groups()[0]
        width = geometry.group_ring_depth(theta, operand, group, depths)
        assert not hasattr(operand.fragment, "va_includes_regions")
        assert not hasattr(operand.fragment, "va_time_shares_regions")
        assert geometry.requested_read_ahead(theta, operand, depths) == plr
        assert schedule.want(operand) == plr
        assert schedule.steps(operand, group, plr) == plr
        assert geometry.ring_positions(theta, operand, group, width) == period

    a_slot = geometry.ring_slot(theta, theta.op("A"), "g0", 3)
    assert a_slot.evaluate({"M_split": 1, "K_inner": 0, "M_inner": 0}) == 2
    assert a_slot.evaluate({"M_split": 1, "K_inner": 1, "M_inner": 0}) == 0
    assert prefetch_refusals(theta, depths) == ()

    decoded = decode_theta(theta)
    split_axes = {
        axis.name
        for bindings in decoded.profile.read_issue_bindings.values()
        for axis, _value in bindings
    }
    assert split_axes == set()
    program = lower_to_gir(decoded.program, decoded.profile)
    run_pipeline(program, passes=pipeline())
    assert check_plan(program) == []
    reads = [action for action in plan_block(program, "steady") if action.kind == "read"]
    keys = [(action.at["tc"], action.at["tile_flat"], action.at["k_flat"],
             tuple(action.at.get("issue") or ()))
            for action in reads]
    assert len(keys) == len(set(keys))


def test_register_owner_calculates_all_eight_operand_parameters():
    from Tensile.Components.LoopModel import Order, Registers

    kernel = _f8_split_register_kernel()
    kernel.update({
        "PrefetchLocalReadA": -1,
        "PrefetchLocalReadB": -1,
        "ClusterLocalReadA": -1,
        "ClusterLocalReadB": -1,
        "VgprGroupA": -1,
        "VgprGroupB": -1,
        "VgprAllocA": -1,
        "VgprAllocB": -1,
    })
    Order.calculate(kernel)
    Registers.calculate(kernel)

    assert all(
        int(kernel[base + side]) >= 0
        for base in (
            "PrefetchLocalRead",
            "ClusterLocalRead",
            "VgprGroup",
            "VgprAlloc",
        )
        for side in ("A", "B")
    )


def test_adapter_rejects_calculated_ring_smaller_than_required_depth():
    from Tensile.Components.LoopModel import Order, Registers
    from Tensile.LoopModel import adapter

    kernel = _f8_split_register_kernel(va_b=1)
    Order.calculate(kernel)
    Registers.calculate(kernel)
    with pytest.raises(RuntimeError, match="VgprAllocB=1.*smaller.*rotation"):
        adapter.params_to_theta(kernel)


def test_wio1_clr1_va1_holds_its_complete_two_tile_prefetch_unit():
    from Tensile.Components.LoopModel import Order, Registers
    from Tensile.LoopModel import adapter

    kernel = _f8_split_register_kernel()
    kernel.update(
        MatrixInstruction=[16, 16, 128, 1, 1, 2, 2, 2, 2],
        MIWaveTile=[2, 2],
        MIWaveTileA=2,
        MIWaveTileB=2,
        DepthU=128,
        LoopIters=1,
        WmmaInnerOrder=1,
        WmmaOuterOrder=0,
        PrefetchLocalRead=2,
        ClusterLocalRead=1,
        ClusterLocalReadA=-1,
        ClusterLocalReadB=-1,
        VgprGroupA=-1,
        VgprGroupB=-1,
        VgprAllocA=-1,
        VgprAllocB=-1,
        TDMSplitA=0,
        TDMSplitB=0,
    )
    Order.calculate(kernel)
    Registers.calculate(kernel)

    assert kernel["VgprAllocA"] == 1
    adapter.params_to_theta(kernel)

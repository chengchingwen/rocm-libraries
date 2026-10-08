# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import pytest

pytestmark = pytest.mark.unit

_LEAVES = Path(__file__).resolve().parents[2] / "Lowering" / "leaves.py"
_KERNEL_WRITER_ASSEMBLY = Path(__file__).resolve().parents[2] / "KernelWriterAssembly.py"
_TDM_SPLIT = Path(__file__).resolve().parents[2] / "Components" / "TDMSplit.py"
_LOOPMODEL_REGISTERS = (Path(__file__).resolve().parents[2]
                        / "Components" / "LoopModel" / "Registers.py")
_AXIS_MT = "MT"
_AXIS_DU = "DU"
_SPLIT_OF = None
_VALU_ALLOC = None


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


def _split_of():
    global _SPLIT_OF
    if _SPLIT_OF is None:
        _SPLIT_OF = _function(
            _TDM_SPLIT,
            "split_of",
            {"TDM_SPLIT_AXIS": {0: (1, 1), 1: (2, 1), 2: (1, 2)},
             "AXIS_MT": _AXIS_MT, "AXIS_DU": _AXIS_DU},
        )
    return _SPLIT_OF


def _valu_alloc(kernel, tc, bpe):
    writer = SimpleNamespace(states=SimpleNamespace(
        bpr=4, packDTVA=False, convDTVA=False, packDTVB=False, convDTVB=False))
    size = _function(
        _LOOPMODEL_REGISTERS,
        "operandValuPuVaVg",
        {"split_of": _split_of(), "AXIS_MT": _AXIS_MT, "AXIS_DU": _AXIS_DU,
         "ValuAlloc": _valu_alloc_type()},
    )
    return size(writer, kernel, tc, bpe)


def _valu_alloc_type():
    global _VALU_ALLOC
    if _VALU_ALLOC is None:
        source = _LOOPMODEL_REGISTERS.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(_LOOPMODEL_REGISTERS))
        cls = next(node for node in ast.walk(tree)
                   if isinstance(node, ast.ClassDef) and node.name == "ValuAlloc")
        module = ast.Module(body=[cls], type_ignores=[])
        ast.fix_missing_locations(module)
        namespace = {"NamedTuple": NamedTuple}
        exec(compile(module, str(_LOOPMODEL_REGISTERS), "exec"), namespace)
        _VALU_ALLOC = namespace["ValuAlloc"]
    return _VALU_ALLOC


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


def test_clr1_without_stated_va_uses_the_outer_loop_size():
    kernel = _nmk_kernel()
    kernel["VgprAllocA"] = -1

    per, va = _valu_size(kernel, "A", 1)

    assert va == 8
    assert per * va == 256


def test_nkm_clr1_counts_the_split_half_as_an_outer_loop():
    # WIO6 = NKM, so A's prefetch unit is M.  TDMSplit MT halves it to 4 tiles,
    # and the other half is one of the loops outside the unit, beside K.  CLR
    # on with no stated VA holds them all: 16 tiles / 4 = 4 buffers.
    kernel = _nmk_kernel()
    kernel.update(WmmaInnerOrder=6, TDMSplitA=1, VgprAllocA=-1, VgprGroupA=-1,
                  ClusterLocalReadA=1, PrefetchLocalReadA=1)

    per, va = _valu_size(kernel, "A", 1)

    assert (per, va) == (64, 4)
    assert per * va == 8 * 2 // 1 * 16


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


def test_the_allocation_carries_the_terms_of_its_own_formula():
    # The IR dump states PU * VA / VG, so the allocation reports the terms and
    # not only their product.  MKN splits B's prefetch unit from 8 tiles to 4.
    kernel = _nmk_kernel()
    kernel.update(WmmaInnerOrder=4, TDMSplitB=1, VgprGroupB=2, VgprAllocB=3,
                  ClusterLocalReadB=0, PrefetchLocalReadB=2)

    alloc = _valu_alloc(kernel, "B", 1)

    assert (alloc.puTiles, alloc.puRegs, alloc.vg, alloc.va) == (4, 64, 2, 3)
    assert alloc.perBuffer == alloc.puRegs // alloc.vg
    assert alloc.total == alloc.perBuffer * alloc.va == 96


def test_halfplr_shape_is_vg2_va3():
    kernel = _nmk_kernel()
    kernel.update(WmmaInnerOrder=1, VgprGroupA=2, VgprAllocA=3, ClusterLocalReadA=0,
                  PrefetchLocalReadA=2)

    per, va = _valu_size(kernel, "A", 1)

    assert (per, va) == (64, 3)
    assert per * va == 8 * 3 // 2 * 16

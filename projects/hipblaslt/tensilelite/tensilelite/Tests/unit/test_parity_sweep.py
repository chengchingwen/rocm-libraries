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


def test_pgr3_is_unreachable_under_ulm():
    """`PrefetchGlobalRead >= 3` cannot be generated under UseLoopModel — checked, not assumed."""
    import importlib.util

    # Read the FILE, not a bound attribute: several of the ULM rejects this reasoning depends on
    # live outside `assignDerivedParameters`, and `SolutionStructs` re-exports the class over the
    # module name, so attribute lookup is the wrong instrument here.
    src = open(importlib.util.find_spec("Tensile.SolutionStructs.Solution").origin).read()
    assert '"PrefetchGlobalRead>=3 Supports only DirectToLdsA and DirectToLdsB"' in src
    assert '"PrefetchGlobalRead>=3 Supports only ScheduleIterAlg == 3"' in src
    # ...and that ULM refuses both of those, which is what makes the combination empty.
    assert "UseLoopModel does not support DirectToLds" in src
    assert "UseLoopModel supports only ScheduleIterAlg 0 or 4" in src


def test_the_decoder_really_does_assert_at_pgr3():
    """Non-vacuity for the test above: the unreachability claim is only interesting if the
    decoder would in fact fail there.  If this ever starts passing, PGR>=3 has been fixed and
    both this test and ledger note should be retired."""
    k = {**_BASE, "PrefetchGlobalRead": 3, "NumLdsBlk": 3}
    # RuntimeError, not AssertionError: an inexpressible config is a per-kernel refusal.
    with pytest.raises(RuntimeError, match="JOIN whose predecessors"):
        build_gir(adapter.params_to_theta(kernel_to_params(k)))

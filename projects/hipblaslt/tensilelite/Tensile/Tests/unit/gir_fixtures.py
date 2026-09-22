# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Shared fixtures for the GIR unit tests (test_gir.py, test_gir_to_rocisa.py).
The canonical bf16 NT KMN kernel (MI 16x16x32, DepthU 64, PGR2) and its PLR0 / single-LDS
variants, plus the theta->GIR helpers.  Kept rocisa-free so the GIR tests run under a plain pytest.
"""

from Tensile.LoopModel import adapter


def mi_inputs(kernel):
    """Add what a Solution derives about read COUNTS, so a fixture counts reads like a real kernel.

    `MIInputPerThread*` is `SolutionStructs.Validators.MatrixInstruction`'s M (or N) x K x B /
    wavefront, then `// MXBlock * (32 // MatrixInstM)` for a scale.  `ReadInstructions` is the
    resolved per-operand count, set here as well because a fixture handed straight to
    `params_to_theta` never passes through `kernel_to_params` that would resolve it.
    Derived, not written as literals, so changing a fixture's MatrixInstruction cannot leave a
    stale count behind.
    """
    mi, wave = kernel["MatrixInstruction"], kernel.get("WavefrontSize") or 32
    pt = kernel.get("ProblemType", {})
    out = {"MIInputPerThreadA": mi[0] * mi[2] * mi[3] // wave,
           "MIInputPerThreadB": mi[1] * mi[2] * mi[3] // wave}
    for tc, mn in (("A", mi[0]), ("B", mi[1])):
        block = pt.get("MXBlock%s" % tc)
        if block:
            out["MIInputPerThreadMXS%s" % tc] = out["MIInputPerThread%s" % tc] // block * (32 // mn)
    merged = {**kernel, **out}
    # A scale's read width is kernel-wide, an input's is per-operand -- LraTileAssignment's rule.
    reads = {}
    for tc, per_thread in out.items():
        name = tc[len("MIInputPerThread"):]
        key = "LocalReadVectorWidthMXS" if "MXS" in name else "LocalReadVectorWidth%s" % name
        lrvw = int(merged.get(key) or merged.get("LocalReadVectorWidth") or 0)
        if lrvw > 0 and per_thread >= lrvw and not per_thread % lrvw:
            reads[name] = per_thread // lrvw
    return {**merged, "ReadInstructions": {**(kernel.get("ReadInstructions") or {}), **reads}}


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
BF16_NT_KMN = mi_inputs(BF16_NT_KMN)
BF16_NT_KMN_PLR0 = {**BF16_NT_KMN, "PrefetchLocalRead": 0}
# The multi-wave wave-separated TDM point: Phi fuses A and B onto ONE cooperative load AND two agents
# cooperate. The agent count is what makes the hazards cross-agent; the fuse is orthogonal.
BF16_NT_KMN_FUSED_XAGENT = {**BF16_NT_KMN_PLR0, "TDMFuse": 0, "NumWaves": 2}
# The case that motivated the rewrite: a LONE operand cooperatively loaded by several agents.  No
# fuse anywhere, and still cross-agent -- the old member-disagreement predicate returned False here.
# No TDM, because a fuse IS the wave selection: with the mover on both tensors and several waves,
# the default row already shares a descriptor.
BF16_NT_KMN_UNFUSED_MULTIAGENT = {**BF16_NT_KMN_PLR0, "NumWaves": 4, "TDMInst": 0}
BF16_NT_KMN_1LDS = {**BF16_NT_KMN, "NumLdsBlk": 1, "1LDSBuffer": 1}


def theta(params):
    """Build theta, re-deriving the read counts from whatever shape the caller ended up with.

    Idempotent, and applied on every call because a test typically spreads a base fixture and then
    overrides MatrixInstruction or a read width -- which would otherwise keep the base's count.
    """
    if params.get("MatrixInstruction"):
        params = mi_inputs(params)
    return adapter.params_to_theta(params)

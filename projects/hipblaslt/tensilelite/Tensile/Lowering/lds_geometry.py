# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
LDS tile geometry -- the PURE arithmetic behind an operand's per-tile LDS address.

Split out of `leaves.py` for the same reason `emit_plan.py` is split out of `gir_to_rocisa.py`.

"""

from __future__ import annotations

from ..tdm_split import split_packs_lds, AXIS_MT, AXIS_DU   # noqa: F401  (AXIS_DU re-exported)

#: Bytes per register.
BPR = 4

#: How many lanes share one range of K positions, so a lane's consecutive local-read chunks sit
#: this many chunk-widths apart along the unroll axis.  The scaffold spells it as a bare `* 2`.
LANE_INTERLEAVE = 2


def read_fragments(blockWidth, bpeDS, lrvw: int, inputPerThUnroll: int, matrixInstK: int) -> tuple:
    """The ds_readS ONE (tile, summation-substep) local-read act decomposes into, as
    `((unrollElems, regOff), ...)` -- both relative to the act's own base, the first in ELEMENTS
    along the unroll axis, the second in registers.
    """
    vwTrLoad = int(blockWidth * BPR / bpeDS)
    nInner = lrvw // vwTrLoad if vwTrLoad else 0
    nOuter = inputPerThUnroll // lrvw if lrvw else 0
    regSpan = int(inputPerThUnroll * bpeDS // BPR)
    if not (nInner and nOuter) or nOuter * nInner * int(blockWidth) != regSpan:
        raise NotImplementedError(
            "local-read fragments do not tile the tile's registers (%s x %s x blockWidth %s != "
            "%s; inputPerThUnroll=%s, lrvw=%s, vwTrLoad=%s, bpeDS=%s)"
            % (nOuter, nInner, blockWidth, regSpan, inputPerThUnroll, lrvw, vwTrLoad, bpeDS))
    if nOuter * nInner * vwTrLoad * LANE_INTERLEAVE != matrixInstK:
        raise NotImplementedError(
            "local-read fragments do not cover MatrixInstK (%s x %s x %s elems x %s lanes != %s; "
            "lrvw=%s, bpeDS=%s)"
            % (nOuter, nInner, vwTrLoad, LANE_INTERLEAVE, matrixInstK, lrvw, bpeDS))
    return tuple(((v + i * nInner * LANE_INTERLEAVE) * vwTrLoad,
                  int(blockWidth) * (v + nInner * i))
                 for i in range(nOuter) for v in range(nInner))


def region_split_is_packed(unrollMajor: bool, nsplit: int, axis=AXIS_MT) -> bool:
    """Does TDMSplit make each region a SEPARATELY PACKED LDS block?

    Delegates to `tdm_split.split_packs_lds`, which states the rule once for the descriptor side
    and this side together.  The answer is a DIAGONAL over (axis, layout) -- a partition packs
    blocks exactly when it cuts the LDS image's INNER axis -- not a property of either alone.
    
    """
    return nsplit > 1 and split_packs_lds(axis, unrollMajor)


def region_row_elems(extent: int, ldsPad: int, unrollMajor: bool, nsplit: int, axis=AXIS_MT) -> int:
    """Row length, in elements, along the LDS image's INNER axis -- of ONE REGION when the split
    packs regions separately, of the whole tile otherwise.
    """
    packed = region_split_is_packed(unrollMajor, nsplit, axis)
    return (extent // max(1, nsplit) if packed else extent) + ldsPad


def fold_inner_offset(innerElems: int, extent: int, unrollMajor: bool, nsplit: int,
                      axis=AXIS_MT) -> tuple:
    """`(region, innerInRow)` -- split an INNER-axis element offset into the region it lands in and
    its position inside that region.  The counterpart of `region_row_elems`: that shortens the row,
    this says which row-block a coordinate past the end belongs to.
    """
    if not region_split_is_packed(unrollMajor, nsplit, axis):
        return (0, innerElems)
    span = max(1, extent // max(1, nsplit))
    return (int(innerElems) // span, int(innerElems) % span)


def addr_coord_on_split_axis(within: int, flat: int, ctx) -> int:
    """Which coordinate the ADDRESS uses along the axis a TDMSplit cut: the WITHIN-REGION one when
    the regions are separately packed, the FLAT one when they are a contiguous continuation.

    ONE RULE, and the two halves are not interchangeable.
    
    """
    return within if region_split_is_packed(
        ctx.unrollMajor, ctx.nsplit, getattr(ctx, "splitAxis", AXIS_MT)) else flat


def tile_row(ctx, t: int) -> int:
    """The LDS ROW index of this operand's wave-tile `t`, in an unroll-major (DU-major) layout.

    The distribution is BY VECTOR GROUP: `vw` adjacent tiles, then the next group starts one
    wave-group of rows further down.
    """
    if not ctx.unrollMajor:
        return t % tiles_per_region(ctx)
    vw = max(1, ctx.vectorWidth)
    group = ctx.MIWaveGroupShape[ctx.tile01] if ctx.MIWaveGroupShape else vw
    return (t // vw) * (getattr(ctx, "segRowShape", 0) or group) + (t % vw)


def component_fold(ctx, row: int) -> tuple:
    """`(rowInComponent, componentBytes)` -- LDSSegmentInterleave stores an operand's two
    components `segWriteStrideBytes` apart, so a row past the end of component 0 belongs to
    component 1 and owes that jump.  Post-pad, exactly like `region_bytes`.
    """
    cols = getattr(ctx, "segCompCols", 0)
    if cols <= 0:
        return (row, 0)
    return (row % cols, (row // cols) * int(ctx.segWriteStrideBytes))


def tiles_per_region(ctx) -> int:
    """How many of this operand's wave tiles live in ONE TDMSplit region."""
    if getattr(ctx, "splitAxis", AXIS_MT) != AXIS_MT:
        return max(1, ctx.miWaveTileAxis)
    return max(1, ctx.miWaveTileAxis // max(1, ctx.nsplit))


def region_bytes(ctx, region: int) -> int:
    """Byte displacement from the operand's LDS base to the start of TDMSplit `region`.

    THIS IS NON-ZERO EXACTLY ON THE PACKED DIAGONAL -- when the split cuts the LDS image's INNER
    axis (`region_split_is_packed`) -- and the asymmetry is a property of the image, not a
    heuristic.  Which axis is inner is layout-decided, so each layout packs on a DIFFERENT split.
    
    """
    if (not region_split_is_packed(ctx.unrollMajor, ctx.nsplit, getattr(ctx, "splitAxis", AXIS_MT))
            or not ctx.splitBoundaryBytes):
        return 0
    return int(region) * int(ctx.splitBoundaryBytes)

# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""`Lowering/tdm_split.py` -- the one TDMSplit derivation."""

import pytest

from Tensile.Components.TDMSplit import derive, split_packs_lds, AXIS_MT, AXIS_DU


@pytest.mark.parametrize("axis,unrollMajor,expect", [
    (AXIS_MT, True,  False),   # [free][unroll], cut the OUTER axis -> contiguous continuation
    (AXIS_MT, False, True),    # [unroll][free], cut the INNER axis -> packed blocks
    (AXIS_DU, True,  True),    # [free][unroll], unroll is INNER   -> packed blocks
    (AXIS_DU, False, False),   # [unroll][free], unroll is OUTER   -> contiguous
    (None,    True,  False),
    (None,    False, False),
])
def test_a_split_packs_lds_blocks_iff_it_cuts_the_images_INNER_axis(axis, unrollMajor, expect):
    """The diagonal, and it is a diagonal -- neither the axis nor the layout decides alone."""
    assert split_packs_lds(axis, unrollMajor) is expect


def test_the_NT_MT_split_reproduces_the_emitted_descriptor():
    """Every field against `Cijk_Ailk_Bjlk MT32x32x64`, which passes 576/576."""
    g = derive(2, AXIS_MT, tlu=True, mt=32, du=64, bpe=2,
               ldsBlockSizePerPad=32, ldsPadBytes=16)
    assert g.splitDim == 0, "free is contiguous under tlu, so the MT split divides dim0"
    assert g.strideAxis == "free"
    assert g.globalConstBytes == 32, "emitted: s_mov_b32 s[sgprtdmAGlobalSplitIncs], 32"
    assert g.ldsStepBytes == 3072, "emitted: s_mov_b32 s[sgprtdmALdsSplitIncs], 3072"
    assert g.packed is True


def test_the_TN_MT_split_keeps_dim1_and_owes_no_region_term():
    """The 864/864 shape.  Unchanged by this module is the point: TN was already correct."""
    g = derive(2, AXIS_MT, tlu=False, mt=32, du=64, bpe=2,
               ldsBlockSizePerPad=32, ldsPadBytes=16)
    assert g.splitDim == 1
    assert g.packed is False


def test_the_DU_split_moves_the_axis_but_NOT_the_lds_partition():
    """`ldsStepBytes` is half the TILE on either axis, because a split partitions one tile however
    the image is ordered.  Only whether the READER owes that displacement differs, and that is
    `packed`.  Getting this backwards would put region 1 at half a row instead of half a tile."""
    mt_g = derive(2, AXIS_MT, tlu=True, mt=32, du=64, bpe=2,
                  ldsBlockSizePerPad=32, ldsPadBytes=16)
    du_g = derive(2, AXIS_DU, tlu=True, mt=32, du=64, bpe=2,
                  ldsBlockSizePerPad=32, ldsPadBytes=16)
    assert du_g.ldsStepBytes == mt_g.ldsStepBytes == 3072
    assert du_g.splitDim == 1 and du_g.strideAxis == "reduction"
    assert du_g.globalConstBytes == 64, "du*bpe/2, NOT mt*bpe/2"
    assert du_g.packed is False


def test_mt_and_du_constants_DIVERGE_off_the_coincidence_shape():
    """At MT32/DU64/bpe2 the two axes' global constants are 32 vs 64 -- but the defect that hid for
    weeks was `mt*bpe/2 == du/2` being TRUE at that shape for the *element* count.  Pin a shape
    where nothing coincides, so a future edit that swaps the extents cannot pass."""
    mt_g = derive(2, AXIS_MT, tlu=True, mt=128, du=32, bpe=2)
    du_g = derive(2, AXIS_DU, tlu=True, mt=128, du=32, bpe=2)
    assert mt_g.globalConstBytes == 128      # 128*2/2
    assert du_g.globalConstBytes == 32       # 32*2/2
    assert mt_g.ldsStepBytes == du_g.ldsStepBytes == 128 * 32 * 2 // 2


@pytest.mark.parametrize("tlu", [True, False])
@pytest.mark.parametrize("axis", [AXIS_MT, AXIS_DU, None])
def test_factor_1_is_INERT_on_every_layout_and_axis(tlu, axis):
    """Emission neutrality, enforced where it is derived: an unsplit operand must produce zero
    displacement and no packing, so every `n == 1` kernel is byte-identical to its predecessor."""
    g = derive(1, axis, tlu=tlu, mt=32, du=64, bpe=2, ldsBlockSizePerPad=32, ldsPadBytes=16)
    assert not g.isSplit
    assert (g.globalConstBytes, g.ldsStepBytes, g.packed) == (0, 0, False)


def test_the_regions_tile_the_operand_exactly_once():
    """`factor` regions of `ldsStepBytes` each must cover the padded tile with no gap or overlap --
    the property that makes the walk's close-to-zero meaningful."""
    for factor in (2, 4):
        for axis in (AXIS_MT, AXIS_DU):
            g = derive(factor, axis, tlu=True, mt=64, du=64, bpe=2)
            assert g.ldsStepBytes * factor == 64 * 64 * 2


# ---------------------------------------------------------------- the param -> theta mapping

@pytest.mark.parametrize("a,b,expect", [
    (0, 0, None),            # control
    (1, 1, (2, 2, 1, 1)),    # exactly the old `TDMSplit: true`
    (2, 2, (1, 1, 2, 2)),
    (1, 2, (2, 1, 1, 2)),    # mixed axes -- expressible in theta, rejected by Solution.py
    (2, 0, (1, 1, 2, 1)),
    (0, 1, (1, 2, 1, 1)),
])
def test_the_knob_maps_onto_thetas_positional_4_list(a, b, expect):
    """`TDMSplitA/B` are a RESTRICTED way of writing `[A_MT, B_MT, A_DU, B_DU]`: one axis per
    operand at factor 2.  theta keeps the general form so the model can be exercised past what L3
    implements, and `(1,1) -> (2,2,1,1)` is the compatibility anchor -- that tuple is what the old
    boolean produced."""
    from Tensile.Components.TDMSplit import split_factors
    k = {"TDMSplitA": a, "TDMSplitB": b, "ProblemType": {"Sparse": 0}}
    assert split_factors(k) == expect


@pytest.mark.parametrize("tc,expect", [("A", (2, AXIS_MT)), ("B", (1, None))])
def test_split_of_is_PER_OPERAND(tc, expect):
    from Tensile.Components.TDMSplit import split_of
    k = {"TDMSplitA": 1, "TDMSplitB": 0, "ProblemType": {"Sparse": 0}}
    assert split_of(k, tc) == expect


@pytest.mark.parametrize("tc", ["MXSA", "MXSB", "Metadata"])
def test_mx_scale_and_metadata_tensors_are_NEVER_split(tc):
    """The `not MXS and not Sparse` guard was re-spelled at nine call sites and had already
    drifted (some excluded metadata, some did not).  It lives in `split_of` now, so a caller
    cannot forget it."""
    from Tensile.Components.TDMSplit import split_of
    assert split_of({"TDMSplitA": 1, "TDMSplitB": 1, "ProblemType": {"Sparse": 0}}, tc) == (1, None)


def test_a_sparse_problem_is_never_split_on_either_operand():
    from Tensile.Components.TDMSplit import split_of
    k = {"TDMSplitA": 1, "TDMSplitB": 2, "ProblemType": {"Sparse": 1}}
    assert split_of(k, "A") == (1, None) and split_of(k, "B") == (1, None)


def test_the_per_region_dim1_span_is_ZERO_when_the_split_took_dim0():
    """`_tdmSplitDim1ForRegion` derives region 1's dim1 extent as `H0 - dim1SpanPerRegion`.  When
    the split cut dim0 the regions share a dim1 extent, and returning 0 makes `H1 == H0` fall out
    instead of needing a branch.  It replaced a hardcoded `MacroTile // 2`, which was wrong on
    both counts once the axis and the factor became parameters."""
    mt_tlu = derive(2, AXIS_MT, tlu=True,  mt=32, du=64, bpe=2)   # splitDim 0
    du_tlu = derive(2, AXIS_DU, tlu=True,  mt=32, du=64, bpe=2)   # splitDim 1
    mt_um  = derive(2, AXIS_MT, tlu=False, mt=32, du=64, bpe=2)   # splitDim 1
    assert mt_tlu.dim1SpanPerRegion == 0
    assert du_tlu.dim1SpanPerRegion == 32      # du/2
    assert mt_um.dim1SpanPerRegion == 16       # mt/2


@pytest.mark.parametrize("axis,unrollMajor,packed,want", [
    # the SAME 2x2 diagonal as `split_packs_lds`, read from the address side
    (AXIS_DU, True,  True,  "within"),   # TN + DU: packed   -> region_bytes carries the region
    (AXIS_DU, False, False, "flat"),     # NT + DU: contiguous -> the coord must carry it
    (AXIS_MT, False, True,  "within"),   # NT + MT: packed
    (AXIS_MT, True,  False, "flat"),     # TN + MT: contiguous
])
def test_the_address_uses_the_WITHIN_REGION_coord_only_when_the_region_is_PACKED(
        axis, unrollMajor, packed, want):
    """`region_bytes` and the split-axis coordinate are two halves of ONE address and must agree.
    Packed: `region * splitBoundary` carries the region, so the coordinate is within-region.
    """
    import types
    from Tensile.Lowering.lds_geometry import (addr_coord_on_split_axis, region_bytes,
                                               region_split_is_packed)
    ctx = types.SimpleNamespace(unrollMajor=unrollMajor, nsplit=2, splitAxis=axis,
                                splitBoundaryBytes=2176)
    assert region_split_is_packed(unrollMajor, 2, axis) is packed
    got = addr_coord_on_split_axis(3, 7, ctx)             # within=3, flat=7
    assert got == (3 if want == "within" else 7)
    # and the two halves are consistent: a region term exists exactly when the coord is in-region
    assert (region_bytes(ctx, 1) != 0) is packed


def test_an_unsplit_or_same_valued_read_is_UNAFFECTED_by_the_selection():
    """`kFlat == kIdx` for every unsplit read and for a split on the OTHER axis, so the rule is
    inert there -- which is why this could be added without re-measuring the green cells."""
    import types
    from Tensile.Lowering.lds_geometry import addr_coord_on_split_axis
    for um in (True, False):
        for ax in (AXIS_MT, AXIS_DU, None):
            ctx = types.SimpleNamespace(unrollMajor=um, nsplit=1, splitAxis=ax)
            assert addr_coord_on_split_axis(5, 5, ctx) == 5


# ===========================================================================================
# read_fragments -- which ds_reads one (tile, substep) local-read act decomposes into.

@pytest.mark.parametrize("name,blockWidth,bpeDS,lrvw,inputPerThUnroll,miK,want", [
    # bf16, LDS-transpose (`ds_load_tr16_b128`, blockWidth 4 = 8 elements = lrvw) at MI k 32.
    # NT MT32x32 DU128: per act offsets 0 and 1536 bytes at UnrollStride 48 elements
    # x 2 bytes -> 0 and 16 elements; destination registers +0 and +4.
    ("bf16 LDSTr", 4, 2, 8, 16, 32, ((0, 0), (16, 4))),
    # bf16, general unroll-major (`ds_load_b128`).  TN MT32x32 DU64: 0 and 32 bytes at
    # UnrollStride 1 x 2 bytes -> 0 and 16 elements; registers +0 and +4.  SAME table as the
    # transpose path -- the read path changes the strides, not the decomposition.
    ("bf16 general", 4, 2, 8, 16, 32, ((0, 0), (16, 4))),
    # fp8, LDS-transpose (`ds_load_b64_tr_b8`, blockWidth 2 = 8 elements) against lrvw 16 at MI
    # k 64: the INNER loop is live, so 4 reads.  The scaffold spells these as
    # `(v * 8 + i * 32)` elements and `2 * (innerIdx + 2 * outerIdx)` registers.
    ("fp8 LDSTr", 2, 1, 16, 32, 64, ((0, 0), (8, 2), (32, 4), (40, 6))),
    # fp8, general unroll-major (`ds_load_b128` = 16 elements = lrvw) at MI k 128: FOUR outer
    # chunks, and this is the row that discriminates.
    ("fp8 general", 4, 1, 16, 64, 128, ((0, 0), (32, 4), (64, 8), (96, 12))),
])
def test_read_fragments_reproduces_the_emitted_ds_read_table(
        name, blockWidth, bpeDS, lrvw, inputPerThUnroll, miK, want):
    from Tensile.Lowering.lds_geometry import read_fragments
    assert read_fragments(blockWidth, bpeDS, lrvw, inputPerThUnroll, miK) == want


def test_read_fragments_tile_the_registers_and_the_K_extent():
    """The two invariants, stated over the same four shapes: the register offsets are dense and
    exactly fill `inputPerThUnroll * bpeDS / 4`, and the elements this lane reads times the lanes
    it interleaves with are the whole `MatrixInstK`."""
    from Tensile.Lowering.lds_geometry import read_fragments, LANE_INTERLEAVE, BPR
    for blockWidth, bpeDS, lrvw, ipt, miK in ((4, 2, 8, 16, 32), (2, 1, 16, 32, 64),
                                              (4, 1, 16, 64, 128)):
        frags = read_fragments(blockWidth, bpeDS, lrvw, ipt, miK)
        regs = sorted(r for _, r in frags)
        assert regs == list(range(0, int(ipt * bpeDS // BPR), int(blockWidth)))
        elemsPerRead = int(blockWidth * BPR / bpeDS)
        assert len(frags) * elemsPerRead * LANE_INTERLEAVE == miK
        # and no two reads land on the same elements
        assert len({u for u, _ in frags}) == len(frags)


@pytest.mark.parametrize("blockWidth,bpeDS,lrvw,ipt,miK,why", [
    # lrvw not a multiple of the per-read width -> the chunk cannot be tiled at all
    (4, 2, 6, 16, 32, "registers"),
    # a per-read width that does not divide the chunk (b96 against lrvw 8): the fragments cover
    # 6 of the tile's 8 registers, so two would go unwritten and the wmma would read them
    (3, 2, 8, 16, 32, "registers"),
    # registers tile, but the lane's elements do not add up to MatrixInstK -- the case a
    # register-only check would wave through while every read lands in the wrong K position
    (4, 2, 8, 16, 64, "MatrixInstK"),
])
def test_read_fragments_REFUSES_a_shape_it_cannot_tile(blockWidth, bpeDS, lrvw, ipt, miK, why):
    """It raises rather than returning a plausible table, because the caller emits from it: a
    wrong table is silently wrong numerics, and there is nothing in the assembly that names it."""
    from Tensile.Lowering.lds_geometry import read_fragments
    with pytest.raises(NotImplementedError) as e:
        read_fragments(blockWidth, bpeDS, lrvw, ipt, miK)
    assert why in str(e.value)


# ------------------------------------------------------------------- wave region span
def _wrs_kernel(wt, wg, vw=None, bytesPerElem=0.25, split=1, mi=16, seg=None):
    """A kernel dict with just the keys `wave_region_span` reads."""
    class _DT:
        def __init__(self, r): self._r = r
        def numRegisters(self): return self._r
    k = {"MatrixInstruction": [mi, mi, 128, 1, 1, wt, wt, wg, wg],
         "MIWaveGroup": [wg, wg], "MIWaveTile": [wt, wt],
         "TDMSplitA": split, "TDMSplitB": split,
         "ProblemType": {"MacDataTypeA": _DT(bytesPerElem), "MacDataTypeB": _DT(bytesPerElem),
                         "Sparse": 0}}
    if vw:
        k["VectorWidthA"] = k["VectorWidthB"] = vw
    if seg:
        k["LDSSegmentInterleave"] = 1
        k["LDSSegInterleaveOffsets"] = seg
    return k


def test_a_waves_region_span_is_NOT_the_split_factor_when_MIWaveGroup_moves_it():
    """`nsplit` is how many regions the TILE is cut into; a WAVE reaches only the ones its
    `VectorWidth` x `MIWaveGroup` distribution lands in, and the two differ.
    """
    from Tensile.Components.TDMSplit import wave_region_span
    assert wave_region_span(_wrs_kernel(2, 2), "A") == 1, \
        "MIWaveGroup[2,2] MIWaveTile2: both of a wave's tiles are in region 0"
    assert wave_region_span(_wrs_kernel(4, 1, vw=1, bytesPerElem=0.5), "A") == 2
    assert wave_region_span(_wrs_kernel(4, 1, vw=2, bytesPerElem=0.5), "A") == 2
    assert wave_region_span(_wrs_kernel(2, 2, split=0), "A") == 1, "unsplit is always one region"


def test_the_wave_span_agrees_with_the_region_derivation_ORACLE():
    """The span and `region_derivation_check` are two derivations of one fact, so they must agree:
    a span BELOW the split factor is exactly the condition under which the index-derived region
    disagrees with the position-derived one."""
    from Tensile.Components.TDMSplit import wave_region_span, split_of, AXIS_MT
    from Tensile.Lowering.tool.region_derivation_check import TileCtx, check_tiles
    for wt, wg, vw, bpe in ((2, 2, 2, 0.25), (4, 1, 1, 0.5), (4, 1, 2, 0.5)):
        k = _wrs_kernel(wt, wg, vw=vw, bytesPerElem=bpe)
        n, _axis = split_of(k, "A")
        ctx = TileCtx(unrollMajor=True, nsplit=n, splitAxis=AXIS_MT, vectorWidth=vw,
                      MIWaveGroupShape=(16 * wg * vw,) * 2, tile01=0,
                      miWaveTileAxis=wt, macroTile=wt * wg * 16)
        mislabelled = bool(check_tiles(ctx))
        assert (wave_region_span(k, "A") < n) == mislabelled, \
            f"WT{wt} WG{wg} VW{vw}: span={wave_region_span(k,'A')} n={n} oracle={mislabelled}"


def test_port_split_shortens_the_region_extent_with_the_row_stride():
    """The target: MIWT[8,8], VW[4,4], WG[2,2], TDMSplitA/B=1, LDSSI=1.

    `portSplitA` drops A's per-vIdx row stride from the whole wave-group shape to one wave shape.
    The write side makes the matching change: `tdmSplitLdsBoundary` divides the baseline region
    footprint by `numVectorsPerTile` and stacks A's two TDMSplit regions inside each segment.
    Applying only the row-stride half made A span one region while baseline B spanned two, so the
    model widened A and forced both region loads to complete.  Both operands physically span both
    regions across their two vIdx instructions and are coordinate-selected.
    """
    from Tensile.Components.TDMSplit import wave_region_span
    seg = {"portSplitA": True, "footprintPacked": True, "writeStrideBytes": 1 << 16}
    baseline = _wrs_kernel(8, 2, vw=4)
    interleaved = _wrs_kernel(8, 2, vw=4, seg=seg)
    assert wave_region_span(baseline, "A") == wave_region_span(baseline, "B") == 2
    assert wave_region_span(interleaved, "A") == wave_region_span(interleaved, "B") == 2


def test_TDMFuse_admits_a_mixed_AXIS_pair_and_still_rejects_a_mixed_COUNT_pair():
    """The fuse gate's predicate is the region COUNT, not parameter equality."""
    from Tensile.Components.TDMSplit import split_of

    def counts(a, b):
        k = {"TDMSplitA": a, "TDMSplitB": b, "ProblemType": {"Sparse": 0}}
        return split_of(k, "A")[0], split_of(k, "B")[0]

    # the four pairs the gate must still reject -- a real 2-vs-1 count mismatch
    for a, b in ((1, 0), (0, 1), (2, 0), (0, 2)):
        na, nb = counts(a, b)
        assert na != nb, f"({a},{b}) must be a count mismatch, got {na} vs {nb}"

    # the two pairs the gate must now ADMIT -- same count, different axis
    for a, b in ((1, 2), (2, 1)):
        na, nb = counts(a, b)
        assert na == nb == 2, f"({a},{b}) must be two regions each, got {na} vs {nb}"
        assert split_of({"TDMSplitA": a, "TDMSplitB": b, "ProblemType": {"Sparse": 0}}, "A")[1] \
            != split_of({"TDMSplitA": a, "TDMSplitB": b, "ProblemType": {"Sparse": 0}}, "B")[1], \
            f"({a},{b}) is supposed to be the MIXED-AXIS case"

    # and the symmetric pairs stay admitted
    for a, b in ((0, 0), (1, 1), (2, 2)):
        na, nb = counts(a, b)
        assert na == nb, f"({a},{b}) symmetric must agree, got {na} vs {nb}"

    # SPARSE: `split_of` answers (1, None) for every operand, so a count test is vacuously equal
    # while the OLD raw-parameter test still fired -- a second over-rejection the narrowing fixes.
    sk = {"TDMSplitA": 1, "TDMSplitB": 0, "ProblemType": {"Sparse": 1}}
    assert split_of(sk, "A")[0] == split_of(sk, "B")[0] == 1
    assert int(sk["TDMSplitA"]) != int(sk["TDMSplitB"]), "the raw params DO differ here"


def test_tdmsplit_factorizes_an_axis_exactly():
    from Tensile.Components.TDMSplit import factor_axis

    assert factor_axis(8, 1, "WT") == 8
    assert factor_axis(8, 2, "WT") == 4
    assert factor_axis(8, 4, "WT") == 2
    with pytest.raises(ValueError, match="factorize WT"):
        factor_axis(7, 2, "WT")
    with pytest.raises(ValueError, match="factorize WT"):
        factor_axis(1, 2, "WT")


def test_tiles_per_region_DEPENDS_ON_the_Solution_divisibility_gate():
    """WHY THE `MIWaveTile % nsplit` GATE CANNOT BE REMOVED."""
    from Tensile.Components.TDMSplit import AXIS_MT
    from Tensile.Lowering.lds_geometry import tiles_per_region
    from Tensile.Lowering.tool.region_derivation_check import TileCtx, check_tiles

    def ctx(wt, n):
        return TileCtx(unrollMajor=True, nsplit=n, splitAxis=AXIS_MT, vectorWidth=1,
                       MIWaveGroupShape=(32, 32), tile01=0,
                       miWaveTileAxis=wt, macroTile=wt * 2 * 16)

    # THE TRAP: the floor fabricates a divisor of 1 where the true value is 1/2 a tile.
    assert tiles_per_region(ctx(1, 2)) == 1, \
        "if this stops being 1 the floor changed and this whole hazard needs re-measuring"
    assert 1 // 2 == 0, "the true tiles-per-region at MIWaveTile 1, nsplit 2 is ZERO, not 1"

    # AND THE ORACLE CANNOT FLAG IT -- it shares the same floor, so it reports agreement.
    assert not [f for f in check_tiles(ctx(1, 2)) if f.tile >= 0], (
        "check_tiles reports NO index-vs-position disagreement here, which is exactly why it must "
        "not be used as evidence that this shape is safe")

    # the divisible shapes the gate admits: the floor is exact, so the oracle is meaningful there
    for wt in (2, 4, 8):
        assert tiles_per_region(ctx(wt, 2)) == wt // 2 and wt % 2 == 0, \
            f"MIWaveTile {wt} must divide evenly -- these are the shapes the gate lets through"


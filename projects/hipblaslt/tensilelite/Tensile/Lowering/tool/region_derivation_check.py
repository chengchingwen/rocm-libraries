# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
RegionDerivationCheck -- is a wave tile's TDMSplit region the same whether you derive it from
the TILE INDEX or from the tile's actual FREE-AXIS POSITION?
WHY THE TWO CAN DISAGREE. `lds_geometry` already holds both derivations, for different callers.

"""

from __future__ import annotations

from dataclasses import dataclass

from ..lds_geometry import tile_row, tiles_per_region, region_split_is_packed
from ...Components.TDMSplit import AXIS_MT


@dataclass
class TileCtx:
    """The shape `lds_geometry` needs, with the same field names its functions read."""
    unrollMajor:      bool
    nsplit:           int
    splitAxis:        str
    vectorWidth:      int
    MIWaveGroupShape: tuple
    tile01:           int
    miWaveTileAxis:   int
    macroTile:        int
    splitBoundaryBytes: int = 1        # only its truthiness matters here


@dataclass
class Finding:
    """One wave tile whose region comes out differently by index than by position."""
    tile:          int
    by_index:      int
    by_position:   int
    row:           int
    note:          str


def rows_per_region(ctx: TileCtx) -> int:
    """Free-axis rows in one region.  The split divides the MacroTile, not the wave's tile count."""
    return max(1, ctx.macroTile // max(1, ctx.nsplit))


def check_tiles(ctx: TileCtx):
    """Findings for every wave tile whose two region derivations disagree, plus the floor check.

    Only meaningful for an MT (free-axis) split: a summation-axis split leaves every tile present
    in every region, which `tiles_per_region` already degenerates for."""
    out = []
    if ctx.nsplit <= 1 or ctx.splitAxis != AXIS_MT:
        return out
    per = tiles_per_region(ctx)
    rpr = rows_per_region(ctx)
    if ctx.miWaveTileAxis % ctx.nsplit:
        out.append(Finding(-1, per, -1, -1,
                           f"MIWaveTile {ctx.miWaveTileAxis} is not divisible by nsplit "
                           f"{ctx.nsplit}: tiles_per_region floors to {per}, so tile "
                           f"{per * ctx.nsplit}..{ctx.miWaveTileAxis - 1} fall outside every region"))
    for t in range(ctx.miWaveTileAxis):
        row = tile_row(ctx, t)
        by_index = t // max(1, per)
        by_position = row // rpr
        if by_index != by_position:
            out.append(Finding(t, by_index, by_position, row,
                               "tile index and free position name different regions"))
    return out


def describe(ctx: TileCtx) -> str:
    per, rpr = tiles_per_region(ctx), rows_per_region(ctx)
    packed = region_split_is_packed(ctx.unrollMajor, ctx.nsplit, ctx.splitAxis)
    return (f"MIWaveTile={ctx.miWaveTileAxis} VW={ctx.vectorWidth} "
            f"groupShape={ctx.MIWaveGroupShape[ctx.tile01]} MT={ctx.macroTile} "
            f"nsplit={ctx.nsplit} {'packed' if packed else 'contiguous'} "
            f"-> tiles/region={per}, rows/region={rpr}")

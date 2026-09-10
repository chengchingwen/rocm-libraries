# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""LoopModel -- the theta schedule-model decoder, ported into TensileLite (was `spacetime` in the
decoder_proto repo).
"""

from . import adapter, traversal
from .checks import Obligation, build_ledger, check_ledger_discharged, validate_loopir
from .emit import build_ir, emit_mainloop
from .traversal import (MmaShape, frag_regs, free_tiles, group_live_peak, group_ring_size, k_tiles,
                       load_regs, mma_shape, operand_buffer_regs, operand_footprint_regs,
                       operand_generations, operand_tile_bytes, plan_transfers)
from .ir import (Await, Bind, Branch, Cond, Counter, Expr, HOP_COUNTER, Inst, Load, Loop, Mma,
                 Peel, Placement, Pred, Space, cst)
from .render import render_geometry, render_ir, render_ir_raw, render_stream
from .schedule import build_S, derive_S, group_width
from .theta import DepthMap, Fragment, Hop, Axis, Operand, Theta

__all__ = [
    "Space", "Counter", "HOP_COUNTER",
    "Load", "Mma", "Expr", "cst", "Placement", "Await", "Inst",
    "Loop", "Branch", "Bind", "Peel", "Cond", "Pred",
    "Mode", "Fragment", "Hop", "Operand", "Theta", "DepthMap",
    "build_S", "derive_S", "group_rate", "group_live_peak", "group_width",
    "MmaShape", "plan_transfers", "mma_shape",
    "frag_regs", "free_tiles", "k_tiles",
    "operand_buffer_regs", "load_regs", "operand_tile_bytes",
    "operand_generations", "operand_footprint_regs",
    "Obligation", "build_ledger", "build_ir",
    "check_ledger_discharged", "emit_mainloop",
    "render_ir", "render_ir_raw", "render_stream", "render_geometry",
    "validate_loopir",
    "traversal", "adapter",
]

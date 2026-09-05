# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
PlacementPass -- resolve each PendingMark's Region to a concrete anchor.

Policy travels with the Region (design "placement by swap TYPE -- a user directive, not just
sigma"), because it differs by what the Mark mutates.

"""

from __future__ import annotations

from .base import Pass
from ..nodes import BLOCK_ENTRY, BLOCK_EXIT, MIDPOINT


class PlacementPass(Pass):
    """Resolve each PendingMark's legal window to one concrete anchor. Mutates no IR."""
    def run(self, prog, am):
        for pm in prog.pending:
            pm.anchor = self._resolve(prog, pm.region)
        return ()                        # no IR mutation

    @classmethod
    def _resolve(cls, prog, region):
        body = prog.block(region.block).body
        lo = cls._slot(body, region.after, opening=True)     # first legal slot
        hi = cls._slot(body, region.before, opening=False)   # one past the last legal slot
        if hi < lo:
            return (region.block, region.before)             # malformed window: stated bound
        idx = lo + (hi - lo) // 2 if region.policy == MIDPOINT else lo
        return (region.block, body[idx] if idx < len(body) else BLOCK_EXIT)

    @staticmethod
    def _slot(body, anchor, opening):
        """`after` -> the slot just past it; `before` -> the slot it occupies."""
        if anchor is BLOCK_ENTRY or anchor == BLOCK_ENTRY:
            return 0
        if anchor is BLOCK_EXIT or anchor == BLOCK_EXIT:
            return len(body)
        for i, node in enumerate(body):
            if node is anchor:
                return i + 1 if opening else i
        return 0 if opening else len(body)

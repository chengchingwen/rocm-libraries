# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
Pass base class.
"""

from __future__ import annotations


class Pass:
    """A pass that may edit block BODIES (insert/remove instructions) but not the CFG.

    `structural = False` is the default and is a CLAIM the pass makes: block set, terminators and
    preds/succs are the same on the way out as on the way in.  A pass that changes any of those is
    a `StructuralPass` -- see below for why the distinction is worth a type."""

    structural = False

    @property
    def name(self):
        return type(self).__name__

    def run(self, prog, am):
        """Mutate `prog` in place; return an iterable of invalidated analysis names (may be ())."""
        raise NotImplementedError


class StructuralPass(Pass):
    """A pass that may CHANGE THE CFG: add or remove blocks, retarget terminators, move the test."""

    structural = True

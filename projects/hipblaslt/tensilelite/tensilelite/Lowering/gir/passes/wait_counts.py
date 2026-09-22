# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The residuals into the body as `Mark('waitcnt', ...)`.

Runs last: a residual counts the emitted order.  A counter key is present only when charged;
absence says nothing, it is not zero.
"""

from __future__ import annotations

from .base import Pass
from ..nodes import Mark
from ..analyses.wait_counts import WaitCounts


class WaitCntPass(Pass):
    """Insert one `waitcnt` Mark before each instruction that owes a residual."""

    def run(self, prog, am):
        charged = am.get(WaitCounts(), prog).essential()
        if not len(charged):
            return prog
        by_block = {}
        for site in charged:
            by_block.setdefault(site.block, {}).setdefault(site.pos, []).append(site)

        for label, at_pos in by_block.items():
            block = prog.block(label)
            if block.model_only:
                continue  # nothing emits it, so nothing waits
            body = block.body
            for pos in sorted(at_pos, reverse=True):
                if pos > len(body):
                    continue
                sites = sorted(at_pos[pos], key=lambda site: site.counter)
                if len({site.counter for site in sites}) != len(sites):
                    raise RuntimeError(
                        "WaitCntPass: solver returned several waits for one shared counter at "
                        f"{label}:{pos}")
                marks = []
                for site in sites:
                    fact = {
                        site.counter: site.n,
                        "hazard": site.kind,
                        "from": site.producer,
                        "frames": site.frames,
                        "relations": tuple(relation.dump() for relation in site.relations),
                    }
                    marks.append(Mark("waitcnt", fact))
                body[pos:pos] = marks
        prog.bump()
        return prog

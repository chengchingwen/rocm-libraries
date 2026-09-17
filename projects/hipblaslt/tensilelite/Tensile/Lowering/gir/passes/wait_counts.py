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
                sites = at_pos[pos]
                if pos >= len(body):
                    continue
                fact = {site.counter: site.n for site in sites}
                lead = min(sites, key=lambda s: (s.n, s.counter))
                raw = {t for s in sites for t in s.tokens}
                fact.update(hazard=lead.kind, **{"from": lead.producer},
                            frames=max(s.frames for s in sites),
                            tokens=tuple(sorted(raw)),
                            order_tokens=tuple(sorted({t for s in sites for t in s.order_tokens}
                                                      - raw)))
                body.insert(pos, Mark("waitcnt", fact))
        prog.bump()
        return prog

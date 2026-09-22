# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Stable wait-count analysis façade.

`CounterFlow` is the production implementation.  `BackwardWaitCounts` remains available for
directionally independent differential tests while both consume the same PotentialWait set.
"""

from __future__ import annotations

from .backward_wait_counts import BackwardWaitCounts, potential_ranks
from .counter_flow import CounterFlow
from .potential_waits import (IssueInfo, IssueKey, PotentialWait, PotentialWaitSet,
                              PotentialWaits, anchor_site)
from .wait_common import (DSCNT, FANOUT_META, TENSORCNT, WaitCountSet, WaitRelation,
                          WaitSite, counter_for, counter_of, edge_domain,
                          emitting_reads, fan_out, issued, is_register_fill,
                          predecessors, trip_domain)


class WaitCounts(CounterFlow):
    """Production GIR residual analysis, implemented by forward CounterFlow."""

    _counter_for = staticmethod(counter_for)
    _is_register_fill = staticmethod(is_register_fill)


def differential_wait_counts(prog, am):
    """Return exact site/residual disagreements between the two implementations."""
    forward = am.get(CounterFlow(), prog)
    backward = am.get(BackwardWaitCounts(), prog)
    a = {(site.block, site.pos, site.counter): site.n for site in forward}
    b = {(site.block, site.pos, site.counter): site.n for site in backward}
    return tuple((key, a.get(key), b.get(key))
                 for key in sorted(set(a) | set(b)) if a.get(key) != b.get(key))


# Temporary private aliases for downstream diagnostics migrating from the old monolith.
_anchor_site = anchor_site
_predecessors = predecessors
_trip_domain = trip_domain
_edge_domain = edge_domain
_issued = issued

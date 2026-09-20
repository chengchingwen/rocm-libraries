# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Backward completion-counter ranking over GIR's finite frame graph."""

from __future__ import annotations

from ..analysis import Analysis
from ..nodes import Mark
from .frame_map import FrameMap
from .lds_buffers import LdsBufferIds
from .potential_waits import PotentialWaits
from .wait_common import (WaitCountSet, WaitSite, edge_domain, feasible_domains,
                          merge_relations, predecessors, trip_domain, wait_relation)


def _consume_issue(info, counter, acc, budget):
    if info is None or info.counter != counter:
        return acc, budget, False
    acc += info.width
    if budget is None:
        return acc, budget, False
    budget -= info.width
    return acc, budget, budget <= 0


def potential_ranks(prog, fm, potential_waits, potential, decisions=(), feasible=None):
    """Ranks on paths where ``potential.producer`` remains in flight.

    ``decisions`` is a static ``{(block, slot, counter): residual}`` map.  Encountering one while
    walking backward restricts the older search to the suffix that wait retained.
    """
    decisions = dict(decisions)
    pred = predecessors(fm)
    feasible = feasible if feasible is not None else feasible_domains(prog, fm)
    source = potential.producer
    start = (potential.block, potential.frame)
    initial_domain = trip_domain(prog)
    # node, cursor, accumulated rank, retained older-instruction budget, T-domain, first anchor
    stack = [(start, potential.pos, 0, None, initial_domain, True)]
    seen = {}
    ranks = []

    while stack:
        node, cursor, acc, budget, domain, first = stack.pop()
        domain = frozenset(domain) & feasible.get(node, frozenset())
        if not domain:
            continue
        state = (node, cursor, budget, domain)
        if not first and state in seen and seen[state] <= acc:
            continue
        seen[state] = min(acc, seen.get(state, acc))

        block = prog.block(node[0])
        frame = node[1]
        retired = False
        while cursor >= 0:
            site = (node[0], cursor, potential.counter)
            if not (first and cursor == potential.pos) and site in decisions:
                n = int(decisions[site])
                budget = n if budget is None else min(budget, n)
                if budget <= 0:
                    retired = True
                    break
            first = False
            if cursor == 0:
                break

            pos = cursor - 1
            if (node[0] == source.block and frame == source.frame and pos == source.pos):
                if budget is None or budget > 0:
                    ranks.append(acc)
                retired = True
                break

            info = potential_waits.issue(node[0], pos, frame)
            acc, budget, retired = _consume_issue(info, potential.counter, acc, budget)
            if retired:
                break
            cursor -= 1

        if retired:
            continue
        for pnode in pred.get(node, ()):
            incoming = edge_domain(prog, pnode[0], node[0], domain)
            if incoming:
                stack.append((pnode, len(prog.block(pnode[0]).body), acc,
                              budget, incoming, False))
    return tuple(sorted(set(ranks)))


def _rpo_index(prog):
    return {block.label: i for i, block in enumerate(prog.walk_rpo())}


def _existing_decisions(prog):
    out = {}
    for block in prog.blocks.values():
        for pos, node in enumerate(block.body):
            if not isinstance(node, Mark) or node.kind != "waitcnt":
                continue
            for counter in ("tensorcnt", "dscnt"):
                if counter in (node.at or {}):
                    out[(block.label, pos, counter)] = int(node.at[counter])
    return out


def _direct_decisions(prog, fm, potentials, feasible=None, fixed=()):
    values = {}
    fixed = dict(fixed)
    for potential in potentials:
        if prog.block(potential.block).model_only:
            continue
        ranks = potential_ranks(
            prog, fm, potentials, potential, fixed, feasible=feasible)
        if not ranks:
            continue
        n = min(ranks)
        site = potential.site
        values[site] = min(n, values.get(site, n))
    return values


def _essential_decisions(prog, fm, potentials, direct, feasible=None, fixed=()):
    """Resolve all prior-wait retention over the bounded frame state."""
    decisions = dict(direct)
    fixed = dict(fixed)
    order = _rpo_index(prog)
    sites = sorted(direct, key=lambda s: (order.get(s[0], len(order)), s[1], s[2]))
    for _ in range(len(fm.nodes()) + 1):
        resolved = {}
        for site in sites:
            trial = dict(fixed)
            trial.update(decisions)
            trial.pop(site, None)
            ranks = [rank for potential in potentials.for_site(site)
                     for rank in potential_ranks(
                         prog, fm, potentials, potential, trial, feasible=feasible)]
            if ranks:
                resolved[site] = min(ranks)
        decisions = resolved
    return decisions


def materialize_sites(prog, fm, storage, potentials, decisions, feasible=None,
                      ranking_decisions=None, promotions=None):
    """Attach per-frame diagnostic relations to a static decision map."""
    charged = {}
    ranking_decisions = decisions if ranking_decisions is None else ranking_decisions
    promotions = promotions or {}
    for site, fallback in sorted(decisions.items()):
        ranked = list(promotions.get(site, ()))
        if not ranked:
            for potential in potentials.for_site(site):
                ranks = potential_ranks(
                    prog, fm, potentials, potential, ranking_decisions, feasible=feasible)
                for rank in ranks:
                    ranked.append((rank, potential))
        if not ranked:
            continue
        n = min(rank for rank, _potential in ranked)
        if site not in promotions and n != int(fallback):
            raise RuntimeError(
                "BackwardWaitCounts disagrees at %s: plan=%d backward=%d"
                % (site, int(fallback), n))
        relations = merge_relations(*[
            (wait_relation(potential, rank, fm, storage),)
            for rank, potential in ranked])
        witness = min(ranked, key=lambda item: (
            item[0], item[1].hazard.producer.operand,
            item[1].producer.block, item[1].producer.pos))[1]
        charged[site] = WaitSite(
            block=site[0], pos=site[1], counter=site[2], n=n,
            kind=witness.hazard.kind,
            producer=witness.hazard.producer.operand,
            frames=len(ranked), relations=relations)
    return WaitCountSet(charged)


class BackwardWaitCounts(Analysis):
    """Corrected backward solver with retained-window semantics."""

    def run(self, prog, am):
        fm = am.get(FrameMap(), prog)
        storage = am.get(LdsBufferIds(), prog)
        potentials = am.get(PotentialWaits(), prog)
        if potentials.unresolved:
            raise RuntimeError(
                "BackwardWaitCounts: unresolved shared frame identity; refusing to infer no wait "
                "for %s" % (potentials.unresolved[:3],))
        fixed = _existing_decisions(prog)
        feasible = feasible_domains(prog, fm)
        from .counter_flow import derive_counterflow_plan
        decisions, trace, _observed = derive_counterflow_plan(prog, fm, potentials)
        all_decisions = dict(fixed)
        all_decisions.update(decisions)
        return materialize_sites(prog, fm, storage, potentials, decisions, feasible,
                                 ranking_decisions=all_decisions,
                                 promotions=trace.promotions)

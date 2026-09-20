# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Forward TENSORCNT/DSCNT flow over GIR's finite ring-frame state."""

from __future__ import annotations

from dataclasses import dataclass

from ..analysis import Analysis
from ..nodes import Mark
from .frame_map import FrameMap
from .lds_buffers import LdsBufferIds
from .potential_waits import PotentialWaits, issue_occurrence
from .wait_common import (DSCNT, FANOUT_META, TENSORCNT, WaitCountSet, WaitSite,
                          edge_domain, issued, merge_relations, trip_domain, wait_relation)


@dataclass(frozen=True)
class _CounterState:
    """Finite weighted FIFO of frame-issue identities, one tuple per counter."""
    tensor: tuple = ()
    ds: tuple = ()

    def queue(self, counter):
        return self.tensor if counter == TENSORCNT else self.ds

    def replace(self, counter, queue):
        queue = tuple(queue)
        return (_CounterState(queue, self.ds) if counter == TENSORCNT
                else _CounterState(self.tensor, queue))

    def issue(self, info, occurrence):
        # `(block, frame, position)` is a complete dynamic identity in GIR's ring quotient.  A
        # repeated occurrence replaces its prior queue copy rather than manufacturing a trip id.
        queue = tuple(entry for entry in self.queue(info.counter) if entry[0] != occurrence)
        queue += ((occurrence, info.width),)
        return self.replace(info.counter, queue)

    def wait(self, counter, n):
        queue = self.queue(counter)
        if n <= 0:
            return self.replace(counter, ())
        kept = []
        remaining = n
        for occurrence, width in reversed(queue):
            if remaining <= 0:
                break
            take = min(width, remaining)
            kept.append((occurrence, take))
            remaining -= take
        return self.replace(counter, reversed(kept))

    def age(self, producer, potentials):
        info = potentials.issue(producer.block, producer.pos, producer.frame)
        if info is None:
            return None
        occurrence = (producer.block, producer.pos, producer.frame)
        queue = self.queue(producer.counter)
        for pos, (candidate, _width) in enumerate(queue):
            if candidate == occurrence:
                return sum(width for _later, width in queue[pos + 1:])
        return None

    def required_ages(self, counter, required):
        """Ages of required occurrences present in this queue, in one reverse scan."""
        out = {}
        newer = 0
        for occurrence, width in reversed(self.queue(counter)):
            key = (counter,) + occurrence
            if key in required:
                out[key] = newer
            newer += width
        return out


@dataclass(frozen=True)
class FlowViolation:
    site: tuple
    producer: object
    age: int
    emitted: object


@dataclass
class CounterFlowTrace:
    """One fixed-plan finite-state traversal, also used by the independent checker."""
    requirements: dict
    requirements_by_pred: dict
    violations: tuple
    states: int
    promotions: dict = None

    @property
    def safe(self):
        return not self.violations

    def ages_at(self, site):
        return tuple(sorted(self.requirements.get(tuple(site), ())))


def _existing_wait(node):
    if not isinstance(node, Mark) or node.kind != "waitcnt":
        return ()
    out = []
    for counter in (TENSORCNT, DSCNT):
        if counter in (node.at or {}):
            out.append((counter, int(node.at[counter])))
    return tuple(out)


def simulate_counterflow(prog, fm, potentials, decisions):
    """Enumerate all canonical frame/counter states under one static wait plan."""
    decisions = dict(decisions)
    requirements = {}
    requirements_by_pred = {}
    observed = {}
    observed_by_pred = {}
    violations = []
    initial = trip_domain(prog)
    entries = fm.frames(prog.entry)
    queue = [((prog.entry, frame), _CounterState(), initial, None) for frame in entries]
    seen = set()

    while queue:
        node, state, domain, predecessor = queue.pop()
        canonical = (node, state, domain, predecessor)
        if canonical in seen:
            continue
        seen.add(canonical)
        label, frame = node
        block = prog.block(label)

        positions = () if block.model_only else range(len(block.body) + 1)
        for pos in positions:
            # A static wait executes in every frame that enters this block.  Requirements are
            # frame-specific and absent producers impose no constraint.
            for counter in (TENSORCNT, DSCNT):
                site = (label, pos, counter)
                emitted = decisions.get(site)
                required = potentials.producer_map_at(label, pos, counter, frame)
                for occurrence, age in state.required_ages(counter, required).items():
                    producer = required[occurrence]
                    requirements.setdefault(site, []).append(age)
                    requirements_by_pred.setdefault((site, predecessor), []).append(age)
                    observed.setdefault((site, issue_occurrence(producer)), []).append(age)
                    observed_by_pred.setdefault(
                        (site, predecessor, issue_occurrence(producer)), []).append(age)
                    if emitted is None or emitted > age:
                        violations.append(FlowViolation(site, producer, age, emitted))
                if emitted is not None:
                    state = state.wait(counter, int(emitted))

            if pos == len(block.body):
                break
            body_node = block.body[pos]
            for counter, n in _existing_wait(body_node):
                state = state.wait(counter, n)
            info = potentials.issue(label, pos, frame)
            if info is not None:
                state = state.issue(info, (label, pos, frame))

        for succ in fm.successors(node):
            incoming = edge_domain(prog, label, succ[0], domain)
            if incoming:
                queue.append((succ, state, incoming, label))

    return CounterFlowTrace(
        requirements={site: tuple(values) for site, values in requirements.items()},
        requirements_by_pred={
            key: tuple(values) for key, values in requirements_by_pred.items()},
        violations=tuple(violations), states=len(seen)), (observed, observed_by_pred)


def _predecessor_promotions(prog, potentials, decisions, trace, observed_by_pred):
    counts = (prog.meta or {}).get(FANOUT_META) or {}
    promoted, evidence = {}, {}
    for site, current in sorted(decisions.items()):
        block, pos, counter = site
        per_pred = {
            pred: min(ages) for (at, pred), ages in trace.requirements_by_pred.items()
            if at == site and pred is not None and ages
        }
        if len(per_pred) < 2:
            continue
        common = max(per_pred.values())
        if common <= current:
            continue
        prefix = issued(prog.block(block).body, 0, pos, counter, counts)
        if any(counter in (node.at or {}) for node in prog.block(block).body[:pos]
               if isinstance(node, Mark) and node.kind == "waitcnt"):
            continue
        by_occurrence = {}
        for potential in potentials.for_site(site):
            by_occurrence.setdefault(issue_occurrence(potential.producer), []).append(potential)
        local = {}
        local_evidence = {}
        for pred, need in per_pred.items():
            if need >= common:
                continue
            pred_block = prog.block(pred)
            if pred_block.model_only or tuple(pred_block.succs) != (block,):
                local = {}
                break
            edge_site = (pred, len(pred_block.body), counter)
            ranked = []
            for (at, via, occurrence), ages in observed_by_pred.items():
                if at != site or via != pred:
                    continue
                for age in ages:
                    if age >= common:
                        continue
                    edge_age = age - prefix
                    if edge_age < 0:
                        local = {}
                        break
                    ranked.extend((edge_age, potential)
                                  for potential in by_occurrence.get(occurrence, ()))
            if not ranked:
                local = {}
                break
            local[edge_site] = min(rank for rank, _potential in ranked)
            local_evidence[edge_site] = tuple(ranked)
        if not local:
            continue
        promoted[site] = common
        promoted.update(local)
        evidence.update(local_evidence)
    return promoted, evidence


def derive_counterflow_plan(prog, fm, potentials):
    """Choose waits by a statically bounded number of finite frame-flow rounds."""
    sites = [site for site in potentials.sites() if not prog.block(site[0]).model_only]
    decisions = {site: 0 for site in sites}

    # One round carries the effect of the current potential waits across every canonical state.
    # `FrameMap.nodes()` is the complete modulo-ring state set, so that many rounds plus the seed
    # round exposes every wait-to-wait relation without a runtime convergence/iteration limit.
    for _ in range(len(fm.nodes()) + 1):
        trace, _observed = simulate_counterflow(prog, fm, potentials, decisions)
        decisions = {site: min(ages) for site, ages in trace.requirements.items()}

    compensation = {}
    promotion_evidence = {}
    join_sites = [site for site in sites if len(prog.block(site[0]).preds) > 1]
    for _ in range(len(join_sites) + 1):
        trace, (_observed, observed_by_pred) = simulate_counterflow(
            prog, fm, potentials, decisions)
        promoted, evidence = _predecessor_promotions(
            prog, potentials, decisions, trace, observed_by_pred)
        changed = False
        for site, n in promoted.items():
            if site in decisions and site not in compensation:
                if decisions[site] != n:
                    decisions[site] = n
                    changed = True
            else:
                prior = compensation.get(site)
                value = n if prior is None else min(prior, n)
                if prior != value:
                    compensation[site] = value
                    decisions[site] = value
                    changed = True
        for site, ranked in evidence.items():
            promotion_evidence[site] = (
                promotion_evidence.get(site, ()) + tuple(ranked))
        if not changed:
            break
        for _ in range(len(fm.nodes()) + 1):
            trace, _observed = simulate_counterflow(
                prog, fm, potentials, decisions)
            decisions = {
                **{site: min(ages) for site, ages in trace.requirements.items()},
                **compensation,
            }
    final, (observed, _observed_by_pred) = simulate_counterflow(
        prog, fm, potentials, decisions)
    final.promotions = promotion_evidence
    derived = {site: min(ages) for site, ages in final.requirements.items()}
    ordinary = {site: n for site, n in decisions.items() if site not in compensation}
    if derived != ordinary:
        raise RuntimeError(
            "CounterFlow frame bound did not close the static wait plan: %d sites became %d"
            % (len(ordinary), len(derived)))
    if not final.safe:
        first = final.violations[0]
        raise RuntimeError(
            "CounterFlow produced an unsafe plan at %s: producer age %d, emitted %r"
            % (first.site, first.age, first.emitted))
    return decisions, final, observed


def _materialize(prog, fm, storage, potentials, decisions, observed, promotions=None):
    charged = {}
    promotions = promotions or {}
    for site, n in sorted(decisions.items()):
        ranked = list(promotions.get(site, ()))
        if not ranked:
            for potential in potentials.for_site(site):
                for rank in observed.get((site, issue_occurrence(potential.producer)), ()):
                    ranked.append((rank, potential))
        if not ranked:
            continue
        relations = merge_relations(*[
            (wait_relation(potential, rank, fm, storage),)
            for rank, potential in ranked])
        witness = min(ranked, key=lambda item: (
            item[0], item[1].hazard.producer.operand,
            item[1].producer.block, item[1].producer.pos))[1]
        charged[site] = WaitSite(
            block=site[0], pos=site[1], counter=site[2], n=int(n),
            kind=witness.hazard.kind,
            producer=witness.hazard.producer.operand,
            frames=len(ranked), relations=relations)
    return WaitCountSet(charged)


class CounterFlow(Analysis):
    """Forward solver for both GIR completion counters."""

    def run(self, prog, am):
        fm = am.get(FrameMap(), prog)
        storage = am.get(LdsBufferIds(), prog)
        potentials = am.get(PotentialWaits(), prog)
        if potentials.unresolved:
            raise RuntimeError(
                "CounterFlow: unresolved shared frame identity; refusing to infer no wait for %s"
                % (potentials.unresolved[:3],))
        decisions, trace, observed = derive_counterflow_plan(prog, fm, potentials)
        return _materialize(
            prog, fm, storage, potentials, decisions, observed, trace.promotions)

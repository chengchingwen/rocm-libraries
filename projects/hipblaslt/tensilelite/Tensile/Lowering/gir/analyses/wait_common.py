# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Shared completion-counter vocabulary and result types.

GIR keeps logical counter ranks exact.  Hardware-field legalization belongs downstream.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..coverage import plan_coverage
from ..emit_plan import plan_block
from ..nodes import CondChain, CondGoto, Goto, LoopBack, Move


TENSORCNT = "tensorcnt"   # global -> shared
DSCNT = "dscnt"           # shared -> register

#: `Program.meta` key: `{operand: read instructions one fill issues}`, from Fragment.
FANOUT_META = "read_instructions"


def counter_of(move):
    """The counter this Move increments, or ``None``."""
    dst = move.dsts[0].tile.space if move.dsts else None
    src = move.srcs[0].tile.space if move.srcs else None
    if dst == "shared" and src == "global":
        return TENSORCNT
    if dst == "register" and src == "shared":
        return DSCNT
    return None


def fan_out(move, counts) -> int:
    """How many hardware instructions one counted GIR Move issues."""
    counter = counter_of(move)
    if counter == TENSORCNT:
        return 1
    if counter != DSCNT:
        raise RuntimeError("WaitCounts: asked to count a Move that increments no tracked counter")
    ref = move.dsts[0] if move.dsts else None
    operand = getattr(ref.tile, "operand", None) if ref is not None else None
    n = int((counts or {}).get(operand, 0) or 0)
    if n < 1:
        raise RuntimeError(
            "WaitCounts: no read-instruction count for operand %r (have %s).  The completion "
            "counter counts INSTRUCTIONS; without this the wait is a drain, not a wait."
            % (operand, sorted((counts or {}))))
    return n


def issued(body, lo, hi, counter, counts) -> int:
    """Count every instruction on ``counter`` in ``body[lo:hi]``."""
    total = 0
    for pos in range(max(0, lo), min(hi, len(body))):
        node = body[pos]
        if isinstance(node, Move) and counter_of(node) == counter:
            total += fan_out(node, counts)
    return total


def emitting_reads(prog, phase) -> int:
    """How many read acts of ``phase`` issue, by the emitter's coverage plan."""
    try:
        acts = plan_block(prog, phase)
    except Exception:
        return -1
    meta = prog.meta or {}
    coverage = plan_coverage(
        acts, lambda op: (meta.get("read_quantum", {}) or {}).get(op),
        folded_of=lambda op: bool((meta.get("read_fold", {}) or {}).get(op))
        or bool((meta.get("coverage_axes", {}) or {}).get(op)))
    out = 0
    for i, act in enumerate(acts):
        if act.kind != "read":
            continue
        decision = coverage.decision(i)
        if decision is None or getattr(decision, "emit", True):
            out += 1
    return out


@dataclass(frozen=True)
class WaitRelation:
    """One frame-specific completion relation retained for diagnostics."""
    kind: str
    counter: str
    producer: tuple
    consumer: tuple
    gap: int
    scope: str
    rank: int

    def dump(self):
        keys = ("block", "pos", "operand", "space", "frame", "ring",
                "gdelta", "absolute", "token", "regions", "register", "advance")
        return {
            "kind": self.kind, "counter": self.counter, "rank": self.rank,
            "gap": self.gap, "scope": self.scope,
            "producer": dict(zip(keys, self.producer)),
            "consumer": dict(zip(keys, self.consumer)),
        }


@dataclass(frozen=True)
class WaitSite:
    """One static wait site on one shared hardware counter."""
    block: str
    pos: int
    counter: str
    n: int
    kind: str
    producer: str
    frames: int
    tokens: tuple = ()
    order_tokens: tuple = ()
    relations: tuple = ()

    @property
    def is_drain(self) -> bool:
        return self.n == 0


class WaitCountSet:
    """Query API over residuals keyed by ``(block, position, counter)``."""

    def __init__(self, charged=()):
        self._charged = dict(charged)

    def __len__(self):
        return len(self._charged)

    def __iter__(self):
        return iter(sorted(self._charged.values(),
                           key=lambda site: (site.block, site.pos, site.counter)))

    def at(self, block, pos):
        return {counter: site for (label, p, counter), site in self._charged.items()
                if label == block and p == pos}

    def drains(self):
        return tuple(site for site in self if site.is_drain)

    def essential(self):
        """CounterFlow already accounts for prior waits; every returned site is essential."""
        return self

    def as_dict(self):
        return dict(self._charged)


def _access_relation(touch, frame, fm, storage):
    ref = touch.ref
    is_register = hasattr(touch, "ring_pos")
    if is_register:
        ring = max(1, int(getattr(ref, "reg_ring", 1) or 1))
        relation = int(getattr(touch.inst, "advance", 0) or 0)
        return (touch.block, touch.pos, touch.operand, ref.tile.space, None, ring,
                relation, False, touch.ring_pos[-2], (), touch.ring_pos, relation)
    ring = max(1, int(storage.depth_of(ref.tile.operand)))
    absolute = getattr(ref, "abs_gen", None) is not None
    relation = int(ref.abs_gen) if absolute else int(getattr(ref, "gdelta", 0) or 0)
    token = fm.generation(ref, frame)
    return (touch.block, touch.pos, touch.operand, ref.tile.space, fm.render(frame), ring,
            relation, absolute, token,
            tuple(tuple(sorted(values)) for values in getattr(touch, "regions", ())), (),
            int(getattr(touch.inst, "advance", 0) or 0))


def wait_relation(potential, rank, fm, storage):
    """Resolve one PotentialWait's diagnostic relation at ``rank``."""
    return WaitRelation(
        kind=potential.hazard.kind, counter=potential.counter,
        producer=_access_relation(potential.hazard.producer, potential.producer_frame, fm, storage),
        consumer=_access_relation(potential.hazard.consumer, potential.consumer_frame, fm, storage),
        gap=int(potential.hazard.gap),
        scope="block" if getattr(potential.hazard, "cross_agent", False) else "wave",
        rank=int(rank),
    )


def merge_relations(*groups):
    by_value = {relation: relation for group in groups for relation in (group or ())}
    return tuple(sorted(by_value.values(), key=repr))


def is_register_fill(hazard, prog):
    body = prog.block(hazard.producer.block).body
    node = body[hazard.producer.pos] if hazard.producer.pos < len(body) else None
    return isinstance(node, Move) and counter_of(node) == DSCNT


def counter_for(hazard, prog):
    """Counter that must retire the hazard's vacating/producing operation."""
    body = prog.block(hazard.producer.block).body
    node = body[hazard.producer.pos] if hazard.producer.pos < len(body) else None
    if hazard.kind == "RAW":
        return counter_of(node) if isinstance(node, Move) else None
    if isinstance(node, Move) and counter_of(node) == DSCNT:
        return DSCNT
    consumer_body = prog.block(hazard.consumer.block).body
    reader = (consumer_body[hazard.consumer.pos]
              if hazard.consumer.pos < len(consumer_body) else None)
    return DSCNT if isinstance(reader, Move) and counter_of(reader) == DSCNT else TENSORCNT


def predecessors(fm):
    back = {}
    for node in fm.nodes():
        for succ in fm.successors(node):
            back.setdefault(succ, []).append(node)
    return {key: tuple(value) for key, value in back.items()}


def _pred_value(pred, trip):
    if pred.lhs != "T":
        return None
    rhs = int(pred.rhs.const) + (trip if pred.rhs.var == "T" else 0)
    return {
        "==": trip == rhs, "!=": trip != rhs, "<": trip < rhs,
        "<=": trip <= rhs, ">": trip > rhs, ">=": trip >= rhs,
    }[pred.op]


def edge_allows(term, target, trip):
    if isinstance(term, Goto):
        return term.target == target
    if isinstance(term, CondGoto):
        value = _pred_value(term.pred, trip)
        return (target in (term.t_target, term.f_target) if value is None
                else target == (term.t_target if value else term.f_target))
    if isinstance(term, CondChain):
        for pred, arm in term.arms:
            value = _pred_value(pred, trip)
            if value is None:
                return target in tuple(t for _p, t in term.arms) + (term.default,)
            if value:
                return target == arm
        return target == term.default
    if isinstance(term, LoopBack):
        return target in (term.body, term.exit_target)
    return False


def trip_domain(prog):
    """One representative of each scaffold ``T`` truth class."""
    limit = max(4, int((prog.meta or {}).get("M", 0) or 0) + 3)
    return frozenset(range(1, limit + 1))


def edge_domain(prog, src, dst, domain):
    term = prog.block(src).term
    return frozenset(trip for trip in domain if edge_allows(term, dst, trip))


def feasible_domains(prog, fm):
    """Representative trip values that can reach each frame node from program entry."""
    values = {node: set() for node in fm.nodes()}
    queue = [((prog.entry, frame), trip)
             for frame in fm.frames(prog.entry) for trip in trip_domain(prog)]
    seen = set()
    while queue:
        node, trip = queue.pop()
        if (node, trip) in seen:
            continue
        seen.add((node, trip))
        values.setdefault(node, set()).add(trip)
        for succ in fm.successors(node):
            if edge_allows(prog.block(node[0]).term, succ[0], trip):
                queue.append((succ, trip))
    return {node: frozenset(domain) for node, domain in values.items()}

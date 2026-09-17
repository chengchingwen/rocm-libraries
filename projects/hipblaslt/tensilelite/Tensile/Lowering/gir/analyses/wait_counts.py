# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The completion-counter residual each hazard consumer owes, per counter class.

`n` is a RANK in the counter's FIFO, counted between `(block, frame)` endpoints -- a rotation
advances the frame without moving in the block CFG.  Rounds DOWN: a site takes the minimum.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from ..analysis import Analysis
from ..nodes import Mark, Mma, Move
from .dep_tokens import DependenceTokens
from .fence_regions import fences_of, separating
from .frame_hazards import FrameHazards
from .frame_map import FrameMap
from .reg_hazards import RegHazards
from ..coverage import plan_coverage
from ..emit_plan import plan_block
from ..loop_wait import (
    LoopWaitCounter, LoopWaitDependencyField, LoopWaitDependencyKind, LoopWaitScope,
)

TENSORCNT = "tensorcnt"   # global -> shared
DSCNT = "dscnt"           # shared -> register

# Values in stinkytofu::waitcnt::CounterKind and the LoopWaitData transport schema.
_COUNTER_IDS = {DSCNT: int(LoopWaitCounter.DS), TENSORCNT: int(LoopWaitCounter.TENSOR)}
_KIND_IDS = {
    "RAW": int(LoopWaitDependencyKind.RAW),
    "WAR": int(LoopWaitDependencyKind.WAR),
    "WAW": int(LoopWaitDependencyKind.WAW),
}

#: `Program.meta` key: `{operand: read instructions one fill issues}`, from Fragment.
FANOUT_META = "read_instructions"


def counter_of(move) -> str:
    """The counter this Move's instructions increment, or None."""
    dst = move.dsts[0].tile.space if move.dsts else None
    src = move.srcs[0].tile.space if move.srcs else None
    if dst == "shared" and src == "global":
        return TENSORCNT
    if dst == "register" and src == "shared":
        return DSCNT
    return None


def fan_out(move, counts) -> int:
    """Instructions this Move becomes.  There is no unknown: GIR carries the count for every read
    operand, and a Move counted as zero would make every residual over it a full drain."""
    if counter_of(move) == TENSORCNT:
        return 1
    ref = move.dsts[0] if move.dsts else None
    operand = getattr(ref.tile, "operand", None) if ref is not None else None
    n = int((counts or {}).get(operand, 0) or 0)
    if n < 1:
        raise RuntimeError(
            "WaitCounts: no read-instruction count for operand %r (have %s).  The completion "
            "counter counts INSTRUCTIONS; without this the wait is a drain, not a wait."
            % (operand, sorted((counts or {}))))
    return n


def emitting_reads(prog, phase) -> int:
    """How many read acts of `phase` issue an instruction, by the emitter's own coverage plan."""
    try:
        acts = plan_block(prog, phase)
    except Exception:
        return -1
    meta = prog.meta or {}
    plan = plan_coverage(
        acts, lambda op: (meta.get("read_quantum", {}) or {}).get(op),
        folded_of=lambda op: bool((meta.get("read_fold", {}) or {}).get(op))
                          or bool((meta.get("coverage_axes", {}) or {}).get(op)))
    out = 0
    for i, act in enumerate(acts):
        if act.kind != "read":
            continue
        d = plan.decision(i)
        if d is None or getattr(d, "emit", True):
            out += 1
    return out


@dataclass(frozen=True)
class WaitDependency:
    """One explicit frame-hazard relation, independent of its temporary numeric rank."""
    producer_op_id:    int
    consumer_op_id:    int
    producer_frame_id: int
    consumer_frame_id: int
    generation_gap:    int
    counter:           int
    kind:              int
    scope:             int

    def flatten(self) -> tuple:
        record = [0] * int(LoopWaitDependencyField.COUNT)
        record[LoopWaitDependencyField.PRODUCER_OP_ID] = self.producer_op_id
        record[LoopWaitDependencyField.CONSUMER_OP_ID] = self.consumer_op_id
        record[LoopWaitDependencyField.PRODUCER_FRAME_ID] = self.producer_frame_id
        record[LoopWaitDependencyField.CONSUMER_FRAME_ID] = self.consumer_frame_id
        record[LoopWaitDependencyField.GENERATION_GAP] = self.generation_gap
        record[LoopWaitDependencyField.COUNTER] = self.counter
        record[LoopWaitDependencyField.KIND] = self.kind
        record[LoopWaitDependencyField.SCOPE] = self.scope
        return tuple(record)


@dataclass(frozen=True)
class WaitSite:
    """One consumer and the residual it owes on one counter class."""
    block:    str
    pos:      int
    counter:  str
    n:        int          # 0 = full drain
    kind:     str
    producer: str
    frames:   int
    #: Storage the wait WAITS FOR (RAW), and storage it merely ORDERS (WAR) -- the same split
    #: `_emit_fence` makes between `memoryToken` and `orderToken`.
    tokens:       tuple = ()
    order_tokens: tuple = ()
    #: Frame-hazard relations carried to StinkyTofu. Register hazards deliberately stay numeric
    #: here because register SSA reconstructs dscnt dependencies after scheduling.
    dependencies: tuple = ()

    @property
    def is_drain(self) -> bool:
        return self.n == 0


def _issued(body, lo, hi, counter, regs) -> int:
    total = 0
    for pos in range(max(0, lo), min(hi, len(body))):
        node = body[pos]
        if isinstance(node, Move) and counter_of(node) == counter:
            total += fan_out(node, regs)
    return total




class _Paths:
    """Whole-block cost between `(block, frame)` nodes, memoised per program.

    The walk state is FIXED SIZE -- one phase per ring -- so `(block, frame)` enumerates it all.
    Counting is exact: a FIFO retires in order, so nothing younger than the producer leaves first
    and no intervening wait changes how many sit between the pair."""

    def __init__(self, prog, fm, regs):
        self._prog, self._fm, self._regs = prog, fm, regs
        self._whole, self._reach = {}, {}

    def frames(self, block):
        return self._fm.frames(block)

    def whole(self, label, counter) -> int:
        """Same-class instructions one execution of `label` issues; a `model_only` block none."""
        key = (label, counter)
        if key not in self._whole:
            blk = self._prog.block(label)
            self._whole[key] = (0 if getattr(blk, "model_only", False)
                                else _issued(blk.body, 0, len(blk.body), counter, self._regs))
        return self._whole[key]

    def between(self, start, counter, steps):
        """Cheapest whole-block cost to each node, endpoints excluded; `steps` None = any length."""
        key = (start, counter, steps)
        if key not in self._reach:
            self._reach[key] = (self._exact(start, counter, steps) if steps is not None
                                else self._any(start, counter))
        return self._reach[key]

    def _exact(self, start, counter, steps):
        layer, first = {start: 0}, True
        for _ in range(steps):
            nxt = {}
            for node, cost in layer.items():
                out = cost + (0 if first else self.whole(node[0], counter))
                for succ in self._fm.successors(node):
                    if succ not in nxt or out < nxt[succ]:
                        nxt[succ] = out
            layer, first = nxt, False
        return layer

    def _any(self, start, counter):
        """Cheapest cost per node.  Bounded by the state set: a longer walk only repeats a state."""
        best, layer, first = {}, {start: 0}, True
        for _ in range(len(self._fm.nodes())):
            nxt = {}
            for node, cost in layer.items():
                out = cost + (0 if first else self.whole(node[0], counter))
                for succ in self._fm.successors(node):
                    if succ not in nxt or out < nxt[succ]:
                        nxt[succ] = out
                    if succ not in best or out < best[succ]:
                        best[succ] = out
            layer, first = nxt, False
            if not layer:
                break
        return best


def _predecessors(fm):
    """Reverse of the frame graph: `{node: (node, ...)}`.  A backward walk needs the edges the
    forward one supplies, negated -- stepping back across one is the advance undone."""
    back = {}
    for node in fm.nodes():
        for succ in fm.successors(node):
            back.setdefault(succ, []).append(node)
    return {k: tuple(v) for k, v in back.items()}


def _fills(fm, body, pos, frame, operand, slots) -> bool:
    """Does the instruction at `pos` fill the consumer's storage in `frame`?

    A Phi-fused copy is ONE instruction over several members, so its members must all be offered --
    but only the member of the CONSUMER'S OPERAND can carry its data.  Matching on any member makes
    an MXSA fill answer for an MXSB read, which stops the walk early and inflates the count."""
    inst = body[pos] if pos < len(body) else None
    for ref in tuple(getattr(inst, "dsts", ()) or ()):
        if (ref.tile.space == "shared" and ref.tile.operand == operand
                and (fm.touches(ref, frame) & slots)):
            return True
    return False


def _names(prog, block, pos):
    """The storage a fence at `(block, pos)` pins: what it waits for, plus what it only orders."""
    body = prog.block(block).body
    node = body[pos] if pos < len(body) else None
    if not (isinstance(node, Mark) and node.kind == "fence"):
        return None
    return set(node.at.get("tokens") or ()) | set(node.at.get("order_tokens") or ())


def _anchor_site(prog, hazard, fences, toks=None):
    """Where the wait goes: `(block, pos, at_producer)`, before the FENCE on a cross-wave edge.

    A completion counter retires only this wave's own ops, so at the consumer it names a FIFO that
    never held the producing wave's op and proves nothing for any value.  The barrier publishes the
    write, so the wait must precede it -- the fence raised for THIS hazard, by `separating`."""
    if not getattr(hazard, "cross_agent", False):
        return hazard.consumer.block, hazard.consumer.pos, False
    sep = separating(hazard, fences, lambda lab: len(prog.block(lab).body) + 1)
    # `separating` answers POSITION -- it stands between the ends.  A barrier orders an access only
    # if it NAMES that access's storage, so a fence that does not is no anchor: the wait would sit
    # at a barrier the overwriting access may be hoisted across.
    want = set(toks.tokens_for(hazard.consumer.ref)) if toks is not None else set()
    if want:
        sep = [(b, i) for b, i in sep if (_names(prog, b, i) or set()) >= want] or sep
    # Latest of those: the one the consumer actually waits on, so the wait stays as late as
    # correctness allows.  Its own block's comes first -- a producer-block fence is the one-shot case.
    here = [i for b, i in sep if b == hazard.consumer.block]
    if here:
        return hazard.consumer.block, max(here), False
    there = [i for b, i in sep if b == hazard.producer.block]
    if there:
        return hazard.producer.block, max(there), True
    return hazard.consumer.block, hazard.consumer.pos, False


def _rank(prog, fm, preds, hazard, fp, fc, counter, regs, anchor):
    """In-flight count owed at `anchor`, by walking BACKWARD from it to the producer.

    Counts every issue of this counter class on the way -- ALL operands share one FIFO, so a TDM
    load of any tensor sits in it.  Stepping back across a frame edge undoes that edge's advance,
    so the frame is always the one that instance really ran in and the count is exact rather than
    a shortest-path estimate."""
    sblk, target = hazard.producer.block, hazard.producer.pos
    consumer_slots = fm.touches(hazard.consumer.ref, fc)
    c_operand = hazard.consumer.ref.tile.operand
    ablock, aframe, apos = anchor
    start = (ablock, aframe)
    best, seen = None, {}
    stack = [(start, apos, 0)]
    while stack:
        node, upto, acc = stack.pop()
        body = prog.block(node[0]).body
        # A source is an instance that COLLIDES: same block and position is not enough, the
        # storage `(frame + gdelta) % ring` resolves to must intersect what the consumer reads.
        # Two instances of one copy in different frames fill different generations.  Every
        # colliding instance is a source and the nearest binds, so take the minimum.  The producer
        # itself is never counted -- it is the entry the wait must RETIRE.
        if node[0] == sblk and target < upto and _fills(fm, body, target, node[1], c_operand, consumer_slots):
            n = acc + _issued(body, target + 1, upto, counter, regs)
            best = n if best is None else min(best, n)
            continue
        onward = acc + _issued(body, 0, upto, counter, regs)
        if node in seen and seen[node] <= onward:
            continue
        seen[node] = onward
        for pnode in preds.get(node, ()):
            stack.append((pnode, len(prog.block(pnode[0]).body), onward))
    return best


def _instance_ranks(prog, fm, preds, hazard, fp, fc, counter, regs, site):
    """A frame hazard names its instance; a register hazard names none, so every pair is walked."""
    block, pos, at_producer = site
    if fp is not None:
        return (_rank(prog, fm, preds, hazard, fp, fc, counter, regs,
                      (block, fp if at_producer else fc, pos)),)
    p, c = hazard.producer.block, hazard.consumer.block
    return tuple(_rank(prog, fm, preds, hazard, a, b, counter, regs,
                       (block, a if at_producer else b, pos))
                 for a in fm.frames(p) for b in fm.frames(c)) or (None,)


def _merge_dependencies(*groups):
    """Deterministic set-union of explicit dependency records."""
    by_record = {dep.flatten(): dep for group in groups for dep in (group or ())}
    return tuple(by_record[key] for key in sorted(by_record))


def _prune_redundant(prog, charged, regs):
    """Drop sites an earlier wait on the same counter already discharged, within a block."""
    keep, by_line = {}, {}
    for key, site in charged.items():
        by_line.setdefault((site.block, site.counter), []).append(site)
    for (block, counter), sites in by_line.items():
        body = prog.block(block).body
        forced, held = None, None
        for site in sorted(sites, key=lambda s: s.pos):
            need = _issued(body, 0, site.pos, counter, regs) - site.n
            if forced is not None and need <= forced:
                # The surviving wait discharges this hazard too, so it inherits the storage: a
                # dropped site's tokens move to the wait that covers it, or nothing orders them.
                if held is not None:
                    keep[held] = replace(
                        keep[held],
                        tokens=tuple(sorted(set(keep[held].tokens) | set(site.tokens))),
                        order_tokens=tuple(sorted(set(keep[held].order_tokens)
                                                  | set(site.order_tokens))),
                        dependencies=_merge_dependencies(
                            keep[held].dependencies, site.dependencies))
                continue
            forced = need if forced is None else max(forced, need)
            held = (site.block, site.pos, site.counter)
            keep[held] = site
    return keep


class WaitCounts(Analysis):
    """The residual every hazard consumer owes, keyed `(block, pos, counter)`.  Pure."""

    def run(self, prog, am):
        regs = (prog.meta or {}).get(FANOUT_META) or {}
        fm = am.get(FrameMap(), prog)
        frame_ids = {frame: i for i, frame in enumerate(
            sorted({frame for _block, frame in fm.nodes()}))}
        preds = _predecessors(fm)
        fences = fences_of(prog)
        toks = am.get(DependenceTokens(), prog)
        charged = {}
        # FrameHazards is LDS-only; the read->wmma RAW that gates a fragment is RegHazards'.
        # Each INSTANCE is charged, not each edge: the frames are what set the rank apart.
        edges = [(h, self._counter_for(h, prog), fp, fc)
                 for fp, fc, h in am.get(FrameHazards(), prog).instances()]
        edges += [(h, DSCNT, None, None) for h in am.get(RegHazards(), prog).edges(kind="RAW")
                  if self._is_register_fill(h, prog)]
        for hazard, counter, fp, fc in edges:
            if counter is None:
                continue
            site = _anchor_site(prog, hazard, fences, toks)
            ranks = [r for r in _instance_ranks(prog, fm, preds, hazard, fp, fc, counter, regs,
                                                site)
                     if r is not None]
            n = min(ranks) if ranks else 0
            key = (site[0], site[1], counter)
            prior = charged.get(key)
            # A RAW is WAITED FOR, anything else only ORDERED -- the split `_emit_fence` makes.
            # `_prune_redundant` hands a dropped site's storage to the wait that absorbs it.
            seen = set(toks.tokens_for(hazard.consumer.ref))
            isRaw = hazard.kind == "RAW"
            raw = tuple(sorted(set(prior.tokens if prior else ()) | (seen if isRaw else set())))
            war = tuple(sorted(set(prior.order_tokens if prior else ())
                               | (set() if isRaw else seen)))
            explicit = ()
            if fp is not None and fc is not None:
                explicit = (WaitDependency(
                    producer_op_id=int(getattr(hazard.producer.inst, "op_id", -1)),
                    consumer_op_id=int(getattr(hazard.consumer.inst, "op_id", -1)),
                    producer_frame_id=frame_ids[fp],
                    consumer_frame_id=frame_ids[fc],
                    generation_gap=int(hazard.gap),
                    counter=_COUNTER_IDS[counter],
                    kind=_KIND_IDS[hazard.kind],
                    scope=int(LoopWaitScope.WORKGROUP
                              if hazard.cross_agent else LoopWaitScope.WAVE)),)
            dependencies = _merge_dependencies(
                prior.dependencies if prior else (), explicit)
            if prior is None or n < prior.n:
                charged[key] = WaitSite(
                    block=site[0], pos=site[1], counter=counter, n=n,
                    kind=hazard.kind, producer=hazard.producer.operand,
                    frames=(prior.frames + 1) if prior else 1,
                    tokens=raw, order_tokens=war, dependencies=dependencies)
            else:
                charged[key] = WaitSite(
                    block=prior.block, pos=prior.pos, counter=prior.counter, n=prior.n,
                    kind=prior.kind, producer=prior.producer, frames=prior.frames + 1,
                    tokens=raw, order_tokens=war, dependencies=dependencies)
        return WaitCountSet(charged, lambda: _prune_redundant(prog, charged, regs))

    @staticmethod
    def _is_register_fill(hazard, prog):
        """A shared->register read feeding a consumer."""
        body = prog.block(hazard.producer.block).body
        node = body[hazard.producer.pos] if hazard.producer.pos < len(body) else None
        return isinstance(node, Move) and counter_of(node) == DSCNT

    @staticmethod
    def _counter_for(hazard, prog):
        """RAW discharges against the producer's class, WAR/WAW against the vacating read's."""
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


class WaitCountSet:
    """Query API over the charged residuals."""

    def __init__(self, charged, prune=None):
        self._charged = dict(charged)
        self._prune = prune

    def __len__(self):
        return len(self._charged)

    def __iter__(self):
        return iter(sorted(self._charged.values(), key=lambda s: (s.block, s.pos, s.counter)))

    def at(self, block, pos):
        return {counter: site for (blk, p, counter), site in self._charged.items()
                if blk == block and p == pos}

    def drains(self):
        return tuple(site for site in self if site.is_drain)

    def essential(self):
        """Sites not already discharged; the set keeps all of them, for reading."""
        return WaitCountSet(self._prune() if self._prune else self._charged)

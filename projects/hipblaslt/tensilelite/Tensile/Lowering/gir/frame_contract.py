# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The GIR frame facts that belong to NO instruction, and the parser for them.

An action's identity, kind, anchor and the shared buffers it touches ride on the instruction as
`mod.gir_action`; only the generation table and the phi edges -- facts about the loop, not about
any one instruction -- travel here.

The format is line-oriented `tag [positional] key=value ...`, readable and writeable by hand.
An absent optional key means its default, so there are no sentinel values to decode. The grammar
is in `docs/frame-contract.md`; `parse_contract` is the reference implementation and must stay in
step with `stinkytofu/src/analysis/asm/GirFrameAnalysis.cpp`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .analysis import AnalysisManager
from .analyses.frame_map import FrameMap
from .analyses.lds_buffers import LdsBufferIds
from .emit_plan import plan_program
from .nodes import Mark, Move


#: The module metadata slot the contract travels in.
CONTRACT_KEY = "gir.frame_contract"

#: Identifies the blob; NOT a version. The producer and the consumer ship together, so a contract
#: that does not match its parser is a build error, never something to negotiate at runtime.
CONTRACT_MARKER = "gir-frame-contract"

#: An operand name is written bare, so it must be a plain token. Escaping instead would put an
#: encode/decode pair in two languages to serve a name no kernel has ever produced.
_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")

_HAZARD_KINDS = {"RAW": ("copy", "read"), "WAR": ("read", "copy"), "WAW": ("copy", "copy")}


# --- the facts ---------------------------------------------------------------


@dataclass(frozen=True)
class Generation:
    """A rotating buffer: how deep it is, where it enters, how far each trip advances it."""
    id: int
    ring: int
    entry: int = 0
    advance: int = 0


@dataclass(frozen=True)
class Action:
    """One schedulable unit, and the action whose position its block is pinned to."""
    id: int
    kind: str
    anchor: int


@dataclass(frozen=True)
class SharedAccess:
    """One shared-memory touch an action makes."""
    action: int
    is_write: bool
    operand: str
    ring: int
    gen: int = -1              # -1 = not bound to a generation
    gdelta: int = 0
    absolute: int = -1         # -1 = the generation is relative, not pinned
    cross_agent: bool = False
    region: int = -1           # -1 = the whole operand


@dataclass(frozen=True)
class Edge:
    """A generation's phase arriving at `dst` from `src`.

    `relative` distinguishes the two ways GIR states it: a `transfer` forwards the predecessor's
    phase shifted by `delta`, an `incoming` pins it to an absolute `value`.
    """
    dst: int
    src: int
    gen: int
    value: int
    relative: bool


@dataclass(frozen=True)
class Relation:
    """A hazard the fence at `fence` separates, and how many generations lie between its ends."""
    fence: int
    kind: str
    producer: int
    consumer: int
    gap: int = 0


@dataclass
class Contract:
    generations: dict = field(default_factory=dict)
    actions: dict = field(default_factory=dict)
    accesses: list = field(default_factory=list)
    edges: list = field(default_factory=list)
    relations: list = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not (self.generations or self.actions or self.accesses
                    or self.edges or self.relations)


# --- deriving the facts from a program ---------------------------------------


def _shared_refs(action):
    """The shared-memory refs one action touches, each with whether it writes."""
    source = action.source
    if not isinstance(source, Move):
        return ()
    if action.kind == "read":
        return tuple((ref, False) for ref in source.srcs if ref.tile.space == "shared")
    if action.kind == "copy":
        return tuple((ref, True) for ref in source.dsts if ref.tile.space == "shared")
    return ()


def _generations(prog) -> dict:
    """Every rotating buffer the program names, with its ring, entry phase and per-trip advance."""
    facts = {}

    def ensure(gen, ring=None):
        if gen is None:
            return None
        depth = max(1, int(ring if ring is not None else gen.ring))
        fact = facts.setdefault(int(gen.id), {"ring": depth, "entry": None, "advance": None})
        if fact["ring"] != depth:
            raise RuntimeError("frame contract: inconsistent ring for Gen %d" % gen.id)
        return fact

    for block in prog.blocks.values():
        for phi in block.phis:
            fact = ensure(phi.gen)
            if phi.entry_val is None:
                continue
            value = int(phi.entry_val) % fact["ring"]
            if fact["entry"] not in (None, value):
                raise RuntimeError("frame contract: inconsistent phi entry for Gen %d" % phi.gen.id)
            fact["entry"] = value
        for xfer in block.xfers:
            fact = ensure(xfer.gen, xfer.ring)
            value = int(xfer.adv) % fact["ring"]
            if fact["advance"] not in (None, value):
                raise RuntimeError("frame contract: inconsistent transfer for Gen %d" % xfer.gen.id)
            fact["advance"] = value
        for inst in block.body:
            if isinstance(inst, Move):
                for ref in tuple(inst.srcs) + tuple(inst.dsts):
                    ensure(getattr(ref, "gen", None))

    return {gid: Generation(gid, fact["ring"], fact["entry"] or 0, fact["advance"] or 0)
            for gid, fact in facts.items()}


def _anchor_actions(prog, plans, label, first, seen=()):
    """The action ids that stand for `label` on an incoming (`first`) or outgoing edge.

    The FIRST action is a block's canonical identity: ST scheduling may reorder physical
    instructions, so an outgoing frame must never depend on whichever action is emitted last.
    An empty block forwards the question to its neighbours.
    """
    if label in seen:
        return ()
    actions = plans.get(label, ())
    if actions:
        return (actions[0].action_id,)
    neighbours = prog.blocks[label].succs if first else prog.blocks[label].preds
    out = set()
    for neighbour in neighbours:
        if neighbour in prog.blocks:
            out.update(_anchor_actions(prog, plans, neighbour, first, tuple(seen) + (label,)))
    return tuple(sorted(out))


def _forward_delta(frames, predecessor, destination, gen) -> int:
    """The affine phase transfer of one forwarding phi input.

    `None` in a GenPhi does not mean "no edge semantics": it forwards the value after GIR's edge
    transfer. Deriving that from FrameMap is what stops the serialized contract substituting ST's
    loop-edge convention for GIR's exact dataflow.
    """
    ring = max(1, int(gen.ring))
    deltas = {
        (target.of(gen.id) - source.of(gen.id)) % ring
        for source in frames.frames(predecessor)
        for label, target in frames.successors((predecessor, source))
        if label == destination
    }
    if len(deltas) != 1:
        raise RuntimeError(
            "frame contract: phi transfer %s->%s for Gen %d is not one affine delta: %r"
            % (predecessor, destination, gen.id, sorted(deltas)))
    return next(iter(deltas))


def _region_of(operand, regions):
    """The one region a frame instance selects, or -1 for the whole operand.

    `frame_region_instances_of` already enumerates one instance per physical agent, so an instance
    names a single region. A MAY-set would have to widen every consumer, so refuse instead."""
    axes = [tuple(sorted(axis)) for axis in (regions or ())]
    if not axes:
        return -1
    if len(axes) > 1 or len(axes[0]) != 1:
        raise RuntimeError(
            "frame contract: operand %r selects %r, but one instance must name one region. "
            "Widen the contract to a region set rather than dropping the extra values."
            % (operand, axes))
    return int(axes[0][0])


def _accesses_of(action, storage, distributed):
    """Every shared touch of one action, one `SharedAccess` per frame-region instance."""
    out = []
    for ref, is_write in _shared_refs(action):
        operand = str(ref.tile.operand)
        if not _TOKEN.match(operand):
            raise RuntimeError(
                "frame contract: operand name %r is not a bare token, so it cannot be written "
                "unescaped. Rename the operand rather than adding an escape to both parsers."
                % (operand,))
        gen = getattr(ref, "gen", None)
        absolute = getattr(ref, "abs_gen", None)
        for regions in storage.geometry.frame_region_instances_of(ref, is_write):
            out.append(SharedAccess(
                action=action.action_id,
                is_write=is_write,
                operand=operand,
                ring=max(1, int(storage.depth_of(ref.tile.operand))),
                gen=-1 if gen is None else int(gen.id),
                gdelta=int(getattr(ref, "gdelta", 0) or 0),
                absolute=-1 if absolute is None else int(absolute),
                cross_agent=bool(distributed.get(ref.tile.operand)),
                region=_region_of(operand, regions)))
    return out


def _edges_of(prog, plans, frames):
    """Every phi input, as the phase one block's generation receives from another."""
    out = set()
    for block in prog.blocks.values():
        destinations = _anchor_actions(prog, plans, block.label, True)
        for phi in block.phis:
            for pred, value in phi.incomings:
                sources = _anchor_actions(prog, plans, pred, False)
                if not sources or not destinations:
                    raise RuntimeError(
                        "frame contract: phi input %s->%s has no physical action anchor"
                        % (pred, block.label))
                relative = value is None
                phase = (_forward_delta(frames, pred, block.label, phi.gen) if relative
                         else int(value) % max(1, int(phi.gen.ring)))
                for source in sources:
                    for destination in destinations:
                        out.add(Edge(int(destination), int(source), int(phi.gen.id),
                                     phase, relative))
    return sorted(out, key=lambda e: (e.relative, e.dst, e.src, e.gen, e.value))


def _relations_of(actions, actions_by_source):
    """Every hazard a fence action separates, with both ends resolved to action ids."""
    def endpoint(identity, want_kind):
        candidates = [a for a in actions_by_source.get(int(identity), ()) if a.kind == want_kind]
        if len(candidates) != 1:
            raise RuntimeError("frame contract: relation endpoint %r maps to %d %s actions"
                               % (identity, len(candidates), want_kind))
        return candidates[0].action_id

    out = []
    for action in actions:
        source = action.source
        if action.kind != "fence" or not isinstance(source, Mark):
            continue
        for relation in source.at.get("relations") or ():
            kind = str(relation.get("kind"))
            if kind not in _HAZARD_KINDS:
                continue
            producer_kind, consumer_kind = _HAZARD_KINDS[kind]
            out.append(Relation(
                fence=action.action_id, kind=kind,
                producer=endpoint((relation.get("producer") or {}).get("identity"), producer_kind),
                consumer=endpoint((relation.get("consumer") or {}).get("identity"), consumer_kind),
                gap=int(relation.get("gap", 0) or 0)))
    return out


def build_contract(prog) -> Contract:
    """The frame facts of `prog`, with no GIR CFG topology in them."""
    analyses = AnalysisManager()
    storage = analyses.get(LdsBufferIds(), prog)
    if storage.unresolved:
        raise RuntimeError("frame contract: unresolved shared references %r"
                           % (storage.unresolved[:4],))
    frames = analyses.get(FrameMap(), prog)
    plans = plan_program(prog)

    actions = [action for block_actions in plans.values() for action in block_actions]
    anchors = {action.action_id: block_actions[0].action_id
               for block_actions in plans.values() if block_actions
               for action in block_actions}
    by_source = {}
    for action in actions:
        if action.source is not None:
            by_source.setdefault(id(action.source), []).append(action)

    distributed = prog.meta.get("agent_distributed", {}) or {}
    contract = Contract(generations=_generations(prog))
    for action in actions:
        contract.actions[action.action_id] = Action(
            action.action_id, action.kind, anchors[action.action_id])
        contract.accesses.extend(_accesses_of(action, storage, distributed))
    contract.edges = _edges_of(prog, plans, frames)
    contract.relations = _relations_of(actions, by_source)
    return contract


# --- text --------------------------------------------------------------------


def render_contract(contract) -> str:
    """The contract as text: deterministic, and the same thing `parse_contract` reads."""
    lines = [CONTRACT_MARKER, ""]
    for generation in sorted(contract.generations.values(), key=lambda g: g.id):
        lines.append("gen %d  ring=%d entry=%d advance=%d"
                     % (generation.id, generation.ring, generation.entry, generation.advance))

    if contract.edges:
        lines.append("")
    for edge in contract.edges:
        lines.append("%s  dst=%d src=%d gen=%d %s=%d"
                     % ("transfer" if edge.relative else "incoming",
                        edge.dst, edge.src, edge.gen,
                        "delta" if edge.relative else "value", edge.value))

    if contract.relations:
        lines.append("")
    for relation in contract.relations:
        text = ("rel %d  %s  producer=%d consumer=%d"
                % (relation.fence, relation.kind, relation.producer, relation.consumer))
        lines.append(text + ("  gap=%d" % relation.gap if relation.gap else ""))
    return "\n".join(lines) + "\n"


def _fields(words, line):
    """The `key=value` words of a record, as a dict; bare words become `True` flags."""
    out = {}
    for word in words:
        key, sep, value = word.partition("=")
        if key in out:
            raise RuntimeError("frame contract: duplicate field %r in %r" % (key, line))
        out[key] = value if sep else True
    return out


def _int(fields, key, line, default=None):
    if key not in fields:
        if default is None:
            raise RuntimeError("frame contract: %r needs a %s=" % (line, key))
        return default
    try:
        return int(fields[key])
    except (TypeError, ValueError):
        raise RuntimeError("frame contract: %s= is not an integer in %r" % (key, line)) from None


def parse_contract(text: str) -> Contract:
    """Read a contract back. The reference for the C++ parser, and what makes the format testable."""
    contract = Contract()
    lines = [line.split("#", 1)[0].strip() for line in (text or "").splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        return contract
    if lines[0] != CONTRACT_MARKER:
        raise RuntimeError("frame contract: expected %r, got %r" % (CONTRACT_MARKER, lines[0]))

    for line in lines[1:]:
        words = line.split()
        tag = words[0]
        if tag == "gen":
            fields = _fields(words[2:], line)
            gid = int(words[1])
            contract.generations[gid] = Generation(
                gid, _int(fields, "ring", line), _int(fields, "entry", line, 0),
                _int(fields, "advance", line, 0))
        elif tag in ("incoming", "transfer"):
            fields = _fields(words[1:], line)
            relative = tag == "transfer"
            contract.edges.append(Edge(
                _int(fields, "dst", line), _int(fields, "src", line), _int(fields, "gen", line),
                _int(fields, "delta" if relative else "value", line), relative))
        elif tag == "rel":
            kind = words[2]
            if kind not in _HAZARD_KINDS:
                raise RuntimeError("frame contract: unknown hazard %r in %r" % (kind, line))
            fields = _fields(words[3:], line)
            contract.relations.append(Relation(
                int(words[1]), kind, _int(fields, "producer", line),
                _int(fields, "consumer", line), _int(fields, "gap", line, 0)))
        else:
            raise RuntimeError("frame contract: unknown record %r in %r" % (tag, line))
    return contract


def encode_frame_contract(prog) -> str:
    """Build and render the contract for `prog` -- the one call the kernel writer makes."""
    return render_contract(build_contract(prog))

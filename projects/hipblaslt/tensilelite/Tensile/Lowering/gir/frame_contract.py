# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Versioned, CFG-free GIR frame facts consumed by StinkyTofu."""

from __future__ import annotations

from urllib.parse import quote

from .analysis import AnalysisManager
from .analyses.lds_buffers import LdsBufferIds
from .emit_plan import plan_program
from .nodes import Mark, Move


CONTRACT_KEY = "gir.frame_contract"
CONTRACT_HEADER = "GIR_FRAME_CONTRACT_V1"


def _shared_refs(action):
    source = action.source
    if not isinstance(source, Move):
        return ()
    if action.kind == "read":
        return tuple((ref, False) for ref in source.srcs if ref.tile.space == "shared")
    if action.kind == "copy":
        return tuple((ref, True) for ref in source.dsts if ref.tile.space == "shared")
    return ()


def _gens(prog):
    facts = {}

    def ensure(gen, ring=None):
        if gen is None:
            return None
        gid = int(gen.id)
        fact = facts.setdefault(gid, {
            "ring": max(1, int(ring if ring is not None else gen.ring)),
            "entry": None,
            "advance": None,
        })
        if fact["ring"] != max(1, int(ring if ring is not None else gen.ring)):
            raise RuntimeError("frame contract: inconsistent ring for Gen %d" % gid)
        return fact

    for block in prog.blocks.values():
        for phi in block.phis:
            fact = ensure(phi.gen)
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
            if not isinstance(inst, Move):
                continue
            for ref in tuple(inst.srcs) + tuple(inst.dsts):
                ensure(getattr(ref, "gen", None))

    for gid, fact in facts.items():
        if fact["entry"] is None:
            fact["entry"] = 0
        if fact["advance"] is None:
            fact["advance"] = 0
    return facts


def _action_for_identity(actions_by_source, identity, want_kind):
    candidates = [a for a in actions_by_source.get(int(identity), ()) if a.kind == want_kind]
    if len(candidates) != 1:
        raise RuntimeError(
            "frame contract: relation endpoint %r maps to %d %s actions"
            % (identity, len(candidates), want_kind))
    return candidates[0].action_id


def encode_frame_contract(prog) -> str:
    """Return a deterministic line-oriented contract with no GIR CFG topology."""
    analyses = AnalysisManager()
    storage = analyses.get(LdsBufferIds(), prog)
    if storage.unresolved:
        raise RuntimeError("frame contract: unresolved shared references %r" %
                           (storage.unresolved[:4],))

    plans = plan_program(prog)
    actions = [action for block_actions in plans.values() for action in block_actions]
    actions_by_source = {}
    for action in actions:
        if action.source is not None:
            actions_by_source.setdefault(id(action.source), []).append(action)

    lines = [CONTRACT_HEADER]
    for gid, fact in sorted(_gens(prog).items()):
        lines.append("GEN %d %d %d %d" %
                     (gid, fact["ring"], fact["entry"], fact["advance"]))

    distributed = prog.meta.get("agent_distributed", {}) or {}
    for action in actions:
        lines.append("ACTION %d %s" % (action.action_id, action.kind))
        for ref, is_write in _shared_refs(action):
            ring = max(1, int(storage.depth_of(ref.tile.operand)))
            gen = getattr(ref, "gen", None)
            gen_id = -1 if gen is None else int(gen.id)
            abs_gen = getattr(ref, "abs_gen", None)
            absolute = abs_gen is not None
            regions = storage.geometry.frame_regions_of(ref, is_write)
            encoded_regions = (
                "/".join(",".join(str(value) for value in sorted(axis_values))
                         for axis_values in regions)
                if regions else "-"
            )
            lines.append("ACCESS %d %d %d %d %d %d %d %s %s" % (
                action.action_id,
                int(is_write),
                gen_id,
                ring,
                int(getattr(ref, "gdelta", 0) or 0),
                int(abs_gen) if absolute else -1,
                int(bool(distributed.get(ref.tile.operand))),
                quote(str(ref.tile.operand), safe=""),
                encoded_regions,
            ))

    kind_endpoint = {
        "RAW": ("copy", "read"),
        "WAR": ("read", "copy"),
        "WAW": ("copy", "copy"),
    }
    for action in actions:
        source = action.source
        if action.kind != "fence" or not isinstance(source, Mark):
            continue
        for relation in source.at.get("relations") or ():
            kind = str(relation.get("kind"))
            if kind not in kind_endpoint:
                continue
            producer = relation.get("producer") or {}
            consumer = relation.get("consumer") or {}
            producer_action = _action_for_identity(
                actions_by_source, producer.get("identity"), kind_endpoint[kind][0])
            consumer_action = _action_for_identity(
                actions_by_source, consumer.get("identity"), kind_endpoint[kind][1])
            lines.append("REL %d %s %d %d %d" % (
                action.action_id, kind, producer_action, consumer_action,
                int(relation.get("gap", 0) or 0)))

    return "\n".join(lines) + "\n"


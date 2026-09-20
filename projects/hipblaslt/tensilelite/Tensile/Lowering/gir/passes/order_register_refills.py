# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""OrderRegisterRefillsPass -- keep a compact VGPR value live through its last logical use."""

from __future__ import annotations

from ..emit_plan import plan_block
from .base import Pass


def _read_writes(at):
    """``(physical slot, logical sources)`` written by one planned register read."""
    issue = tuple(at.get("issue") or ())
    fills = at.get("fills") or (
        (None, None, at.get("tile_flat", at["tile"]), None,
         at.get("k_flat", at["k"])),)
    logical = tuple((entry[2], entry[4], issue) for entry in fills)
    units = tuple(at.get("unit_indexes") or
                  (at.get("unit_index", at.get("tile_flat", at["tile"])),))
    if len(units) == len(logical):
        by_unit = {}
        for unit, source in zip(units, logical):
            by_unit.setdefault(unit, set()).add(source)
    else:
        by_unit = {units[0]: set(logical)}
    return tuple(
        ((at["tc"], at.get("group"), at["reg_buf"], unit), frozenset(sources))
        for unit, sources in by_unit.items())


def _mma_reads(at):
    """``(physical slot, logical source)`` consumed by one planned wmma."""
    issues = at.get("source_issues", {}) or {}
    return tuple(
        ((op, group, slot, unit), (logical, at["u"], tuple(issues.get(op, ()))))
        for op, group, slot, unit, logical in (at.get("srcs") or ()))


def _clobber_floors(prog, label):
    """Read node -> last consumer node it must follow, from logical def-use over physical names."""
    actions = plan_block(prog, label)
    writes = {}
    consumers = []
    for pos, action in enumerate(actions):
        if action.kind == "read":
            for slot, sources in _read_writes(action.at):
                writes.setdefault(slot, []).append((pos, action.source, sources))
        elif action.kind == "wmma":
            for slot, need in _mma_reads(action.at):
                consumers.append((pos, action.source, slot, need))

    floors = {}
    for consumer_pos, consumer, slot, need in consumers:
        history = writes.get(slot, ())
        definitions = [event for event in history
                       if event[0] < consumer_pos and need in event[2]]
        # A drain commonly consumes the final steady trip's live-out value, so its definition is
        # outside this block. Treat that as position -1; the dataflow verifier decides separately
        # whether such a live-in really exists.
        definition_pos = definitions[-1][0] if definitions else -1
        for write_pos, write, sources in history:
            if definition_pos < write_pos < consumer_pos and need not in sources:
                floors[id(write)] = (write, consumer)
    return floors


def order_register_refills(prog, label):
    """Sink only proven early clobbers; return whether the block changed."""
    block = prog.block(label)
    changed = False
    for _round in range(len(block.body) + 1):
        floors = _clobber_floors(prog, label)
        if not floors:
            return changed
        body = block.body
        index = {id(node): pos for pos, node in enumerate(body)}
        after = {}
        for node, consumer in floors.values():
            if id(node) not in index or id(consumer) not in index:
                continue
            if index[id(consumer)] <= index[id(node)]:
                continue
            after.setdefault(id(consumer), []).append(node)
        if not after:
            return changed
        movers = {id(node) for nodes in after.values() for node in nodes}

        out = []
        for node in body:
            if id(node) not in movers:
                out.append(node)
            out.extend(after.get(id(node), ()))
        block.body[:] = out
        prog.bump()
        changed = True
    raise RuntimeError(f"register-refill ordering did not converge in block {label!r}")


class OrderRegisterRefillsPass(Pass):
    """Use physical-register def-use, not memory tokens, to repair early refill schedules."""

    def run(self, prog, am):
        changed = False
        for block in prog.blocks.values():
            # Steady schedules have explicit in-block definitions and are already validated by the
            # LoopIR ledger. The ambiguity this pass resolves is drain live-in state: its defining
            # read is in the preceding trip, represented by a negative generation relation here.
            if block.gen_rel is None or int(block.gen_rel) >= 0:
                continue
            changed |= order_register_refills(prog, block.label)
        if changed:
            am.invalidate()
            return ("body",)
        return ()

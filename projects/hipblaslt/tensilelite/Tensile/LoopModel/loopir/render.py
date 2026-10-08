# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Total human-readable rendering for LoopIR."""

from __future__ import annotations

from .expr import TRUE
from .model import (
    Compute,
    Cond,
    Let,
    Loop,
    Movement,
    Program,
    Region,
    Switch,
)
from .values import IMMEDIATE


def _tile(tile):
    return f"{tile.operand.name}[{', '.join(value.render() for value in tile.indices)}]"


def _where(predicate):
    return (
        ""
        if predicate is TRUE or predicate.evaluate({}) is True
        else f" where {predicate.render()}"
    )


def _render_completion(value):
    if value is IMMEDIATE:
        return "immediate"
    return f"{value.name}@{value.counter_scope.name.lower()}"


def _render_body(nodes, indent, out):
    pad = "  " * indent
    for node in nodes:
        if isinstance(node, Region):
            out.append(f"{pad}@{node.tag.name.lower()}")
        elif isinstance(node, Movement):
            out.append(
                f"{pad}{node.op.__name__.lower()} "
                f"completion={_render_completion(node.completion)}"
            )
            for index, transfer in enumerate(node.transfers):
                effect = node.effect_for(index)
                out.append(
                    f"{pad}  effect={_tile(effect.tile)} "
                    f"local={_tile(transfer.local_tile)} "
                    f"{transfer.source.name.lower()} -> "
                    f"{transfer.destination.name.lower()}"
                    f"{_where(effect.where)}"
                )
        elif isinstance(node, Compute):
            reads = ", ".join(
                f"{_tile(value)}@register"
                for value in node.reads
            )
            writes = ", ".join(
                f"{_tile(value)}@register"
                for value in node.writes
            )
            out.append(
                f"{pad}{node.op.__name__.lower()} "
                f"read({reads}) write({writes})"
            )
        elif isinstance(node, Loop):
            out.append(
                f"{pad}for {node.axis.name} in "
                f"[{node.lo.render()}, {node.hi.render()})"
            )
            _render_body(node.body, indent + 1, out)
        elif isinstance(node, Cond):
            out.append(f"{pad}if {node.pred.render()}")
            _render_body(node.then_body, indent + 1, out)
            if node.else_body:
                out.append(f"{pad}else")
                _render_body(node.else_body, indent + 1, out)
        elif isinstance(node, Let):
            out.append(f"{pad}let {node.axis.name} = {node.value.render()}")
            _render_body(node.body, indent + 1, out)
        elif isinstance(node, Switch):
            out.append(f"{pad}switch {node.expression.render()}")
            for value, body in node.cases:
                out.append(f"{pad}  case {value}")
                _render_body(body, indent + 2, out)
            if node.otherwise:
                out.append(f"{pad}  otherwise")
                _render_body(node.otherwise, indent + 2, out)
        else:
            out.append(f"{pad}<{type(node).__name__}>")


def render(program: Program) -> str:
    params = ", ".join(param.name for param in program.params) or "-"
    axes = ", ".join(
        f"{axis.name}:{axis.extent.render()}"
        for axis in program.axes
    ) or "-"
    agents = ", ".join(
        f"{axis.name}:{axis.extent.render()}@{scope.name.lower()}"
        for axis, scope in program.agents
    ) or "-"
    out = [
        "program",
        f"  params: {params}",
        f"  axes: {axes}",
        f"  agents: {agents}",
        "  operands:",
        *(
            f"    {operand.name}[{', '.join(axis.name for axis in operand.axes)}] "
            f"{operand.role.name.lower()}"
            for operand in program.operands
        ),
        "  body:",
    ]
    _render_body(program.body, 2, out)
    return "\n".join(out)


__all__ = ["render"]

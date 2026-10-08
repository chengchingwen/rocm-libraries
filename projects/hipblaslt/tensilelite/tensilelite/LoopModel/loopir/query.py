# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Read-only structural queries over LoopIR."""

from __future__ import annotations

from dataclasses import dataclass, field

from .values import IRTypeError, TransformRefused
from .expr import Axis, Not, expr_equivalent
from .model import (
    Compute,
    Cond,
    Let,
    Loop,
    LoopIR,
    Movement,
    Node,
    Program,
    Region,
    Switch,
)


Path = tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class Occurrence:
    """One structural node occurrence in one immutable Program."""

    node: Node = field(repr=False)
    path: Path
    _program: Program = field(repr=False)
    ancestors: tuple[Node, ...] = field(default=(), repr=False)
    domains: tuple = ()
    bindings: tuple = ()
    predicates: tuple = ()

    def __post_init__(self):
        if not isinstance(self.node, LoopIR):
            raise IRTypeError("Occurrence.node must be a LoopIR node")

    def __getattr__(self, name):
        return getattr(self.node, name)

@dataclass(frozen=True)
class Selection:
    """An ordered tuple of structural occurrences from one Program."""

    occurrences: tuple[Occurrence, ...]
    _program: Program = field(repr=False)

    def __post_init__(self):
        if not isinstance(self.occurrences, tuple):
            raise IRTypeError("Selection.occurrences must be a tuple")
        if any(item._program is not self._program for item in self.occurrences):
            raise IRTypeError("a Selection cannot mix Programs")

    def __len__(self) -> int:
        return len(self.occurrences)

    def __getitem__(self, index):
        return self.occurrences[index]

def _selection_of(program: Program, occurrences) -> Selection:
    if not isinstance(program, Program):
        raise IRTypeError("selection_of expects a Program")
    values = tuple(occurrences)
    if not all(isinstance(item, Occurrence) for item in values):
        raise IRTypeError("selection values must be Occurrence objects")
    return Selection(values, program)


def _walk_body(
    program,
    body,
    path=(),
    ancestors=(),
    domains=(),
    bindings=(),
    predicates=(),
):
    for index, node in enumerate(body):
        node_path = path + (("body", index),)
        occurrence = Occurrence(
            node,
            node_path,
            program,
            ancestors,
            domains,
            bindings,
            predicates,
        )
        yield occurrence
        nested = ancestors + (node,)
        if isinstance(node, (Loop, Let)):
            yield from _walk_body(
                program,
                node.body,
                node_path + ((type(node).__name__.lower(), 0),),
                nested,
                domains + ((node.axis, node.lo, node.hi),)
                if isinstance(node, Loop) else domains,
                bindings + ((node.axis, node.value),)
                if isinstance(node, Let) else bindings,
                predicates,
            )
        elif isinstance(node, Cond):
            yield from _walk_body(
                program,
                node.then_body,
                node_path + (("then", 0),),
                nested,
                domains,
                bindings,
                predicates + (node.pred,),
            )
            yield from _walk_body(
                program,
                node.else_body,
                node_path + (("else", 0),),
                nested,
                domains,
                bindings,
                predicates + (Not(node.pred),),
            )
        elif isinstance(node, Switch):
            for case_index, (_value, case_body) in enumerate(node.cases):
                yield from _walk_body(
                    program,
                    case_body,
                    node_path + (("case", case_index),),
                    nested,
                    domains,
                    bindings,
                    predicates + (node.expression == _value,),
                )
            yield from _walk_body(
                program,
                node.otherwise,
                node_path + (("otherwise", 0),),
                nested,
                domains,
                bindings,
                predicates + tuple(
                    node.expression != value for value, _body in node.cases
                ),
            )


def occurrences(program: Program) -> tuple[Occurrence, ...]:
    if not isinstance(program, Program):
        raise IRTypeError("occurrences expects a Program")
    return tuple(_walk_body(program, program.body))


def _find(program: Program, predicate) -> Selection:
    if not isinstance(program, Program):
        raise IRTypeError("find expects a Program")
    if not callable(predicate):
        raise IRTypeError("find.predicate must be callable")
    return _selection_of(
        program,
        (item for item in occurrences(program) if bool(predicate(item))),
    )


def _one(selection: Selection, name: str) -> Occurrence:
    if len(selection) != 1:
        raise TransformRefused(
            f"{name} expected exactly one match, found {len(selection)}"
        )
    return selection[0]


def find_loop(
    program: Program,
    axis: Axis,
    *,
    lo=None,
    hi=None,
) -> Occurrence:
    if not isinstance(axis, Axis):
        raise IRTypeError("find_loop.axis must be an Axis")

    def matches(item):
        node = item.node
        return isinstance(node, Loop) and node.axis is axis \
            and (lo is None or expr_equivalent(node.lo, lo)) \
            and (hi is None or expr_equivalent(node.hi, hi))

    return _one(_find(program, matches), "find_loop")


def operations(program: Program) -> tuple[Movement | Compute, ...]:
    if not isinstance(program, Program):
        raise IRTypeError("operations expects a Program")
    result = []
    identities = set()
    for item in occurrences(program):
        node = item.node
        if isinstance(node, (Movement, Compute)) and id(node._identity) not in identities:
            identities.add(id(node._identity))
            result.append(node)
    return tuple(result)


def regions(program: Program) -> tuple[Region, ...]:
    if not isinstance(program, Program):
        raise IRTypeError("regions expects a Program")
    return tuple(
        item.node for item in occurrences(program) if isinstance(item.node, Region)
    )


__all__ = [
    "Occurrence",
    "occurrences", "find_loop", "operations", "regions",
]

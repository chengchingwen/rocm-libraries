# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Named pure Program transformations for LoopIR generation."""

from __future__ import annotations

from collections.abc import Callable

from .values import IRTypeError, TransformRefused
from .expr import (
    And,
    Axis,
    Compare,
    Pred,
    TRUE,
    as_pred,
    expr_equivalent,
)
from .model import (
    DRAIN,
    PROLOGUE,
    SHORT,
    STEADY,
    Compute,
    Cond,
    Effect,
    Let,
    Loop,
    LoopIR,
    Movement,
    Program,
    Region,
    Switch,
    Tile,
    Transfer,
    same_operation,
)
from .rewrite import insert, replace, rewrite
Transform = Callable[[Program], Program]


def compose(*transforms: Transform) -> Transform:
    """Compose named ``Program -> Program`` functions left to right."""

    if not all(callable(transform) for transform in transforms):
        raise IRTypeError("compose expects only callable Program transformations")

    def composed(program: Program) -> Program:
        current = program
        for transform in transforms:
            current = transform(current)
            if not isinstance(current, Program):
                raise IRTypeError("a Program transformation returned a non-Program")
        return current

    return composed


def append(program: Program, nodes: tuple[LoopIR, ...]) -> Program:
    """Append nodes to the root body."""

    return insert(program, program.body, len(program.body), nodes)


def materialize_steady_copies(program: Program, copies: int) -> Program:
    """Materialize the target-derived steady ring period in Program structure."""

    if not isinstance(copies, int) or isinstance(copies, bool) or copies < 1:
        raise IRTypeError("steady copy count must be a positive integer")
    if copies == 1:
        return program
    copy_axis = Axis("_steady_copy", copies)
    changed = 0

    def edit_body(body, active=None):
        nonlocal changed
        current = active
        output = []
        for node in body:
            if isinstance(node, Region):
                current = node.tag
                output.append(node)
                continue
            if isinstance(node, Loop):
                child = edit_body(node.body, current)
                if current is STEADY and node.hi.free_params():
                    child = tuple(
                        Let(copy_axis, index, child)
                        for index in range(copies)
                    )
                    changed += 1
                node = Loop(node.axis, node.lo, node.hi, child)
            elif isinstance(node, Cond):
                node = Cond(
                    node.pred,
                    edit_body(node.then_body, current),
                    edit_body(node.else_body, current),
                )
            elif isinstance(node, Let):
                node = Let(node.axis, node.value, edit_body(node.body, current))
            elif isinstance(node, Switch):
                node = Switch(
                    node.expression,
                    tuple(
                        (value, edit_body(case, current))
                        for value, case in node.cases
                    ),
                    edit_body(node.otherwise, current),
                )
            output.append(node)
        return tuple(output)

    body = edit_body(program.body)
    if changed != 1:
        raise TransformRefused(
            f"expected one symbolic steady Loop, found {changed}"
        )
    return Program(
        program.params,
        (*program.axes, copy_axis),
        program.operands,
        body,
        program.agents,
        _identity=program._identity,
    )


def _replace_predicate(original: Pred, extra: Pred) -> Pred:
    return extra if original is TRUE else And(original, extra)


def _restrict_effect(effect: Effect, predicate: Pred) -> Effect:
    return Effect(
        effect.tile,
        _replace_predicate(effect.where, predicate),
    )


def partition_movement(
    program: Program,
    movement: Movement,
    predicates: tuple[Pred, ...],
    completions: tuple,
) -> Program:
    """Replace one Movement with independently-issued domain partitions.

    This is a generic checked structural primitive.  TensorLoad-side
    generation supplies TDMSplit predicates and completion values.
    """

    if not isinstance(movement, Movement):
        raise IRTypeError("partition_movement expects a Movement")
    if not isinstance(predicates, tuple) or not predicates:
        raise IRTypeError("partition predicates must be a non-empty tuple")
    if not isinstance(completions, tuple) or len(completions) != len(predicates):
        raise IRTypeError("partition completions must match predicates")
    parts = tuple(
        Movement(
            movement.op,
            tuple(
                Transfer(
                    transfer.local_tile,
                    transfer.source,
                    transfer.destination,
                    (
                        _restrict_effect(transfer.effect, as_pred(predicate))
                        if transfer.effect is not None else None
                    ),
                )
                for transfer in movement.transfers
            ),
            completion,
            effect=(
                _restrict_effect(movement.effect, as_pred(predicate))
                if movement.effect is not None else None
            ),
        )
        for predicate, completion in zip(predicates, completions)
    )
    return replace(program, movement, parts)


def fuse_movements(
    program: Program,
    movements: tuple[Movement, ...],
    completion,
) -> Program:
    """Fuse adjacent compatible Movements into one issue/completion."""

    if not isinstance(movements, tuple) or len(movements) < 2 \
            or not all(isinstance(value, Movement) for value in movements):
        raise IRTypeError("fuse_movements requires at least two Movements")
    if len({movement.op for movement in movements}) != 1:
        raise TransformRefused("fused Movements must have the same kind")
    transfers = []
    for movement in movements:
        for index, transfer in enumerate(movement.transfers):
            transfers.append(Transfer(
                transfer.local_tile,
                transfer.source,
                transfer.destination,
                movement.effect_for(index),
            ))
    fused = Movement(movements[0].op, tuple(transfers), completion)
    return replace(program, movements, (fused,))


def _substitute_tile(tile: Tile, replacements) -> Tile:
    return Tile(
        tile.operand,
        tuple(expression.substitute(replacements) for expression in tile.indices),
    )


def _substitute_operation(operation, replacements):
    if isinstance(operation, Movement):
        return Movement(
            operation.op,
            tuple(
                Transfer(
                    _substitute_tile(transfer.local_tile, replacements),
                    transfer.source,
                    transfer.destination,
                    (
                        Effect(
                            _substitute_tile(
                                transfer.effect.tile,
                                replacements,
                            ),
                            transfer.effect.where.substitute(replacements),
                        )
                        if transfer.effect is not None else None
                    ),
                )
                for transfer in operation.transfers
            ),
            operation.completion,
            effect=(
                Effect(
                    _substitute_tile(operation.effect.tile, replacements),
                    operation.effect.where.substitute(replacements),
                )
                if operation.effect is not None else None
            ),
            _identity=operation._identity,
        )
    if isinstance(operation, Compute):
        return Compute(
            operation.op,
            tuple(_substitute_tile(value, replacements) for value in operation.reads),
            tuple(_substitute_tile(value, replacements) for value in operation.writes),
            operation.completion,
            _identity=operation._identity,
        )
    raise IRTypeError("operation substitution expects Movement or Compute")


def substitute_operation(operation, replacements):
    """Return one operation with Axis expressions substituted, preserving identity."""

    return _substitute_operation(operation, dict(replacements))


def rewrite_operation(
    program: Program,
    operation: Movement | Compute,
    rule: Callable[[Movement | Compute], Movement | Compute],
) -> Program:
    """Apply a typed semantic operation rewrite to every logical occurrence."""

    if not callable(rule):
        raise IRTypeError("rewrite_operation.rule must be callable")

    def apply(node):
        if isinstance(node, (Movement, Compute)) and same_operation(node, operation):
            replacement = rule(node)
            if not isinstance(replacement, (Movement, Compute)):
                raise IRTypeError("operation rewrite must return an operation")
            return replacement
        return node

    return rewrite(program, apply)


def _operation_map(body):
    result = {}

    def visit(nodes):
        for node in nodes:
            if isinstance(node, (Movement, Compute)):
                result.setdefault(node._identity, node)
            if isinstance(node, Loop):
                visit(node.body)
            elif isinstance(node, Cond):
                visit(node.then_body)
                visit(node.else_body)
            elif isinstance(node, Let):
                visit(node.body)
            elif isinstance(node, Switch):
                for _value, case in node.cases:
                    visit(case)
                visit(node.otherwise)

    visit(body)
    return result


def _pipeline_body(body, include, variants):
    output = []
    for node in body:
        if isinstance(node, (Movement, Compute)):
            if include(node._identity):
                output.append(variants.get(node._identity, node))
            continue
        if isinstance(node, Region):
            output.append(node)
        elif isinstance(node, Loop):
            child = _pipeline_body(node.body, include, variants)
            if child:
                output.append(Loop(
                    node.axis, node.lo, node.hi, child,
                ))
        elif isinstance(node, Let):
            child = _pipeline_body(node.body, include, variants)
            if child:
                output.append(Let(node.axis, node.value, child))
        elif isinstance(node, Cond):
            then_body = _pipeline_body(node.then_body, include, variants)
            else_body = _pipeline_body(node.else_body, include, variants)
            if then_body or else_body:
                output.append(Cond(node.pred, then_body, else_body))
        elif isinstance(node, Switch):
            cases = tuple(
                (value, _pipeline_body(case, include, variants))
                for value, case in node.cases
            )
            otherwise = _pipeline_body(node.otherwise, include, variants)
            if any(case for _value, case in cases) or otherwise:
                output.append(Switch(node.expression, cases, otherwise))
    return tuple(output)


def pipeline(
    program: Program,
    loop: Loop,
    specifications: tuple[tuple[Movement | Compute, int], ...],
) -> Program:
    """Build prologue/steady/drain/short structure from operation advances.

    Each specification names a logical operation and a non-negative advance
    along ``loop.axis``.  The result contains only ordinary LoopIR nodes and
    Region markers.
    """

    if not isinstance(loop, Loop):
        raise TransformRefused("pipeline requires one serial Loop")
    if not isinstance(specifications, tuple):
        raise IRTypeError("pipeline specifications must be a tuple")
    originals = _operation_map(loop.body)
    depths = {}
    variants = {}
    for operation, depth in specifications:
        if not isinstance(operation, (Movement, Compute)):
            raise IRTypeError("pipeline specifications require operations")
        if not isinstance(depth, int) or isinstance(depth, bool) or depth < 0:
            raise IRTypeError("pipeline depth must be a non-negative integer")
        if operation._identity not in originals:
            raise TransformRefused("pipeline operation is not inside the selected Loop")
        if depth:
            depths[operation._identity] = depth
            variants[operation._identity] = _substitute_operation(
                originals[operation._identity],
                {loop.axis: loop.axis + depth},
            )
    maximum = max(depths.values(), default=0)
    if maximum == 0:
        return program

    prologue = []
    for step in range(maximum):
        threshold = maximum - 1 - step
        included = {
            identity for identity, depth in depths.items()
            if depth > threshold
        }
        body = _pipeline_body(
            loop.body,
            lambda identity: identity in included,
            variants,
        )
        if body:
            prologue.append(Let(loop.axis, loop.lo + step - maximum, body))

    steady = Loop(
        loop.axis,
        loop.lo,
        loop.hi - maximum,
        _pipeline_body(loop.body, lambda _identity: True, variants),
    )

    selected = set(depths)
    drain = []
    for step in range(maximum):
        limit = maximum - 1 - step
        included = {
            identity for identity, depth in depths.items()
            if depth <= limit
        }
        body = _pipeline_body(
            loop.body,
            lambda identity: identity not in selected or identity in included,
            {**originals, **{
                identity: variants[identity]
                for identity in included
            }},
        )
        if body:
            drain.append(Let(loop.axis, loop.hi - maximum + step, body))

    replacement = Cond(
        loop.hi - loop.lo >= maximum,
        (
            Region(PROLOGUE),
            *prologue,
            Region(STEADY),
            steady,
            Region(DRAIN),
            *drain,
        ),
        (
            Region(SHORT),
            Loop(
                loop.axis, loop.lo, loop.hi, loop.body,
            ),
        ),
    )
    return replace(program, loop, (replacement,))


def suppress_drain(program: Program) -> Program:
    """Remove pipeline drain/short arms while retaining prologue and steady.

    The result contains the same ordinary Region markers and Loop nodes; this
    transform adds no special no-load-loop node to LoopIR.
    """

    def rule(node):
        if not isinstance(node, Cond):
            return node
        tags = {
            candidate.tag
            for candidate in node.then_body
            if isinstance(candidate, Region)
        }
        if not {PROLOGUE, STEADY, DRAIN} <= tags:
            return node
        original_loop = next(
            (
                candidate for candidate in node.else_body
                if isinstance(candidate, Loop)
            ),
            None,
        )
        kept = []
        for candidate in node.then_body:
            if isinstance(candidate, Region) and candidate.tag is DRAIN:
                break
            if (
                original_loop is not None
                and isinstance(candidate, Loop)
                and candidate.axis is original_loop.axis
            ):
                candidate = Loop(
                    candidate.axis,
                    candidate.lo,
                    original_loop.hi,
                    candidate.body,
                )
            kept.append(candidate)
        return tuple(kept)

    return rewrite(program, rule)


def _first_touch_value(predicate, axis):
    if not isinstance(predicate, Compare) or predicate.op != "==":
        return None
    axis_expression = axis._as_expr()
    if expr_equivalent(predicate.lhs, axis_expression):
        return predicate.rhs.constant_value
    if expr_equivalent(predicate.rhs, axis_expression):
        return predicate.lhs.constant_value
    return None


def _resolve_first_touch(body, axis, value):
    output = []
    for node in body:
        if isinstance(node, Cond):
            expected = _first_touch_value(node.pred, axis)
            if expected is not None:
                branch = node.then_body if int(value) == expected else node.else_body
                output.extend(_resolve_first_touch(branch, axis, value))
                continue
            node = Cond(
                node.pred,
                _resolve_first_touch(node.then_body, axis, value),
                _resolve_first_touch(node.else_body, axis, value),
            )
        elif isinstance(node, Loop):
            node = Loop(
                node.axis,
                node.lo,
                node.hi,
                _resolve_first_touch(node.body, axis, value),
            )
        elif isinstance(node, Let):
            node = Let(
                node.axis,
                node.value,
                _resolve_first_touch(node.body, axis, value),
            )
        elif isinstance(node, Switch):
            node = Switch(
                node.expression,
                tuple(
                    (
                        case,
                        _resolve_first_touch(case_body, axis, value),
                    )
                    for case, case_body in node.cases
                ),
                _resolve_first_touch(node.otherwise, axis, value),
            )
        output.append(node)
    return tuple(output)


def specialize_first_touch(program: Program, region=None) -> Program:
    """Specialize finite loop bodies at simple first-touch equality guards."""

    def edit_body(body, active_region=None):
        output = []
        current_region = active_region
        for node in body:
            if isinstance(node, Region):
                current_region = node.tag
                output.append(node)
                continue
            if isinstance(node, Loop):
                child = edit_body(node.body, current_region)
                lo = node.lo.constant_value
                hi = node.hi.constant_value
                if (
                    (region is None or current_region is region)
                    and lo is not None
                    and hi is not None
                    and hi > lo
                ):
                    cases = tuple(
                        (
                            value,
                            _resolve_first_touch(
                                child, node.axis, value
                            ),
                        )
                        for value in range(lo, hi)
                    )
                    if not all(
                        case_body == cases[0][1]
                        for _value, case_body in cases[1:]
                    ):
                        child = (Switch(node.axis, cases, ()),)
                    else:
                        child = cases[0][1]
                node = Loop(
                    node.axis,
                    node.lo,
                    node.hi,
                    child,
                )
            elif isinstance(node, Cond):
                node = Cond(
                    node.pred,
                    edit_body(node.then_body, current_region),
                    edit_body(node.else_body, current_region),
                )
            elif isinstance(node, Let):
                node = Let(
                    node.axis,
                    node.value,
                    edit_body(node.body, current_region),
                )
            elif isinstance(node, Switch):
                node = Switch(
                    node.expression,
                    tuple(
                        (
                            value,
                            edit_body(case, current_region),
                        )
                        for value, case in node.cases
                    ),
                    edit_body(node.otherwise, current_region),
                )
            output.append(node)
        return tuple(output)

    return program.with_body(edit_body(program.body))


def defer_single_path_operations(
    program: Program,
    region,
    operations: tuple[Movement | Compute, ...],
) -> Program:
    """Defer selected operations only through single-path structural wrappers."""

    identities = {operation._identity for operation in operations}

    def selected_identity(node):
        while True:
            if isinstance(node, (Movement, Compute)):
                return (
                    node._identity
                    if node._identity in identities else None
                )
            if isinstance(node, Loop) and len(node.body) == 1:
                node = node.body[0]
                continue
            if (
                isinstance(node, Cond)
                and not node.else_body
                and len(node.then_body) == 1
            ):
                node = node.then_body[0]
                continue
            return None

    def edit_body(body, active_region=None):
        rebuilt = []
        current_region = active_region
        for node in body:
            if isinstance(node, Region):
                current_region = node.tag
                rebuilt.append(node)
                continue
            if isinstance(node, Loop):
                node = Loop(
                    node.axis,
                    node.lo,
                    node.hi,
                    edit_body(node.body, current_region),
                )
            elif isinstance(node, Cond):
                node = Cond(
                    node.pred,
                    edit_body(node.then_body, current_region),
                    edit_body(node.else_body, current_region),
                )
            elif isinstance(node, Let):
                node = Let(
                    node.axis,
                    node.value,
                    edit_body(node.body, current_region),
                )
            elif isinstance(node, Switch):
                node = Switch(
                    node.expression,
                    tuple(
                        (
                            value,
                            edit_body(case, current_region),
                        )
                        for value, case in node.cases
                    ),
                    edit_body(node.otherwise, current_region),
                )
            rebuilt.append(node)
        def reordered(nodes):
            return (
                tuple(
                    node for node in nodes
                    if selected_identity(node) is None
                )
                + tuple(
                    node for node in nodes
                    if selected_identity(node) is not None
                )
            )

        if not any(isinstance(node, Region) for node in rebuilt):
            return (
                reordered(tuple(rebuilt))
                if active_region is region else tuple(rebuilt)
            )
        output = []
        index = 0
        segment_region = active_region
        while index < len(rebuilt):
            node = rebuilt[index]
            if isinstance(node, Region):
                segment_region = node.tag
                output.append(node)
                index += 1
                continue
            stop = index
            while stop < len(rebuilt) \
                    and not isinstance(rebuilt[stop], Region):
                stop += 1
            segment = tuple(rebuilt[index:stop])
            output.extend(
                reordered(segment)
                if segment_region is region else segment
            )
            index = stop
        return tuple(output)

    return program.with_body(edit_body(program.body))


def defer_operations(
    program: Program,
    region,
    operations: tuple[Movement | Compute, ...],
) -> Program:
    """Move selected direct-body operations to the end of Loops in one region."""

    identities = {
        operation._identity
        for operation in operations
        if isinstance(operation, (Movement, Compute))
    }
    if not identities:
        return program

    def operation_identities(node):
        if isinstance(node, (Movement, Compute)):
            return {node._identity}
        result = set()
        if isinstance(node, Loop):
            children = (node.body,)
        elif isinstance(node, Cond):
            children = (node.then_body, node.else_body)
        elif isinstance(node, Let):
            children = (node.body,)
        elif isinstance(node, Switch):
            children = tuple(case for _value, case in node.cases) + (
                node.otherwise,
            )
        else:
            children = ()
        for child in children:
            for item in child:
                result.update(operation_identities(item))
        return result

    def selected_unit(node):
        present = operation_identities(node)
        return bool(present) and present <= identities

    def edit_body(body, active_region=None):
        output = []
        current_region = active_region
        for node in body:
            if isinstance(node, Region):
                current_region = node.tag
                output.append(node)
                continue
            rebuilt = node
            if isinstance(node, Loop):
                child = edit_body(node.body, current_region)
                if current_region is region:
                    leading = tuple(
                        item for item in child
                        if not selected_unit(item)
                    )
                    trailing = tuple(
                        item for item in child
                        if selected_unit(item)
                    )
                    child = leading + trailing
                rebuilt = Loop(
                    node.axis, node.lo, node.hi, child,
                )
            elif isinstance(node, Cond):
                rebuilt = Cond(
                    node.pred,
                    edit_body(node.then_body, current_region),
                    edit_body(node.else_body, current_region),
                )
            elif isinstance(node, Let):
                rebuilt = Let(
                    node.axis,
                    node.value,
                    edit_body(node.body, current_region),
                )
            elif isinstance(node, Switch):
                rebuilt = Switch(
                    node.expression,
                    tuple(
                        (value, edit_body(case, current_region))
                        for value, case in node.cases
                    ),
                    edit_body(node.otherwise, current_region),
                )
            output.append(rebuilt)
        return tuple(output)

    return program.with_body(edit_body(program.body))


def prioritize_operations(
    program: Program,
    region,
    operations: tuple[Movement | Compute, ...],
) -> Program:
    """Move selected operation-only structural units to body fronts in one region."""

    identities = {operation._identity for operation in operations}
    if not identities:
        return program

    def operation_identities(node):
        if isinstance(node, (Movement, Compute)):
            return {node._identity}
        children = ()
        if isinstance(node, Loop):
            children = (node.body,)
        elif isinstance(node, Cond):
            children = (node.then_body, node.else_body)
        elif isinstance(node, Let):
            children = (node.body,)
        elif isinstance(node, Switch):
            children = tuple(case for _value, case in node.cases) + (
                node.otherwise,
            )
        return {
            identity
            for child in children
            for item in child
            for identity in operation_identities(item)
        }

    def selected_unit(node):
        present = operation_identities(node)
        return bool(present) and present <= identities

    def edit_body(body, active_region=None):
        output = []
        current_region = active_region
        for node in body:
            if isinstance(node, Region):
                current_region = node.tag
                output.append(node)
                continue
            if isinstance(node, Loop):
                node = Loop(
                    node.axis,
                    node.lo,
                    node.hi,
                    edit_body(node.body, current_region),
                )
            elif isinstance(node, Cond):
                node = Cond(
                    node.pred,
                    edit_body(node.then_body, current_region),
                    edit_body(node.else_body, current_region),
                )
            elif isinstance(node, Let):
                node = Let(node.axis, node.value, edit_body(node.body, current_region))
            elif isinstance(node, Switch):
                node = Switch(
                    node.expression,
                    tuple(
                        (value, edit_body(case, current_region))
                        for value, case in node.cases
                    ),
                    edit_body(node.otherwise, current_region),
                )
            output.append(node)
        output = tuple(output)
        if current_region is not region:
            return output
        return (
            tuple(item for item in output if selected_unit(item))
            + tuple(item for item in output if not selected_unit(item))
        )

    return program.with_body(edit_body(program.body))


def materialize_region_operations(
    program: Program,
    region,
    selector: Callable[[Movement | Compute], bool],
    key: Callable[[Movement | Compute], object],
) -> Program:
    """Unroll and order operation-only regions using caller-owned policy.

    The structural rewrite is generic.  Theta generation supplies the
    operation selector and ordering key; lowering never reconstructs them.
    """

    if not callable(selector) or not callable(key):
        raise IRTypeError("materialize_region_operations expects callables")

    def operations_in(nodes):
        result = []
        for node in nodes:
            if isinstance(node, (Movement, Compute)):
                result.append(node)
            elif isinstance(node, Loop):
                result.extend(operations_in(node.body))
            elif isinstance(node, Cond):
                result.extend(operations_in(node.then_body))
                result.extend(operations_in(node.else_body))
            elif isinstance(node, Let):
                result.extend(operations_in(node.body))
            elif isinstance(node, Switch):
                for _value, case in node.cases:
                    result.extend(operations_in(case))
                result.extend(operations_in(node.otherwise))
        return result

    def flatten(nodes, bindings):
        output = []
        for node in nodes:
            if isinstance(node, (Movement, Compute)):
                output.append(_substitute_operation(node, bindings))
            elif isinstance(node, Let):
                value = node.value.substitute(bindings)
                if value.constant_value is None:
                    raise TransformRefused("materialized Let is not constant")
                flatten_bindings = {**bindings, node.axis: value.constant_value}
                output.extend(flatten(node.body, flatten_bindings))
            elif isinstance(node, Loop):
                lower = node.lo.substitute(bindings)
                upper = node.hi.substitute(bindings)
                if lower.constant_value is None or upper.constant_value is None:
                    raise TransformRefused("materialized Loop is not finite")
                for value in range(lower.constant_value, upper.constant_value):
                    output.extend(flatten(node.body, {**bindings, node.axis: value}))
            elif isinstance(node, Cond):
                predicate = node.pred.substitute(bindings)
                verdict = predicate.evaluate({})
                if verdict is None:
                    raise TransformRefused("materialized Cond is unresolved")
                output.extend(flatten(
                    node.then_body if verdict else node.else_body,
                    bindings,
                ))
            elif isinstance(node, Switch):
                expression = node.expression.substitute(bindings)
                if expression.constant_value is None:
                    raise TransformRefused("materialized Switch is unresolved")
                output.extend(flatten(
                    dict(node.cases).get(expression.constant_value, node.otherwise),
                    bindings,
                ))
            elif isinstance(node, Region):
                continue
        return tuple(output)

    def edit_body(body):
        rebuilt = []
        for node in body:
            if isinstance(node, Loop):
                node = Loop(
                    node.axis,
                    node.lo,
                    node.hi,
                    edit_body(node.body),
                )
            elif isinstance(node, Cond):
                node = Cond(
                    node.pred,
                    edit_body(node.then_body),
                    edit_body(node.else_body),
                )
            elif isinstance(node, Let):
                node = Let(node.axis, node.value, edit_body(node.body))
            elif isinstance(node, Switch):
                node = Switch(
                    node.expression,
                    tuple(
                        (value, edit_body(case))
                        for value, case in node.cases
                    ),
                    edit_body(node.otherwise),
                )
            rebuilt.append(node)
        body = tuple(rebuilt)
        output = []
        index = 0
        while index < len(body):
            node = body[index]
            output.append(node)
            if not isinstance(node, Region) or node.tag is not region:
                index += 1
                continue
            stop = index + 1
            while stop < len(body) and not isinstance(body[stop], Region):
                stop += 1
            segment = body[index + 1:stop]
            found = operations_in(segment)
            if found and all(selector(operation) for operation in found):
                materialized = flatten(segment, {})
                output.extend(sorted(materialized, key=key))
            else:
                output.extend(segment)
            index = stop
        return tuple(output)

    return program.with_body(edit_body(program.body))


def materialize_operation_runs(
    program: Program,
    region,
    selector: Callable[[Movement | Compute], bool],
    key: Callable[[Movement | Compute], object],
) -> Program:
    """Unroll and order consecutive selected-operation structural units."""

    def operations_in(node):
        if isinstance(node, (Movement, Compute)):
            return (node,)
        children = ()
        if isinstance(node, Loop):
            children = (node.body,)
        elif isinstance(node, Cond):
            children = (node.then_body, node.else_body)
        elif isinstance(node, Let):
            children = (node.body,)
        elif isinstance(node, Switch):
            children = tuple(case for _value, case in node.cases) + (
                node.otherwise,
            )
        return tuple(
            operation
            for child in children
            for item in child
            for operation in operations_in(item)
        )

    def selected_unit(node):
        found = operations_in(node)
        return bool(found) and all(selector(operation) for operation in found)

    def flatten(nodes, bindings):
        output = []
        for node in nodes:
            if isinstance(node, (Movement, Compute)):
                output.append(_substitute_operation(node, bindings))
            elif isinstance(node, Let):
                value = node.value.substitute(bindings)
                if value.constant_value is None:
                    raise TransformRefused("materialized Let is not constant")
                output.extend(flatten(
                    node.body,
                    {**bindings, node.axis: value.constant_value},
                ))
            elif isinstance(node, Loop):
                lower = node.lo.substitute(bindings)
                upper = node.hi.substitute(bindings)
                if lower.constant_value is None or upper.constant_value is None:
                    raise TransformRefused("materialized Loop is not finite")
                for value in range(lower.constant_value, upper.constant_value):
                    output.extend(flatten(
                        node.body,
                        {**bindings, node.axis: value},
                    ))
            elif isinstance(node, Cond):
                predicate = node.pred.substitute(bindings)
                verdict = predicate.evaluate({})
                if verdict is None:
                    raise TransformRefused("materialized Cond is unresolved")
                output.extend(flatten(
                    node.then_body if verdict else node.else_body,
                    bindings,
                ))
            elif isinstance(node, Switch):
                expression = node.expression.substitute(bindings)
                if expression.constant_value is None:
                    raise TransformRefused("materialized Switch is unresolved")
                output.extend(flatten(
                    dict(node.cases).get(expression.constant_value, node.otherwise),
                    bindings,
                ))
        return tuple(output)

    def edit_body(body, active_region=None):
        rebuilt = []
        current_region = active_region
        for node in body:
            if isinstance(node, Region):
                current_region = node.tag
                rebuilt.append(node)
                continue
            if isinstance(node, Loop):
                node = Loop(
                    node.axis, node.lo, node.hi,
                    edit_body(node.body, current_region),
                )
            elif isinstance(node, Cond):
                node = Cond(
                    node.pred,
                    edit_body(node.then_body, current_region),
                    edit_body(node.else_body, current_region),
                )
            elif isinstance(node, Let):
                node = Let(
                    node.axis, node.value, edit_body(node.body, current_region)
                )
            elif isinstance(node, Switch):
                node = Switch(
                    node.expression,
                    tuple(
                        (value, edit_body(case, current_region))
                        for value, case in node.cases
                    ),
                    edit_body(node.otherwise, current_region),
                )
            rebuilt.append(node)
        body = tuple(rebuilt)
        if current_region is not region:
            return body
        output = []
        index = 0
        while index < len(body):
            if not selected_unit(body[index]):
                output.append(body[index])
                index += 1
                continue
            stop = index
            while stop < len(body) and selected_unit(body[stop]):
                stop += 1
            try:
                materialized = flatten(body[index:stop], {})
            except TransformRefused:
                output.extend(body[index:stop])
            else:
                output.extend(sorted(materialized, key=key))
            index = stop
        return tuple(output)

    return program.with_body(edit_body(program.body))


def extend_region(
    program: Program,
    region,
    nodes: tuple[LoopIR, ...],
) -> Program:
    """Append nodes to every matching marker-delimited region in the Program."""

    if not isinstance(nodes, tuple) \
            or not all(isinstance(node, LoopIR) for node in nodes):
        raise IRTypeError("extend_region.nodes must be LoopIR values")

    def edit_body(body):
        rebuilt = []
        for node in body:
            if isinstance(node, Loop):
                node = Loop(
                    node.axis, node.lo, node.hi, edit_body(node.body),
                )
            elif isinstance(node, Cond):
                node = Cond(
                    node.pred,
                    edit_body(node.then_body),
                    edit_body(node.else_body),
                )
            elif isinstance(node, Let):
                node = Let(node.axis, node.value, edit_body(node.body))
            elif isinstance(node, Switch):
                node = Switch(
                    node.expression,
                    tuple((value, edit_body(case)) for value, case in node.cases),
                    edit_body(node.otherwise),
                )
            rebuilt.append(node)
        body = tuple(rebuilt)
        output = []
        index = 0
        while index < len(body):
            marker = body[index]
            output.append(marker)
            index += 1
            if not isinstance(marker, Region) or marker.tag is not region:
                continue
            while index < len(body) and not isinstance(body[index], Region):
                output.append(body[index])
                index += 1
            output.extend(nodes)
        return tuple(output)

    return program.with_body(edit_body(program.body))


def replace_region_operations(
    program: Program,
    region,
    replacements: dict,
) -> Program:
    """Replace logical operations only while a marker-delimited region is active."""

    replacements = dict(replacements)

    def edit_body(body, active_region=None):
        output = []
        current_region = active_region
        for node in body:
            if isinstance(node, Region):
                current_region = node.tag
                output.append(node)
                continue
            if isinstance(node, (Movement, Compute)):
                replacement = (
                    replacements.get(node._identity)
                    if current_region is region else None
                )
                output.append(replacement or node)
            elif isinstance(node, Loop):
                output.append(Loop(
                    node.axis,
                    node.lo,
                    node.hi,
                    edit_body(node.body, current_region),
                ))
            elif isinstance(node, Cond):
                output.append(Cond(
                    node.pred,
                    edit_body(node.then_body, current_region),
                    edit_body(node.else_body, current_region),
                ))
            elif isinstance(node, Let):
                output.append(Let(
                    node.axis,
                    node.value,
                    edit_body(node.body, current_region),
                ))
            elif isinstance(node, Switch):
                output.append(Switch(
                    node.expression,
                    tuple(
                        (value, edit_body(case, current_region))
                        for value, case in node.cases
                    ),
                    edit_body(node.otherwise, current_region),
                ))
        return tuple(output)

    return program.with_body(edit_body(program.body))


def replace_operation_units_in_last_let(
    program: Program,
    region,
    replacements: dict,
) -> Program:
    """Replace whole single-operation units in the final Let of one region.

    A unit may be wrapped by scheduling ``Cond`` nodes.  Replacing the unit,
    rather than only its leaf operation, lets a later transformation state a
    new guard without retaining the superseded one.
    """

    replacements = dict(replacements)

    def identities(node):
        if isinstance(node, (Movement, Compute)):
            return {node._identity}
        children = ()
        if isinstance(node, Loop):
            children = (node.body,)
        elif isinstance(node, Cond):
            children = (node.then_body, node.else_body)
        elif isinstance(node, Let):
            children = (node.body,)
        elif isinstance(node, Switch):
            children = tuple(case for _value, case in node.cases) + (
                node.otherwise,
            )
        return {
            identity
            for child in children
            for item in child
            for identity in identities(item)
        }

    def replace_units(body):
        output = []
        for node in body:
            present = identities(node)
            selected = present & replacements.keys()
            if isinstance(node, Cond) and len(present) == 1 and selected:
                output.append(replacements[next(iter(selected))])
            elif isinstance(node, (Movement, Compute)) \
                    and node._identity in replacements:
                output.append(replacements[node._identity])
            elif isinstance(node, Loop):
                output.append(Loop(
                    node.axis,
                    node.lo,
                    node.hi,
                    replace_units(node.body),
                ))
            elif isinstance(node, Cond):
                output.append(Cond(
                    node.pred,
                    replace_units(node.then_body),
                    replace_units(node.else_body),
                ))
            elif isinstance(node, Let):
                output.append(Let(
                    node.axis,
                    node.value,
                    replace_units(node.body),
                ))
            elif isinstance(node, Switch):
                output.append(Switch(
                    node.expression,
                    tuple(
                        (value, replace_units(case))
                        for value, case in node.cases
                    ),
                    replace_units(node.otherwise),
                ))
            else:
                output.append(node)
        return tuple(output)

    def edit_body(body, active_region=None):
        rebuilt = []
        current_region = active_region
        for node in body:
            if isinstance(node, Region):
                current_region = node.tag
                rebuilt.append(node)
            elif isinstance(node, Loop):
                rebuilt.append(Loop(
                    node.axis,
                    node.lo,
                    node.hi,
                    edit_body(node.body, current_region),
                ))
            elif isinstance(node, Cond):
                rebuilt.append(Cond(
                    node.pred,
                    edit_body(node.then_body, current_region),
                    edit_body(node.else_body, current_region),
                ))
            elif isinstance(node, Let):
                rebuilt.append(Let(
                    node.axis,
                    node.value,
                    edit_body(node.body, current_region),
                ))
            elif isinstance(node, Switch):
                rebuilt.append(Switch(
                    node.expression,
                    tuple(
                        (value, edit_body(case, current_region))
                        for value, case in node.cases
                    ),
                    edit_body(node.otherwise, current_region),
                ))
            else:
                rebuilt.append(node)

        candidates = []
        segment_region = active_region
        for index, node in enumerate(rebuilt):
            if isinstance(node, Region):
                segment_region = node.tag
            elif segment_region is region and isinstance(node, Let):
                candidates.append(index)
        if candidates:
            index = candidates[-1]
            target = rebuilt[index]
            rebuilt[index] = Let(
                target.axis,
                target.value,
                replace_units(target.body),
            )
        return tuple(rebuilt)

    return program.with_body(edit_body(program.body))


def reset_operation_guards(
    program: Program,
    region,
    operations: tuple[Movement | Compute, ...],
) -> Program:
    """Reset invariant-axis equality guards for selected operations in one region."""

    identities = {operation._identity for operation in operations}

    def selected_operations(nodes):
        found = []
        for node in nodes:
            if isinstance(node, (Movement, Compute)) and node._identity in identities:
                found.append(node)
            children = ()
            if isinstance(node, Loop):
                children = (node.body,)
            elif isinstance(node, Cond):
                children = (node.then_body, node.else_body)
            elif isinstance(node, Let):
                children = (node.body,)
            elif isinstance(node, Switch):
                children = tuple(case for _value, case in node.cases) + (
                    node.otherwise,
                )
            for child in children:
                found.extend(selected_operations(child))
        return found

    def edit_body(body, active_region=None):
        output = []
        current_region = active_region
        for node in body:
            if isinstance(node, Region):
                current_region = node.tag
                output.append(node)
                continue
            if isinstance(node, Loop):
                node = Loop(
                    node.axis, node.lo, node.hi,
                    edit_body(node.body, current_region),
                )
            elif isinstance(node, Cond):
                then_body = edit_body(node.then_body, current_region)
                else_body = edit_body(node.else_body, current_region)
                then_selected = selected_operations(then_body)
                else_selected = selected_operations(else_body)
                if (
                    current_region is region
                    and then_selected
                    and else_selected
                    and {
                        operation._identity for operation in then_selected
                    } == {
                        operation._identity for operation in else_selected
                    }
                ):
                    output.extend(then_body)
                    continue
                predicate = node.pred
                selected = then_selected
                if current_region is region and selected:
                    active_ids = set()
                    for operation in selected:
                        if isinstance(operation, Movement):
                            coordinates = {}
                            for index, transfer in enumerate(operation.transfers):
                                tiles = (
                                    transfer.local_tile,
                                    operation.effect_for(index).tile,
                                )
                                for tile in tiles:
                                    for coordinate, expression in enumerate(tile.indices):
                                        coordinates.setdefault(
                                            (tile.operand, coordinate),
                                            [],
                                        ).append(expression)
                            for expressions in coordinates.values():
                                free_axes = {
                                    id(axis): axis
                                    for expression in expressions
                                    for axis in expression.free_axes()
                                }
                                for axis_id, axis in free_axes.items():
                                    extent = axis.extent.constant_value
                                    if extent is None:
                                        active_ids.add(axis_id)
                                        continue
                                    signatures = []
                                    for value in range(int(extent)):
                                        environment = {
                                            other: 0
                                            for other in free_axes.values()
                                        }
                                        environment[axis] = value
                                        signatures.append(tuple(sorted(
                                            expression.evaluate(environment)
                                            for expression in expressions
                                        )))
                                    if any(
                                        signature != signatures[0]
                                        for signature in signatures[1:]
                                    ):
                                        active_ids.add(axis_id)
                        else:
                            active_ids.update(
                                id(axis)
                                for tile in operation.reads + operation.writes
                                for expression in tile.indices
                                for axis in expression.free_axes()
                            )
                    guards = tuple(
                        axis for axis in predicate.free_axes()
                        if id(axis) not in active_ids
                    )
                    bare_guards = tuple(
                        axis for axis in guards
                        if _first_touch_value(predicate, axis) is not None
                    )
                    if len(bare_guards) == 1:
                        predicate = bare_guards[0] == 0
                node = Cond(predicate, then_body, else_body)
            elif isinstance(node, Let):
                node = Let(
                    node.axis, node.value, edit_body(node.body, current_region)
                )
            elif isinstance(node, Switch):
                node = Switch(
                    node.expression,
                    tuple(
                        (value, edit_body(case, current_region))
                        for value, case in node.cases
                    ),
                    edit_body(node.otherwise, current_region),
                )
            output.append(node)
        return tuple(output)

    return program.with_body(edit_body(program.body))


__all__ = [
    "Transform", "compose", "append", "partition_movement",
    "fuse_movements",
    "materialize_steady_copies",
    "substitute_operation", "rewrite_operation", "pipeline",
    "suppress_drain",
    "specialize_first_touch", "defer_single_path_operations",
    "defer_operations", "materialize_region_operations",
    "materialize_operation_runs",
    "prioritize_operations",
    "extend_region", "replace_region_operations",
    "replace_operation_units_in_last_let",
    "reset_operation_guards",
]

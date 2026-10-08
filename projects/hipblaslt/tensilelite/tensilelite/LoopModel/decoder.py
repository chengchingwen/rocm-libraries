# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Theta frontend for the address-agnostic LoopIR."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
import itertools

from . import traversal as geometry
from .loopir import (
    DS_LOAD,
    GLOBAL,
    IMMEDIATE,
    INPUT,
    OUTPUT,
    PROLOGUE,
    REGISTER,
    SHARED,
    SHORT,
    STEADY,
    TENSOR_LOAD,
    WAVE,
    Axis,
    CompletionClass,
    Compute,
    Cond,
    DRAIN,
    Effect,
    Expr,
    Loop,
    Movement,
    Operand,
    Param,
    Program,
    Region,
    Switch,
    Transfer,
    WMMA,
    append,
    compose,
    defer_operations,
    defer_single_path_operations,
    extend_region,
    find_loop,
    materialize_region_operations,
    materialize_operation_runs,
    materialize_steady_copies,
    operations,
    pipeline,
    prioritize_operations,
    replace_region_operations,
    replace_operation_units_in_last_let,
    reset_operation_guards,
    rewrite_operation,
    substitute_operation,
    suppress_drain,
    specialize_first_touch,
    walk,
)
from .schedule import (
    Schedule,
    _read_placement,
    _readahead_shift,
    build_S,
    copy_first_operands,
    loads_in_place,
    preloaded_tiles,
)


READ_COMPLETION = CompletionClass("read", WAVE)
LOAD_COMPLETION = CompletionClass("load", WAVE)


@dataclass(frozen=True)
class ThetaTargetProfile:
    """Target/lowering context deliberately kept outside LoopIR."""

    theta: object
    depths: object
    schedule: object
    read_prefetch: object
    read_advances: object
    read_issue_bindings: object
    read_signatures: object
    register_inplace: frozenset
    shared_inplace: frozenset

    def shared_ring(self, operand: str) -> int:
        if self.theta.op(operand).trajectory.shared is None:
            return 0
        return max(1, int(self.depths.shared(operand, 1) or 1))

    def register_refill_in_place(self, operand: str) -> bool:
        """Whether the finalized read schedule reuses its physical register slot."""

        return operand in self.register_inplace

    def shared_refill_in_place(self, operand: str) -> bool:
        """Whether the finalized copy schedule overwrites its LDS generation."""

        return operand in self.shared_inplace

    def read_prefetch_steps(self, operation):
        return int(self.read_prefetch.get(operation._identity, 0))

    def read_advance(self, operation):
        return int(self.read_advances.get(operation._identity, 0))

    def read_issue(self, operation, environment):
        return tuple(
            (axis.name, int(value.evaluate(environment)))
            for axis, value in self.read_issue_bindings.get(
                operation._identity,
                (),
            )
        )

    def read_issue_axes(self, operation):
        return tuple(
            axis.name
            for axis, _value in self.read_issue_bindings.get(
                operation._identity,
                (),
            )
        )

    @staticmethod
    def _read_signature(operation):
        return tuple(
            (
                transfer.local_tile.operand.name,
                tuple(
                    expression.render()
                    for expression in transfer.local_tile.indices
                ),
                tuple(
                    expression.render()
                    for expression in operation.effect_for(index).tile.indices
                ),
                operation.effect_for(index).where.render(),
            )
            for index, transfer in enumerate(operation.transfers)
        )

    def validate(self, program):
        for operation in operations(program):
            if not isinstance(operation, Movement) \
                    or operation.op is not DS_LOAD:
                continue
            if self._read_signature(operation) not in self.read_signatures.get(
                operation._identity,
                (),
            ):
                raise RuntimeError(
                    "LoopIR read was changed after target facts were finalized; "
                    "re-finalize the decoded schedule before lowering"
                )

    def register_location(
        self,
        operand: str,
        coordinate,
        prefetch_steps=0,
        copy_phase=0,
        placement_environment=None,
        issue_axes=(),
    ):
        coordinate_key = tuple(sorted(
            (getattr(key, "name", key), value)
            for key, value in dict(coordinate).items()
        ))
        placement_key = tuple(sorted(
            (getattr(key, "name", key), value)
            for key, value in dict(placement_environment or {}).items()
        ))
        key = (
            operand,
            coordinate_key,
            int(prefetch_steps),
            int(copy_phase),
            placement_key,
            tuple(issue_axes),
        )
        cached = self._register_locations.get(key)
        if cached is not None:
            return cached

        descriptor = self.theta.op(operand)
        if descriptor.fragment is None:
            return 0, 0, 1
        environment = dict(coordinate)
        labels = tuple(descriptor.fragment.groups()) or ("g0",)
        try:
            group = int(geometry.group_index(self.theta, descriptor, environment))
        except Exception:
            group = 0
        group = min(max(0, group), len(labels) - 1)
        label = labels[group]
        template_key = (
            operand,
            label,
            int(prefetch_steps),
            tuple(issue_axes),
        )
        template = self._register_placement_templates.get(template_key)
        if template is None:
            slots, _source = _read_placement(
                self.theta,
                descriptor,
                self.depths,
                prefetch_steps=int(prefetch_steps),
                groups=(label,),
                plans=self.schedule,
                repeat_axes=tuple(issue_axes),
            )
            expression = slots[0][1]
            width = (
                int(expression.args[1])
                if expression.op == "mod" else 1
            )
            carries = geometry.vgpr_ring_carries(
                self.theta,
                descriptor,
                label,
                width,
            )
            positions = (
                geometry.ring_positions(
                    self.theta,
                    descriptor,
                    label,
                    width,
                )
                if carries else 0
            )
            template = expression, width, carries, positions
            self._register_placement_templates[template_key] = template
        expression, width, carries, positions = template
        bindings = dict(placement_environment or environment)
        for axis in expression.free_axes():
            value = bindings.get(axis.name, bindings.get(axis, 0))
            bindings[axis] = value
            bindings[axis.name] = value
        try:
            slot = int(expression.evaluate(bindings))
        except Exception:
            slot = 0
        if carries:
            phase = int(copy_phase or 0)
            direction = -1 if issue_axes else 1
            slot = (
                slot
                + direction * phase * positions
            ) % width
        result = group, slot, width
        self._register_locations[key] = result
        return result

    @cached_property
    def _register_locations(self):
        return {}

    @cached_property
    def _register_placement_templates(self):
        return {}

    def unit_index(self, operand: str, coordinate) -> int:
        coordinate_key = tuple(sorted(
            (getattr(key, "name", key), value)
            for key, value in dict(coordinate).items()
        ))
        key = operand, coordinate_key
        if key in self._unit_indices:
            return self._unit_indices[key]
        try:
            result = int(geometry.unit_tile_index(
                self.theta,
                self.theta.op(operand),
                dict(coordinate),
            ))
        except Exception:
            result = 0
        self._unit_indices[key] = result
        return result

    @cached_property
    def _unit_indices(self):
        return {}

    def bulk_bytes(self, operand: str) -> int:
        descriptor = self.theta.op(operand)
        return (
            int(geometry.operand_tile_bytes(self.theta, descriptor))
            // max(1, int(descriptor.split))
        )

    def fragment_regs(self, operand: str) -> int:
        descriptor = self.theta.op(operand)
        return (
            int(geometry.frag_regs(self.theta, descriptor))
            if descriptor.fragment is not None else 0
        )

    def region_index(self, operand: str, coordinate) -> int:
        values = tuple(self.theta.op(operand).region_of(coordinate))
        if not values:
            return 0
        result = 0
        extents = {
            axis.name: max(1, int(axis.extent.constant_value))
            for axis in self.theta.inner_axes()
        }
        descriptor = self.theta.op(operand)
        for axis_name, value in zip(descriptor.region_axes, values):
            span = max(1, int(descriptor.region_span.get(axis_name, 1)))
            radix = max(1, extents.get(axis_name, 1) // span)
            result = result * radix + int(value)
        return result

    @cached_property
    def metadata(self) -> dict:
        inner = tuple(self.theta.inner_axes())
        reads = [
            (operand, operand.trajectory.fragment_fill)
            for operand in self.theta.operands
            if operand.trajectory.fragment_fill is not None
        ]
        coverage_factors = {
            operand.name: geometry.coverage_factors(
                self.theta,
                operand,
                movement.coverage,
            )
            for operand, movement in reads
            if movement.coverage is not None
        }
        axis_extents = {
            axis.name: int(axis.extent.constant_value)
            for axis in self.theta.inner_axes()
        }
        coverage_axes = {
            name: tuple(
                (axis, int(factor))
                for axis, factor in factors.items()
                if axis_extents.get(axis) == int(factor)
            )
            for name, factors in coverage_factors.items()
        }
        read_fold = {
            name: {
                axis: int(factor)
                for axis, factor in factors.items()
                if axis_extents.get(axis) != int(factor)
            }
            for name, factors in coverage_factors.items()
        }
        read_fold = {
            name: factors for name, factors in read_fold.items() if factors
        }
        metadata = {
            "buffer_depths": self.depths,
            "vgpr_subdivided": any(
                operand.fragment is not None
                and geometry.vgpr_group_axis(self.theta, operand) is not None
                for operand in self.theta.operands
            ),
            "inner_axis_order": [axis.name for axis in inner],
            "axis_extents": {
                axis.name: int(axis.extent.constant_value)
                for axis in inner
            },
            "summation_axes": sorted(geometry.summation_names(self.theta)),
            "free_axes": {
                operand.name: geometry.free_names(self.theta, operand)
                for operand in self.theta.operands
            },
            "register_layout": geometry.register_layout(
                self.theta,
                self.depths,
            ),
            "region_axes": {
                operand.name: tuple(operand.region_axes)
                for operand in self.theta.operands
            },
            "operand_regions": {
                operand.name: max(1, operand.split)
                for operand in self.theta.operands
            },
            "region_span": {
                operand.name: dict(operand.region_span)
                for operand in self.theta.operands
            },
            "fused_copy_groups": [
                tuple(group) for group in self.theta.fused_copy_groups
            ],
            "unit_regions": {
                key: regions
                for key, _members, regions in self.theta.movement_units()
            },
            "unit_member_regions": {
                key: {
                    member.name: max(1, member.split)
                    for member in members
                }
                for key, members, _regions in self.theta.movement_units()
            },
            "agent_distributed": {
                operand.name: self.theta.agent_distributed(operand.name)
                for operand in self.theta.operands
            },
            "wave_relative_regions": {
                operand.name
                for operand in self.theta.operands
                if operand.trajectory.shared is not None
                and operand.trajectory.shared.regions.wave_relative
            },
            "read_coverage": {
                operand.name: movement.coverage
                for operand, movement in reads
                if movement.coverage is not None
            },
            "coverage_axes": coverage_axes,
            "read_fold": read_fold,
            "mma_inputs": [
                operand.name
                for operand in self.theta.operands
                if operand.is_input and operand.fragment is not None
            ][:2],
            "mma_scales": [
                operand.name
                for operand in self.theta.operands
                if operand.is_input and operand.fragment is not None
            ][2:],
            "per_region_completion": bool(self.theta.per_region_completion),
        }
        return metadata

    def as_meta(self) -> dict:
        return self.metadata


@dataclass(frozen=True)
class DecodedSchedule:
    program: Program
    profile: ThetaTargetProfile


class _Decoder:
    def __init__(self, theta):
        self.theta = theta
        self.depths, _ = build_S(theta)
        self.schedule = Schedule(theta, self.depths)
        requested_steps = self.schedule.read_ahead_steps
        self._register_inplace = {
            (operand.name, group)
            for operand in theta.operands
            if operand.trajectory.fragment_fill is not None
            for group, depth in self.depths.groups_of(operand.name).items()
            if depth < 2 or loads_in_place(
                theta,
                operand,
                requested_steps,
                self.depths,
                groups=(group,),
                plans=self.schedule,
            )
        }
        chunk = theta.summation_chunk_name()
        self._shared_inplace = frozenset(
            operand.name
            for operand in theta.operands
            if operand.trajectory.shared is not None
            and (
                operand.trajectory.shared.ring_depth < 2
                or operand.trajectory.shared.ring_depth
                <= (
                    theta.off_at(operand, TENSOR_LOAD, chunk)
                    if chunk is not None else 0
                )
            )
        )
        self.copy_first = frozenset(
            copy_first_operands(theta, self.depths, self.schedule)
        )
        self.params = []
        self.axes = []
        self.axis_by_name = {}
        self._declare_axes()
        self.wave_axis = None
        self.agents = ()
        if int(getattr(theta, "wave_count", 1) or 1) > 1:
            self.wave_axis = Axis("wave", int(theta.wave_count))
            self.agents = ((self.wave_axis, WAVE),)
        self.operands = {}
        self._declare_operands()
        self.operation_offsets = {}
        self.operation_operand = {}
        self.operation_groups = {}
        self.copy_completions = {}
        self._operand_order = {
            operand.name: index for index, operand in enumerate(theta.operands)
        }
        levels = [
            axis.name for axis in theta.inner_axes()
            if axis.name not in theta.wave_served_axes()
        ]
        positions = {name: index for index, name in enumerate(levels)}
        copy_operands = [
            operand for operand in theta.operands
            if operand.trajectory.shared_fill is not None
        ]

        def read_site(operand):
            level = geometry.operand_level(theta, operand, DS_LOAD)
            return positions.get(level, -1)

        def copy_priority(operand):
            unit = next(
                (
                    members for _key, members, _regions in theta.movement_units()
                    if operand in members
                ),
                [operand],
            )
            return (
                min(read_site(member) for member in unit),
                sum(
                    geometry.operand_tile_bytes(theta, member)
                    // max(1, member.split)
                    for member in unit
                ),
                self._operand_order[operand.name],
            )

        self.copy_operands = tuple(sorted(copy_operands, key=copy_priority))
        self.copy_rank = {}
        for rank, operand in enumerate(self.copy_operands):
            key = next(
                (
                    key for key, members, _regions in theta.movement_units()
                    if operand in members
                ),
                (operand.name,),
            )
            for name in key:
                self.copy_rank.setdefault(name, rank)
        self.read_operands = tuple(
            operand for operand in theta.operands
            if operand.trajectory.fragment_fill is not None
            and not operand.is_output
        )
        self.read_level = {
            operand.name: geometry.operand_level(theta, operand, DS_LOAD)
            for operand in self.read_operands
        }
        self.inner_levels = tuple(
            descriptor for descriptor in theta.inner_axes()
            if descriptor.name not in theta.wave_served_axes()
        )
        self.read_operands = tuple(sorted(
            self.read_operands,
            key=self._read_issue_priority_descriptor,
        ))
        wmma_axes = (
            {
                axis
                for operand in theta.output_operands()
                for axis in geometry.presence(theta, operand)
            }
            | set(geometry.summation_names(theta))
        )
        self.leaf_axis = next(
            (
                descriptor.name
                for descriptor in reversed(self.inner_levels)
                if descriptor.name in wmma_axes
            ),
            self.inner_levels[-1].name if self.inner_levels else None,
        )
        self._read_cache = {}
        self._compute_cache = None

    def _declare_axes(self):
        runtime = 0
        for descriptor in self.theta.ord:
            static_extent = descriptor.extent.constant_value
            if static_extent is not None and int(static_extent) > 0:
                extent = int(static_extent)
            else:
                name = "T" if runtime == 0 else f"{descriptor.name}_extent"
                extent = Param(name)
                self.params.append(extent)
                runtime += 1
            axis = Axis(descriptor.name, extent)
            self.axes.append(axis)
            self.axis_by_name[descriptor.name] = axis

    def _operand_axis_names(self, descriptor):
        names = []
        if descriptor.is_input:
            names.extend(axis.name for axis in self.theta.outer_axes())
        names.extend(geometry.presence(self.theta, descriptor))
        return tuple(dict.fromkeys(name for name in names if name in self.axis_by_name))

    def _declare_operands(self):
        for descriptor in self.theta.operands:
            self.operands[descriptor.name] = Operand(
                descriptor.name,
                tuple(
                    self.axis_by_name[name]
                    for name in self._operand_axis_names(descriptor)
                ),
                OUTPUT if descriptor.is_output else INPUT,
            )

    def _tile(self, descriptor, offsets=None, replacements=None):
        offsets = dict(offsets or ())
        replacements = dict(replacements or ())
        operand = self.operands[descriptor.name]
        return operand[tuple(
            (axis + int(offsets.get(axis.name, 0) or 0)).substitute(replacements)
            for axis in operand.axes
        )]

    def _effect(
        self,
        descriptor,
        local_tile,
        *,
        semantic_tile=None,
        region=None,
    ):
        """Ask the frontend wiring for this explicit semantic effect."""

        tile = semantic_tile or local_tile
        if descriptor.effect is not None:
            return descriptor.effect(tile, self.wave_axis, region)
        return Effect(tile)

    def _read_tiles(self, descriptor):
        tile = self._tile(descriptor)
        movement = descriptor.trajectory.fragment_fill
        factors = (
            geometry.coverage_factors(
                self.theta,
                descriptor,
                movement.coverage,
            )
            if movement is not None and movement.coverage is not None
            else {}
        )
        choices = []
        for axis, expression in zip(tile.operand.axes, tile.indices):
            extent = max(1, int(factors.get(axis.name, 1) or 1))
            base = expression - expression % extent
            choices.append(tuple(base + offset for offset in range(extent)))
        return tuple(
            tile.operand[tuple(indices)]
            for indices in itertools.product(*choices)
        )

    def _movement(self, descriptor, movement):
        op = movement.op
        tiles = (
            (self._tile(descriptor),)
            if movement.op is TENSOR_LOAD
            else self._read_tiles(descriptor)
        )
        operation = Movement(
            op,
            tuple(
                Transfer(
                    tile,
                    movement.source.space,
                    movement.destination.space,
                    effect=self._effect(descriptor, tile),
                )
                for tile in tiles
            ),
            (
                self._copy_completion((descriptor.name,), 0)
                if movement.op is TENSOR_LOAD
                else READ_COMPLETION
                if movement.src is SHARED
                else LOAD_COMPLETION
            ),
        )
        self.operation_offsets[operation._identity] = dict(movement.offsets or {})
        self.operation_operand[operation._identity] = descriptor
        return operation

    def _copy_completion(self, unit, region):
        key = (
            tuple(unit),
            int(region) if self.theta.per_region_completion else None,
        )
        return self.copy_completions.setdefault(
            key,
            CompletionClass("copy", WAVE),
        )

    def _nested(self, node, axis_names):
        body = (node,)
        for descriptor in reversed(self.theta.inner_axes()):
            if descriptor.name not in axis_names:
                continue
            axis = self.axis_by_name[descriptor.name]
            body = (Loop(axis, 0, axis.extent, body),)
        return body

    def _movement_axes(self, descriptor, movement):
        if movement.op is TENSOR_LOAD:
            return set(descriptor.region_axes)
        try:
            return {
                axis.name
                for axis in geometry.presence_axes(
                    self.theta,
                    descriptor,
                    hop=movement,
                )
            }
        except Exception:
            return set(self._operand_axis_names(descriptor))

    def _copy_nodes(self):
        by_name = {operand.name: operand for operand in self.theta.operands}
        output = []
        handled = set()
        units = list(self.theta.movement_units())
        units.sort(key=lambda unit: min(
            self.copy_rank.get(member.name, len(self.copy_rank))
            for member in unit[1]
        ))

        def region_values(member, region):
            result = {}
            value = int(region)
            for name in reversed(tuple(member.region_axes)):
                axis = self.axis_by_name[name]
                span = max(1, int(member.region_span.get(name, 1)))
                radix = max(
                    1,
                    int(axis.extent.constant_value) // span,
                )
                result[axis] = (value % radix) * span
                value //= radix
            return result

        for _key, members, region_count in units:
            members = [
                member for member in members
                if member.name in by_name
                and member.trajectory.shared_fill is not None
            ]
            if not members:
                continue
            driver = max(members, key=lambda member: max(1, member.split))
            active = set(driver.region_axes)
            for region in range(max(1, int(region_count))):
                transfers = []
                for member in members:
                    if region >= max(1, int(member.split)):
                        continue
                    movement = member.trajectory.shared_fill
                    local_tile = self._tile(
                        member,
                        replacements=region_values(member, region),
                    )
                    semantic_tile = self._tile(member)
                    covered_axes = tuple(
                        axis for axis in local_tile.operand.axes
                        if axis.name in active
                        and axis.name not in member.region_axes
                        and axis.extent.constant_value is not None
                        and int(axis.extent.constant_value) > 1
                    )
                    covered_values = tuple(
                        range(int(axis.extent.constant_value))
                        for axis in covered_axes
                    )
                    for values in (
                        itertools.product(*covered_values)
                        if covered_values else ((),)
                    ):
                        covered_by_id = {
                            id(axis): value
                            for axis, value in zip(covered_axes, values)
                        }
                        member_tile = (
                            local_tile.operand[tuple(
                                (
                                    Expr.constant(covered_by_id[id(operand_axis)])
                                    if id(operand_axis) in covered_by_id
                                    else expression
                                )
                                for operand_axis, expression in zip(
                                    local_tile.operand.axes,
                                    local_tile.indices,
                                )
                            )]
                            if covered_axes else local_tile
                        )
                        semantic_member = (
                            semantic_tile.operand[tuple(
                                (
                                    Expr.constant(covered_by_id[id(operand_axis)])
                                    if id(operand_axis) in covered_by_id
                                    else expression
                                )
                                for operand_axis, expression in zip(
                                    semantic_tile.operand.axes,
                                    semantic_tile.indices,
                                )
                            )]
                            if covered_axes else semantic_tile
                        )
                        transfers.append(Transfer(
                            member_tile,
                            movement.source.space,
                            movement.destination.space,
                            effect=self._effect(
                                member,
                                member_tile,
                                semantic_tile=semantic_member,
                                region=region,
                            ),
                        ))
                    handled.add(member.name)
                if transfers:
                    operation = Movement(
                        TENSOR_LOAD,
                        tuple(transfers),
                        self._copy_completion(
                            tuple(member.name for member in members),
                            region,
                        ),
                    )
                    offsets = {}
                    for member in members:
                        offsets.update(dict(
                            member.trajectory.shared_fill.offsets or {}
                        ))
                    self.operation_offsets[operation._identity] = offsets
                    output.append(operation)
        for descriptor in self.copy_operands:
            movement = descriptor.trajectory.shared_fill
            if movement is None or descriptor.name in handled:
                continue
            operation = self._movement(descriptor, movement)
            output.extend(self._nested(
                operation,
                self._movement_axes(descriptor, movement),
            ))
        return tuple(output)

    def _reads_for_operand(self, descriptor):
        cached = self._read_cache.get(descriptor.name)
        if cached is not None:
            return cached
        movement = descriptor.trajectory.fragment_fill
        groups = tuple(descriptor.fragment.groups()) or ("g0",)
        values = []
        for group in groups:
            operation = self._movement(descriptor, movement)
            self.operation_groups[operation._identity] = (group,)
            values.append(operation)
        values = tuple(values)
        self._read_cache[descriptor.name] = values
        return values

    def _compute_node(self):
        if self._compute_cache is not None:
            return self._compute_cache
        reads = []
        writes = []
        read_names = {descriptor.name for descriptor in self.read_operands}
        for descriptor in self.theta.operands:
            if descriptor.name not in read_names:
                continue
            access = self._tile(descriptor)
            reads.extend(
                access for _group in (
                    tuple(descriptor.fragment.groups()) or ("g0",)
                )
            )
        for descriptor in self.theta.output_operands():
            if descriptor.fragment is not None:
                writes.append(self._tile(descriptor))
        self._compute_cache = Compute(
            WMMA,
            tuple(reads),
            tuple(writes),
            IMMEDIATE,
        )
        return self._compute_cache

    def _repeat_read_axes(self, descriptor):
        if descriptor.fragment is None \
                or descriptor.name.startswith("MXS") \
                or geometry.reloads_whole_set(self.theta, descriptor):
            return ()
        want = self.schedule.want(descriptor)
        shallow = any(
            geometry.group_ring_depth(
                self.theta, descriptor, group, self.depths
            ) < geometry.group_reuse_floor(
                self.theta,
                descriptor,
                group,
                charge_lead=False,
                lead=0,
            )
            or (
                descriptor.fragment.va_time_shares_regions
                and geometry.group_ring_depth(
                    self.theta, descriptor, group, self.depths
                ) < geometry.group_live_peak(
                    self.theta,
                    descriptor,
                    group,
                    off=geometry.prefetch_distance_for(
                        self.theta,
                        descriptor,
                        self.schedule.steps(descriptor, group, want),
                    ),
                )
            )
            for group in descriptor.fragment.groups()
        )
        if not shallow or self.read_level[descriptor.name] not in {
            axis.name for axis in self.inner_levels
        }:
            return ()
        varying = set(geometry.varying_axes(self.theta, descriptor))
        level = next(
            index for index, axis in enumerate(self.inner_levels)
            if axis.name == self.read_level[descriptor.name]
        )
        return tuple(
            axis.name for axis in self.inner_levels[:level]
            if axis.name not in varying
            and int(axis.extent.constant_value) > 1
        )

    def _read_issue_priority_descriptor(self, descriptor):
        groups = tuple(descriptor.fragment.groups()) or ("g0",)
        wanted = self.schedule.want(descriptor)
        depth = min(
            self.schedule.steps(descriptor, group, wanted)
            for group in groups
        )
        advance, strides, span, extents = _readahead_shift(
            self.theta,
            descriptor,
            depth,
            self.depths,
            plans=self.schedule,
            repeat_axes=self._repeat_read_axes(descriptor),
        )
        coverage = geometry.read_coverage(self.theta, descriptor)
        varying = set(geometry.varying_axes(self.theta, descriptor))
        environment = {
            axis.name: 0 for axis in self.theta.inner_axes()
        }
        coordinate = tuple(
            int(geometry._shifted_coord(
                self.theta,
                axis,
                advance,
                strides,
                span,
                extents,
                coverage,
            ).evaluate(environment))
            if axis.name in varying else 0
            for axis in self.theta.inner_axes()
        )
        width = min(
            max(1, int(self.depths.get(descriptor.name, group)))
            for group in groups
        )
        return (
            coordinate,
            self.copy_rank.get(descriptor.name, len(self.copy_rank)),
            geometry.frag_regs(self.theta, descriptor),
            depth,
            width,
            self._operand_order[descriptor.name],
        )

    def _first_touch_axes(self, descriptor, depth):
        varying = set(geometry.varying_axes(self.theta, descriptor))
        repeated = set(self._repeat_read_axes(descriptor))
        values = [
            (axis.name, 0)
            for axis in self.inner_levels[:depth]
            if axis.name not in varying and axis.name not in repeated
        ]
        coverage = geometry.read_coverage(self.theta, descriptor)
        values.extend(
            (axis.name, int(coverage[axis.name]))
            for axis in self.inner_levels
            if axis.name in coverage
        )
        return tuple(dict(values).items())

    def _first_touch_guard(self, node, first_touch, values):
        body = (node,)
        for name, step in reversed(first_touch):
            axis = self.axis_by_name[name]
            expression = axis % step if step > 1 else axis
            body = (Cond(expression == int(values.get(name, 0)), body, ()),)
        return body[0]

    def _reload_cycle(self, descriptor, groups, span):
        if geometry.reloads_whole_set(self.theta, descriptor) \
                or descriptor.name.startswith("MXS"):
            return max(1, int(span))
        selected = tuple(groups or descriptor.fragment.groups())
        width = min(
            geometry.group_ring_depth(
                self.theta, descriptor, group, self.depths
            )
            for group in selected
        )
        time_shares_regions = any(
            geometry.regions_lead_the_ring(self.theta, descriptor, group)
            and descriptor.fragment.va_time_shares_regions
            for group in selected
        )
        unit_positions = (
            geometry.group_prefetch_unit_positions(self.theta, descriptor)
            if time_shares_regions
            or geometry.vgpr_group_axis(self.theta, descriptor) is not None
            else geometry.rotation_unit_positions(self.theta, descriptor)
        )
        return max(
            1,
            geometry.group_register_positions(self.theta, descriptor),
            width * unit_positions,
        )

    def _read_anchor(self, descriptor, groups, depth, invariant):
        if not invariant:
            return {}
        wanted = {}
        for group in (groups or descriptor.fragment.groups()):
            positions = geometry.reload_anchor(
                self.theta,
                descriptor,
                group,
                max(1, self.depths.get(descriptor.name, group)),
                depth,
            )
            if positions is None:
                return {}
            for coordinate in positions.values():
                for name, value in coordinate.items():
                    if name in invariant and value:
                        wanted.setdefault(name, value)
        return wanted

    def _place_read(self, operation, depth):
        descriptor = self.operation_operand[operation._identity]
        groups = self.operation_groups.get(operation._identity)
        advance, span, position, _replacements = self._read_retiming(operation)
        first_touch = self._first_touch_axes(descriptor, depth)
        invariant = {name for name, step in first_touch if step == 0}
        reload_cycle = self._reload_cycle(descriptor, groups, span)
        remainder = advance % reload_cycle if advance else 0
        late = {
            name: max(0, int(self.axis_by_name[name].extent.constant_value) - 1)
            for name in invariant
        }
        if remainder and invariant:
            selected = tuple(groups or descriptor.fragment.groups())
            carries = any(
                geometry.vgpr_ring_carries(
                    self.theta,
                    descriptor,
                    group,
                    geometry.group_ring_depth(
                        self.theta, descriptor, group, self.depths
                    ),
                )
                for group in selected
            )
            guard_cycle = (
                max(reload_cycle, span)
                if carries and not self._repeat_read_axes(descriptor)
                else reload_cycle
            )
            nowrap = position % guard_cycle < reload_cycle - remainder
            return (
                Cond(
                    nowrap,
                    (self._first_touch_guard(operation, first_touch, {}),),
                    (self._first_touch_guard(operation, first_touch, late),),
                ),
                True,
            )
        if advance >= reload_cycle and not remainder and invariant:
            anchor = late
        else:
            anchor = self._read_anchor(
                descriptor, groups, advance, invariant
            )
        return (
            self._first_touch_guard(operation, first_touch, anchor),
            bool(advance),
        )

    def _suppressed_read(self, operation):
        descriptor = self.operation_operand[operation._identity]
        groups = self.operation_groups.get(operation._identity)
        advance, span, position, _replacements = self._read_retiming(operation)
        depth = next(
            (
                index for index, axis in enumerate(self.inner_levels)
                if axis.name == self.read_level[descriptor.name]
            ),
            None,
        )
        if depth is None:
            return operation
        first_touch = self._first_touch_axes(descriptor, depth)
        invariant = {name for name, step in first_touch if step == 0}
        anchor = self._read_anchor(
            descriptor,
            groups,
            advance,
            invariant,
        )
        node = self._first_touch_guard(operation, first_touch, anchor)
        return Cond(
            position + advance < span,
            (node,),
            (),
        )

    def _inner_nest(self):
        if not self.inner_levels:
            return (self._compute_node(),)

        by_level = {descriptor.name: [] for descriptor in self.inner_levels}
        head = []
        for descriptor in self.read_operands:
            target = by_level.get(self.read_level[descriptor.name], head)
            target.extend(self._reads_for_operand(descriptor))

        def build(depth):
            descriptor = self.inner_levels[depth]
            candidates = by_level[descriptor.name]
            leading = []
            trailing = []
            for operation in candidates:
                node, runs_ahead = self._place_read(operation, depth)
                (trailing if runs_ahead else leading).append(node)
            body = list(leading)
            if descriptor.name == self.leaf_axis:
                body.append(self._compute_node())
            if depth + 1 < len(self.inner_levels):
                body.append(build(depth + 1))
            body.extend(trailing)
            axis = self.axis_by_name[descriptor.name]
            return Loop(axis, 0, axis.extent, tuple(body))

        return tuple(head) + (build(0),)

    def _store_nodes(self):
        output = []
        for descriptor in self.theta.operands:
            if not descriptor.is_output:
                continue
            for movement in descriptor.movements:
                operation = self._movement(descriptor, movement)
                output.extend(self._nested(
                    operation,
                    set(self._operand_axis_names(descriptor)),
                ))
        return tuple(output)

    def _iteration_tree(self):
        body = self._copy_nodes() + self._inner_nest() + self._store_nodes()
        for descriptor in reversed(self.theta.outer_axes()):
            axis = self.axis_by_name[descriptor.name]
            body = (Loop(axis, 0, axis.extent, body),)
        return body

    def _apply_pipelines(self, program):
        current = program
        for descriptor in reversed(self.theta.ord):
            if descriptor.extent.constant_value != 0:
                continue
            axis = self.axis_by_name[descriptor.name]
            specs = tuple(
                (operation, int(self.operation_offsets[operation._identity][descriptor.name]))
                for operation in operations(current)
                if int(self.operation_offsets.get(operation._identity, {}).get(
                    descriptor.name, 0
                ) or 0) > 0
            )
            if not specs:
                continue
            loop = find_loop(current, axis)
            current = pipeline(current, loop.node, specs)
        return current

    def _operation_runs_ahead(self, operation):
        descriptor = self.operation_operand.get(operation._identity)
        return (
            isinstance(operation, Movement)
            and operation.op is DS_LOAD
            and descriptor is not None
            and int(self.schedule.want(descriptor) or 0) > 0
        )

    def _operation_needs_defer(self, operation):
        if not self._operation_runs_ahead(operation):
            return False
        descriptor = self.operation_operand[operation._identity]
        groups = self.operation_groups.get(operation._identity)
        selected = tuple(groups or descriptor.fragment.groups())
        return any(
            (descriptor.name, group) in self._register_inplace
            for group in selected
        )

    def _read_retiming(self, operation):
        descriptor = self.operation_operand.get(operation._identity)
        if descriptor is None or not self._operation_runs_ahead(operation):
            return 0, 1, 0, {}
        groups = self.operation_groups.get(operation._identity)
        wanted = self.schedule.want(descriptor)
        depth = min(
            self.schedule.steps(descriptor, group, wanted)
            for group in (groups or descriptor.fragment.groups())
        )
        advance, strides, span, extents = _readahead_shift(
            self.theta,
            descriptor,
            depth,
            self.depths,
            groups=groups,
            plans=self.schedule,
            repeat_axes=self._repeat_read_axes(descriptor),
        )
        coverage = geometry.read_coverage(self.theta, descriptor)
        position = Expr.constant(0)
        for name, coefficient in strides.items():
            axis = self.axis_by_name[name]
            position = position + coefficient * (
                axis // max(1, int(coverage.get(name, 1)))
            )
        shifted = position + int(advance)
        replacements = {}
        for name, stride in strides.items():
            axis = self.axis_by_name[name]
            extent = max(1, int(extents.get(name, 1)))
            fold = max(1, int(coverage.get(name, 1)))
            digit = (shifted // max(1, int(stride))) % extent
            replacements[axis] = fold * digit if fold > 1 else digit
        outer = self.theta.reduction_chunk_mode()
        if outer is not None and outer.name in self.axis_by_name:
            axis = self.axis_by_name[outer.name]
            replacements[axis] = axis + shifted // max(1, int(span))
        return int(advance), max(1, int(span)), position, replacements

    def _apply_inner_shifts(self, program):
        current = program
        for operation in tuple(operations(current)):
            advance, _span, _position, replacements = self._read_retiming(
                operation
            )
            if advance:
                current = rewrite_operation(
                    current,
                    operation,
                    lambda node, replacements=replacements: substitute_operation(
                        node,
                        replacements,
                    ),
                )
        return current

    def _prefetch_order_key(self, operation):
        descriptor = self.operation_operand[operation._identity]
        tile = operation.transfers[0].local_tile
        by_axis = {
            axis.name: expression.constant_value
            for axis, expression in zip(tile.operand.axes, tile.indices)
        }
        coordinate = tuple(
            int(by_axis.get(axis.name) or 0)
            for axis in self.theta.inner_axes()
        )
        return (
            coordinate,
            self.copy_rank.get(descriptor.name, len(self.copy_rank)),
            geometry.frag_regs(self.theta, descriptor),
            self._operand_order[descriptor.name],
        )

    def _copy_order_key(self, operation):
        coordinates = []
        for transfer in operation.transfers:
            by_axis = {
                axis.name: expression.constant_value
                for axis, expression in zip(
                    transfer.local_tile.operand.axes,
                    transfer.local_tile.indices,
                )
            }
            coordinates.append(tuple(
                int(by_axis.get(axis.name) or 0)
                for axis in self.theta.ord
            ))
        unit = tuple(
            transfer.local_tile.operand.name
            for transfer in operation.transfers
        )
        return (
            min(coordinates),
            min(self.copy_rank.get(name, len(self.copy_rank)) for name in unit),
            sum(
                geometry.operand_tile_bytes(self.theta, self.theta.op(name))
                // max(1, self.theta.op(name).split)
                for name in unit
            ),
            min(self._operand_order[name] for name in unit),
        )

    def _fill_reads(self):
        events = []
        axis_names = tuple(axis.name for axis in self.theta.inner_axes())
        for descriptor in self.read_operands:
            hop_coverage = descriptor.trajectory.fragment_fill.coverage
            coverage = (
                geometry.coverage_factors(
                    self.theta, descriptor, hop_coverage
                )
                if hop_coverage is not None else {}
            )
            seen_carriers = set()
            coordinates = preloaded_tiles(
                self.theta,
                descriptor,
                self.schedule.want(descriptor),
                self.depths,
                self.schedule,
            )
            operations_for_operand = self._reads_for_operand(descriptor)
            for ordinal, coordinate in enumerate(coordinates):
                carrier = tuple(
                    (
                        name,
                        int(coordinate.get(name, 0))
                        // max(1, int(coverage.get(name, 1))),
                    )
                    for name in axis_names
                )
                if carrier in seen_carriers:
                    continue
                seen_carriers.add(carrier)
                try:
                    group = geometry.group_index(
                        self.theta,
                        descriptor,
                        coordinate,
                    )
                except Exception:
                    group = 0
                group = min(max(0, int(group)), len(operations_for_operand) - 1)
                operation = substitute_operation(
                    operations_for_operand[group],
                    {
                        self.axis_by_name[name]: value
                        for name, value in coordinate.items()
                        if name in self.axis_by_name
                    },
                )
                use = tuple(int(coordinate.get(name, 0)) for name in axis_names)
                groups = tuple(descriptor.fragment.groups()) or ("g0",)
                width = min(
                    max(1, int(self.depths.get(descriptor.name, group)))
                    for group in groups
                )
                events.append((
                    use,
                    self.copy_rank.get(descriptor.name, len(self.copy_rank)),
                    geometry.frag_regs(self.theta, descriptor),
                    0,
                    width,
                    self._operand_order[descriptor.name],
                    ordinal,
                    operation,
                ))
        events.sort(key=lambda event: event[:-1])
        return tuple(event[-1] for event in events)

    def build(self):
        iteration_tree = self._iteration_tree()
        seed = Program(
            tuple(self.params),
            tuple(self.axes),
            tuple(self.operands.values()),
            (),
            self.agents,
        )

        def insert_steady(program):
            return append(program, iteration_tree)

        program = compose(
            insert_steady,
            self._apply_inner_shifts,
            self._apply_pipelines,
        )(seed)
        if self.theta.suppress_no_load_loop:
            program = suppress_drain(program)
        shifted = {
            operation._identity: operation
            for operation in operations(program)
            if isinstance(operation, Movement)
            and operation.op is DS_LOAD
        }
        suppressed = {
            identity: self._suppressed_read(operation)
            for identity, operation in shifted.items()
            if self._operation_runs_ahead(operation)
        }
        if suppressed:
            program = replace_operation_units_in_last_let(
                program,
                DRAIN,
                suppressed,
            )
        program = replace_region_operations(
            program,
            SHORT,
            {
                operation._identity: operation
                for operations_for_operand in self._read_cache.values()
                for operation in operations_for_operand
            },
        )
        program = reset_operation_guards(
            program,
            SHORT,
            tuple(
                operation
                for operations_for_operand in self._read_cache.values()
                for operation in operations_for_operand
            ),
        )
        program = prioritize_operations(
            program,
            SHORT,
            tuple(
                operation
                for operations_for_operand in self._read_cache.values()
                for operation in operations_for_operand
            ),
        )
        program = prioritize_operations(
            program,
            SHORT,
            tuple(
                operation for operation in operations(program)
                if isinstance(operation, Movement)
                and operation.op is TENSOR_LOAD
            ),
        )
        fills = self._fill_reads()
        if fills:
            program = extend_region(program, PROLOGUE, fills)
        program = materialize_region_operations(
            program,
            PROLOGUE,
            lambda operation: (
                isinstance(operation, Movement)
                and operation.op is DS_LOAD
            ),
            self._prefetch_order_key,
        )
        program = specialize_first_touch(program, STEADY)
        program = specialize_first_touch(program, DRAIN)
        in_place_reads = tuple(
            operation for operation in operations(program)
            if isinstance(operation, Movement)
            and operation.op is DS_LOAD
            and self._operation_needs_defer(operation)
        )
        for tag in (STEADY, DRAIN):
            program = defer_single_path_operations(
                program,
                tag,
                in_place_reads,
            )
        program = defer_operations(
            program,
            STEADY,
            tuple(
                operation for operation in operations(program)
                if isinstance(operation, Movement)
                and operation.op is TENSOR_LOAD
                and not any(
                    transfer.local_tile.operand.name in self.copy_first
                    for transfer in operation.transfers
                )
            ),
        )
        for tag in (PROLOGUE, STEADY, DRAIN, SHORT):
            program = materialize_operation_runs(
                program,
                tag,
                lambda operation: (
                    isinstance(operation, Movement)
                    and operation.op is TENSOR_LOAD
                ),
                self._copy_order_key,
            )
        program = materialize_steady_copies(
            program,
            max(1, int(geometry.ring_trip_copies(
                self.theta,
                self.depths,
            ))),
        )
        if not any(isinstance(node, Region) for node in walk(program)):
            program = program.with_body(
                (Region(STEADY), *program.body),
                check=False,
            )
        program.check()
        final_reads = tuple(
            operation for operation in operations(program)
            if isinstance(operation, Movement)
            and operation.op is DS_LOAD
        )
        read_prefetch = {
            operation._identity: int(
                self.schedule.want(
                    self.operation_operand[operation._identity]
                ) or 0
            )
            for operation in final_reads
        }
        read_advances = {
            operation._identity: int(self._read_retiming(operation)[0])
            for operation in final_reads
        }
        read_issue_bindings = {}
        for operation in final_reads:
            descriptor = self.operation_operand[operation._identity]
            replacements = self._read_retiming(operation)[3]
            read_issue_bindings[operation._identity] = tuple(
                (
                    self.axis_by_name[name],
                    replacements.get(
                        self.axis_by_name[name],
                        Expr.axis_value(self.axis_by_name[name]),
                    ),
                )
                for name in self._repeat_read_axes(descriptor)
            )
        read_signatures = {}
        for operation in final_reads:
            read_signatures.setdefault(
                operation._identity,
                set(),
            ).add(ThetaTargetProfile._read_signature(operation))
        return DecodedSchedule(
            program,
            ThetaTargetProfile(
                self.theta,
                self.depths,
                self.schedule,
                read_prefetch,
                read_advances,
                read_issue_bindings,
                {
                    identity: frozenset(signatures)
                    for identity, signatures in read_signatures.items()
                },
                frozenset(
                    operand
                    for operand, _group in self._register_inplace
                ),
                self._shared_inplace,
            ),
        )


def decode_theta(theta) -> DecodedSchedule:
    return _Decoder(theta).build()

# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Every derived decision: how many buffers, how far ahead, where each reload sits."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

from . import traversal as geometry
from .traversal import (_inner_steps, group_live_peak, group_reuse_floor, group_ring_size,
                        lds_buffers, readahead_level)
from .ir import COPY, Expr, Placement, READ, REVERSE, SHARED_GROUP, Space
from .theta import DepthMap, path_direction
from .traversal import (CLOBBER, _divisors, _summation_ring_axis,
                        axis_strides, broadcast_width, chunks_crossed,
                        transfer_extents, is_uniform_over, operand_level,
                        position_terms, prefetch_axis_name,
                        prefetch_distance_for, coverage_tile_cap, read_coverage,
                        requested_read_ahead,
                        register_reuse_verdict, reload_modes, reload_positions,
                        reloads_whole_set, ring_axes, ring_slot,
                        varying_axes)
from .traversal import free_axes, presence_axes


# --- prefetch --------------------------------------------------------------

def readahead_reach(theta, depths, plans) -> int:
    """How many whole reduction chunks the prefetch crosses into, over every group."""
    deepest = 0
    for operand in theta.operands:
        if not (operand.hops and operand.hops[-1].dst == Space.REGISTER
                and operand.hops[-1].src == Space.SHARED):
            continue  # only shared->register reads read ahead
        prefetch_axis = prefetch_axis_name(theta, operand)  # PER operand: the axis it varies on
        if prefetch_axis is None:
            continue
        prefetch_steps = plans.want(operand)
        if prefetch_steps <= 0:
            continue
        for group in operand.fragment.groups():
            steps = plans.steps(operand, group, prefetch_steps)
            deepest = max(deepest, chunks_crossed(theta, operand, steps))
    return deepest


def copy_must_be_first(theta, operand, depths, plans=None):
    plans = Schedule(theta, depths) if plans is None else plans
    outer = theta.summation_chunk_name()
    if outer is None:
        return False
    prefetch_steps = plans.want(operand)
    crossed = max((chunks_crossed(theta, operand, plans.steps(operand, group, prefetch_steps))
                   for group in operand.fragment.groups()), default=0)
    copy_offset = theta.off_at(operand.name, COPY, outer)
    # Equality can lead only when the copy lands in a different LDS slot.  At S_shared <= off it
    # wraps onto the slab still being read and remains a WAR that must trail the reads.
    distinct_slot = lds_buffers(theta, operand) > copy_offset
    return crossed >= 1 and copy_offset == crossed and distinct_slot


def copy_first_operands(theta, depths, plans=None):
    """{operand name} whose shared copy must lead this trip's reads -- the reorder input."""
    plans = Schedule(theta, depths) if plans is None else plans
    return frozenset(operand.name for operand in theta.operands
                     if any(hop.dst == Space.SHARED for hop in operand.hops)
                     and copy_must_be_first(theta, operand, depths, plans))


def chunk_crossing_violations(theta, depths, plans=None):
    """The coupling `off(copy, iter) >= r + 1`, per operand -- [] when theta is emittable."""
    plans = Schedule(theta, depths) if plans is None else plans
    outer = theta.summation_chunk_name()
    if outer is None:
        return []
    bad = []
    for operand in theta.operands:
        if not any(hop.dst == Space.SHARED for hop in operand.hops):
            continue  # no shared hop -> no shared residency
        # The DERIVED depth, as `copy_must_be_first` reads it: the raw `off` charges a full-inner
        # operand a crossing its clamped depth never makes, and refuses the kernel for it.
        prefetch_steps = plans.want(operand)
        for group in operand.fragment.groups():
            steps = plans.steps(operand, group, prefetch_steps)
            crossed = chunks_crossed(theta, operand, steps)
            if crossed <= 0:
                continue
            copy_offset = theta.off_at(operand.name, COPY, outer)
            need = crossed if copy_must_be_first(theta, operand, depths, plans) else crossed + 1
            if copy_offset < need:
                bad.append((operand.name, steps, crossed, copy_offset, need))
                break  # one report per operand is enough
    return bad


@dataclass(frozen=True)
class PeelDepths:
    """The peel depth per level: how many iterations each level lifts out of the loop."""
    per_level: dict
    offsets: dict
    requested_steps: int
    chunk: object
    reach: int = 0
    deferrals: dict = field(default_factory=dict)

    def depth(self, level) -> int:
        """How many steps of `level` are lifted out of the loop; 0 means it is not peeled."""
        return int(self.per_level.get(level, 0))

    @property
    def chunk_depth(self) -> int:
        """The peel at the reduction chunk -- the one level the emitter currently peels."""
        return self.depth(self.chunk)

    def peeled_levels(self) -> list:
        """Every level with `M_l > 0`, outer->inner."""
        return list(self.per_level)

    def copy_off(self) -> dict:
        return {name: depth for (name, role, level), depth in self.offsets.items()
                if role == COPY and level == self.chunk}


def _read_offsets(theta):
    """Every non-zero `off` theta carries, split by sign: forward peels and rejected deferrals."""
    offsets, deferrals = {}, {}
    for operand in theta.operands:
        if operand.hops and path_direction(operand) == REVERSE:
            raise RuntimeError(
                f"operand {operand.name!r} has a REVERSE trajectory "
                f"({[f'{h.src}->{h.dst}' for h in operand.hops]}): its hops run register->shared"
                f"->global, and the mirrored peel that shape needs is not emitted yet. Refusing "
                f"rather than peeling it forward, because the hop role still says 'read' for the "
                f"read-back leg, so a role-keyed peel would look correct and run backwards.")
        for hop in operand.hops:
            for axis in theta.ord:
                depth = theta.off_at(operand, hop.role, axis.name)
                if depth > 0:
                    offsets[(operand.name, hop.role, axis.name)] = depth
                elif depth < 0:
                    deferrals[(operand.name, hop.role, axis.name)] = depth
    return offsets, deferrals


def _deepest_read_ahead(theta, depths=None):
    """The largest read-ahead any shared->register read asks for, in its own axis's steps."""
    asked = [requested_read_ahead(theta, operand, depths) for operand in theta.operands
             if operand.hops and operand.hops[-1].dst == Space.REGISTER
             and operand.hops[-1].src == Space.SHARED]
    return max(0, max(asked, default=0))


def peel_depths(theta, depths, plans=None):
    """The peel depth of every level, and the offsets it was derived from."""
    plans = Schedule(theta, depths) if plans is None else plans
    chunk = theta.summation_chunk_name()
    offsets, deferrals = _read_offsets(theta)

    per_level = {}
    for (_name, _role, level), depth in offsets.items():
        per_level[level] = max(depth, per_level.get(level, 0))

    prefetch_steps = max((plans.want(operand) for operand in theta.operands
                          if operand.hops and operand.hops[-1].dst == Space.REGISTER
                          and operand.hops[-1].src == Space.SHARED), default=0)
    reach = readahead_reach(theta, depths, plans) if prefetch_steps else 0
    if reach and chunk is not None:
        per_level[chunk] = max(reach, per_level.get(chunk, 0))
    per_level = {axis.name: per_level[axis.name] for axis in theta.ord if axis.name in per_level}
    return PeelDepths(per_level=per_level, offsets=offsets, requested_steps=prefetch_steps,
                      chunk=chunk, reach=reach, deferrals=deferrals)


@dataclass(frozen=True)
class BoundaryHoist:
    """One instance of's cross-level boundary term."""
    op: str
    role: str
    level: str
    home: str
    delta: int
    coords: tuple


def _leading_inner_coords(home, delta):
    return tuple({home: i} for i in range(max(0, delta)))


def boundary_hoists(theta, pd):
    """Every cross-level boundary term `theta` carries (see `BoundaryHoist`), outer level first."""
    names = [axis.name for axis in theta.ord]
    outer = {axis.name for axis in theta.outer_axes()}
    out = []
    for (opname, role, level), delta in pd.offsets.items():
        if level not in outer:
            continue  # (1) inner level -> no boundary to cross
        operand = theta.op(opname)
        home = operand_level(theta, operand, role)
        if home is None or home not in names:
            continue
        if names.index(home) <= names.index(level):
            continue  # (2) home is at or outer to level
        if home not in outer:
            continue  # (3) unrolled home -> ordinary prefetch
        out.append(BoundaryHoist(op=opname, role=role, level=level, home=home, delta=delta,
                                 coords=_leading_inner_coords(home, delta)))
    return sorted(out, key=lambda hop: (names.index(hop.level), hop.op, hop.role))


def _all_varying_coords(theta, operand, varying):
    """Every coordinate the operand varies over -- the whole set, in loop order."""
    axes = [axis for axis in theta.inner_axes() if axis.name in varying]
    names = [axis.name for axis in axes]
    return [dict(zip(names, values))
            for values in product(*[range(max(1, axis.extent)) for axis in axes])]

def preloaded_tiles(theta, operand, prefetch_steps, depths=None, plans=None):
    """The leading physical tiles covered by the realized operand prefetch."""
    if not prefetch_steps:
        return []
    varying = set(geometry.presence(theta, operand))
    if not varying:
        return []
    if depths is not None and plans is None:
        plans = Schedule(theta, depths)
    steps = plans.want(operand) if plans is not None else requested_read_ahead(theta, operand)
    if steps < 1:
        return []
    coords = _all_varying_coords(theta, operand, varying)
    read = next((hop for hop in operand.hops
                 if not hop.is_bulk and hop.dst == Space.REGISTER), None)
    factors = geometry.coverage_factors(theta, operand, getattr(read, "quantum", None)) \
        if read is not None else {}
    leaders = [coord for coord in coords
               if all(int(coord.get(axis, 0)) % max(1, int(factor)) == 0
                      for axis, factor in factors.items())]
    count = steps * geometry.group_prefetch_unit_positions(theta, operand)
    return leaders[:min(len(leaders), count)]


def _prefetching_groups(operand):
    return tuple(group for group in operand.fragment.groups()
                 if operand.fragment.policy_of(group) != "inplace")


def prefetch_steps_for(theta, operand, group, requested, depths) -> int:
    """Realized lead for one group; zero when its capped policy must read in place."""
    if operand.fragment.policy_of(group) == "inplace":
        return 0
    if max(1, depths.get(operand.name, group)) < group_reuse_floor(theta, operand, group):
        return 0
    return min(max(0, int(requested)), requested_read_ahead(theta, operand, depths))


BAND_EMPTY = "band-empty"
CAPACITY = "capacity"
ASSIGNMENT = "assignment"
UNKNOWN = "unknown"


@dataclass(frozen=True)
class PrefetchRefusal:
    """Why one group could not read as far ahead as PrefetchLocalRead asked."""
    operand: str
    group: str
    requested: int
    derived: int
    reason: str
    values: dict

    def __getitem__(self, index):
        """Positional access, so the render tables can stay column-oriented."""
        return (self.operand, self.group, self.requested, self.derived, self.reason,
                self.values)[index]


def prefetch_refusal(theta, operand, group, prefetch_steps, depths):
    """The first depth the derivation would not grant, and what stopped it."""
    if not prefetch_steps:
        return None
    derived = prefetch_steps_for(theta, operand, group, prefetch_steps, depths)
    if derived >= prefetch_steps:
        return None
    buffers = max(1, depths.get(operand.name, group))
    refused = derived + 1

    def refusal(reason, **values):
        return PrefetchRefusal(operand.name, group, prefetch_steps, derived, reason, values)

    if refused > min(prefetch_steps, max(1, buffers - 1)):
        return refusal(BAND_EMPTY, depth=refused, buffers=buffers,
                       rate=group_ring_size(theta, operand, group),
                       fan=broadcast_width(theta, operand))
    peak = group_live_peak(theta, operand, group, _inner_steps(theta), off=refused)
    if peak > buffers:
        return refusal(CAPACITY, depth=refused, buffers=buffers, peak=peak)
    distance = prefetch_distance_for(theta, operand, refused)
    clobbered = [other for other in operand.fragment.groups()
                 if register_reuse_verdict(theta, operand, other,
                                           max(1, depths.get(operand.name, other)),
                                           distance) == CLOBBER]
    if clobbered:
        return refusal(ASSIGNMENT, depth=refused, buffers=buffers, distance=distance,
                       groups=clobbered, fan=broadcast_width(theta, operand))
    return refusal(UNKNOWN, depth=refused, buffers=buffers)


def prefetch_refusals(theta, depths, prefetch_steps=None):
    """Every (op, group) whose PLR request the derivation could not honour."""
    if prefetch_steps is None:
        try:
            prefetch_steps = peel_depths(theta, depths).requested_steps
        except Exception:
            return ()
    out = []
    for operand in theta.operands:
        if getattr(operand, "is_output", False):
            continue
        operand_steps = min(prefetch_steps, requested_read_ahead(theta, operand, depths))
        for group in operand.fragment.groups():
            refusal = prefetch_refusal(theta, operand, group, operand_steps, depths)
            if refusal is not None:
                out.append(refusal)
    return tuple(out)


def _readahead_depth(theta, operand, prefetch_steps, depths, groups=None, plans=None) -> int:
    """The shallowest depth across the selected groups: one instruction carries one advance."""
    selected = operand.fragment.groups() if groups is None else tuple(groups)
    if not selected:
        return 0
    plans = Schedule(theta, depths) if plans is None else plans
    return min(plans.steps(operand, group, prefetch_steps) for group in selected)


def _readahead_shift(theta, operand, prefetch_steps, depths, groups=None, plans=None):
    """The load-order offset the read-ahead becomes, with the strides it was measured in."""
    extents = transfer_extents(theta, operand)
    strides, reload_span = axis_strides(theta, reload_modes(theta, operand), extents)
    steps = _readahead_depth(theta, operand, prefetch_steps, depths, groups, plans)
    # `prefetch_distance_for` is the ONE answer -- re-testing the level against `strides` here is
    # how an operand invariant over it came back with no read-ahead at all.
    # An unrestricted read serves EVERY group, so it walks the whole fan rather than one residue.
    serves = tuple(groups) if groups else tuple(operand.fragment.groups())
    distance = prefetch_distance_for(theta, operand, steps, serves) if steps else 0
    return distance, strides, reload_span, extents


def loads_in_place(theta, operand, prefetch_steps, depths, groups=None, plans=None) -> bool:
    """Does the read-ahead land back on the value it started from, so the refill is in place?"""
    shift, _strides, reload_span, _extents = _readahead_shift(theta, operand, prefetch_steps,
                                                              depths, groups, plans)
    return bool(shift) and shift % max(1, reload_span) == 0


def _read_placement(theta, operand, depths, prefetch_steps=0, groups=None, plans=None):
    shift, strides, reload_span, extents = _readahead_shift(theta, operand, prefetch_steps,
                                                            depths, groups, plans)
    slots = []
    all_groups = operand.fragment.groups()
    selected = all_groups if groups is None else tuple(g for g in all_groups if g in groups)
    for group in selected:
        buffers = depths.get(operand.name, group)
        label = group if len(all_groups) > 1 else ""
        slot = ring_slot(theta, operand, group, buffers, advance=shift)
        slots.append((label, slot))
    src_slot = None
    if operand.hops and operand.hops[-1].src == Space.SHARED:
        lds_depth = depths.get(operand.name, SHARED_GROUP)
        outer_var = theta.summation_chunk_name()
        # The source generation is the chunk the read's position falls in, whatever the ring
        # rotates on: a shift past the reload span reads the NEXT chunk's LDS buffer.
        carried = position_terms(strides, read_coverage(theta, operand))
        src_slot = Expr(var=outer_var, mod=max(1, lds_depth),
                        carry=(carried, shift, reload_span))
    return Placement(space=Space.REGISTER, slots=tuple(slots), src_slot=src_slot)


def resolve_coverage(theta, depths=None) -> dict:
    return {operand.name: hop.quantum
            for operand in theta.operands for hop in operand.hops
            if not hop.is_bulk and hop.dst == Space.REGISTER and hop.quantum is not None}


def _generation_map(theta, operand, depths, prefetch_steps):
    placement = _read_placement(theta, operand, depths, prefetch_steps=prefetch_steps)
    varying = list(presence_axes(theta, operand))
    chunk = theta.summation_chunk_name()
    out = {}
    for combo in product(*[range(max(1, axis.extent)) for axis in varying]):
        env = {chunk: 0}
        env.update({axis.name: v for axis, v in zip(varying, combo)})
        source = placement.src_slot
        generation = source.eval(env) if hasattr(source, "eval") else source
        slots = tuple(slot.eval(env) if hasattr(slot, "eval") else slot
                      for _group, slot in placement.slots)
        out[tuple(zip((axis.name for axis in varying), combo))] = (generation, slots)
    return out, varying


def transfer_coverage_axes(theta, operand, hop, depths) -> tuple:
    """The axes one instruction of this hop covers, and by what factor (the movement coverage)."""
    cap = coverage_tile_cap(theta, operand, hop)
    if cap <= 1:
        return ()
    free = [axis for axis in free_axes(theta, operand) if axis.extent > 1]
    if not free:
        return ()
    steps = theta.off_of(operand, hop, prefetch_axis_name(theta, operand))
    generations, varying = _generation_map(theta, operand, depths, steps)
    best, covered = (), 1
    for depth in range(1, len(free) + 1):
        whole = free[-(depth - 1):] if depth > 1 else []  # outer axes, spanned whole
        inner = free[-depth]  # the one that may take a factor
        outer_width = 1
        for axis in whole:
            outer_width *= max(1, axis.extent)
        if outer_width > cap:
            break
        room = cap // outer_width
        factor = max((f for f in _divisors(inner.extent) if f <= room), default=1)
        if factor <= 1 and not whole:
            break  # nothing to merge at this depth
        candidate = tuple([(axis.name, axis.extent) for axis in whole]
                          + ([(inner.name, factor)] if factor > 1 else []))
        if is_uniform_over(generations, candidate) and outer_width * factor > covered:
            best, covered = candidate, outer_width * factor
        if factor < inner.extent:
            break
    return best


# --- schedule --------------------------------------------------------------

@dataclass(frozen=True)
class OperandPlan:
    """What we decided for one operand's register group."""

    #: how many copies of this operand's registers exist, so a load never overwrites a live one
    register_buffers: int
    #: which loop axis the lookahead is counted along, derived per operand from the loop order
    prefetch_axis: str
    #: how many steps along that axis a load runs ahead; 0 means load right before use
    prefetch_steps: int
    #: the same lookahead as an offset in this operand's own load order, which is what the
    #: address arithmetic and the drain bound consume
    prefetch_distance: int
    #: the loop position a reload must follow; empty means it may sit at first use
    reload_after: dict
    #: tiles loaded before the loop starts, so the first iteration has data
    preloaded_tiles: tuple
    #: how many LDS copies the global-to-LDS side keeps
    lds_buffers: int
    #: why `prefetch_steps` came out below what was requested, or None
    prefetch_refused: object

    @property
    def loads_ahead(self) -> bool:
        return self.prefetch_steps >= 1


class Schedule:
    """The decision table for one theta, computed on demand and cached."""

    def __init__(self, theta, depths, requested_depth=None):
        self.theta = theta
        self.depths = depths                     # the register depth map
        self.requested_depth = (_deepest_read_ahead(theta, depths) if requested_depth is None
                                else requested_depth)
        self._plans = {}
        self._steps = {}   # a plan needs EVERY group's steps, so this layer resolves first

    def want(self, operand) -> int:
        """The selected scheme's one realized internal PLR for this operand."""
        return requested_read_ahead(self.theta, operand, self.depths)

    def steps(self, operand, group, want) -> int:
        """How far ahead this group actually reads, at the depth `want` asks for.

        Kept below `plan`, not inside it: a plan's `preloaded_tiles` spans every group of the
        fragment, so deriving one plan asks for the others' steps. Two layers, no cycle.
        """
        key = (operand.name, group, want)
        if key not in self._steps:
            self._steps[key] = prefetch_steps_for(self.theta, operand, group, want, self.depths)
        return self._steps[key]

    def plan(self, operand, group, depth=None) -> OperandPlan:
        """The plan for one (operand, group), at `depth` or theta's requested depth."""
        want = self.want(operand) if depth is None else min(depth, self.want(operand))
        key = (operand.name, group, want)
        if key not in self._plans:
            self._plans[key] = self._derive(operand, group, want)
        return self._plans[key]

    def _derive(self, operand, group, want) -> OperandPlan:
        theta, depths = self.theta, self.depths
        width = max(1, depths.get(operand.name, group))
        steps = self.steps(operand, group, want)
        asked = min(want, self.want(operand))
        distance = prefetch_distance_for(theta, operand, steps)
        anchors = reload_positions(theta, operand, group, width, distance) or {}
        reload_after = {}
        for coord in anchors.values():
            reload_after.update(coord)
        level = readahead_level(theta, operand)
        return OperandPlan(
            register_buffers=width,
            prefetch_axis=level[0] if level else None,
            prefetch_steps=steps,
            prefetch_distance=distance,
            reload_after=reload_after,
            preloaded_tiles=tuple(map(_freeze,
                                      preloaded_tiles(theta, operand, want, depths, self))),
            lds_buffers=lds_buffers(theta, operand),
            prefetch_refused=prefetch_refusal(theta, operand, group, want, depths),
        )

    def operands(self):
        """(op, group) for every register-resident operand, in theta order."""
        for operand in self.theta.operands:
            if getattr(operand, "is_output", False) or not getattr(operand, "fragment", None):
                continue
            for group in operand.fragment.groups():
                yield operand, group


def _freeze(coord):
    return tuple(sorted(coord.items())) if isinstance(coord, dict) else coord


# ---  ------------------------------------------------------------------------------------------------

def group_width(theta, operand, group, steps) -> int:
    """Rotation-unit width selected for one group."""
    read = next((hop for hop in operand.hops
                 if not hop.is_bulk and hop.dst == Space.REGISTER), None)
    if read is not None and not presence_axes(theta, operand, hop=read):
        return 1
    if operand.name.startswith("MXS") and read is not None and read.quantum is not None:
        return 1
    R = group_ring_size(theta, operand, group)
    L_floor = group_live_peak(theta, operand, group, steps)  # post-WAR (band floor)
    serves = _prefetching_groups(operand)
    distance = prefetch_distance_for(
        theta, operand, requested_read_ahead(theta, operand),
        serves if group in serves else (group,))
    L_prefetch = group_live_peak(theta, operand, group, steps, off=distance)
    choice = operand.fragment.policy_of(group)
    if choice == "inplace":
        return L_floor
    if isinstance(choice, int):
        return max(L_floor, choice)
    if choice == "unroll":
        return max(L_floor, L_prefetch, R)
    if choice == "pipeline":
        need = max(L_floor, L_prefetch)
        if reloads_whole_set(theta, operand):
            return need
        return next((width for width in range(need, R + 1) if R % width == 0), R)
    return max(L_floor, L_prefetch)

def derive_S(theta) -> DepthMap:
    """The per-tile-group buffer-ring depth map S."""
    steps = _inner_steps(theta)
    depths = DepthMap()
    for operand in theta.operands:
        if operand.is_output:
            continue  # the accumulator holds one value per output tile across the
        for group in operand.fragment.groups():
            depths.set(operand.name, group, group_width(theta, operand, group, steps))
    return depths

def build_S(theta):
    floor = theta.S if theta.S is not None else derive_S(theta)
    depths = DepthMap(dict(floor.depths))
    for operand in theta.operands:
        if any(hop.dst == Space.SHARED for hop in operand.hops):
            depths.set(operand.name, "shared", lds_buffers(theta, operand))
    return depths, floor

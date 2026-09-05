# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Every derived decision: how many buffers, how far ahead, where each reload sits."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

from . import traversal as geometry
from .traversal import _inner_steps, group_live_peak, group_ring_size, lds_buffers, readahead_level
from .ir import COPY, Expr, Placement, READ, REVERSE, SHARED_GROUP, Space
from .theta import DepthMap, path_direction
from .traversal import (CLOBBER, _divisors, _summation_ring_axis, _shifted_ring_slot,
                        axis_is_live, axis_strides, broadcast_width, chunks_crossed,
                        group_value_span, transfer_extents, is_uniform_over, operand_level,
                        position_terms, prefetch_axis_mode, prefetch_axis_name,
                        prefetch_distance_for, coverage_tile_cap, read_coverage,
                        requested_read_ahead,
                        register_reuse_verdict, reload_anchor, reload_modes, reload_positions,
                        level_band_is_stepless, reloads_whole_set, ring_axes, ring_slot,
                        sibling_free_axes, varying_axes)
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
        prefetch_steps = requested_read_ahead(theta, operand)
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
    prefetch_steps = requested_read_ahead(theta, operand)
    crossed = max((chunks_crossed(theta, operand, plans.steps(operand, group, prefetch_steps))
                   for group in operand.fragment.groups()), default=0)
    return crossed >= 1 and theta.off_at(operand.name, COPY, outer) == crossed


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
        prefetch_steps = requested_read_ahead(theta, operand)
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


def _deepest_read_ahead(theta):
    """The largest read-ahead any shared->register read asks for, in its own axis's steps."""
    asked = [requested_read_ahead(theta, operand) for operand in theta.operands
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

    prefetch_steps = _deepest_read_ahead(theta)
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


@dataclass(frozen=True)
class TraversalSplit:
    """The inner traversal cut at the prefetch axis: what lies outside it, what lies inside."""
    prefetch_axis: object
    outer_axes: list           # varying axes outer to the prefetch axis
    inner_axes: list           # varying axes inside it
    inside_block: list         # every axis inside it, varying or not

    @property
    def steps_along_axis(self) -> int:
        return self.prefetch_axis.extent


def _split_at_prefetch_axis(theta, operand, varying):
    axis = prefetch_axis_mode(theta, operand)
    if axis is None:
        return None
    inner_axes = list(theta.inner_axes())
    at = inner_axes.index(axis)
    inside = inner_axes[at + 1:]
    return TraversalSplit(prefetch_axis=axis,
                          outer_axes=[m for m in inner_axes[:at] if m.name in varying],
                          inner_axes=[m for m in inside if m.name in varying],
                          inside_block=inside)


def _outer_axis_values(theta, operand, split):
    """Outer axes split two ways: pinned to 0, or fanned over because the read covers them."""
    read_hop = next((hop for hop in operand.hops
                     if not hop.is_bulk and hop.dst == Space.REGISTER
                     and getattr(hop, "quantum", None) is not None), None)
    coverage = geometry.transfer_coverage(theta, operand, read_hop) if read_hop is not None else {}
    siblings = set(sibling_free_axes(theta, operand))
    pinned = {axis.name: 0 for axis in split.outer_axes if axis.name not in siblings}
    fanned = [(axis.name, range(int(coverage[axis.name])))
              for axis in split.outer_axes if axis.name in coverage]
    fanned += [(axis.name, range(max(1, axis.extent))) for axis in split.outer_axes
               if axis.name in siblings and axis.name not in coverage]
    return pinned, fanned


def preloads_for_group(theta, operand, split, prefetch_steps, grouping_axis, grouping_values):
    """The coordinates one register group needs loaded before the loop starts.

    Every coordinate is one point of a cartesian product over four kinds of axis, listed here
    outermost-varying first: the grouping axis when it sits outside the prefetch axis, the
    prefetch axis itself, the axes inside it, and the outer axes the read fans over.  Outer axes
    it does not fan over are pinned to 0.
    """
    if prefetch_steps <= 0:
        return []
    steps = (split.steps_along_axis if not axis_is_live(split.inside_block)
             else min(prefetch_steps, split.steps_along_axis))
    pinned, fanned = _outer_axis_values(theta, operand, split)

    def values_of(axis):
        if grouping_axis is not None and axis.name == grouping_axis:
            return grouping_values
        return range(axis.extent)

    # A step is one ring SLOT: consecutive slots sit `extent // cycled` tiles apart.
    axis = split.prefetch_axis
    cycled = dict(ring_axes(theta, operand, operand.fragment.groups()[0])).get(axis.name, axis.extent)
    stride = max(1, int(axis.extent) // max(1, int(cycled)))
    grouping_is_outer = any(axis_.name == grouping_axis for axis_ in split.outer_axes)
    axes = ([(grouping_axis, grouping_values)] if grouping_is_outer else [])
    # The level is theta-global: an operand invariant over it holds no such coordinate, and naming
    # one keys the preload where no read can ever discharge it.
    if axis.name in varying_axes(theta, operand):
        axes.append((axis.name,
                     [v for v in range(0, steps * stride, stride) if v < axis.extent]))
    axes += [(axis.name, values_of(axis)) for axis in split.inner_axes]
    axes += fanned

    names = [name for name, _values in axes]
    # a name listed twice (grouping axis that the read also fans over) takes its last value,
    # which is the fan -- the same precedence the sequential build had
    return [dict(pinned, **dict(zip(names, point)))
            for point in product(*[values for _name, values in axes])]


def _any_group_anchors(theta, operand, steps_per_group, depths):
    """Does any register group have a legal reload position, i.e. does it really run ahead?"""
    if depths is None:
        return False
    for group, steps in steps_per_group.items():
        distance = prefetch_distance_for(theta, operand, steps)
        width = max(1, depths.get(operand.name, group))
        if reload_positions(theta, operand, group, width, distance):
            return True
    return False


def _all_varying_coords(theta, operand, varying):
    """Every coordinate the operand varies over -- the whole set, in loop order."""
    axes = [axis for axis in theta.inner_axes() if axis.name in varying]
    names = [axis.name for axis in axes]
    return [dict(zip(names, values))
            for values in product(*[range(max(1, axis.extent)) for axis in axes])]


def _preloads_over_groups(theta, operand, split, steps_per_group):
    """The union over register groups, in group order, without duplicates."""
    seen, coords = set(), []
    for group, steps in steps_per_group.items():
        grouping_axis, grouping_values = group_value_span(theta, operand, group)
        for coord in preloads_for_group(theta, operand, split, steps,
                                        grouping_axis, grouping_values):
            key = tuple(sorted(coord.items()))
            if key not in seen:
                seen.add(key)
                coords.append(coord)
    return coords


def preloaded_tiles(theta, operand, prefetch_steps, depths=None, plans=None):
    """The coords loaded before the loop, so the first trip has data.

    Three shapes, in order: the whole set at once when every register stays live across an
    outer pass, one group's worth when the fragment is not partitioned, and the union over
    groups when it is.
    """
    if not prefetch_steps:
        return []
    varying = set(varying_axes(theta, operand))
    if not varying:
        return []
    groups = operand.fragment.groups()
    if depths is not None and plans is None:
        plans = Schedule(theta, depths)
    steps_per_group = {group: (plans.steps(operand, group, prefetch_steps)
                               if depths is not None else prefetch_steps)
                       for group in groups}
    # An advance that wraps to zero leaves the steady reading the same values every trip, so that
    # GROUP has nothing to pre-fill.  `_readahead_shift` is the one derivation of it; asking it here
    # keeps the prologue and the steady from disagreeing about which groups prefetch.
    if depths is not None:
        steps_per_group = {
            group: (0 if not _readahead_shift(theta, operand, prefetch_steps, depths,
                                              (group,), plans)[0] else steps)
            for group, steps in steps_per_group.items()}
    if max(steps_per_group.values(), default=0) < 1:
        return []  # nothing runs ahead: every group refills in place

    if (reloads_whole_set(theta, operand)
            or _any_group_anchors(theta, operand, steps_per_group, depths)):
        return _all_varying_coords(theta, operand, varying)

    split = _split_at_prefetch_axis(theta, operand, varying)
    if split is None:
        return []
    if depths is None:  # no depth map, so use the request rather than a derived depth
        return preloads_for_group(theta, operand, split, prefetch_steps, None, None)
    if operand.fragment.grouping_mode is None or len(groups) <= 1:
        return preloads_for_group(theta, operand, split, steps_per_group[groups[0]], None, None)
    return _preloads_over_groups(theta, operand, split, steps_per_group)


def prefetch_steps_for(theta, operand, group, requested, depths) -> int:
    """The deepest read-ahead this group can actually carry, at or below `requested`."""
    if not requested or level_band_is_stepless(theta, operand):
        return 0
    buffers = max(1, depths.get(operand.name, group))
    steps = _inner_steps(theta)
    deepest = 0
    # No `buffers - 1` bound: what a depth costs is `group_live_peak(off=depth)`, which the body
    # already tests against `buffers`.  Capping the loop as well is the same rule derived twice,
    # and the cruder one refused depths the real test admits.
    for depth in range(1, requested + 1):
        distance = prefetch_distance_for(theta, operand, depth)
        if group_live_peak(theta, operand, group, steps, off=distance) > buffers:
            break
        # Only groups that OWE a read-ahead can veto one: an in-place group holds no ring to
        # anchor in, and letting it answer here refuses its siblings for its own policy.
        if any(reload_anchor(theta, operand, other,
                             max(1, depths.get(operand.name, other)), distance) is None
               for other in operand.fragment.groups()
               if operand.fragment.policy_of(other) != "inplace"):
            break
        deepest = depth
    return deepest


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
        for group in operand.fragment.groups():
            refusal = prefetch_refusal(theta, operand, group, prefetch_steps, depths)
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
        slot = ring_slot(theta, operand, group, buffers)
        # A SHIFTED READ OWES A SHIFTED SLOT, whatever the ring rotates on.  At
        # DepthU == MatrixInstK the summation substep has extent 1 and drops out, so the ring
        # turns on a free axis; requiring a summation axis here left the slot unshifted.
        if shift and buffers > 1:
            slot = _shifted_ring_slot(theta, operand, group, shift, strides, buffers,
                                      read_coverage(theta, operand)) or slot
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
        self.requested_depth = (_deepest_read_ahead(theta) if requested_depth is None
                                else requested_depth)
        self._plans = {}
        self._steps = {}   # a plan needs EVERY group's steps, so this layer resolves first

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
        want = self.requested_depth if depth is None else depth
        key = (operand.name, group, want)
        if key not in self._plans:
            self._plans[key] = self._derive(operand, group, want)
        return self._plans[key]

    def _derive(self, operand, group, want) -> OperandPlan:
        theta, depths = self.theta, self.depths
        width = max(1, depths.get(operand.name, group))
        steps = self.steps(operand, group, want)
        asked = min(want, requested_read_ahead(theta, operand))
        if steps < asked and operand.fragment.policy_of(group) != "inplace":
            raise RuntimeError(
                f"read-ahead: {operand.name}/{group} was asked for {asked} step(s) along "
                f"{prefetch_axis_name(theta, operand)} and only {steps} can be scheduled "
                f"({prefetch_refusal(theta, operand, group, asked, depths)}) -- choose a register "
                f"grouping whose ring can carry it, or ask for less")
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

def _assignment_width(theta, operand, group, lo, hi) -> int:
    if not any(hop.src == Space.SHARED and hop.dst == Space.REGISTER for hop in (operand.hops or ())):
        return lo
    prefetch_steps = requested_read_ahead(theta, operand)
    if not prefetch_steps:
        return lo
    want = max(lo, prefetch_steps + 1)
    for w in range(want, max(want, hi) + 1):
        if register_reuse_verdict(theta, operand, group, w, prefetch_distance_for(theta, operand, prefetch_steps)) != "clobber":
            return w
    return lo

def group_width(theta, operand, group, steps) -> int:
    """The rotation width this group stores, searched inside its derived band."""
    R = group_ring_size(theta, operand, group)
    L_floor = group_live_peak(theta, operand, group, steps)  # post-WAR (band floor)
    L_prefetch = group_live_peak(theta, operand, group, steps, post_war=False)  # prefetch-overlap peak
    choice = operand.fragment.policy_of(group)
    if choice == "inplace":  # W=1 where sequentially refillable, else L
        return max(1, L_floor)
    # R is where the search starts, never a ceiling: a read-ahead past the end of the trip needs
    # more buffers than one trip has.  `off` is a DISTANCE, not a step count -- the peak must be
    # measured with the displacement the read actually takes, or its width cannot carry it.
    served = group_live_peak(theta, operand, group, steps,
                             off=prefetch_distance_for(theta, operand,
                                                       requested_read_ahead(theta, operand)))
    if isinstance(choice, int):
        # An explicit W is a REQUEST like any other: honoured as given, or refused downstream when
        # it cannot carry the read-ahead.  Widening it here would be the clamp's mirror image.
        return max(L_floor, min(choice, R))
    if choice == "pipeline":
        need = max(L_prefetch, min(2, R))
        want = max(need, _assignment_width(theta, operand, group, need, R))
        return max(L_floor, served, min(next((w for w in range(want, R + 1) if R % w == 0), R), R))
    W = R if choice == "unroll" else L_prefetch  # 'unroll'->R ceiling; 'overlap'/'floor'->L_pf
    return max(L_floor, served, min(W, R))

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

# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Everything true of one operand's traversal: the tile and register arithmetic, which
axes it varies over, and how its register ring turns."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product as _iproduct
from math import ceil

from .ir import COPY, Expr, CoverageMap, READ, SHARED_GROUP, Space, COVERAGE_VAR, cst
from .theta import Axis


# --- tile and register arithmetic ------------------------------------------------------------

# ===========================================================================

def axis_tiles(macro_elems: int, atom_elems: int) -> int:
    return max(1, n_tiles(macro_elems, atom_elems))


# ===========================================================================

def _inner_axes(theta):
    return [(axis.name, axis.extent) for axis in theta.inner_axes()]


def broadcast_axes(theta, operand) -> set:
    """The intra-iteration loop axes this operand is constant over (does NOT depend on)."""
    inner = {name for name, _ext in _inner_axes(theta)}
    # An operand's OWN region axes are axes it varies over -- a scale carries its parent's split
    # axis and changes with it, so riding a fused walk does not make it constant over that axis.
    unsplit_regions = set()
    return ({name for name in operand.fragment.group_broadcast() if name in inner}
            | {name for name in theta.wave_served_axes() if name in inner}
            | unsplit_regions)


def _coverage_candidate_axes(theta, operand):
    bcast = broadcast_axes(theta, operand)
    summation = {axis.name for axis in summation_axes(theta)}
    return [(axis.name, max(1, axis.extent)) for axis in theta.inner_axes()
            if axis.name not in bcast and axis.name not in summation]


def derive_coverage(hop):
    """'s carrier group as `Phi` applied to `rho` -- the `ir.CoverageMap` derived, not supplied."""
    phi = max(1, int(getattr(hop, "phi_width", 1) or 1))
    rho = int(getattr(hop, "rho_span", 0) or 0)
    if phi <= 1 and not rho:
        return None
    tile = COVERAGE_VAR
    carrier = Expr(digits=(((tile, 1),), 0, ((1, phi, 0),)))  # tile // phi, unbounded
    if rho > 1:
        if phi % rho:
            return None
        slot = Expr(digits=(((tile, 1),), 0, ((1, rho, phi // rho),)))  # (t mod phi) // rho
    else:
        slot = Expr(digits=(((tile, 1),), 0, ((1, 1, phi),)))  # t mod phi
    return CoverageMap(carrier=carrier, slot=slot)


def coverage_factors(theta, operand, quantum) -> dict:
    if quantum is None:
        return {}
    axes = _coverage_candidate_axes(theta, operand)
    stride, total = {}, 1
    for name, extents in reversed(axes):
        stride[name] = total
        total *= extents
    out = {}
    for name, extents in axes:
        if extents <= 1:
            continue  # a degenerate axis is spanned by everything; say nothing
        step = stride[name]
        best = 1
        for factor in range(2, extents + 1):
            if extents % factor:
                continue  # only a uniform coverage -- regularity, no ragged group
            if all(quantum.carrier_of(tile - ((tile // step) % factor) * step) == quantum.carrier_of(tile)
                   for tile in range(total)):
                best = factor
        if best > 1:
            out[name] = best
    return out


def coverage_axes(theta, operand, quantum) -> set:
    """The intra-iteration axes ONE instruction of a per-lane hop spans whole."""
    extents = dict(_coverage_candidate_axes(theta, operand))
    return {name for name, factor in coverage_factors(theta, operand, quantum).items()
            if factor >= extents.get(name, 0)}


def transfer_broadcast(theta, operand, hop) -> set:
    inner = [name for name, _ext in _inner_axes(theta)]
    if not hop.is_bulk:
        return broadcast_axes(theta, operand) | coverage_axes(theta, operand, getattr(hop, "quantum", None))
    present = ({rm for rm in getattr(operand, "region_axes", ()) if rm in inner}
               if max(1, getattr(operand, "split", 1)) > 1 else set())
    return {name for name in inner if name not in present}


def transfer_coverage(theta, operand, hop) -> dict:
    if hop is None or getattr(hop, "is_bulk", False):
        return {}
    extents = dict(_coverage_candidate_axes(theta, operand))
    return {name: factor
            for name, factor in coverage_factors(theta, operand, getattr(hop, "quantum", None)).items()
            if 1 < factor < extents.get(name, 0)}


# ===========================================================================

def free_tiles(theta, operand) -> int:
    exts = [axis.extent for axis in free_axes(theta, operand)]
    return max(1, product(exts)) if exts else 1


def k_tiles(theta) -> int:
    exts = [axis.extent for axis in summation_axes(theta)]
    return max(1, product(exts)) if exts else 1


def summation_tiles(theta, operand) -> int:
    summation = {axis.name for axis in summation_axes(theta)}
    exts = [axis.extent for axis in presence_axes(theta, operand) if axis.name in summation]
    return max(1, product(exts)) if exts else 1


def presence_tiles(theta, operand) -> int:
    exts = [axis.extent for axis in presence_axes(theta, operand)]
    return max(1, product(exts)) if exts else 1


def wmma_grid(theta) -> tuple:
    inputs = [operand for operand in theta.operands
              if operand.is_input and any(hop.dst == Space.REGISTER for hop in operand.hops)]
    f0 = free_tiles(theta, inputs[0]) if len(inputs) > 0 else 1
    f1 = free_tiles(theta, inputs[1]) if len(inputs) > 1 else 1
    return (f0, f1, k_tiles(theta))


# ===========================================================================

def frag_regs(theta, operand) -> int:
    elems = max(1, operand.frag_elems)
    if theta.reg_bytes % max(1, operand.elem_bytes) == 0:
        scale = theta.reg_bytes // operand.elem_bytes  # elements per register
        return max(1, recast_count(elems, scale))  # ceil(elems/scale); was recast+coshape
    return max(1, -(-elems * operand.elem_bytes // theta.reg_bytes))  # ceil, sub-register elems


# ---  ------------------------------------------------------------------------------------------------

def _inner_steps(theta):
    axes = theta.inner_axes()
    ranges = [[(axis.name, value) for value in range(axis.extent)] for axis in axes]
    return [dict(c) for c in _iproduct(*ranges)] if ranges else [dict()]


def readahead_level_of(inner, region_names, broadcast_names=(), rotation_regions=()):
    """`(name, extent, span)` of the axis one PLR step advances: the OUTERMOST non-region axis.

 A degenerate axis is still the level -- one step of an extent-1 axis moves into the next value of
 whatever encloses it.  `span` counts every axis INSIDE the level, REGION AXES INCLUDED, because
 one step covers them; a split outside the level is not covered, so the read-ahead re-issues there.
    """
    bcast = set(broadcast_names or ())
    inner = [axis for axis in inner if axis.name not in bcast]
    regions = set(region_names or ())
    at = next((i for i, axis in enumerate(inner) if axis.name not in regions), None)
    if at is None:
        return None
    span = 1
    for axis in inner[at + 1:]:
        span *= max(1, int(axis.extent))
    return inner[at].name, max(1, int(inner[at].extent)), span


def readahead_level(theta, op=None):
    """The level is a property of `ord`, not of an operand: every read walks the same nest.

 `op` is accepted so call sites read the same either way, and ignored: an operand INVARIANT over
 the level is the whole-set case, which `reloads_whole_set` answers, not a different level."""
    regions = set()
    for operand in theta.operands:
        regions |= set(getattr(operand, "region_axes", ()) or ())
    return readahead_level_of(theta.inner_axes(), regions)


def group_owned_tiles(theta, operand) -> int:
    """The fan tiles one register group owns -- its share of the grouping mode's values."""
    grouping_mode = operand.fragment.grouping_mode
    if grouping_mode is None:
        return 1
    return max(1, theta.free_extent(grouping_mode) // max(1, len(operand.fragment.groups())))


def group_unit_tiles(theta, operand, group) -> int:
    """the unit: fan tiles ONE buffer of this group holds, derived from `loop order` AND the split."""
    if operand.fragment.grouping_mode is None:
        return 1
    return max(1, min(group_owned_tiles(theta, operand), resident_free_tiles(theta, operand)))


def group_fan_reloads(theta, operand, group) -> int:
    """Buffer-generations the fan contributes to `R` = owned tiles / tiles-per-buffer."""
    grouping_mode = operand.fragment.grouping_mode
    if grouping_mode is None:
        return 1
    hop = next((h for h in (operand.hops or ())
                if not getattr(h, "is_bulk", False)
                and h.dst == Space.REGISTER and h.src == Space.SHARED), None)
    owned = group_owned_tiles(theta, operand)
    if hop is not None:
        varying = {axis.name for axis in presence_axes(theta, operand, hop=hop)}
        if grouping_mode not in varying:
            return 1  # fully spanned: one instruction, one generation
        quantum = int((transfer_coverage(theta, operand, hop) or {}).get(grouping_mode, 1) or 1)
        owned = max(1, owned // max(1, quantum))
    unit = group_unit_tiles(theta, operand, group)
    return max(1, -(-owned // unit))


def _ring_broadcast(theta, operand, group) -> set:
    summation = set(summation_names(theta))
    broadcast = set(operand.fragment.group_broadcast(group)) \
        | {axis for axis in getattr(operand, "region_axes", ()) if axis not in summation}
    grouping_mode = operand.fragment.grouping_mode
    if grouping_mode is not None and group_fan_reloads(theta, operand, group) == 1:
        broadcast.add(grouping_mode)  # the grouping mode's values partition storage, not reload
    return broadcast


def _group_value_seq(theta, operand, group, steps):
    broadcast = _ring_broadcast(theta, operand, group)
    return [tuple((axis, value) for axis, value in sorted(step.items()) if axis not in broadcast) for step in steps]


def group_ring_size(theta, operand, group) -> int:
    broadcast = _ring_broadcast(theta, operand, group)
    _rd = next((hop for hop in (operand.hops or ())
                if not getattr(hop, "is_bulk", False) and hop.dst == Space.REGISTER
                and getattr(hop, "quantum", None) is not None), None)
    fold = transfer_coverage(theta, operand, _rd) if _rd is not None else {}
    grouping_mode = operand.fragment.grouping_mode
    r = 1
    for axis in theta.inner_axes():
        if axis.name in broadcast:
            continue
        extents = max(1, axis.extent // max(1, int(fold.get(axis.name, 1))))
        if axis.name == grouping_mode:
            extents = group_fan_reloads(theta, operand, group)
        r *= extents
    return r


def group_live_peak(theta, operand, group, steps, post_war=True, off=None) -> int:
    """The most values of this group live at once, which is the floor on its ring width."""
    nsteps = len(steps)
    if nsteps == 0:
        return 1
    if off is None:
        off = 0 if post_war else prefetch_distance_for(
            theta, operand, requested_read_ahead(theta, operand))
    broadcast = _ring_broadcast(theta, operand, group)
    varying = set(presence(theta, operand))
    spatial = tuple(sorted(axis for axis in broadcast if axis in varying))
    rate = _group_value_seq(theta, operand, group, steps)
    names = [tuple((axis, step[axis]) for axis in spatial) for step in steps]
    quantum = _quantum_over(transfer_coverage(theta, operand, _register_read_hop(operand)),
                            [a.name for a in theta.inner_axes()], broadcast)
    # `off` arrives as a displacement in the operand's LOAD ORDER; `_name_live_peak` indexes ONE
    # name's own steps.  Convert, or a distance that is one step of the level reads as many.
    _strides, span = axis_strides(theta, reload_modes(theta, operand),
                                  transfer_extents(theta, operand))
    peak = 0
    for nm in set(names):
        own = [i for i in range(nsteps) if names[i] == nm]  # this tile's own step list
        if own:
            seq = [rate[i] for i in own]
            # The divisor is how many DISTINCT values this name takes, not how many steps it has:
            # an operand invariant over an axis repeats each value, and counting the repeats makes
            # one step of the level read as a whole traversal.
            per_own = max(1, span // max(1, len(set(seq))))
            peak = max(peak, _name_live_peak(seq, off // per_own, quantum))
    return max(1, peak)


def _register_read_hop(operand):
    """The shared->register hop, or None -- the transfer whose coverage sets the fill granularity."""
    return next((h for h in (operand.hops or ())
                 if not getattr(h, "is_bulk", False)
                 and h.dst == Space.REGISTER and h.src == Space.SHARED), None)


def _quantum_over(cover, axes, broadcast) -> int:
    """Positions of the rotating axes one transfer spans."""
    return max([int(cover.get(a, 1) or 1) for a in axes if a not in broadcast] or [1])


def _name_live_peak(rate, off, quantum, niter=3) -> int:
    """The most generations of ONE tile live at once, given a read-ahead of `off` POSITIONS.

    `off` is the displacement the read actually takes -- `prefetch_distance_for` -- not a count of
    level steps, so the peak grows with the distance the way the ring's capacity check does."""
    span = len(rate)
    stride = next((j for j in range(1, span) if rate[j] == rate[0]), span)
    stride = -(-stride // max(1, quantum)) * max(1, quantum)   # no transfer is freed part-way
    gens = {}
    for s in range(niter):
        for j, value in enumerate(rate):
            at = s * span + j
            gens.setdefault((s, value), [at, at])[1] = at
    intervals = [(first - off, last) for first, last in gens.values()]
    return max((sum(1 for f, l in intervals if f <= p <= l) for p in range(span, 2 * span)),
               default=0)


def requested_read_ahead(theta, operand) -> int:
    """How far ahead this operand's read runs, in its own axis's steps.

 `PrefetchLocalRead` names the operand that steps INSIDE a chunk.  A FULL-INNER one -- whose whole
 set is a chunk, so one step of its level IS a chunk -- does not take that number: its depth is
 what the copy side stages, since the copy must stay a chunk ahead of the read it feeds."""
    if not operand.hops:
        return 0
    axis = prefetch_axis_name(theta, operand)
    if axis is None:
        return 0
    want = max(0, int(theta.off_at(operand.name, READ, axis) or 0))
    if not want or level_band_is_stepless(theta, operand):
        return 0
    outer = theta.summation_chunk_name()
    staged = int(theta.off_at(operand.name, COPY, outer) or 0) if outer else 0
    # Only the FULL-INNER operand takes its depth from the buffer.  Clamping every operand that
    # crosses a chunk was measured: it kills the read-ahead broadly, so a crossing that outruns the
    # staging is refused by `chunk_crossing_violations` instead, per PLR-is-faithful.
    if reloads_whole_set(theta, operand):
        return max(0, min(want, staged - 1))
    return want


_reg_read_off = requested_read_ahead      # the old name, kept for the sweep that still uses it


def _free_tile_index_seq(theta, operand, steps):
    free = set(free_names(theta, operand))
    free_axes = [axis for axis in theta.inner_axes() if axis.name in free]
    seq = []
    for step in steps:
        idx = 0
        for j, axis in enumerate(free_axes):
            inner = 1
            for m2 in free_axes[j + 1:]:
                inner *= m2.extent
            idx += step.get(axis.name, 0) * inner
        seq.append(idx)
    return seq


def resident_free_tiles(theta, operand) -> int:
    steps = _inner_steps(theta)
    T = len(steps)
    if T == 0:
        return 1
    off = _reg_read_off(theta, operand)
    seq = _free_tile_index_seq(theta, operand, steps)
    if not any(seq):  # operand has no own free mode -> 1 tile
        return 1
    gens = {}
    niter = 3
    for s in range(niter):
        for i, value in enumerate(seq):
            step = s * T + i
            gens.setdefault((s, value), [step, step])[1] = step
    intervals = [(first - off, second) for first, second in gens.values()]
    peak = 0
    for p in range(T, 2 * T):  # sample the steady iteration
        peak = max(peak, sum(1 for first, second in intervals if first <= p <= second))
    return max(1, peak)


def lds_buffers(theta, operand) -> int:
    """S_shared: the shared ring buffer depth (per reduction chunk)/."""
    if not any(hop.dst == Space.SHARED for hop in operand.hops):
        raise RuntimeError(
            f"{operand.name} has no shared placement (hops={[f'{hop.src}->{hop.dst}' for hop in operand.hops]}), "
            f"so S_shared is undefined for it -- the caller should have guarded on the operand "
            f"touching SHARED before asking for a shared-ring depth.")
    outer_var = theta.summation_chunk_name()
    d = theta.off_at(operand, COPY, outer_var) if outer_var is not None else 0
    if operand.lds_buffers:
        return max(1, int(operand.lds_buffers))
    return max(1, max(0, d))  # unstated -> delta (see the docstring's warning)


# ---  ------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class MmaShape:
    """The compute grid decoded for one kiter."""
    m_tiles: int; n_tiles: int; k_tiles: int
    @property
    def count(self) -> int:
        return self.m_tiles * self.n_tiles * self.k_tiles


# ===========================================================================

def operand_generations(operand, depths) -> int:
    """sum over register rate-groups of the group's depth (generation count) -- from derive_S."""
    groups = {group: d for group, d in depths.groups_of(operand.name).items() if group != SHARED_GROUP}
    return sum(groups.values()) if groups else 1


def group_regs(theta, operand, group) -> int:
    """Register width of ONE reuse group = its share of the fragment's registers."""
    nslots = operand.fragment.total_slots() or 1
    return max(1, frag_regs(theta, operand) * operand.fragment.size_of(group) // nslots)


def operand_footprint_regs(theta, operand, depths) -> int:
    """Physical VGPRs for this operand."""
    if operand.is_output:
        return max(1, free_tiles(theta, operand) * frag_regs(theta, operand))
    if operand.fragment.grouping_mode is not None:
        region = _region_count(theta, operand)
        gm_extent = theta.free_extent(operand.fragment.grouping_mode)  # in-region fan (grouping vals)
        groups = operand.fragment.groups()
        n_parts = len(groups)
        if gm_extent % n_parts != 0:
            raise RuntimeError(
                f"{operand.name}: grouping mode '{operand.fragment.grouping_mode}' extent {gm_extent} is "
                f"not divisible by {n_parts} parts -- uneven register partitions are OUT OF SCOPE "
                f"(accepted exception to; only equal partitions are supported.")
        part_vals = group_unit_tiles(theta, operand, groups[0])
        sum_w_times_part = sum(depths.get(operand.name, g) * part_vals for g in groups)
        return max(1, sum_w_times_part) * frag_regs(theta, operand) * region
    total = 0
    for group in operand.fragment.groups():
        depth = depths.get(operand.name, group)
        total += depth * group_regs(theta, operand, group)
    return max(1, total) * resident_free_tiles(theta, operand)


def operand_emitted_regs(theta, operand, depths) -> int:
    if operand.is_output:
        return operand_footprint_regs(theta, operand, depths)
    widths = [depths.get(operand.name, group) for group in operand.fragment.groups()] or [1]
    fan = group_unit_tiles(theta, operand, operand.fragment.groups()[0]) \
        * max(1, len(operand.fragment.groups())) if operand.fragment.grouping_mode else 1
    return max(1, max(widths)) * max(1, fan) * frag_regs(theta, operand) * _region_count(theta, operand)


def _region_count(theta, operand) -> int:
    return max(1, int(getattr(operand, "wave_region_span", 1)))


# ===========================================================================

def operand_buffer_regs(theta, operand) -> int:
    """Registers this operand loads per kiter = the axes it varies over tiles x frag_regs."""
    return presence_tiles(theta, operand) * frag_regs(theta, operand)


def load_regs(theta, operand, hop) -> int:
    """Registers moved by ONE per-lane instruction = vector width packed into regs."""
    vw = hop.vector_elems or operand.frag_elems
    return max(1, ceil(vw * operand.elem_bytes / theta.reg_bytes))


def operand_tile_bytes(theta, operand) -> int:
    return presence_tiles(theta, operand) * operand.frag_elems * operand.elem_bytes * theta.lanes


def plan_transfers(theta, operand, hop) -> int:
    """Instructions this hop issues per kiter."""
    if hop.is_bulk:
        return max(1, hop.split)
    return max(1, ceil(operand_buffer_regs(theta, operand) / load_regs(theta, operand, hop)))


# ===========================================================================

def mma_shape(theta) -> MmaShape:
    """m_tiles x n_tiles x k_tiles for one kiter, derived from wmma_grid."""
    axis, n, k = wmma_grid(theta)
    return MmaShape(axis, n, k)


# --- integer layout arithmetic ------------------------------------------------------------

def tile_split(n: int, t: int) -> tuple:
    n = max(1, n)
    t = max(1, t)
    if n % t == 0:
        return (t, n // t)
    return (min(t, n), -(-n // t))


def n_tiles(n: int, t: int) -> int:
    """The tile count along an axis = `tile_split(n,t)[1]` = n//t (divisible)."""
    return max(1, tile_split(n, t)[1])


def per_tile(n: int, t: int) -> int:
    """The per-tile extent = `tile_split(n,t)[0]` = t (divisible)."""
    return max(1, tile_split(n, t)[0])


def product(extents) -> int:
    """Size of a `Layout(tuple(extents))` = product of the extents."""
    p = 1
    for e in extents:
        p *= max(1, int(e))
    return max(1, p)


def recast_count(elems: int, scale: int) -> int:
    elems = max(1, elems)
    scale = max(1, scale)
    return max(1, -(-elems // scale))


# --- what an operand varies over -------------------------------------------------------------

def summation_axes(theta):
    """The axes the output is not present on: the summation (K) axes."""
    on_output = set()
    for operand in theta.output_operands():
        on_output.update(axis.name for axis in presence_axes(theta, operand))
    served_by_waves = theta.wave_served_axes()
    return [axis for axis in theta.inner_axes()
            if axis.name not in on_output and axis.name not in served_by_waves]


def summation_names(theta):
    return {axis.name for axis in summation_axes(theta)}


def free_axes(theta, operand):
    """The operand's own free (M or N) axes: the ones it varies over that are not summation."""
    summation = summation_names(theta)
    return [axis for axis in presence_axes(theta, operand) if axis.name not in summation]


def free_names(theta, operand):
    return [axis.name for axis in free_axes(theta, operand)]


def presence(theta, operand):
    return [axis.name for axis in presence_axes(theta, operand)]


def presence_axes(theta, operand, hop=None):
    """The axes this operand's data changes over on one transfer, or over all of them."""
    constant_over = (broadcast_axes(theta, operand) if hop is None
                     else transfer_broadcast(theta, operand, hop))
    coverage = {} if hop is None else transfer_coverage(theta, operand, hop)
    return [axis if axis.name not in coverage
            else Axis(axis.name, max(1, axis.extent // coverage[axis.name]))
            for axis in theta.inner_axes() if axis.name not in constant_over]


# --- the axes it varies over --------------------------------------------------------------

def prefetch_axis_name(theta, operand=None):
    axis = readahead_level(theta, operand)
    return axis[0] if axis else None


def operand_level(theta, operand, role):
    chunk = theta.summation_chunk_name()
    if role == COPY:
        inner = {axis.name for axis in theta.inner_axes()}
        region = next((r for r in getattr(operand, "region_axes", ()) if r in inner), None)
        return region if (region is not None and max(1, operand.split) > 1) else chunk
    varying = varying_axes(theta, operand)
    return varying[-1] if varying else chunk


def group_value_span(theta, operand, group):
    fragment = operand.fragment
    grouping_mode = fragment.grouping_mode
    groups = fragment.groups()
    if grouping_mode is None or len(groups) <= 1:
        return None, None
    # INTERLEAVED, not contiguous halves: a group owns one residue class, so its own next block is
    # `parts` away and a read-ahead by a multiple of that never leaves it.  The unit is the READ's
    # block, not one value -- a folded read covers `q` consecutive values and may not straddle.
    index = list(groups).index(group)
    quantum = group_block_quantum(theta, operand)
    return grouping_mode, [value for value in range(theta.free_extent(grouping_mode))
                           if (value // quantum) % len(groups) == index]


def group_block_quantum(theta, operand) -> int:
    """Level values one read covers, so the partition interleaves BLOCKS a read cannot straddle."""
    grouping_mode = getattr(getattr(operand, "fragment", None), "grouping_mode", None)
    if grouping_mode is None:
        return 1
    return max(1, int((read_coverage(theta, operand) or {}).get(grouping_mode, 1) or 1))


def varying_axes(theta, operand):
    hop = _local_read(operand)
    constant_over = (broadcast_axes(theta, operand) if hop is None
                     else transfer_broadcast(theta, operand, hop))
    return [axis.name for axis in theta.inner_axes() if axis.name not in constant_over]


def reload_modes(theta, operand):
    """The modes the prefetch advances through: the axes it varies over, minus the sibling
 enumerators that lie OUTSIDE the level.  A storage region INSIDE the level is walked, because one
 step of the level covers it; one outside it is a separate issue of the whole read-ahead.  The
 register-group enumerator is a walk axis only when it IS the level -- otherwise its groups are
 separate rings."""
    siblings = set(sibling_free_axes(theta, operand))
    grouping = getattr(getattr(operand, "fragment", None), "grouping_mode", None)
    keep = _axes_inside_level(theta) & siblings - ({grouping} if grouping else set())
    if group_step_on_level(theta, operand) > 1:
        keep = keep | {grouping}
    return [name for name in varying_axes(theta, operand)
            if name not in siblings or name in keep]


def group_step_on_level(theta, operand, groups=None) -> int:
    """Level values one register group skips per step, `1` unless the partition CUTS the level.

    The partition is interleaved, so a group owns a residue class: its own next value is `parts`
    away, and a read-ahead by that much stays inside the group and wraps into its next generation.
    A read serving EVERY group walks the whole fan, so it steps by one like an unpartitioned one.
    """
    fragment = getattr(operand, "fragment", None)
    grouping = getattr(fragment, "grouping_mode", None)
    parts = len(getattr(fragment, "labels", ()) or ()) or 1
    level = readahead_level(theta)
    if parts < 2 or not grouping or level is None or grouping != level[0]:
        return 1
    if groups is not None and len(set(groups)) >= parts:
        return 1
    return parts if grouping in varying_axes(theta, operand) else 1


def level_band_is_stepless(theta, operand) -> bool:
    """Does the partition leave each group a SINGLE level value, so it has no step to run ahead?"""
    step = group_step_on_level(theta, operand)
    level = readahead_level(theta)
    return step > 1 and level is not None and max(1, int(level[1]) // step) < 2


def _axes_inside_level(theta) -> set:
    """The axis names one step of the read-ahead level covers."""
    level = readahead_level(theta)
    names = [axis.name for axis in theta.inner_axes()]
    if level is None or level[0] not in names:
        return set()
    return set(names[names.index(level[0]) + 1:])


def sibling_free_axes(theta, operand) -> tuple:
    summation = set(summation_names(theta))
    named = tuple(r for r in getattr(operand, "region_axes", ()) if r not in summation)
    fragment = getattr(operand, "fragment", None)
    grouping_mode = getattr(fragment, "grouping_mode", None)
    if grouping_mode and grouping_mode not in summation \
            and len(getattr(fragment, "parts", (1,))) > 1:
        named = named + ((grouping_mode,) if grouping_mode not in named else ())
    if named or max(1, int(getattr(operand, "free_split", 1))) <= 1:
        return named
    own_axes = {operand.free_mode, grouping_mode}
    return tuple(axis.name for axis in theta.inner_axes()
                 if axis.name in own_axes and axis.name not in summation)[:1]


def _sibling_outer_to_reduction(theta, operand) -> bool:
    names = [axis.name for axis in theta.inner_axes()]
    summation = _summation_ring_axis(theta, operand, operand.fragment.groups()[0])
    if not summation or summation[0] not in names:
        return False
    at = names.index(summation[0])
    return any(name in names and names.index(name) < at
               for name in sibling_free_axes(theta, operand))


def axis_is_live(axes) -> bool:
    """Does this loop order block contain any axis with more than one value?"""
    return any(axis.extent > 1 for axis in axes)


def prefetch_axis_mode(theta, operand=None):
    """The reduction substep axis: the innermost reduction inner mode with more than ONE value."""
    inner = list(theta.inner_axes())
    axis = readahead_level(theta, operand)
    if axis is None:
        return None
    return next((mode for mode in inner if mode.name == axis[0]), None)


def ring_axes(theta, operand, group):
    """The inner modes one register group cycles distinct values over, in loop order order."""
    constant_over = _ring_broadcast(theta, operand, group)
    extents = transfer_extents(theta, operand)
    return [(axis.name, max(1, extents.get(axis.name, axis.extent)))
            for axis in theta.inner_axes()
            if axis.name not in constant_over and axis.extent > 1]


def ring_slot(theta, operand, group, buffers):
    """Which buffer of the ring a coordinate lands in, as an expression in the loop indices."""
    if buffers <= 1:
        return cst(0)
    ring = ring_axes(theta, operand, group)
    if not ring:
        return cst(0)
    strides, place = {}, 1
    for name, extent in reversed(ring):
        strides[name] = place
        place *= max(1, extent)
    in_ord_order = {name: strides[name] for name, _extent in ring}
    return Expr(mod=max(1, buffers),
                terms=position_terms(in_ord_order, read_coverage(theta, operand)))


def _summation_ring_axis(theta, operand, group):
    summation = summation_names(theta)
    on_summation = [(name, extent) for name, extent in ring_axes(theta, operand, group)
                    if name in summation]
    return on_summation[-1] if on_summation else None


def axis_strides(theta, modes=None, extents=None):
    axes = list(theta.inner_axes())
    if modes is not None:
        axes = [axis for axis in axes if axis.name in modes]
    strides, place = {}, 1
    for axis in reversed(axes):  # innermost axis has stride 1
        strides[axis.name] = place
        place *= (extents or {}).get(axis.name, axis.extent)
    return strides, place       # place is now the product over `modes`


def transfer_extents(theta, operand) -> dict:
    """`{mode: extent}` as the operand's read hop sees them -- its the axes it varies over extents."""
    hop = _local_read(operand)
    extents = {axis.name: axis.extent for axis in theta.inner_axes()}
    if hop is not None:
        extents.update({axis.name: axis.extent for axis in presence_axes(theta, operand, hop)})
    return extents


def _local_read(operand):
    return next((hop for hop in operand.hops if not hop.is_bulk and hop.dst == Space.REGISTER), None)


def read_coverage(theta, operand) -> dict:
    """`{mode: q}` -- the partial coverage factor of the operand's read hop or `{}`."""
    hop = _local_read(operand)
    return {} if hop is None else transfer_coverage(theta, operand, hop)


def broadcast_width(theta, operand) -> int:
    varying = set(varying_axes(theta, operand))
    if not varying:
        return 1
    inner_names = [axis.name for axis in theta.inner_axes()]
    extents = {axis.name: axis.extent for axis in theta.inner_axes()}
    outermost_varying = min(inner_names.index(name) for name in varying)
    width = 1
    for name in inner_names[:outermost_varying]:
        if name not in varying and extents.get(name, 1) > 1:
            width *= extents[name]
    return width


def position_terms(strides, fold=None):
    return tuple((name, coef, max(1, int((fold or {}).get(name, 1))))
                 for name, coef in strides.items())


def _shifted_coord(axis, shift, strides, _reload_span, extents, fold=None):
    if axis not in strides:
        return Expr(var=axis)
    terms = position_terms(strides, fold)
    quantum = max(1, int((fold or {}).get(axis, 1)))
    stride = max(1, strides[axis])
    if quantum > 1:
        return Expr(digits=(terms, shift, ((quantum, stride, max(1, extents.get(axis, 1))),)))
    return Expr(carry=(terms, shift, stride), mod=max(1, extents.get(axis, 1)))


def _shifted_ring_slot(theta, operand, group, shift, strides, buffers, fold=None):
    """The rotation slot the prefetched read writes, as an Expr."""
    ring = [(name, extent) for name, extent in ring_axes(theta, operand, group)
            if name in strides]
    if not ring:
        return None
    digits, place = [], 1
    terms = position_terms(strides, fold)
    for name, extent in reversed(ring):  # the innermost axis is the fastest radix
        digits.append((place, max(1, strides[name]), extent))
        place *= max(1, extent)
    return Expr(digits=(terms, shift, tuple(reversed(digits))), mod=max(1, buffers))


def coverage_tile_cap(theta, operand, hop) -> int:
    if hop.is_bulk or not hop.vector_elems:
        return 1
    per_fragment = max(1, int(operand.frag_elems))
    return max(1, int(hop.vector_elems) // per_fragment)


def _divisors(value: int):
    value = max(1, int(value))
    return [d for d in range(value, 0, -1) if value % d == 0]


def is_uniform_over(generations, axes) -> bool:
    """Is the generation map constant within each block of `axes`?"""
    block = dict(axes)
    seen = {}
    for coord, generation in generations.items():
        key = tuple((name, value // block[name] if name in block else value)
                    for name, value in coord)
        if seen.setdefault(key, generation) != generation:
            return False
    return True


# --- registers -------------------------------------------------------------

SAFE = "safe"
IN_PLACE = "inplace"
CLOBBER = "clobber"

TRIPS_WALKED = 3


def chunks_crossed(theta, operand, prefetch_steps):
    """How many reduction chunks this read-ahead reaches into -- the DISTANCE over the chunk.

 Asked of `prefetch_distance_for`, never re-derived: a second distance here is how the whole-set
 arm went unseen, so the staging bound passed a schedule needing a chunk the copy had not loaded
 and it surfaced as a clobber in the ring walk instead of a refusal at the adapter."""
    if prefetch_steps <= 0:
        return 0
    distance = prefetch_distance_for(theta, operand, prefetch_steps)
    if distance <= 0:
        return 0
    _strides, span = axis_strides(theta, reload_modes(theta, operand),
                                  transfer_extents(theta, operand))
    span = max(1, span)
    return (span - 1 + distance) // span


def prefetch_distance_for(theta, operand, prefetch_steps, groups=None) -> int:
    if not prefetch_steps:
        return 0
    reload_axes = reload_modes(theta, operand)
    if not reload_axes:
        return 0
    strides, reload_span = axis_strides(theta, reload_axes, transfer_extents(theta, operand))
    # THE WHOLE-SET ARM COMES FIRST: an operand broadcast over the level does not step it at all,
    # so asking for its stride would report "no read-ahead" for exactly the operands that need one.
    if reloads_whole_set(theta, operand):
        return prefetch_steps * reload_span
    axis = readahead_level(theta, operand)
    if axis is None:
        return 0
    # ONE STEP IS ONE STEP, whatever is inside the level: the axes inside it are what the step
    # covers, so a nest that degenerated to nothing still advances by one, not by the whole axis.
    if axis[0] in strides:
        # CAPPED AT THE LEVEL'S OWN VALUES: a read cannot run further ahead than the level holds
        # without leaving the chunk, and the chunk-crossing rule -- not the distance -- governs that.
        step = group_step_on_level(theta, operand, groups)
        return min(prefetch_steps, max(1, axis[1] // step)) * step * strides[axis[0]]
    # This operand does not step the level, so the level is not a displacement in ITS order: one
    # step of it covers every read of this operand that lies inside it.
    extents = transfer_extents(theta, operand)
    inside = _axes_inside_level(theta)
    covered = 1
    for name in reload_axes:
        if name in inside:
            covered *= max(1, extents.get(name, 1))
    return prefetch_steps * covered


def reloads_whole_set(theta, operand) -> bool:
    varying = varying_axes(theta, operand)
    inner_names = [axis.name for axis in theta.inner_axes()]
    extents = {axis.name: axis.extent for axis in theta.inner_axes()}
    reload_axes = [name for name in (reload_modes(theta, operand) or varying)
                   if extents.get(name, 1) > 1]
    if not reload_axes:
        return False
    outermost_reload = min(inner_names.index(name) for name in reload_axes)
    return any(name not in varying and extents.get(name, 1) > 1
               for name in inner_names[:outermost_reload])


@dataclass(frozen=True, order=True)
class Generation:
    """Which load put a value in a register, and the coordinates that load did not move."""
    trip: int
    position: int
    pinned: tuple


@dataclass(frozen=True, order=True)
class Event:
    """A register is loaded, or read, at one point in the walk."""
    time: int
    is_load: bool
    generation: Generation


@dataclass(frozen=True)
class RegisterTimeline:
    """Per register, every load and every read, over three trips of the loop body."""
    events: dict
    steps: tuple
    steps_per_trip: int

    def is_steady(self, time) -> bool:
        return self.steps_per_trip <= time < 2 * self.steps_per_trip


@dataclass(frozen=True)
class _RingWalk:
    """The fixed geometry one register-timeline walk needs, so the walk itself is just a loop."""
    varying: tuple
    varying_strides: dict
    reload_strides: dict
    reload_span: int
    extents: dict
    fan_axes: tuple      # the axes that name a distinct register, not a slot in the ring
    pinned_axes: tuple   # the axes a reload does not move
    ring: tuple
    buffers: int

    def register_of(self, coord):
        """Which register this coordinate lives in: its fan position plus its ring slot."""
        slot, place = 0, 1
        for name, extent in reversed(self.ring):
            slot += coord.get(name, 0) * place
            place *= extent
        return (tuple(coord.get(name, 0) for name in self.fan_axes), slot % max(1, self.buffers))

    def position(self, coord):
        return sum(self.varying_strides[name] * coord[name] for name in self.varying)

    def reload_position(self, coord):
        return sum(self.reload_strides[name] * coord[name] for name in self.reload_strides)

    def advanced(self, coord, position):
        """`coord` moved to the reload position `position`, leaving the pinned axes alone."""
        out = dict(coord)
        out.update({name: (position // self.reload_strides[name])
                    % max(1, self.extents.get(name, 1)) for name in self.reload_strides})
        return out


def _ring_walk(theta, operand, group, buffers, varying):
    reload_axes = reload_modes(theta, operand) or varying
    varying_strides, _ = axis_strides(theta, varying)
    reload_strides, reload_span = axis_strides(theta, reload_axes)
    summation = set(summation_names(theta))
    return _RingWalk(
        varying=tuple(varying), varying_strides=varying_strides,
        reload_strides=reload_strides, reload_span=reload_span,
        extents={axis.name: axis.extent for axis in theta.inner_axes()},
        fan_axes=tuple(name for name in varying if name not in summation),
        pinned_axes=tuple(name for name in varying if name not in reload_strides),
        ring=tuple(ring_axes(theta, operand, group)), buffers=buffers)


def _register_timeline(theta, operand, group, buffers, distance, at=None):
    """Walk TRIPS_WALKED trips, recording when each register is read and when it is loaded."""
    varying = varying_axes(theta, operand)
    if not varying or not distance:
        return None
    walk = _ring_walk(theta, operand, group, buffers, varying)
    steps = _inner_steps(theta)
    per_trip = len(steps)
    events = {}
    for trip in range(TRIPS_WALKED):
        loaded_positions = set()
        for index, step in enumerate(steps):
            time = trip * per_trip + index
            coord = {name: step.get(name, 0) for name in walk.varying}
            pinned = tuple(coord[name] for name in walk.pinned_axes)
            reload_position = walk.reload_position(coord)
            events.setdefault(walk.register_of(coord), []).append(
                Event(time, False, Generation(trip, reload_position, pinned)))

            if at and any(step.get(name, 0) != value for name, value in at.items()):
                continue
            position = walk.position(coord)
            if position in loaded_positions:
                continue  # first touch of this position issues the one read
            loaded_positions.add(position)

            reached = reload_position + distance
            events.setdefault(walk.register_of(walk.advanced(coord, reached)), []).append(
                Event(time, True, Generation(trip + reached // max(1, walk.reload_span),
                                             reached % max(1, walk.reload_span), pinned)))
    return RegisterTimeline(events, tuple(steps), per_trip)


def _uses_before_load(timeline):
    """Steady-trip reads that see a generation no load put there."""
    stale = {}
    for register, events in timeline.events.items():
        loaded = None
        for event in sorted(events):
            if event.is_load:
                loaded = event.generation
            elif loaded is not None and timeline.is_steady(event.time) \
                    and loaded != event.generation:
                stale.setdefault(register, []).append((event.time, event.generation, loaded))
    return stale


def _overwrites_live_register(timeline):
    """Steady-trip loads that land on a generation still read afterwards."""
    conflicts = {}
    for register, events in timeline.events.items():
        loads = sorted(event for event in events if event.is_load)
        reads = [event for event in events if not event.is_load]
        for order, load in enumerate(loads):
            if not order or not timeline.is_steady(load.time):
                continue
            overwritten = loads[order - 1].generation
            if overwritten == load.generation:
                continue
            still_read = sorted(read.time for read in reads
                                if read.generation == overwritten and read.time >= load.time)
            if still_read:
                conflicts.setdefault(register, []).append((load.time, still_read))
    return conflicts


def register_reuse_verdict(theta, operand, group, buffers, distance, at=None) -> str:
    """Can this many buffers carry a read-ahead of `distance`: SAFE, IN_PLACE, or CLOBBER?"""
    timeline = _register_timeline(theta, operand, group, buffers, distance, at=at)
    if timeline is None:
        return SAFE
    if _uses_before_load(timeline):
        return CLOBBER
    verdict = SAFE
    for _register, conflicts in _overwrites_live_register(timeline).items():
        for load_time, still_read in conflicts:
            if any(read_time > load_time for read_time in still_read):
                return CLOBBER
            verdict = IN_PLACE
    return verdict


def reload_anchor(theta, operand, group, buffers, distance):
    """Where a prefetching reload sits -- the ONE answer, searched or derived.

 `reload_positions` looks for the earliest slot the ring admits: one that follows the last reader
 of the value being overwritten, which only exists when there are buffers to spare.  When there are
 not, the END of every invariant pass is legal unconditionally, because every consumer in that pass
 has already run.  None only when even the latest position clobbers.
    """
    found = reload_positions(theta, operand, group, buffers, distance)
    if found is not None:
        return found
    extents = {axis.name: axis.extent for axis in theta.inner_axes()}
    invariant = {axis.name for axis in theta.inner_axes()} - set(varying_axes(theta, operand))
    at = {name: extents[name] - 1 for name in invariant if max(1, extents.get(name, 1)) > 1}
    if not at or register_reuse_verdict(theta, operand, group, buffers, distance,
                                        at=at) == CLOBBER:
        return None
    return {None: at}


def reload_positions(theta, operand, group, buffers, distance):
    """Where each reload must sit so it follows the last read of the value it overwrites.

    None means no such position exists, so this width cannot carry this read-ahead.
    """
    timeline = _register_timeline(theta, operand, group, buffers, distance)
    if timeline is None:
        return {}
    steps, steps_per_trip = timeline.steps, timeline.steps_per_trip

    after = {}
    for register, conflicts in _overwrites_live_register(timeline).items():
        for _load_time, still_read in conflicts:
            last_read = still_read[-1]
            if last_read >= 2 * steps_per_trip:
                return None
            after[register] = dict(steps[last_read % steps_per_trip])

    invariant_axes = ({axis.name for axis in theta.inner_axes()}
                      - set(varying_axes(theta, operand))) | set(read_coverage(theta, operand) or {})
    at = {}
    for name in invariant_axes:
        values = {coord[name] for coord in after.values() if name in coord}
        if len(values) > 1:
            return None
        if values:
            at[name] = next(iter(values))

    if at and register_reuse_verdict(theta, operand, group, buffers, distance, at=at) == CLOBBER:
        return None
    return after

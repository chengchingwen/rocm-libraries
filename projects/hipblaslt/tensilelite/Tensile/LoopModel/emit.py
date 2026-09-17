# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""LoopIR instructions: the copies, the reads, the wmmas, and the loop nest.

The emitted shape is always the same four phases:

    if T > M:  Peel(prologue)  Loop(iter < T-M)  Peel(drain)
    else:      M straight steps with no read-ahead

`M` is the software-pipeline depth in K-chunks and `T` the K-chunk trip count, unknown until
run time.  Each phase is the same two calls -- `copy_insts` for global->LDS and `nest` for
LDS->register plus the wmma -- differing only in which operands are in flight and whether the
buffer index is a loop expression or a constant.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from . import traversal as geometry
from .checks import (build_ledger, check_ledger_discharged, discharge_once,
                     movement_name as _movement_name, validate_loopir as _validate)
from .ir import (Await, Bind, Cond, Expr, Inst, Load, Loop, Mma, Peel, Placement, Pred, READ,
                 Space, child_bodies, cst, is_war, move_reloads_after_last_use)
from .schedule import (Schedule, _read_placement, _readahead_shift,
                       boundary_hoists as _boundary_hoists, build_S, copy_first_operands,
                       peel_depths, preloaded_tiles, resolve_coverage)
from .traversal import (group_block_quantum, requested_read_ahead,
                        _shifted_coord, lds_buffers, operand_level, position_terms, presence,
                        presence_axes, read_coverage, readahead_level, reload_anchor, reload_positions,
                        summation_names, varying_axes)

STEADY, PROLOGUE, DRAIN, SHORT = "steady", "prologue", "drain", "short"


# --- first-touch guard surgery (pure LoopIR, no theta) ---------------------

def _is_first_touch_on(node, axis):
    return (isinstance(node, Cond) and node.kind == "first_touch"
            and getattr(node.pred.lhs, "var", None) == axis and not node.pred.lhs.mod
            and node.pred.op == "==" and node.pred.rhs == 0)


def _resolve_first_touch(nodes, axis, taken):
    """Rewrite a body with every `axis == 0` guard decided one way."""
    out = []
    for node in nodes:
        if _is_first_touch_on(node, axis):
            if taken:
                out.extend(_resolve_first_touch(node.then, axis, taken))
            continue  # not taken -> the read is not issued
        if isinstance(node, Loop):
            out.append(Loop(axis=node.axis, trip=node.trip, outer=node.outer,
                            bodies=[_resolve_first_touch(body, axis, taken) for body in node.bodies],
                            body_ranges=node.body_ranges))
        elif isinstance(node, Cond):
            out.append(Cond(pred=node.pred, kind=node.kind, label=node.label,
                            then=_resolve_first_touch(node.then, axis, taken),
                            els=_resolve_first_touch(node.els, axis, taken)))
        elif isinstance(node, Peel):
            out.append(Peel(kind=node.kind, axis=node.axis, k=node.k, outer=node.outer,
                            body=_resolve_first_touch(node.body, axis, taken)))
        elif isinstance(node, Bind):
            out.append(Bind(axis=node.axis, value=node.value,
                            body=_resolve_first_touch(node.body, axis, taken)))
        else:
            out.append(node)
    return out


def _has_first_touch(nodes, axis):
    return any(_is_first_touch_on(node, axis)
               or any(_has_first_touch(body, axis) for body in child_bodies(node))
               for node in nodes)


def _split_first_iteration(loop):
    """Give a loop two bodies -- iteration 0 and the rest -- so its `axis == 0` guards vanish."""
    if not isinstance(loop.trip, int) or loop.trip < 2 or len(loop.bodies) != 1:
        return loop
    if not _has_first_touch(loop.bodies[0], loop.axis):
        return loop
    first = _resolve_first_touch(loop.bodies[0], loop.axis, taken=True)
    rest = _resolve_first_touch(loop.bodies[0], loop.axis, taken=False)
    return Loop(axis=loop.axis, trip=loop.trip, outer=False,
                bodies=[first, rest], body_ranges=((0, 1), (1, loop.trip)))


# --- facts read off theta --------------------------------------------------

@dataclass(frozen=True)
class _CopyShape:
    """Where one global->LDS copy sits in the nest and how many regions it spans."""
    level: str
    regions: int
    spans: tuple
    axes: tuple


def _region_axes(theta, operand):
    """The region axes of this operand's global->LDS transfer, or None if it has no such move."""
    hop = next((h for h in (operand.hops or ()) if h.is_bulk), None)
    if hop is None:
        return None
    return tuple(axis.name for axis in presence_axes(theta, operand, hop))


def _register_hop_coverage(theta, operand):
    for hop in operand.hops:
        if not hop.is_bulk and hop.dst == Space.REGISTER and hop.quantum is not None:
            return hop.quantum
    return None


def _scope_of(theta, obligation, opname):
    if obligation.kind in ("inplace-WAR", "RAW-residency"):
        return "block" if theta.agent_distributed(opname) else "wave"
    return "wave"


def _consume_role(obligation):
    if obligation.kind == "RAW-residency":
        return (obligation.consumer.op, "read" if obligation.consumer.at == Space.SHARED else "wmma")
    return (obligation.consumer.op, "copy" if obligation.consumer.at == "shared" else "read")


def _group_enum_axes(theta, member_ops):
    """The axes a movement's copies are enumerated over: those of the member that OWNS THE WALK.

 The longest member defines it -- a fused group may pair a split operand with an unsplit one, and
 taking whichever member came first in the key would enumerate the short one and never visit the
 long one's later regions.  Uniform splits keep the first member, as before.
    """
    bulk = [(o, _region_axes(theta, o)) for o in member_ops]
    bulk = [(o, axes) for o, axes in bulk if axes is not None]
    if not bulk:
        return ()
    return max(bulk, key=lambda pair: max(1, pair[0].split))[1]


def _group_spanned_axes(theta, member_ops):
    """Every region axis a movement spans -- the union over its members, not one member's."""
    out = []
    for operand in member_ops:
        for axis in (_region_axes(theta, operand) or ()):
            if axis not in out:
                out.append(axis)
    return tuple(out)


def _axis_stride(axis, spans, regions, extents):
    """How many values of `axis` this movement skips between issues."""
    if axis not in spans:
        return max(1, extents.get(axis, 1))  # invariant over the axis -> issue once
    others = 1
    for other in spans:
        if other != axis:
            others *= max(1, extents.get(other, 1))
    own = max(1, regions // max(1, others))
    full = max(1, extents.get(axis, 1))
    if own > full or full % own:
        raise RuntimeError(
            f"copy spanning {list(spans)} has {regions} regions, which gives a count of {own} on "
            f"axis {axis!r} of extent {full}: the deficit is not attributable to one axis, so "
            f"its stride is ambiguous.")
    return full // own


def _group_guard(theta, operand, groups):
    """`Cond` predicates restricting a group-restricted read to the coordinates it owns."""
    if not groups:
        return ()
    labels = operand.fragment.groups()
    grouping_axis = getattr(operand.fragment, "grouping_mode", None)
    if len(labels) < 2 or not grouping_axis:
        return ()
    idx = sorted(labels.index(group) for group in groups if group in labels)
    if len(idx) == len(labels) or not idx:
        return ()  # the whole fragment -- nothing to exclude
    if idx != list(range(idx[0], idx[-1] + 1)):
        raise RuntimeError(
            "register groups %r of %s are not a contiguous run of %r -- a group guard would "
            "need a disjunction, which Pred cannot express" % (groups, operand.name, labels))
    extent = next((axis.extent for axis in theta.inner_axes() if axis.name == grouping_axis), 0)
    if not extent or extent % len(labels):
        raise RuntimeError(
            "grouping axis %r of %s has extent %r, which %d register groups do not divide "
            "evenly" % (grouping_axis, operand.name, extent, len(labels)))
    # The partition is INTERLEAVED, so a group's index is the RESIDUE of its block, not a range.
    quantum = group_block_quantum(theta, operand)
    guard_expr = Expr(carry=(((grouping_axis, 1),), 0, quantum), mod=len(labels))
    out = []
    if idx[0] > 0:
        out.append(Pred(guard_expr, ">=", idx[0]))
    if idx[-1] < len(labels) - 1:
        out.append(Pred(guard_expr, "<=", idx[-1]))
    return tuple(out)


def _check_peel_is_emittable(theta, peel, chunk):
    outer = {axis.name for axis in theta.outer_axes()}
    extra = [level for level in peel.peeled_levels() if level in outer and level != chunk]
    if extra:
        raise NotImplementedError(
            f"theta asks for a peel at outer level(s) {extra} besides the summation chunk "
            f"{chunk!r}, with depths { {l: peel.depth(l) for l in extra} }. Peeling those needs an "
            f"enclosing loop this emitter does not build. Refusing, rather than emitting the "
            f"chunk-only peel and silently dropping them.")
    outer_reads = [(operand, level, depth) for (operand, role, level), depth in peel.offsets.items()
                   if role == READ and depth and level in outer]
    if outer_reads:
        raise NotImplementedError(
            "theta asks a read to run ahead at an outer level: "
            + "; ".join(f"off({op}:read, {level})={depth}" for op, level, depth in outer_reads)
            + ". This emitter only shifts the copy, so the read would keep its old slot while the "
              "peel deepened around it, and the obligation check would still pass. Refusing, "
              "rather than applying half of it.")
    hoists = _boundary_hoists(theta, peel)
    if hoists:
        raise NotImplementedError(
            "theta asks for a copy to move across a level boundary: "
            + "; ".join(f"off({hop.op}:{hop.role}, {hop.level})={hop.delta} relocates the leading "
                        f"{hop.delta} instance(s) of level {hop.home}" for hop in hoists)
            + ". Moving them into the previous outer iteration's drain is not implemented. "
              "Refusing, rather than dropping them.")
    if peel.deferrals:
        raise RuntimeError(
            "a negative read-ahead offset is not meaningful: "
            + "; ".join(f"off({op}:{role}, {level})={depth}"
                        for (op, role, level), depth in sorted(peel.deferrals.items()))
            + ". Offsets count forward. Deferring a read is a positive offset on an operand that "
              "runs in reverse, not a negative offset on one that runs forward.")


@dataclass(frozen=True)
class _NestShape:
    """The loop levels of one theta, and which operand's read is emitted at each."""
    all_levels: list       # every inner axis as (name, extent), in loop order
    levels: list           # the same, minus the axes a wave serves -- the ones we emit loops for
    position: dict         # axis name -> its index in `levels`
    leaf_axis: str         # the innermost level the wmma sits at
    copy_ops: list         # operands with a global->LDS move
    reads: list            # operands with a ->register move
    read_level: dict       # operand name -> the axis its read is emitted at
    scale: bool            # more than two reads means MX scale operands are present
    units: dict            # movement group key -> (members, region count)


def _nest_shape(theta) -> _NestShape:
    wave_served = theta.wave_served_axes()
    all_levels = [(axis.name, axis.extent) for axis in theta.inner_axes()]
    levels = [(name, extent) for name, extent in all_levels if name not in wave_served]
    reads = [o for o in theta.operands if o.hops and o.hops[-1].dst == Space.REGISTER]
    wmma_axes = ({axis for o in theta.output_operands() for axis in presence(theta, o)}
                 | set(summation_names(theta)))
    return _NestShape(
        all_levels=all_levels,
        levels=levels,
        position={name: i for i, (name, _extent) in enumerate(levels)},
        leaf_axis=next((name for name, _e in reversed(levels) if name in wmma_axes),
                       levels[-1][0] if levels else None),
        copy_ops=[o for o in theta.operands if o.hops and o.hops[0].dst == Space.SHARED],
        reads=reads,
        read_level={o.name: operand_level(theta, o, READ) for o in reads},
        scale=len(reads) > 2,
        units={key: (members, regions) for key, members, regions in theta.movement_units()})


# --- the emitter -----------------------------------------------------------

class _Emitter:
    """Builds the rolled loop nest for one theta: prologue, steady loop, drain, short path.

    Everything is symbolic.  A copy targets LDS buffer `(iter + off) mod buffers` rather than a
    named buffer, and a read's register slot is an `Expr` in the same way, so software pipelining
    is arithmetic in the address instead of duplicated code.  Nothing is unrolled here; the
    renderer and the GIR lowering do that.
    """

    def __init__(self, theta, depths, ledger=None, plans=None):
        self.theta, self.S = theta, depths
        self.plans = plans or Schedule(theta, depths)  # ONE table; everything below reads it
        self.peel = peel_depths(theta, depths, self.plans)
        self.off = self.peel.copy_off()          # per operand: how many chunks ahead it loads
        self.dr = self.peel.requested_steps      # register read-ahead depth
        self.M = self.peel.chunk_depth           # pipeline depth, in K-chunks
        self.outer_var = theta.summation_chunk_name("iter")
        self.reads_ahead = readahead_level(theta) is not None
        self._copy_first = copy_first_operands(theta, depths, self.plans)
        self._copy_shapes = {}
        _check_peel_is_emittable(theta, self.peel, self.outer_var)

        self.awaits_by_site = {}
        for obligation in (build_ledger(theta, depths, self.plans) if ledger is None else ledger):
            if obligation.kind != "readahead-residency":
                self.awaits_by_site.setdefault(_consume_role(obligation), []).append(obligation)

        shape = _nest_shape(theta)
        self.all_inner, self.inner = shape.all_levels, shape.levels
        self.pos, self.leaf_axis = shape.position, shape.leaf_axis
        self.copy_ops, self.reads = shape.copy_ops, shape.reads
        self.read_level, self.scale, self._units = shape.read_level, shape.scale, shape.units

        for group in theta.fuse_groups:
            members = [name for name in group if name in self.off]
            if len({self.off[name] for name in members}) > 1:
                raise RuntimeError(
                    f"fuse group {group} mixes prefetch offsets "
                    f"{{{', '.join(f'{n}:off={self.off[n]}' for n in members)}}} -- a fused "
                    f"cooperative load must share one offset; split the group or equalize it.")

    # --- the four phases ---------------------------------------------------

    def build(self):
        loop = self._steady_loop()
        if self.M == 0:
            return self._prologue() + [loop]
        return [Cond(pred=Pred(Expr(var="T"), ">", self.M),
                     then=self._prologue() + [loop] + self._drain(),
                     els=self._short_trip(), kind="peel_validity")]

    def _prologue(self):
        """Ramp in: the copies for the M chunks ahead of iteration 0, then the register fill."""
        if self.M == 0 and not self.dr:
            return []
        body = []
        for step in range(self.M):
            copies = self._nest_copy_runs(self.copy_insts(PROLOGUE, step))
            if copies:
                body.append(Bind(axis=self.outer_var, value=cst(step - self.M), body=copies))
        body += self.fill_reads()
        return [Peel(kind="prologue", axis=self.outer_var, k=self.M, body=body)]

    def _steady_loop(self):
        """The rolled body: one copy per operand group, one read per operand, one wmma."""
        body = self._defer_reloads(self.copy_insts(STEADY) + self.nest())
        return Loop(axis=self.outer_var,
                    trip=Pred(Expr(var=self.outer_var), "<", Expr(var="T", add=-self.M)),
                    outer=True, bodies=[body])

    def _drain(self):
        """Ramp out: the M chunks still in flight when the steady loop stops."""
        if self.M == 0:
            return []
        first_suppressed = self.M - max(1, self.peel.reach)
        body = [self._drain_step(step, bool(self.dr and step >= first_suppressed))
                for step in range(self.M)]
        return [Peel(kind="drain", axis=self.outer_var, k=self.M, body=body)]

    def _drain_step(self, step, suppress_ahead):
        copies = self._nest_copy_runs(self.copy_insts(DRAIN, step))
        body = self._defer_reloads(copies + self.nest(suppress_ahead=suppress_ahead))
        return Bind(axis=self.outer_var, value=Expr(var="T", add=(step - self.M)), body=body)

    def _short_trip(self):
        """T <= M: the pipeline never fills, so emit M plain steps with no read-ahead."""
        out = []
        for step in range(self.M):
            copies = self._nest_copy_runs(self.copy_insts(SHORT, step))
            node = Bind(axis=self.outer_var, value=cst(step),
                        body=copies + self.nest(read_shift=0))
            if step:
                node = Cond(pred=Pred(Expr(var="T"), ">", step), then=[node], els=[],
                            kind="short_step_validity")
            out.append(node)
        return out

    # --- copies: global -> LDS ---------------------------------------------

    def copy_insts(self, phase, step=0):
        """One `Load` per movement group, targeting the LDS buffer that phase assigns it."""
        by_name = {operand.name: operand for operand in self.copy_ops}
        out, emitted = [], set()
        for operand in self.copy_ops:
            key = next((k for k in self._units if operand.name in k), (operand.name,))
            if key in emitted:
                continue
            emitted.add(key)
            member_ops = [by_name[name] for name in key]
            regions = self._units[key][1]
            off, buffers = self.off[operand.name], lds_buffers(self.theta, operand)
            slot = self._copy_slot(phase, off, buffers, step)
            if slot is None:
                continue  # this operand is not in flight at this step
            war = phase == STEADY or phase == DRAIN or (phase == SHORT and step >= max(1, buffers))
            inst = self._copy_inst(member_ops, slot, regions, war=war)
            axes = _group_enum_axes(self.theta, member_ops) if regions > 1 else ()
            order = {name: i for i, (name, _extent) in enumerate(self.inner)}
            self._copy_shapes[id(inst)] = _CopyShape(
                level=max(axes, key=lambda a: order.get(a, -1)) if axes else None,
                regions=regions,
                spans=_group_spanned_axes(self.theta, member_ops) if regions > 1 else (),
                axes=axes)
            out.append(inst)
        return out

    def _copy_slot(self, phase, off, buffers, step):
        """The LDS buffer this copy writes, or None if the operand is not in flight yet/still."""
        if phase == STEADY:
            return Expr(var=self.outer_var, mod=buffers, add=off)
        if phase == PROLOGUE:
            if off < self.M - step:
                return None  # has not entered the pipeline yet
            return cst((step + off - self.M) % max(1, buffers))
        if phase == SHORT:
            return cst(step % max(1, buffers))
        if off > self.M - 1 - step:
            return None  # already drained
        return Expr(var=self.outer_var, mod=buffers, add=off)

    def _copy_inst(self, member_ops, slot, regions, war=True):
        hop = member_ops[0].hops[0]
        size = sum(geometry.operand_tile_bytes(self.theta, operand) // max(1, operand.split)
                   for operand in member_ops)
        axes = _group_enum_axes(self.theta, member_ops) if regions > 1 else ()
        awaits, seen = (), set()
        for operand in member_ops:
            for await_node in self._site_awaits(operand.name, "copy"):
                if not war and is_war(await_node.kind):
                    continue  # prologue: no read has vacated the slot yet
                key = (str(await_node.dep), await_node.counter, await_node.kind, await_node.scope)
                if key not in seen:
                    seen.add(key)
                    awaits += (await_node,)
        load = Load(tuple(operand.name for operand in member_ops), hop.src, hop.dst, 0,
                    part=0, n_parts=1, coord=tuple((a, None) for a in axes), size_bytes=size)
        return Inst(op=load, placement=Placement(space=Space.SHARED, slots=(("", slot),)), awaits=awaits)

    def _defer_reloads(self, body):
        return self._nest_copy_runs(move_reloads_after_last_use(body, self._copy_first))

    def _nest_copy_runs(self, body):
        """Fold each maximal run of copies into one loop nest ordered by the loop order."""
        out, run = [], []

        def flush():
            if run:
                out.extend(self._nest_copies([(self._copy_shapes[id(node)], node) for node in run]))
                run.clear()

        for node in body:
            if id(node) in self._copy_shapes:
                run.append(node)
            else:
                flush()
                out.append(node)
        flush()
        return out

    def _nest_copies(self, made):
        """Put split copies under loops over their region axes, first-touch guarded."""
        flat = [inst for shape, inst in made if shape.level is None]
        order = [name for name, _extent in self.all_inner]
        extents = dict(self.all_inner)
        levels = []
        for shape, _inst in made:
            for axis in (shape.axes or ((shape.level,) if shape.level is not None else ())):
                if axis not in levels:
                    levels.append(axis)
        levels.sort(key=order.index)
        if not levels:
            return flat

        def build(depth):
            axis = levels[depth]
            body = []
            for shape, inst in made:
                if shape.level != axis:
                    continue
                node = inst
                for outer_axis in reversed(levels[:depth]):
                    stride = _axis_stride(outer_axis, shape.spans, shape.regions, extents)
                    if stride > 1:  # not present at every value of the outer axis
                        node = Cond(pred=Pred(Expr(var=outer_axis, mod=stride), "==", 0),
                                    then=[node], els=[], kind="first_touch")
                body.append(node)
            if depth + 1 < len(levels):
                body.append(build(depth + 1))
            return Loop(axis=axis, trip=extents[axis], outer=False, bodies=[body])

        return flat + [build(0)]

    # --- reads: LDS -> registers -------------------------------------------

    def read_inst(self, operand, kiter_note, shift=None, groups=None):
        """One `Load` into registers, its coordinate advanced by the read-ahead distance."""
        # PER OPERAND.  `self.dr` is the deepest ANY operand asks for; a full-inner one derives a
        # shallower depth, and priming it for the peer's is what leaves its obligation undischarged.
        distance = requested_read_ahead(self.theta, operand) if shift is None else shift
        placement = _read_placement(self.theta, operand, self.S, prefetch_steps=distance,
                                    groups=groups, plans=self.plans)
        hop = operand.hops[-1]
        awaits = self._site_awaits(operand.name, "read", kiter_note=kiter_note, groups=groups)
        advance, strides, span, extents = _readahead_shift(self.theta, operand, distance, self.S,
                                                           groups, self.plans)
        coverage = read_coverage(self.theta, operand)
        coord = tuple((axis, _shifted_coord(axis, advance, strides, span, extents, coverage))
                      if advance else (axis, None)
                      for axis in varying_axes(self.theta, operand))
        load = Load((operand.name,), hop.src, Space.REGISTER, 0, coord=coord,
                    size_regs=geometry.frag_regs(self.theta, operand), advance=advance,
                    quantum=_register_hop_coverage(self.theta, operand))
        return Inst(op=load, placement=placement, awaits=awaits)

    def _read_groups(self, operand):
        """Split the register groups into those that prefetch and those that load in place."""
        groups = operand.fragment.groups()
        if not self.dr or len(groups) < 2:
            return (None,)
        depth = {g: self.plans.plan(operand, g, self.dr).prefetch_steps for g in groups}
        ahead = tuple(g for g in groups if depth[g] >= 1)
        in_place = tuple(g for g in groups if depth[g] == 0)
        if not ahead or not in_place:
            return (None,)  # one shape for the whole fragment -- one instruction
        return (ahead, in_place)

    def _first_touch_axes(self, operand):
        """Axes this read does not vary over, so it is issued once instead of every iteration."""
        varying = set(varying_axes(self.theta, operand))
        if self.read_level[operand.name] not in self.pos:
            return []
        level = self.pos[self.read_level[operand.name]]
        out = [(name, 0) for (name, _extent) in self.inner[:level] if name not in varying]
        coverage = read_coverage(self.theta, operand)
        out += [(name, int(coverage[name])) for (name, _extent) in self.inner if name in coverage]
        return out

    def _read_anchor(self, operand, groups, depth):
        """The invariant-axis coordinates a prefetching read must sit at, if any."""
        distance = _readahead_shift(self.theta, operand, depth, self.S, groups, self.plans)[0]
        if not distance:
            return {}
        invariant = {name for name, _step in self._first_touch_axes(operand)}
        if not invariant:
            return {}
        want = {}
        for group in (groups or operand.fragment.groups()):
            positions = reload_anchor(self.theta, operand, group,
                                      max(1, self.S.get(operand.name, group)), distance)
            if positions is None:
                return {}  # no legal placement -- this group does not read ahead
            for coord in positions.values():
                for axis, value in coord.items():
                    if axis not in invariant or not value:
                        continue
                    assert want.setdefault(axis, value) == value, (
                        "read-ahead of %s wants its refill at two values of the invariant axis "
                        "%r (%r and %r); `reload_positions` should have refused this"
                        % (operand.name, axis, want[axis], value))
        return want

    def fill_reads(self):
        """The reads that fill the register pipeline before iteration 0."""
        return [self._fill_read_inst(operand, coord)
                for operand in self.reads
                for coord in preloaded_tiles(self.theta, operand,
                                             requested_read_ahead(self.theta, operand),
                                             self.S, self.plans)]

    def _fill_read_inst(self, operand, coord_dict):
        placement = _read_placement(self.theta, operand, self.S, prefetch_steps=0, plans=self.plans).at(coord_dict)
        hop = operand.hops[-1]
        awaits = tuple(a for a in self._site_awaits(operand.name, "read",
                                                   kiter_note=self.outer_var)
                       if not is_war(a.kind))  # nothing has been read yet, so nothing to vacate
        load = Load((operand.name,), hop.src, Space.REGISTER, 0,
                    coord=tuple((axis, coord_dict[axis])
                                for axis in varying_axes(self.theta, operand)),
                    size_regs=geometry.frag_regs(self.theta, operand),
                    quantum=_register_hop_coverage(self.theta, operand))
        return Inst(op=load, placement=placement, awaits=awaits)

    # --- the loop nest ------------------------------------------------------

    def nest(self, suppress_ahead=False, with_wmma=True, read_shift=None):
        """The inner loop nest, preceded by any read placed above it."""
        distance = self.dr if read_shift is None else read_shift
        head = []
        for operand in self.reads:
            if self.read_level[operand.name] in self.pos:
                continue  # an inner level -- `build_level` emits it
            for groups in self._read_groups(operand):
                if groups and len(operand.fragment.groups()) > 1:
                    raise RuntimeError(
                        "operand %s is register-partitioned into %r but its read is hoisted above "
                        "the inner nest, where the grouping axis %r is not in scope -- the "
                        "group's `covers` must be restricted instead of guarded"
                        % (operand.name, operand.fragment.groups(),
                           operand.fragment.grouping_mode))
                head.append(self.read_inst(operand, self.outer_var, shift=distance, groups=groups))
        if self.inner:
            return head + [self.build_level(0, suppress_ahead, with_wmma, read_shift)]
        return head + ([self.wmma_inst()] if with_wmma else [])

    def build_level(self, depth, suppress_ahead=False, with_wmma=True, read_shift=None):
        """One loop level: the reads placed here, the wmma if this is the leaf, then inward."""
        distance = self.dr if read_shift is None else read_shift
        axis, extent = self.inner[depth]
        body, deferred = [], []
        for operand in [o for o in self.reads if self.read_level[o.name] == axis]:
            for groups in self._read_groups(operand):
                node, runs_ahead = self._guarded_read(operand, groups, distance, suppress_ahead)
                (deferred if runs_ahead else body).append(node)
        if with_wmma and axis == self.leaf_axis:
            body.append(self.wmma_inst())
        if depth + 1 < len(self.inner):
            body.append(self.build_level(depth + 1, suppress_ahead, with_wmma, read_shift))
        body.extend(deferred)  # a read that runs ahead sits after the reads of the old value
        return _split_first_iteration(Loop(axis=axis, trip=extent, outer=False, bodies=[body]))

    def _guarded_read(self, operand, groups, ceiling, suppress_ahead):
        """One read plus the guards its placement needs; the flag says it must be deferred."""
        # `ceiling` is the deepest ANY operand asks for.  A full-inner one derives a shallower
        # depth, and placing its read at the peer's depth anchors a FIRST TOUCH at the last pass of
        # its invariant axis -- where every wmma in that pass has already read it.
        depth = min(ceiling, requested_read_ahead(self.theta, operand))
        node = self.read_inst(operand, self.outer_var, shift=depth, groups=groups)
        advance, strides, span, _extents = _readahead_shift(self.theta, operand, depth, self.S,
                                                            groups, self.plans)
        coverage = read_coverage(self.theta, operand)
        if suppress_ahead and advance and self.reads_ahead:
            guard = Pred(Expr(terms=position_terms(strides, coverage), add=advance), "<", span)
            node = Cond(pred=guard, then=[node], els=[], kind="readahead_suppress")
        for group_pred in _group_guard(self.theta, operand, groups):
            node = Cond(pred=group_pred, then=[node], els=[], kind="reg_group")
        anchor = self._read_anchor(operand, groups, depth)
        if anchor and isinstance(node, Inst):
            node = replace(node, anchor=tuple(sorted(anchor.items())))
        for guard_axis, step in reversed(self._first_touch_axes(operand)):
            lhs = Expr(var=guard_axis, mod=step) if step > 1 else Expr(var=guard_axis)
            node = Cond(pred=Pred(lhs, "==", anchor.get(guard_axis, 0)), then=[node], els=[],
                        kind="first_touch")
        return node, bool(anchor)

    def wmma_inst(self):
        scales = ("MXSA[m,k]", "MXSB[n,k]") if self.scale else ()
        mma = Mma(a="A[m,k]", b="B[n,k]", acc="C[m,n]", kiter=0, block="A(k)*B(k)",
                  coord=tuple((name, None) for name, _extent in self.inner), scales=scales)
        awaits = ()
        for operand in self.reads:
            awaits += self._site_awaits(operand.name, "wmma")
        placement = {operand.name: _read_placement(self.theta, operand, self.S, prefetch_steps=0, plans=self.plans)
                     for operand in self.reads}
        return Inst(op=mma, placement=placement, awaits=awaits)

    # --- shared -------------------------------------------------------------

    def _site_awaits(self, opname, role, kiter_note=None, groups=None):
        """The obligations the ledger says this site must wait on, as `Await` nodes."""
        out = []
        operand = next((o for o in self.theta.operands if o.name == opname), None)
        region = None
        if self.theta.per_region_completion and operand is not None:
            region = next(iter(_region_axes(self.theta, operand) or ()), None)
        labels = set(operand.fragment.groups()) if operand is not None else set()
        obligations = list(self.awaits_by_site.get((opname, role), []))
        movement = _movement_name(self.theta, opname)
        if movement != opname:
            obligations += self.awaits_by_site.get((movement, role), [])
        for obligation in obligations:
            if (groups is not None and is_war(obligation.kind)
                    and obligation.producer.at in labels
                    and obligation.producer.at not in groups):
                continue  # another split's register-ring WAR
            quals = (kiter_note, region) if obligation.kind == "RAW-residency" else ()
            note = ("refill WAR: slot vacated by prior read" if is_war(obligation.kind)
                    else f"{opname} resident")
            out.append(Await(dep=obligation.producer.qualified(*quals),
                             counter=obligation.counter,
                             scope=_scope_of(self.theta, obligation, opname),
                             kind=obligation.kind, note=note))
        return tuple(out)


# --- entry points ----------------------------------------------------------

def build_ir(theta, depths, ledger=None, plans=None):
    return _Emitter(theta, depths, ledger, plans).build()


def emit_mainloop(theta):
    depths, floor = build_S(theta)
    resolve_coverage(theta, depths)
    plans = Schedule(theta, depths)
    ledger = build_ledger(theta, depths, plans)
    ir = discharge_once(build_ir(theta, depths, ledger, plans))
    undischarged = check_ledger_discharged(ledger, ir)
    defects = _validate(theta, ir, depths, undischarged=undischarged)
    if defects:
        shape = "ord=[%s] off=%s S_reg=%s" % (
            ",".join(axis.name for axis in theta.inner_axes()),
            dict(getattr(theta, "off_map", {}) or {}),
            {operand.name: dict(depths.groups_of(operand.name)) for operand in theta.operands
             if getattr(operand, "role", "input") == "input"})
        raise RuntimeError("LoopIR structural gate failed:\n  " + "\n  ".join(defects)
                           + "\n  theta: " + shape)
    return {"S": depths, "floor": floor, "ir": ir, "ledger": ledger,
            "ledger_empty": not undischarged, "undischarged": undischarged,
            "n_obl": len(ledger),
            "tensor_per_kiter": theta.tensor_instrs_per_kiter()}

# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""theta -- the canonical Spacetime-layout point the decoder consumes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .ir import COPY, FORWARD, HOP_COUNTER, READ, REVERSE, STORE, Space


# ---  ------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Axis:
    """One tiled traversal axis: a name and its extent."""
    name: str
    extent: int = 0  # static trip count; >0 = inner (unrolled), 0 = outer (runtime loop)

    @property
    def is_outer(self) -> bool:
        return self.extent == 0


@dataclass
class Fragment:
    """The register-buffer structure of an operand's VGPR placement."""
    broadcast_axes: set = field(default_factory=set)
    parts: tuple = (1,)  # slot count per reuse group; (1,)=one group, (1,1)=lo/hi
    labels: tuple = None  # optional pretty label per group (default g0,g1,...)
    grouping_mode: str = None  # tile mode whose values the groups partitionform)
    group_policy: dict = None  # label -> 'unroll'|'overlap'|'inplace'|int W; "*"=default

    def __post_init__(self):
        if isinstance(self.parts, int):
            if self.parts < 1:
                raise RuntimeError(f"Fragment.parts={self.parts} must be >=1")
            self.parts = (1,) * self.parts

    def group_labels(self) -> tuple:
        if self.labels is not None:
            return tuple(self.labels)
        return tuple(f"g{i}" for i in range(len(self.parts)))

    def groups(self) -> tuple:
        return self.group_labels()

    def size_of(self, label) -> int:
        labs = self.group_labels()
        return self.parts[labs.index(label)] if label in labs else 1

    def total_slots(self) -> int:
        return sum(self.parts)

    def group_broadcast(self, group=None) -> set:
        return set(self.broadcast_axes)

    def policy_of(self, group):
        gp = self.group_policy or {}
        return gp.get(group, gp.get("*", "unroll"))


# --------------------------------------------------------------------------- The OP-CLASS role

HOP_ROLE = {
    (Space.GLOBAL,   Space.SHARED):   COPY,
    (Space.SHARED,   Space.REGISTER): READ,
    (Space.GLOBAL,   Space.REGISTER): READ,
    (Space.REGISTER, Space.SHARED):   STORE,
    (Space.SHARED,   Space.GLOBAL):   STORE,
    (Space.REGISTER, Space.GLOBAL):   STORE,
}


# --------------------------------------------------------------------------- TRAJECTORY SENSE --

_SPACE_RANK = {Space.GLOBAL: 0, Space.SHARED: 1, Space.REGISTER: 2}


def path_direction(op) -> str:
    """forward (toward the compute) or reverse (away from it) for one operand."""
    sense = REVERSE if op.is_output else FORWARD
    ranks = [(_SPACE_RANK.get(h.src), _SPACE_RANK.get(h.dst)) for h in (op.hops or ())]
    net = [b - a for a, b in ranks if a is not None and b is not None]
    if net and ((sense == FORWARD) != (sum(net) > 0)):
        raise RuntimeError(
            f"op-class {op.name!r} is is_output={op.is_output} (sense {sense}) but its hops "
            f"{[f'{h.src}->{h.dst}' for h in op.hops]} run the other way -- the trajectory sense "
            f"and the path must agree.  Sense is per OP-CLASS; the role is per HOP.")
    return sense


@dataclass
class Hop:
    """One leg of an operand's staging path (path)."""
    src: str; dst: str
    vector_elems: int = 0  # elements moved per instruction per lane (0 = whole frag)
    kind: str = None  # 'tdm' | 'load'; None => inferred from (src,dst)
    split: int = 1  # TDM region-split count (only meaningful for kind='tdm')
    quantum: object = None  # ir.CoverageMap None = identity, one tile per instruction
    phi_width: int = 1  # Phi: movement instances merged into one instruction
    rho_span: int = 0  # rho: sub-agent span in tiles (0 = no sub-agent distribution)
    region_agent_relative: bool = False

    def __post_init__(self):
        if self.kind is None:
            self.kind = "tdm" if (self.src, self.dst) == (Space.GLOBAL, Space.SHARED) else "load"

    @property
    def is_bulk(self) -> bool:
        return self.kind == "tdm"

    @property
    def role(self) -> str:
        """This hop's operand role -- `copy` / `read` / `store` (see `HOP_ROLE`)."""
        try:
            return HOP_ROLE[(self.src, self.dst)]
        except KeyError:
            raise RuntimeError(
                f"hop {self.src}->{self.dst} has no op-class role; extend HOP_ROLE `off` is "
                f"indexed by op-class, and an op-class is (operand, hop))") from None

    @property
    def counter(self): return HOP_COUNTER[(self.src, self.dst)]


@dataclass
class Operand:
    """One tensor path in generic geometry."""
    name: str
    free_mode: str
    hops: list
    fragment: Fragment
    frag_elems: int = 4
    elem_bytes: int = 2
    lds_buffers: int = 0  # S_shared for this operand; 0 = unstated (falls back to delta)
    region_axes: tuple = ()
    #: {region axis: how many values of it one of THIS operand's regions spans}.  The K walk is
    #: shared, so an operand that splits K coarser than the axis steps holds a region across
    #: several of its values; 1 means the axis value IS the region index.
    region_span: dict = field(default_factory=dict)
    free_split: int = 1
    wave_region_span: int = 1
    role: str = "input"  # 'input' (loaded operand) | 'output' (accumulator operand)

    @property
    def is_output(self) -> bool:
        return self.role == "output"

    @property
    def is_input(self) -> bool:
        return self.role == "input"

    def region_of(self, coord) -> tuple:
        """The region THIS operand's coordinate names, one value per region axis.

        The shared axis counts steps; this operand's region spans `region_span` of them, so the
        index is the axis value divided by that span -- the ONE place that division happens."""
        at = dict(coord)
        return tuple(int(at.get(axis, 0)) // max(1, int(self.region_span.get(axis, 1)))
                     for axis in self.region_axes)

    @property
    def split(self) -> int:
        """TDM region-split count = the `split` of this operand's bulk (tdm) hop, else 1."""
        for hop in self.hops:
            if hop.is_bulk:
                return max(1, hop.split)
        return 1

    def has_bulk(self) -> bool:
        return any(hop.is_bulk for hop in self.hops)


# ---  ------------------------------------------------------------------------------------------------

AGENT_LEVELS = ("block", "wave", "subwave")

#: The finest level a synchronization scope may name
SYNC_FLOOR_LEVEL = "wave"


RESORT_ROLES = ("read", "copy")


@dataclass(frozen=True)
class Resort:
    """A looped mode traversed by the agents instead of by the coordinate."""
    axis:   str
    level:  str
    extent: int
    origin: tuple = ()
    role:   str = "read"

    def __post_init__(self):
        if self.level not in AGENT_LEVELS:
            raise RuntimeError("resort level %r not in %r" % (self.level, AGENT_LEVELS))
        if int(self.extent) < 1:
            raise RuntimeError("resort extent must be >= 1, got %r" % (self.extent,))
        if self.role not in RESORT_ROLES:
            raise RuntimeError("resort role %r not in %r" % (self.role, RESORT_ROLES))

    @property
    def opclass(self) -> str:
        return self.origin[0] if self.origin else ""

    @property
    def rank(self) -> int:
        return AGENT_LEVELS.index(self.level)


@dataclass(frozen=True)
class Rho:
    """The agent assignment and role partition."""
    resort: tuple = ()  # tuple[Resort]
    roles:  tuple = ()  # the producer/consumer partition -- unimplemented, see docs/RHO_DESIGN.md

    def __post_init__(self):
        """`resort` is frozen at construction, so derive the served set here, once."""
        limit = AGENT_LEVELS.index("wave")
        object.__setattr__(self, "_served",
                           frozenset(r.axis for r in self.resort
                                     if r.rank <= limit and r.role == "read"))

    def at(self, *levels, role: str = "read") -> tuple:
        return tuple(r for r in self.resort if r.level in levels and r.role == role)

    def coarser_than(self, level: str, role: str = "read") -> tuple:
        lim = AGENT_LEVELS.index(level)
        return tuple(r for r in self.resort if r.rank <= lim and r.role == role)

    def copy_agents(self, opname: str, default: int = 1) -> int:
        hits = [role for role in self.resort if role.role == "copy" and role.opclass == opname]
        if not hits:
            return default
        return max(int(role.extent) for role in hits)

    def served_axes(self) -> frozenset:
        """The axes served above wave level. Computed once: `resort` is a frozen tuple."""
        return self._served

    def span_over(self, axes, level: str = "subwave") -> int:
        hits = [role for role in self.at(level) if role.axis in axes]
        if not hits:
            return 0
        if len(hits) > 1:
            raise RuntimeError(
                "rho: %d modes resorted to %s within one presence set (%s) -- the span would be a "
                "product and names only the single sub-agent partition; no shipping "
                "configuration produces this."
                % (len(hits), level, ", ".join(sorted(role.axis for role in hits))))
        return int(hits[0].extent)


@dataclass
class Theta:
    operands: list
    ord: tuple  # the full permutation of looped axes, outer -> inner
    reg_bytes: int = 4
    lanes: int = 32
    fuse_groups: list = field(default_factory=list)  # Phi: lists of operand tokens
    fuse: int = 0
    lds: dict = field(default_factory=dict)  # op -> {pad, block}
    rho: "Rho" = field(default_factory=lambda: Rho())
    waves: int = 1
    off_map: dict = field(default_factory=dict)
    per_region_completion: bool = False
    S: object = None

    def __post_init__(self):
        """The axis nest is fixed at construction, so split it here rather than on every read.

        Nothing rebinds `theta.ord`; use `dataclasses.replace` to get a theta with a different
        nest, which runs this again.
        """
        self.ord = tuple(self.ord)
        self.inner = tuple(axis for axis in self.ord if not axis.is_outer)
        self.outer = tuple(axis for axis in self.ord if axis.is_outer)

    # --- convenience lookups ------------------------------------------------
    def op(self, name):
        """The operand of this name."""
        for operand in self.operands:
            if operand.name == name:
                return operand
        raise RuntimeError("no operand %r in theta (have %s)"
                         % (name, ", ".join(operand.name for operand in self.operands)))

    def agent_distributed(self, opname) -> bool:
        if not any(hop.dst == Space.SHARED for hop in self.op(opname).hops):
            return False  # never staged in shared memory: no cross-agent buffer
        return self.waves > 1

    def movement_units(self):
        copy_ops = [operand for operand in self.operands if operand.hops and operand.hops[0].dst == Space.SHARED]
        by_name = {operand.name: operand for operand in copy_ops}
        units, seen = [], set()
        for op in copy_ops:
            key = next((tuple(tile for tile in group if tile in by_name) for group in self.fuse_groups
                        if op.name in group), (op.name,))
            if key in seen:
                continue
            seen.add(key)
            members = [by_name[name] for name in key]
            # THE WALK IS AS LONG AS THE LONGEST MEMBER.  A member with fewer regions than the
            # unit does not move on every step; `Operand.split` says how many it owns, and the
            # copy side skips it on the steps it does not (see `loopir_to_gir._absent_at_region`).
            units.append((key, members, max(max(1, operand.split) for operand in members)))
        return units

    def levels(self):
        return list(self.ord)

    def outer_axes(self):
        return self.outer

    def reduction_chunk_mode(self):
        outer = self.outer_axes()
        return outer[-1] if outer else None

    def summation_chunk_name(self, default=None):
        axis = self.reduction_chunk_mode()
        return axis.name if axis is not None else default

    def inner_axes(self):
        return self.inner

    def free_extent(self, mode_name):
        for axis in self.ord:
            if axis.name == mode_name:
                return axis.extent
        return 1

    def off_at(self, opname, role, level_name) -> int:
        name = opname if isinstance(opname, str) else opname.name
        return int(self.off_map.get((name, role, level_name), 0))

    def off_of(self, operand, hop, level_name) -> int:
        return self.off_at(operand, hop.role, level_name)

        # --- the axes it varies over & reduction ---

    def output_operands(self):
        return [operand for operand in self.operands if operand.is_output]

    def wave_served_axes(self) -> set:
        return self.rho.served_axes()




    def tensor_instrs_per_kiter(self) -> int:
        """distinct global->shared instructions per kiter (one shared counter tracks all)."""
        byname = {operand.name: operand for operand in self.operands if operand.hops and operand.hops[0].dst == Space.SHARED}
        fused = {tile for group in self.fuse_groups for tile in group}
        n = 0
        for group in self.fuse_groups:  # each fused group: one instr per region
            members = [byname[tile] for tile in group if tile in byname]
            n += max((max(1, operand.split) for operand in members), default=1)
        for name, operand in byname.items():  # unfused copies
            if name not in fused:
                n += max(1, operand.split)
        return max(1, n)


@dataclass
class DepthMap:
    """Buffer depth per PLACEMENT and rotating coordinate -- not per operand."""
    depths: dict = field(default_factory=dict)  # (op, region, group) -> depth

    def set(self, operand, group, depth, region=None):
        self.depths[(operand, region, group)] = depth

    def get(self, operand, group, default=1, region=None):
        exact = self.depths.get((operand, region, group))
        if exact is not None:
            return exact
        if region is not None:
            return self.depths.get((operand, None, group), default)
        vals = {depth for (o, _role, g), depth in self.depths.items() if o == operand and g == group}
        if not vals:
            return default
        if len(vals) > 1:
            raise RuntimeError(
                f"S({operand}, {group}) is asked WITHOUT a region but its placements disagree "
                f"({sorted(vals)}) -- a per-placement depth cannot be collapsed to per-operand "
                f"(the depth map is indexed by placement, not by operand).  Name the region.")
        return vals.pop()

    def regions_of(self, operand, group):
        rs = sorted({role for (o, role, g) in self.depths if o == operand and g == group},
                    key=lambda role: (role is not None, role))
        return rs or [None]

    def groups_of(self, operand, region=None):
        groups = {group for (o, _role, group) in self.depths if o == operand}
        return {group: self.get(operand, group, region=region) for group in groups}

    def __repr__(self):
        by = {}
        for (operand, role, group), depth in sorted(self.depths.items(), key=lambda kv: (kv[0][0], str(kv[0][1]),
                                                                        kv[0][2])):
            by.setdefault(operand, []).append(f"{group}:{depth}" if role is None else f"{group}@r{role}:{depth}")
        return " | ".join(f"{operand}=[{', '.join(v)}]" for operand, v in by.items())

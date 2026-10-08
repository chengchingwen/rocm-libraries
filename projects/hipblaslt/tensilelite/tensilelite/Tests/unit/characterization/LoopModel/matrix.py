# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The LoopModel decode matrix, and the digest taken over each cell.

Shared by the golden suite. Every configuration here is one that the shipping
`loopmodel_*.yaml` blocks can produce, so a digest change is a real emitted-kernel change.
"""

import hashlib
import itertools
import dataclasses

# The bf16 base is the existing shared fixture; the MX base mirrors the shipping
# `loopmodel_mxf8_gfx1250.yaml` block (MT64x64, MI 16x16x128, MXBlock 32, fused TDM).
from gir_fixtures import BF16_NT_KMN, mi_inputs

LOOP_ORDERS = ("KMN", "KNM", "MKN", "NKM", "MNK", "NMK", "KMNKMN", "KMNMNK", "KMKNMN")


class _OneByte:
    """Element type stub: MX operands are 1 byte per element."""

    def numBytes(self):
        return 1


def _waves(count):
    """MIWaveGroup for a wave count."""
    return {1: [1, 1], 2: [2, 1], 4: [2, 2]}[count]


def bf16_cells(order):
    """Every bf16 cell for one loop order: waves x split x PLR x PGR x DepthU."""
    for waves, (split_a, split_b), plr, pgr, depth_u in itertools.product(
            (1, 4), itertools.product((0, 1, 2), repeat=2), (0, 1, 2), (1, 2), (64, 128)):
        group = _waves(waves)
        yield ("bf16 %-7s w%d %d/%d plr%d pgr%d du%-3d"
               % (order, waves, split_a, split_b, plr, pgr, depth_u),
               {**BF16_NT_KMN, "LoopOrder": order, "MIWaveGroup": group,
                "MatrixInstruction": [16, 16, 32, 1, 1, 2, 2, group[0], group[1]],
                "DepthU": depth_u, "PrefetchGlobalRead": pgr, "PrefetchLocalRead": plr,
                "MIWaveTileA": 2, "MIWaveTileB": 2, "TDMInst": 3,
                "enableTDMA": True, "enableTDMB": True,
                "TDMSplitA": split_a, "TDMSplitB": split_b,
                "VectorWidthA": 1, "VectorWidthB": 1})


def bf16_fused_cells(order):
    """The shipping bf16 shape with the default fused row and a ONE-SIDED split.

    `bf16_cells` leaves TDMFuse unset, so the fused descriptor -- which every shipping bf16 kernel
    uses -- was uncharacterized, and with it the mixed-region group where one member rides a single
    region of the other's walk.
    """
    for (split_a, split_b), plr, pgr in itertools.product(
            ((2, 0), (0, 2), (2, 2), (1, 0), (0, 1)), (0, 1), (1, 2)):
        yield ("bffuse %-7s w4 %d/%d plr%d pgr%d" % (order, split_a, split_b, plr, pgr),
               {**BF16_NT_KMN, "LoopOrder": order, "MIWaveGroup": [2, 2],
                "MatrixInstruction": [16, 16, 32, 1, 1, 2, 2, 2, 2],
                "DepthU": 64, "PrefetchGlobalRead": pgr, "PrefetchLocalRead": plr,
                "MIWaveTileA": 2, "MIWaveTileB": 2, "TDMInst": 3, "TDMFuse": 0,
                "enableTDMA": True, "enableTDMB": True,
                "TDMSplitA": split_a, "TDMSplitB": split_b,
                "VectorWidthA": 1, "VectorWidthB": 1})


def mx_cells(order):
    """Every mxf8 cell for one loop order: split x PLR x PGR x DepthU."""
    for split, plr, pgr, depth_u in itertools.product((0, 1, 2), (0, 1, 2), (1, 2), (128, 256)):
        yield ("mxf8 %-7s w4 %d/%d plr%d pgr%d du%-3d"
               % (order, split, split, plr, pgr, depth_u),
               {"MatrixInstruction": [16, 16, 128, 1, 1, 2, 2, 2, 2], "DepthU": depth_u,
                "MIWaveTileA": 2, "MIWaveTileB": 2, "MatrixInstK": 128,
                "ProblemType": {"DataType": _OneByte(), "MXBlockA": 32, "MXBlockB": 32},
                "PrefetchGlobalRead": pgr, "PrefetchLocalRead": plr, "NumWaves": 4,
                "enableTDMA": 1, "enableTDMB": 1, "NumLdsBlk": 2, "LoopOrder": order,
                "TDMSplitA": split, "TDMSplitB": split, "TDMFuse": 0,
                "VectorWidthA": 2, "VectorWidthB": 2, "LocalReadVectorWidthMXS": 8,
                # The emitter reads mxf8 with ds_load_*_b128 (blockWidth 4 regs, bpeDS 1), so the
                # width is 16.  Unset, the adapter defaults to 4 and counts FOUR TIMES the reads a
                # real kernel issues -- every wait derived from a fixture would then be wrong.
                "LocalReadVectorWidthA": 16, "LocalReadVectorWidthB": 16,
                "UnrollMajorLDSMXSA": 1, "UnrollMajorLDSMXSB": 1})


def mx_mixed_cells(order):
    """Cells whose FUSE GROUP mixes region counts: a scale (always one region) fused with a split
    data operand, and the default row's asymmetric pair.  `mx_cells` cannot reach these -- the
    default row groups like with like, so every member of a set there owns the same number of
    regions.
    """
    shapes = [(fuse, a, b) for fuse in (1, 2, 3)
              for a, b in ((1, 0), (0, 1), (1, 1), (2, 0), (0, 2))]
    shapes += [(0, 1, 0), (0, 0, 1), (0, 2, 0), (0, 0, 2)]
    for fuse, split_a, split_b in shapes:
        yield ("mxmix %-7s f%d %d/%d" % (order, fuse, split_a, split_b),
               {"MatrixInstruction": [16, 16, 128, 1, 1, 2, 2, 2, 2], "DepthU": 256,
                "MIWaveTileA": 2, "MIWaveTileB": 2, "MatrixInstK": 128,
                "ProblemType": {"DataType": _OneByte(), "MXBlockA": 32, "MXBlockB": 32},
                "PrefetchGlobalRead": 2, "PrefetchLocalRead": 1, "NumWaves": 4,
                "enableTDMA": 1, "enableTDMB": 1, "NumLdsBlk": 2, "LoopOrder": order,
                "TDMSplitA": split_a, "TDMSplitB": split_b, "TDMFuse": fuse,
                "VectorWidthA": 2, "VectorWidthB": 2, "LocalReadVectorWidthMXS": 8,
                # The emitter reads mxf8 with ds_load_*_b128 (blockWidth 4 regs, bpeDS 1), so the
                # width is 16.  Unset, the adapter defaults to 4 and counts FOUR TIMES the reads a
                # real kernel issues -- every wait derived from a fixture would then be wrong.
                "LocalReadVectorWidthA": 16, "LocalReadVectorWidthB": 16,
                "UnrollMajorLDSMXSA": 1, "UnrollMajorLDSMXSB": 1})


def cells(order):
    """Every cell for one loop order, bf16 then mxf8.

    Read counts are re-derived HERE, after each cell's overrides: a cell that changes
    MatrixInstruction or a read width would otherwise inherit the base fixture's stale count.
    """
    for family in (bf16_cells, bf16_fused_cells, mx_cells, mx_mixed_cells):
        for key, kernel in family(order):
            yield key, mi_inputs(kernel)


def find(order, wanted):
    """The kernel whose cell key matches `wanted`, ignoring runs of whitespace.

    Keys are column-aligned for readability, so matching them literally is fragile; this lets a
    caller name a cell as "bf16 KMNKMN w1 1/1 plr1 pgr1 du64" and not count spaces.
    """
    squash = lambda text: " ".join(text.split())
    for key, kernel in cells(order):
        if squash(key) == squash(wanted):
            return kernel
    raise KeyError("no cell %r in %s" % (wanted, order))


def decode(kernel):
    """Decode one kernel to its three rendered artefacts, or the error that stopped it.

    Returns `(loop_ir, gir, ledger)` as text. A raise is part of the characterized behaviour and
    is captured rather than propagated, so a config that legitimately refuses still pins a golden.
    """
    from Tensile.LoopModel import adapter
    from Tensile.LoopModel.decoder import decode_theta
    from Tensile.LoopModel.loopir import dependences, render
    from Tensile.Lowering import lower_to_gir
    from Tensile.Lowering.gir.passes import pipeline, run_pipeline
    from Tensile.Lowering.gir.render import render_gir
    try:
        theta = adapter.params_to_theta(adapter.kernel_to_params(kernel))
        decoded = decode_theta(theta)
        edges = "\n".join(
            repr(edge)
            for edge in sorted(dependences(decoded.program), key=repr)
        )
        gir = lower_to_gir(decoded.program, decoded.profile)
        run_pipeline(gir, passes=pipeline())
        return render(decoded.program), render_gir(gir), edges
    except Exception as exc:                       # a refusal is behaviour worth pinning
        return ("%s: %s" % (type(exc).__name__, str(exc).split("\n")[0]),) * 3


def digest(kernel):
    """A short stable hash of everything one config emits."""
    return hashlib.md5("\x00".join(decode(kernel)).encode()).hexdigest()[:16]


_DISPLAY_FIELDS = frozenset({"label", "note", "version"})
_SEMANTIC_TAGS = {"CoverageMap": "TransferCoverage"}
_META_KEYS = {
    "M": "peel_depth",
    "S": "buffer_depths",
    "fuse_groups": "fused_copy_groups",
    "read_quantum": "read_coverage",
    "region_agent_relative": "wave_relative_regions",
}


def _emitted(result, name, old_name=None):
    if isinstance(result, dict):
        return result[name if name in result else old_name]
    return getattr(result, name)


#: Keys whose value is `id(some_object)`. A raw address is not a fact about the schedule -- it
#: changes every process -- but WHICH endpoints share one is, so each is replaced by its ordinal
#: in first-visit order. Dropping the key instead would lose the producer/consumer pairing.
_IDENTITY_KEYS = frozenset({"identity"})


def _semantic_value(value, identities=None):
    """Stable object structure without display-only text, field names, or object addresses."""
    identities = {} if identities is None else identities

    def recur(item):
        return _semantic_value(item, identities)

    if type(value).__name__ == "DepthMap":
        return ("BufferDepths", tuple(sorted(
            ((recur(key), recur(item)) for key, item in value.depths.items()), key=repr)))
    if type(value).__name__ == "BufferDepths":
        entries = {}
        for operand in value.theta.operands:
            for group, depth in value.groups_of(operand.name).items():
                entries[(operand.name, None, group)] = depth
            if value.include_shared and operand.trajectory.shared is not None:
                entries[(operand.name, None, "shared")] = value.shared(operand.name)
        return ("BufferDepths", tuple(sorted(
            ((recur(key), recur(item)) for key, item in entries.items()), key=repr)))
    if dataclasses.is_dataclass(value):
        tag = _SEMANTIC_TAGS.get(type(value).__name__, type(value).__name__)
        fields = tuple(
            identities.setdefault(
                id(getattr(value, item.name)),
                len(identities),
            )
            if item.name == "_identity"
            else recur(getattr(value, item.name))
            for item in dataclasses.fields(value)
            if item.name not in _DISPLAY_FIELDS
        )
        return (tag, fields)
    if isinstance(value, dict):
        items = [(recur(_META_KEYS.get(key, key)),
                  identities.setdefault(item, len(identities))
                  if key in _IDENTITY_KEYS and isinstance(item, int) else recur(item))
                 for key, item in value.items()]
        return tuple(sorted(items, key=repr))
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((recur(item) for item in value), key=repr))
    if isinstance(value, (tuple, list)):
        return tuple(recur(item) for item in value)
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return value
    return repr(value)


def _effect_domain_value(domain):
    return (
        tuple(
            (axis.name, lower.render(), upper.render())
            for axis, lower, upper in domain.loops
        ),
        tuple(
            (axis.name, value.render())
            for axis, value in domain.bindings
        ),
        tuple(predicate.render() for predicate in domain.predicates),
        tuple(axis.name for axis in domain.local_axes),
        tuple((axis.name, scope.name) for axis, scope in domain.agent_axes),
    )


def _dependence_value(dependence, identities):
    def effect(value):
        operation = value.operation
        endpoint = value.endpoint
        endpoint_value = tuple(
            (item.name, getattr(endpoint, item.name))
            for item in dataclasses.fields(endpoint)
        )
        return (
            identities[id(operation._identity)],
            value.occurrence.path,
            type(endpoint).__name__,
            endpoint_value,
            value.kind.name,
            value.level.name,
            value.tile.operand.name,
            tuple(expression.render() for expression in value.tile.indices),
            value.predicate.render(),
        )

    domain = dependence.domain
    return (
        dependence.kind.name,
        effect(dependence.complete),
        effect(dependence.issue),
        _effect_domain_value(domain.complete_domain),
        _effect_domain_value(domain.issue_domain),
        tuple(
            (equality.complete.render(), equality.issue.render())
            for equality in domain.equalities
        ),
        tuple(
            (constraint.axis.name, constraint.distance)
            for constraint in domain.iteration_constraints
        ),
        domain.conservative,
    )


def _dependence_values(program):
    from Tensile.LoopModel.loopir import dependences, operations

    identities = {
        id(operation._identity): index
        for index, operation in enumerate(operations(program))
    }
    return tuple(
        _dependence_value(dependence, identities)
        for dependence in dependences(program)
    )


def semantic_digest(kernel):
    """Hash schedule, LoopIR, ledger, and GIR structure independent of rendering."""
    from Tensile.LoopModel import adapter
    from Tensile.LoopModel.decoder import decode_theta
    from Tensile.Lowering import lower_to_gir
    from Tensile.Lowering.gir.passes import pipeline, run_pipeline
    try:
        theta = adapter.params_to_theta(adapter.kernel_to_params(kernel))
        decoded = decode_theta(theta)
        program = lower_to_gir(decoded.program, decoded.profile)
        run_pipeline(program, passes=pipeline())
        value = (
            decoded.profile.depths,
            decoded.program,
            _dependence_values(decoded.program),
            program,
        )
    except Exception as exc:
        value = ("error", type(exc).__name__)
    return hashlib.md5(repr(_semantic_value(value)).encode()).hexdigest()[:16]

# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""TensileLite Solution -> theta. The only place TensileLite vocabulary appears."""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd as _gcd

from . import traversal as geometry
from .ir import READ, Space
from .theta import (AgentAssignment, AgentAxisAssignment, Fragment, Global,
                    RegionLayout, Shared, Trajectory, Axis, Operand, Theta, group_labels)
from .traversal import readahead_level
from ..Components import TDMSplit as _tdm_split
from ..Components.TDMFuse import tdmFusedGroups, tdmWavePartition
WAVE32 = 32  # gfx1250 wave width
REG_BYTES = 4  # gfx1250 register width in bytes

def _per_operand(kernel, base, operand, preset):
    value = kernel.get("%s%s" % (base, operand))
    return preset if value is None or value < 0 else int(value)


def _k_tiles(kernel):
    mi_k = kernel["MatrixInstruction"][2]
    return max(1, kernel["DepthU"] // mi_k) * kernel["InnerUnroll"]


def _frag_elems(kernel, free_mi):
    mi_k = kernel["MatrixInstruction"][2]
    mi_b = kernel["MatrixInstruction"][3] if len(kernel["MatrixInstruction"]) > 3 else 1
    return max(1, free_mi * mi_k * mi_b // WAVE32)


def _acc_frag_elems(kernel):
    mi_b = kernel["MatrixInstruction"][3] if len(kernel["MatrixInstruction"]) > 3 else 1
    return max(1, kernel["MatrixInstruction"][0] * kernel["MatrixInstruction"][1] * mi_b // WAVE32)


def _mx_frag_elems(kernel, free_mi, mxblock):
    data = _frag_elems(kernel, free_mi)
    dup = max(1, WAVE32 // max(1, kernel["MatrixInstruction"][0]))
    return max(1, data // max(1, mxblock) * dup)


def _coalesce(kernel):
    return max(1, kernel["LocalReadVectorWidth"] // max(1, kernel["VectorWidthA"]))


CANON_NAMES = ["K_split", "K_inner",
               "M_split", "M_inner", "N_split", "N_inner"]

#: tile axis -> its storage-region axis
INNER_TO_SPLIT = {"K_inner": "K_split", "M_inner": "M_split", "N_inner": "N_split"}


# ---------------------------------------------------------------------------
def _loop_order_word(order):
    """Normalize a LoopOrder param to a 6-letter axis word (each of K,M,N twice)."""
    word = str(order).upper()
    if len(word) == 3 and sorted(word) == ["K", "M", "N"]:
        word = "".join(c * 2 for c in word)  # "kmn" -> "kkmmnn"
    if len(word) != 6 or sorted(word) != ["K", "K", "M", "M", "N", "N"]:
        raise ValueError(
            f"LoopOrder={order!r}: must be a 3-letter axis order (e.g. 'KMN','NMK') or a "
            f"6-letter word with each of K,M,N twice (first occ=split, second=inner, e.g. "
            f"'KMKNMN'); inner->split orders are not expressible (by design).")
    return word


def canonical_loop_order(order):
    word = _loop_order_word(order)
    if all(word[i] == word[i + 1] for i in (0, 2, 4)):
        return word[0] + word[2] + word[4]
    return word


#: `WmmaInnerOrder` is an index into this: the order the TILE axes nest in, innermost last.
#: The numbering is the one `Common/ValidParameters.py` documents.
WMMA_INNER_ORDERS = ("KMN", "KNM", "MNK", "MKN", "NMK", "NKM")

#: `WmmaOuterOrder` names the axis whose SPLIT sits outermost; 0 keeps each split beside its tile.
WMMA_OUTER_AXES = (None, "M", "N", "K")


def wmma_loop_order(params) -> str:
    """The traversal word for a solution.

    `WmmaInnerOrder`/`WmmaOuterOrder` are the parameters; `LoopOrder` is accepted only as a
    direct spelling for hand-built thetas, which is what the unit fixtures pass.
    """
    if "WmmaInnerOrder" in params or "WmmaOuterOrder" in params:
        return loop_order_of(params.get("WmmaInnerOrder", 1), params.get("WmmaOuterOrder", 0))
    return params.get("LoopOrder", "KMN")


def loop_order_of(inner_order: int, outer_order: int) -> str:
    """The loop-order word `WmmaInnerOrder` and `WmmaOuterOrder` name.

    `outer_order` 0 keeps every split beside the tile it divides, which is the 3-letter order.
    Otherwise that axis's split leads and the rest follow in tile order; at most two splits are
    ever live, so naming the outer one fixes the traversal and the third's position is free.
    """
    try:
        inner = WMMA_INNER_ORDERS[int(inner_order) - 1]
    except (IndexError, TypeError, ValueError):
        raise ValueError(
            f"WmmaInnerOrder={inner_order!r} must be 1..{len(WMMA_INNER_ORDERS)} "
            f"({', '.join(WMMA_INNER_ORDERS)})")
    if not 0 <= int(outer_order) < len(WMMA_OUTER_AXES):
        raise ValueError(f"WmmaOuterOrder={outer_order!r} must be 0..{len(WMMA_OUTER_AXES) - 1}")
    outer = WMMA_OUTER_AXES[int(outer_order)]
    if outer is None:
        return inner
    return outer + "".join(a for a in inner if a != outer) + inner


def _word_to_modes(word):
    AXIS = {"K": "K", "M": "M", "N": "N"}
    seen = set()
    roles = []
    for c in word:
        axis = AXIS[c]
        roles.append(f"{axis}_split" if axis not in seen else f"{axis}_inner")
        seen.add(axis)
    return roles


def _canonical_ord(substeps, splits, m_inner, n_inner, order):
    """The eight canonical axes with their extents, ordered by the LoopOrder word."""
    # K_inner is the longest run both operands can take without changing
    # storage region. K_split counts those runs.
    # kResA and kResB are operand metadata, not loop axes.
    a_regions = max(1, splits.kSplit * splits.kResA)
    b_regions = max(1, splits.kSplit * splits.kResB)
    k_inner = _gcd(max(1, substeps // a_regions), max(1, substeps // b_regions))
    extents = {
        "K_inner": max(1, k_inner),
        "K_split": max(1, substeps // max(1, k_inner)),
        "M_split": max(1, splits.mSplit), "M_inner": max(1, m_inner),
        "N_split": max(1, splits.nSplit), "N_inner": max(1, n_inner),
    }
    axes = {role: Axis(role, extents[role]) for role in CANON_NAMES}
    roles = _word_to_modes(_loop_order_word(order))
    def _keep(role):
        # The loop order determines which tile axes exist, regardless of their
        # extents, so keep every tile axis in the nest.
        if role in INNER_TO_SPLIT:
            return True
        return extents[role] > 1
    # ord carries only the shared link. Each operand keeps its own residual
    # region information rather than adding it to the shared nest.
    inner_ord = [axes[role] for role in roles if _keep(role)]
    if not inner_ord:  # fully degenerate: keep K_inner as the 1 loop
        inner_ord = [axes["K_inner"]]
    iter_mode = Axis("iter")  # extent=0 => is_outer (the reduction chunk / LDS-ring axis)
    axes["iter"] = iter_mode
    ord_ = [iter_mode] + inner_ord
    return ord_, axes, extents


# --- build -----------------------------------------------------------------


DEFAULTS = {
    "DepthU": 128, "MatrixInstruction": [16, 16, 64, 1], "MIWaveTile": [1, 2],
    "PrefetchGlobalRead": 2, "PrefetchLocalRead": 1, "HalfPLR": 0,
    "TDMInst": 3, "TDMSplit": [1, 1], "TDMIterateMode": 0,
    "GlobalReadVectorWidthA": 8, "GlobalReadVectorWidthB": 8,
    "VectorWidthA": 2, "VectorWidthB": 2, "LocalReadVectorWidth": 4,
    "DirectToVgprA": False, "DirectToVgprB": False,
    "WaveSeparateGlobalReadA": 0, "WaveSeparateGlobalReadB": 0,
    "1LDSBuffer": 0, "LdsPadA": -1, "LdsPadB": -1, "LdsBlockSizePerPad": 0,
    "InnerUnroll": 1, "ElemBytes": 2,
    "MXBlockA": 0, "MXBlockB": 0, "TDMFuse": 0,
    "NumWaves": 1,
    "PrefetchGlobalReadA": -1, "PrefetchGlobalReadB": -1,
    "PrefetchLocalReadA": -1, "PrefetchLocalReadB": -1,
    "ClusterLocalRead": 0, "ClusterLocalReadA": -1, "ClusterLocalReadB": -1,
    "VgprGroupA": -1, "VgprGroupB": -1, "VgprAllocA": -1, "VgprAllocB": -1,
    "LDSBufferA": -1, "LDSBufferB": -1,
    "LDSBufferMXSA": -1, "LDSBufferMXSB": -1,
    "ReadVectorElems": {},
    "ReadQuantum": {},
    "PerRegionCompletion": False,
}


@dataclass(frozen=True)
class Splits:
    """TDMSplit axis factors plus the independent physical wave-to-region span."""
    aMT: int; bMT: int; aDU: int; bDU: int
    mSplit: int; nSplit: int  # the M/N region extent = that operand's MT split
    kSplit: int; kResA: int; kResB: int  # shared K link (gcd), then each operand's residual
    waveRegionsA: int; waveRegionsB: int  # regions one wave's read addresses may occupy


@dataclass(frozen=True)
class Ord:
    """The loop nest this LoopOrder gives: the axes in order, and each operand's regions."""
    ord_: list
    M_BCAST: set; N_BCAST: set
    aRegions: tuple; bRegions: tuple
    mInner: int; nInner: int
    #: {operand: {region axis: axis values ONE of its regions spans}} -- 1 when the axis value
    #: is the region index, >1 when the shared K walk steps finer than that operand splits.
    regionSpan: dict = None


@dataclass(frozen=True)
class OperandFragment:
    """One operand's register fragment, the elements it holds, and the LDS ring behind it."""
    fragment: object
    frag_elems: int
    regions: int          # how many storage regions its copy writes
    lds_buffers: int


@dataclass(frozen=True)
class Fragments:
    """The fragment of each operand, by name, plus the builder the MX scale rings reuse."""
    by_operand: dict
    reg_fragment: object

    def __getitem__(self, name) -> OperandFragment:
        return self.by_operand[name]


@dataclass(frozen=True)
class MovementInputs:
    """Target movement widths, grouping, and explicit coverage."""
    grvwA: int; grvwB: int; lrvw: int
    read_elements: dict; instances_per_instruction: dict; read_coverage: dict; agent_group_sizes: dict


def _read_params(p):
    """The Solution params this decode reads, with the defaults applied."""
    kernel = dict(DEFAULTS); kernel.update(p)
    matrix_instruction = kernel["MatrixInstruction"]
    substeps = _k_tiles(kernel)  # reduction substeps per kiter (DU/MI_K x InnerUnroll)
    elem_bytes = kernel["ElemBytes"]
    fanM, fanN = kernel["MIWaveTile"][0], kernel["MIWaveTile"][1]
    copy_depth = max(0, kernel["PrefetchGlobalRead"])
    read_depth = max(0, kernel["PrefetchLocalRead"])
    coalesce = _coalesce(kernel)  # substeps sharing one hi-register (reuse rate)

    return copy_depth, read_depth, elem_bytes, fanM, fanN, kernel, substeps, matrix_instruction


def _read_splits(kernel):
    """TDMSplit as canonical region extents: per-operand MT, shared DU."""
    # TDMSplit: [A_MT, B_MT, A_DU, B_DU] -> canonical region extents.
    split_factors = kernel["TDMSplit"]
    if isinstance(split_factors, bool):
        split_factors = [2, 2] if split_factors else [1, 1]
    split_factors = list(split_factors) + [1] * (4 - len(split_factors))
    aMT, bMT, aDU, bDU = (max(1, split_factors[0]), max(1, split_factors[1]), max(1, split_factors[2]), max(1, split_factors[3]))
    mSplit, nSplit = aMT, bMT  # M region (A's fan), N region (B's fan)
    _split_wave_regions = list(kernel.get("TDMSplitWaveRegions", []) or []) + [0, 0]
    waveRegionsA = max(1, min(aMT, int(_split_wave_regions[0]) or aMT))
    waveRegionsB = max(1, min(bMT, int(_split_wave_regions[1]) or bMT))

    kSplit = _gcd(aDU, bDU)  # the shared link: both operands prefix it
    kResA = aDU // max(1, kSplit)  # A's own residual beyond the shared link
    kResB = bDU // max(1, kSplit)  # B's own residual

    return Splits(aMT, bMT, aDU, bDU, mSplit, nSplit, kSplit, kResA, kResB,
                  waveRegionsA, waveRegionsB)


def _build_ord(fanM, fanN, kernel, substeps, splits):
    """The canonical six-mode loop order for this LoopOrder word, plus each operand's regions."""
    # TDMSplit is a coordinate factorization, independent of which regions a physical wave reads.
    m_inner = _tdm_split.factor_axis(fanM, splits.mSplit, "MIWaveTileA")
    n_inner = _tdm_split.factor_axis(fanN, splits.nSplit, "MIWaveTileB")
    order_word = _loop_order_word(wmma_loop_order(kernel))
    ord_, canonical_axes, canonical_extents = _canonical_ord(
        substeps, splits, m_inner, n_inner, order_word)

    N_BCAST = {"N_split", "N_inner"}  # A is broadcast over N modes
    M_BCAST = {"M_split", "M_inner"}  # B is broadcast over M modes
    # K_split is a region axis only for the operand that actually splits K.
    aK, bK = max(1, splits.kSplit * splits.kResA), max(1, splits.kSplit * splits.kResB)
    aRegions = tuple(role for role in ("M_split", "K_split")
                     if canonical_extents[role] > 1 and (role != "K_split" or aK > 1))
    bRegions = tuple(role for role in ("N_split", "K_split")
                     if canonical_extents[role] > 1 and (role != "K_split" or bK > 1))
    kSplitExtent = max(1, canonical_extents["K_split"])
    span = {"A": {"K_split": max(1, kSplitExtent // aK)} if "K_split" in aRegions else {},
            "B": {"K_split": max(1, kSplitExtent // bK)} if "K_split" in bRegions else {}}

    return Ord(ord_, M_BCAST, N_BCAST, aRegions, bRegions, m_inner, n_inner, span)


def _read_movements(kernel):
    """The per-hop payload the target supplies: vector widths, the movement coverage, Phi."""
    # --- hops (per-hop delta = retime) ------------------------------------------------
    grvwA, grvwB = kernel["GlobalReadVectorWidthA"], kernel["GlobalReadVectorWidthB"]
    lrvw = kernel["LocalReadVectorWidth"]

    read_elements = dict(kernel.get("ReadVectorElems", {}) or {})

    read_coverage = dict(kernel.get("ReadQuantum", {}) or {})

    instances_per_instruction = dict(kernel.get("ReadPhi", {}) or {})
    agent_group_sizes = dict(kernel.get("ReadRho", {}) or {})

    return MovementInputs(grvwA, grvwB, lrvw, read_elements,
                          instances_per_instruction, read_coverage, agent_group_sizes)


def _build_agent_assignment(movements, nest):
    """Build agent-axis assignments and adapter movement inputs."""
    agent_inputs = {
        "A": "M_inner", "MXSA": "M_inner",
        "B": "N_inner", "MXSB": "N_inner",
    }
    _by_mode = {}

    def _assign(axis, level, extent, origin):
        prev = _by_mode.get(axis)
        if prev is not None:
            if (prev.level, prev.extent) != (level, int(extent)):
                raise RuntimeError(
                    "axis %r assigned twice with different answers -- %s/%d from %s vs "
                    "%s/%d from %s. An axis has ONE agent level."
                    % (axis, prev.level, prev.extent, prev.origin, level, extent, origin))
            return
        _by_mode[axis] = AgentAxisAssignment(
            axis=axis, level=level, extent=int(extent), origin=origin)

    for _name, _free in agent_inputs.items():
        _sub = int(movements.agent_group_sizes.get(_name, 0) or 0)
        if _sub > 1 and _free:  # sub-wave: the distributed carrier coverage
            _assign(_free, "subwave", _sub, (_name, READ))
    assignment = AgentAssignment(
        axis_assignments=tuple(_by_mode[m] for m in sorted(_by_mode)))

    def _configure_read(fragment, name, default):
        merged = max(1, int(movements.instances_per_instruction.get(name, 1) or 1))
        _free = agent_inputs.get(name)
        agent_group_size = (
            assignment.group_size_over({_free} if _free else set(), level="subwave")
            if int(movements.agent_group_sizes.get(name, 0) or 0) > 1 else 0)
        fragment.vector_elements = int(movements.read_elements.get(name, default) or default)
        fragment.instances_per_instruction = merged
        fragment.agent_group_size = agent_group_size
        fragment.coverage = movements.read_coverage.get(name)
        if fragment.coverage is None:
            fragment.coverage = geometry.derive_transfer_coverage(fragment)
        return fragment

    return _configure_read, agent_inputs, assignment


def _build_fragments(kernel, matrix_instruction, nest, splits):
    """The operands: their paths, fragments, register widths and MX scale rings."""
    # --- register width: DERIVED, not a parameter
    def reg_fragment(bcast, grp_mode, fan_extent, tc=None):
        return Fragment(broadcast_axes=set(bcast), parts=1, labels=group_labels(1),
                        grouping_mode=grp_mode, group_policy={"*": "pipeline"})

    fragA = reg_fragment(nest.N_BCAST, "M_inner", nest.mInner, "A")
    fragB = reg_fragment(nest.M_BCAST, "N_inner", nest.nInner, "B")

    fragElA = _frag_elems(kernel, matrix_instruction[0])
    fragElB = _frag_elems(kernel, matrix_instruction[1])
    splitA = splits.mSplit * splits.aDU
    splitB = splits.nSplit * splits.bDU

    numLdsBlk = kernel.get("NumLdsBlk", 2)
    ldsBufA = 0 if kernel["DirectToVgprA"] else _per_operand(kernel, "LDSBuffer", "A", numLdsBlk)
    ldsBufB = 0 if kernel["DirectToVgprB"] else _per_operand(kernel, "LDSBuffer", "B", numLdsBlk)
    ldsBufMXSA = _per_operand(kernel, "LDSBuffer", "MXSA", ldsBufA)
    ldsBufMXSB = _per_operand(kernel, "LDSBuffer", "MXSB", ldsBufB)
    for _name, _v in (("A", ldsBufA), ("B", ldsBufB),
                    ("MXSA", ldsBufMXSA), ("MXSB", ldsBufMXSB)):
        if _v and _v > numLdsBlk:
            raise RuntimeError(
                "LDSBuffer%s=%d exceeds the kernel's allocated NumLdsBlk=%d -- the LDS allocation "
                "is sized by Solution.py against MaxLDS, so a deeper ring is an out-of-bounds LDS "
                "write, not a tighter schedule.  Raise 1LDSBuffer/DepthU so the allocation grows, "
                "or keep LDSBuffer%s <= %d." % (_name, _v, numLdsBlk, _name, numLdsBlk))
    return Fragments(reg_fragment=reg_fragment, by_operand={
        "A":    OperandFragment(fragA, fragElA, splitA, ldsBufA),
        "B":    OperandFragment(fragB, fragElB, splitB, ldsBufB),
        "MXSA": OperandFragment(None, 0, splitA, ldsBufMXSA),
        "MXSB": OperandFragment(None, 0, splitB, ldsBufMXSB),
    })


def _build_operands(configure_read, elem_bytes, frags, movements, kernel, matrix_instruction,
                    nest, splits):
    """Build tensor identities and placement paths."""
    def region_layout(parent):
        is_a = parent == "A"
        return RegionLayout(
            axes=nest.aRegions if is_a else nest.bRegions,
            span=(nest.regionSpan or {}).get(parent, {}),
            free_split=max(1, splits.aMT if is_a else splits.bMT),
            wave_span=splits.waveRegionsA if is_a else splits.waveRegionsB)

    def input_operand(name, free_mode, direct, copy_width, parent):
        spec = frags[name]
        fragment = spec.fragment
        fragment.fragment_elements = spec.frag_elems
        if direct:
            fragment.vector_elements = copy_width
            path = Trajectory(Global(), fragment)
        else:
            shared = Shared(vector_elements=copy_width, split=spec.regions,
                            ring_depth=spec.lds_buffers, regions=region_layout(parent))
            path = Trajectory(Global(), shared,
                              configure_read(fragment, name, movements.lrvw))
        return Operand(name, free_mode, trajectory=path, elem_bytes=elem_bytes)

    A = input_operand("A", "M_inner", kernel["DirectToVgprA"], movements.grvwA, "A")
    B = input_operand("B", "N_inner", kernel["DirectToVgprB"], movements.grvwB, "B")

    accumulator = Fragment(
        broadcast_axes={"K_split", "K_inner"},
        fragment_elements=_acc_frag_elems(kernel))
    C = Operand("C", "M_inner", trajectory=Trajectory(accumulator),
                elem_bytes=REG_BYTES, role="output")
    operands = [A, B, C]

    hasMXA, hasMXB = kernel["MXBlockA"] > 0, kernel["MXBlockB"] > 0

    def add_scale(name, free_mode, parent, copy_width, broadcast, free_mi, block):
        fragment = frags.reg_fragment(broadcast, free_mode, 1, name)
        fragment.fragment_elements = _mx_frag_elems(kernel, free_mi, block)
        # A SCALE IS NEVER SPLIT, so it owns no region axis.  Borrowing the parent's gave it
        # `region_axes` with `split=1`, and every region-count reader had to work around that.
        shared = Shared(vector_elements=copy_width, ring_depth=frags[name].lds_buffers)
        operands.append(Operand(
            name, free_mode,
            trajectory=Trajectory(Global(), shared,
                                  configure_read(fragment, name, movements.lrvw)),
            elem_bytes=1))

    if hasMXA:
        add_scale("MXSA", "M_inner", "A", movements.grvwA, nest.N_BCAST,
                  matrix_instruction[0], kernel["MXBlockA"])
    if hasMXB:
        add_scale("MXSB", "N_inner", "B", movements.grvwB, nest.M_BCAST,
                  matrix_instruction[1], kernel["MXBlockB"])

    return hasMXA, hasMXB, operands


def _fuse_view(hasMXA, hasMXB, kernel):
    """The solution-shaped mapping `Components.TDMFuse` reads, built from the theta params.

    A row that needs scales the problem does not have declines and falls back to the default,
    which is `tdmGrouping`'s own answer rather than a second rule stated here.
    """
    tdmInst = int(kernel.get("TDMInst", 0) or 0)
    return {"TDMFuse": int(kernel.get("TDMFuse", 0) or 0),
            "NumWaves": max(1, int(kernel["NumWaves"])),
            "TDMInst": tdmInst,
            "enableTDMA": bool(tdmInst & 0x01), "enableTDMB": bool(tdmInst & 0x02),
            "ProblemType": {"MXBlockA": hasMXA, "MXBlockB": hasMXB}}


def _build_fuse(hasMXA, hasMXB, kernel, operands):
    """The Phi fusion groups: which operands share one load."""
    view = _fuse_view(hasMXA, hasMXB, kernel)
    present = {operand.name for operand in operands}
    groups = [[t for t in group if t in present] for group in tdmFusedGroups(view)]
    return view, [group for group in groups if len(group) > 1]


def _read_off_requests(kernel, operands, outer_level, copy_depth, read_depth):
    """PrefetchGlobalRead becomes off(copy, iter); PrefetchLocalRead is a per-operand request."""
    read_requests = {}
    for operand in operands:
        copies = _per_operand(kernel, "PrefetchGlobalRead", operand.name, copy_depth)
        # A scale reads the local-read dials of the tensor it scales, as its ring depth does.
        reads = _per_operand(kernel, "PrefetchLocalRead",
                             _REGISTER_SIDE.get(operand.name, operand.name), read_depth)
        if operand.movements and operand.movements[0].dst == Space.SHARED and copies > 0:
            movement = operand.movements[0]
            movement.destination.offsets[outer_level] = copies
        if (operand.movements and operand.movements[-1].dst == Space.REGISTER
                and operand.movements[-1].src == Space.SHARED and reads > 0):
            read_requests[operand.name] = reads
    return read_requests


def _copy_wave_shares(view, groups, operands, waves):
    """How many waves cooperate on each operand's global->LDS copy."""
    names = {operand.name for operand in operands}
    share = {}
    for group in groups:
        for name in group:
            share[name] = max(1, len(tdmWavePartition(view, name)[1]))
    for name in names:  # an operand in no group keeps every wave
        if not any(name in group for group in groups):
            share[name] = waves
    return {name: count for name, count in share.items() if name in names}


def _with_copy_shares(assignment, agent_inputs, shares):
    """Add copy-side wave assignments."""
    entries = tuple(AgentAxisAssignment(
                        axis=agent_inputs[name], level="wave", extent=count,
                        origin=(name, "copy"), role="copy")
                    for name, count in sorted(shares.items())
                    if name in agent_inputs and agent_inputs[name])
    return (AgentAssignment(
                axis_assignments=assignment.axis_assignments + entries)
            if entries else assignment)


def _build_theta(agent_inputs, copy_depth, read_depth, view, groups, kernel, nest, operands,
                 assignment):
    """The retime matrix `off`, the copy-side wave partition, and the assembled Theta."""
    waves = max(1, int(kernel["NumWaves"]))
    read_requests = _read_off_requests(
        kernel, operands, nest.ord_[0].name, copy_depth, read_depth)
    if waves > 1:
        assignment = _with_copy_shares(
            assignment, agent_inputs, _copy_wave_shares(view, groups, operands, waves))
    theta = Theta(operands=operands, ord=tuple(nest.ord_), reg_bytes=REG_BYTES, lanes=WAVE32,
                  fused_copy_groups=groups, agent_assignment=assignment, wave_count=waves,
                  per_region_completion=bool(kernel["PerRegionCompletion"]),
                  suppress_no_load_loop=bool(kernel.get("SuppressNoLoadLoop", False)))
    for operand in theta.operands:
        if operand.movements and operand.fragment:
            operand.fragment.grouping_mode = geometry.grouping_mode_name(theta, operand)
    _place_read_offsets(theta, read_requests)
    _set_register_depths(theta, kernel)
    return theta


#: an operand reads the PLR/CLR of the data tensor it belongs to
_REGISTER_SIDE = {"A": "A", "MXSA": "A", "B": "B", "MXSB": "B"}

class _StatedDepth:
    """A `depths`-shaped view of the depth just assigned, so `group_ring_depth` stays the one
    place that turns a per-group VA into the slots the emitter addresses."""

    def __init__(self, depth):
        self._depth = max(1, int(depth))

    def get(self, _name, _group):
        return self._depth


def _fit_read_ahead(theta, operand, group, depth) -> None:
    """Pull this operand's read-ahead back to what `depth` buffers can hold in flight.

    A lead of `L` keeps `L + 1` values live, so a ring of `depth` carries a lead of `depth - 1`
    and no more.  Asking for more does not deepen the pipeline, it refills a slot still in use.
    """
    axis = geometry.readahead_level(theta, operand)
    if axis is None:
        return
    offsets = operand.trajectory.fragment_fill.destination.offsets
    while int(offsets.get(axis[0], 0) or 0) > 0 and _lead_of(theta, operand, group) + 1 > depth:
        offsets[axis[0]] = int(offsets[axis[0]]) - 1


def _lead_of(theta, operand, group) -> int:
    return int(geometry.prefetch_distance_for(
        theta, operand, geometry.requested_read_ahead(theta, operand), (group,)))


def _share_regions(theta, operand, group, depth, regions) -> int:
    """`depth` cut to ONE region's share, if the ring walk says the regions may share it.

    They may when the regions LEAD the ring -- the outer loop finishes region 0 before region 1
    begins -- and when the read-ahead does not run past the region edge, which is the one thing
    that overlaps them.  The lead is pulled back to what one region can carry and the shared ring
    is then walked for real: anything short of a clobber is a ring that holds what it claims.
    `regions_lead_the_ring` alone is NOT enough -- it passed 56/384 cells that clobber.
    """
    if regions <= 1 or not geometry.regions_lead_the_ring(theta, operand, group):
        return depth
    # AN ALL-INNER OPERAND HAS NOTHING TO DEFER.  Its resident set already spans every region --
    # that is what rereading the whole set every trip means -- so one buffer per region IS the
    # share, and halving again leaves one buffer holding two regions' worth.
    if geometry.reloads_whole_set(theta, operand):
        return depth
    # AND NOTHING IT MUST OUTLIVE IS NESTED ABOVE THE RING.  An axis the operand is INVARIANT
    # over is a pass the resident set has to survive: inside the region it means the next region
    # arrives while this one is still being reread, outside it means the set is wanted again on
    # the next turn of that axis.  Either way a shared slot serves the wrong region to a wmma --
    # 48/384 for the inner case, 16/384 for the outer.  Both operands split is where it bites:
    # each one's ring straddles the OTHER's region axis.
    order = [axis.name for axis in theta.ord]
    varies = set(geometry.varying_axes(theta, operand))
    regional = {name for name in (getattr(operand, "region_axes", ()) or ()) if name in order}
    ringed = [name for name, _extent in geometry.ring_axes(theta, operand, group)
              if name in order]
    last = min(order.index(name) for name in ringed)
    if any(order[index] not in varies and order[index] not in regional
           for index in range(1, last)):          # 0 is `iter`, the trip, which nothing outlives
        return depth
    shared = max(1, depth // regions)
    axis = geometry.readahead_level(theta, operand)
    offsets = operand.trajectory.fragment_fill.destination.offsets if axis else {}
    restore = dict(offsets)
    if axis is not None:
        while int(offsets.get(axis[0], 0) or 0) > 0 and _lead_of(theta, operand, group) + 1 > shared:
            offsets[axis[0]] = int(offsets[axis[0]]) - 1
    # THE FLOOR IS THE AUTHORITY, asked of the SHARED ring: the regions no longer each keep a
    # copy of what is live, so it is the per-region reuse windows that decide.  A share below it
    # aliases two live generations however outer the region axis is.
    floor = geometry.group_reuse_floor(theta, operand, group, shared_regions=True)
    verdict = geometry.register_reuse_verdict(
        theta, operand, group, shared, _lead_of(theta, operand, group))
    if shared >= floor and verdict != geometry.CLOBBER:
        return shared
    offsets.clear()
    offsets.update(restore)                      # the share was refused; leave the lead alone
    return depth


def _scheduled_lead(theta, operand, group) -> int:
    """The read-ahead distance this operand will actually be SCHEDULED at, not the one asked for.

    `schedule._placeable_read_ahead` lowers the request when the refill has no legal position, so
    a refusal that charges the raw request judges a pipeline depth the kernel never gets.
    """
    from .schedule import BufferDepths, _placeable_read_ahead
    depths = BufferDepths(theta, include_shared=False)
    return int(geometry.prefetch_distance_for(
        theta, operand, _placeable_read_ahead(theta, operand, depths), (group,)))


def _set_register_depths(theta, kernel):
    """The ring depth is STATED by the solution, not searched for.

    This is the ROTATION count, not the width.  `ClusterLocalRead` keeps the operand's whole set
    resident, and what that costs depends on whether the operand has an OUTER axis:

      all-inner  -- it rereads the same set every trip, so nothing rotates and the ring is 1
      has-outer  -- it advances along its outer axis, so a full buffer is one slot per value

    Without clustering the ring is the read-ahead depth plus the value in use, raised to the
    reuse floor: a value consumed again after an axis the operand is invariant over has to
    survive that pass, and a ring that does not divide its positions aliases two live values.
    Every arm is measured on the operand's OWN axis, which is why the depth is per operand at
    all.  This replaces the depth SEARCH and nothing else -- the grouping is left as it was.
    """
    for operand in theta.operands:
        if not (operand.movements and operand.fragment):
            continue
        side = _REGISTER_SIDE.get(operand.name)
        if side is None:
            continue
        cluster = _per_operand(kernel, "ClusterLocalRead", side,
                               int(kernel.get("ClusterLocalRead", 0) or 0))
        prefetch = _per_operand(kernel, "PrefetchLocalRead", side,
                                int(kernel.get("PrefetchLocalRead", 0) or 0))
        # POSITIONS THE RING ENUMERATES, off the same `ring_axes` the slot expression walks, so a
        # buffer can never out-count its slots.  A REGION AXIS IS ONE OF THEM: it is a loop
        # coordinate like any other, so it is counted here, once, and nothing multiplies the
        # depth by it afterwards.  VG and VA then stand free of TDMSplit, PLR and CLR.
        group = operand.fragment.groups()[0]
        # VG IS SETTLED FIRST, because the derivation below ASKS the geometry questions and VG
        # changes their answers: `group_reuse_floor` searches up to the ring size in BUFFERS, so
        # asking it before the fragment carries VG capped the search at the un-cut ring, found
        # no width, and silently returned that ceiling -- which the floor's own cache then froze.
        asked_vg = kernel.get("VgprGroup" + side)
        asked_va = kernel.get("VgprAlloc" + side)
        split = max(1, geometry.split_unit_divisor(theta, operand))
        vg = split if asked_vg is None or int(asked_vg) == -1 else max(1, int(asked_vg))
        operand.fragment.vgpr_group = max(1, vg)
        region_axes = set(getattr(operand, "region_axes", ()) or ())
        ring = [max(1, int(extent))
                for name, extent in geometry.ring_axes(theta, operand, group)
                if name not in region_axes]
        # A region appears at its own multiplicity, not its axis extent, so it is taken from
        # `region_positions` rather than from the enumerator -- counted here and nowhere else.
        regions = max(1, geometry.region_positions(theta, operand))
        positions = max(1, geometry.product(ring) if ring else 1) * regions
        if cluster:
            # All-inner holds one buffer; otherwise the whole ring.  A read-ahead still fits: the
            # refill is in place, deferred past the last use.
            depth = regions if geometry.reloads_whole_set(theta, operand) else positions
            # CLUSTERING OVER A SPLIT KEEPS ONE REGION RESIDENT where the regions can share
            # the ring -- the halved full, which is what CLR=1 over a split should cost.
            depth = _share_regions(theta, operand, group, depth, regions)
            # A CLUSTERED RING STILL HAS TO CARRY ITS OWN LEAD.  This arm sizes the ring to the
            # resident set and never asked the floor, so an all-inner operand took ONE buffer
            # while `PrefetchLocalRead` gave it a lead of 2 -- the refill then landed on the
            # value in use.  The set is resident; reading ahead inside it buys nothing.
            _fit_read_ahead(theta, operand, group, depth)
            # AND IT MUST COVER WHAT THE READS REACH.  "The whole set is resident" says what the
            # ring has to HOLD, not that one buffer can hold it: where the set turns through more
            # than one rotation place, one buffer serves the wrong tile to a wmma.  The reuse
            # FLOOR is the wrong bound here -- it over-reports on a clustered ring (8/384) --
            # the right one is the places the reads actually write.
            depth = max(depth, geometry.ring_reached_positions(theta, operand, group, depth))
        else:
            # A scale takes its tensor's read-ahead but enumerates its own ring, and one scale
            # read can cover every free tile -- so the depth it asks for is bounded by the ring.
            # PLR IS CONVERTED TO BUFFERS HERE, regions included: they are concurrent, so a
            # lead of `PLR` keeps `PLR + 1` live in each of them.  The floor already speaks
            # buffers, so both sides of the max are the unit VA counts and the result IS VA.
            want = max(max(1, int(prefetch) + 1) * regions,
                       geometry.group_reuse_floor(theta, operand, group))
            # A WIDTH BELOW THE REGION DIGIT'S PLACE ERASES IT.  The region digit sits above
            # the enumerator, so a modulus at or under the enumerator's span folds every region
            # onto the same slot and two concurrent regions alias.  Where the regions rotate,
            # the ring has to span them.
            if regions > 1:
                want = max(want, positions)
            depth = next((width for width in range(want, positions + 1)
                          if positions % width == 0), positions)
        # VG/VA ARE THE AUTHORITY INSIDE THETA.  `ClusterLocalRead`/`PrefetchLocalRead` reach no
        # further than this function: they are one way of SAYING a (VG, VA) pair, and the pair is
        # what the model carries.  Stating them directly says shapes the pair cannot -- 1.5
        # buffers is VG=2/VA=3, and an all-inner operand's `U + P/2` is VG=2/VA=2n+1.
        # ONLY WHAT VG DIVIDES BEYOND THE SPLIT changes anything here: the split's own share is
        # already spent as the per-region unit.  A sub-iteration still consumes a whole unit, so
        # a unit cut `extra` ways takes `extra` buffers; at the derived VG that factor is 1 and
        # the depth is the one PLR/CLR gave.
        # A VG THE SPLIT DOES NOT DIVIDE HAS NO MEANING, and floors away silently: the split has
        # already cut the unit `split` ways and the buffer cannot be bigger than what it left.
        if asked_vg is not None and int(asked_vg) != -1 and vg % split:
            raise RuntimeError(
                "VgprGroup%s=%d: the split already cuts %s's prefetch unit %d ways, so a buffer "
                "is at most a %d-th of it and %d is not a multiple of %d.  Use a multiple of %d, "
                "or leave it at -1 to derive." % (side, vg, operand.name, split, split, vg,
                                                  split, split))
        extra = max(1, vg // split)
        # SCALED ON BOTH ARMS, clustered included: VG cuts the buffer, so the same resident set
        # takes that many more of them.  Leaving CLR=1 unscaled halved the block and kept the
        # unhalved ring, which is a depth that cannot hold what it claims.  CLR=1 still
        # OVERWRITES VA -- the whole set is resident by definition -- it just honours VG too.
        depth *= extra
        # VA IS THE BUFFER COUNT OUTRIGHT -- regions included, not multiplied on afterwards.
        if asked_va is not None and int(asked_va) != -1 and not cluster:
            depth = int(asked_va)
        operand.fragment.vgpr_group = max(1, vg)
        operand.fragment.ring_depths = {
            label: depth for label in operand.fragment.groups()}
        # THESE TWO CHECK A STATED PAIR, NOT A DERIVED ONE.  VG cuts the buffer, so VA has to
        # count more of them, and the caller is the one who has to get that right: the floor is
        # the smallest ring that keeps a value alive across every consumer pass, the ceiling is
        # all the positions the ring has to write.  A DERIVED depth is whatever PLR/CLR already
        # settled, and re-judging it here would refuse kernels that have always generated.
        if asked_vg is None and asked_va is None:
            continue
        if int(asked_vg if asked_vg is not None else -1) == -1 \
                and int(asked_va if asked_va is not None else -1) == -1:
            continue
        # VG HAS TO DIVIDE THE BLOCK IT CUTS.  A block of one tile has no half, and the cut
        # floors away silently -- leaving the allocation scaled for buffers the reads never fill.
        block = geometry.group_unit_tiles(theta, operand, operand.fragment.groups()[0])
        if extra > 1 and block % extra:
            raise RuntimeError(
                "VgprGroup%s=%d: %s holds %d tile(s) per buffer under this loop order, which %d "
                "does not divide, so there is no whole-tile %d-th of it to allocate.  Use a VG "
                "that divides %d, or leave it at -1 to derive."
                % (side, vg, operand.name, block, extra, extra, block))
        for label in operand.fragment.groups():
            # CHECK THE SLOTS THE EMITTER ADDRESSES, not the per-group VA.  `group_ring_depth`
            # gives each region its own turn, and `_check_dead_slots` and the slot modulus both
            # count that way; judging `depth` instead agrees only at one region, so a split
            # operand's over-ask used to walk past here and surface as a check_plan defect.
            slots = geometry.group_ring_depth(theta, operand, label, _StatedDepth(depth))
            # THE FLOOR IS A SINGLE-TRIP MODEL.  It asks what stays live inside ONE turn of the
            # ring, so it cannot judge a ring the trip does NOT close -- 3 buffers over 4
            # positions is the 1.5-buffer rotation, which closes after three trips and carries a
            # lead the single-trip bound calls impossible (`lead + 1` of 5 on a 4-position ring).
            # Those widths are the ones `ring_trip_copies` replicates the body for, and they
            # verify clean; `check_plan` is their gate, not this.
            period = geometry.ring_positions(theta, operand, label, slots)
            if period % max(1, slots):
                continue
            # THE REFUSAL ASKS FOR THE UNCHARGED FLOOR -- what actually aliases, not what the
            # lead would cost if its reload were never anchored.  The charged number is right
            # for the scheduler and too strict as a gate.
            floor = geometry.group_reuse_floor(theta, operand, label, charge_lead=False,
                                               lead=_scheduled_lead(theta, operand, label))
            if slots < floor:
                raise RuntimeError(
                    "VgprGroup%s=%d with VgprAlloc%s=%d: %s:%s rotates through %d buffers but "
                    "needs at least %d to keep one read alive across every pass that consumes "
                    "it -- fewer would alias two live generations.  Raise VgprAlloc%s."
                    % (side, vg, side, depth, operand.name, label, slots, floor, side))
            # NO CEILING.  A VgprAlloc above what the reads reach leaves slots nobody writes,
            # which is WASTED REGISTERS, not a defect -- measured: VA 5..12 on a 4-position ring
            # all verify clean once the dead-slot check stops treating an unused slot as an
            # error.  VA is the caller's to spend; only a ring too SHALLOW aliases live values.


def _check_ring_closes(theta, kernel) -> None:
    """A ring the trip does not close needs the drain-free lowering, and must say so by name.

    The body repeats only every `ring_trip_copies` trips, so a drain would have to enter at the
    right phase -- it would need the trip count modulo that period, which it cannot know.
    `SuppressNoLoadLoop` removes the drain; `Solution.py` clears that flag unless BufferLoad and
    PrefetchGlobalRead==1 and GlobalSplitU<=1, so the value that survives is what matters.
    Asked HERE rather than at the Solution because `VgprGroup > 1` is only a proxy for it: 42 of
    94 VG=2 configurations close every trip and take a drain perfectly well.
    """
    if kernel.get("SuppressNoLoadLoop"):
        return
    from .schedule import BufferDepths
    copies = geometry.ring_trip_copies(theta, BufferDepths(theta, include_shared=False))
    if copies <= 1:
        return
    raise RuntimeError(
        "this ring closes only every %d trips, so it needs SuppressNoLoadLoop, which this "
        "solution cannot keep (that wants BufferLoad, PrefetchGlobalRead=1 and GlobalSplitU<=1): "
        "a drain cannot know the trip count modulo %d, so it would enter at the wrong phase."
        % (copies, copies))


def params_to_theta(p: dict) -> Theta:
    """Translate a TensileLite-param dict into a canonical theta point."""
    copy_depth, read_depth, elem_bytes, fanM, fanN, kernel, substeps, matrix_instruction = _read_params(p)
    splits = _read_splits(kernel)
    nest = _build_ord(fanM, fanN, kernel, substeps, splits)
    movements = _read_movements(kernel)
    configure_read, agent_inputs, assignment = _build_agent_assignment(movements, nest)
    frags = _build_fragments(kernel, matrix_instruction, nest, splits)
    hasMXA, hasMXB, operands = _build_operands(
        configure_read, elem_bytes, frags, movements, kernel, matrix_instruction, nest, splits)
    view, groups = _build_fuse(hasMXA, hasMXB, kernel, operands)
    theta = _build_theta(agent_inputs, copy_depth, read_depth, view, groups, kernel, nest,
                         operands, assignment)
    return theta


def _place_read_offsets(theta, requests):
    for operand in theta.operands:
        depth = requests.get(operand.name, 0)
        if depth <= 0:
            continue
        axis = readahead_level(theta, operand)
        if axis is None:  # no intra-region axis this operand advances along
            continue
        movement = operand.trajectory.fragment_fill
        movement.destination.offsets[axis[0]] = depth


# --- solution --------------------------------------------------------------

def _shape_params(elem_bytes, kernel, mi, wt0, wt1):
    """The kernel's own shape: instruction, depth, fans, vector widths, prefetch depths."""
    return {
        "MatrixInstruction": mi,
        "DepthU": kernel["DepthU"],
        "MIWaveTile": [wt0, wt1],
        "ElemBytes": max(1, elem_bytes),
        "PrefetchGlobalRead": kernel.get("PrefetchGlobalRead", 2),
        # Both spellings: the solution keeps the pair only where the operands differ.
        "PrefetchLocalRead": kernel.get("PrefetchLocalRead", 1),
        "PrefetchLocalReadA": kernel.get("PrefetchLocalReadA", -1),
        "PrefetchLocalReadB": kernel.get("PrefetchLocalReadB", -1),
        "ClusterLocalRead": kernel.get("ClusterLocalRead", 0),
        "ClusterLocalReadA": kernel.get("ClusterLocalReadA", -1),
        "VgprGroupA": kernel.get("VgprGroupA", -1),
        "VgprGroupB": kernel.get("VgprGroupB", -1),
        "VgprAllocA": kernel.get("VgprAllocA", -1),
        "VgprAllocB": kernel.get("VgprAllocB", -1),
        "ClusterLocalReadB": kernel.get("ClusterLocalReadB", -1),
        # The BITMASK and its two bits: HalfPLR=1 is A alone, so the pair is what theta reads.
        "HalfPLR": kernel.get("HalfPLR", 0),
        "HalfPLRA": bool(kernel.get("HalfPLRA", int(kernel.get("HalfPLR", 0) or 0) & 0x1)),
        "HalfPLRB": bool(kernel.get("HalfPLRB", int(kernel.get("HalfPLR", 0) or 0) & 0x2)),
        # DERIVED by Solution, not asked for: HalfPLR and RAP both turn it on.
        "SuppressNoLoadLoop": bool(kernel.get("SuppressNoLoadLoop", False)),
        "GlobalReadVectorWidthA": kernel.get("GlobalReadVectorWidthA", 8) or 8,
        "GlobalReadVectorWidthB": kernel.get("GlobalReadVectorWidthB", 8) or 8,
        "VectorWidthA": kernel.get("VectorWidthA", 2) or 2,
        "VectorWidthB": kernel.get("VectorWidthB", 2) or 2,
        "LocalReadVectorWidth": kernel.get("LocalReadVectorWidth", 4) or 4,
        "DirectToVgprA": bool(kernel.get("DirectToVgprA", False)),
        "DirectToVgprB": bool(kernel.get("DirectToVgprB", False)),
        "WaveSeparateGlobalReadA": kernel.get("WaveSeparateGlobalReadA", 0),
        "WaveSeparateGlobalReadB": kernel.get("WaveSeparateGlobalReadB", 0),
        "InnerUnroll": kernel.get("InnerUnroll", 1) or 1,
        "NumLdsBlk": kernel.get("NumLdsBlk", 2),
        "1LDSBuffer": kernel.get("1LDSBuffer", 0),
    }


def _movement_params(kernel, target):
    """What the target and the layout supply: fusion, agents, splits, the coverage."""
    return {
        "TDMFuse": int(kernel.get("TDMFuse", 0) or 0),
        # A fuse is a wave selection between the movements the TDM makes, so theta needs to know
        # which tensors it moves -- `enableTDM{A,B}` is the resolved form of `TDMInst`.
        "TDMInst": ((0x01 if kernel.get("enableTDMA") else 0)
                    | (0x02 if kernel.get("enableTDMB") else 0)
                    or int(kernel.get("TDMInst", 0) or 0)),
        "NumWaves": kernel.get("NumWaves", 1),
        "TDMSplit": _tdm_split.split_factors(kernel) or [1, 1, 1, 1],
        "TDMSplitWaveRegions": [_tdm_split.wave_region_span(kernel, "A"),
                                _tdm_split.wave_region_span(kernel, "B")],
        "MXBlockA": int(kernel["ProblemType"].get("MXBlockA", 0) or 0),
        "MXBlockB": int(kernel["ProblemType"].get("MXBlockB", 0) or 0),
        "LoopOrder": wmma_loop_order(kernel),
        "ReadVectorElems": _read_vector_elems(kernel, target),
        "ReadQuantum": _read_coverage(kernel, target),
        # The MX half-wave fold is DERIVED from the kernel, not only supplied by a target: one
        # ds_load carries blocks 2g and 2g+1 across the two half-waves (a Phi over a rho span).
        "ReadPhi": dict({tc: f[0] for tc in ("MXSA", "MXSB")
                         for f in (_mxs_fold(kernel, tc),) if f[0] > 1},
                        **((target or {}).get("ReadPhi", {}) or {})),
        "ReadRho": dict({tc: f[1] for tc in ("MXSA", "MXSB")
                         for f in (_mxs_fold(kernel, tc),) if f[1] > 1},
                        **((target or {}).get("ReadRho", {}) or {})),
        "RegisterBudget": (target or {}).get("RegisterBudget"),
    }


def kernel_to_params(kernel, target=None) -> dict:
    """Translate a TensileLite Solution dict into the decoder's TensileLite-param dict."""
    mi = list(kernel["MatrixInstruction"])
    wt0 = kernel.get("MIWaveTileA", kernel.get("MIWaveTile", [1, 1])[0])
    wt1 = kernel.get("MIWaveTileB", kernel.get("MIWaveTile", [1, 1])[1]
                     if isinstance(kernel.get("MIWaveTile"), (list, tuple)) else 1)

    try:
        elem_bytes = int(kernel["ProblemType"]["DataType"].numBytes())
    except Exception:
        elem_bytes = 2

    p = _shape_params(elem_bytes, kernel, mi, wt0, wt1)
    p.update(_movement_params(kernel, target))
    return p


def _mxs_fold(kernel, tc):
    """The MX scale read fold as `(phi, rho)` -- how many tiles one ds_read carries, and over how
    many sub-agents.

    TWO folds compose.  CONTIGUOUS: one wide ds_read covers `VectorWidth{tc}` adjacent tiles, whose
    scales are contiguous in LDS (`tilePerRead` in `leaves.buildMxScaleReadContext`).  DISTRIBUTED:
    the half-wave pair below, block `2g` low and `2g+1` high -- paper 2.2's scattered carrier group.
    """
    contiguous = max(1, int(kernel.get("VectorWidth%s" % tc, 1) or 1))
    distributed = _mxs_tile_span(kernel, tc)
    phi = contiguous * distributed
    return (phi, distributed if distributed > 1 else 0)


def _mxs_tile_span(kernel, tc) -> int:
    """The MX scale half-wave fold: `2` when one ds_load carries two scale blocks, else 1.

    theta's own form of `LocalRead.getMxsTileSpanInfo`.  One instruction holds block `2g` on the
    lower half-wave and `2g+1` on the upper, so it is a DISTRIBUTED carrier group -- Phi over a
    rho sub-agent span -- and modelling it here is what lets GIR carry one Move per instruction.
    """
    tile01 = 0 if tc.endswith("A") else 1
    pt = kernel.get("ProblemType", {}) or {}
    if "MXS" not in tc or not int(pt.get("MXBlock%s" % tc[-1], 0) or 0):
        return 1
    if tuple(kernel.get("ISA") or ()) != (12, 5, 0) \
            or kernel.get("MXScaleFormat") != "InMemorySwizzle":
        return 1
    vw = int(kernel.get("VectorWidth%s" % tc, 0) or 0)
    wave_tile = (kernel.get("MIWaveTile") or [0, 0])[tile01]
    if vw < 1 or not wave_tile:
        return 1
    vectors = int(wave_tile) // vw
    if vectors < 2 or vectors % 2:
        return 1
    inst_t = kernel.get("MatrixInstM" if tile01 == 0 else "MatrixInstN")
    if not inst_t or int(inst_t) != int(kernel.get("WavefrontSize", 0) or 0) // 2:
        return 1
    return 2


def _read_coverage(kernel, target) -> dict:
    return dict((target or {}).get("ReadQuantum", {}) or {})


def _read_vector_elems(kernel, target) -> dict:
    """{operand: elements one shared->register instruction moves per lane} --'s coverage domain."""
    out = dict((target or {}).get("ReadVectorElems", {}) or {})
    pt = kernel.get("ProblemType", {})
    for operand_letter, mx in (("MXSA", "MXBlockA"), ("MXSB", "MXBlockB")):
        if operand_letter in out or not int(pt.get(mx, 0) or 0):
            continue
        if kernel.get("UnrollMajorLDS%s" % operand_letter, kernel.get("UnrollMajorLDSA", 1)):
            out[operand_letter] = int(kernel.get("LocalReadVectorWidthMXS", 0) or 0) or 1
        else:
            out[operand_letter] = 1
    return out



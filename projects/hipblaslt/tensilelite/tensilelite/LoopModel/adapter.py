# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""TensileLite Solution -> theta. The only place TensileLite vocabulary appears."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import gcd as _gcd

from . import traversal as geometry
from ..Components import TDMSplit as _tdm_split
from ..Lowering.lds_geometry import read_fragments
from .traversal import readahead_level
from .ir import Space
from .schedule import select_register_depths
from .theta import (AgentAssignment, AgentAxisAssignment, Fragment, Global,
                    RegionLayout, Shared, Trajectory, Axis, Operand, Theta, group_labels)
from .ir import COPY, READ


# --- fragments -------------------------------------------------------------

#: fixed facts about the target, not about any one kernel
WAVE32 = 32      # gfx1250 wave width
REG_BYTES = 4    # gfx1250 register width in bytes

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


# --- fuse ------------------------------------------------------------------

# `TDMFuse` is a Solution parameter, so its groupings live beside the other TDM descriptor facts
# and are read here, not owned here.
from ..Components.TDMFuse import tdmFusedGroups, tdmWavePartition


# --- loop order ------------------------------------------------------------

# ===========================================================================


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
    # ONE SHARED K WALK.  `K_inner` is the longest run both operands can take without either
    # changing storage region; `K_split` counts those runs, so `K_split * K_inner == K` exactly.
    # `kResA`/`kResB` are REGION INFO on the operand, not axes -- see `Operand.region_div`.
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
        if extents[role] > 1:
            return True
        if role == "K_inner":  # the K chain is live when the shared split is
            return extents["K_split"] > 1
        return extents.get(INNER_TO_SPLIT.get(role, ""), 1) > 1
    # `ord` CARRIES ONLY THE SHARED LINK.  `kResA`/`kResB` are one operand's own residue: putting
    # them on the shared nest asks the other operand to have an opinion about an axis that is not
    # its.  Each side gets its own nest, the shared one with its residue spliced back in.
    inner_ord = [axes[role] for role in roles if _keep(role)]
    if not inner_ord:  # fully degenerate: keep K_inner as the 1 loop
        inner_ord = [axes["K_inner"]]
    iter_mode = Axis("iter")  # extent=0 => is_outer (the reduction chunk / LDS-ring axis)
    axes["iter"] = iter_mode
    ord_ = [iter_mode] + inner_ord
    return ord_, axes, extents


def inner_ord_of(ord_):
    """The inner (intra-chunk) modes of an loop order list -- those with a concrete (>0) extent."""
    return [axis for axis in ord_ if not axis.is_outer]


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
    "LoopOrder": "KMN",
    "PrefetchGlobalReadA": -1, "PrefetchGlobalReadB": -1,
    "PrefetchLocalReadA": -1, "PrefetchLocalReadB": -1,
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
    # --- TDMSplit: [A_MT, B_MT, A_DU, B_DU] -> canonical region extents --------------
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
    order_word = _loop_order_word(kernel["LoopOrder"])
    ord_, canonical_axes, canonical_extents = _canonical_ord(
        substeps, splits, m_inner, n_inner, order_word)

    N_BCAST = {"N_split", "N_inner"}  # A is broadcast over N modes
    M_BCAST = {"M_split", "M_inner"}  # B is broadcast over M modes
    # A REGION AXIS IS THE OPERAND'S OWN: `K_split` is a region for the operand that actually
    # splits K, and merely a loop axis for the one that does not.
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


def _read_width(kernel, tc) -> int:
    """The LDS read width for `tc`.

    A scale's is kernel-wide and an input's per-operand -- `LraTileAssignment` states the same rule
    for the read addresses -- and the final default is the one `_shape_params` applies to this very
    parameter.  One chain, so the count and the addresses cannot disagree about the width.

    A fixture that states no width takes the `4`, which is NOT what the emitter picks; only a
    Solution-supplied width makes the count match the issued reads.
    """
    key = "LocalReadVectorWidthMXS" if "MXS" in tc else "LocalReadVectorWidth%s" % tc
    return int(kernel.get(key) or kernel.get("LocalReadVectorWidth") or 4)


def _mx_block(kernel, tc) -> int:
    """`MXBlock{A,B}`, which a param dict carries at the top level and a kernel under ProblemType."""
    name = "MXBlock%s" % tc[-1]
    return int(kernel.get(name) or (kernel.get("ProblemType") or {}).get(name) or 0)


def _fragment_count(kernel, tc, per_thread, lrvw) -> int:
    """`len(read_fragments(...))` -- the count the EMITTER issues, from the emitter's own function.

    The leaf emits one ds_read per fragment, so asking `read_fragments` is the only way this cannot
    drift from what is emitted; deriving the same number a second way here is what lets the two
    disagree the moment a width clamps."""
    bpe_ds = _elem_bytes(kernel)
    mi_k = int(kernel.get("MatrixInstK") or 0)
    if not (bpe_ds and mi_k and per_thread and lrvw):
        return 0
    try:
        return len(read_fragments(lrvw * bpe_ds / 4.0, bpe_ds, lrvw, per_thread, mi_k))
    except Exception:
        return 0                      # a shape it cannot tile falls back to the outer factor


def _elem_bytes(kernel) -> int:
    """Bytes per element, or 0 when the kernel carries no type object."""
    try:
        return int(kernel["ProblemType"]["DataType"].numBytes())
    except Exception:
        return 0


def _read_instructions(kernel, tc) -> int:
    """Read instructions one fill of this operand's register placement issues.

    Taken from `read_fragments` -- the emitter's own decomposition -- so the count the waits charge
    is the count the leaf issues.  Falls back to the outer factor only for a shape `read_fragments`
    refuses, which is a shape the emitter cannot read either.
    """
    if not tc:
        return 0
    # A scale's width is kernel-wide, an input's is per-operand -- `LraTileAssignment` states the
    # same rule for the read addresses, and a second spelling here is how MXS counted 0.
    per_thread = int(kernel.get("MIInputPerThread%s" % tc, 0) or 0)
    lrvw, key = _read_width(kernel, tc), "read width"
    if per_thread < 1 or lrvw < 1:
        # Solution did not supply it, so derive from the geometry rather than leave it unknown.
        derived = _derive_read_instructions(kernel, kernel.get("MatrixInstruction") or (), tc)
        if derived > 0:
            return derived
        raise RuntimeError(
            "UseLoopModel: cannot count %s's read instructions -- MIInputPerThread%s=%s, %s=%s, "
            "MatrixInstruction=%s.  The completion counter counts INSTRUCTIONS, so a wait derived "
            "without this number is a full drain, not a wait."
            % (tc, tc, per_thread or None, key, lrvw or None, kernel.get("MatrixInstruction")))
    # The EMITTER's decomposition first; the outer factor only where it cannot tile the registers.
    return _fragment_count(kernel, tc, per_thread, lrvw) or max(1, -(-per_thread // lrvw))


def _derive_read_instructions(kernel, mi, tc) -> int:
    """The read-instruction count when no Solution supplied one.

    The kernel path gets `MIInputPerThread{tc}` from Solution; the PARAM path has no Solution to
    ask, so the same formula -- M (or N) x K x B / wavefront, then `// MXBlock * (32 // M)` for a
    scale -- is applied to the 4-item matrix instruction here.  Deriving beats leaving it unknown:
    a count of 0 makes every wait over the operand a full drain.
    """
    if not tc or len(mi) < 4:
        return 0
    wave = int(kernel.get("WavefrontSize") or 32)
    mn = mi[0] if tc.endswith("A") else mi[1]
    per_thread = mn * mi[2] * mi[3] // max(1, wave)
    if "MXS" in tc:
        block = _mx_block(kernel, tc)
        if not block:
            return 0
        # `max(1, ...)` for the same reason `_mx_frag_elems` has it: a data fill smaller than one
        # MX block still carries a scale, so the count floors at one rather than vanishing.
        per_thread = max(1, per_thread // block * max(1, 32 // max(1, mn)))
    lrvw = _read_width(kernel, tc)
    if lrvw < 1 or per_thread < 1:
        return 0
    return max(1, -(-per_thread // lrvw))   # a fill narrower than the width is still ONE read


def _build_fragments(kernel, matrix_instruction, nest, splits):
    """The operands: their paths, fragments, register widths and MX scale rings."""
    # --- register width: DERIVED, not a parameter
    def reg_fragment(bcast, grp_mode, fan_extent, tc=None):
        supplied = (kernel.get("ReadInstructions") or {}).get(tc, 0)
        return Fragment(broadcast_axes=set(bcast), parts=1, labels=group_labels(1),
                        grouping_mode=grp_mode, group_policy={"*": "pipeline"},
                        instructions=int(supplied
                                         or _derive_read_instructions(kernel, matrix_instruction,
                                                                      tc)))

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
        shared = Shared(vector_elements=copy_width, ring_depth=frags[name].lds_buffers,
                        regions=region_layout(parent))
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
        reads = _per_operand(kernel, "PrefetchLocalRead", operand.name, read_depth)
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
                axis_assignments=assignment.axis_assignments + entries,
                roles=assignment.roles)
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
                  per_region_completion=bool(kernel["PerRegionCompletion"]))
    for operand in theta.operands:
        if operand.movements and operand.fragment:
            operand.fragment.grouping_mode = geometry.grouping_mode_name(theta, operand)
    _place_read_offsets(theta, read_requests)
    select_register_depths(theta, kernel.get("RegisterBudget"))
    return theta


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
        "PrefetchLocalRead": kernel.get("PrefetchLocalRead", 1),
        "GlobalReadVectorWidthA": kernel.get("GlobalReadVectorWidthA", 8) or 8,
        "GlobalReadVectorWidthB": kernel.get("GlobalReadVectorWidthB", 8) or 8,
        "VectorWidthA": kernel.get("VectorWidthA", 2) or 2,
        "VectorWidthB": kernel.get("VectorWidthB", 2) or 2,
        "LocalReadVectorWidth": kernel.get("LocalReadVectorWidth", 4) or 4,
        # per-operand and RESOLVED, unlike the kernel-wide one, which is the AUTO -1
        # A scale is asked about only when the problem HAS one; a kernel with no MXBlock has no
        # MXS operand, so there is nothing to count and nothing to refuse.
        "ReadInstructions": {tc: _read_instructions(kernel, tc)
                             for tc in ("A", "B", "MXSA", "MXSB")
                             if tc in ("A", "B")
                             or (kernel.get("ProblemType") or {}).get("MXBlock%s" % tc[-1])},
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
        "LoopOrder": kernel.get("LoopOrder", "KMN"),
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



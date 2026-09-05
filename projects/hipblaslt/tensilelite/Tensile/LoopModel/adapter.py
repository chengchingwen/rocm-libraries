# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""TensileLite Solution -> theta. The only place TensileLite vocabulary appears."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from math import gcd as _gcd

from . import traversal as geometry
from .. import tdm_split as _tdm_split
from .emit import emit_mainloop
from .traversal import readahead_level
from .ir import Space
from .render import grouping_note as _grouping_note, render_geometry, render_stream
from .schedule import (build_S, chunk_crossing_violations, derive_S,
                       prefetch_steps_for)
from .traversal import requested_read_ahead
from .theta import Fragment, Hop, Axis, Operand, Resort, Rho, Theta
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


def _dc_replace(fragment, **kw):
    return dataclasses.replace(fragment, **kw)


# --- fuse ------------------------------------------------------------------

# `TDMFuse` is a TensileLite Solution parameter, so its groupings live beside the other TDM
# descriptor facts and are read here, not owned here -- the scaffold and the cluster-load
# masks need the same table without importing the decoder.
from ..tdm_fuse import (FUSE_GROUPS, FUSE_NAME, FUSE_WAVE_SHARES, EMITTABLE,  # noqa: F401
                        wave_ranges, fuse_groups, needs_mx, crosses_data_and_scale,
                        max_group_size)


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


def _region_labels(n_parts):
    if n_parts == 2:
        return ("lo", "hi")
    return tuple(f"g{i}" for i in range(n_parts))


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
    """How TDMSplit cuts each operand's tile, and how many of those regions one wave covers."""
    aMT: int; bMT: int; aDU: int; bDU: int
    mSplit: int; nSplit: int  # the M/N region extent = that operand's MT split
    kSplit: int; kResA: int; kResB: int  # shared K link (gcd), then each operand's residual
    _waveA: int; _waveB: int  # regions one wave occupies <= the MT split


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
class HopSpec:
    """What the target supplies per hop: vector widths, the movement coverage, and Phi."""
    grvwA: int; grvwB: int; lrvw: int
    read_elems: dict; read_phi: dict; read_quantum: dict; read_rho: dict
    _wave_span: dict


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
    _waveA = max(1, min(aMT, int(_split_wave_regions[0]) or aMT))
    _waveB = max(1, min(bMT, int(_split_wave_regions[1]) or bMT))

    kSplit = _gcd(aDU, bDU)  # the shared link: both operands prefix it
    kResA = aDU // max(1, kSplit)  # A's own residual beyond the shared link
    kResB = bDU // max(1, kSplit)  # B's own residual

    return Splits(aMT, bMT, aDU, bDU, mSplit, nSplit, kSplit, kResA, kResB,
                  _waveA, _waveB)


def _build_ord(fanM, fanN, kernel, substeps, splits):
    """The canonical six-mode loop order for this LoopOrder word, plus each operand's regions."""
    # --- canonical 6-mode loop order, ordered by the LoopOrder word/ ----------------------
    m_inner = max(1, fanM // splits._waveA)
    n_inner = max(1, fanN // splits._waveB)
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


def _read_hops(kernel, splits):
    """The per-hop payload the target supplies: vector widths, the movement coverage, Phi."""
    # --- hops (per-hop delta = retime) ------------------------------------------------
    grvwA, grvwB = kernel["GlobalReadVectorWidthA"], kernel["GlobalReadVectorWidthB"]
    lrvw = kernel["LocalReadVectorWidth"]

    read_elems = dict(kernel.get("ReadVectorElems", {}) or {})

    read_quantum = dict(kernel.get("ReadQuantum", {}) or {})

    _wave_span = {"A": max(1, splits.aMT // max(1, splits._waveA)), "B": max(1, splits.bMT // max(1, splits._waveB))}

    read_phi = dict(kernel.get("ReadPhi", {}) or {})
    read_rho = dict(kernel.get("ReadRho", {}) or {})

    return HopSpec(grvwA, grvwB, lrvw, read_elems, read_phi, read_quantum, read_rho,
                   _wave_span)


def _build_rho(hopspec, nest):
    """rho -- the agent assignment -- and the hops that derive from it."""
    # --- rho: the AGENT ASSIGNMENT, built BEFORE the hops that derive from itR3) ----------
    rho_inputs = {
        "A":    (nest.aRegions, "M_inner", hopspec._wave_span["A"]),
        "MXSA": (nest.aRegions, "M_inner", hopspec._wave_span["A"]),
        "B":    (nest.bRegions, "N_inner", hopspec._wave_span["B"]),
        "MXSB": (nest.bRegions, "N_inner", hopspec._wave_span["B"]),
    }
    _by_mode = {}

    def _resort(axis, level, extent, origin):
        prev = _by_mode.get(axis)
        if prev is not None:
            if (prev.level, prev.extent) != (level, int(extent)):
                raise RuntimeError(
                    "rho: mode %r resorted twice with different answers -- %s/%d from %s vs "
                    "%s/%d from %s. An axis has ONE agent level."
                    % (axis, prev.level, prev.extent, prev.origin, level, extent, origin))
            return
        _by_mode[axis] = Resort(axis=axis, level=level, extent=int(extent), origin=origin)

    for _name, (_regs, _free, _span) in rho_inputs.items():
        if _span > 1:
            for _m in _regs:  # wave: the agents traverse the region axis
                _resort(_m, "wave", _span, (_name, READ))
        _sub = int(hopspec.read_rho.get(_name, 0) or 0)
        if _sub > 1 and _free:  # sub-wave: the distributed carrier coverage
            _resort(_free, "subwave", _sub, (_name, READ))
    rho = Rho(resort=tuple(_by_mode[m] for m in sorted(_by_mode)))

    def _local_read(name, default):
        phi = max(1, int(hopspec.read_phi.get(name, 1) or 1))
        _regs, _free, _ = rho_inputs.get(name, ((), None, 1))
        rho_span = rho.span_over({_free} if _free else set(), level="subwave")
        agent_rel = bool(rho.at("wave", "block")) and bool(
            set(_regs) & {role.axis for role in rho.coarser_than("wave")})
        return Hop(Space.SHARED, Space.REGISTER,
                   vector_elems=int(hopspec.read_elems.get(name, default) or default),
                   phi_width=phi, rho_span=rho_span,
                   quantum=(hopspec.read_quantum.get(name)
                            if hopspec.read_quantum.get(name) is not None
                            else geometry.derive_coverage(
                                Hop(Space.SHARED, Space.REGISTER,
                                    phi_width=phi, rho_span=rho_span))),
                   region_agent_relative=agent_rel)

    def hops(dtv, grvw, split=1, name=None):
        if dtv:  # DirectToVgpr: one per-lane global->register hop
            return [Hop(Space.GLOBAL, Space.REGISTER, vector_elems=grvw)]
        return [Hop(Space.GLOBAL, Space.SHARED, vector_elems=grvw, kind="tdm", split=max(1, split)),
                _local_read(name, hopspec.lrvw)]

    return _local_read, rho_inputs, hops, rho


def _build_fragments(kernel, matrix_instruction, nest, splits):
    """The operands: their paths, fragments, register widths and MX scale rings."""
    # --- register width: DERIVED, not a parameter
    def reg_fragment(bcast, grp_mode, fan_extent):
        grouping_mode = grp_mode if fan_extent > 1 else None
        return Fragment(broadcast_axes=set(bcast), parts=1, labels=_region_labels(1),
                        grouping_mode=grouping_mode, group_policy={"*": "pipeline"})

    fragA = reg_fragment(nest.N_BCAST, "M_inner", nest.mInner)
    fragB = reg_fragment(nest.M_BCAST, "N_inner", nest.nInner)

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


def _build_operands(_local_read, elem_bytes, frags, hops, hopspec, kernel, matrix_instruction,
                    nest, splits):
    """The operands themselves: one path of hops each, carrying its fragment."""
    A = Operand("A", "M_inner", hops(kernel["DirectToVgprA"], hopspec.grvwA, frags["A"].regions, "A"), frags["A"].fragment,
                frag_elems=frags["A"].frag_elems, elem_bytes=elem_bytes,
                lds_buffers=frags["A"].lds_buffers, region_axes=nest.aRegions, region_span=(nest.regionSpan or {}).get("A", {}),
                free_split=max(1, splits.aMT),
                wave_region_span=splits._waveA)
    B = Operand("B", "N_inner", hops(kernel["DirectToVgprB"], hopspec.grvwB, frags["B"].regions, "B"), frags["B"].fragment,
                frag_elems=frags["B"].frag_elems, elem_bytes=elem_bytes,
                lds_buffers=frags["B"].lds_buffers, region_axes=nest.bRegions, region_span=(nest.regionSpan or {}).get("B", {}),
                free_split=max(1, splits.bMT),
                wave_region_span=splits._waveB)

    C_BCAST = {"K_split", "K_inner"}  # reduction axes = C's broadcast
    accElems = _acc_frag_elems(kernel)  # accumulator fragment (per-lane)
    C = Operand("C", "M_inner", [], Fragment(broadcast_axes=set(C_BCAST)),
                frag_elems=accElems, elem_bytes=REG_BYTES, role="output")
    operands = [A, B, C]

    hasMXA, hasMXB = kernel["MXBlockA"] > 0, kernel["MXBlockB"] > 0
    if hasMXA:
        operands.append(Operand("MXSA", "M_inner",
                                [Hop(Space.GLOBAL, Space.SHARED, vector_elems=hopspec.grvwA),
                                 _local_read("MXSA", hopspec.lrvw)],
                                frags.reg_fragment(nest.N_BCAST, "M_inner", nest.mInner),
                                frag_elems=_mx_frag_elems(kernel, matrix_instruction[0], kernel["MXBlockA"]), elem_bytes=1,
                                lds_buffers=frags["MXSA"].lds_buffers, region_axes=nest.aRegions,
                                region_span=(nest.regionSpan or {}).get("A", {}),
                                free_split=max(1, splits.aMT)))  # inherits the parent's, like
    if hasMXB:
        operands.append(Operand("MXSB", "N_inner",  # parent's builder -- see mxsa above
                                [Hop(Space.GLOBAL, Space.SHARED, vector_elems=hopspec.grvwB),
                                 _local_read("MXSB", hopspec.lrvw)],
                                frags.reg_fragment(nest.M_BCAST, "N_inner", nest.nInner),
                                frag_elems=_mx_frag_elems(kernel, matrix_instruction[1], kernel["MXBlockB"]), elem_bytes=1,
                                lds_buffers=frags["MXSB"].lds_buffers, region_axes=nest.bRegions,
                                region_span=(nest.regionSpan or {}).get("B", {}),
                                free_split=max(1, splits.bMT)))  # inherits the parent's, like


    return hasMXA, hasMXB, operands


def _build_fuse(hasMXA, hasMXB, kernel, operands, splitA, splitB):
    """The Phi fusion groups: which operands share one load."""
    # --- Phi fusion ---------------------------------------------------------------------
    fuse = kernel["TDMFuse"]
    tdm_split_on = (splitA > 1 or splitB > 1)  # any TDM region split present
    present = {operand.name for operand in operands}
    if fuse in (2, 3, 4, 5) and not (hasMXA or hasMXB):
        fuse = 1 if tdm_split_on else 0
    groups = [[t for t in group if t in present] for group in FUSE_GROUPS.get(fuse, [])]
    groups = [group for group in groups if len(group) > 1]
    return fuse, groups


def _read_padding(kernel):
    """The LDS padding the target asks for."""
    # --- lds padding ------------------------------------------------------------------
    lds = {}
    for opn, padk in (("A", "LdsPadA"), ("B", "LdsPadB")):
        pad = kernel[padk]
        if pad not in (-1, 0):
            lds[opn] = {"pad": pad, "block": kernel["LdsBlockSizePerPad"]}
    return lds


def _read_off_requests(kernel, operands, outer_level, copy_depth, read_depth):
    """PrefetchGlobalRead becomes off(copy, iter); PrefetchLocalRead is a per-operand request."""
    off_map, read_requests = {}, {}
    for operand in operands:
        copies = _per_operand(kernel, "PrefetchGlobalRead", operand.name, copy_depth)
        reads = _per_operand(kernel, "PrefetchLocalRead", operand.name, read_depth)
        if operand.hops and operand.hops[0].dst == Space.SHARED and copies > 0:
            off_map[(operand.name, operand.hops[0].role, outer_level)] = copies
        if (operand.hops and operand.hops[-1].dst == Space.REGISTER
                and operand.hops[-1].src == Space.SHARED and reads > 0):
            read_requests[operand.name] = reads
    return off_map, read_requests


def _copy_wave_shares(fuse, groups, operands, waves):
    """How many waves cooperate on each operand's global->LDS copy."""
    names = {operand.name for operand in operands}
    share = {}
    for index, group in enumerate(groups):
        ranges = wave_ranges(fuse, index, waves)
        if ranges is None:  # no explicit split: the group divides its waves evenly
            for name in group:
                share[name] = max(1, waves // max(1, len(group)))
            continue
        for member, _first, count in ranges:
            share[group[member]] = max(1, count)
    for name in names:  # an operand in no group keeps every wave
        if not any(name in group for group in groups):
            share[name] = waves
    return {name: count for name, count in share.items() if name in names}


def _with_copy_shares(rho, rho_inputs, shares):
    """Record the copy-side wave partition on rho, one Resort per operand that has an axis."""
    entries = tuple(Resort(axis=rho_inputs[name][1], level="wave", extent=count,
                           origin=(name, "copy"), role="copy")
                    for name, count in sorted(shares.items())
                    if name in rho_inputs and rho_inputs[name][1])
    return Rho(resort=rho.resort + entries, roles=rho.roles) if entries else rho


def _place_offsets(rho_inputs, copy_depth, read_depth, fuse, groups, kernel, lds, nest, operands,
                   rho):
    """The retime matrix `off`, the copy-side wave partition, and the assembled Theta."""
    waves = max(1, int(kernel["NumWaves"]))
    off_map, read_requests = _read_off_requests(kernel, operands, nest.ord_[0].name,
                                                copy_depth, read_depth)
    if waves > 1:
        rho = _with_copy_shares(rho, rho_inputs,
                                _copy_wave_shares(fuse, groups, operands, waves))
    theta = Theta(operands=operands, ord=tuple(nest.ord_), reg_bytes=REG_BYTES, lanes=WAVE32,
                  fuse=fuse, fuse_groups=groups, lds=lds, rho=rho, waves=waves,
                  off_map=off_map, per_region_completion=bool(kernel["PerRegionCompletion"]))
    _place_read_offsets(theta, read_requests)
    theta.S = _build_vgpr_S(theta, kernel.get("RegisterBudget"))
    return theta


def params_to_theta(p: dict) -> Theta:
    """Translate a TensileLite-param dict into a canonical theta point."""
    copy_depth, read_depth, elem_bytes, fanM, fanN, kernel, substeps, matrix_instruction = _read_params(p)
    splits = _read_splits(kernel)
    nest = _build_ord(fanM, fanN, kernel, substeps, splits)
    hopspec = _read_hops(kernel, splits)
    _local_read, rho_inputs, hops, rho = _build_rho(hopspec, nest)
    frags = _build_fragments(kernel, matrix_instruction, nest, splits)
    hasMXA, hasMXB, operands = _build_operands(_local_read, elem_bytes, frags, hops, hopspec, kernel,
                                               matrix_instruction, nest, splits)
    fuse, groups = _build_fuse(hasMXA, hasMXB, kernel, operands, frags["A"].regions,
                               frags["B"].regions)
    lds = _read_padding(kernel)
    theta = _place_offsets(rho_inputs, copy_depth, read_depth, fuse, groups, kernel, lds, nest,
                           operands, rho)
    return theta


def _place_read_offsets(theta, requests):
    for operand in theta.operands:
        depth = requests.get(operand.name, 0)
        if depth <= 0:
            continue
        axis = readahead_level(theta, operand)
        if axis is None:  # no intra-region axis this operand advances along
            continue
        theta.off_map[(operand.name, operand.hops[-1].role, axis[0])] = depth


def _prefetch_served(theta) -> bool:
    """Every pipelined group can carry the read-ahead its off_map asks for."""
    depths = derive_S(theta)
    for operand in theta.operands:
        if operand.is_output or not operand.fragment:
            continue
        want = requested_read_ahead(theta, operand)
        if not want:
            continue
        for group in operand.fragment.groups():
            if operand.fragment.policy_of(group) == "inplace":
                continue  # refilling at the point of use is a declared policy, not a shortfall
            if prefetch_steps_for(theta, operand, group, want, depths) < want:
                return False
    return True


#: Register groupings to try, coarsest first: one whole fan, then splits.  A split raises the
#: reload count the ring sees, which is what lets a deeper read-ahead be assigned at all; the
#: `inplace` tail is the variant that buys registers back when the budget is tight.
_GROUPINGS = ((1, None), (2, ("pipeline", "pipeline")), (2, ("pipeline", "inplace")),
              (4, ("pipeline",) * 4), (4, ("pipeline",) + ("inplace",) * 3),
              (8, ("pipeline",) * 8), (8, ("pipeline",) + ("inplace",) * 7))


class _SBuilder:
    """Builds S: the register grouping and the ring size per group, within the budget.

    theta supplies the facts (loop order, transfer quantum, split extents, the read-ahead its
    off_map asks for); the budget picks among the groupings those facts allow.
    """

    def __init__(self, theta, budget):
        self.theta, self.budget, self.cache = theta, budget, {}
        self.splittable = [op for op in theta.operands
                           if op.hops and op.fragment.grouping_mode is not None
                           and theta.free_extent(op.fragment.grouping_mode) > 1
                           # a microscale ring RIDES WITH ITS PARENT, so it is not ours to choose
                           and not op.name.startswith("MXS")]
        self.original = {op.name: op.fragment for op in self.splittable}

    def admissible(self, operand, share) -> bool:
        """Whether this operand's fan may be cut `share` ways.

        A group holds WHOLE transfers of the grouping axis: one shared->register instruction covers
        `quantum` tiles along it, and a boundary inside one cuts an instruction in half.  A grouping
        axis that is also a storage region is left whole rather than guessing its alignment.
        """
        mode = operand.fragment.grouping_mode
        fan = self.theta.free_extent(mode)
        if share < 1 or fan % share or fan // share < 1:
            return False
        if share == 1:
            return True
        if mode in (getattr(operand, "region_axes", ()) or ()):
            return False
        hop = next((h for h in (operand.hops or ())
                    if not getattr(h, "is_bulk", False)
                    and h.dst == Space.REGISTER and h.src == Space.SHARED), None)
        if hop is None:
            return True
        quantum = int((geometry.transfer_coverage(self.theta, operand, hop) or {}).get(mode, 1) or 1)
        return (fan // share) % max(1, quantum) == 0

    def apply(self, share, policies) -> bool:
        """Give every splittable operand `share` groups on `policies`."""
        for operand in self.splittable:
            if share == 1:
                operand.fragment = self.original[operand.name]
                continue
            if not self.admissible(operand, share):
                return False
            labels = _region_labels(share)
            operand.fragment = _dc_replace(self.original[operand.name], parts=share, labels=labels,
                                           group_policy=dict(zip(labels, policies)))
        return True

    def unserved(self, depths):
        """Why this S fails the requested read-ahead, or None when every group carries it."""
        for operand in self.theta.operands:
            if operand.is_output or not operand.fragment:
                continue
            want = requested_read_ahead(self.theta, operand)
            if not want:
                continue
            for group in operand.fragment.groups():
                if operand.fragment.policy_of(group) == "inplace":
                    continue  # refilling at the point of use is a declared policy
                got = prefetch_steps_for(self.theta, operand, group, want, depths)
                if got < want:
                    return ("ring", operand.name, got, want)
        # A read-ahead crossing into the next chunk is readable only if the copy side staged it.
        crossing = chunk_crossing_violations(self.theta, depths)
        if crossing:
            name, _steps, crossed, offset, need = crossing[0]
            return ("crossing", name, crossed, offset, need)
        return None

    def cost(self, depths) -> int:
        """Registers this S holds -- summed per group, so a ragged partition is visible."""
        return sum(geometry.operand_footprint_regs(self.theta, operand, depths)
                   for operand in self.theta.operands)

    def build(self):
        best, tried = None, []
        for share, policies in _GROUPINGS:
            if not self.apply(share, policies):
                continue
            try:
                depths = derive_S(self.theta)
            except RuntimeError as exc:
                tried.append((share, policies, None, ("derive", exc)))
                continue
            spent, why = self.cost(depths), self.unserved(depths)
            tried.append((share, policies, spent, why))
            if why is None and (not self.budget or spent <= self.budget):
                return depths                     # shallowest grouping that serves and fits
            if why is None:
                best = spent if best is None else min(best, spent)
        self.apply(1, None)
        raise RuntimeError(
            "no register grouping gives S: %s -- tried %s"
            % ("every candidate exceeds the RegisterBudget of %d (cheapest %s)" % (self.budget, best)
               if best is not None else "none serves the requested read-ahead",
               "; ".join(_grouping_note(*note) for note in tried)))


def _build_vgpr_S(theta, budget):
    """S -- theta's INPUT, built once here and handed over, never re-derived downstream."""
    return _SBuilder(theta, budget).build()


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
        "NumWaves": kernel.get("NumWaves", 1),
        "TDMSplit": _tdm_split.split_factors(kernel) or [1, 1, 1, 1],
        "TDMSplitWaveRegions": [_tdm_split.wave_region_span(kernel, "A"),
                                _tdm_split.wave_region_span(kernel, "B")],
        "MXBlockA": int(kernel["ProblemType"].get("MXBlockA", 0) or 0),
        "MXBlockB": int(kernel["ProblemType"].get("MXBlockB", 0) or 0),
        "LoopOrder": kernel.get("LoopOrder", "KMN"),
        "ReadVectorElems": _read_vector_elems(kernel, target),
        "ReadQuantum": _read_quantum(kernel, target),
        "ReadPhi": dict((target or {}).get("ReadPhi", {}) or {}),
        "ReadRho": dict((target or {}).get("ReadRho", {}) or {}),
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


def _read_quantum(kernel, target) -> dict:
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


def kernel_to_ir_text(kernel) -> str:
    try:
        params = kernel_to_params(kernel)
        theta = params_to_theta(params)
        r = emit_mainloop(theta)
    except Exception as e:
        return f"# LoopModel IR unavailable (decode error): {e}\n"

    lines = ["# LoopModel theta-IR dump (observability; not the emitted asm)",
             f"# ledger_empty = {r['ledger_empty']}  ({r['n_obl']} obligations)"]
    if r["undischarged"]:
        for o in r["undischarged"]:
            lines.append(f"#   UNDISCHARGED: {o.kind} {o.producer}->{o.consumer} on {o.counter}")
    lines.append("")
    try:
        lines.append(render_geometry(theta))
        lines.append("")
    except Exception as e:
        lines.append(f"# (geometry render failed: {e})")
    try:
        lines.append(render_stream(r["ir"]))
    except Exception as e:
        lines.append(f"# (IR render failed: {e})")
    return "\n".join(lines) + "\n"

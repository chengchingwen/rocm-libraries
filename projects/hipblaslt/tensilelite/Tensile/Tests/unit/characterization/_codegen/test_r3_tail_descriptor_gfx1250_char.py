# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
"""gfx1250 ULM tail/stagger coverage over every TDMSplit x TDMFuse cell (CPU-only).

Two invariants, asserted on fully generated assembly:

1. **Every descriptor is rewound.** A descriptor that issues its own ``tensor_load_to_lds``
   owns a chunk position, so it owes a tail rewind. Enumerating ``tensorParametersA/B``
   missed mxf8's ``(MXSA,MXSB)`` descriptor entirely.
2. **The WrapU hoist implies the A/B parity pair.** ``WrapUA = parity ? WrapUB : WrapUA`` is
   this descriptor's wrap only when A and B are the parity group; under a grouping that
   crosses data with scale it clobbers the wrap of a descriptor A owns alone.

Driven from the shipping mxf8 config's own fork group, so the cell list cannot drift from what
ships. CPU-only: no GPU required.
"""

import collections
import re

import pytest

import codegen_harness as _ch
import config_harness as _cfgh

pytestmark = pytest.mark.unit

_CFG = "Tensile/Tests/common/gemm/gfx12/loopmodel_mxf8_gfx1250.yaml"

_ISSUER = re.compile(r"tensor_load_to_lds s\[sgprtdm(\w+?)Group0:")
_REWIND = re.compile(r"s_sub_u32 s\[sgprtdm(\w+?)Group0\+2\].*rewind to the chunk")
_HOIST = re.compile(r"hoist: WrapU(\w+) = parity \? WrapU(\w+)")
#: A=even/B=odd is this descriptor's partition only when A and B share it. Elsewhere the split
#: tensor is alone and every wave serves it, so a parity gate leaves half the waves in region 0.
_ABPARITY = re.compile(r"wave parity \(A=even/B=odd\)")
_TAIL = re.compile(r"/\* Tail Loop")
#: `count = 0/1` on the shared descriptor: the per-wave NULL that keeps a group's SHORT member
#: off a region step it does not carry.
_MASK = re.compile(r"s\[sgprtdm\w+Group0\+0\]")

def _groups(fuse):
    """The sets `fuse` names, read from the component so a renumbering cannot strand this test."""
    from Tensile.Components.TDMFuse import TDM_FUSE_GROUPING, TDM_GROUPS
    return TDM_GROUPS[TDM_FUSE_GROUPING[fuse]].groups


def _descriptor(fuse, member):
    """The set OWNER whose SGPRs carry `member` -- every other member RegSets onto it.

    An issuer emits under its own alias name, so names must be resolved to descriptors before
    an issue set and a rewind set can be compared.
    """
    from Tensile.Components.TDMFuse import TDM_DATA_TENSORS
    for g in _groups(fuse):
        if member in g:
            return next((m for m in g if m in TDM_DATA_TENSORS), g[0])
    return member


def _ab_parity(fuse):
    """Does `fuse` seat A and B on ONE set, so wave parity selects between them?"""
    return any(len(g) == 2 and set(g) == {"A", "B"} for g in _groups(fuse))


def _mixed_group(fuse, owner, sa, sb):
    """Does the set OWNED by `owner` hold both a split and an unsplit member?

    Asked per descriptor, not per kernel: under `B_MX` the (B,MXSA,MXSB) set is mixed while A
    rides its own, so a kernel-wide answer demands a NULL on a descriptor with nothing to null.
    """
    factor = {"A": 2 if sa else 1, "B": 2 if sb else 1, "MXSA": 1, "MXSB": 1}
    for g in _groups(fuse):
        if _descriptor(fuse, g[0]) == owner:
            return len({factor[m] > 1 for m in g}) > 1
    return False


def _fuse_group_config(tmp_path):
    """The shipping config trimmed to the fork group that sweeps every TDMFuse value.

    `_solutions_from_config_unguarded` expands only the first group; the crossing groupings
    live in a later one.
    """
    lines = open(_CFG).read().split("\n")
    head = next(i for i, l in enumerate(lines) if l.startswith("BenchmarkProblems:"))
    starts = [i for i, l in enumerate(lines)
              if i > head and (re.match(r"^  - ", l) or re.match(r"^  -$", l))]
    want = next((i for i, l in enumerate(lines) if "TDMFuse:" in l and "2" in l and "3" in l), None)
    if want is None:
        pytest.skip("no fork group sweeps several TDMFuse values")
    lo = max(s for s in starts if s < want)
    hi = min([s for s in starts if s > lo] + [len(lines)])
    out = tmp_path / "mxf8_fuse.yaml"
    out.write_text("\n".join(lines[:head + 1] + lines[lo:hi]))
    return str(out)


def _cells(cfg):
    """One kernel per distinct (TDMFuse, TDMSplitA, TDMSplitB), with its emitted assembly."""
    from Tensile.Common.Types import DebugConfig
    from Tensile.KernelWriterAssembly import KernelWriterAssembly
    from Tensile.TensileCreateLibrary.Run import generateKernelObjectsFromSolutions

    assembler, iim = _cfgh._toolchain_for("gfx1250")
    with _cfgh._isolated_globals_with_isa(iim):
        sols = _cfgh._solutions_from_config_unguarded(cfg, assembler, iim, limit_solutions=4000)
        pool = collections.OrderedDict()
        for k in generateKernelObjectsFromSolutions(sols):
            if not k.get("UseLoopModel"):
                continue
            pool.setdefault((int(k.get("TDMFuse", 0)), int(k.get("TDMSplitA", 0)),
                             int(k.get("TDMSplitB", 0))), k)
        kwa = KernelWriterAssembly(assembler, DebugConfig())
        sel = list(pool.items())
        if sel and not _ch._WARMED:
            _cfgh._emit_one(kwa, sel[0][1], False, False)   # scheduler warm-up
            _ch._WARMED = True
        return [(key, _cfgh._emit_one(kwa, k, False, False)) for key, k in sel]


@pytest.mark.gfx1250
def test_every_tdmsplit_tdmfuse_cell_rewinds_and_wraps_per_descriptor(tmp_path):
    cells = _cells(_fuse_group_config(tmp_path))
    assert cells, "no UseLoopModel kernels in the fork group"
    assert len({k[0] for k, _ in cells}) > 1, "fork group did not sweep TDMFuse"

    problems = []
    for (fuse, sa, sb), (base, src, err) in cells:
        issuers = {_descriptor(fuse, m) for m in _ISSUER.findall(src)}
        rewound = {_descriptor(fuse, m) for m in _REWIND.findall(src)}
        why = []
        if err != 0:
            why.append("emit err=%d" % err)
        if not issuers:
            why.append("no descriptor issues a load")
        if issuers != rewound:
            why.append("issue=%s rewind=%s" % (sorted(issuers), sorted(rewound)))
        if rewound:
            guard = src.find("skip peel rewind when the main loop never ran")
            branch = src.find("K < DepthU: peel issued no descriptor advance", guard)
            first_rewind = min((m.start() for m in _REWIND.finditer(src)), default=-1)
            skip_label = src.find("label_SkipTdmTailUndoPeel", first_rewind)
            if not (0 <= guard < branch < first_rewind < skip_label):
                why.append("peel rewind is not guarded on OrigLoopCounter != 0")
        # The hoist writes A/B parity into WrapUA; only the (A,B) parity set may use it.
        if _HOIST.search(src) and not _ab_parity(fuse):
            why.append("A/B WrapU parity hoist under a crossing grouping")
        # Defect B: a crossing grouping must gate NOTHING on A/B parity -- the split tensor is
        # alone on its descriptor, so the step and the dim1 extent are wave-uniform.
        if not _ab_parity(fuse) and _ABPARITY.search(src):
            why.append("%d A/B-parity gate(s) under a crossing grouping"
                       % len(_ABPARITY.findall(src)))
        if "overflowed resources" in src:
            why.append("SGPR pool overflowed")
        # 3. A group whose members own DIFFERENT region counts must null the short ones across
        # each region step they do not carry.  GIR brackets its copies; the tail runs the
        # scaffold's own region loop, which had no such bracket at all.
        cut = _TAIL.search(src)
        if cut:
            tail = src[cut.start():]
            counts = collections.Counter(_descriptor(fuse, m) for m in _ISSUER.findall(tail))
            walked = [d for d, n in counts.items() if n > 1 and _mixed_group(fuse, d, sa, sb)]
            if walked and not _MASK.search(tail):
                why.append("tail walks %s with an unsplit group member and no descriptor NULL"
                           % sorted(walked))
        if why:
            problems.append("TDMFuse=%d TDMSplitA=%d TDMSplitB=%d (%s): %s"
                            % (fuse, sa, sb, base, "; ".join(why)))
    assert not problems, "%d/%d cells break the tail/stagger contract:\n  %s" % (
        len(problems), len(cells), "\n  ".join(problems))

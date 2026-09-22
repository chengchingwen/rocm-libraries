# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
The barrier-uniformity check over the EMITTED instruction stream.

`gir/analyses/barrier_uniformity.py` asks the same question of the GIR CFG; this one asks it of
the assembly, where the wave-discriminating branches the scaffold owns actually are.  A barrier
inside one of those arms is reached by some waves and not others.
"""

from __future__ import annotations

import atexit
import os
import re


# Sweep instrumentation: a check that never fires has to be able to SHOW it ran over real
# structure.  `TENSILE_BARRIER_UNIFORMITY_REPORT=1` prints the totals at exit; off, this is three
_STATS = {"kernels": 0, "regions": 0, "owned": 0, "inside": 0}
_REPORT = bool(os.environ.get("TENSILE_BARRIER_UNIFORMITY_REPORT"))
if _REPORT:
    atexit.register(lambda: print("[barrier-uniformity] kernels=%(kernels)d "
                                  "wave-discriminating-arms=%(regions)d proc-scoped-selectors="
                                  "%(owned)d inside-an-arm=%(inside)d" % _STATS, flush=True))


# Registers that NAME AN WAVE.  `sgprWaveIdx` is the wave index within the workgroup (what the
# TDM wave-separation branches test); `vgprSerial` is the flat thread id, so anything compared
AGENT_ID_REGS = re.compile(r"\bsgprWaveIdx\b|\bvgprSerial\b|\bsgprWaveId\b")

# Producers of SCC / VCC.  A branch's condition comes from the nearest preceding one of these.
_SETS_COND = re.compile(r"^\s*(s_bitcmp[01]_b(?:32|64)|s_cmp_\w+|s_cmpk_\w+|s_and_b(?:32|64)"
                        r"|s_andn2_b(?:32|64)|s_or_b(?:32|64)|s_test_\w+|v_cmp\w*)\b")
_VECTOR_CMP = re.compile(r"^\s*v_cmp\w*\b")

# How far back to look for the condition's producer.  The emitters put it 1-2 instructions before
# the branch; 12 is slack, not a tuning knob.
_LOOKBACK = 12


def _text(item):
    """The instruction text with its comment stripped -- the operands, and nothing a comment says."""
    try:
        s = str(item)
    except Exception:
        return ""
    return s.split("//")[0].rstrip()


def _is_label(item):
    return type(item).__name__ == "Label"


def _branch_target(item):
    """The label a conditional branch jumps to, or None if `item` is not a conditional branch.

    Matched on the CLASS NAME rather than by importing every branch type, because the set grows
    (`SCBranchExecZ`, `SCBranchSCC1`, ...) and a missing import would silently make this check skip a
    region -- the one failure axis a safety check must not have.
    """
    n = type(item).__name__
    if not (n.startswith("SCBranch") or n.startswith("SCLongBranch")):
        return None
    return getattr(item, "labelName", None)


def _condition_kind(items, i):
    """Is the condition of the branch at `items[i]` wave-discriminating?  Returns a reason
    string, or None when the branch is wave-uniform as far as this check can tell."""
    n = type(items[i]).__name__
    if "Exec" in n:
        return "branch on EXEC (lane-discriminating)"
    for j in range(i - 1, max(-1, i - 1 - _LOOKBACK), -1):
        t = _text(items[j])
        if not t.strip():
            continue
        if _SETS_COND.match(t):
            if _VECTOR_CMP.match(t):
                return "branch on VCC from a vector compare: " + t.strip()
            if AGENT_ID_REGS.search(t):
                return "condition reads an wave id: " + t.strip()
            return None                       # a uniform scalar compare (trip count, GSU, sizes)
    return None                               # no producer found within the window


def agent_discriminating_regions(items):
    """`[(start, end, target, reason)]` -- the half-open spans only SOME waves execute.

    A span is opened by an wave-discriminating conditional branch and closed by the FIRST later
    occurrence of its target label.  A backward branch (a loop back-edge) opens nothing: every
    wave that runs the body runs all of it."""
    label_at = {}
    for i, it in enumerate(items):
        if _is_label(it):
            label_at.setdefault(it.getLabelName(), i)
    out = []
    for i, it in enumerate(items):
        tgt = _branch_target(it)
        if tgt is None:
            continue
        why = _condition_kind(items, i)
        if why is None:
            continue
        # branch.labelName is the bare name; Label.getLabelName() carries the 'label_' prefix.
        end = label_at.get(tgt, label_at.get("label_" + tgt))
        if end is None or end <= i:
            continue
        out.append((i, end, tgt, why))
    return out


def check_barrier_uniformity(rootModule, ownedBarriers, kernelName=""):
    """Verify no barrier in `ownedBarriers` was emitted inside an wave-discriminating skip region.

    `ownedBarriers` is matched by OBJECT IDENTITY (the same currency `_stripBarriers(keep=...)` uses,
    and for the same reason: rocisa nodes reject an added attribute, so identity is the only honest
    signal that a barrier is ours).  Returns the findings; raises if there are any.
    """
    items = list(rootModule.flatitems())
    regions = agent_discriminating_regions(items)
    owned = {id(b) for b in (ownedBarriers or ())}
    at = {}
    for i, it in enumerate(items):
        if id(it) in owned:
            at.setdefault(id(it), i)
    _STATS["kernels"] += 1
    _STATS["regions"] += len(regions)
    _STATS["owned"] += len(at)

    findings = []
    for bid, pos in at.items():
        for (s, e, tgt, why) in regions:
            if s < pos < e:
                findings.append("%s: proc-scoped selector at item %d lies inside the arm "
                                "%d..%d -> %s (%s)" % (kernelName or "<kernel>", pos, s, e, tgt, why))
    _STATS["inside"] += len(findings)
    if findings:
        raise RuntimeError(
            "uniform placement : %d GIR fence(s) emitted where only some waves "
            "arrive -- the workgroup rendezvous HANGS.  %s" % (len(findings), "; ".join(findings[:4])))
    return findings

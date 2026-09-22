# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
G-UNIFORM (paper the design / placement premise) — the liveness check the empty-ledger gate
does not subsume, on both layers it has to be asked on:

  * `gir/analyses/barrier_uniformity.py`  — the design's tree-structural test over GIR's region tree
  * `Lowering/tool/barrier_uniformity_asm.py`       — the same question over the EMITTED stream, which is the
                                           only place the scaffold's wave-parity arms exist

Every test here builds the failing shape explicitly, because the passing case is the one the
codebase already produces and a check that has never been seen to fire is not a check.
"""

import pytest

from Tensile.Lowering.gir.nodes import (Program, Block, Mark, Pred, Bound, Goto, CondGoto,
                                        CondChain)
from Tensile.Lowering.gir.analyses.barrier_uniformity import BarrierUniformity
from Tensile.Lowering.gir.analysis import AnalysisManager


def _fence(scope="block"):
    return Mark("fence", {"buffers": ("A",), "tokens": (0,), "scope": scope,
                          "kinds": ("WAR",), "edges": 1})


def _two_block(role_steady="all", role_pro="all", pred=None, meta=None):
    """prologue -> steady, with the fence in `steady`."""
    prog = Program(entry="prologue")
    term = (CondGoto(pred, "steady", "steady") if pred is not None else Goto("steady"))
    prog.add_block(Block(phase="prologue", succs=("steady",), term=term, role=role_pro))
    prog.add_block(Block(phase="steady", preds=("prologue",), body=[_fence()],
                         term=Goto("end"), role=role_steady))
    prog.meta.update(meta or {})
    return prog


def _run(prog):
    return AnalysisManager().get(BarrierUniformity(), prog)


# ---------------------------------------------------------------------------
# GIR layer
# ---------------------------------------------------------------------------
def test_uniformly_placed_fence_is_accepted():
    """The shape the lowering produces today: every block role='all', every guard a trip-count
    predicate.  Zero violations, and the check must not invent one."""
    assert _run(_two_block()) == []


def test_fence_in_a_role_restricted_block_is_a_violation():
    """ρ: only some waves run a role-restricted block, so a WORKGROUP rendezvous inside it
    is a hang.  This is the shape wave-specialization will introduce."""
    v = _run(_two_block(role_steady="producer"))
    assert len(v) == 1 and v[0].reason == "role", v
    assert "role='producer'" in repr(v[0])


def test_a_role_restricted_ANCESTOR_is_also_a_violation():
    """Domination, not containment: a restricted block on the way to the fence keeps the excluded
    agents from ever reaching it, even though the fence's own block admits everyone."""
    v = _run(_two_block(role_pro="consumer"))
    assert len(v) == 1 and v[0].reason == "role" and "prologue" in v[0].where, v


def test_an_agent_varying_guard_is_a_violation():
    """A `Cond` arm that discriminates among agents.  GIR cannot spell one today, so the
    program must DECLARE the symbol as agent-varying — that declaration is the extension point,
    and this test is what proves the check consumes it."""
    prog = _two_block(pred=Pred("waveId", "==", Bound(const=0)),
                      meta={"agent_varying_symbols": ("waveId",)})
    v = _run(prog)
    assert len(v) == 1 and v[0].reason == "guard", v
    assert "waveId" in v[0].where


def test_a_trip_count_guard_on_the_same_shape_is_NOT_a_violation():
    """The control-flow GIR really has: a comparison on the trip symbol.  All waves of a workgroup
    share it, so it splits nobody — the check must not fire on the loop's own guard."""
    prog = _two_block(pred=Pred("iter", "<", Bound(var="T")),
                      meta={"agent_varying_symbols": ("waveId",)})
    assert _run(prog) == []


def test_a_multi_exit_header_arm_is_inspected_too():
    """A `CondChain` header (G1) tests several predicates; every arm is a place an agent could be
    steered away, so all of them are checked, not just the first."""
    prog = Program(entry="head")
    prog.add_block(Block(phase="head", succs=("steady", "drain0"),
                         term=CondChain(arms=((Pred("c", "<=", Bound(const=1)), "drain0"),
                                              (Pred("laneParity", "==", Bound(const=0)), "steady")),
                                        default="steady")))
    prog.add_block(Block(phase="steady", preds=("head",), body=[_fence()], term=Goto("end")))
    prog.add_block(Block(phase="drain0", preds=("head",), term=Goto("end")))
    prog.meta["agent_varying_symbols"] = ("laneParity",)
    v = _run(prog)
    assert len(v) == 1 and v[0].reason == "guard" and "laneParity" in v[0].where, v


def test_a_non_proc_scope_selector_is_not_this_checks_business():
    """A wave-scoped discharge is carried by program order on one agent — there is no rendezvous
    and nothing to be uniform about."""
    prog = _two_block(role_steady="producer")
    prog.blocks["steady"].body[0].at["scope"] = "wave"
    assert _run(prog) == []


def test_G_UNIFORM_is_wired_into_verify_gir():
    """The analysis is only worth having if the verifier runs it — the defect being fixed is
    precisely that `scope` reached nothing but a comment string."""
    from Tensile.Lowering.gir.verify import _check_barrier_uniformity
    prog = _two_block(role_steady="producer")
    with pytest.raises(RuntimeError, match="G-UNIFORM"):
        _check_barrier_uniformity(prog, AnalysisManager())


# ---------------------------------------------------------------------------
# Emission layer — the scaffold's wave-parity arms, which GIR cannot see
# ---------------------------------------------------------------------------
rocisa = pytest.importorskip("rocisa")


def _stream(inner_before_barrier):
    """A module shaped like `_applyStaggerTDM`: parity test, skip branch, body, join label."""
    from rocisa.code import Module, Label
    from rocisa.instruction import SBitcmp1B32, SCBranchSCC1, SBarrier, SCmpEQU32, SAddU32
    from rocisa.container import sgpr
    m = Module("t")
    m.add(SAddU32(dst=sgpr("Tmp"), src0=sgpr("Tmp"), src1=1, comment="filler"))
    bar_before = SBarrier()
    if not inner_before_barrier:
        m.add(bar_before)
    m.add(SBitcmp1B32(src0=sgpr("WaveIdx"), src1=0, comment="check wave parity"))
    m.add(SCBranchSCC1(labelName="SkipStaggerA", comment="skip: odd waves handle B"))
    bar_inside = SBarrier()
    if inner_before_barrier:
        m.add(bar_inside)
    m.add(SAddU32(dst=sgpr("tdmAGroup0+2"), src0=sgpr("tdmAGroup0+2"), src1=sgpr("Tmp"),
                  comment="TDM addr += stagger offset (lo)"))
    m.add(Label(label="SkipStaggerA", comment=""))
    return m, (bar_inside if inner_before_barrier else bar_before)


def test_emission_check_catches_a_selector_inside_a_wave_parity_arm():
    """THE deadlock L733 names, in the exact form this backend can produce it."""
    from Tensile.Lowering.tool.barrier_uniformity_asm import check_barrier_uniformity
    m, bar = _stream(inner_before_barrier=True)
    with pytest.raises(RuntimeError, match="HANGS"):
        check_barrier_uniformity(m, [bar], kernelName="probe")


def test_emission_check_accepts_the_same_selector_outside_the_arm():
    """Same module, barrier hoisted above the parity test — the placement the emitters produce."""
    from Tensile.Lowering.tool.barrier_uniformity_asm import check_barrier_uniformity
    m, bar = _stream(inner_before_barrier=False)
    assert check_barrier_uniformity(m, [bar], kernelName="probe") == []


def test_a_trip_count_arm_does_not_count_as_agent_discriminating():
    """A barrier inside a GSU / size guard is fine: every wave evaluates the same scalar.  Without
    this the check would fire on nearly every fence in the mainloop."""
    from rocisa.code import Module, Label
    from rocisa.instruction import SCmpEQU32, SCBranchSCC1, SBarrier
    from rocisa.container import sgpr
    from Tensile.Lowering.tool.barrier_uniformity_asm import check_barrier_uniformity
    m = Module("t")
    m.add(SCmpEQU32(src0=sgpr("GSU"), src1=1, comment="GSU == 1?"))
    m.add(SCBranchSCC1(labelName="GSU_1", comment="branch if GSU == 1"))
    bar = SBarrier()
    m.add(bar)
    m.add(Label(label="GSU_1", comment=""))
    assert check_barrier_uniformity(m, [bar]) == []


def test_a_backward_branch_opens_no_region():
    """A loop back-edge is not a skip: every agent that runs the body runs all of it.  Treating it
    as a region would put every steady-body fence 'inside an arm'."""
    from rocisa.code import Module, Label
    from rocisa.instruction import SBitcmp1B32, SCBranchSCC1, SBarrier
    from rocisa.container import sgpr
    from Tensile.Lowering.tool.barrier_uniformity_asm import agent_discriminating_regions
    m = Module("t")
    m.add(Label(label="Top", comment=""))
    m.add(SBarrier())
    m.add(SBitcmp1B32(src0=sgpr("WaveIdx"), src1=0, comment="check wave parity"))
    m.add(SCBranchSCC1(labelName="Top", comment="back edge"))
    assert agent_discriminating_regions(list(m.flatitems())) == []


def test_only_GIR_OWNED_barriers_are_judged():
    """A scaffold barrier inside a scaffold arm is the scaffold's business — it may well be
    intentional.  Identity is the only signal that a barrier is ours (rocisa nodes take no added
    attribute), and it is the same currency `_stripBarriers(keep=...)` already uses."""
    from Tensile.Lowering.tool.barrier_uniformity_asm import check_barrier_uniformity
    m, _bar = _stream(inner_before_barrier=True)
    assert check_barrier_uniformity(m, []) == []

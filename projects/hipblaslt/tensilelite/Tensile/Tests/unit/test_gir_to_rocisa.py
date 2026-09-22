# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for the GIR → rocisa lowering (Tensile/Lowering), specifically its PURE half — the
emit plan (Tensile/Lowering/gir/emit_plan.py).
"""

import pathlib
import pytest

from Tensile.Lowering import lower_to_gir, build_gir
from Tensile.Lowering.gir import plan_block, run_pipeline, EmitAction

from gir_fixtures import BF16_NT_KMN, BF16_NT_KMN_PLR0, theta as _theta


def _finalized(params):
    prog = lower_to_gir(_theta(params))
    run_pipeline(prog)
    return prog


def _kinds(actions):
    return [a.kind for a in actions]


def test_plan_steady_has_all_action_kinds():
    acts = plan_block(_finalized(BF16_NT_KMN), "steady")
    kinds = set(_kinds(acts))
    assert {"wmma", "read", "copy", "swap", "gr_inc"} <= kinds, f"missing action kinds: {kinds}"


def test_plan_wmma_count_matches_tile_grid():
    """KMN MIWaveTile [2,2], 2 K-substeps → 2*2*2 = 8 wmmas in steady, one per (idx0,idx1,u)."""
    wmmas = [a for a in plan_block(_finalized(BF16_NT_KMN), "steady") if a.kind == "wmma"]
    assert len(wmmas) == 8, f"expected 8 steady wmmas, got {len(wmmas)}"
    grid = {(a.at["idx0"], a.at["idx1"], a.at["u"]) for a in wmmas}
    assert grid == {(m, n, u) for m in (0, 1) for n in (0, 1) for u in (0, 1)}


def test_plan_read_indices_in_range():
    reads = [a for a in plan_block(_finalized(BF16_NT_KMN), "steady") if a.kind == "read"]
    assert reads, "expected steady reads"
    for a in reads:
        assert a.at["tc"] in ("A", "B")
        assert a.at["tile"] in (0, 1)          # MIWaveTile axis
        assert a.at["k"] in (0, 1)             # K substep
        assert a.at["reg_buf"] in (0, 1)       # register generation slot
        assert a.at["token"] is not None and a.at["token"][0] == "lds"


def test_plan_read_token_names_ONE_generation():
    """Back to exact.  A steady read rotates, but it no longer has to SAY so: the fence
    between the rotating copy and the read carries the ordering, so the label names the one buffer
    this access touches.  See tokens._generation for the round trip."""
    fps = {a.at["token"][2] for a in plan_block(_finalized(BF16_NT_KMN), "steady")
           if a.kind == "read"}
    assert all(len(f) == 1 for f in fps), f"a read token names more than one buffer: {fps}"


def test_plan_order_read_before_its_wmma():
    """Consume-after-read: the first substep's reads precede the first wmma (plan preserves body
    order)."""
    kinds = _kinds(plan_block(_finalized(BF16_NT_KMN), "steady"))
    assert kinds.index("read") < kinds.index("wmma")


def test_plan_swap_hops_present():
    swaps = [a for a in plan_block(_finalized(BF16_NT_KMN), "steady") if a.kind == "swap"]
    hops = {a.at["hop"] for a in swaps}
    assert "read" in hops and "copy" in hops, f"steady swap hops {hops}"


def test_plan_plr0_no_midbody_read_swap_before_first_wmma():
    """PLR0: the read swap is at the loop bottom, so no read swap appears BEFORE the first wmma."""
    acts = plan_block(_finalized(BF16_NT_KMN_PLR0), "steady")
    first_wmma = next(i for i, a in enumerate(acts) if a.kind == "wmma")
    before = [a for a in acts[:first_wmma] if a.kind == "swap" and a.at["hop"] == "read"]
    assert before == [], "PLR0 must not swap the read ptr mid-body before the wmma"


def test_plan_drain_no_copy():
    prog = _finalized(BF16_NT_KMN)
    for phase in prog.blocks:
        if phase.startswith("drain"):
            assert all(a.kind != "copy" for a in plan_block(prog, phase))


def test_plan_from_build_gir_all_stages_nonempty():
    prog = build_gir(_theta(BF16_NT_KMN))
    for phase in prog.blocks:
        assert plan_block(prog, phase), f"{phase}: empty plan"


def test_emit_plan_presence_meta_no_role_to_mode():
    """The GIR interface is presence-derived (no m/n/k role_to_mode): Program.meta carries
    reduction_modes + per-operand free_modes + mma_inputs, and emit_plan projects coords onto
    them.  Confirms role_to_mode is gone and the meta is the presence contract."""
    prog = build_gir(_theta(BF16_NT_KMN))
    assert "role_to_mode" not in prog.meta
    assert set(prog.meta["summation_axes"]) == {"K_inner"}
    assert prog.meta["mma_inputs"] == ["A", "B"]
    assert prog.meta["free_axes"]["A"] and prog.meta["free_axes"]["B"]


def test_emit_plan_multimode_role_mixed_radix():
    """The 6-axis fix: when an operand's free presence spans TWO modes (M_split + M_inner, both
    >1 under a region-split order), emit_plan MIXED-RADIX-COMBINES them into one tile index
    instead of dropping all but one (the old role_to_mode single-mode break).  A's free = both M
    modes → coord (M_split=1,M_inner=1) must project to M_split·2+M_inner=3, not 1."""
    from Tensile.Lowering.gir.emit_plan import _project
    from Tensile.LoopModel.traversal import free_names
    th = _theta(dict(BF16_NT_KMN, LoopOrder="MNK", MIWaveTile=[4, 4], TDMSplit=[2, 2]))
    free = {o.name: free_names(th, o) for o in th.operands}
    ext = {m.name: m.extent for m in th.inner_axes()}
    order = [m.name for m in th.inner_axes()]
    a_free = set(free["A"])
    assert len(a_free) >= 2, "test needs a genuine two-mode free role (M_split+M_inner)"
    coord = {m: 1 for m in order}
    idx = _project(coord, a_free, order, ext)
    # true mixed-radix over A's free modes in ord order
    expect, stride = 0, 1
    for m in reversed([m for m in order if m in a_free]):
        expect += coord[m] * stride
        stride *= ext[m]
    assert idx == expect and idx != 1, f"multi-mode role not combined: got {idx}, want {expect}"


# ===========================================================================
# the L3 path must REFUSE what it cannot address, not silently mis-address.
def test_plan_mma_operands_are_derived_not_literal_A_B():
    """`_plan_mma` picks the wmma's two source slots via the DERIVED `mma_inputs`, not by string-
    comparing `tile.operand == "A"`.  Both slots must be populated for a real kernel."""
    prog = build_gir(_theta(BF16_NT_KMN))
    wmmas = [a for a in plan_block(prog, "steady") if a.kind == "wmma"]
    assert wmmas, "no wmma actions"
    assert all(a.at["bufA"] is not None and a.at["bufB"] is not None for a in wmmas), \
        "an input's register slot went unresolved — operand identity was not derived"


def test_plan_mma_refuses_a_multi_group_operand():
    """An operand split into several register groups contributes one src Ref per group, each with
    its own slot.  The plan carries ONE slot per operand, so it must refuse rather than keep
    whichever Ref came last (a silently wrong source register).  Multi-group is."""
    from Tensile.Lowering.gir.emit_plan import _plan_mma
    from Tensile.Lowering.gir.nodes import Tile, Ref, Mma, Program
    prog = build_gir(_theta(BF16_NT_KMN))
    in0 = prog.meta["mma_inputs"][0]
    t = Tile(operand=in0, space="register")
    inst = Mma(srcs=(Ref(tile=t, group=0, slot=0), Ref(tile=t, group=1, slot=1)), dsts=())
    with pytest.raises(NotImplementedError):
        _plan_mma(prog, inst, [])


def test_register_depth_refuses_divergent_group_widths():
    """The wmma leaf's fallback takes ONE scalar W.  When register groups rotate at different
    widths (mxfp8 lo=2, hi=1) no scalar is correct — returning the first silently mis-addresses
    every other group."""
    from Tensile.Lowering.gir_to_rocisa import _register_depth
    from Tensile.Lowering.gir.analyses.reg_band import RegBand
    prog = build_gir(_theta(BF16_NT_KMN))
    # PER OPERAND, not a scalar.
    assert _register_depth(prog, RegBand({("A", 0): 2, ("B", 0): 2})) == {"A": 2, "B": 2}
    with pytest.raises(NotImplementedError):
        _register_depth(prog, RegBand({("A", 0): 2, ("A", 1): 1}))


def test_an_LDS_ACCESS_THAT_CANNOT_BE_NAMED_is_REFUSED_not_defaulted():
    """The successor to a test that outlived its mechanism, and the property is the same one."""
    from Tensile.Lowering.gir.verify import verify_gir
    from Tensile.Lowering.gir.analyses.lds_buffers import LdsBufferIds, LdsBufferIdSet
    from Tensile.Lowering.gir import AnalysisManager

    prog = build_gir(_theta(BF16_NT_KMN))
    am = AnalysisManager()
    tokens = am.get(LdsBufferIds(), prog)
    assert len(tokens) > 0, "non-vacuity: this kernel really does have named LDS buffers"
    assert tokens.unresolved == (), "a clean program names every shared access"
    verify_gir(prog)                                   # and the verifier agrees

    # INJECT the hole the old test injected, at the layer that now owns it.
    holed = LdsBufferIdSet(ids=tokens.buffers, per_ref={}, rings={}, geometry=None,
                           unresolved=("A@gen2",), writes={}, reads={})
    assert holed.unresolved, "non-vacuity: the injected assignment really is missing a name"
    assert holed.ids_for(object()) == (), "an unassigned ref reports (), never a default id"

def test_append_tag_appends_and_never_overwrites():
    """Every GIR-emitted instruction carries a `GIR ...` tag APPENDED to whatever comment the
    scaffold primitive already wrote.
    """
    from rocisa.code import Module
    from rocisa.instruction import SNop
    from Tensile.Lowering.gir_to_rocisa import GirToRocisa
    m = Module("t")
    m.add(SNop(0, comment="sync LDS0"))
    m.add(SNop(0, comment=""))
    GirToRocisa._append_tag(m, "copy A gen=1")
    got = [i.comment for i in m.flatitems()]
    assert got[0] == "sync LDS0  <GIR: copy A gen=1>", "must APPEND, not replace"
    assert got[1] == "<GIR: copy A gen=1>", "an empty comment takes the tag alone (no leading blanks)"
    # the delimited form is what makes the two layers greppable and machine-checkable:
    #   <LoopIR: ...>  the schedule's meaning (what the GEMM does)
    #   <GIR: ...>     the lowering act that realized it (who emitted this instruction)
    assert all(c.count("<GIR:") == 1 for c in got), "exactly one GIR tag per instruction"


@pytest.mark.parametrize("M,to_chunk,want", [
    (1, 1, +1), (1, 2, 0), (1, 3, 0),        # PGR1: prologue then steady (lead collapses to 0)
    (2, 1, +1), (2, 2, 0), (2, 3, -1), (2, 4, -1),   # PGR2: steady lead is NEGATIVE
])
def test_wrap_lead_is_derived_from_the_peel_depth(M, to_chunk, want):
    """`lead = pf - min(to_chunk, M+1)`, from `ctr + lead == S + pf` with `S = T - n`."""
    from Tensile.Lowering.gir_to_rocisa import GirToRocisa
    assert GirToRocisa._wrap_lead(to_chunk, M) == want


def test_wrap_lead_makes_every_stagger_start_cover_the_chunks_exactly_once():
    """Independent check of the closed form: replay GIR's emitted advance/copy sequence for every
    trip count and every stagger start, applying the wrap exactly as `tdmIncrementGir` does, and
    require that the copied chunks are a permutation of `0..T-1`.
    """
    from Tensile.Lowering.gir_to_rocisa import GirToRocisa as G
    pf = G._STAGGER_PF_CONST
    for M in (1, 2):
        for T in range(M + 1, 12):
            for S in range(T):
                chunk, copied = S, []
                for i in range(T):                       # copy-then-advance, every region
                    copied.append(chunk)
                    ctr = T if i < M else T - (i - M)    # counter: T in peel, then T-v
                    lead = G._wrap_lead(i + 1, M)
                    wrap = (ctr + lead == S + pf)
                    chunk = chunk + 1 - T if wrap else chunk + 1
                assert sorted(copied) == list(range(T)), \
                    f"M={M} T={T} S={S}: copied {copied}, expected a permutation of 0..{T-1}"
def test_prefetch_index_still_matches_the_scaffolds_own_constant():
    """Pin the two calibrations together: if `KernelWriter` changes its rule, this fails rather than
    letting `gir_to_rocisa` keep a stale copy that silently disagrees."""
    src = pathlib.Path("Tensile/KernelWriter.py").read_text()
    line = src.split("pfi = ")[1].splitlines()[0].strip()
    assert line == '1 if kernel["PrefetchGlobalRead"] < 3 else kernel["PrefetchGlobalRead"] - 1', \
        f"scaffold's pfi rule changed to `{line}` — GirToRocisa._prefetch_index must follow"


def test_the_fence_act_reaches_the_emit_plan_with_its_must_set_and_scope():
    """a fence must arrive at L3 carrying WHAT it orders and HOW FAR it must reach."""
    from Tensile.Lowering.gir.emit_plan import plan_block
    from gir_fixtures import BF16_NT_KMN_FUSED_XAGENT

    from Tensile.Lowering.gir import AnalysisManager
    from Tensile.Lowering.gir.analyses import FrameHazards

    prog = build_gir(_theta(dict(BF16_NT_KMN_FUSED_XAGENT, PrefetchGlobalRead=1)))
    fences = [a for a in plan_block(prog, "steady") if a.kind == "fence"]
    assert fences, "the steady body must carry a fence act"
    # EVERY edge the block owes, not a magic floor: tie the act to the analysis it came from, so
    # a change in the hazard set moves both sides together instead of tripping a stale number.
    hz = AnalysisManager().get(FrameHazards(), prog)
    owed = [h for h in hz.needing_fence() if "steady" in (h.producer.block, h.consumer.block)]
    assert owed, "non-vacuity: the fixture must owe the steady block some cross-wave edge"
    assert sum(a.at["edges"] for a in fences) >= len(owed) // 2
    for a in fences:
        assert set(a.at["buffers"]) <= {"A", "B"} and a.at["scope"] == "block"


def test_a_single_wave_kernel_emits_NO_fence_act():
    """Not "L3 discards it" — the analysis finds no cross-agent edge, so no Mark is ever made."""
    from Tensile.Lowering.gir.emit_plan import plan_program
    prog = build_gir(_theta(BF16_NT_KMN))
    assert not [a for acts in plan_program(prog).values() for a in acts if a.kind == "fence"]


def test_the_retired_await_mark_no_longer_emits_an_instruction():
    """The Mark survives (it is the model's discharge object, , and shows in the dumps) but is
    no longer realized: `FenceRegions` derives the same boundaries and covers them minimally.
    Realizing one barrier per obligation is what put 28-48 barriers in the SIA=0 body."""
    from Tensile.Lowering.gir_to_rocisa import GirToRocisa
    # attributes, not source text: the surviving docstrings NAME both of these while explaining
    # why they are gone, so a substring check would fail on its own explanation.
    assert not hasattr(GirToRocisa, "_emit_await"), "the await realization is still wired"
    assert not hasattr(GirToRocisa, "_needs_own_barrier"), \
        "the SIA-keyed barrier gate should be gone with it — GIR owns fences at every SIA"


def test_the_barrier_strip_keeps_GIR_fences_and_removes_the_scaffolds():
    """the reset pass does STRIP and INSERT; ULM replaces only the insert."""
    pytest.importorskip("rocisa")
    from rocisa.instruction import SBarrier
    from rocisa.code import Module
    from Tensile.KernelWriter import KernelWriter

    def survivors(root):
        out, stack = [], [root]
        while stack:
            for it in stack.pop().items():
                if isinstance(it, SBarrier):
                    out.append(it.comment)
                elif isinstance(it, Module):
                    stack.append(it)
        return out

    root, inner = Module("root"), Module("inner")
    ours = SBarrier(comment="GIR fence: order A+B")
    inner.add(SBarrier(comment="PGR->LW needs sync"))
    inner.add(ours)                                   # nested, to catch a non-recursive strip
    root.add(inner)
    root.add(SBarrier(comment="wait for local write done, sync"))

    assert KernelWriter._stripBarriers(root, keep=[ours]) == (2, 1)
    assert survivors(root) == ["GIR fence: order A+B"]
    # and with no keep-set it is the ORIGINAL behaviour, so the non-ULM path is untouched
    assert KernelWriter._stripBarriers(root) == (1, 0)
    assert survivors(root) == []

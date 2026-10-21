# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
The symbolic executor must FIND things, so every test here perturbs a working loop and demands the
finding: a check that only ever returns nothing is indistinguishable from a check that is dead.
Each assertion names both sides -- what the correct loop reports and what the broken one reports --
because an inequality that holds for a trivial reason is how a test goes green without testing.
"""

import pytest

from Tensile.Lowering.tool.symexec import (COUNTER_UNBOUNDED, OVERWRITE_BEFORE_READ,
                                           READ_BEFORE_LAND, READ_UNWRITTEN, WMMA_UNLANDED,
                                           execute, gir_disagreement, observed_pairs)
from Tensile.Lowering.tool.symexec.machine import ModelError
from Tensile.Lowering.tool.symexec.program import DecodeError, decode_range


def asm(*lines):
    return decode_range(list(lines), 0, len(lines))


def kinds(findings):
    return {f.kind for f in findings}


READ = ("ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3], v[vgprLocalReadAddrA+0] offset:0"
        " // <GIR: read A[tile=0,k=0]->X0>")
WMMA = "v_wmma_f32_16x16x128 v[vgprValuC+0:vgprValuC+0+7], v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3]"
COPY = ("tensor_load_to_lds s[sgprtdmAGroup0:sgprtdmAGroup0+3], s[sgprtdmAGroup1:sgprtdmAGroup1+7]"
        " // <GIR: copy A gen=0>")
SWAP_DESC = "s_xor_b32 s[sgprtdmAGroup0+1], s[sgprtdmAGroup0+1], 0x10000 // <GIR: swap A copy-hop>"
SWAP_READ = "v_xor_b32 v[vgprLocalReadAddrA], 0x10000, v[vgprLocalReadAddrA]"

BODY_OK = (READ, "s_wait_dscnt 0", WMMA, SWAP_DESC, SWAP_READ, COPY,
           "s_wait_tensorcnt 0", "s_wait_alu depctr_vm_vsrc(0)",
           "s_wait_alu depctr_va_vdst(0)")

#: A REAL PROLOGUE FILLS, WAITS, THEN READS.  That last read is what pins the read pointer against
#: the descriptor: an entry assignment under which it reads an unfilled buffer is one the kernel
#: itself refutes, so the relation is solved for rather than assumed.
PROLOGUE = (COPY, "s_wait_tensorcnt 0", READ, "s_wait_dscnt 0")


def test_correct_loop_has_no_findings():
    assert execute(asm(*PROLOGUE), asm(*BODY_OK)).findings == []


def test_dropping_the_tensorcnt_wait_is_caught():
    """The defect that made mxf8 PGR2xPLR1 wrong from the SECOND trip on."""
    body = asm(*[l for l in BODY_OK if "s_wait_tensorcnt" not in l])
    assert READ_BEFORE_LAND in kinds(execute(asm(*PROLOGUE), body).findings)


def test_an_undrained_counter_is_caught():
    """No `s_wait_tensorcnt` anywhere means the counter grows every trip until the hardware stalls."""
    body = asm(*[l for l in BODY_OK if "s_wait_tensorcnt" not in l])
    assert COUNTER_UNBOUNDED in kinds(execute(asm(*PROLOGUE), body).findings)


def test_raising_the_wait_count_is_caught():
    """`s_wait_tensorcnt 1` leaves the load this trip's read needs still in flight."""
    body = asm(*[l.replace("s_wait_tensorcnt 0", "s_wait_tensorcnt 1") for l in BODY_OK])
    assert READ_BEFORE_LAND in kinds(execute(asm(*PROLOGUE), body).findings)


def test_dscnt_wait_keeps_exactly_the_newest_n_read_instructions():
    """The older read lands at dscnt 1; dscnt 2 keeps both reads outstanding."""
    newer = ("ds_load_b128 v[vgprValuA_X0_I0+4:vgprValuA_X0_I0+4+3], "
             "v[vgprLocalReadAddrA+0] offset:16 // <GIR: read A[tile=1,k=0]->X0>")
    wait1 = execute(asm(*PROLOGUE), asm(READ, newer, "s_wait_dscnt 1", WMMA), limit=1)
    wait2 = execute(asm(*PROLOGUE), asm(READ, newer, "s_wait_dscnt 2", WMMA), limit=1)
    assert WMMA_UNLANDED not in kinds(wait1.findings)
    assert WMMA_UNLANDED in kinds(wait2.findings)


def test_wmma_before_its_ds_load_lands_is_caught():
    body = asm(*[l for l in BODY_OK if "s_wait_dscnt" not in l])
    assert WMMA_UNLANDED in kinds(execute(asm(*PROLOGUE), body).findings)


def test_refill_over_an_outstanding_read_is_caught():
    """Single-buffered and unwaited: the refill lands on the buffer being read."""
    body = asm(*[l for l in BODY_OK if "s_wait_dscnt" not in l and "0x10000" not in l])
    assert OVERWRITE_BEFORE_READ in kinds(execute(asm(*PROLOGUE), body).findings)


def test_reading_a_buffer_nothing_filled_is_caught():
    """A read whose ring position no copy ever fills -- the unprimed half of a double buffer."""
    body = asm(SWAP_READ, READ, SWAP_READ)
    assert READ_UNWRITTEN in kinds(execute(asm(*PROLOGUE), body).findings)


def test_the_parity_model_is_what_separates_safe_from_unsafe():
    """Two bodies identical but for the pointer swaps, with the read left outstanding on `dscnt`.

    Double-buffered, the refill targets the buffer the read is NOT on and there is no WAR; single-
    buffered it targets the one being read and there is.  Nothing but the tracked parity can produce
    that difference, so this is the test that the parity is load-bearing rather than decorative."""
    double = asm(READ, WMMA, SWAP_DESC, SWAP_READ, COPY, "s_wait_dscnt 0", "s_wait_tensorcnt 0")
    single = asm(READ, WMMA, COPY, "s_wait_dscnt 0", "s_wait_tensorcnt 0")
    assert OVERWRITE_BEFORE_READ in kinds(execute(asm(*PROLOGUE), single).findings)
    assert OVERWRITE_BEFORE_READ not in kinds(execute(asm(*PROLOGUE), double).findings)


def test_an_older_in_flight_writer_is_not_hidden_by_a_newer_one():
    """Two copies fill one buffer; the newer retires, the older does not.

    Keeping only the newest writer would report this clean, which is a false negative in the check
    whose whole value is that it has none."""
    body = asm(COPY, COPY, "s_wait_tensorcnt 1", READ, "s_wait_dscnt 0")
    assert READ_BEFORE_LAND in kinds(execute(asm(*PROLOGUE), body).findings)


def test_observed_pairs_and_gir_disagreement_report_the_executed_tensor():
    body = asm(*[l for l in BODY_OK if "s_wait_tensorcnt" not in l])
    ex = execute(asm(*PROLOGUE), body)
    assert {t for (t, _b) in observed_pairs(ex)} == {"A"}

    class _End:
        def __init__(self, operand):
            self.operand = operand

    class _Hazard:
        def __init__(self, a, b):
            self.producer, self.consumer = _End(a), _End(b)

    assert gir_disagreement(ex, [_Hazard("B", "B")]) == {"executed_not_modelled": ["A"],
                                                         "modelled_not_executed": ["B"]}


# ---- refusals: the machine must decline rather than answer from a guess --------------------

def test_unmodelled_access_width_is_refused_not_skipped():
    with pytest.raises(DecodeError):
        asm("ds_load_b999 v[vgprValuA_X0_I0+0], v[vgprLocalReadAddrA+0] offset:0")


def test_a_tdm_load_with_no_descriptor_is_refused():
    with pytest.raises(DecodeError):
        asm("tensor_load_to_lds s[sgprSomething:sgprSomething+3] // <GIR: copy A gen=0>")


def test_a_branch_in_the_executed_range_is_refused():
    with pytest.raises(DecodeError):
        asm("s_cbranch_scc0 label_Somewhere")


def test_multi_bit_xor_mask_is_refused_not_guessed():
    with pytest.raises(ModelError):
        execute(asm(), asm("v_xor_b32 v[vgprLocalReadAddrA], 0x30000, v[vgprLocalReadAddrA]"))


def test_a_non_xor_write_to_a_buffer_select_register_is_refused():
    """The blocker that made the first draft unsound: `v_add`/`s_mov` into the select register moves
    the buffer without the machine seeing it, and every answer afterwards is confidently wrong."""
    with pytest.raises(ModelError):
        execute(asm(), asm("v_add_nc_u32 v[vgprLocalReadAddrA], v4, v0"))
    with pytest.raises(ModelError):
        execute(asm(), asm("s_mov_b32 s[sgprtdmAGroup0+1], s88"))


def test_lds_read_with_no_tensor_is_refused():
    with pytest.raises(ModelError):
        execute(asm(), asm("ds_load_b128 v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3],"
                           " v[vgprSomethingElse+0]"))


def test_ds_store_is_refused_rather_than_given_the_read_pointer_parity():
    with pytest.raises(ModelError):
        execute(asm(), asm("ds_store_b128 v[vgprLocalWriteAddrA+0],"
                           " v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3]"))


def test_the_descriptor_frame_quotient_agrees_with_the_full_enumeration():
    """Pinning the descriptor side is a symmetry quotient, so it must give the SAME findings as
    enumerating both sides.  If it ever does not, the symmetry claim is false."""
    from itertools import product

    from Tensile.Lowering.tool.symexec.checks import _run_once, _signature, select_bits
    prologue = asm("v_add_nc_u32 v[vgprLocalReadAddrA], v4, v0",
                   "s_mov_b32 s[sgprtdmAGroup0+1], s88", *PROLOGUE)
    body = asm(*[l for l in BODY_OK if "s_wait_tensorcnt" not in l])
    bits = select_bits(prologue, body)
    keys = sorted(_run_once(prologue, body, 2, 0, {}, 16, bits).unknown)
    assert {"read:A+0", "desc:A+1"} <= set(keys)

    def sigs(ks):
        runs = [_run_once(prologue, body, 2, 0, dict(zip(ks, c)), 16, bits)
                for c in product(range(2), repeat=len(ks))]
        return set.intersection(*[{_signature(f) for f in r.findings} for r in runs])

    assert sigs(keys) == sigs([k for k in keys if not k.startswith("desc:")])


def test_a_fused_group_swap_moves_every_member():
    """`swap A+B copy-hop` is written on A's descriptor and moves B's buffer too.

    Without the tag the instruction says only that A moved, so B's copies would pile onto one buffer
    while its reads alternate -- 45 corpus kernels reported exactly that before this was modelled."""
    copy_b = ("tensor_load_to_lds s[sgprtdmBGroup0:sgprtdmBGroup0+3],"
              " s[sgprtdmBGroup1:sgprtdmBGroup1+7] // <GIR: copy B gen=0>")
    read_b = ("ds_load_b128 v[vgprValuB_X0_I0+0:vgprValuB_X0_I0+0+3],"
              " v[vgprLocalReadAddrB+0] offset:0 // <GIR: read B[tile=0,k=0]->X0>")
    swap_ab = ("s_xor_b32 s[sgprtdmAGroup0+1], s[sgprtdmAGroup0+1], 0x10000"
               " // <GIR: swap A+B copy-hop>")
    swap_rb = ("v_xor_b32 v[vgprLocalReadAddrB], 0x10000, v[vgprLocalReadAddrB]"
               " // <GIR: swap B read-hop>")
    prologue = asm(copy_b, "s_wait_tensorcnt 0", read_b, "s_wait_dscnt 0")
    body = asm(read_b, "s_wait_dscnt 0", swap_ab, swap_rb, copy_b, "s_wait_tensorcnt 0")
    assert READ_UNWRITTEN not in kinds(execute(prologue, body).findings)


def test_the_entry_hypothesis_is_applied_once_and_stays_on_the_ring():
    """A seed folded in twice put a pointer on a ring position no writer can name, which discarded
    every non-zero hypothesis by arithmetic and made the witness look far stronger than it is."""
    from Tensile.Lowering.tool.symexec.machine import Machine
    m = Machine(ring=2, seed={"read:A+0": 1})
    m.establishing = True
    m.run(asm("v_add_nc_u32 v[vgprLocalReadAddrA], v4, v0"))
    assert m.read_buffer("read:A+0") == 1
    m.run(asm(SWAP_READ))
    assert m.read_buffer("read:A+0") == 0


def test_reading_a_select_register_is_not_mistaken_for_writing_it():
    """`s_mov_b32 s88, s[sgprtdmAGroup0+1]` is a READ; splitting operands on `"],"` swallowed the
    later ones and made every such instruction look like a write to the select."""
    reads = asm("s_mov_b32 s88, s[sgprtdmAGroup0+1]",
                "s_cselect_b32 s88, s[sgprtdmAGroup0+1], s90")
    assert [i.writes_select for i in reads] == [None, None]
    execute(asm(), reads)


def test_the_wmma_finding_names_the_wmma_and_the_load_it_outran():
    body = asm(*[l for l in BODY_OK if "s_wait_dscnt" not in l])
    bad = [f for f in execute(asm(*PROLOGUE), body).findings if f.kind == WMMA_UNLANDED]
    assert bad and all(f.at is not f.other for f in bad)
    assert all(f.at.inst.kind == "wmma" and f.other.inst.kind == "ds_load" for f in bad)


def test_a_region_stride_the_body_rewrites_is_refused():
    """Counting steps is only exact if the same VALUE is added and taken away; a scratch SGPR the
    body reassigns between the two breaks that, and the claim must not be made anyway."""
    body = asm("s_mov_b32 s93, 8448",
               "s_add_u32 s[sgprtdmAGroup0+1], s[sgprtdmAGroup0+1], s93",
               "s_mov_b32 s93, 4352",
               "s_sub_u32 s[sgprtdmAGroup0+1], s[sgprtdmAGroup0+1], s93",
               COPY)
    with pytest.raises(ModelError):
        execute(asm(), body)


def test_a_region_stride_rewritten_between_BALANCED_pairs_is_not_refused():
    """The refusal is per outstanding displacement, not per register: a pair that closes before the
    stride is reassigned names one value, so the load's region is derivable."""
    body = asm("s_mov_b32 s93, 8448",
               "s_add_u32 s[sgprtdmAGroup0+1], s[sgprtdmAGroup0+1], s93",
               "s_sub_u32 s[sgprtdmAGroup0+1], s[sgprtdmAGroup0+1], s93",
               "s_mov_b32 s93, 4352",
               COPY)
    execute(asm(), body)


def test_a_symbolic_region_stride_is_tracked_like_an_allocated_sgpr():
    """Generated assembly keeps the split stride's symbolic name until final assembly."""
    body = asm(
        "s_mov_b32 s[sgprtdmALdsSplitIncs], 2176",
        "s_add_u32 s[sgprtdmAGroup0+1], s[sgprtdmAGroup0+1], "
        "s[sgprtdmALdsSplitIncs]",
        "s_sub_u32 s[sgprtdmAGroup0+1], s[sgprtdmAGroup0+1], "
        "s[sgprtdmALdsSplitIncs]",
        COPY)
    execute(asm(), body)


def test_a_displacement_out_of_the_buffer_is_refused_not_carried():
    """Whether a subtraction borrows into the select bit depends on the base, which is unknown."""
    body = asm(SWAP_DESC, "s_sub_u32 s[sgprtdmAGroup0+1], s[sgprtdmAGroup0+1], 128")
    with pytest.raises(ModelError):
        execute(asm(), body)


def test_the_vgpr_msb_does_not_change_register_identity():
    """`s_set_vgpr_msb` is four 2-bit PER-SLOT banks, not one bank number.

    An earlier version keyed registers on the raw immediate, which sent a load and its consumer to
    different keys and made the finding vanish.  Until the per-slot layout is decoded the bank is
    assumed equal at a load and its consumer, so the instruction must not perturb the chain."""
    without = asm(*[l for l in BODY_OK if "s_wait_dscnt" not in l])
    with_msb = asm(*([BODY_OK[0], "s_set_vgpr_msb 16452"]
                     + [l for l in BODY_OK[1:] if "s_wait_dscnt" not in l]))
    assert kinds(execute(asm(*PROLOGUE), without).findings) \
        == kinds(execute(asm(*PROLOGUE), with_msb).findings)
    assert WMMA_UNLANDED in kinds(execute(asm(*PROLOGUE), without).findings)


def test_a_range_form_scalar_write_marks_both_halves_written():
    from Tensile.Lowering.tool.symexec.program import scalar_dsts
    assert set(scalar_dsts(asm("s_add_u64 s[88:89], s[88:89], s[90:91]")[0])) == {"s88", "s89"}


def test_a_displacement_on_a_fused_member_lands_on_the_owner():
    """Every reader redirects a member to its group owner, so a write recorded under the member's
    own key is written where nothing looks.  14 corpus kernels silently discarded a region that
    way."""
    from Tensile.Lowering.tool.symexec.checks import _run_once, fused_aliases, select_bits
    swap = ("s_xor_b32 s[sgprtdmAGroup0+1], s[sgprtdmAGroup0+1], 0x10000"
            " // <GIR: swap A+B copy-hop>")
    body = asm(swap, "s_add_u32 s[sgprtdmBGroup0+1], s[sgprtdmBGroup0+1], 8704",
               "s_sub_u32 s[sgprtdmBGroup0+1], s[sgprtdmBGroup0+1], 8704")
    alias = fused_aliases(body)
    assert alias.get("desc:B+1") == "desc:A+1"
    ex = _run_once(asm(), body, 2, 0, {}, 16, select_bits(body), alias)
    assert "desc:B+1" not in ex.region


def test_two_groups_filling_one_tensor_refuse_the_quotient():
    """The quotient pins the descriptor side; that is only a change of frame if each tensor has one
    descriptor.  Unreachable on today's corpus, so the assert is what keeps it true rather than
    lucky."""
    copy_a = ("tensor_load_to_lds s[sgprtdmAGroup0:sgprtdmAGroup0+3],"
              " s[sgprtdmAGroup1:sgprtdmAGroup1+7] // <GIR: copy A gen=0>")
    copy_a2 = ("tensor_load_to_lds s[sgprtdmBGroup0:sgprtdmBGroup0+3],"
               " s[sgprtdmBGroup1:sgprtdmBGroup1+7] // <GIR: copy A gen=0>")
    with pytest.raises(ModelError):
        execute(asm(), asm(copy_a, copy_a2, READ, "s_wait_dscnt 0", SWAP_READ))

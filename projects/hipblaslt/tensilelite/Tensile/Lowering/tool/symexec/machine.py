# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
The machine state the emitted main loop runs on: completion counters, LDS buffers, registers.

An LDS address register carries three things this machine tracks separately, each from executed
instructions rather than from an assumption.  One bit selects the BUFFER -- WHICH bit is learned per
kernel from the xor the kernel emits, since it is the buffer stride (0x8000/0x10000/0x20000 all
occur).  Lower bits carry a REGION displacement a split kernel applies and later undoes.  The BYTES
come from the immediate offset and the opcode width.

FIVE conditions make the tracking sound, and each is enforced rather than assumed: the toggle is a
single bit; a register is xored by one mask only; a region displacement uses one stride per
register; an outstanding displacement does not span a rewrite of that stride register; and a literal
displacement stays inside its buffer.  Counting region steps is exact ONLY because of the fourth --
a stride reassigned between the add and the sub is two values under one name.  The ENTRY value of a
pointer is not derivable at all, so it is marked unknown and the caller enumerates every value it
could hold.

`s_set_vgpr_msb` is NOT modelled.  Its immediate is four 2-bit per-operand-slot banks
(`[1:0]`=src0, `[3:2]`=src1, `[5:4]`=src2, `[7:6]`=dst), so register identity here assumes the bank
is the same at a load and at its consumer.  An earlier version hashed register keys on the raw
immediate, which broke the def-use chain outright; ignoring it keeps the chain intact.
"""

from __future__ import annotations

import copy
import re
from collections import deque

from .program import (COUNTER, SELECT_INDEX, scalar_srcs, writes_first_operand, _first_operand)

#: The scalar operand a region displacement adds -- a REGISTER, never a literal, because the stride
#: is a runtime LDS size. Keep symbolic assembly names as well as allocated ``sNN`` names: generated
#: `.s` files deliberately retain ``s[sgpr...LdsSplitIncs]``.
_STRIDE_RE = re.compile(r"s\[[^\]]+\]|\bs\d+\b(?!\s*[:\]])")

#: any scalar register as written in the asm -- numbered, ranged, or symbolic.
_SREG = re.compile(r"s\[[^\]]*\]|\bs\d+\b")

#: SCC is an input to the value of a select, and an output of every compare and most ALU ops.
#: the ALU dependency counters an `s_wait_alu depctr_*` drains.
ALU_COUNTERS = ("vm_vsrc", "va_vdst")
#: their hardware width.  Past it the oldest entries have necessarily completed, so holding them
#: would model a dependence the hardware cannot express.
ALU_DEPTH = 16

_SCC_READERS = ("s_cselect", "s_addc", "s_subb", "s_cbranch_scc")
_SCC_WRITERS = ("s_cmp", "s_bitcmp", "s_add_u32", "s_sub_u32", "s_addc", "s_subb", "s_and_",
                "s_or_", "s_xor_", "s_lshl", "s_lshr", "s_ashr", "s_bfe", "s_abs")


#: `copy A+B gen=0` -- the movement's members, in the order the emitter listed them.
_UNIT_RE = re.compile(r"copy\s+([A-Za-z0-9_]+(?:\+[A-Za-z0-9_]+)*)")


def _token_per_member(inst):
    """`[(tensor, slot)]` from the emitted `sync LDS [...]` tag, or None when it does not pair.

    The tag is the frame-resolved storage this load fills, so it decides aliasing exactly where a
    descriptor's parity only guesses.  A fused movement lists its members and its slots in the same
    order; anything else is left to the pointer model rather than paired on a hunch."""
    m = _UNIT_RE.search(inst.gir or "")
    if not m or not inst.lds_tokens:
        return None
    members = m.group(1).split("+")
    slots = sorted(inst.lds_tokens)
    return list(zip(members, slots)) if len(members) == len(slots) else None


def _dst_tokens(inst):
    """The scalar names this instruction writes: the destination as spelled, plus a range's halves."""
    if not writes_first_operand(inst.op):
        return ()
    m = _SREG.match(_first_operand(inst.text.partition(" ")[2])[0].strip())
    if not m:
        return ()
    rng = re.match(r"s\[\s*(\d+)\s*:\s*(\d+)\s*\]", m.group(0))
    return (m.group(0),) + (tuple("s%d" % i for i in range(int(rng.group(1)),
                                                           int(rng.group(2)) + 1)) if rng else ())


class ModelError(RuntimeError):
    """The machine met state it cannot represent faithfully.

    Raised rather than approximated: an aliasing question answered by a guess is worse than one not
    answered at all, because the report looks the same."""


class Access:
    """One tensor-buffer access: which buffer, which REGION of it, and which bytes.

    `lo is None` means the whole buffer, which is what a TDM load fills.  `region is None` means the
    region is not derivable and the access may touch any of them -- true of every LDS read, whose
    region is folded into an immediate this machine cannot resolve against a symbolic stride.

    So region NEVER narrows an overlap today: every comparison the checks make has a read on one
    side, and a read's region is always None.  What it does do is lengthen the period and gate the
    refusals; it will only discriminate once reads carry a region."""

    __slots__ = ("tensor", "buffer", "region", "lo", "hi")

    def __init__(self, tensor, buffer, region=None, lo=None, hi=None):
        self.tensor, self.buffer, self.region = tensor, buffer, region
        self.lo, self.hi = lo, hi

    def overlaps(self, other):
        if self.tensor != other.tensor or self.buffer != other.buffer:
            return False
        if (self.region is not None and other.region is not None
                and self.region != other.region):
            return False
        if self.lo is None or other.lo is None:
            return True
        return self.lo < other.hi and other.lo < self.hi

    def __repr__(self):
        span = "whole" if self.lo is None else f"[{self.lo},{self.hi})"
        where = "" if self.region is None else f"r{self.region}"
        return f"{self.tensor}#{self.buffer}{where}{span}"


class Event:
    """An instruction as executed: when it issued, what it accessed, when it completed.

    One Event per INSTRUCTION, carrying every access it makes, because that is what the hardware
    counter counts -- a fused TDM load filling four tensors increments `tensorcnt` once."""

    __slots__ = ("inst", "wave", "clock", "accesses", "retired_at", "epoch")

    def __init__(self, inst, wave, clock, accesses, epoch):
        self.inst, self.wave, self.clock = inst, wave, clock
        self.accesses = tuple(accesses)
        self.epoch, self.retired_at = epoch, None

    @property
    def landed(self):
        return self.retired_at is not None

    def overlaps(self, access):
        return any(a.overlaps(access) for a in self.accesses)

    def __repr__(self):
        what = ",".join(repr(a) for a in self.accesses)
        return f"{self.inst!r}@t{self.clock}w{self.wave}{'[' + what + ']' if what else ''}"


def _resolved(inst, alias):
    """`inst` with its select operands redirected through the fused-group alias."""
    dst = alias.get(inst.writes_select, inst.writes_select)
    src = alias.get(inst.src_select, inst.src_select)
    if dst == inst.writes_select and src == inst.src_select:
        return inst
    view = copy.copy(inst)
    view.writes_select, view.src_select = dst, src
    return view


class Machine:
    """One wave's architectural state.

    The counters are in-order FIFOs -- `s_wait_<c> N` retires from the front until at most N remain
    -- which is the whole of the gfx1250 completion model, and the only thing that makes a memory
    instruction's result observable."""

    def __init__(self, wave=0, ring=2, seed=None, select_bits=None):
        self.wave = wave
        self.ring = ring
        self.clock = 0
        self.epoch = 0                      # barrier WAITS passed: the synchronisation interval
        self.fifo = {c: deque() for c in set(COUNTER.values()) | set(ALU_COUNTERS)}
        self.parity = {}                    # buffer-select key -> toggles so far, mod ring
        self.defined_by = {}                # register cell -> the Event that loads it
        self.trace = []
        self.establishing = False           # inside the prologue, where a pointer may be built
        self.unknown = set()                # select keys whose entry buffer is not derivable
        self.seed = dict(seed or {})        # the entry buffer being explored for those keys
        self.region = {}                    # select key -> net region displacement in BYTES
        self.steps = {}                     # select key -> {stride generation: net displacements}
        self.stride = {}                    # select key -> the operand those displacements use
        self.alias = {}                     # a fused member's descriptor -> the group owner's
        self.value = {}                     # scalar -> the tag of the value it holds
        self.scc = None                     # the tag of the value SCC holds
        self._tags = {}                     # canonical expression -> its tag
        # THE SELECT BIT IS THE BUFFER STRIDE, so it is a property of the KERNEL's LDS layout, not a
        # constant: it is read off the xor the kernel itself emits.
        self.select_bit = dict(select_bits or {})

    # ---- buffer parity ------------------------------------------------------------------
    def _toggle(self, target, mask):
        if mask is None:
            raise ModelError(f"xor on {target} with no literal mask: the buffer select cannot be "
                             f"tracked, so no aliasing answer is available")
        if mask & (mask - 1):
            raise ModelError(f"xor mask {mask:#x} on {target} is not a single bit, so it does not "
                             f"select one of {self.ring} buffers; the ring model does not apply")
        self.parity[target] = (self.parity.get(target, 0) + 1) % self.ring

    def _region_step(self, inst):
        """A balanced region displacement of a descriptor, or None.

        `Group0+1` carries two fields: the select bit picks the BUFFER and the lower bits carry a
        REGION offset a TDMSplit kernel advances to aim the next copy at the other half, then undoes.
        Counting the steps stands in for the value only while the stride holds ONE value, so the
        steps are filed under its value tag: recomputing it identically is the same displacement."""
        if inst.op not in ("s_add_u32", "s_sub_u32") or not inst.writes_select:
            return None
        # The destination descriptor is also the first source. The region stride is the final
        # addend; searching the whole source tail would pick that descriptor when names are still
        # symbolic (`s[sgpr...]`).
        stride = _STRIDE_RE.search(inst.text.rsplit(",", 1)[-1])
        if stride is None:
            return None
        return (stride.group(0), self.value.get(stride.group(0), stride.group(0)),
                1 if inst.op == "s_add_u32" else -1)

    def _net_steps(self, key, inst):
        """This descriptor's outstanding symbolic displacement, as a step count.

        Steps add up per VALUE of the stride register; a displacement left outstanding across a
        change of that value names two different displacements, so it is refused."""
        live = {gen: n for gen, n in (self.steps.get(key) or {}).items() if n}
        if len(live) > 1:
            raise ModelError(
                f"line {inst.line}: {key} is displaced by {self.stride.get(key)} across "
                f"{len(live)} distinct values of it, so the outstanding steps do not name one "
                f"displacement: {inst.text!r}")
        return next(iter(live.values()), 0)

    def _literal_displacement(self, inst):
        """An add/sub of a KNOWN constant to a select register.  True if it was accounted for.

        A constant is decidable against the select bit, so the two fields never have to be guessed
        apart: adding exactly the bit moves to the other BUFFER (which is how a sibling pointer to
        the far half is built), and adding less than it moves within a buffer, to another REGION.
        Anything at or above the bit that is not the bit itself would carry into the select and is
        refused."""
        if inst.op not in ("s_add_u32", "s_sub_u32", "v_add_nc_u32", "v_sub_nc_u32",
                           "v_add_u32", "v_sub_u32") or inst.literal is None:
            return False
        dst, src = inst.writes_select, inst.src_select or inst.writes_select
        bit = self.select_bit.get(dst) or self.select_bit.get(src)
        moved = self.region.get(src, 0) + (-1 if "sub" in inst.op else 1) * inst.literal
        if bit is None:
            # NO XOR ANYWHERE ON THIS POINTER means the kernel keeps one buffer for it, so there is
            # no stride to carry into and the whole displacement is a region move.  No bound is
            # needed because with one buffer there is nothing for an overflow to alias onto.
            self.parity[dst] = self.parity.get(src, 0)
            self.region[dst] = moved
            return True
        # NO INVENTED CARRY.  Whether a displacement borrows into the select bit depends on the
        # pointer's BASE, which is exactly what this machine does not know, so a result outside the
        # buffer is refused rather than resolved by assuming the base is zero.
        if not 0 <= moved < bit:
            raise ModelError(f"line {inst.line}: displacing {dst} to {moved} leaves the buffer of "
                             f"stride {bit:#x}; whether that carries into the select bit depends on "
                             f"the base, which is not derivable: {inst.text!r}")
        self.parity[dst] = self.parity.get(src, 0)
        self.region[dst] = moved
        return True

    def _check_select_write(self, inst):
        """Handle a write to a buffer-select register that is not a single-bit toggle.

        In the PROLOGUE such a write establishes the pointer, and its buffer bit is not derivable
        without evaluating the address arithmetic, so the key is marked UNKNOWN and takes whatever
        seed the caller is exploring -- the caller enumerates both values and keeps only findings
        that hold either way.  In the BODY it is refused: there the pointer is supposed to move by
        toggling, and an unmodelled write would move it without the machine seeing it."""
        if not inst.writes_select or inst.kind == "xor":
            return
        # RESOLVE ONCE, BEFORE ANY WRITE.  Every reader of this state redirects a fused member to
        # its group owner, so a write recorded under the member's own key is written where nothing
        # will ever look.  Deriving the same fact in two places is the defect shape this file has
        # produced four times; this is the fifth and last.
        inst = _resolved(inst, self.alias)
        if self._literal_displacement(inst):
            return
        step = self._region_step(inst)
        if step is not None:
            stride, gen, sign = step
            key = inst.writes_select
            if self.stride.setdefault(key, stride) != stride:
                raise ModelError(f"line {inst.line}: {key} is displaced by two different strides "
                                 f"({self.stride[key]} and {stride}), so a region count no longer "
                                 f"names one displacement")
            nets = self.steps.setdefault(key, {})
            nets[gen] = nets.get(gen, 0) + sign
            return
        if not self.establishing:
            raise ModelError(
                f"line {inst.line}: {inst.op} writes the buffer-select register "
                f"{inst.writes_select} inside the loop body, where this machine models the pointer "
                f"as moving only by single-bit xor: {inst.text!r}")
        self.unknown.add(inst.writes_select)
        self.parity[inst.writes_select] = 0

    def read_buffer(self, key):
        # A POINTER'S ENTRY BUFFER IS NEVER DERIVABLE, whether the prologue built it or it arrived
        # already built, so every select key the machine touches is an unknown for the caller to
        # enumerate.  Assuming 0 for the ones the prologue happens not to write is the same guess
        # in a quieter place.
        key = self.alias.get(key, key)
        self.unknown.add(key)
        return self._buffer_of(key)

    def desc_key(self, group):
        """The parity key of a TDM group's descriptor -- the same spelling `select_target` yields."""
        return "desc:%s%+d" % (group, SELECT_INDEX)

    def write_buffer(self, group):
        return self._buffer_of(self.alias.get(self.desc_key(group), self.desc_key(group)))

    def _buffer_of(self, key):
        """The ring position of a pointer: its toggles plus the entry buffer being explored.

        The seed is applied HERE and only here, and reduced mod the ring: folding it in a second
        place put pointers on a position no writer can name, which silently eliminated every
        non-zero hypothesis and made the witness look far stronger than it is."""
        return (self.parity.get(key, 0) + self.seed.get(key, 0)) % self.ring

    # ---- access construction ------------------------------------------------------------
    def _accesses(self, inst):
        if inst.kind == "ds_load":
            tensor = inst.addr_tensor or (sorted(inst.tensors)[0] if inst.tensors else None)
            if tensor is None or inst.addr_select is None:
                raise ModelError(f"line {inst.line}: an LDS read naming no tensor -- neither its "
                                 f"address register nor its GIR tag says which buffer it touches")
            lo = inst.offset
            # THE STORAGE IS THE EMITTED `sync LDS N` -- the frame-resolved slot, which already
            # carries `(frame + gdelta) % ring`.  The machine supplies the TIMING (issue/retire from
            # the real wait sequence); the slot is the identity, and both sides read it the same way
            # so the comparison is in one domain.  Falls back to the pointer parity when untagged.
            slots = frozenset(inst.lds_tokens or ())
            buf = self.read_buffer(inst.addr_select)
            if slots:
                return [Access(tensor, buf, s, lo, lo + inst.width) for s in sorted(slots)]
            return [Access(tensor, buf, None, lo, lo + inst.width)]
        if inst.kind == "ds_store":
            raise ModelError(f"line {inst.line}: `ds_store` to LDS is not modelled -- its buffer "
                             f"select lives in a write-address register this machine does not "
                             f"track, so its aliasing cannot be answered")
        if inst.kind == "tensor_load":
            key = self.alias.get(self.desc_key(inst.desc_group), self.desc_key(inst.desc_group))
            self.unknown.add(key)
            # Validate that a symbolic region displacement still denotes one value before using
            # this descriptor. Access identity remains the emitted sync-LDS slot below.
            self._net_steps(key, inst)
            buf = self.write_buffer(inst.desc_group)
            # SAME DOMAIN AS THE READ: the slots this movement fills, per its own `sync LDS [..]`.
            # A fused load lists one slot per member, so each member gets its own access.
            paired = _token_per_member(inst)
            if paired:
                return [Access(t, buf, s) for t, s in paired if t in inst.tensors]
            slots = sorted(inst.lds_tokens or ())
            if slots:
                return [Access(t, buf, s) for t in sorted(inst.tensors) for s in slots]
            return [Access(t, buf, None) for t in sorted(inst.tensors)]
        return []

    # ---- scalar values ------------------------------------------------------------------
    def _intern(self, expr):
        return self._tags.setdefault(expr, len(self._tags))

    def _revalue(self, inst):
        """Tag every scalar this instruction writes with the VALUE it now holds.

        The tag is the computation itself -- opcode, source tags, and SCC where the opcode reads it
        -- so recomputing a stride the same way from the same inputs yields one tag, while any
        change to an input yields another.  A register never written keeps its own name as its tag."""
        expr = inst.op + " " + _SREG.sub(
            lambda m: "#%s" % self.value.get(m.group(0), m.group(0)), scalar_srcs(inst))
        if inst.op.startswith(_SCC_READERS):
            expr += " scc#%s" % self.scc
        tag = self._intern(expr)
        for i, name in enumerate(_dst_tokens(inst)):
            self.value[name] = tag if i == 0 else self._intern("%s @%d" % (expr, i))
        if inst.op.startswith(_SCC_WRITERS):
            self.scc = self._intern(expr + " ->scc")

    # ---- execution ----------------------------------------------------------------------
    def issue(self, inst):
        """Execute one instruction; returns the Event it created, or None."""
        self.clock += 1
        self._check_select_write(inst)
        self._revalue(inst)
        if inst.kind == "xor":
            owner = inst.xor_target
            for target in (inst.xor_targets or ()):
                if owner and target != owner:
                    # ONE PHYSICAL SELECT BIT PER DESCRIPTOR: a fused member does not get a key of
                    # its own, it resolves to the register the swap is actually written on.
                    self.alias[target] = owner
            target = owner or (inst.xor_targets or (None,))[0]
            if target is None:
                # NAMES NO SELECT KEY -- neither a GIR `swap` tag nor a descriptor/read-addr
                # spelling -- so it moves no buffer; it is scalar arithmetic, already valued.
                return None
            self._toggle(self.alias.get(target, target), inst.xor_mask)
            return None
        if inst.kind == "barrier":
            self.epoch += 1
            return None
        if inst.kind in ("barrier_signal", "other", "vgpr_msb"):
            return None
        event = Event(inst, self.wave, self.clock, self._accesses(inst), self.epoch)
        if inst.counter:
            self.fifo[inst.counter].append(event)
        if inst.kind in ("valu", "wmma"):
            # A VALU is outstanding on the ALU counters until its sources are read (`vm_vsrc`) and
            # its result written (`va_vdst`); `s_wait_alu` retires it, and so does the counter's
            # own width.
            for c in ALU_COUNTERS:
                q = self.fifo[c]
                q.append(event)
                while len(q) > ALU_DEPTH:
                    q.popleft()
        self.trace.append(event)
        for r in inst.defs:
            self.defined_by[r] = event
        return event

    def wait(self, counter, n):
        """Retire from the front until at most `n` entries remain."""
        q = self.fifo.get(counter)
        if q is None:
            return []
        out = []
        while len(q) > n:
            e = q.popleft()
            e.retired_at = self.clock
            out.append(e)
        return out

    def outstanding(self, counter):
        return list(self.fifo.get(counter, ()))

    def run(self, program):
        """Execute a decoded stream once."""
        for item in program:
            if isinstance(item, tuple):
                self.wait(item[1], item[2])
            else:
                self.issue(item)
        return self

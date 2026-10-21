# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
What the executed trace proves about the emitted loop.

Every finding names an instruction pair and the executed state that makes it a violation -- an entry
still in a completion FIFO, a buffer no copy filled, a counter that only grows.

There is no cross-wave check here.  Whether two WAVES collide is a question about the per-lane part
of the address, and this machine deliberately discards it: single-wave checks compare accesses
through the same base register, where the base cancels and offset+width is exact, but two waves hold
different bases and it does not cancel.  A check written anyway would fire on correct code that
merely partitions one buffer between waves, which is how the previous tooling reached 2459 findings
on hardware-validated kernels.  It needs a wave-dependent address term first.
"""

from __future__ import annotations

from .machine import Machine, ModelError
from .program import decode_range, loop_body, setup_new_tile
from .walk import walk_all_entries

#: a read whose data has not landed: the TDM load that fills its buffer is still on `tensorcnt`.
READ_BEFORE_LAND = "read_before_land"
#: a refill issued while a read of the SAME buffer is still on `dscnt`.
OVERWRITE_BEFORE_READ = "overwrite_before_read"
#: a wmma consuming a register whose `ds_load` has not retired.
WMMA_UNLANDED = "wmma_unlanded"
#: any other VALU consuming a register whose `ds_load` has not retired.
VALU_UNLANDED = "valu_unlanded"
#: a `ds_load` writing a register still queued as an unread source of an earlier VALU.
VMEM_CLOBBERS_VALU_SRC = "vmem_clobbers_valu_src"
#: a read of a buffer no TDM load has ever filled.
READ_UNWRITTEN = "read_unwritten"
#: a completion counter the loop never drains, so it grows every trip until the hardware stalls.
COUNTER_UNBOUNDED = "counter_unbounded"
#: a `ds_load` landing on a register whose previous value no wmma has consumed yet.
REGISTER_CLOBBERED = "register_clobbered"

class Finding:
    __slots__ = ("kind", "at", "other", "detail")

    def __init__(self, kind, at, other=None, **detail):
        self.kind, self.at, self.other, self.detail = kind, at, other, detail

    def __repr__(self):
        where = f"{self.kind}: {self.at!r}"
        if self.other is not None:
            where += f" vs {self.other!r}"
        return where + (f" {self.detail}" if self.detail else "")


class Executor(Machine):
    """A machine that records why each memory access was or was not safe when it issued."""

    def __init__(self, wave=0, ring=2, seed=None, select_bits=None):
        super().__init__(wave=wave, ring=ring, seed=seed, select_bits=select_bits)
        self.writers = {}           # (tensor, buffer) -> [Events that filled it, oldest first]
        self.unconsumed = {}        # register -> the load that defined it, until a wmma reads it
        self.findings = []
        self.reporting = True       # off while priming, so entry state is established quietly

    def _unlanded_writers(self, access):
        """Writers of this buffer still in flight, per the counter -- NOT per `landed`.

        `writers` is an append-only history: a write the walk left behind keeps `retired_at is None`
        forever, so `landed` reports it outstanding no matter how many waits follow.  The counter's
        own queue is the only thing that says what has not completed."""
        live = {id(e) for e in self.outstanding("tensorcnt")}
        return [w for w in self.writers.get((access.tensor, access.buffer), ())
                if id(w) in live and w.overlaps(access)]

    def issue(self, inst):
        event = super().issue(inst)
        if event is None:
            return event
        for acc in event.accesses:
            key = (acc.tensor, acc.buffer)
            if inst.kind == "ds_load":
                seen = self.writers.get(key)
                if not seen:
                    if self.reporting:
                        self.findings.append(Finding(READ_UNWRITTEN, event, buffer=key))
                else:
                    # EVERY unretired writer is a violation, not just the newest: an older one
                    # still in flight can land after the reader and clobber what it read.
                    for w in self._unlanded_writers(acc):
                        if self.reporting:
                            self.findings.append(Finding(
                                READ_BEFORE_LAND, event, w,
                                tensorcnt_depth=len(self.outstanding("tensorcnt"))))
            elif inst.kind == "tensor_load":
                live = [e for e in self.outstanding("dscnt") if e.overlaps(acc)]
                if live and self.reporting:
                    self.findings.append(Finding(OVERWRITE_BEFORE_READ, event, live[0],
                                                 outstanding=len(live)))
                self.writers.setdefault(key, []).append(event)
        if inst.kind == "ds_load":
            # `vm_vsrc` is the counter that says a VALU has READ its sources.  Writing one of those
            # registers while it is still queued clobbers the source before the VALU consumed it.
            for e in self.outstanding("vm_vsrc"):
                hit = set(inst.defs) & set(e.inst.uses)
                if hit and self.reporting:
                    self.findings.append(Finding(VMEM_CLOBBERS_VALU_SRC, event, e,
                                                 registers=sorted(hit)[:3]))
            # A REGISTER IS A BUFFER TOO: refilling one whose value no wmma has taken loses that
            # value outright, and nothing in the counter model forbids it.
            for reg in sorted(inst.defs):
                prior = self.unconsumed.get(reg)
                if prior is not None and prior is not event and self.reporting:
                    self.findings.append(Finding(REGISTER_CLOBBERED, event, prior, register=reg))
                self.unconsumed[reg] = event
        if inst.kind == "wmma":
            for reg in inst.uses:
                self.unconsumed.pop(reg, None)
            if self.reporting:
                self._check_wmma(event)
        elif inst.kind == "valu":
            # EVERY consumer, not just the wmma: the drain reads accumulators back with v_mov and
            # v_cvt, and a dscnt that does not cover those is the same defect one instruction later.
            for reg in inst.uses:
                self.unconsumed.pop(reg, None)
                d = self.defined_by.get(reg)
                if d is not None and not d.landed and self.reporting:
                    self.findings.append(Finding(VALU_UNLANDED, event, d, register=reg))
        return event

    def _check_wmma(self, event):
        """The finding names the WMMA and the load it outran.

        Naming the load twice collapsed every consumer of one register to a single signature, so the
        cross-hypothesis intersection could not tell two wmmas apart."""
        for reg in sorted(event.inst.uses):
            d = self.defined_by.get(reg)
            if d is not None and not d.landed:
                self.findings.append(Finding(WMMA_UNLANDED, event, d, register=reg))


def _parity_state(ex):
    """Everything that decides which buffer bytes the next trip names: parity AND region.

    Covering parity alone leaves a split kernel reported at an arbitrary phase of its REGION cycle,
    which is the same phase dependence the period exists to remove, on the other axis.  Untouched
    entries are dropped so a key first seen mid-run compares equal to its absence."""
    return frozenset([("p", k, v) for k, v in ex.parity.items() if v]
                     + [("r", k, v) for k, v in ex.region.items() if v]
                     # the NET, not the per-generation split: a generation number counts trips, so
                     # keying on it would make every trip a new state and the period never close.
                     + [("s", k, sum(gens.values())) for k, gens in ex.steps.items()
                        if sum(gens.values())])


def _parity_period(ex, body, limit=16):
    """How many trips until the buffer parities return to where they started.

    The alias domain is the parity, so this is the loop's real period: reporting on any single trip
    reports one PHASE of it, which is why a fixed trip count answers differently for different
    parities of that count."""
    start = _parity_state(ex)
    for n in range(1, limit + 1):
        ex.run(body)
        if _parity_state(ex) == start:
            return n
    raise ModelError(f"the buffer and region state did not return to its entry value within "
                     f"{limit} trips, so the loop has no period to report a steady trip for")


def select_bits(*programs):
    """`{select key: buffer stride}` read off the xors the kernel emits.

    The stride is a property of the kernel's LDS layout, not a constant, so it is learned rather
    than assumed; a register xored by two different masks has no single stride and is refused."""
    out = {}
    for program in programs:
        for item in program:
            if isinstance(item, tuple) or item.kind != "xor":
                continue
            mask = item.xor_mask
            if mask is None or mask & (mask - 1):
                continue
            for target in (item.xor_targets or ()):
                if out.setdefault(target, mask) != mask:
                    raise ModelError(f"{target} is xored by both {out[target]:#x} and {mask:#x}, "
                                     f"so it has no single buffer stride")
    return out


def fused_aliases(*programs):
    """`{member key: owner key}` for every fused swap, built from the decoded stream.

    Established before execution, not at the first xor: a copy issued BEFORE the swap would
    otherwise resolve through an unaliased key and land on a buffer of its own."""
    out = {}
    for program in programs:
        for item in program:
            if isinstance(item, tuple) or item.kind != "xor" or not item.xor_target:
                continue
            for target in (item.xor_targets or ()):
                if target != item.xor_target:
                    out[target] = item.xor_target
    return out


def _run_once(prologue, body, ring, wave, seed, limit, bits, alias=None):
    """One execution under one hypothesis about the entry buffer of each established pointer."""
    ex = Executor(wave=wave, ring=ring, seed=seed, select_bits=bits)
    ex.alias.update(alias or {})
    # THE PROLOGUE IS THE WITNESS that pins the read pointer against the descriptor: it fills a
    # buffer and then reads it, so a hypothesis under which those name different buffers is one the
    # kernel refutes.  Its findings are collected for that purpose, not reported.
    ex.reporting = True
    ex.establishing = True
    ex.run(prologue)
    ex.establishing = False
    ex.entry_findings, ex.findings = ex.findings, []
    ex.reporting = False
    period = _parity_period(ex, body, limit=limit)
    for _ in range(period):
        ex.run(body)
    ex.reporting = True
    depth_before = {c: len(q) for c, q in ex.fifo.items()}
    for _ in range(period):
        ex.run(body)
    for counter, before in depth_before.items():
        after = len(ex.fifo[counter])
        if after > before:
            ex.findings.append(Finding(COUNTER_UNBOUNDED, ex.trace[-1] if ex.trace else None,
                                       counter=counter, per_period=after - before))
    ex.period = period
    return ex


def _one_group_per_tensor(ex):
    """The descriptor-frame quotient is only a symmetry if each tensor has ONE filling group.

    Pinning the descriptor side is justified by every comparison being a tensor's read pointer
    against its descriptor; two groups filling one tensor gives it two descriptors and the pinning
    stops being a change of frame."""
    groups = {}
    for event in ex.trace:
        if event.inst.kind != "tensor_load":
            continue
        for acc in event.accesses:
            groups.setdefault(acc.tensor, set()).add(event.inst.desc_group)
    bad = {t: sorted(g) for t, g in groups.items() if len(g) > 1}
    if bad:
        raise ModelError(f"these tensors are filled by more than one descriptor group ({bad}), so "
                         f"pinning the descriptor side is not a change of frame and the hypothesis "
                         f"space cannot be quotiented by it")


def _signature(f):
    """What makes two findings from different runs the same finding."""
    if f.kind == COUNTER_UNBOUNDED:
        # A counter finding is about the LOOP, not an access; its `at` is just the last event, so
        # folding that event's buffers in would make two runs disagree about the same fact.
        return (f.kind, f.detail.get("counter"))
    return (f.kind,
            getattr(getattr(f.at, "inst", None), "line", None),
            getattr(getattr(f.other, "inst", None), "line", None),
            f.detail.get("register"), f.detail.get("counter"), f.detail.get("buffer"),
            tuple(sorted((a.tensor, a.buffer) for a in getattr(f.at, "accesses", ()))))


def execute(prologue, body, ring=2, wave=0, limit=16, max_unknown=10):
    """Report the findings that hold under EVERY entry buffer the prologue could have produced.

    A pointer built by address arithmetic (`v_add`, `s_mov`) has a buffer bit this machine cannot
    derive, so assuming one would make every later answer rest on a guess.  Instead each such
    pointer is an unknown, all assignments are executed, and only findings present in all of them
    are returned -- a finding that survives every hypothesis does not depend on any of them.

    Reporting over one full parity PERIOD rather than a fixed trip removes the other phase
    dependence: every buffer the ring visits is reported exactly once.
    """
    bits = select_bits(prologue, body)
    alias = fused_aliases(prologue, body)
    probe = _run_once(prologue, body, ring, wave, {}, limit, bits, alias)
    # ONLY RELATIVE PARITY IS OBSERVABLE: every comparison is between a tensor's read pointer and
    # its descriptor, so shifting both leaves every overlap unchanged.  Pinning the descriptor side
    # as the reference frame is a quotient by that symmetry, not a sample of the space.
    _one_group_per_tensor(probe)
    keys = sorted(k for k in probe.unknown if not k.startswith("desc:"))
    if not keys:
        return probe
    if len(keys) > max_unknown:
        raise ModelError(f"{len(keys)} pointers have an entry buffer this machine cannot derive "
                         f"({', '.join(keys[:4])}...); enumerating {ring}**{len(keys)} hypotheses "
                         f"is not a check, so no answer is offered")
    from itertools import product
    runs = [_run_once(prologue, body, ring, wave, dict(zip(keys, combo)), limit, bits, alias)
            for combo in product(range(ring), repeat=len(keys))]
    consistent = [r for r in runs
                  if not any(f.kind == READ_UNWRITTEN for f in r.entry_findings)]
    if not consistent:
        raise ModelError(f"no entry buffer assignment over {sorted(keys)} lets the prologue read "
                         f"what it filled, so the pointers cannot be placed against each other and "
                         f"nothing about the loop follows")
    runs = consistent
    common = set.intersection(*[{_signature(f) for f in r.findings} for r in runs])
    held = [f for f in runs[0].findings if _signature(f) in common]
    runs[0].findings = held
    runs[0].hypotheses = len(runs)
    runs[0].unknown_pointers = keys
    return runs[0]


def observed_pairs(ex):
    """`{(tensor, buffer): [(writer_inst, reader_inst)]}` -- the dependences the run exercised."""
    pairs = {}
    for f in ex.findings:
        if f.kind in (READ_BEFORE_LAND, OVERWRITE_BEFORE_READ) and f.other is not None:
            for acc in f.at.accesses:
                pairs.setdefault((acc.tensor, acc.buffer), []).append((f.other.inst, f.at.inst))
    return pairs



class Report:
    """Findings plus the hypothesis accounting they rest on.

    An EMPTY finding list means one of two unrelated things -- nothing was found, or several
    admissible entry phases were found and they disagreed, so nothing survived the intersection.
    Reporting the count alongside is what keeps the second from reading as the first."""

    __slots__ = ("findings", "floor", "seeds_at_floor", "seeds_tried", "per_seed")

    def __init__(self, findings, floor, seeds_at_floor, seeds_tried, per_seed):
        self.findings, self.floor = findings, floor
        self.seeds_at_floor, self.seeds_tried, self.per_seed = seeds_at_floor, seeds_tried, per_seed

    @property
    def undecided(self):
        """The empty answer is an intersection of hypotheses that disagreed, not a clean kernel."""
        return not self.findings and self.seeds_at_floor > 1 and any(self.per_seed)

    def __repr__(self):
        return ("Report(%d findings, floor=%d, seeds %d/%d at floor, per-seed %s%s)"
                % (len(self.findings), self.floor, self.seeds_at_floor, self.seeds_tried,
                   self.per_seed, ", UNDECIDED" if self.undecided else ""))


def check_paths(lines, ring=2, wave=0, head="label_LoopBeginL", **kw):
    """`[Finding]` over every reachable path; see `report_paths` for the hypothesis accounting."""
    return report_paths(lines, ring=ring, wave=wave, head=head, **kw).findings


def report_paths(lines, ring=2, wave=0, head="label_LoopBeginL", **kw):
    """Findings over EVERY reachable path of a whole kernel, not one trace through it.

    `execute` answers for one straight-line stream, so it can only ever describe one arm of every
    branch it walks past -- for a GEMM that means the drain and both no-load-loop arms are unseen.
    This builds the CFG, forks the machine state at each conditional, and reports what holds under
    every entry phase the kernel's own reads do not refute.

    The walk starts at `setupNewTile`, where a tile's own state begins: above it the kernel only
    loads arguments and picks an ArgType, branching heavily and touching no LDS.
    """
    span = loop_body(lines, head)
    if span is None:
        raise ModelError(f"no rolled loop under {head!r}: nothing to establish the steady state "
                         f"against, so no answer is offered")
    lo, _hi = span
    start = setup_new_tile(lines)
    if start is None:
        raise ModelError("no 'Begin setupNewTile' marker: where a tile's own state starts cannot be "
                         "read off the kernel, and walking from the top adds argument-loading paths "
                         "that touch no LDS")
    prologue = decode_range(lines, start, lo - 1, allow_branches=True)
    body = decode_range(lines, lo, _hi, allow_branches=True)
    bits, alias = select_bits(prologue, body), fused_aliases(prologue, body)

    def make_state():
        ex = Executor(wave=wave, ring=ring, seed={}, select_bits=bits)
        ex.alias.update(alias)
        return ex

    runs, common = walk_all_entries(lines, make_state, ring=ring, establish_before=lo,
                                    entry_line=start, **kw)
    if not runs:
        raise ModelError("no entry buffer hypothesis could be walked, so the pointers cannot be "
                         "placed against each other and nothing follows")
    floor = min(r.unwritten or 0 for _seed, r in runs)
    if floor:
        raise ModelError(f"the best entry buffer hypothesis still reads storage nothing filled "
                         f"{floor} times, so it is refuted by the kernel it was chosen for and no "
                         f"finding rests on it")
    by_key, per_seed = {}, []
    for _seed, r in runs:
        if not getattr(r, "consistent", True):
            continue
        found = r.unique()
        per_seed.append(len(found))
        for path, f in found:
            at, other = getattr(f.at, "inst", None), getattr(f.other, "inst", None)
            by_key[(f.kind, at, other)] = (path, f)
    return Report([by_key[k][1] for k in common if k in by_key], floor,
                  len(per_seed), len(runs), tuple(per_seed))

def gir_disagreement(ex, hazards):
    """Tensors the run exercised a dependence on that the GIR hazard set does not name, and back.

    `hazards` is any iterable of objects with `.producer.operand` / `.consumer.operand`, i.e. an
    `FrameHazardSet`.  A dependence the machine executed but the model never predicted is a hole in
    the model; a model edge on a tensor the machine never touched is dead weight."""
    modelled = {h.producer.operand for h in hazards} | {h.consumer.operand for h in hazards}
    executed = {t for (t, _b) in observed_pairs(ex)}
    return {"executed_not_modelled": sorted(executed - modelled),
            "modelled_not_executed": sorted(modelled - executed)}

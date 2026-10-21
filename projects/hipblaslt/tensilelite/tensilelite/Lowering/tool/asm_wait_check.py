# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
Symbolic execution of the emitted main loop against the hardware completion counters.

Each counter is an in-order FIFO of the instructions that landed on it, and `s_wait_<c> N` retires
from the front until at most N remain -- so whether a producer has completed is a question of how
many instructions issued after it, never of what a comment says.  Dependencies come from the
operands: a `ds_load` defines VGPRs and a `wmma` uses them.  The `<GIR: ...>` tag is carried along
only to name which GIR node a finding belongs to.
"""

from __future__ import annotations

import re
from collections import deque

from ..gir_tag import GIR_TAG_RE

#: `s_wait_dscnt 3`.  gfx1250 splits the counters; there is no lgkmcnt.
_WAIT = re.compile(r"^\s*s_wait_(tensorcnt|dscnt|loadcnt|storecnt|asynccnt|kmcnt|xcnt)\s+(\d+)")
#: the loop head is a label DEFINITION, not any line mentioning the name.
_LABEL = re.compile(r"^\s*([A-Za-z_]\w*)\s*:")
#: `v[vgprValuA_X0_I0+0+0+0:vgprValuA_X0_I0+0+0+0+15]`, `v[vgprValuMXSA_X0_I0+0]`
_VREG = re.compile(r"v\[\s*(vgpr\w+)((?:[+-]\d+)*)\s*(?::\s*vgpr\w+((?:[+-]\d+)*))?\s*\]")
#: the emitter owns the tag grammar; a second spelling of it here is how three read-side checks
#: silently matched nothing.
_GIR = GIR_TAG_RE
#: `sync LDS0` or `sync LDS [0, 1, 3]` -- the emitter writes both forms.  This is the LDS a memory
#: instruction touches, and pairing a copy with a read on anything coarser invents hazards.
_SYNC = re.compile(r"sync LDS\s*\[([0-9,\s]*)\]|sync LDS(\d+)")


def _buffers(text):
    out = set()
    for grp, one in _SYNC.findall(text):
        if one:
            out.add(int(one))
        else:
            out.update(int(t) for t in grp.split(",") if t.strip())
    return frozenset(out)

#: which FIFO each instruction lands on.  A wmma lands on none; it only consumes.
COUNTER = {"tensor_load": "tensorcnt", "ds_load": "dscnt", "ds_store": "dscnt",
           "global_load": "loadcnt", "global_store": "storecnt"}


class Inst:
    """One decoded instruction: what it lands on, what it defines, what it uses."""

    __slots__ = ("pos", "line", "op", "kind", "defs", "uses", "gir", "text", "buffers")

    def __init__(self, pos, line, op, kind, defs, uses, gir, text, buffers=frozenset()):
        self.pos, self.line, self.op, self.kind = pos, line, op, kind
        self.defs, self.uses, self.gir, self.text = defs, uses, gir, text
        self.buffers = buffers

    @property
    def counter(self):
        return COUNTER.get(self.kind)

    def __repr__(self):
        return f"{self.op}@{self.line}" + (f" <{self.gir}>" if self.gir else "")


def _offsets(expr):
    return sum(int(t) for t in re.findall(r"[+-]\d+", expr or ""))


def _regs(text):
    """The `(name, index)` cells an operand list names, expanding `v[x+a:x+a+n]`."""
    out = set()
    for name, lo, hi in _VREG.findall(text):
        a, b = _offsets(lo), _offsets(hi) if hi else _offsets(lo)
        out.update((name, i) for i in range(min(a, b), max(a, b) + 1))
    return frozenset(out)


def _kind(op):
    if op.startswith("tensor_load"):
        return "tensor_load"
    if op.startswith("ds_load") or op.startswith("ds_read"):
        return "ds_load"
    if op.startswith("ds_store") or op.startswith("ds_write"):
        return "ds_store"
    if op.startswith("buffer_load") or op.startswith("global_load"):
        return "global_load"
    if op.startswith("buffer_store") or op.startswith("global_store"):
        return "global_store"
    if op.startswith("v_wmma") or op.startswith("v_swmmac"):
        return "wmma"
    if op.startswith("s_barrier"):
        return "barrier"
    return None


def loop_body(lines, head="label_LoopBeginL"):
    """`(first, last)` line indices of the rolled body: the label DEFINITION to its back edge."""
    start = None
    for i, line in enumerate(lines):
        m = _LABEL.match(line)
        if m and m.group(1) == head:
            start = i
            break
    if start is None:
        return None
    back = re.compile(r"\s*s_cbranch\w*\s+" + re.escape(head) + r"\b")
    end = next((i for i in range(start + 1, len(lines)) if back.match(lines[i])), None)
    return None if end is None else (start + 1, end)


#: `<GIR: steady waitcnt tensorcnt=0,dscnt=2 hazard=RAW from=A frames=1>`.  The tag rides the
#: instruction it guards, so one instruction can carry several.
_MARK = re.compile(r"<GIR: \S+ waitcnt ([^<>]*)>")
_MARK_CNT = re.compile(r"(tensorcnt|dscnt)=(\d+)")


class WaitMark:
    """A GIR `waitcnt` tag: the residual GIR computed for each counter at this point."""

    __slots__ = ("line", "claims", "text")

    def __init__(self, line, claims, text):
        self.line, self.claims, self.text = line, claims, text

    def __repr__(self):
        return "mark@%d %s" % (self.line, self.claims)


def decode(path, head="label_LoopBeginL", marks=False):
    """The loop body as `[Inst | ('wait', counter, n, line)]`, in issue order.

    `marks` additionally keeps GIR's own `waitcnt` tags, which are comments and so invisible to
    every other reader here."""
    lines = open(path, errors="replace").read().splitlines()
    span = loop_body(lines, head)
    if span is None:
        raise ValueError(f"{path}: no rolled body for {head!r}")
    out, pos = [], 0
    for i in range(*span):
        raw = lines[i]
        code = raw.split("//")[0]
        if marks:
            for g in _MARK.finditer(raw):
                out.append(WaitMark(i + 1,
                                    {c: int(n) for c, n in _MARK_CNT.findall(g.group(1))},
                                    g.group(1).strip()))
        m = _WAIT.match(code)
        if m:
            out.append(("wait", m.group(1), int(m.group(2)), i + 1))
            continue
        toks = code.split()
        if not toks:
            continue
        kind = _kind(toks[0])
        if kind is None:
            continue
        _op, _, args = code.strip().partition(" ")
        dst, _, rest = args.partition("],")
        g = _GIR.search(raw)
        out.append(Inst(pos, i + 1, toks[0], kind,
                        _regs(dst + "]") if kind in ("ds_load", "global_load") else frozenset(),
                        _regs(rest) if kind == "wmma" else frozenset(),
                        g.group(1) if g else None, code.strip(), _buffers(raw)))
        pos += 1
    return out


class Machine:
    """The completion counters as in-order FIFOs, plus which instruction last defined a register."""

    def __init__(self):
        self.fifo = {c: deque() for c in set(COUNTER.values())}
        self.defined_by = {}

    def issue(self, inst):
        for r in inst.defs:
            self.defined_by[r] = inst
        if inst.counter:
            self.fifo[inst.counter].append(inst)

    def wait(self, counter, n):
        """Retire from the front until at most `n` remain.  Returns what retired."""
        q = self.fifo.get(counter)
        if q is None:
            return []
        out = []
        while len(q) > n:
            out.append(q.popleft())
        return out

    def outstanding(self, counter):
        return list(self.fifo.get(counter, ()))

    def pending_def(self, reg):
        """The instruction that defined `reg`, if it has not yet retired from its FIFO."""
        d = self.defined_by.get(reg)
        return d if d is not None and d.counter and d in self.fifo[d.counter] else None


def run(program, trips=3):
    """Execute `trips` iterations.  Returns `(violations, machine)` for the final trip.

    A violation is a `wmma` reading a register whose defining `ds_load` is still outstanding --
    exact, from the operands and the counter, with no reliance on any comment.
    """
    m, bad = Machine(), []
    for trip in range(trips):
        for item in program:
            if isinstance(item, tuple):
                m.wait(item[1], item[2])
                continue
            if item.kind == "wmma":
                for r in sorted(item.uses):
                    d = m.pending_def(r)
                    if d is not None and trip == trips - 1:
                        bad.append((item, r, d, len(m.outstanding("dscnt"))))
            m.issue(item)
    return bad, m


def unwaited_wmma(path, trips=3, head="label_LoopBeginL"):
    """`[(wmma, reg, defining_load, dscnt_depth)]` -- a wmma reading an unlanded register."""
    return run(decode(path, head), trips=trips)[0]


def tensor_state_at_reads(path, trips=3, head="label_LoopBeginL"):
    """`[(ds_load, [outstanding tensor_loads])]` -- the raw material for GIR to adjudicate.

    Which LDS bytes a TDM load fills is not in its operands -- they name a descriptor -- so this
    reports the tensor FIFO at each read and leaves the pairing to the GIR hazard set."""
    program = decode(path, head)
    m, out = Machine(), []
    for trip in range(trips):
        for item in program:
            if isinstance(item, tuple):
                m.wait(item[1], item[2])
                continue
            if item.kind == "ds_load" and trip == trips - 1:
                out.append((item, m.outstanding("tensorcnt")))
            m.issue(item)
    return out


#: `read A[tile=0,k=1]->X1` -- the GIR read names the register buffer it fills.
_READ_TAG = re.compile(r"read (\w+)\[([^\]]*)\]->X(\d+)")
#: `copy A+B gen=0` -- the GIR copy names its operands and the generation it fills.
_COPY_TAG = re.compile(r"copy ([\w+]+) gen=(\d+)")
#: `fence A+B+MXSA+MXSB (RAW,WAR,WAW)` -- what a GIR fence claims to order.
_FENCE_TAG = re.compile(r"fence ([\w+]+)(?: \(([^)]*)\))?")
#: `vgprValuA_X1_I0` -- the register name carries the buffer index the ring rotates through.
_VBUF = re.compile(r"vgprValu(\w+?)_X(\d+)_I(\d+)")


def wrong_generation(path, trips=3, head="label_LoopBeginL"):
    """`[(wmma, reg, load, said, used)]` -- a wmma reading a buffer the load did not fill.

    The register name carries the ring buffer (`..._X1_...`) and the GIR read tag says which
    buffer that load was meant to fill, so the two must agree or the wmma is on the wrong
    generation."""
    program = decode(path, head)
    m, bad = Machine(), []
    for trip in range(trips):
        for item in program:
            if isinstance(item, tuple):
                m.wait(item[1], item[2])
                continue
            if item.kind == "wmma" and trip == trips - 1:
                for reg in sorted(item.uses):
                    d = m.defined_by.get(reg)
                    tag = _READ_TAG.search(d.gir or "") if d is not None else None
                    buf = _VBUF.search(reg[0])
                    if tag and buf and int(tag.group(3)) != int(buf.group(2)):
                        bad.append((item, reg, d, int(tag.group(3)), int(buf.group(2))))
            m.issue(item)
    return bad


def overwritten_early(path, trips=3, head="label_LoopBeginL"):
    """`[(tdm_load, [reads still outstanding on the SAME LDS], fence_between)]`.

    Paired on the buffers, never on issue order alone: a refill for the next generation running
    while this one's reads are outstanding is what a pipeline IS, and on disjoint LDS it orders
    nothing.  `fence_between` is the GIR fence tag seen since the oldest of them, if any."""
    program = decode(path, head)
    m, out, last_fence = Machine(), [], None
    for trip in range(trips):
        for item in program:
            if isinstance(item, tuple):
                m.wait(item[1], item[2])
                continue
            if item.kind == "barrier":
                last_fence = item.gir or last_fence
            if item.kind == "tensor_load" and trip == trips - 1:
                live = [r for r in m.outstanding("dscnt")
                        if r.kind == "ds_load" and (r.buffers & item.buffers)]
                if live:
                    out.append((item, live, last_fence))
            m.issue(item)
    return out


def cross_agent_cover(path, head="label_LoopBeginL"):
    """`(fences, operands_read, operands_copied)` -- what the emitted fences claim to order.

    Every operand that is both copied and read in the body needs a fence naming it; an operand
    read but never named by any fence has no cross-wave ordering at all."""
    program = decode(path, head)
    fences, read_ops, copy_ops = [], set(), set()
    for item in program:
        if isinstance(item, tuple):
            continue
        tag = item.gir or ""
        if item.kind == "barrier":
            f = _FENCE_TAG.search(tag)
            if f:
                fences.append((frozenset(f.group(1).split("+")),
                               frozenset((f.group(2) or "").split(",")) - {""}))
        r = _READ_TAG.search(tag)
        if r:
            read_ops.add(r.group(1))
        c = _COPY_TAG.search(tag)
        if c:
            copy_ops.update(c.group(1).split("+"))
    named = frozenset().union(*[f[0] for f in fences]) if fences else frozenset()
    return fences, read_ops, copy_ops, (read_ops & copy_ops) - named


def segments(program):
    """The body split at barriers.  Within a segment waves interleave freely -- a barrier is the
    only thing that orders one wave's memory against another's."""
    out, cur = [], []
    for item in program:
        if not isinstance(item, tuple) and item.kind == "barrier":
            out.append(cur)
            cur = []
            continue
        cur.append(item)
    out.append(cur)
    return out


def unsynchronised_cross_wave(path, head="label_LoopBeginL"):
    """`[(segment, operand, copy, reads)]` -- an operand refilled and read in ONE barrier interval.

    Waves interleave inside a segment, so one wave's TDM refill of an operand can land while
    another wave is still reading it.  Only a barrier separates them, and inside a segment there
    is none: this is the cross-agent WAR the token layer exists to force a fence for."""
    out = []
    for n, seg in enumerate(segments(decode(path, head))):
        copies, reads = {}, {}
        for item in seg:
            if isinstance(item, tuple):
                continue
            c = _COPY_TAG.search(item.gir or "")
            if c:
                for op in c.group(1).split("+"):
                    copies.setdefault(op, []).append(item)
            r = _READ_TAG.search(item.gir or "")
            if r:
                reads.setdefault(r.group(1), []).append(item)
        for op in sorted(set(copies) & set(reads)):
            out.append((n, op, copies[op], reads[op]))
    return out


_BRANCH = re.compile(r"^\s*(s_cbranch\w*|s_branch)\s+(\S+)")
_END = re.compile(r"^\s*s_endpgm\b")


def decode_all(path):
    """`(blocks, succs, order)` -- the WHOLE kernel as a CFG of decoded basic blocks.

    A block starts at a label definition and ends at a branch, an endpgm, or the next label.
    `s_cbranch` has two successors (taken and fallthrough); `s_branch` one; `s_endpgm` none."""
    lines = open(path, errors="replace").read().splitlines()
    starts, label_at = [0], {}
    for i, line in enumerate(lines):
        m = _LABEL.match(line)
        if m:
            label_at[m.group(1)] = i
            starts.append(i)
        elif _BRANCH.match(line) or _END.match(line):
            starts.append(i + 1)
    starts = sorted(set(s for s in starts if s < len(lines)))
    bounds = list(zip(starts, starts[1:] + [len(lines)]))
    blocks, succs, order = {}, {}, []
    for bid, (lo, hi) in enumerate(bounds):
        blocks[bid] = _decode_range(lines, lo, hi)
        order.append(bid)
    at = {lo: bid for bid, (lo, _hi) in enumerate(bounds)}
    for bid, (lo, hi) in enumerate(bounds):
        nxt, tail = [], None
        for i in range(lo, hi):
            if _END.match(lines[i]):
                tail = "end"
                break
            m = _BRANCH.match(lines[i])
            if m:
                tgt = label_at.get(m.group(2))
                if tgt is not None and tgt in at:
                    nxt.append(at[tgt])
                tail = m.group(1)
                break
        if tail != "end" and tail != "s_branch" and bid + 1 < len(bounds):
            nxt.append(bid + 1)
        succs[bid] = tuple(dict.fromkeys(nxt))
    return blocks, succs, order


def _decode_range(lines, lo, hi):
    out, pos = [], 0
    for i in range(lo, hi):
        raw = lines[i]
        code = raw.split("//")[0]
        m = _WAIT.match(code)
        if m:
            out.append(("wait", m.group(1), int(m.group(2)), i + 1))
            continue
        toks = code.split()
        if not toks:
            continue
        kind = _kind(toks[0])
        if kind is None:
            continue
        _op, _, args = code.strip().partition(" ")
        dst, _, rest = args.partition("],")
        g = _GIR.search(raw)
        out.append(Inst(pos, i + 1, toks[0], kind,
                        _regs(dst + "]") if kind in ("ds_load", "global_load") else frozenset(),
                        _regs(rest) if kind == "wmma" else frozenset(),
                        g.group(1) if g else None, code.strip(), _buffers(raw)))
        pos += 1
    return out


def paths(succs, entry=0, max_visits=2, cap=4000):
    """Every path from `entry`, each block entered at most `max_visits` times (so a loop is seen
    in steady state), bounded by `cap` so a wide CFG cannot explode."""
    out, stack = [], [(entry, (entry,), {entry: 1})]
    while stack and len(out) < cap:
        bid, route, seen = stack.pop()
        nxt = [s for s in succs.get(bid, ()) if seen.get(s, 0) < max_visits]
        if not nxt:
            out.append(route)
            continue
        for s in nxt:
            stack.append((s, route + (s,), {**seen, s: seen.get(s, 0) + 1}))
    return out


def execute_all_paths(path, max_visits=2, cap=400):
    """`[(route, violations)]` -- every path carrying TDM/DS/WMMA, executed on the counters."""
    blocks, succs, _order = decode_all(path)
    interesting = {b for b, prog in blocks.items()
                   if any(not isinstance(i, tuple) and i.kind in
                          ("tensor_load", "ds_load", "ds_store", "wmma") for i in prog)}
    out = []
    for route in paths(succs, cap=cap):
        if not (set(route) & interesting):
            continue
        m, bad = Machine(), []
        for bid in route:
            for item in blocks[bid]:
                if isinstance(item, tuple):
                    m.wait(item[1], item[2])
                    continue
                if item.kind == "wmma":
                    for r in sorted(item.uses):
                        d = m.pending_def(r)
                        if d is not None:
                            bad.append((item, r, d))
                m.issue(item)
        if bad:
            out.append((route, bad))
    return out


class Wave:
    """One wave's position in the program plus its own completion counters."""

    def __init__(self, wid, program):
        self.wid, self.program, self.pc = wid, program, 0
        self.machine = Machine()
        self.reading = {}          # (operand, gen) -> the ds_load still outstanding for it

    @property
    def done(self):
        return self.pc >= len(self.program)

    def at_barrier(self):
        """Only a barrier WAIT blocks.  `s_barrier_signal` marks arrival and the wave runs on --
        modelling it as a join over-synchronises and hides every race the split barrier allows."""
        item = self.program[self.pc] if not self.done else None
        return (item is not None and not isinstance(item, tuple)
                and item.kind == "barrier" and "wait" in item.op)

    def at_signal(self):
        item = self.program[self.pc] if not self.done else None
        return (item is not None and not isinstance(item, tuple)
                and item.kind == "barrier" and "wait" not in item.op)

    def step(self):
        """Execute one instruction.  Returns `(kind, payload)` for the arbiter."""
        item = self.program[self.pc]
        self.pc += 1
        if isinstance(item, tuple):
            for done in self.machine.wait(item[1], item[2]):
                for key, rd in list(self.reading.items()):
                    if rd is done:
                        self.reading.pop(key)
            return ("wait", item)
        self.machine.issue(item)
        if item.kind == "ds_load":
            for b in item.buffers:
                self.reading[b] = item
        return (item.kind, item)


def _copy_targets(inst):
    """The LDS buffers a TDM copy fills.

    The BUFFERS, not the operand from the GIR tag: a `copy A+B` and a `read B` can name entirely
    disjoint LDS, and pairing on the operand invents a hazard the scheduler is right to ignore."""
    return sorted(inst.buffers)


def multiwave_races(path, waves=2, head="label_LoopBeginL", trips=2, schedules=None):
    """`[(writer_wave, copy, reader_wave, read)]` -- a refill landing under another wave's read.

    Waves execute the same program and are ordered ONLY by barriers: at a barrier every wave
    stops until all arrive.  Between barriers the arbiter tries each interleaving in `schedules`
    (default: run each wave to its next barrier in turn, in both orders), and a race is a TDM
    copy of `(operand, gen)` issued by one wave while another wave still has an outstanding
    `ds_load` of that operand -- the cross-agent WAR the fence exists to prevent.
    """
    program = decode(path, head) * trips
    # A segment has no barrier in it, so waves may be arbitrarily SKEWED inside one: lockstep
    # round-robin never overlaps a wave's reads with another's refill and finds nothing.
    orders = schedules or [list(range(waves)), list(reversed(range(waves)))]
    skews = [0, 1, 2, 4, 8, 16, 32, 64]
    seen, out = set(), []
    for order in orders:
      for skew in skews:
        ws = [Wave(i, program) for i in range(waves)]
        for _ in range(skew):                      # let the leader run ahead inside the segment
            lead = ws[order[0]]
            if lead.done or lead.at_barrier():
                break
            kind, item = lead.step()
            if kind == "tensor_load":
                for op in _copy_targets(item):
                    for other in ws:
                        if other is not lead and other.reading.get(op) is not None:
                            rd = other.reading[op]
                            key = (lead.wid, item.line, other.wid, rd.line)
                            if key not in seen:
                                seen.add(key)
                                out.append((lead.wid, item, other.wid, rd))
        while not all(w.done for w in ws):
            progressed = False
            # ONE instruction per wave per round: between barriers the waves interleave, and
            # running each to its own barrier would serialise away every race there is.
            for i in order:
                w = ws[i]
                if w.done or w.at_barrier():
                    continue
                kind, item = w.step()
                progressed = True
                if kind != "tensor_load":
                    continue
                for op in _copy_targets(item):
                    for other in ws:
                        if other is w:
                            continue
                        rd = other.reading.get(op)
                        if rd is None:
                            continue
                        key = (w.wid, item.line, other.wid, rd.line)
                        if key not in seen:
                            seen.add(key)
                            out.append((w.wid, item, other.wid, rd))
            if all(w.done or w.at_barrier() for w in ws):
                for w in ws:                       # every wave has arrived: release the barrier
                    if not w.done:
                        w.pc += 1
                progressed = True
            if not progressed:
                break
    return out


def multiwave_races_kernel(path, waves=2, cap=200, max_visits=2):
    """`multiwave_races` over the WHOLE kernel, not just the rolled body.

    Three quarters of the memory traffic is in the prologue, the peel and the drains, and a race
    there is as fatal as one in the loop.  Each CFG path is linearised and executed in turn."""
    blocks, succs, _order = decode_all(path)
    interesting = {b for b, prog in blocks.items()
                   if any(not isinstance(i, tuple) and i.kind in
                          ("tensor_load", "ds_load", "ds_store") for i in prog)}
    seen, out = set(), []
    for route in paths(succs, cap=cap, max_visits=max_visits):
        if not (set(route) & interesting):
            continue
        program = [i for b in route for i in blocks[b]]
        for w, cp, o, rd in multiwave_races_program(program, waves=waves):
            key = (w, cp.line, o, rd.line)
            if key not in seen:
                seen.add(key)
                out.append((route, w, cp, o, rd))
    return out


def multiwave_races_program(program, waves=2):
    """The multi-wave arbiter over an already-linearised program."""
    orders = [list(range(waves)), list(reversed(range(waves)))]
    skews = [0, 1, 2, 4, 8, 16, 32, 64]
    seen, out = set(), []
    for order in orders:
        for skew in skews:
            ws = [Wave(i, program) for i in range(waves)]

            def note(lead, item):
                for op in _copy_targets(item):
                    for other in ws:
                        if other is lead:
                            continue
                        rd = other.reading.get(op)
                        if rd is None:
                            continue
                        key = (lead.wid, item.line, other.wid, rd.line)
                        if key not in seen:
                            seen.add(key)
                            out.append((lead.wid, item, other.wid, rd))

            for _ in range(skew):
                lead = ws[order[0]]
                if lead.done or lead.at_barrier():
                    break
                kind, item = lead.step()
                if kind == "tensor_load":
                    note(lead, item)
            while not all(w.done for w in ws):
                progressed = False
                for i in order:
                    w = ws[i]
                    if w.done or w.at_barrier():
                        continue
                    kind, item = w.step()
                    progressed = True
                    if kind == "tensor_load":
                        note(w, item)
                if all(w.done or w.at_barrier() for w in ws):
                    for w in ws:
                        if not w.done:
                            w.pc += 1
                    progressed = True
                if not progressed:
                    break
    return out

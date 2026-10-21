# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
GIR's computed waitcnt residuals against the emitted assembly, two ways.

Both sides are in the `.s`: GIR tags each residual onto the instruction it guards, and the hardware
counters are an in-order FIFO that `s_wait_<c> N` retires down to `N`.

COMPARE reads what the emitted code has outstanding where our mark stands.  It is DESCRIPTIVE, not
an oracle: a wait that is merely sufficient never proves what was necessary, and the emitted counts
are not currently a trustworthy reference.  Read it as a distribution, not a verdict.

SUBSTITUTE is the oracle.  It throws away every emitted wait, applies ours instead, and asks
whether a wmma then reads a register whose `ds_load` has not landed.  That is decided from the
operands, so it holds for `dscnt`; the `tensorcnt` side needs an LDS address model this does not
have.  The same run with the emitted waits is the control and must report nothing.

It answers one question: would GIR's residuals ALONE cover the RAW, in the order the kernel is
actually emitted in.  It consumes the emitted ORDER, never an emitted wait value, so it stands
whatever those values are.  A failure says a residual computed over GIR's order does not transfer
to the order the kernel ships in -- `n` counts issues, and the two orders differ.

    python -m Tensile.Lowering.tool.waitcnt_compare a.s b.s ...
"""

from __future__ import annotations

import argparse
import collections
import os
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor

from .asm_wait_check import Machine, WaitMark, decode

STRICTER, MATCH, LOOSER = "stricter", "match", "looser"


class Cell:
    """One counter of one mark: what GIR claimed, and what the assembly had outstanding there."""

    __slots__ = ("line", "counter", "ours", "theirs", "text")

    def __init__(self, line, counter, ours, theirs, text):
        self.line, self.counter = line, counter
        self.ours, self.theirs, self.text = ours, theirs, text

    @property
    def verdict(self):
        return STRICTER if self.ours < self.theirs else MATCH if self.ours == self.theirs else LOOSER

    def __repr__(self):
        return "%s@%d %s ours=%d theirs=%d" % (self.verdict, self.line, self.counter,
                                               self.ours, self.theirs)


def _walk(program, trips, apply_ours, on_issue):
    """Run `trips` iterations, applying either our marks or the emitted waits."""
    machine, pending = Machine(), []
    for trip in range(trips):
        for item in program:
            if isinstance(item, tuple):
                if not apply_ours:
                    machine.wait(item[1], item[2])
                continue
            if isinstance(item, WaitMark):
                pending.append(item)
                continue
            # A pending mark stands for a wait that PRECEDES this instruction, so it has to be in
            # effect before the instruction is judged.  Applying it afterwards made every wait
            # sitting immediately before its wmma -- the correct, common shape -- read as absent.
            if apply_ours:
                for mark in pending:
                    for counter, n in mark.claims.items():
                        machine.wait(counter, n)
            on_issue(trip, machine, pending, item)
            pending = []
            machine.issue(item)
    return machine


def compare(path, trips=3, head="label_LoopBeginL"):
    """`[Cell]` for the steady trip -- earlier trips are still filling the loop-carried FIFOs."""
    out = []

    def at_issue(trip, machine, pending, _item):
        if trip != trips - 1:
            return
        for mark in pending:
            for counter, ours in mark.claims.items():
                out.append(Cell(mark.line, counter, ours,
                                len(machine.outstanding(counter)), mark.text))

    _walk(decode(path, head, marks=True), trips, False, at_issue)
    return out


def substitute(path, trips=3, head="label_LoopBeginL", ours=True):
    """`[(wmma, reg, load, needed, left)]` -- a wmma reading a register still in flight.

    `needed` is the residual that would have forced `load` out here, counted in the EMITTED order;
    `left` is what the applied wait actually permitted.  `left > needed` is the whole failure."""
    bad = []

    def at_issue(trip, machine, _pending, item):
        if item.kind != "wmma" or trip != trips - 1:
            return
        for reg in sorted(item.uses):
            held = machine.pending_def(reg)
            if held is None:
                continue
            queue = list(machine.outstanding(held.counter))
            bad.append((item, reg, held, len(queue) - 1 - queue.index(held), len(queue)))

    _walk(decode(path, head, marks=True), trips, ours, at_issue)
    return bad


def _one(path):
    try:
        wmmas = sum(1 for i in decode(path) if getattr(i, "kind", None) == "wmma")
        return (path, compare(path), substitute(path, ours=True),
                substitute(path, ours=False), wmmas, None)
    except (ValueError, OSError) as e:
        return path, [], [], [], 0, "%s: %s" % (type(e).__name__, e)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+", help="`.s` files, or a directory to walk for them")
    ap.add_argument("-j", "--jobs", type=int, default=min(64, (os.cpu_count() or 8)))
    ap.add_argument("-v", "--verbose", action="store_true", help="list every kernel, not the first")
    a = ap.parse_args(argv)
    # A corpus sweep is thousands of kernels, which is more than one argv can carry.
    a.paths = [str(f) for p in a.paths
               for f in (sorted(pathlib.Path(p).rglob("*.s")) if os.path.isdir(p) else [p])]

    tally, shapes = collections.Counter(), collections.Counter()
    gaps = collections.Counter()
    skipped, unmarked = [], 0
    broken_ours, broken_theirs, compared = [], [], 0
    wmma_total, wmma_bad, carried = 0, 0, 0
    with ProcessPoolExecutor(max_workers=a.jobs) as pool:
        for path, cells, ours, theirs, wmmas, err in pool.map(_one, a.paths, chunksize=8):
            if err:
                skipped.append((path, err))
                continue
            if not cells:
                unmarked += 1
                continue
            compared += 1
            wmma_total += wmmas
            wmma_bad += len({w.line for w, _r, _l, _n, _k in ours})
            # the defining load sits AFTER its wmma in the text: GIR's order is not this one.
            carried += len({w.line for w, _r, ld, _n, _k in ours if ld.line > w.line})
            for _w, _r, _l, need, left in ours:
                gaps[left - need] += 1
            for c in cells:
                tally[c.verdict] += 1
                shapes[(c.verdict, c.counter, c.ours, c.theirs)] += 1
            if ours:
                broken_ours.append((path, ours))
            if theirs:
                broken_theirs.append((path, theirs))

    print("kernels: %d compared, %d with no GIR mark, %d skipped"
          % (compared, unmarked, len(skipped)))
    print("cells:   %d  " % sum(tally.values())
          + "  ".join("%s=%d" % (v, tally[v]) for v in (STRICTER, MATCH, LOOSER)))
    for (verdict, counter, ours_n, theirs_n), n in shapes.most_common(20):
        print("  %-8s %-9s ours=%-3d emitted=%-3d  x%d" % (verdict, counter, ours_n, theirs_n, n))

    print("\nSUBSTITUTE (our marks as the only waits): %d/%d kernel(s), %d/%d wmma(s) read an "
          "unlanded register" % (len(broken_ours), compared, wmma_bad, wmma_total))
    print("           of those, %d wmma(s) are fed by a load the text places AFTER them" % carried)
    print("CONTROL    (emitted waits):                %d/%d -- the tool's own check"
          % (len(broken_theirs), compared))
    print("           left-minus-needed, the size of the miss: %s"
          % "  ".join("+%d x%d" % (g, n) for g, n in sorted(gaps.items())[:12]))
    for label, group in (("ours", broken_ours), ("control", broken_theirs)):
        for path, bad in group[: (len(group) if a.verbose else 3)]:
            print("  [%s] %s" % (label, os.path.basename(path)))
            for wmma, reg, load, need, left in bad[:3]:
                print("      %s reads %s still held by %s (needed %d, left %d)"
                      % (wmma, "%s+%d" % reg, load, need, left))
    for path, err in skipped[:5]:
        print("skip %s: %s" % (os.path.basename(path), err), file=sys.stderr)
    return 1 if broken_ours or broken_theirs else 0


if __name__ == "__main__":
    sys.exit(main())

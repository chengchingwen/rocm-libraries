# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
Path exploration: every branch combination, each carrying its own machine state.

A hazard on ANY reachable path is a hazard, so a straight-line trace can only ever be a lower
bound -- it answers for one arm of every branch it walked past.  This walks the CFG instead: a
conditional forks the state, a loop is re-entered until its state stops changing, and the findings
are the union over all paths.
"""

from __future__ import annotations

import copy

from .cfg import build_cfg, entry_label
from .machine import ModelError

#: A path may not revisit one block more times than this -- a runaway loop is a defect, not an answer.
_MAX_VISITS = 8
#: Total blocks executed across the whole walk, so an exponential fan-out fails loudly.
_MAX_STEPS = 20000


def _summary(ex):
    """What decides the future: everything a later hazard can be found from.

    Rotation phase and counter depth are not enough -- the checks read which loads are outstanding
    and which register each was destined for, so two states equal only in phase still differ in what
    they will report.  Summarising on less merges paths and drops findings."""
    return (tuple(sorted(ex.parity.items())),
            tuple(sorted(ex.region.items())),
            tuple(sorted((c, tuple(e.inst.line for e in q)) for c, q in ex.fifo.items())),
            tuple(sorted((k, tuple(sorted(v.items())))
                         for k, v in ex.steps.items() if any(v.values()))),
            tuple(sorted((k, tuple((e.inst.line, e.landed) for e in v))
                         for k, v in getattr(ex, "writers", {}).items())),
            tuple(sorted((r, e.inst.line, e.landed)
                         for r, e in getattr(ex, "defined_by", {}).items())))


class PathResult:
    """Findings over the whole graph, each keyed by where it was found."""

    def __init__(self):
        self.findings = []          # (path tuple, Finding)
        self.unknown = set()        # select keys whose entry buffer the walk could not derive
        self.entry_findings = []    # what the establishing blocks reported, for reporting only
        self.unwritten = None       # reads of storage nothing filled: how refuted this seed is
        self.consistent = True      # this seed is among the least refuted
        self.visited = set()        # (label, summary)
        self.steps = 0
        self.blocks_seen = set()

    def add(self, path, findings):
        for f in findings:
            self.findings.append((path, f))

    def unique(self):
        """One entry per distinct (kind, producer line, consumer line)."""
        out = {}
        for path, f in self.findings:
            at = getattr(f.at, "inst", None)
            other = getattr(f.other, "inst", None)
            key = (f.kind, getattr(at, "line", None), getattr(other, "line", None))
            out.setdefault(key, (path, f))
        return [v for _k, v in sorted(out.items(), key=lambda kv: (str(kv[0][0]), kv[0][1] or 0))]

    def kinds(self):
        c = {}
        for _k, f in self.unique():
            c[f.kind] = c.get(f.kind, 0) + 1
        return c


def walk(lines, make_state, entry=None, stop=None, max_steps=_MAX_STEPS, establish_before=None,
         entry_line=0):
    """Execute every reachable path; return a `PathResult`.

    `make_state` builds a fresh Executor.  Blocks are run in full; a conditional pushes BOTH
    successors with independent copies of the state, so no arm is assumed.

    `establish_before` is the line at which the steady state begins: up to it a pointer may be
    BUILT by address arithmetic this machine cannot evaluate, which is recorded as an unknown
    rather than refused."""
    cfg = build_cfg(lines, entry=entry_line, stop=stop)
    start = entry or entry_label(cfg, lines)
    res = PathResult()
    work = [(start, make_state(), (start,), {})]
    while work:
        label, state, path, visits = work.pop()
        if label not in cfg:
            continue
        res.steps += 1
        if res.steps > max_steps:
            raise ModelError("path walk exceeded %d block executions: the graph is fanning out "
                             "faster than the state converges" % max_steps)
        block = cfg[label]
        res.blocks_seen.add(label)
        state.findings = []
        state.establishing = establish_before is not None and block.lo < establish_before
        state.run(block.body)
        if state.establishing:
            res.entry_findings.extend(state.findings)
        res.add(path, state.findings)
        state.findings = []
        key = (label, _summary(state))
        if key in res.visited:
            continue                       # this state already continued from here
        res.visited.add(key)
        res.unknown |= state.unknown
        succs = block.succs
        for i, s in enumerate(succs):
            seen = dict(visits)
            seen[s] = seen.get(s, 0) + 1
            if seen[s] > _MAX_VISITS:
                continue
            nxt = copy.deepcopy(state) if i + 1 < len(succs) else state
            work.append((s, nxt, path + (s,), seen))
    return res


def walk_all_entries(lines, make_state, ring=2, max_unknown=8, **kw):
    """`(PathResult per seed, findings common to every seed)`.

    ONLY THE RELATIVE PHASE IS OBSERVABLE between a read pointer and its descriptor, so the entry
    buffer of each pointer the prologue builds is a hypothesis, not a fact.  Every assignment is
    walked in full and only findings common to the surviving ones are reported.

    A WRONG ENTRY BUFFER AIMS EVERY READ AT STORAGE NOTHING FILLED, so `read_unwritten` over the
    whole run counts how far a hypothesis is refuted and the least-refuted survive.  Scoring on the
    prologue alone instead picks seeds the loop then contradicts, which is worse than no answer.

    THE EVIDENCE IS THE ZERO-TRIP PATHS, measured: once the loop has run it has filled everything
    under any hypothesis, so restricting the count to steady paths scores every seed 0 and
    discriminates nothing.

    A NONZERO FLOOR MEANS NO HYPOTHESIS EXPLAINS THE KERNEL and the caller is expected to refuse
    rather than report the survivors."""
    from itertools import product
    probe = walk(lines, make_state, **kw)
    keys = sorted(k for k in probe.unknown if not k.startswith("desc:"))
    if not keys:
        return [probe], probe.unique()
    if len(keys) > max_unknown:
        raise ModelError("%d pointers have an entry buffer this machine cannot derive; the "
                         "hypothesis space is too large to walk" % len(keys))
    runs = []
    for combo in product(range(ring), repeat=len(keys)):
        seed = dict(zip(keys, combo))
        r = walk(lines, lambda s=seed: _seeded(make_state(), s), **kw)
        r.unwritten = sum(1 for _p, f in r.unique() if f.kind == "read_unwritten")
        runs.append((seed, r))
    floor, common = min(r.unwritten for _s, r in runs), None
    for _s, r in runs:
        r.consistent = r.unwritten == floor
        if not r.consistent:
            continue
        sig = {(f.kind, getattr(getattr(f, "at", None), "inst", None),
                getattr(getattr(f, "other", None), "inst", None)) for _p, f in r.unique()}
        common = sig if common is None else (common & sig)
    return runs, (common or set())


def _seeded(ex, seed):
    ex.seed = dict(seed)
    return ex

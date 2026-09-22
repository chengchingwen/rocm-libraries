# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
Run `symexec` over many kernels at once.

The unit of work is one (kernel, entry-buffer hypothesis) walk, not one kernel: a kernel with four
unknown pointers is sixteen independent walks, so fanning out per kernel leaves most cores idle and
the slowest kernel sets the wall clock on its own.

    python -m Tensile.Lowering.tool.symexec_table a.s b.s ...
"""

from __future__ import annotations

import argparse
import collections
import os
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import product

from .symexec import Executor, ModelError
from .symexec.checks import fused_aliases, select_bits
from .symexec.program import DecodeError, decode_range, loop_body, setup_new_tile
from .symexec.walk import _seeded, walk


def _setup(path, ring, wave):
    """`(lines, entry line, establish line, state factory)` for one kernel."""
    lines = open(path, errors="replace").read().splitlines()
    span = loop_body(lines)
    start = setup_new_tile(lines)
    if span is None or start is None:
        raise ModelError("no rolled loop, or no setupNewTile marker")
    lo, hi = span
    bits = select_bits(decode_range(lines, start, lo - 1, allow_branches=True),
                       decode_range(lines, lo, hi, allow_branches=True))
    alias = fused_aliases(decode_range(lines, start, lo - 1, allow_branches=True),
                          decode_range(lines, lo, hi, allow_branches=True))

    def make_state():
        ex = Executor(wave=wave, ring=ring, seed={}, select_bits=bits)
        ex.alias.update(alias)
        return ex

    return lines, start, lo, make_state


def _why(e):
    """Why a kernel was refused, in the one spelling both the probe and the walk report it in."""
    return "%s: %s" % (type(e).__name__, e)


def _kind(path, finding):
    """`@entry` for a path that never re-enters the loop: its hazards are about the fill, not the
    steady state, and reporting both as one hides a real violation among known artifacts."""
    # The loop HEAD, not any label containing it: `label_InitCIterWmma_label_LoopBeginL_0` is the
    # entry trampoline, and counting it as a trip calls a prologue hazard steady.
    steady = any(b == "label_LoopBeginL" for b in path)
    return finding.kind if steady else finding.kind + "@entry"


def _one(job):
    """One walk under one hypothesis, as `(path, seed, unwritten, signature, findings)`."""
    path, seed, ring, wave, refusal = job
    # The probe already refused this kernel, and its reason is the answer -- re-deriving it here
    # assumed the refusal came from `_setup`, which it no longer need: a walk can raise too.
    if refusal is not None:
        return path, None, None, None, refusal
    try:
        lines, start, lo, make_state = _setup(path, ring, wave)
        r = walk(lines, lambda: _seeded(make_state(), seed), establish_before=lo, entry_line=start)
    except (ModelError, DecodeError, ValueError) as e:
        return path, seed, None, None, _why(e)
    uniq = r.unique()
    unwritten = sum(1 for _p, f in uniq if f.kind == "read_unwritten")
    sig = {(_kind(p, f), getattr(getattr(f, "at", None), "inst", None),
            getattr(getattr(f, "other", None), "inst", None)) for p, f in uniq}
    return path, seed, unwritten, sig, collections.Counter(_kind(p, f) for p, f in uniq)


def _hypotheses(path, ring, wave):
    """Every entry buffer assignment worth walking for one kernel."""
    lines, start, lo, make_state = _setup(path, ring, wave)
    probe = walk(lines, make_state, establish_before=lo, entry_line=start)
    keys = sorted(k for k in probe.unknown if not k.startswith("desc:"))
    if not keys:
        return [{}]
    return [dict(zip(keys, c)) for c in product(range(ring), repeat=len(keys))]


def _probe(job):
    """One kernel's hypotheses, as `(path, hypotheses, refusal)`."""
    path, ring, wave = job
    try:
        return path, _hypotheses(path, ring, wave), None
    except Exception as e:                          # a kernel that cannot even be set up
        return path, None, _why(e)


def run(paths, ring=2, wave=0, workers=None):
    """`{path: (floor, seeds at floor, seeds tried, kind counter or refusal)}`."""
    workers = workers or max(1, (os.cpu_count() or 8) - 2)
    # Enumerating the hypotheses costs a full probe walk per kernel, so at corpus scale it
    # dwarfs the walks it feeds -- 2.9 h serial against 12 min for the fan-out.  `map` keeps
    # input order, so the job list is the same one a serial loop would build.
    jobs = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        probes = [(p, ring, wave) for p in paths]
        for path, hypotheses, refusal in pool.map(_probe, probes, chunksize=1):
            if refusal is not None:
                jobs.append((path, None, ring, wave, refusal))
                print("%-34s SETUP FAILED %s" % (os.path.basename(path), refusal),
                      file=sys.stderr)
            else:
                jobs += [(path, s, ring, wave, None) for s in hypotheses]
    out = collections.defaultdict(list)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for path, seed, unwritten, sig, kinds in pool.map(_one, jobs, chunksize=1):
            out[path].append((seed, unwritten, sig, kinds))
    table = {}
    for p, runs in out.items():
        live = [r for r in runs if r[1] is not None]
        if not live:
            table[p] = (None, 0, len(runs), runs[0][3])
            continue
        floor = min(r[1] for r in live)
        at = [r for r in live if r[1] == floor]
        common = set.intersection(*[r[2] for r in at]) if at else set()
        kinds = collections.Counter()
        for kind, _a, _b in common:
            kinds[kind] += 1
        table[p] = (floor, len(at), len(runs), kinds)
    return table


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+", help="`.s` files, or a directory to walk for them")
    ap.add_argument("--ring", type=int, default=2)
    ap.add_argument("--wave", type=int, default=0)
    ap.add_argument("-j", "--workers", type=int, default=None)
    ap.add_argument("--summary", action="store_true", help="counts by outcome, not a row per kernel")
    a = ap.parse_args(argv)
    # A corpus sweep is thousands of kernels, which is more than one argv can carry.
    a.paths = [str(f) for p in a.paths
               for f in (sorted(pathlib.Path(p).rglob("*.s")) if os.path.isdir(p) else [p])]
    table = run(a.paths, ring=a.ring, wave=a.wave, workers=a.workers)
    if a.summary:
        tally, kindsum = collections.Counter(), collections.Counter()
        for p in a.paths:
            if p not in table:
                tally["not-run"] += 1
                continue
            floor, at, _n, kinds = table[p]
            tally["SETUP" if floor is None else "REFUSE" if floor
                  else "UNDEC" if (not kinds and at > 1)
                  else "CLEAN" if not kinds else "FINDING"] += 1
            # A setup refusal carries its message here, not a counter, and counting a string
            # tallies its CHARACTERS -- which buried the real kinds under 40 rows of letters.
            if floor is not None:
                kindsum += collections.Counter(kinds)
        total = sum(tally.values())
        print("%d kernels: %s" % (total, dict(tally.most_common())))
        if kindsum:
            print("finding kinds: %s" % dict(kindsum.most_common()))
        return 1 if (tally["FINDING"] or tally["UNDEC"]) else 0
    print("%-34s %6s %12s   %s" % ("kernel", "found", "seeds(ok/n)", "kinds"))
    for p in a.paths:
        if p not in table:
            continue
        floor, at, n, kinds = table[p]
        name = os.path.basename(p)
        if floor is None:
            print("%-34s %6s %12s   %s" % (name, "SETUP", "-", kinds))
        elif floor:
            print("%-34s %6s %12s   refuted %d times under the best hypothesis"
                  % (name, "REFUSE", "%d/%d" % (at, n), floor))
        elif not kinds and at > 1:
            # NOT CLEAN: several admissible phases disagreed, so the intersection emptied.
            print("%-34s %6s %12s   hypotheses disagree; no finding rests on all of them"
                  % (name, "UNDEC", "%d/%d" % (at, n)))
        else:
            print("%-34s %6d %12s   %s" % (name, sum(kinds.values()), "%d/%d" % (at, n), dict(kinds)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

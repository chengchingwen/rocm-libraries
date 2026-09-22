# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Sweep the scaffold tail's handoff contract over the whole decode matrix, in parallel.

The `K % DepthU` tail is scaffold-owned (#94), so what GIR owes it is an exit state.  Theta never
reads `AssertSummationElementMultiple`, so the program built for a tail-free config is the program
a tail-bearing one gets -- the existing matrix is a valid sample and no codegen run is needed.

    PYTHONPATH=.:Tensile/Tests/unit:Tensile/Tests/unit/characterization/LoopModel \\
        python Tensile/Tests/unit/characterization/LoopModel/check_tail_handoff.py

`--resets` assumes the scaffold's `numReadsIterCoalesced > 1` gate, under which the tail rebinds
both LDS pointers to buffer 0; that is kernel state the model cannot see, so it is a flag.
"""

import argparse
import collections
import multiprocessing
import sys

import matrix


def _one(args):
    """`(cell key, [violation])` for one configuration; a refusal carries no verdict."""
    order, key, resets = args
    from Tensile.LoopModel import adapter
    from Tensile.Lowering import build_gir
    from Tensile.Lowering.gir.analyses.tail_handoff import handoff_violations
    try:
        theta = adapter.params_to_theta(adapter.kernel_to_params(matrix.find(order, key)))
        return key, handoff_violations(build_gir(theta), tail_resets_lds=resets)
    except Exception as exc:
        return key, ["REFUSED: %s: %s" % (type(exc).__name__, str(exc).split("\n")[0])]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-j", "--jobs", type=int, default=multiprocessing.cpu_count())
    ap.add_argument("--resets", action="store_true", help="assume the tail rebinds both pointers")
    ap.add_argument("-v", "--verbose", action="store_true", help="print every failing cell")
    opts = ap.parse_args(argv)

    cells = [(order, key, opts.resets)
             for order in matrix.LOOP_ORDERS for key, _kernel in matrix.cells(order)]
    with multiprocessing.Pool(opts.jobs) as pool:
        verdicts = dict(pool.map(_one, cells, chunksize=8))

    by_kind = collections.Counter(v.split(":")[0] for vs in verdicts.values() for v in vs)
    clean = sum(1 for vs in verdicts.values() if not vs)
    print("%d cells, %d clean, %d with a violation"
          % (len(verdicts), clean, len(verdicts) - clean))
    for kind, n in by_kind.most_common():
        print("  %-14s %d" % (kind, n))
    if opts.verbose:
        for key, violations in sorted(verdicts.items()):
            for v in violations:
                print("  %-40s %s" % (key, v))
    return 1 if by_kind else 0


if __name__ == "__main__":
    sys.exit(main())

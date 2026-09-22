# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Fast parallel golden check -- the refactor inner loop.

Reads exactly the files `test_loopmodel_golden_char.py` reads and reports the same verdict; the
pytest suite stays the gate, this is what you run between edits. Serial pytest takes ~8 minutes,
this takes seconds.

    PYTHONPATH=.:Tensile/Tests/unit:Tensile/Tests/unit/characterization/LoopModel \\
        python Tensile/Tests/unit/characterization/LoopModel/check_goldens.py

Exit status is 0 when every cell matches, 1 otherwise; changed cells are printed with the loop
order, the stored digest and the produced one.
"""

import json
import multiprocessing
import pathlib
import sys

import matrix

GOLDENS = pathlib.Path(__file__).parent / "__goldens__"


def _order(order):
    """Digest rendered and structural output for one loop order."""
    cells = list(matrix.cells(order))
    return order, {
        "digests": {key: matrix.digest(kernel) for key, kernel in cells},
        "semantic": {key: matrix.semantic_digest(kernel) for key, kernel in cells},
    }


def main():
    with multiprocessing.Pool(min(len(matrix.LOOP_ORDERS), multiprocessing.cpu_count())) as pool:
        produced = dict(pool.map(_order, matrix.LOOP_ORDERS))

    moved, checked = [], 0
    for order, kinds in sorted(produced.items()):
        for kind, cells in kinds.items():
            stored = json.loads((GOLDENS / ("%s_%s.json" % (kind, order))).read_text())
            for key, got in cells.items():
                checked += 1
                if stored.get(key) != got:
                    moved.append(("%s/%s" % (kind, key), stored.get(key), got))

    for key, was, now in moved[:40]:
        print("  CHANGED %s  %s -> %s" % (key, was, now))
    if len(moved) > 40:
        print("  ... and %d more" % (len(moved) - 40))

    # The six full-text goldens are checked by the pytest suite; digests alone catch the change,
    # and the texts exist to explain it.
    print("%d cells checked, %d changed" % (checked, len(moved)))
    return 1 if moved else 0


if __name__ == "__main__":
    sys.exit(main())

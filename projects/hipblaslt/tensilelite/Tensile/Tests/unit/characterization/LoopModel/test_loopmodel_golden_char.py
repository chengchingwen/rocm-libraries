# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Characterization: what LoopModel emits today, over its whole reachable matrix.

Pins the decoder's output so it can be restructured with proof rather than hope. Two layers:

  * `test_matrix_digest` -- one digest per configuration, 2268 cells, sharded by loop order.
    Catches any change anywhere.
  * `test_representative_ir` -- full text for six cells spanning the shapes that have broken
    before. Catches nothing extra; explains what the digests caught.

Behaviour is pinned as-is, including refusals: a config that raises records its exception, because
"this configuration is rejected" is part of what the decoder does today.

Goldens are plain files under `__goldens__/`, not syrupy snapshots -- a departure from the suite
protocol recorded in `DECISIONS.md`. Regenerate with `LOOPMODEL_GOLDEN_UPDATE=1`.

CPU-only. No GPU, no rocisa.
"""

import json
import os
import pathlib

import pytest

import matrix

GOLDENS = pathlib.Path(__file__).parent / "__goldens__"
UPDATING = os.environ.get("LOOPMODEL_GOLDEN_UPDATE") == "1"

# Cells whose full text is stored. Each is a shape with a failure history, so when a digest moves
# the readable diff is likely to be here.
REPRESENTATIVE = {
    "baseline":        ("KMN", "bf16 KMN w1 0/0 plr1 pgr1 du64"),
    "no-prefetch":     ("KMN", "bf16 KMN w1 0/0 plr0 pgr1 du64"),
    "mixed-mt-du":     ("MNK", "bf16 MNK w1 1/2 plr1 pgr1 du64"),
    "six-axis":        ("KMNKMN", "bf16 KMNKMN w1 1/1 plr1 pgr1 du64"),
    "broadcast-outer": ("NKM", "bf16 NKM w4 0/0 plr1 pgr2 du128"),
    "mx-no-k-axis":    ("MNK", "mxf8 MNK w4 0/0 plr1 pgr1 du128"),
}


def _check(name, produced, dump=str):
    """Compare against the stored golden, or write it and return None when regenerating.

    It must NOT skip here: a test that writes two goldens would stop after the first, which is
    how `semantic_*.json` stayed unwritable while `digests_*.json` refreshed. Skip in the caller,
    after every golden it owns has been written.
    """
    path = GOLDENS / name
    if UPDATING:
        path.parent.mkdir(exist_ok=True)
        path.write_text(dump(produced))
        return None
    assert path.exists(), "missing golden %s; regenerate with LOOPMODEL_GOLDEN_UPDATE=1" % name
    return path.read_text()


@pytest.mark.unit
@pytest.mark.parametrize("order", matrix.LOOP_ORDERS)
def test_matrix_digest(order):
    """Every configuration's rendered and structural output."""
    cells = list(matrix.cells(order))
    rendered = {key: matrix.digest(kernel) for key, kernel in cells}
    semantic = {key: matrix.semantic_digest(kernel) for key, kernel in cells}
    for prefix, produced in (("digests", rendered), ("semantic", semantic)):
        stored = _check("%s_%s.json" % (prefix, order), produced,
                        dump=lambda d: json.dumps(d, indent=1, sort_keys=True) + "\n")
        if stored is None:
            continue
        expected = json.loads(stored)
        moved = {k: (expected.get(k), v) for k, v in produced.items()
                 if expected.get(k) != v}
        assert not moved, "%s: %d cell(s) changed, e.g. %s" % (
            prefix, len(moved), list(moved.items())[:3])
        assert set(produced) == set(expected), "the matrix itself changed shape"
    if UPDATING:
        pytest.skip("goldens for %s written" % order)


@pytest.mark.unit
@pytest.mark.parametrize("name", sorted(REPRESENTATIVE))
def test_representative_ir(name):
    """Full LoopIR, GIR and ledger text for one cell, so a digest change is readable."""
    order, key = REPRESENTATIVE[name]
    loop_ir, gir, ledger = matrix.decode(matrix.find(order, key))
    produced = "\n\n".join(("=== LoopIR ===", loop_ir, "=== GIR ===", gir,
                            "=== ledger ===", ledger))
    stored = _check("ir_%s.txt" % name, produced)
    if stored is None:
        pytest.skip("golden ir_%s.txt written" % name)
    assert produced == stored


@pytest.mark.unit
def test_the_matrix_has_not_collapsed():
    """The goldens above prove nothing if the matrix is empty or folded onto a few outputs.

    Shape is checked over the whole matrix (cheap -- no decoding); distinctness over one order,
    which is enough to catch a collapse and avoids decoding all 2268 cells a second time.
    """
    keys = [key for order in matrix.LOOP_ORDERS for key, _ in matrix.cells(order)]
    assert len(keys) == 2619 and len(set(keys)) == 2619, "matrix shape changed"
    distinct = {matrix.digest(kernel) for _, kernel in matrix.cells("KMN")}
    assert len(distinct) > 60, "KMN collapsed to %d distinct outputs of 263" % len(distinct)


@pytest.mark.unit
def test_every_representative_names_a_real_cell():
    """A typo in REPRESENTATIVE would silently drop a readable golden."""
    for name, (order, key) in sorted(REPRESENTATIVE.items()):
        assert matrix.find(order, key) is not None, name

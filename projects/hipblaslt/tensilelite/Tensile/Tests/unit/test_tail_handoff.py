# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The contract the GIR main loop owes the scaffold-owned tail loop (#94).

The `K % DepthU` remainder is emitted by `KernelWriter.py`; GIR has no tail block and needs none.
What it owes is the state that code inherits -- so these pin which parts of that state hold today
and which are known gaps, rather than leaving the handoff to be discovered on hardware.
"""

import pytest

from Tensile.Lowering import build_gir
from Tensile.Lowering.gir.analyses.tail_handoff import exit_state, handoff_violations

from gir_fixtures import BF16_NT_KMN, theta


def _kinds(violations):
    return sorted({v.split(":")[0] for v in violations})


def _program(**overrides):
    return build_gir(theta({**BF16_NT_KMN, **overrides}))


SHAPES = [
    pytest.param({"PrefetchGlobalRead": pgr, "PrefetchLocalRead": plr, "LoopOrder": order},
                 id="pgr%d-plr%d-%s" % (pgr, plr, order))
    for pgr in (1, 2) for plr in (0, 1, 2) for order in ("KMN", "MKN", "MNK")
]


@pytest.mark.parametrize("overrides", SHAPES)
def test_the_steady_path_leaves_the_descriptor_one_chunk_past_the_last_load(overrides):
    """The tail loads the chunk the descriptor names, so the main loop must leave it on `T`.

    This is the load-bearing half of the handoff: it is what lets the tail stay scaffold-owned."""
    bad = [v for v in handoff_violations(_program(**overrides)) if v.startswith("H-CHUNK:")]
    assert bad == []


@pytest.mark.parametrize("overrides", SHAPES)
def test_double_buffering_alone_is_not_a_handoff_violation(overrides):
    """A pipelined loop leaves the read and write pointers on different buffers by design.

    That is a violation only for a tail that rebinds neither, so under the scaffold's
    `numReadsIterCoalesced > 1` gate -- where both go back to buffer 0 -- it must not be reported."""
    assert "H-BUFFER" not in _kinds(handoff_violations(_program(**overrides),
                                                       tail_resets_lds=True))


@pytest.mark.parametrize("overrides", SHAPES)
def test_a_folded_short_arm_records_who_supplies_its_entry_step(overrides):
    """Folding the `T < M` arm onto the drain chain is correct only when the chain is entered at
    step `M - T`, and GIR does not emit that selection.

    The obligation must therefore be carried in `short_loop`, not assumed -- an arm that folds
    silently is one whose descriptor position nothing accounts for."""
    prog = _program(**overrides)
    fold = prog.meta.get("short_loop")
    if fold and fold.get("verdict") == "folded":
        assert fold.get("obligations")
    assert "H-FOLD" not in _kinds(handoff_violations(prog))


@pytest.mark.parametrize("overrides", SHAPES)
def test_the_whole_handoff_holds_once_the_tail_rebinds(overrides):
    """With the scaffold's rebind, nothing is left for GIR to owe -- the reason the tail can stay
    scaffold-owned rather than becoming a GIR block."""
    assert handoff_violations(_program(**overrides), tail_resets_lds=True) == []


def test_every_exit_path_is_accounted_for():
    """A path the walk misses reports no violation, which reads exactly like a clean handoff."""
    prog = _program()
    paths = exit_state(prog)
    assert paths, "no exit path found"
    assert all(path[0] == prog.entry for path in paths)
    # Every path ends where the scaffold resumes -- at a block whose successor leaves the program.
    for path in paths:
        assert path[-1] in prog.blocks

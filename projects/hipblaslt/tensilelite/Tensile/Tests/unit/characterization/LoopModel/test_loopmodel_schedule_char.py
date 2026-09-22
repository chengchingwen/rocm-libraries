# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Characterization: the DECISION TABLE -- what the decoder decided, per operand.

`Schedule.render()` is the answer to "what changes when PLR goes 1 -> 2" and "why does this operand
get one register buffer and that one four". Pinning it means those answers cannot move silently, and
gives a readable diff when the emitted goldens shift.

Covers the shapes with a failure history, and a PLR sweep on the baseline so the parameter's effect
is visible in one file.

CPU-only. No GPU, no rocisa.
"""

import os
import pathlib

import pytest

import matrix

GOLDENS = pathlib.Path(__file__).parent / "__goldens__"
UPDATING = os.environ.get("LOOPMODEL_GOLDEN_UPDATE") == "1"

CASES = {
    "plr-sweep":       [("KMN", "bf16 KMN w1 0/0 plr%d pgr1 du64" % p, p) for p in (0, 1, 2)],
    "mixed-mt-du":     [("MNK", "bf16 MNK w1 1/2 plr1 pgr1 du64", 1)],
    "six-axis":        [("KMNKMN", "bf16 KMNKMN w1 1/1 plr1 pgr1 du64", 1)],
    "broadcast-outer": [("NKM", "bf16 NKM w4 0/0 plr1 pgr2 du128", 1)],
    "mx-no-k-axis":    [("MNK", "mxf8 MNK w4 0/0 plr%d pgr1 du128" % p, p) for p in (0, 1, 2)],
}


def _table(order, key, plr):
    from Tensile.LoopModel import adapter
    from Tensile.LoopModel.schedule import build_S
    from Tensile.LoopModel.render import render_schedule
    from Tensile.LoopModel.schedule import Schedule
    try:
        theta = adapter.params_to_theta(adapter.kernel_to_params(matrix.find(order, key)))
    except RuntimeError as exc:              # PLR is faithful: no S carries it, so no kernel
        return "%s\n  REFUSED: %s" % (key, str(exc).split(" -- ")[0])
    depths, _ = build_S(theta)
    return "%s\n%s" % (key, render_schedule(Schedule(theta, depths, plr)))


@pytest.mark.unit
@pytest.mark.parametrize("name", sorted(CASES))
def test_decision_table(name):
    """Every decision for one shape, as text."""
    produced = "\n\n".join(_table(*case) for case in CASES[name])
    path = GOLDENS / ("schedule_%s.txt" % name)
    if UPDATING:
        path.write_text(produced)
        pytest.skip("golden written")
    assert produced == path.read_text()


@pytest.mark.unit
def test_a_refused_prefetch_is_reported_not_hidden():
    """A request the derivation cannot honour must show up as a REASON, never as a silent 0.

    `PrefetchLocalRead` is an input; the width and depth that serve it are derived. When the
    derivation cannot meet the request the plan has to say so, or the kernel ships named PLR1 with a
    PLR0 schedule and nothing points at why -- which is how the mxf8 DepthU=128 cells were read as a
    codegen bug for a week.
    """
    from Tensile.LoopModel import adapter
    from Tensile.LoopModel.schedule import build_S, Schedule
    from Tensile.LoopModel.traversal import requested_read_ahead
    # MI_K == DepthU leaves no reduction inner axis.  Every shortfall must be ATTRIBUTABLE: either
    # the operand's own derived request is lower (a full-inner one takes what the copy stages), or
    # the plan carries a refusal.  A bare 0 with neither is the failure this pins.
    theta = adapter.params_to_theta(adapter.kernel_to_params(
        matrix.find("MNK", "mxf8 MNK w4 0/0 plr1 pgr1 du128")))
    depths, _ = build_S(theta)
    schedule = Schedule(theta, depths, 1)
    plans = [(op, g, schedule.plan(op, g)) for op, g in schedule.operands()]
    assert plans, "no register operands -- the check went vacuous"
    shortfalls = 0
    for op, group, plan in plans:
        derived = requested_read_ahead(theta, op)
        if plan.prefetch_steps == 1:
            assert plan.prefetch_refused is None, f"{op.name}: honoured, so nothing to refuse"
            continue
        shortfalls += 1
        assert plan.prefetch_steps == derived or plan.prefetch_refused is not None, (
            f"{op.name}/{group}: PLR1 asked for, {plan.prefetch_steps} scheduled, and neither the "
            f"derived request ({derived}) nor a refusal accounts for it")
    assert shortfalls, "no operand falls short -- the attribution check went vacuous"


@pytest.mark.unit
@pytest.mark.parametrize("name", sorted(CASES))
def test_operand_facts(name):
    """What each operand IS -- pinned beside what we decided for it.

    The two tables are the model/decision split made concrete: this one does not move when PLR or
    the register budget moves, and the schedule table does.
    """
    from Tensile.LoopModel import adapter
    from Tensile.LoopModel.render import render_operand_facts
    produced = []
    for order, key, _plr in CASES[name]:
        try:
            theta = adapter.params_to_theta(adapter.kernel_to_params(matrix.find(order, key)))
        except RuntimeError as exc:          # recorded, not skipped -- a refusal is an outcome
            produced.append("%s\n  REFUSED: %s" % (key, str(exc).split(" -- ")[0]))
            continue
        produced.append("%s\n%s" % (key, render_operand_facts(theta)))
    text = "\n\n".join(produced)
    path = GOLDENS / ("facts_%s.txt" % name)
    if UPDATING:
        path.write_text(text)
        pytest.skip("golden written")
    assert text == path.read_text()

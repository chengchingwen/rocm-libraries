# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Focused tests for the physical main-loop scaffold represented in the GIR frame contract."""

from Tensile.Lowering import build_gir
from Tensile.Lowering.gir.analysis import AnalysisManager
from Tensile.Lowering.gir.analyses.trip_domains import TripDomains
from Tensile.Lowering.gir.emit_plan import plan_program
from Tensile.Lowering.gir.frame_contract import build_contract
from Tensile.Lowering.gir.nodes import CondChain

from gir_fixtures import BF16_NT_KMN, theta


def _pgr2_program():
    return build_gir(theta({
        **BF16_NT_KMN,
        "PrefetchGlobalRead": 2,
        "PrefetchLocalRead": 1,
    }))


def test_scaffold_map_names_all_three_folded_main_loop_exits():
    """T==1, direct T==2, and the eventual steady exit are three physical scaffold paths."""
    prog = _pgr2_program()

    term = prog.blocks["prologue_join"].term
    assert isinstance(term, CondChain)
    assert [(pred.label, target) for pred, target in term.arms] == [
        ("toPGR1", "drain1"),
        ("", "steady"),
    ]
    assert (term.default_label, term.default) == ("LoopEndL", "drain0")
    assert prog.blocks["steady"].term.label == "LoopEndL"


def test_trip_domains_include_direct_and_counted_main_loop_exits():
    """One steady trip exits directly; repeated trips take both the back edge and final exit."""
    prog = _pgr2_program()
    domains = AnalysisManager().get(TripDomains(), prog)

    assert domains.admits("prologue_join", "drain1") == {1}
    assert domains.admits("prologue_join", "drain0") == {2}
    assert 3 in domains.admits("prologue_join", "steady")
    assert 3 in domains.admits("steady", "drain0")
    assert 3 not in domains.admits("steady", "steady")
    assert 4 in domains.admits("steady", "steady")


def test_frame_contract_keeps_three_folded_main_loop_exit_frames():
    """Each scaffold exit gets a guard value; counted-loop edges accept only the steady arm."""
    prog = _pgr2_program()
    plans = plan_program(prog)
    anchors = {label: actions[0].action_id for label, actions in plans.items() if actions}
    source = anchors["prologue_join"]
    destinations = {
        anchors["drain1"]: 1,
        anchors["steady"]: 2,
        anchors["drain0"]: 3,
    }

    contract = build_contract(prog)
    assignments = {}
    for edge in contract.edges:
        if edge.relative or edge.src != source or edge.dst not in destinations:
            continue
        assignments.setdefault(edge.gen, {})[edge.dst] = edge.value
    guards = [(gen, values) for gen, values in assignments.items() if values == destinations]

    assert len(guards) == 1, assignments
    guard_gen = guards[0][0]
    generation = contract.generations[guard_gen]
    assert (generation.ring, generation.entry, generation.advance) == (4, 0, 0)

    required = {(need.dst, need.src, need.gen): need.values for need in contract.requires}
    assert required[(anchors["drain1"], anchors["prologue_join"], guard_gen)] == (0, 1)
    assert required[(anchors["steady"], anchors["prologue_join"], guard_gen)] == (0, 2)
    assert required[(anchors["drain0"], anchors["prologue_join"], guard_gen)] == (0, 3)
    assert required[(anchors["steady"], anchors["steady"], guard_gen)] == (0, 2)
    assert required[(anchors["drain0"], anchors["steady"], guard_gen)] == (0, 2)

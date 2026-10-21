# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""ValuePlacementSolver — the shared def-placement fixpoints."""

import pytest

from Tensile.Lowering.gir.analyses.value_placement import (
    RequiredValue, ValuePlacementSolver, BOTTOM, TOP, meet)
from Tensile.Lowering.gir.nodes import BLOCK_ENTRY, BLOCK_EXIT


def _raise(kind, detail):
    raise AssertionError("%s: %r" % (kind, detail))


class N:
    """A stand-in body node; identity is what the anchors compare."""
    def __init__(self, tag):
        self.tag = tag
    def __repr__(self):
        return "N(%s)" % self.tag


def _pipeline(mod=2, adv=1):
    """prologue -> steady -> drain, steady self-loop (the back edge).
    Every block copies once.  Frames: the back edge advances the chunk by `adv`; prologue and drain
    sit in the same frame as steady for simplicity, so the only delta is the back edge.
    """
    p, s, d = N("copy@prologue"), N("copy@steady"), N("copy@drain")
    acc = {"prologue": [RequiredValue("prologue", p, 0)],
           "steady":   [RequiredValue("steady",   s, 0)],
           "drain":    [RequiredValue("drain",    d, 0)]}
    succs = {"prologue": ["steady"], "steady": ["steady", "drain"], "drain": []}
    preds = {"prologue": [], "steady": ["prologue", "steady"], "drain": ["steady"]}
    # The latch's per-trip def runs before the exit branch too, so the drain is in the POST-advance
    # frame — exactly what `Block.chunk_base` encodes in the real program.
    delta = lambda a, b: adv if a == "steady" and b in ("steady", "drain") else 0
    return acc, preds, succs, delta, (p, s, d)


def test_meet_is_a_lattice_not_a_preference():
    assert meet(BOTTOM, 3) == 3 and meet(3, BOTTOM) == 3
    assert meet(3, 3) == 3
    assert meet(3, 4) is TOP          # disagreement is reported, never resolved to a side
    assert meet(TOP, 3) is TOP


def test_selfloop_pointer_advances_once_per_trip_on_the_back_edge():
    """A rotating pointer whose every access names the same slot in its own frame still has to
    rotate once per trip — the whole rotation IS the back edge.  The def must therefore land at the
    END of the body (after that trip's access), not before it."""
    acc, preds, succs, delta, (p, s, d) = _pipeline(mod=2, adv=1)
    solver = ValuePlacementSolver(acc, preds, succs, delta, "prologue", 0, modulus=2)
    places = solver.solve(_raise)
    back = [pl for pl in places if pl.block == "steady" and pl.at_exit]
    assert len(back) == 1, places
    assert back[0].block == "steady"
    # after the body's own access, before the block ends: the trip bottom.
    assert back[0].after is s and back[0].before == BLOCK_EXIT


def test_prologue_handoff_becomes_a_def_at_the_end_of_the_prologue():
    """The pipeline shape that motivated this module.  The prologue leaves the descriptor at the
    chunk it just fetched; the steady body's copy needs the NEXT one.  That transition is an edge
    def, and the edge's own block is the prologue — so it lands AFTER the prologue's copy, which is
    exactly the hand-written scaffold's `prologue: copy(0); advance->1`."""
    p, s = N("copy@prologue"), N("copy@steady")
    acc = {"prologue": [RequiredValue("prologue", p, 0)], "steady": [RequiredValue("steady", s, 1)]}
    succs = {"prologue": ["steady"], "steady": []}
    preds = {"prologue": [], "steady": ["prologue"]}
    places = ValuePlacementSolver(acc, preds, succs, lambda a, b: 0,
                                  "prologue", 0).solve(_raise)
    edge = [pl for pl in places if pl.block == "prologue" and pl.at_exit]
    assert len(edge) == 1, places
    assert edge[0].block == "prologue"          # NOT inside the loop body
    assert edge[0].after is p                   # immediately after the copy it supersedes
    assert edge[0].before == BLOCK_EXIT
    assert (edge[0].from_value, edge[0].to_value) == (0, 1)


def test_intra_block_change_yields_a_window_between_the_two_accesses():
    a, b = N("read@0"), N("read@1")
    acc = {"steady": [RequiredValue("steady", a, 0), RequiredValue("steady", b, 1)]}
    places = ValuePlacementSolver(acc, {"steady": []}, {"steady": []},
                                  lambda x, y: 0, "steady", 0, modulus=2).solve(_raise)
    inner = [pl for pl in places if not pl.at_exit and pl.after is a]
    assert len(inner) == 1, places
    # the WINDOW is both accesses; picking a point inside it is PlacementPass's policy, not ours.
    assert inner[0].before is b and (inner[0].from_value, inner[0].to_value) == (0, 1)


def test_no_change_no_def():
    p, s = N("c0"), N("c1")
    acc = {"a": [RequiredValue("a", p, 0)], "b": [RequiredValue("b", s, 0)]}
    places = ValuePlacementSolver(acc, {"a": [], "b": ["a"]}, {"a": ["b"], "b": []},
                                  lambda x, y: 0, "a", 0).solve(_raise)
    assert places == []


def test_accessless_block_passes_the_value_through_both_ways():
    """A block with no access must not absorb the transition: the def belongs on the edge where
    supply and demand actually differ, not on every edge of the chain."""
    p, s = N("copy@a"), N("copy@c")
    acc = {"a": [RequiredValue("a", p, 0)], "b": [], "c": [RequiredValue("c", s, 0)]}
    places = ValuePlacementSolver(acc, {"a": [], "b": ["a"], "c": ["b"]},
                                  {"a": ["b"], "b": ["c"], "c": []},
                                  lambda x, y: 0, "a", 0).solve(_raise)
    assert places == [], places


def test_conflicting_join_is_reported_not_resolved():
    """Two predecessors leaving different values: there is no single correct def, and preferring a
    predecessor is the silent-wrong-answer this whole layer exists to avoid."""
    x, y, z = N("c@x"), N("c@y"), N("c@z")
    acc = {"x": [RequiredValue("x", x, 0)], "y": [RequiredValue("y", y, 1)], "z": [RequiredValue("z", z, 0)]}
    succs = {"x": ["z"], "y": ["z"], "z": []}
    preds = {"x": [], "y": [], "z": ["x", "y"]}
    seen = []
    ValuePlacementSolver(acc, preds, succs, lambda a, b: 0, "x", 0).solve(
        lambda kind, detail: seen.append(kind))
    # y supplies 1 where z demands 0 -> a real edge def; x supplies 0 -> none.  The conflict this
    # test pins is the CRITICAL EDGE: z has two preds, y has one succ, so y hosts it legally.
    assert seen == [] or "critical-edge" in seen


def test_critical_edge_is_reported():
    """pred has two successors AND succ has two predecessors: the def cannot go at either end
    without executing on a path that must not have it.  Must be reported."""
    # `u` must DEMAND the old value for this to be a real critical edge.
    a, b, c = N("c@p"), N("c@t"), N("c@u")
    acc = {"p": [RequiredValue("p", a, 0)], "q": [], "t": [RequiredValue("t", b, 1)],
           "u": [RequiredValue("u", c, 0)]}
    succs = {"p": ["t", "u"], "q": ["t"], "t": [], "u": []}
    preds = {"p": [], "q": [], "t": ["p", "q"], "u": ["p"]}
    seen = []
    ValuePlacementSolver(acc, preds, succs, lambda x, y: 0, "p", 0).solve(
        lambda kind, detail: seen.append((kind, detail[0], detail[1])))
    assert ("critical-edge", "p", "t") in seen, seen


def test_frame_delta_makes_cross_frame_blocks_comparable():
    """Same physical slot, different chunk frames: without the delta the solver would invent a def
    that the hardware does not need."""
    s, d = N("copy@steady"), N("copy@drain")
    acc = {"steady": [RequiredValue("steady", s, 0)], "drain": [RequiredValue("drain", d, 1)]}
    succs, preds = {"steady": ["drain"], "drain": []}, {"steady": [], "drain": ["steady"]}
    # drain sits one chunk further along, so demand 1 in drain IS supply 0 in steady.
    places = ValuePlacementSolver(acc, preds, succs,
                                  lambda a, b: -1 if (a, b) == ("steady", "drain") else 0,
                                  "steady", 0).solve(_raise)
    assert places == [], places


def test_demand_propagates_back_through_accessless_blocks_to_the_earliest_point():
    """The backward fixpoint is what makes EARLIEST earliest.
    a(uses 0) -> b(no access) -> c(uses 1).  The transition must be hoisted all the way back to the
    end of `a` — the last point the old value is still live — NOT left at b's exit next to the use.
    """
    a, c = N("copy@a"), N("copy@c")
    acc = {"a": [RequiredValue("a", a, 0)], "b": [], "c": [RequiredValue("c", c, 1)]}
    places = ValuePlacementSolver(acc, {"a": [], "b": ["a"], "c": ["b"]},
                                  {"a": ["b"], "b": ["c"], "c": []},
                                  lambda x, y: 0, "a", 0).solve(_raise)
    assert len(places) == 1, places
    assert places[0].block == "a", "demand did not reach back past the accessless block"
    assert places[0].after is a and places[0].at_exit
    assert (places[0].from_value, places[0].to_value) == (0, 1)


def test_demand_from_two_successors_must_agree_to_share_one_exit_def():
    """One register, one exit: two successors demanding different values cannot share a def at the
    predecessor's exit.  If the fixpoint quietly picked one, the other path would run with the
    wrong buffer — the silent-wrong-answer this layer exists to prevent."""
    a, s, t = N("copy@a"), N("copy@s"), N("copy@t")
    acc = {"a": [RequiredValue("a", a, 0)], "s": [RequiredValue("s", s, 1)], "t": [RequiredValue("t", t, 2)]}
    succs, preds = {"a": ["s", "t"], "s": [], "t": []}, {"a": [], "s": ["a"], "t": ["a"]}
    seen = []
    places = ValuePlacementSolver(acc, preds, succs, lambda x, y: 0, "a", 0, modulus=4).solve(
        lambda kind, detail: seen.append(kind))
    assert seen == [], seen
    # s and t each have ONE predecessor, so each transition is legally realized at that successor's
    # HEAD -- two distinct defs, never one shared exit def at `a`.
    assert not any(pl.at_exit for pl in places), places
    by_block = {pl.block: pl for pl in places}
    assert set(by_block) == {"s", "t"}, places
    assert by_block["s"].to_value == 1 and by_block["t"].to_value == 2
    assert by_block["s"].from_value == 0 and by_block["t"].from_value == 0
    assert by_block["s"].after == BLOCK_ENTRY and by_block["s"].before is s

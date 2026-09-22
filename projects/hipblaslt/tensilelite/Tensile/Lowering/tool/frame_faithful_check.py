# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
Is the LDS hazard model faithful, and is the residual charged against it a real FIFO rank?

  L*  LoopIR   -- does every read name storage some copy fills, in the frame it runs in
  H*  hazards  -- is every instance a real alias, and does its `gap` equal the frame-graph distance
  W*  residual -- does the charged `n` equal the MINIMUM rank over the site's own instances
  F*  fences   -- does a fence retain each cross-wave pair's exact frame/token relation

L and H are independent of what they check. W replays the emitted waits over PotentialWaits and
also compares them with a clean-program CounterFlow derivation.

    python -m Tensile.Lowering.tool.frame_faithful_check [--orders KMN,...] [-j N]
"""

from __future__ import annotations

import argparse
import collections
import copy
import os
import sys
from concurrent.futures import ProcessPoolExecutor

from ..gir.analyses.frame_hazards import FrameHazards, _accesses
from ..gir.analyses.frame_map import FrameMap
from ..gir.analyses.lds_buffers import LdsBufferIds
from ..gir.analyses.fence_regions import fences_of, separating
from ..gir.analyses.backward_wait_counts import BackwardWaitCounts
from ..gir.analyses.counter_flow import CounterFlow, simulate_counterflow
from ..gir.analyses.potential_waits import PotentialWaits
from ..gir import AnalysisManager, Mark


def _distances(fm, start):
    """Edges to every node `start` reaches in ONE OR MORE steps: a loop returns to itself after a
    full cycle, and calling that zero would make every wrap pair look same-trip."""
    out, queue = {}, [(start, 0)]
    while queue:
        node, d = queue.pop(0)
        for nxt in fm.successors(node):
            if nxt not in out:
                out[nxt] = d + 1
                queue.append((nxt, d + 1))
    return out


def audit(prog):
    """Every faithfulness question for one program, as `{code: count}` plus a witness per code."""
    am = AnalysisManager()
    fm, storage = am.get(FrameMap(), prog), am.get(LdsBufferIds(), prog)
    hz = am.get(FrameHazards(), prog)
    acc = _accesses(prog, storage.geometry)
    bad, witness = collections.Counter(), {}

    def flag(code, detail):
        bad[code] += 1
        witness.setdefault(code, detail)

    nodes = set(fm.nodes())
    # ---- L: is LoopIR's naming total, and is every read fed ----------------------------------
    written = collections.defaultdict(set)      # slot -> frames some write fills it in
    for label, touches in acc.items():
        for t in touches:
            for f in fm.frames(label):
                slots = fm.touches(t.ref, f)
                if not slots:
                    flag("L1_unnamed", "%s:%d %s frame=%s" % (label, t.pos, t.operand, f))
                if t.is_write:
                    for s in slots:
                        written[s].add((label, f))
    for label, touches in acc.items():
        for t in touches:
            if t.is_write:
                continue
            for f in fm.frames(label):
                for s in fm.touches(t.ref, f):
                    if not written.get(s):
                        flag("L2_read_unfilled",
                             "%s:%d %s frame=%s slot=%s" % (label, t.pos, t.operand, f, s))

    # ---- H: is every instance a real alias, and is `gap` the frame distance? -------------------
    dist = {}
    raw_cover = collections.defaultdict(set)
    for fp, fc, h in hz.instances():
        if (h.producer.block, fp) not in nodes or (h.consumer.block, fc) not in nodes:
            flag("H1_instance_off_graph", "%s %s->%s" % (h.kind, fp, fc))
            continue
        if not (fm.touches(h.producer.ref, fp) & fm.touches(h.consumer.ref, fc)):
            flag("H2_instance_not_alias",
                 "%s %s:%d->%s:%d frames %s->%s" % (h.kind, h.producer.block, h.producer.pos,
                                                    h.consumer.block, h.consumer.pos, fp, fc))
        src, sink = (h.producer.block, fp), (h.consumer.block, fc)
        if src not in dist:
            dist[src] = _distances(fm, src)
        # Zero is available only to a pair the SAME execution runs in program order.
        d = 0 if (src == sink and h.producer.pos < h.consumer.pos) else dist[src].get(sink)
        if d is None:
            flag("H3_consumer_unreachable", "%s %s->%s" % (h.kind, fp, fc))
        elif d != h.gap:
            # A stored gap describes whichever instance made the edge; the rest would be charged
            # against a distance not theirs.
            flag("H4_gap_not_distance",
                 "%s %s:%d->%s:%d gap=%d distance=%d" % (h.kind, h.producer.block, h.producer.pos,
                                                         h.consumer.block, h.consumer.pos,
                                                         h.gap, d))
        if h.kind == "RAW":
            raw_cover[(h.consumer.block, h.consumer.pos)].add(fc)
    for label, touches in acc.items():
        for t in touches:
            if t.is_write:
                continue
            for f in fm.frames(label):
                if f not in raw_cover.get((label, t.pos), ()):
                    flag("H5_read_frame_uncovered",
                         "%s:%d %s frame=%s" % (label, t.pos, t.operand, f))

    # ---- W: replay the emitted waits, then compare with a clean-program forward derivation ----
    potentials = am.get(PotentialWaits(), prog)
    replay, _observed = simulate_counterflow(prog, fm, potentials, {})
    for violation in replay.violations:
        flag("W1_under_wait",
             "%s:%d %s emitted=%r producer_age=%d"
             % (violation.site[0], violation.site[1], violation.site[2],
                violation.emitted, violation.age))

    clean = copy.deepcopy(prog)
    identity_map = {
        id(old): id(new)
        for label, block in prog.blocks.items()
        for old, new in zip(block.body, clean.block(label).body)
    }
    for block in clean.blocks.values():
        for node in block.body:
            for relation in (getattr(node, "at", {}) or {}).get("relations", ()):
                for endpoint in (relation.get("producer") or {},
                                 relation.get("consumer") or {}):
                    if endpoint.get("identity") in identity_map:
                        endpoint["identity"] = identity_map[endpoint["identity"]]
    actual = {}
    for label, block in clean.blocks.items():
        body = []
        for node in block.body:
            if isinstance(node, Mark) and node.kind == "waitcnt":
                logical_pos = len(body)
                for counter in ("tensorcnt", "dscnt"):
                    if counter in (node.at or {}):
                        actual[(label, logical_pos, counter)] = int(node.at[counter])
                continue
            body.append(node)
        block.body = body
    clean.bump()
    clean_am = AnalysisManager()
    expected = {(site.block, site.pos, site.counter): site.n
                for site in clean_am.get(CounterFlow(), clean)}
    backward = {(site.block, site.pos, site.counter): site.n
                for site in clean_am.get(BackwardWaitCounts(), clean)}
    for site in sorted(set(expected) | set(backward)):
        if expected.get(site) != backward.get(site):
            flag("W3_directional_disagreement",
                 "%s:%d %s forward=%r backward=%r"
                 % (*site, expected.get(site), backward.get(site)))
    for site in sorted(set(actual) | set(expected)):
        have, need = actual.get(site), expected.get(site)
        if have is None:
            flag("W1_under_wait", "%s:%d %s missing, rank=%d" % (*site, need))
        elif need is None:
            flag("W2_over_wait", "%s:%d %s emitted=%d but producer is retired" % (*site, have))
        elif have > need:
            flag("W1_under_wait", "%s:%d %s emitted=%d rank=%d" % (*site, have, need))
        elif have < need:
            flag("W2_over_wait", "%s:%d %s emitted=%d rank=%d" % (*site, have, need))

    # ---- F: does the separating fence carry this exact frame relation? ------------------------
    fence_slots = fences_of(prog)
    fence_relations = {}
    for label, block in prog.blocks.items():
        for pos, node in enumerate(block.body):
            if isinstance(node, Mark) and node.kind == "fence":
                fence_relations[(label, pos)] = tuple(node.at.get("relations") or ())

    def logical_at(touch):
        pos = 0
        for node in prog.block(touch.block).body:
            if node is touch.inst:
                return f"{touch.block}[{pos}]"
            if not (isinstance(node, Mark) and node.kind == "waitcnt"):
                pos += 1
        return None

    def end_matches(fact, touch, frame):
        return ((fact.get("identity") == id(touch.inst) or fact.get("at") == logical_at(touch))
                and fact.get("operand") == touch.operand
                and fact.get("frame") == fm.render(frame)
                and fact.get("token") == fm.generation(touch.ref, frame)
                and tuple(fact.get("regions") or ()) == tuple(
                    tuple(sorted(values)) for values in getattr(touch, "regions", ())))

    def relation_matches(relation, hazard, fp, fc):
        return (relation.get("kind") == hazard.kind
                and int(relation.get("gap", -1)) == int(hazard.gap)
                and end_matches(relation.get("producer") or {}, hazard.producer, fp)
                and end_matches(relation.get("consumer") or {}, hazard.consumer, fc))

    nslots = lambda label: len(prog.block(label).body) + 1
    for fp, fc, hazard in hz.instances():
        if not hazard.cross_agent:
            continue
        between = separating(hazard, fence_slots, nslots)
        if not between:
            code = "F3_war_no_fence_between" if hazard.kind == "WAR" else "F2_no_fence_between"
            flag(code, "%s %s:%d->%s:%d gap=%d"
                 % (hazard.kind, hazard.producer.block, hazard.producer.pos,
                    hazard.consumer.block, hazard.consumer.pos, hazard.gap))
            continue
        if not any(any(relation_matches(relation, hazard, fp, fc)
                           for relation in fence_relations.get(site, ()))
                   for site in between):
            code = "F4_war_fence_omits_relation" if hazard.kind == "WAR" \
                else "F1_fence_omits_relation"
            flag(code, "%s %s:%d->%s:%d frames %s->%s"
                 % (hazard.kind, hazard.producer.block, hazard.producer.pos,
                    hazard.consumer.block, hazard.consumer.pos, fp, fc))
    return bad, witness


def _one(job):
    order, key, kernel = job
    try:
        from Tensile.LoopModel import adapter
        from Tensile.Lowering import build_gir
        prog = build_gir(adapter.params_to_theta(adapter.kernel_to_params(kernel)))
    except Exception as exc:
        return order, key, {"SKIP_" + type(exc).__name__: 1}, {}
    try:
        bad, witness = audit(prog)
        return order, key, dict(bad), witness
    except Exception as exc:
        return order, key, {"AUDIT_" + type(exc).__name__: 1}, {"AUDIT": str(exc)[:200]}


def main(argv=None):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                    "Tests", "unit", "characterization", "LoopModel"))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "Tests", "unit"))
    import matrix

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--orders", default=",".join(matrix.LOOP_ORDERS))
    ap.add_argument("--family", default="", help="only cells whose key starts with this")
    ap.add_argument("-j", "--workers", type=int, default=max(1, (os.cpu_count() or 8) - 2))
    ap.add_argument("-v", "--witness", action="store_true", help="one example per code")
    a = ap.parse_args(argv)

    jobs = [(o, k, kern) for o in a.orders.split(",")
            for k, kern in matrix.cells(o) if k.startswith(a.family)]
    # PGR3 is off the golden matrix and the only setting that builds `model_only` blocks.
    jobs += [(o, k + " pgr3", dict(kern, PrefetchGlobalRead=3))
             for o, k, kern in list(jobs) if "pgr1" in k or "f1 " in k]
    tally, per_code, shown = collections.Counter(), collections.Counter(), {}
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for order, key, bad, witness in pool.map(_one, jobs, chunksize=4):
            real = {c: n for c, n in bad.items() if not c.startswith("SKIP_")}
            tally["configs"] += 1
            tally["skipped" if any(c.startswith("SKIP_") for c in bad)
                  else "dirty" if real else "clean"] += 1
            for code, n in bad.items():
                per_code[code] += n
                if code not in shown:
                    shown[code] = "%s %s | %s" % (order, key, witness.get(code, ""))
    print("%d configs: %s" % (tally["configs"], dict(tally.most_common())))
    for code, n in sorted(per_code.items()):
        print("  %-26s %8d%s" % (code, n, ("   e.g. " + shown[code]) if a.witness else ""))
    return 1 if any(not c.startswith("SKIP_") for c in per_code) else 0


if __name__ == "__main__":
    sys.exit(main())

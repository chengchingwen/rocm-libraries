# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Frame-specific completion hazards before a residual has been chosen."""

from __future__ import annotations

from dataclasses import dataclass

from ..analysis import Analysis
from ..nodes import Mark, Move
from .fence_regions import fences_of, separating
from .frame_hazards import FrameHazards
from .frame_map import FrameMap
from .reg_hazards import RegHazards
from .wait_common import (DSCNT, FANOUT_META, counter_for, counter_of, fan_out,
                          is_register_fill)


@dataclass(frozen=True)
class IssueKey:
    """One dynamic producer occurrence in GIR's finite frame quotient."""
    counter: str
    block: str
    pos: int
    frame: object
    identity: tuple


@dataclass(frozen=True)
class IssueInfo:
    """One counted Move in one frame; keys share this one issue width."""
    counter: str
    width: int
    keys: tuple = ()


def issue_occurrence(key):
    """Counter FIFO identity; storage/register aliases on one Move share one completion."""
    return key.counter, key.block, key.pos, key.frame


def _issue_order(key):
    return (key.counter, key.block, key.pos,
            getattr(key.frame, "phases", ()), key.identity)


@dataclass(frozen=True)
class PotentialWait:
    """A hazard anchor whose counter residual has not yet been derived."""
    block: str
    pos: int
    frame: object
    counter: str
    producer: IssueKey
    producer_frame: object
    consumer_frame: object
    hazard: object
    at_producer: bool = False

    @property
    def site(self):
        return self.block, self.pos, self.counter


def _relation_fence_sites(prog, hazard, fp, fc, fm):
    producer_frame, consumer_frame = fm.render(fp), fm.render(fc)
    current_ids = {id(node) for block in prog.blocks.values() for node in block.body}
    producer_regions = tuple(tuple(sorted(values))
                             for values in getattr(hazard.producer, "regions", ()))
    consumer_regions = tuple(tuple(sorted(values))
                             for values in getattr(hazard.consumer, "regions", ()))
    out = []
    for block in prog.blocks.values():
        for pos, node in enumerate(block.body):
            if not isinstance(node, Mark) or node.kind != "fence":
                continue
            for relation in node.at.get("relations") or ():
                producer = relation.get("producer") or {}
                consumer = relation.get("consumer") or {}
                producer_identity = producer.get("identity")
                consumer_identity = consumer.get("identity")
                producer_matches = (
                    producer_identity == id(hazard.producer.inst)
                    if producer_identity in current_ids else
                    producer.get("at") == f"{hazard.producer.block}[{hazard.producer.pos}]")
                consumer_matches = (
                    consumer_identity == id(hazard.consumer.inst)
                    if consumer_identity in current_ids else
                    consumer.get("at") == f"{hazard.consumer.block}[{hazard.consumer.pos}]")
                if (relation.get("kind") == hazard.kind
                        and int(relation.get("gap", -1)) == int(hazard.gap)
                        and producer_matches and consumer_matches
                        and producer.get("operand") == hazard.producer.operand
                        and consumer.get("operand") == hazard.consumer.operand
                        and tuple(producer.get("regions") or ()) == producer_regions
                        and tuple(consumer.get("regions") or ()) == consumer_regions
                        and producer.get("frame") == producer_frame
                        and consumer.get("frame") == consumer_frame):
                    out.append((block.label, pos))
                    break
    return tuple(dict.fromkeys(out))


def anchor_site(prog, hazard, fences, counter, fm=None, fp=None, fc=None):
    """Return ``(block, slot, at_producer)`` for one hazard."""
    if (counter == DSCNT and hasattr(hazard.producer, "ring_pos")
            and int(hazard.gap) > 0
            and hazard.producer.block == hazard.consumer.block):
        return hazard.producer.block, len(prog.block(hazard.producer.block).body), True
    if not getattr(hazard, "cross_agent", False):
        return hazard.consumer.block, hazard.consumer.pos, False
    if fm is not None and fp is not None and fc is not None:
        sites = _relation_fence_sites(prog, hazard, fp, fc, fm)
        if len(sites) != 1:
            raise RuntimeError(
                "PotentialWaits: cross-agent %s %s[%d] -> %s[%d] frame %s -> %s "
                "has %d relation-owning fences"
                % (hazard.kind, hazard.producer.block, hazard.producer.pos,
                   hazard.consumer.block, hazard.consumer.pos,
                   fm.render(fp), fm.render(fc), len(sites)))
        block, pos = sites[0]
        return block, pos, block == hazard.producer.block
    sep = separating(hazard, fences, lambda label: len(prog.block(label).body) + 1)
    here = [pos for block, pos in sep if block == hazard.consumer.block]
    if here:
        return hazard.consumer.block, max(here), False
    there = [pos for block, pos in sep if block == hazard.producer.block]
    if there:
        return hazard.producer.block, max(there), True
    return hazard.consumer.block, hazard.consumer.pos, False


def _identity(hazard, frame, fm):
    touch = hazard.producer
    if hasattr(touch, "ring_pos"):
        return ("register", touch.operand) + tuple(touch.ring_pos)
    return ("shared",) + tuple(sorted(fm.touches(touch.ref, frame), key=repr))


def _key(hazard, counter, frame, fm):
    return IssueKey(counter, hazard.producer.block, hazard.producer.pos, frame,
                    _identity(hazard, frame, fm))


def _register_frame_pairs(hazard, fm):
    """Concrete frame pairs on which a static register RAW can execute."""
    pframes = fm.frames(hazard.producer.block)
    cframes = fm.frames(hazard.consumer.block)
    out = []
    for fp in pframes:
        src = (hazard.producer.block, fp)
        for fc in cframes:
            dst = (hazard.consumer.block, fc)
            if (src == dst and hazard.producer.pos < hazard.consumer.pos
                    and int(hazard.gap) == 0):
                out.append((fp, fc))
            elif dst in fm.reaches(src):
                out.append((fp, fc))
    return tuple(out)


class PotentialWaitSet:
    """Query API over potential sites and all counted frame issues."""

    def __init__(self, potentials, issues, unresolved=()):
        self._potentials = tuple(sorted(
            potentials,
            key=lambda p: (p.block, p.pos, p.counter,
                           getattr(p.frame, "phases", ()), _issue_order(p.producer))))
        self._issues = dict(issues)
        self._unresolved = tuple(unresolved)
        self._by_site = {}
        self._by_dynamic = {}
        dynamic_producers = {}
        for potential in self._potentials:
            self._by_site.setdefault(potential.site, []).append(potential)
            key = (potential.block, potential.pos, potential.frame, potential.counter)
            self._by_dynamic.setdefault(key, []).append(potential)
            dynamic_producers.setdefault(key, {}).setdefault(
                issue_occurrence(potential.producer), potential.producer)
        self._dynamic_producers = {
            key: dict(producers)
            for key, producers in dynamic_producers.items()
        }

    def __len__(self):
        return len(self._potentials)

    def __iter__(self):
        return iter(self._potentials)

    def sites(self):
        return tuple(sorted(self._by_site))

    def at(self, block, pos, counter=None, frame=None):
        if counter is not None and frame is not None:
            return tuple(self._by_dynamic.get((block, pos, frame, counter), ()))
        out = [p for p in self._potentials if p.block == block and p.pos == pos]
        if counter is not None:
            out = [p for p in out if p.counter == counter]
        if frame is not None:
            out = [p for p in out if p.frame == frame]
        return tuple(out)

    def for_site(self, site):
        return tuple(self._by_site.get(tuple(site), ()))

    def producers_at(self, block, pos, counter, frame):
        """Unique producer requirements; many diagnostic relations may name the same issue."""
        producers = self._dynamic_producers.get((block, pos, frame, counter), {})
        return tuple(sorted(producers.values(), key=_issue_order))

    def producer_map_at(self, block, pos, counter, frame):
        """``{issue occurrence: representative key}`` for one dynamic potential site."""
        return self._dynamic_producers.get((block, pos, frame, counter), {})

    def issue(self, block, pos, frame):
        return self._issues.get((block, pos, frame))

    def issues(self):
        return dict(self._issues)

    @property
    def unresolved(self):
        return self._unresolved


class PotentialWaits(Analysis):
    """Stamp all frame/register hazards before either directional counter analysis."""

    def run(self, prog, am):
        fm = am.get(FrameMap(), prog)
        hazards = am.get(FrameHazards(), prog)
        fences = fences_of(prog)
        potentials = []

        def add(hazard, counter, fp, fc):
            if counter is None:
                return
            block, pos, at_producer = anchor_site(
                prog, hazard, fences, counter, fm=fm, fp=fp, fc=fc)
            frame = fp if at_producer or block == hazard.producer.block else fc
            if frame is None:
                return
            potentials.append(PotentialWait(
                block=block, pos=pos, frame=frame, counter=counter,
                producer=_key(hazard, counter, fp, fm),
                producer_frame=fp, consumer_frame=fc, hazard=hazard,
                at_producer=at_producer))

        for fp, fc, hazard in hazards.instances():
            add(hazard, counter_for(hazard, prog), fp, fc)

        register_hazards = am.get(RegHazards(), prog)
        for hazard in register_hazards.edges(kind="RAW"):
            if not is_register_fill(hazard, prog):
                continue
            for fp, fc in _register_frame_pairs(hazard, fm):
                add(hazard, DSCNT, fp, fc)

        # Deduplicate exact dynamic requirements while preserving distinct register/storage
        # identities carried by different hazards.
        unique = {}
        for potential in potentials:
            key = (potential.block, potential.pos, potential.frame, potential.counter,
                   potential.producer, potential.hazard.kind,
                   potential.consumer_frame, potential.hazard.consumer.block,
                   potential.hazard.consumer.pos)
            unique.setdefault(key, potential)
        potentials = list(unique.values())

        keys_by_issue = {}
        for potential in potentials:
            producer = potential.producer
            key = (producer.block, producer.pos, producer.frame, producer.counter)
            keys_by_issue.setdefault(key, set()).add(producer)

        counts = (prog.meta or {}).get(FANOUT_META) or {}
        issues = {}
        for block in prog.blocks.values():
            if block.model_only:
                continue
            for frame in fm.frames(block.label):
                for pos, node in enumerate(block.body):
                    if not isinstance(node, Move):
                        continue
                    counter = counter_of(node)
                    if counter is None:
                        continue
                    keys = tuple(sorted(
                        keys_by_issue.get((block.label, pos, frame, counter), ()),
                        key=_issue_order))
                    issues[(block.label, pos, frame)] = IssueInfo(
                        counter, fan_out(node, counts), keys)

        unresolved = tuple(hazards.unresolved()) + tuple(fm.unresolved)
        return PotentialWaitSet(potentials, issues, unresolved)

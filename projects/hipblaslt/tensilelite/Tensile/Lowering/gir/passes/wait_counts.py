# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Prepare and annotate GIR's explicit LoopWaitData transport metadata.

The finalized GIR names producer/consumer relations; StinkyTofu computes hardware wait immediates
and performs redundancy pruning after scheduling.
"""

from __future__ import annotations

from .base import Pass
from ..nodes import Mark, Mma, Move, descriptor_unit
from ..analyses.lds_buffers import LdsBufferIds
from ..analyses.wait_counts import WaitCounts


_ACCESS_WRITE = 1 << 0
_ACCESS_ABSOLUTE = 1 << 1
_ACCESS_STRIDE = 5
_DEPENDENCY_STRIDE = 8


def _nodes(prog):
    """GIR operations in stable program/body order, after all semantic Mark placement."""
    for block in prog.blocks.values():
        for node in block.body:
            if isinstance(node, (Move, Mma, Mark)):
                yield node


def _shared_touches(node):
    if not isinstance(node, Move):
        return
    for ref in node.srcs:
        if ref.tile.space == "shared":
            yield ref, False
    for ref in node.dsts:
        if ref.tile.space == "shared":
            yield ref, True


def _class_key(prog, ref):
    """The physical completion class, including a Phi-fused descriptor unit."""
    return tuple(descriptor_unit(prog, (ref.tile.operand,)))


def _region_coords(geometry, ref, is_write):
    """Every concrete region named by a possibly covering shared Ref."""
    coordinates = [()]
    for values in geometry.regions_of(ref, is_write):
        coordinates = [prefix + (int(value),)
                       for prefix in coordinates for value in sorted(values)]
    return tuple(coordinates)


def _access_tables(prog, storage):
    touches = [(ref, is_write, _class_key(prog, ref), region)
               for node in _nodes(prog)
               for ref, is_write in _shared_touches(node)
               for region in _region_coords(storage.geometry, ref, is_write)]
    classes = sorted({class_key for _ref, _write, class_key, _region in touches})
    regions = sorted({(class_key, ref.tile.operand, region)
                      for ref, _write, class_key, region in touches})
    return (touches, {key: i for i, key in enumerate(classes)},
            {key: i for i, key in enumerate(regions)})


def _accesses_for(prog, node, storage, class_ids, region_ids):
    records = []
    for ref, is_write in _shared_touches(node):
        class_key = _class_key(prog, ref)
        absolute = getattr(ref, "abs_gen", None) is not None
        relation = int(ref.abs_gen) if absolute else int(getattr(ref, "gdelta", 0))
        flags = (_ACCESS_WRITE if is_write else 0) | (_ACCESS_ABSOLUTE if absolute else 0)
        for region in _region_coords(storage.geometry, ref, is_write):
            records.extend((
                class_ids[class_key],
                region_ids[(class_key, ref.tile.operand, region)],
                int(storage.depth_of(ref.tile.operand)),
                relation,
                flags,
            ))
    assert len(records) % _ACCESS_STRIDE == 0
    return tuple(records)


class LoopWaitMetadataPass(Pass):
    """Assign stable op ids and deterministic shared-access records."""

    def run(self, prog, am):
        storage = am.get(LdsBufferIds(), prog)
        _touches, class_ids, region_ids = _access_tables(prog, storage)
        changed = False
        for op_id, node in enumerate(_nodes(prog)):
            accesses = (_accesses_for(prog, node, storage, class_ids, region_ids)
                        if isinstance(node, Move) else ())
            if (node.op_id != op_id or node.wait_accesses != accesses
                    or node.wait_dependencies):
                node.op_id = op_id
                node.wait_accesses = accesses
                node.wait_dependencies = ()
                changed = True
        if changed:
            prog.bump()
        return ("loop_wait_metadata",) if changed else ()


def _dependency_records(flat):
    return {tuple(flat[i:i + _DEPENDENCY_STRIDE])
            for i in range(0, len(flat), _DEPENDENCY_STRIDE)
            if len(flat[i:i + _DEPENDENCY_STRIDE]) == _DEPENDENCY_STRIDE}


class WaitDependencyPass(Pass):
    """Annotate each existing wait anchor with explicit frame-hazard dependencies."""

    def run(self, prog, am):
        # Keep every semantic anchor. WaitCounts.essential() prunes a later
        # site when an earlier numeric wait covers it in GIR's pre-scheduling
        # order; StinkyTofu scheduling can reorder counter producers between
        # those sites and invalidate that proof. The post-scheduling dataflow
        # owns redundancy elimination now.
        charged = am.get(WaitCounts(), prog)
        if not len(charged):
            return prog
        by_block = {}
        for site in charged:
            by_block.setdefault(site.block, {}).setdefault(site.pos, []).append(site)

        changed = False
        for label, at_pos in by_block.items():
            block = prog.block(label)
            if block.model_only:
                continue  # nothing emits it, so nothing waits
            body = block.body
            for pos in sorted(at_pos):
                sites = at_pos[pos]
                if pos >= len(body):
                    continue
                node = body[pos]
                records = _dependency_records(node.wait_dependencies)
                records.update(dep.flatten() for site in sites for dep in site.dependencies)
                flattened = tuple(value for record in sorted(records) for value in record)
                if flattened != node.wait_dependencies:
                    node.wait_dependencies = flattened
                    changed = True
        if changed:
            prog.bump()
        return prog


class LegacyWaitCntPass(Pass):
    """Peter's original flow: materialize GIR's pre-scheduling numeric residuals."""

    def run(self, prog, am):
        charged = am.get(WaitCounts(), prog).essential()
        if not len(charged):
            return prog
        by_block = {}
        for site in charged:
            by_block.setdefault(site.block, {}).setdefault(site.pos, []).append(site)

        for label, at_pos in by_block.items():
            block = prog.block(label)
            if block.model_only:
                continue
            body = block.body
            for pos in sorted(at_pos, reverse=True):
                sites = at_pos[pos]
                if pos >= len(body):
                    continue
                fact = {site.counter: site.n for site in sites}
                lead = min(sites, key=lambda site: (site.n, site.counter))
                raw = {token for site in sites for token in site.tokens}
                fact.update(
                    hazard=lead.kind,
                    **{"from": lead.producer},
                    frames=max(site.frames for site in sites),
                    tokens=tuple(sorted(raw)),
                    order_tokens=tuple(sorted(
                        {token for site in sites for token in site.order_tokens} - raw)),
                )
                body.insert(pos, Mark("waitcnt", fact))
        prog.bump()
        return prog


# Preserve the historical public name for the numeric reference flow.
WaitCntPass = LegacyWaitCntPass

# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
TailHandoff -- the state the GIR main loop leaves for the SCAFFOLD-OWNED tail loop.

The `K % DepthU` remainder is emitted by `KernelWriter.py`, not by GIR (#94): the scaffold already
loads it, writes it, barriers and rolls over its `MatrixInstK` slabs, and its TDM path already
clamps the descriptor's K extent to the remainder. GIR owns no tail block. What GIR does own is
the state that code inherits, and the tail re-initializes only some of it -- so the rest is a
contract, checked here.

SCOPE, because getting this wrong reads as a clean kernel: only the STEADY path is checked. The
conditional arms -- the split prefetch guard and the folded `T < M` chain -- deliberately advance
the descriptor on a path that issues no load, which is what `chunk_pin` exists for, so counting
their loads against their advances measures the mechanism rather than a defect. Their agreement is
already enforced by `ValuePlacementSolver`'s join conflict, and their entry step is the fold's own
recorded obligation.
"""

from __future__ import annotations

from ..nodes import Mark, LoopBack, successor_labels


def _trips(blk):
    """`(var, sub, div)` of a loop block's dynamic trip count, or None when it is not a loop."""
    term = blk.term
    return (term.trips.var, term.trips.sub, term.trips.div) if isinstance(term, LoopBack) else None


def _block_state(blk):
    """`(last absolute chunk per descriptor, generation left per pointer)` from one block's marks."""
    chunks, gens = {}, {}
    for node in blk.body:
        if not isinstance(node, Mark):
            continue
        at = node.at if isinstance(node.at, dict) else {}
        if node.kind == "gr_increment":
            chunks[tuple(at.get("unit") or ())] = int(at["to_chunk"])
        elif node.kind == "swap":
            hop = at.get("hop")
            key = (hop, at.get("operand") if hop == "read" else tuple(at.get("unit") or ()))
            gens[key] = int(at["gen_to"])
    return chunks, gens


def exit_paths(prog):
    """Every acyclic entry->exit block sequence."""
    out = []

    def walk(label, seen, acc):
        if label not in prog.blocks:
            out.append(tuple(acc))                 # left the GIR program: this is an exit
            return
        if label in seen:
            return
        acc.append(label)
        for succ in successor_labels(prog.blocks[label]):
            if succ != label:
                walk(succ, seen | {label}, acc)
        acc.pop()

    walk(prog.entry, frozenset(), [])
    return out


def exit_state(prog):
    """`{path: (chunk in the loop's own frame, generations, trips)}` for each way out.

    The chunk is the loop block's `to_chunk`, which `GrIncrementRegions` states on an absolute
    timeline; the trip re-frames it, which `_chunk_violation` undoes rather than re-deriving."""
    states = {}
    for path in exit_paths(prog):
        chunks, gens, trips = {}, {}, None
        for label in path:
            blk = prog.blocks[label]
            block_chunks, block_gens = _block_state(blk)
            if _trips(blk):
                trips, chunks = _trips(blk), dict(block_chunks)
            gens.update(block_gens)
        states[path] = (chunks, gens, trips)
    return states


def _chunk_violation(path, unit, to_chunk, trips):
    """After the last trip the descriptor names `to_chunk + trips - 1`, and that must be `T`.

    With `trips = (T - sub) // div` that holds for every `T` exactly when `div` is 1 and the loop
    body's chunk is one past the peel depth -- the tail then loads the remainder the loop left."""
    var, sub, div = trips
    if div == 1 and to_chunk == sub + 1:
        return None
    return ("H-CHUNK: the path %s leaves the descriptor for %r naming chunk %d + (%s - %d)/%d - 1 "
            "on the last trip, which is not %s -- the tail loads the chunk the descriptor names, "
            "so it must sit exactly one past the last chunk the main loop loaded"
            % (" -> ".join(path), unit, to_chunk, var, sub, div, var))


def _buffer_violations(path, gens):
    """The tail writes its chunk to LDS through the descriptor and reads it back through the read
    pointer, so when it rebinds neither the two must already name one buffer.

    A pipelined loop deliberately leaves them apart -- that is what double buffering is -- so this
    is a violation only for a tail that does not reset, which is why the caller passes the gate."""
    out = []
    for (hop, operand), read_gen in sorted(gens.items(), key=str):
        if hop != "read":
            continue
        for (other_hop, unit), copy_gen in sorted(gens.items(), key=str):
            if other_hop != "copy" or operand not in unit or copy_gen == read_gen:
                continue
            out.append(
                "H-BUFFER: the path %s leaves %r reading LDS buffer %d while the descriptor that "
                "fills it writes buffer %d, and the tail rebinds neither at a coalesced read "
                "count of 1 -- so its reads would miss the chunk it just loaded"
                % (" -> ".join(path), operand, read_gen, copy_gen))
    return out


def _fold_violations(prog):
    """A folded short arm owes its entry step to the scaffold, and must say so.

    Folding collapses the `T < M` arm onto the drain chain, which is only correct when the chain is
    entered at step `M - T`. GIR does not emit that selection, so the obligation has to be carried
    where a reader can find it rather than assumed."""
    fold = prog.meta.get("short_loop")
    if fold and fold.get("verdict") == "folded" and not fold.get("obligations"):
        return ["H-FOLD: the `T < M` arm is folded onto the drain chain but records no obligation "
                "-- entering that chain at step 0 rather than M-T leaves the descriptor short, and "
                "nothing here would say who supplies the entry step"]
    return []


def handoff_violations(prog, tail_resets_lds=False):
    """[] or the human-readable ways this program breaks the state the scaffold tail inherits.

    `tail_resets_lds` is the scaffold's `numReadsIterCoalesced > 1` gate, under which
    `tdmResetTailLdsBuffer` and `recalcLocalReadAddressesAB` put both pointers back on buffer 0.
    It is kernel state GIR cannot see, so it is an argument rather than something derived here."""
    out = _fold_violations(prog)
    for path, (chunks, gens, trips) in exit_state(prog).items():
        if trips is None:
            continue                       # a conditional arm; see the module docstring on scope
        for unit, to_chunk in sorted(chunks.items(), key=str):
            bad = _chunk_violation(path, unit, to_chunk, trips)
            if bad:
                out.append(bad)
        if not tail_resets_lds:
            out += _buffer_violations(path, gens)
    return out

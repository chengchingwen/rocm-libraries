# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
The control-flow graph of an emitted kernel.

A hazard that occurs on ANY reachable path is a hazard, so the executor must walk paths rather than
one straight line.  This module supplies the graph: basic blocks keyed by label, and the successors
each block's terminator names.  It decides nothing about state.
"""

from __future__ import annotations

import re

from .program import LABEL_RE, BRANCH_RE, DecodeError, decode_range

#: Out of scope, so the walk stops here: the epilogue (accumulator -> global) and the tail loop.
#: LoopModel does not emit the tail (task #94), so its TDM loads carry no copy tag to decode.
_EPILOGUE = re.compile(r"^label_(GW_|Summation_End|GSUC_TL|TDMResetTail|TailLoop|SkipTailLoop)")

#: A branch too far for `s_cbranch`: the conditional skips this, so reaching it means jumping.
_LONGJUMP = re.compile(r"^\s*s_setpc_b64\b")
_LONGJUMP_TARGET = re.compile(r"^\s*s_add_i32\s+\S+,\s*(label_\w+)\s*,")


class Block:
    """One basic block: its decoded body and the labels control may reach from it."""

    __slots__ = ("label", "lo", "hi", "body", "succs", "terminator")

    def __init__(self, label, lo, hi, body, succs, terminator):
        self.label, self.lo, self.hi = label, lo, hi
        self.body, self.succs, self.terminator = body, succs, terminator

    @property
    def is_exit(self):
        return not self.succs

    def __repr__(self):
        return "Block(%s, %d insts -> %s)" % (self.label, len(self.body), list(self.succs))


def _long_jump_target(lines, lo, i):
    """The label a `getpc`/`add`/`setpc` trampoline jumps to, read off the `s_add_i32` operand."""
    target = None
    for j in range(lo, i):
        m = _LONGJUMP_TARGET.match(lines[j].split("//")[0])
        if m:
            target = m.group(1)
    if target is None:
        raise ValueError("line %d: s_setpc_b64 with no s_add_i32 naming its label, so where this "
                         "long branch goes cannot be read off the kernel" % (i + 1))
    return target


def _terminator(lines, lo, hi):
    """`(index, opcode, target)` of the block's branch, or None when it falls through."""
    for i in range(lo, hi):
        code = lines[i].split("//")[0]
        if BRANCH_RE.match(code):
            return i, code.split()[0].strip(), code.split()[-1].strip()
        if _LONGJUMP.match(code):
            # A LONG BRANCH IS A TRAMPOLINE, not a fall-through: the conditional above it skips
            # over this, so reading it as one invents an edge to the next label and loses the jump.
            return i, "s_branch", _long_jump_target(lines, lo, i)
    return None


def _leaders(lines, hi):
    """Line indices that start a block: every label, and every line after a branch."""
    out = {0}
    for i in range(hi):
        if LABEL_RE.match(lines[i]):
            out.add(i)
        elif BRANCH_RE.match(lines[i].split("//")[0]) or _LONGJUMP.match(lines[i].split("//")[0]):
            out.add(i + 1)
    return sorted(x for x in out if x < hi)


def _label_at(lines, i):
    m = LABEL_RE.match(lines[i])
    return m.group(1) if m else "_L%d" % (i + 1)


def build_cfg(lines, entry=0, stop=None):
    """`{label: Block}` over `lines[entry:]`, stopping at the epilogue.

    A conditional names two successors -- its target and the fall-through -- and an unconditional
    names one.  A block whose label is the epilogue is kept as an EXIT: reaching it is legal, what
    it then does is out of scope.

    `entry` starts a block of its own and nothing above it is built, so a branch back into that
    region simply has no successor to name."""
    hi = stop if stop is not None else len(lines)
    starts = [s for s in _leaders(lines, hi) if s >= entry]
    if entry < hi and entry not in starts:
        starts.insert(0, entry)
    ends = starts[1:] + [hi]
    at_line = {}
    blocks = {}
    for lo, end in zip(starts, ends):
        label = _label_at(lines, lo)
        at_line[lo] = label
        blocks[label] = (lo, end)

    out = {}
    for label, (lo, end) in blocks.items():
        if _EPILOGUE.match(label):
            out[label] = Block(label, lo, end, [], (), None)
            continue
        term = _terminator(lines, lo, end)
        try:
            body = decode_range(lines, lo, end, allow_branches=True)
        except DecodeError as refusal:
            # The tail loop is the scaffold's, not GIR's, so its loads carry no copy tag. Treat
            # the first one as an EXIT, like the epilogue: reaching it is legal, what it then does
            # is out of scope. Any other refusal stays fatal.
            at = getattr(refusal, "scaffold_line", None)
            if at is None:
                raise
            out[label] = Block(label, lo, end,
                               decode_range(lines, lo, at - 1, allow_branches=True), (), None)
            continue
        succs = []
        if term is None:
            nxt = at_line.get(end)
            if nxt is not None:
                succs.append(nxt)
        else:
            _i, op, target = term
            if target in blocks:
                succs.append(target)
            if not op.startswith("s_branch"):            # conditional: the fall-through too
                nxt = at_line.get(end)
                if nxt is not None:
                    succs.append(nxt)
        out[label] = Block(label, lo, end, body, tuple(dict.fromkeys(succs)), term)
    return out


def entry_label(cfg, lines):
    """The block the kernel starts in -- the one covering the first line."""
    return min(cfg.values(), key=lambda b: b.lo).label


def back_edges(cfg, entry):
    """`{(src, dst)}` whose destination is on the DFS stack -- the loops."""
    seen, stack, out = set(), [], []

    def walk(label):
        seen.add(label)
        stack.append(label)
        for s in cfg[label].succs:
            if s in stack:
                out.append((label, s))
            elif s not in seen:
                walk(s)
        stack.pop()

    if entry in cfg:
        walk(entry)
    return set(out)

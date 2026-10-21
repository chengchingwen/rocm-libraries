# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
Symbolic execution of an emitted gfx1250 main loop.

`check(path)` runs the kernel's prologue and steady loop on a model of the completion counters, LDS
buffers and register file, and reports the accesses whose data had not landed.  Facts come from the
instructions -- opcode width, immediate offset, registers, `s_wait_<c> N` -- and from the
`<GIR: ...>` tag only where the instruction genuinely does not carry them, which is the tensor an
LDS access belongs to.

Control flow is walked, not assumed: a conditional forks the machine state and both arms run, so
the drain and the no-load-loop arms are answered for rather than skipped.  The tail loop and the
epilogue are out of scope and the walk stops at them.

Two things it does NOT do, both deliberate and both refusals rather than guesses: it does not
evaluate per-lane addresses, so it answers nothing about two WAVES colliding (see `checks`); and it
refuses a kernel whose buffer-select register is written by anything but a single-bit xor, or whose
access width or descriptor it cannot read.
"""

from __future__ import annotations

from .cfg import build_cfg
from .checks import (COUNTER_UNBOUNDED, OVERWRITE_BEFORE_READ, READ_BEFORE_LAND,
                     READ_UNWRITTEN, WMMA_UNLANDED, Executor, Finding, Report, check_paths,
                     execute, gir_disagreement, observed_pairs, report_paths)
from .machine import Access, Event, Machine, ModelError
from .program import DecodeError, Inst, decode, decode_prologue, decode_range, loop_body
from .walk import PathResult, walk, walk_all_entries

__all__ = ["check", "check_paths", "report_paths", "Report", "execute", "observed_pairs",
           "gir_disagreement", "Executor", "Machine",
           "Access", "Event", "Finding", "Inst", "decode", "decode_prologue", "decode_range",
           "loop_body", "DecodeError", "ModelError", "READ_BEFORE_LAND", "OVERWRITE_BEFORE_READ",
           "WMMA_UNLANDED", "READ_UNWRITTEN", "COUNTER_UNBOUNDED",
           "build_cfg", "walk", "walk_all_entries", "PathResult"]


def check(path, ring=2, head="label_LoopBeginL"):
    """`[Finding]` over every reachable path of `path` -- prologue, loop, drain and both NLL arms.

    Not one trace: a straight-line run answers for one arm of every branch it passes, and the
    kernel's drain sits behind several.
    """
    lines = open(path, errors="replace").read().splitlines()
    return check_paths(lines, ring=ring, head=head)

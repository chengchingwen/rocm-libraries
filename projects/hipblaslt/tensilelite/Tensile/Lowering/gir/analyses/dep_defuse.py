# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
DepDefuse -- the RAW + WAR dependency edges carried from the LoopIR awaits.
"""

from __future__ import annotations

from ....LoopModel.ir import is_war
from ..nodes import Move, Mma
from ..analysis import Analysis


class DepDefuse:
    """The RAW and WAR edges of each instruction, indexed by that instruction."""
    def __init__(self):
        self._raw = {}       # id(inst) -> [(dep, counter)]
        self._war = {}       # id(inst) -> [(dep, counter)]

    def add(self, inst):
        raw, war = [], []
        for dep, counter, kind, _scope in getattr(inst, "deps", ()):
            (war if is_war(kind) else raw).append((dep, counter))
        self._raw[id(inst)] = raw
        self._war[id(inst)] = war

    def raw(self, inst):
        return self._raw.get(id(inst), [])

    def war(self, inst):
        return self._war.get(id(inst), [])


class DepDefuseAnalysis(Analysis):
    """Index every Move/Mma's awaits into a `DepDefuse` the passes can query."""
    @property
    def cache_key(self):
        return "DepDefuse"

    def run(self, prog, am):
        res = DepDefuse()
        for blk in prog.blocks.values():
            for inst in blk.body:
                if isinstance(inst, (Move, Mma)):
                    res.add(inst)
        return res

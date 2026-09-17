# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Readable names for the flat GIR -> StinkyTofu LoopWaitData transport."""

from enum import IntEnum, IntFlag


class LoopWaitAccessField(IntEnum):
    CLASS_ID = 0
    REGION_ID = 1
    RING_SIZE = 2
    GENERATION_RELATION = 3
    FLAGS = 4
    COUNT = 5


class LoopWaitAccessFlag(IntFlag):
    NONE = 0
    WRITE = 1 << 0
    ABSOLUTE = 1 << 1


class LoopWaitDependencyField(IntEnum):
    PRODUCER_OP_ID = 0
    CONSUMER_OP_ID = 1
    PRODUCER_FRAME_ID = 2
    CONSUMER_FRAME_ID = 3
    GENERATION_GAP = 4
    COUNTER = 5
    KIND = 6
    SCOPE = 7
    COUNT = 8


class LoopWaitCounter(IntEnum):
    DS = 0
    TENSOR = 3


class LoopWaitDependencyKind(IntEnum):
    RAW = 0
    WAR = 1
    WAW = 2


class LoopWaitScope(IntEnum):
    WAVE = 0
    WORKGROUP = 1

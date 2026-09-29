# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""UseLoopModel lowering: LoopIR to GIR and the associated analyses."""

from .loopir_to_gir import lower_to_gir, build_gir, gir_text

__all__ = ["lower_to_gir", "build_gir", "gir_text"]

# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""UseLoopModel lowering: LoopIR to GIR and the associated analyses."""

from .loopir_to_gir import build_gir, gir_text, lower_to_gir, steady_chain

__all__ = ["build_gir", "gir_text", "lower_to_gir", "steady_chain"]

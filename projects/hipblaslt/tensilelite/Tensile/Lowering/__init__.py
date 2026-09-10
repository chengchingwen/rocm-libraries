# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
Tensile.Lowering -- the UseLoopModel lowering stack.

 loopir_to_gir layer 1 (LoopModel LoopIR) -> layer 2 (GIR); build_gir = build-once (R-ONCE)
 gir/ the GIR graph, analyses, passes, verifier (pure Python; no rocisa)
"""

from .loopir_to_gir import (lower_to_gir, lower_params_to_gir, build_gir,
                            gir_text)

__all__ = ["lower_to_gir", "lower_params_to_gir", "build_gir", "gir_text"]

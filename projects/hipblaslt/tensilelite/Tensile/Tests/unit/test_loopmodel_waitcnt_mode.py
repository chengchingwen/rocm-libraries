# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import pytest

from Tensile.Common.GlobalParameters import defaultSolution
from Tensile.Common.ValidParameters import validParameters
from Tensile.Lowering.gir.passes.pipeline import pipeline
from Tensile.Lowering.gir.passes.wait_counts import WaitCntPass


def test_loopmodel_waitcnt_mode_surface_and_pipeline():
    assert validParameters["LoopModelWaitCntMode"] == ["StinkyTofu", "GIR"]
    assert defaultSolution["LoopModelWaitCntMode"] == "StinkyTofu"

    reference = pipeline("GIR")
    assert isinstance(reference[-1], WaitCntPass)

    stinky = pipeline("StinkyTofu")
    assert not any(isinstance(item, WaitCntPass) for item in stinky)


def test_loopmodel_waitcnt_mode_rejects_unknown_value():
    with pytest.raises(ValueError, match="unknown LoopModel wait-count mode"):
        pipeline("unknown")

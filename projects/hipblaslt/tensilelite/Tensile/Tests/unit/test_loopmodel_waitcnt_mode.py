# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from Tensile.Common.GlobalParameters import defaultSolution
from Tensile.Common.ValidParameters import validParameters
from Tensile.Lowering.gir.analyses.wait_counts import WaitSite
from Tensile.Lowering.gir.loop_wait import (
    LoopWaitAccessField,
    LoopWaitAccessFlag,
    LoopWaitCounter,
    LoopWaitDependencyField,
    LoopWaitDependencyKind,
    LoopWaitScope,
)
from Tensile.Lowering.gir.nodes import Block, Mark, Move, Program, Ref, Tile
from Tensile.Lowering.gir.passes.pipeline import pipeline
from Tensile.Lowering.gir.passes.wait_counts import (
    LegacyWaitCntPass,
    LoopWaitMetadataPass,
    WaitDependencyPass,
)


def test_loopmodel_waitcnt_mode_surface_and_pipeline():
    assert validParameters["LoopModelWaitCntMode"] == ["StinkyTofu", "GIR"]
    assert defaultSolution["LoopModelWaitCntMode"] == "StinkyTofu"

    enhanced = pipeline("StinkyTofu")
    assert isinstance(enhanced[-2], LoopWaitMetadataPass)
    assert isinstance(enhanced[-1], WaitDependencyPass)

    reference = pipeline("GIR")
    assert isinstance(reference[-1], LegacyWaitCntPass)
    assert not any(isinstance(p, LoopWaitMetadataPass) for p in reference)


def test_loop_wait_transport_enums_pin_flat_schema():
    assert int(LoopWaitAccessField.COUNT) == 5
    assert int(LoopWaitDependencyField.COUNT) == 8
    assert int(LoopWaitAccessFlag.WRITE | LoopWaitAccessFlag.ABSOLUTE) == 3
    assert int(LoopWaitCounter.DS) == 0
    assert int(LoopWaitCounter.TENSOR) == 3
    assert int(LoopWaitDependencyKind.RAW) == 0
    assert int(LoopWaitDependencyKind.WAR) == 1
    assert int(LoopWaitDependencyKind.WAW) == 2
    assert int(LoopWaitScope.WAVE) == 0
    assert int(LoopWaitScope.WORKGROUP) == 1


def test_legacy_mode_materializes_numeric_wait_mark():
    read = Move(
        (Ref(Tile("A", "shared"), abs_gen=0),),
        (Ref(Tile("A", "register")),),
    )
    prog = Program(blocks={"entry": Block("entry", body=[read])})
    site = WaitSite(
        block="entry",
        pos=0,
        counter="tensorcnt",
        n=2,
        kind="RAW",
        producer="A",
        frames=1,
        tokens=(4,),
    )

    class Charged:
        def essential(self):
            return self

        def __len__(self):
            return 1

        def __iter__(self):
            return iter((site,))

    class FakeAnalysisManager:
        def get(self, _analysis, _prog):
            return Charged()

    LegacyWaitCntPass().run(prog, FakeAnalysisManager())

    wait = prog.block("entry").body[0]
    assert isinstance(wait, Mark)
    assert wait.kind == "waitcnt"
    assert wait.at["tensorcnt"] == 2
    assert wait.at["tokens"] == (4,)

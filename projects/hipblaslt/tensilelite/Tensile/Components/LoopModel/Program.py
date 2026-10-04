# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Build and cache the finalized GIR program for a LoopModel kernel."""

from ...LoopModel import traversal as geometry
from ...LoopModel.emit import emit_mainloop
from ...LoopModel.render import render_theta
from ...Lowering import build_gir, gir_text
from ...Lowering.gir import check_plan
from ...Lowering.gir.analyses import Gl2PrefetchRegions
from ...Lowering.gir.frame_contract import install_frame_contract
from ...Lowering.gir.passes import pipeline as gir_pipeline

from .Theta import loopModelTheta, namingSolution


def loopModelLoopCopies(theta, depths) -> int:
    """Steady blocks one trip unrolls into -- read off the rings, not off a parameter name.

    A ring the trip does not close keeps turning into the next one, so the body is one fixed
    piece of code only once it is replicated through a whole turn.  `VG=2, VA=3` -- the 1.5
    buffers HalfPLR used to name -- asks for three; a ring that divides asks for one.
    """
    return geometry.ring_trip_copies(theta, depths)


def loopModelGirProgram(writer, kernel):
    """Build and cache the finalized GIR program for one kernel."""
    key = getattr(writer.states, "kernelName", None) or id(kernel)
    cache = getattr(writer, "_loopModelGirCache", None)
    if cache is not None and cache[0] == key:
        return cache[1]

    theta, _S = loopModelTheta(writer, kernel)
    with namingSolution(writer, kernel):
        # PrefetchGL2 is placed by an analysis but is not modeled in theta. GIR only
        # chooses where its existing issue/increment pair is emitted.
        mainloop = emit_mainloop(theta)
        prog = build_gir(
            theta,
            mainloop=mainloop,
            params={Gl2PrefetchRegions.PARAM: kernel["PrefetchGL2"]},
            pipeline=gir_pipeline(),
            loop_copies=loopModelLoopCopies(theta, _S),
        )
        violations = check_plan(prog)
        if violations:
            # EVERY semantic violation is a LOWERING DEFECT and stops the build.  A stated
            # (VgprGroup, VgprAlloc) does not downgrade one to a per-kernel skip: a dial can ask
            # for a ring this shape cannot carry, but it cannot ask for the wrong answer.
            raise ValueError(
                "UseLoopModel: the GIR plan fails its own semantic check (%d violation(s)) — "
                "a decoder/lowering defect to repair, do NOT gate it off.\n  %s"
                % (len(violations), "\n  ".join(violations[:8])))
    writer._loopModelMainloopCache = (key, mainloop)
    writer._loopModelGirCache = (key, prog)
    return prog


def loopModelGirProgramCached(writer):
    """Return the cached program for the current named kernel, if any."""
    cache = getattr(writer, "_loopModelGirCache", None)
    if not cache:
        return None
    # Only the name is verifiable here; ids can be recycled by CPython.
    name = getattr(writer.states, "kernelName", None)
    return cache[1] if (name and cache[0] == name) else None


def loopModelIrTextCached(writer):
    """Render the target-aware theta and LoopIR that produced the cached GIR."""
    name = getattr(writer.states, "kernelName", None)
    theta_cache = getattr(writer, "_loopModelThetaCache", None)
    mainloop_cache = getattr(writer, "_loopModelMainloopCache", None)
    if not (
        name
        and theta_cache
        and mainloop_cache
        and theta_cache[0] == name
        and mainloop_cache[0] == name
    ):
        return "# LoopModel IR unavailable: this kernel has no cached target-aware theta/IR\n"
    try:
        theta, _S = theta_cache[1]
        return render_theta(theta, mainloop_cache[1])
    except Exception as exc:
        return f"# LoopModel IR unavailable (cached render error): {exc}\n"


def loopModelGirTextCached(writer):
    """Render the GIR emitted by the cached program."""
    return gir_text(loopModelGirProgramCached(writer))


def attachFrameContract(writer, stModule, kernel):
    """Attach this kernel's frame contract to the StinkyTofu module."""
    install_frame_contract(loopModelGirProgram(writer, kernel), stModule)

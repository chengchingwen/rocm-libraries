# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Build and cache the finalized GIR program for a LoopModel kernel."""

from ...Common.GlobalParameters import globalParameters
from ...LoopModel import traversal as geometry
from ...LoopModel.checks import validate_loopir
from ...LoopModel.decoder import decode_theta
from ...LoopModel.loopir import render as render_loopir
from ...LoopModel.render import render_geometry
from ...Lowering import gir_text
from ...Lowering.loopir_to_gir import lower_to_gir
from ...Lowering.gir import check_plan
from ...Lowering.gir.analyses import Gl2PrefetchRegions
from ...Lowering.gir.frame_contract import install_frame_contract
from ...Lowering.gir.passes import pipeline as gir_pipeline
from ...Lowering.gir.passes import run_pipeline

from .Theta import loopModelTheta, namingSolution


def loopModelGirProgram(writer, kernel):
    """Build and cache the finalized GIR program for one kernel."""
    key = getattr(writer.states, "kernelName", None) or id(kernel)
    verify_ir = bool(globalParameters.get("LoopModelIRVerify", True))
    include_dependencies = verify_ir or bool(
        globalParameters.get("OutputLoopIR", False)
    )
    cache = getattr(writer, "_loopModelGirCache", None)
    if cache is not None and cache[0] == key:
        cached_verified = cache[2] if len(cache) > 2 else True
        cached_dependencies = cache[3] if len(cache) > 3 else True
        if (
            (not verify_ir or cached_verified)
            and (not include_dependencies or cached_dependencies)
        ):
            return cache[1]

    theta, _S = loopModelTheta(writer, kernel)
    with namingSolution(writer, kernel):
        decoded = decode_theta(theta)
        plans = decoded.profile.schedule
        missing = []
        for operand in theta.operands:
            if operand.name not in ("A", "B") or operand.trajectory.shared_read is None:
                continue
            stated_ring = any(
                kernel.get(name) is not None and int(kernel.get(name)) != -1
                for name in ("VgprGroup" + operand.name, "VgprAlloc" + operand.name))
            if not stated_ring:
                continue
            requested = geometry.requested_read_ahead(theta, operand, _S)
            if requested <= 0:
                continue
            for group in operand.fragment.groups():
                actual = plans.plan(operand, group).prefetch_steps
                if actual == 0:
                    missing.append(
                        f"{operand.name}/{group}: requested PLR{requested}, scheduled PLR0")
        if missing:
            raise ValueError(
                "UseLoopModel: nonzero PrefetchLocalRead lost all prefetch — "
                + "; ".join(missing))
        if verify_ir:
            ir_violations = validate_loopir(
                decoded.program,
                decoded.profile,
            )
            if ir_violations:
                raise ValueError(
                    "UseLoopModel: LoopIR validation failed (%d violation(s)) — "
                    "repair the frontend, do not bypass the checker.\n  %s"
                    % (
                        len(ir_violations),
                        "\n  ".join(ir_violations[:8]),
                    )
                )
        prog = lower_to_gir(
            decoded.program,
            decoded.profile,
            include_dependencies=include_dependencies,
        )
        # PrefetchGL2 is a target-side GIR placement parameter, not LoopIR.
        prog.params[Gl2PrefetchRegions.PARAM] = kernel["PrefetchGL2"]
        run_pipeline(
            prog,
            passes=gir_pipeline(),
            verify=verify_ir,
        )
        if verify_ir:
            violations = check_plan(prog)
            if violations:
                raise ValueError(
                    "UseLoopModel: the GIR plan fails its own semantic check (%d violation(s)) — "
                    "a decoder/lowering defect to repair, do NOT gate it off.\n  %s"
                    % (len(violations), "\n  ".join(violations[:8])))
    kinds = {}
    if globalParameters.get("OutputLoopIR", False):
        for block in prog.blocks.values():
            for operation in block.body:
                for _endpoint, _counter, kind, _scope in (
                    getattr(operation, "deps", ()) or ()
                ):
                    kinds[kind] = kinds.get(kind, 0) + 1
    writer._loopModelMainloopCache = (key, decoded.program, kinds)
    writer._loopModelGirCache = (
        key,
        prog,
        verify_ir,
        include_dependencies,
    )
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
    """Render target geometry and the Program that produced the GIR."""
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
        theta, _depths = theta_cache[1]
        program = mainloop_cache[1]
        if len(mainloop_cache) > 2:
            kinds = mainloop_cache[2]
        else:
            kinds = {}
        summary = ", ".join(
            f"{name}={count}"
            for name, count in sorted(kinds.items())
        ) or "none"
        return (
            render_geometry(theta)
            + f"\n  derived_dependences: {summary}"
            + "\n\n"
            + render_loopir(program)
            + "\n"
        )
    except Exception as exc:
        return f"# LoopModel IR unavailable (cached render error): {exc}\n"


def loopModelGirTextCached(writer):
    """Render the GIR emitted by the cached program."""
    return gir_text(loopModelGirProgramCached(writer))


def attachFrameContract(writer, stModule, kernel):
    """Attach this kernel's frame contract to the StinkyTofu module."""
    install_frame_contract(loopModelGirProgram(writer, kernel), stModule)

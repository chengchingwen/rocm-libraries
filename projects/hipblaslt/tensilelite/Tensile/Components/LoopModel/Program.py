# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The finalized GIR Program for a ULM1 kernel -- built once, read many.

The prologue, steady and drain forks emit from one Program, the semantic gate runs on it once,
and the observability dumps render the Program that produced the `.s`.
"""

from ...LoopModel.emit import emit_mainloop
from ...LoopModel.render import render_theta
from ...Lowering import build_gir, gir_text
from ...Lowering.gir import check_plan
from ...Lowering.gir.analyses import Gl2PrefetchRegions
from ...Lowering.gir.frame_contract import install_frame_contract
from ...Lowering.gir.passes import pipeline as gir_pipeline

from .Theta import loopModelTheta


def loopModelGirProgram(writer, kernel):
  """Build and cache this kernel's finalized GIR Program.

  Keyed by kernel name, so a new kernel rebuilds and a kernel's three forks share one Program.
  """
  key = getattr(writer.states, "kernelName", None) or id(kernel)
  cache = getattr(writer, "_loopModelGirCache", None)
  if cache is not None and cache[0] == key:
    return cache[1]
  theta, _S = loopModelTheta(writer, kernel)
  # Presets are kernel parameters a region analysis places but θ does not model: `PrefetchGL2`
  # stages nothing and has no completion class, so GIR only decides where its existing
  # issue/increment pair lands.  The key is shared so producer and consumer cannot drift.
  mainloop = emit_mainloop(theta)
  prog = build_gir(theta, mainloop=mainloop,
                   params={Gl2PrefetchRegions.PARAM: kernel["PrefetchGL2"]},
                   pipeline=gir_pipeline())
  # The semantic gate -- "does this plan compute the right GEMM?" -- runs here and nowhere else.
  viol = check_plan(prog)
  if viol:
    raise ValueError("UseLoopModel: the GIR plan fails its own semantic check (%d violation(s)) "
                     "for kernel %s.  This is a decoder/lowering defect to repair — do NOT gate "
                     "the configuration off.\n  %s"
                     % (len(viol), key, "\n  ".join(viol[:8])))
  writer._loopModelMainloopCache = (key, mainloop)
  writer._loopModelGirCache = (key, prog)
  return prog


def loopModelGirProgramCached(writer):
  """The Program already built for the kernel just emitted, or None if it did not go through GIR.

  Never builds one: the `OutputLoopIR` dump shows what the forks emitted from.  The key is
  honoured, so a kernel that never ran GIR does not get the previous kernel's Program.
  """
  cache = getattr(writer, "_loopModelGirCache", None)
  if not cache:
      return None
  # Only the name is verifiable here; `id()` is unsafe to compare, since CPython recycles them.
  name = getattr(writer.states, "kernelName", None)
  return cache[1] if (name and cache[0] == name) else None


def loopModelIrTextCached(writer):
  """Render the target-aware theta and LoopIR that produced the cached GIR."""
  name = getattr(writer.states, "kernelName", None)
  theta_cache = getattr(writer, "_loopModelThetaCache", None)
  mainloop_cache = getattr(writer, "_loopModelMainloopCache", None)
  if not (name and theta_cache and mainloop_cache
          and theta_cache[0] == name and mainloop_cache[0] == name):
    return "# LoopModel IR unavailable: this kernel has no cached target-aware theta/IR\n"
  try:
    theta, _S = theta_cache[1]
    return render_theta(theta, mainloop_cache[1])
  except Exception as e:
    return f"# LoopModel IR unavailable (cached render error): {e}\n"


def loopModelGirTextCached(writer):
  """Render the GIR the forks emitted from; never rebuilt."""
  return gir_text(loopModelGirProgramCached(writer))


def attachFrameContract(writer, stModule, kernel):
  """Attach this kernel's frame contract to the StinkyTofu module, out of band."""
  install_frame_contract(loopModelGirProgram(writer, kernel), stModule)

# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The three scaffold seams where GIR takes over emission: prologue, steady body, drain.

Each fork replaces the scaffold's own body for that stage, leaving the writer with the
`UseLoopModel` test and one call.
"""

from rocisa.code import Module

from ...Lowering.gir_to_rocisa import GirToRocisa

from .Program import loopModelGirProgram


def girEmitStage(writer, kernel, tPA, tPB, phase, *, internalPointerSwap=False):
  """Emit one GIR stage from the cached Program.

  L3 realizes each swap Mark through the scaffold's per-operand swap primitive; no pre-bundled
  swap Module is passed.  `GirToRocisa` owns the operand -> tensorParameters map, keyed off
  `tensorChar`.
  """
  prog = loopModelGirProgram(writer, kernel)
  return GirToRocisa(writer, kernel, tPA, tPB, prog).emit_block(
      prog, phase, internalPointerSwap=internalPointerSwap)


def loopModelPrologue(writer, kernel, tensorParametersA, tensorParametersB, module, pack, packPre):
  """The prologue fork: the PLR pipe fill and the global prefetch, both GIR-owned.

  Gated on `UseLoopModel` rather than `numItersPLR`, which is 0 at PLR=0 and would leave the
  steady body reading uninitialised LDS -- the read-fill this replaces is PLR-shaped, the copies
  are not.
  """
  # The drain is forked too, so nothing downstream reads localReadDo's offset state and the
  # read-fill loop can go.  pack[]/packPre[] still need empty Modules for the steady loop.
  for plrIdx in range(0, writer.states.numItersPLR):
    packPre[plrIdx] = Module()
    pack[plrIdx] = Module()
  module.add(girEmitStage(
      writer, kernel, tensorParametersA, tensorParametersB, "prologue"))
  # The generation a single-iteration loop never consumes is its own GIR block, so the skipPGR
  # ladder brackets exactly it -- the copies, not the advances both arms need.
  guard = loopModelGirProgram(writer, kernel).meta.get("prefetch_guard")
  if guard:
    module.add(writer.openPrefetchGlobalRead2orMore(kernel, guard["gen"]))
    module.add(girEmitStage(
        writer, kernel, tensorParametersA, tensorParametersB, guard["block"]))
    for idxPgr in range(0, kernel["PrefetchGlobalRead"] + 1):
      module.add(writer.closePrefetchGlobalRead2orMore(
          kernel, tensorParametersA, tensorParametersB, idxPgr))
      if idxPgr == 1 and guard.get("skip"):
        module.add(girEmitStage(
            writer, kernel, tensorParametersA, tensorParametersB, guard["skip"]))
    module.add(girEmitStage(
        writer, kernel, tensorParametersA, tensorParametersB, guard["join"]))
  writer.states.SubTileIdx = (writer.states.SubTileIdx + 1) % kernel["numSubTiles"]


def loopModelSteadyIter(writer, kernel, tensorParametersA, tensorParametersB, module,
                        LoopModelScaffold, u, waitLWCode, syncCode):
  """One steady substep.

  The walker unrolls the whole inner loop, so the order-independent scaffold accumulates across
  u and is emitted once, after the unroll, at the last substep.
  """
  LoopModelScaffold.add(waitLWCode)
  LoopModelScaffold.add(syncCode)
  _perLW = writer.codes.perIterLocalWrite[u][1] if u < len(writer.codes.perIterLocalWrite) else None
  if _perLW is not None and _perLW.count():
    LoopModelScaffold.add(_perLW)
  if u == kernel["LoopIters"] - 1:
    # GIR owns the copies, swaps and GR-increments; each is a leaf like the read and the wmma,
    # which is what lets the prologue and drain forks be GIR-owned too.
    module.add(girEmitStage(
        writer, kernel, tensorParametersA, tensorParametersB, "steady",
        internalPointerSwap=kernel["ExpandPointerSwap"]))
    module.add(LoopModelScaffold)


def loopModelDrainIter(writer, kernel, tensorParametersA, tensorParametersB, module,
                       LoopModelDrainScaffold, u, waitLWCode, syncCode, remainPgr):
  """One drain substep, the NLL mirror of `loopModelSteadyIter`.

  Each call is one step of the staggered drain peel; GIR's drain block carries whatever copies
  that step still issues.
  """
  _lmDrainStep = (kernel["PrefetchGlobalRead"] - 1) - remainPgr
  LoopModelDrainScaffold.add(waitLWCode)
  LoopModelDrainScaffold.add(syncCode)
  # No scaffold per-iteration global read here: draining `perIterGlobalRead[u]` would re-issue
  # the peel's copies on the scaffold's KMN distribution, which a GIR schedule has no use for.
  _perLW = writer.codes.perIterLocalWrite[u][1] \
      if (writer.codes.perIterLocalWrite and u < len(writer.codes.perIterLocalWrite)) else None
  if _perLW is not None and _perLW.count():
    LoopModelDrainScaffold.add(_perLW)
  if u == kernel["LoopIters"] - 1:
    # NGLL, so internalPointerSwap is forced False; GIR places swap and gr_inc itself.
    module.add(girEmitStage(
        writer, kernel, tensorParametersA, tensorParametersB, "drain%d" % _lmDrainStep,
        internalPointerSwap=False))
    module.add(LoopModelDrainScaffold)

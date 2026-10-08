# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Derive LoopModel Valu-ring register allocation from cached theta and S."""

from ...LoopModel import traversal as _geometry

from ..TDMSplit import AXIS_DU, AXIS_MT, split_of
from .Theta import loopModelTheta


def operandValuPuVaVg(writer, kernel, tc, bpe):
  """Registers in one buffer, and the buffer count, for one UseLoopModel operand.

  The operand's final Valu size is ``perBuffer * va``, which is PU * VA / VG.
  Nothing else scales it.

  PU is the innermost of the operand's loop axes: KMN gives A=M and B=N,
  MKN gives A=K and B=N.  Its register count is that many instruction tiles.
  A TDMSplit that cuts this axis has already halved it, so VG is only the
  remaining divisor (a HalfPLR shape on a split macro-tile axis is VG1, not
  VG2).  VA is a stated VgprAlloc, otherwise PLR+1 when CLR is off and the
  outer-loop size when CLR is on.
  """
  orders = ("KMN", "KNM", "MNK", "MKN", "NMK", "NKM")
  order = orders[int(kernel.get("WmmaInnerOrder", 1) or 1) - 1]
  side = "A" if str(tc).endswith("A") else "B"
  roles = ("M", "K") if side == "A" else ("N", "K")
  puAxis = max(roles, key=lambda axis: order.index(axis))

  def dial(name):
    value = kernel.get(name, -1)
    if value is None:
      return -1
    try:
      return int(value)
    except (TypeError, ValueError):
      return -1

  def tiles(axis):
    if axis == "K":
      k = kernel.get("MatrixInstK")
      if not k:
        k = kernel["MatrixInstruction"][2]
      return max(1, int(kernel["DepthU"]) // int(k))
    return int(kernel["MIWaveTileA"] if axis == "M" else kernel["MIWaveTileB"])

  pu = tiles(puAxis)
  splitFactor, splitAxis = split_of(kernel, side)
  cutsPu = ((splitAxis == AXIS_MT and puAxis != "K")
            or (splitAxis == AXIS_DU and puAxis == "K"))
  if cutsPu and splitFactor > 1:
    if pu % splitFactor:
      raise RuntimeError(
          "TDMSplit%s does not divide %s's %d-tile prefetch unit" % (side, tc, pu))
    pu //= splitFactor

  vg = dial("VgprGroup" + side)
  if vg < 1:
    vg = 1
  iptName = "MIInputPerThread" + (tc if str(tc).startswith("MXS") else side)
  ipt = kernel[iptName]
  puRegs = (int(pu * ipt // writer.states.bpr) if bpe is None
            else int(pu * ipt * bpe // writer.states.bpr))
  if puRegs % vg:
    raise RuntimeError(
        "VgprGroup%s=%d does not divide %s's %d-register prefetch unit" % (side, vg, tc, puRegs))
  perBuffer = puRegs // vg

  stated = dial("VgprAlloc" + side)
  if stated > 0:
    va = stated
  else:
    clr = dial("ClusterLocalRead" + side)
    if clr < 0:
      clr = dial("ClusterLocalRead")
    if clr < 0:
      clr = 0
    if clr == 0:
      plr = dial("PrefetchLocalRead" + side)
      if plr < 0:
        plr = dial("PrefetchLocalRead")
      if plr < 0:
        plr = 0
      va = plr + 1
    else:
      # The outer loop is every loop outside the unit, so its size is how
      # many prefetch units the operand's tiles hold.  A TDMSplit that cut
      # the unit left its other half out here, and that half is one of
      # those loops.
      va = max(1, tiles(roles[0]) * tiles(roles[1]) // pu)

  direct = kernel.get("DirectToVgpr" + tc, kernel.get("DirectToVgpr" + side, False))
  packing = ((writer.states.packDTVA or writer.states.convDTVA) if side == "A"
             else (writer.states.packDTVB or writer.states.convDTVB))
  if direct and not packing:
    return 0, va
  return perBuffer, va


def operandNumVgprBuffer(writer, tc):
  """Valu ring depth for one operand.

  vgprAllocationImplClassic stores the per-operand count.  A zero field means
  allocation has not run, so the shared scaffold count still applies.
  """
  stored = {"A": writer.states.numVgprBufferA,
            "B": writer.states.numVgprBufferB,
            "MXSA": writer.states.numVgprBufferMXSA,
            "MXSB": writer.states.numVgprBufferMXSB}[tc]
  return int(stored or writer.states.numVgprBuffer)


def loopModelRegBuffers(writer, kernel):
  """Per-operand ring depth -- that operand's VGPR buffer count, regions included.

  GIR and the `Valu*_X*` table read the same cached θ/S, so every slot GIR names has a matching
  symbol.  Register groups of one operand share the X axis, so that operand's depth is the max
  over its groups.  A, B, and the MX scales do not share a depth; the caller keeps one count
  per operand.
  """
  if not kernel["UseLoopModel"]:
    return {}
  theta, S = loopModelTheta(writer, kernel)
  widths = {}
  for op in theta.operands:
    if op.trajectory.fragment_fill is None:
      continue
    w = max((_geometry.group_ring_depth(theta, op, group, S)
             for group in op.fragment.groups()), default=1)
    widths[op.name] = max(1, int(w))
  return widths


def loopModelValuRegs(writer, kernel):
  """Physical VGPRs per tensorChar for the Valu rings GIR names.

  Each register group owns a compact block for every one of its W rotation slots.
  """
  if not kernel["UseLoopModel"]:
    return {}
  theta, S = loopModelTheta(writer, kernel)
  out = {}
  for op in theta.operands:
    if op.trajectory.fragment_fill is None:
      continue
    side = "A" if op.name in ("A", "MXSA") else "B" if op.name in ("B", "MXSB") else None
    stated = side is not None and any(
        int(kernel.get(name + side, -1) if kernel.get(name + side, -1) is not None else -1) != -1
        for name in ("VgprGroup", "VgprAlloc"))
    sizing = (_geometry.operand_footprint_regs if stated
              else _geometry.operand_emitted_regs)
    out[op.name] = int(sizing(theta, op, S))
  return out


def loopModelRegisterLayout(writer, kernel):
  """Compact per-group physical register offsets from the cached θ/S."""
  if not kernel["UseLoopModel"]:
    return {}
  theta, S = loopModelTheta(writer, kernel)
  return _geometry.register_layout(theta, S)


def applyLoopModelValuRegs(writer, kernel):
  """Overwrite the scaffold's Valu-ring sizing with θ/S's, after it has run."""
  if not kernel["UseLoopModel"]:
    return
  required = writer.states.loopModelValuVgprs
  byOperand = {"A": writer.states.a, "B": writer.states.b}
  if kernel["ProblemType"]["MXBlockA"]:
    byOperand["MXSA"] = writer.states.mxsa
  if kernel["ProblemType"]["MXBlockB"]:
    byOperand["MXSB"] = writer.states.mxsb
  missing = [name for name in byOperand if name not in required]
  if missing:
    raise RuntimeError("UseLoopModel has no Valu VGPR allocation for %s" % ", ".join(missing))
  for name, state in byOperand.items():
    count = int(required[name])
    if count < 1:
      raise RuntimeError("UseLoopModel derived invalid Valu VGPR count %d for %s" % (count, name))
    state.numVgprValu = count

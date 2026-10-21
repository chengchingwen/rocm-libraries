# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
LdsRegionCheck -- does one `ds_load` touch more than ONE storage region of a split tile?
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .asm_lane_eval import LaneEval, UnsupportedInstruction

# double-buffer selector: LDS reads toggle this bit to swap generations, and it is NOT part of the
# region coordinate.  Masked off before the region division, or every swap would read as a region
LDS_BUFFER_BIT = 0x10000


@dataclass
class DsAccess:
    """One emitted `ds_load`, with the per-lane addresses it actually issues."""
    line:      int
    text:      str
    operand:   str                       # A / B / MXSA / MXSB, from the destination register name
    offset:    int
    addrs:     dict = field(default_factory=dict)     # wave -> [addr per lane]

    def regions(self, region_bytes: int, lds_base: int = 0):
        """(per-wave region sets, union across waves)."""
        per = {}
        for wave, a in self.addrs.items():
            per[wave] = {((x & ~LDS_BUFFER_BIT) - lds_base) // region_bytes for x in a}
        allr = set().union(*per.values()) if per else set()
        return per, allr


@dataclass
class RegionViolation:
    access: DsAccess
    kind:   str            # "within-wave" | "across-wave"
    detail: str


# ---------------------------------------------------------------------------------------------
# parsing
_SET = re.compile(r'^\.set\s+([A-Za-z_]\w*)\s*,\s*(.+?)\s*$')
_INS = re.compile(r'^([a-z][a-z0-9_]*)\s+(.*)$')


def parse_symbols(lines):
    """`.set` table, evaluated in file order so later entries may use earlier ones."""
    sym = {}
    for l in lines:
        m = _SET.match(l.strip())
        if not m:
            continue
        try:
            sym[m.group(1)] = int(eval(m.group(2), {"__builtins__": {}}, dict(sym)))
        except Exception:
            pass          # non-numeric .set (labels, strings) -- not an address term
    return sym


def _operand(tok: str, sym: dict):
    """One textual operand -> ('v'|'s'|'lit', value).  A range `v[a:b]` yields its FIRST register,
    which is what an address or a load base is; the width is carried by the opcode.

    An operand this cannot resolve becomes `('unknown', text)` rather than `None`: the evaluator
    then REFUSES the slice naming the token, where a `None` would either crash far from the cause
    or, worse, be read as zero and produce a plausible wrong address."""
    tok = tok.strip()
    if not tok:
        return None
    m = re.match(r'^([vs])\[([^\]]+)\]$', tok)
    if m:
        kind, inner = m.group(1), m.group(2)
        first = inner.split(':', 1)[0]
        try:
            return (kind, int(eval(first, {"__builtins__": {}}, dict(sym))))
        except Exception:
            return ("unknown", tok)
    # BARE register forms (`v4`, `s5`) -- the assembler accepts both spellings and the emitter uses
    # the bare one for unnamed temporaries, which is exactly what the address slice runs through.
    m = re.match(r'^([vs])(\d+)$', tok)
    if m:
        return (m.group(1), int(m.group(2)))
    if re.match(r'^ttmp\d+$', tok):
        return ("s", tok)
    if tok in ("vcc", "vcc_lo", "vcc_hi", "exec", "exec_lo", "null"):
        return ("special", tok)
    try:
        return ("lit", int(tok, 0))
    except ValueError:
        pass
    if tok in sym:
        return ("lit", sym[tok])
    return None


_NO_DST_PREFIXES = ("global_store", "global_prefetch", "buffer_store", "scratch_store",
                    "ds_store", "ds_write", "flat_store", "tensor_load", "tensor_store",
                    "s_store", "image_store")
_NO_DST_EXACT = {"v_nop", "s_nop", "s_endpgm", "s_barrier", "s_barrier_signal", "s_barrier_wait",
                 "s_branch", "s_cbranch_scc0", "s_cbranch_scc1", "s_cbranch_vccz",
                 "s_cbranch_vccnz", "s_setreg_imm32_b32", "s_setprio", "s_sleep",
                 "s_set_vgpr_msb", "s_waitcnt", "s_denorm_mode"}


def _defines_first_operand(op: str) -> bool:
    if op in _NO_DST_EXACT or op.startswith(_NO_DST_PREFIXES):
        return False
    if op.startswith(("s_wait", "s_cmp", "s_branch", "s_cbranch")):
        return False
    return True


def _operand_width(tok: str, sym: dict) -> int:
    """How many consecutive registers the operand `v[a:b]` covers (1 when it is not a range).

    A DEF OF A RANGE DEFINES EVERY REGISTER IN IT, and forgetting that is not a small imprecision.
    """
    m = re.match(r'^[vs]\[([^\]]+)\]$', tok.strip())
    if not m or ':' not in m.group(1):
        return 1
    a, b = m.group(1).split(':', 1)
    try:
        lo = int(eval(a, {"__builtins__": {}}, dict(sym)))
        hi = int(eval(b, {"__builtins__": {}}, dict(sym)))
        return max(1, hi - lo + 1)
    except Exception:
        return 1


def parse_instructions(lines, sym):
    """(line_no, op, dst, srcs, raw, defines, dst_width) -- operands already resolved."""
    out = []
    for i, raw in enumerate(lines):
        l = re.sub(r'//.*', '', raw).strip()
        if not l or l.startswith(('.', '/')) or re.match(r'^[A-Za-z_]\w*:$', l):
            continue
        m = _INS.match(l)
        if not m:
            continue
        op, rest = m.group(1), m.group(2)
        # strip trailing modifiers (offset:, matrix_*, *_fmt) -- they are not operands
        rest = re.sub(r'\s+[a-z_]+:[A-Za-z0-9_]+', '', rest)
        toks = [t for t in rest.split(',')]
        ops = [_operand(t, sym) for t in toks]
        if not ops:
            continue
        defines = _defines_first_operand(op)
        width = _operand_width(toks[0], sym) if defines else 1
        out.append((i + 1, op, ops[0] if defines else None,
                    ops if not defines else ops[1:], l, defines, width))
    return out


def backward_slice(insts, targets):
    """The instructions that define `targets`, transitively.  Numeric AND symbolic registers are
    followed -- the address is built through unnamed temporaries, so a slice that only chases
    `vgprXxx` names stops after a couple of steps and silently returns a short, wrong slice.

    A def is matched against the WHOLE register range it writes (see `_operand_width`), so a wide
    load satisfies a use of any register inside it."""
    live = set(targets)
    keep = []
    for rec in reversed(insts):
        _ln, _op, dst, srcs, _raw, defines, width = rec
        covered = {(dst[0], dst[1] + k) for k in range(width)} & live if (defines and dst and
                                                                         dst[0] in ("v", "s")) else set()
        if covered:
            keep.append(rec)
            live -= covered
            for s in srcs:
                if s and s[0] in ("v", "s"):
                    live.add(s)
    keep.reverse()
    return keep


# ---------------------------------------------------------------------------------------------
# the check
_DS = re.compile(r'^ds_(?:load|read)\w*\s')
_OFF = re.compile(r'offset:(\d+)')


def collect_ds_loads(lines, sym):
    """Every `ds_load`, with its address VGPR, immediate offset, and which operand it fills."""
    out = []
    for i, raw in enumerate(lines):
        l = re.sub(r'//.*', '', raw).strip()
        if not _DS.match(l + " "):
            continue
        mo = _OFF.search(l)
        off = int(mo.group(1)) if mo else 0
        parts = l.split(',')
        addr = _operand(re.sub(r'\s+offset:\d+', '', parts[1]), sym) if len(parts) > 1 else None
        mname = re.search(r'vgprValu([A-Za-z]+)_X', l)
        out.append((i + 1, l, mname.group(1) if mname else "?", off, addr))
    return out


def _expected_offsets(path, sl, ds, accesses, nwaves, wavefront, sgpr_seed, kernarg):
    """Replay the address slice per wave and record the LDS offset each ds_read should use."""
    for wave in range(nwaves):
        m = LaneEval(nlanes=wavefront, serial_vgpr=0, wave=wave,   # v0 = workitem id
                     kernarg=kernarg)
        for k, v in (sgpr_seed or {}).items():
            m.s[k] = v & 0xFFFFFFFF
        for ln, op, dst, srcs, raw, _def, _w in sl:
            try:
                m.step(op, dst, srcs)
            except UnsupportedInstruction as e:
                raise RuntimeError(
                    f"{path}:{ln}: address slice uses an instruction LaneEval does not model "
                    f"({e}) -- refusing rather than evaluating it as a no-op:\n    {raw}")
        for acc, (_ln, _t, _o, off, a) in zip(accesses, ds):
            if a is None:
                continue
            acc.addrs[wave] = [x + off for x in m.lanes_of(a[1])]


def check(path, region_bytes: int, nwaves: int = 1, wavefront: int = 32,
          lds_base: int = 0, operands=None, sgpr_seed=None, kernarg=None):
    """Run the check over one emitted kernel.

    `region_bytes` is `tdmSplitLdsBoundary` for the operand -- the SAME displacement the copy side
    advances by.  Returns (accesses, violations, notes).
    """
    lines = [l.rstrip("\n") for l in open(path)]
    sym = parse_symbols(lines)
    insts = parse_instructions(lines, sym)
    ds = collect_ds_loads(lines, sym)
    if operands:
        ds = [d for d in ds if d[2] in operands]
    if not ds:
        return [], [], ["no ds_load found"]

    addr_regs = {d[4] for d in ds if d[4]}
    sl = backward_slice(insts, addr_regs)
    notes = [f"{len(insts)} instructions, address slice {len(sl)}, {len(ds)} ds_loads"]

    accesses = [DsAccess(line=ln, text=txt, operand=opn, offset=off) for ln, txt, opn, off, _a in ds]
    _expected_offsets(path, sl, ds, accesses, nwaves, wavefront, sgpr_seed, kernarg)
    viol = []
    for acc in accesses:
        if not acc.addrs:
            continue
        per, allr = acc.regions(region_bytes, lds_base)
        for wave, rs in per.items():
            if len(rs) > 1:
                viol.append(RegionViolation(acc, "within-wave",
                                      f"wave {wave} lanes span regions {sorted(rs)}"))
        if len(allr) > 1 and all(len(r) == 1 for r in per.values()):
            viol.append(RegionViolation(acc, "across-wave",
                                  "waves pick different regions: "
                                  + ", ".join(f"w{w}->{sorted(r)[0]}" for w, r in per.items())))
    return accesses, viol, notes

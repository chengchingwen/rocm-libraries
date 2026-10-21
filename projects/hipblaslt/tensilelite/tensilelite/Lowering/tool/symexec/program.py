# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
Decode emitted gfx1250 assembly into the instruction stream the machine executes.

Everything the machine needs comes from the instruction itself -- opcode, registers, immediate
offset, access width -- except which tensor an LDS access belongs to, which is what the `<GIR: ...>`
tag carries.  An instruction the decoder does not model is still decoded far enough to see WHICH
REGISTER IT WRITES, because the machine's state lives in a few named registers and an unmodelled
write to one of them has to be refused rather than skipped.
"""

from __future__ import annotations

import re

from ...gir_tag import GIR_TAG_RE

#: `s_wait_dscnt 3`.  gfx1250 splits the counters; there is no lgkmcnt.
WAIT_RE = re.compile(r"^\s*s_wait_(tensorcnt|dscnt|loadcnt|storecnt|asynccnt|kmcnt|xcnt)\s+(\d+)")

#: `s_wait_alu depctr_vm_vsrc(0)` -- the ALU dependency counters.  Same FIFO discipline as the
#: memory counters: a VALU enqueues, the wait pops down to N.
DEPCTR_RE = re.compile(r"^\s*s_wait_alu\s+depctr_(\w+)\((\d+)\)")
#: a label DEFINITION, not any line mentioning the name.
LABEL_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*:")
#: `v[vgprValuA_X0_I0+0:vgprValuA_X0_I0+0+3]`, `v[vgprValuMXSA_X0_I0+0]`
VREG_RE = re.compile(r"v\[\s*(vgpr\w+)((?:[+-]\d+)*)\s*(?::\s*vgpr\w+((?:[+-]\d+)*))?\s*\]")
#: `offset:8704`
OFFSET_RE = re.compile(r"\boffset:(\d+)")
#: `read A[tile=0,k=1]->X1` -- names the tensor an LDS read belongs to.
READ_RE = re.compile(r"read (\w+)\[([^\]]*)\]->X(\d+)")
#: `copy A+B gen=0` -- names the tensors one TDM load fills.
COPY_RE = re.compile(r"copy ([\w+]+) gen=(\d+)")
#: the two register families that carry a buffer-select bit.  `sgprtdmAGroup0` appears both alone
#: and as the base of a RANGE (`s[sgprtdmAGroup0:sgprtdmAGroup0+3]`), which is how a TDM load always
#: spells its descriptor, so the bracket contents are not anchored.
READ_ADDR_RE = re.compile(r"v\[\s*vgprLocalReadAddr(\w+?)((?:[+-]\d+)*)\s*[\]:]")
#: the TDM descriptor's group NAME, for saying which group a load belongs to.
DESC_RE = re.compile(r"s\[\s*sgprtdm(\w+?)Group(\d+)")
#: the descriptor operand with its sub-index RANGE.  The buffer select is `Group0+1` alone; the
#: 64-bit global pointer the `gr_inc` advances is `Group0+2:+3`, a different register.
DESC_RANGE_RE = re.compile(r"s\[\s*sgprtdm(\w+?)Group(\d+)((?:[+-]\d+)*)"
                           r"\s*(?::\s*sgprtdm\w+?Group\d+((?:[+-]\d+)*))?\s*\]")
#: which sub-index of `Group0` carries the LDS buffer select bit.
SELECT_INDEX = 1
#: `swap A+B copy-hop` / `swap MXSA read-hop` -- a fused group shares one LDS state, so the xor is
#: written on ONE member's descriptor and moves the buffer for EVERY member.  The instruction cannot
#: say that; the tag can.
SWAP_RE = re.compile(r"swap ([\w+]+) (copy|read)-hop")
#: a control transfer.  The machine executes straight-line code, so one inside the decoded range
#: means the trace would include a path that does not run.
BRANCH_RE = re.compile(r"^\s*(s_branch|s_cbranch\w*)\b")

#: `sync LDS3` / `sync LDS [0, 4]` -- the GIR slot ids the instruction names.  This is the
#: frame-resolved storage identity, so it decides aliasing exactly where a pointer parity guesses.
SYNC_RE = re.compile(r"sync LDS\s*(?:\[([^\]]*)\]|(\d+))")

#: which completion counter each memory instruction lands on.  A wmma lands on none.
COUNTER = {"tensor_load": "tensorcnt", "ds_load": "dscnt", "ds_store": "dscnt",
           "global_load": "loadcnt", "global_store": "storecnt"}

#: bytes moved per lane, from the opcode suffix.  Needed to give an access a byte RANGE.
WIDTH = {"b8": 1, "u8": 1, "i8": 1, "b16": 2, "u16": 2, "i16": 2,
         "b32": 4, "b64": 8, "b96": 12, "b128": 16,
         "2addr_b32": 8, "2addr_b64": 16, "2st64_b32": 8}


class DecodeError(RuntimeError):
    """An instruction the decoder cannot model.

    Raised, never skipped: a skipped memory instruction silently removes a dependence, which turns
    a real hazard into a clean report.  `scaffold_line` marks the one refusal a caller may answer
    by ENDING the region instead: a TDM load the scaffold emitted, which carries no GIR copy tag
    because the tail loop is scaffold-owned."""

    def __init__(self, message, scaffold_line=None):
        super().__init__(message)
        self.scaffold_line = scaffold_line


def _offsets(expr):
    return sum(int(t) for t in re.findall(r"[+-]\d+", expr or ""))


def _regs(text):
    """The `(name, index)` register cells an operand list names, expanding `v[x+a:x+a+n]`."""
    out = set()
    for name, lo, hi in VREG_RE.findall(text):
        a, b = _offsets(lo), _offsets(hi) if hi else _offsets(lo)
        out.update((name, i) for i in range(min(a, b), max(a, b) + 1))
    return frozenset(out)


def _width(op):
    """Bytes per lane this opcode moves, from its suffix."""
    for suffix, n in sorted(WIDTH.items(), key=lambda kv: -len(kv[0])):
        if op.endswith("_" + suffix):
            return n
    return None


def classify(op):
    if op.startswith("tensor_load"):
        return "tensor_load"
    if op.startswith("ds_load") or op.startswith("ds_read"):
        return "ds_load"
    if op.startswith("ds_store") or op.startswith("ds_write"):
        return "ds_store"
    if op.startswith("buffer_load") or op.startswith("global_load"):
        return "global_load"
    if op.startswith("buffer_store") or op.startswith("global_store"):
        return "global_store"
    if op.startswith("v_wmma") or op.startswith("v_swmmac"):
        return "wmma"
    if op == "s_barrier_wait":
        return "barrier"
    if op.startswith("s_barrier"):
        return "barrier_signal"
    if op in ("v_xor_b32", "s_xor_b32"):
        return "xor"
    if op == "s_set_vgpr_msb":
        return "vgpr_msb"
    if op.startswith("v_"):
        # Every VALU is a CONSUMER: the drain reads accumulators back with v_mov and v_cvt, and a
        # dscnt that does not cover those is the same defect one instruction later.
        return "valu"
    return "other"


#: opcodes whose first operand is a SOURCE, so naming a select register there is a read.  A TDM
#: load's first operand is its descriptor; a compare's is the value compared.
_READS_FIRST_OPERAND = ("tensor_load", "ds_store", "ds_write", "global_store", "buffer_store",
                        "s_cmp", "v_cmp", "s_bitcmp", "s_barrier", "s_nop", "s_sleep", "s_endpgm",
                        "s_wait", "s_setprio", "s_branch", "s_cbranch")


def select_target(text):
    """The buffer-select register this text names, as the machine's parity key, or None.

    On the descriptor side only `Group0+1` carries the select; `Group0+2:+3` is the 64-bit global
    pointer the `gr_inc` advances, and treating a write to it as a buffer move is how a pointer
    increment gets mistaken for a swap."""
    m = READ_ADDR_RE.search(text)
    if m:
        # THE SUB-INDEX IS PART OF THE NAME: `LocalReadAddrB+1` is a SIBLING pointer holding the
        # other buffer, not another spelling of `LocalReadAddrB`.
        return "read:%s%+d" % (m.group(1), _offsets(m.group(2)))
    m = DESC_RANGE_RE.search(text)
    if m and int(m.group(2)) == 0:
        lo = _offsets(m.group(3))
        hi = _offsets(m.group(4)) if m.group(4) is not None else lo
        if min(lo, hi) <= SELECT_INDEX <= max(lo, hi):
            return "desc:%s%+d" % (m.group(1), SELECT_INDEX)
    return None


def writes_first_operand(op):
    return not any(op.startswith(p) for p in _READS_FIRST_OPERAND)


class Inst:
    """One decoded instruction.

    `tensors` is the set of tensor names this instruction's LDS access belongs to; `addr_tensor`
    is the tensor whose read-address register supplies the buffer parity; `writes_select` is the
    parity key this instruction's DESTINATION operand names, which is what makes an unmodelled
    write to the machine's state visible instead of silent.
    """

    __slots__ = ("line", "op", "kind", "defs", "uses", "gir", "text",
                 "tensors", "addr_tensor", "desc_group", "offset", "width",
                 "xor_target", "xor_mask", "xor_targets", "writes_select", "addr_select",
                 "src_select", "literal", "lds_tokens")

    def __init__(self, **kw):
        for slot in self.__slots__:
            setattr(self, slot, kw.get(slot))

    def __deepcopy__(self, memo):
        """An instruction is immutable, so a forked path shares it rather than copying it."""
        return self

    @property
    def counter(self):
        return COUNTER.get(self.kind)

    def __repr__(self):
        return f"{self.op}@{self.line}" + (f" <{self.gir}>" if self.gir else "")


def _tensors_of(gir, kind, text, line):
    """Which tensors an LDS instruction touches, and which register carries its buffer parity."""
    tensors, addr_tensor = frozenset(), None
    if kind in ("ds_load", "ds_store"):
        m = READ_RE.search(gir or "")
        if m:
            tensors = frozenset([m.group(1)])
        a = READ_ADDR_RE.search(text)
        addr_tensor = a.group(1) if a else None
        if not tensors and addr_tensor:
            tensors = frozenset([addr_tensor])
    elif kind == "tensor_load":
        m = COPY_RE.search(gir or "")
        if not m:
            raise DecodeError("line %d: TDM load with no GIR copy tag, so which tensors it fills "
                              "is unknown: %r" % (line, text.strip()), scaffold_line=line)
        tensors = frozenset(m.group(1).split("+"))
    return tensors, addr_tensor


def _first_operand(args):
    """`(destination, the rest)`.  Splitting on `"],"` swallows later operands whenever the first
    has no bracket, which made a pure READ of a select register look like a write to it."""
    depth = 0
    for i, ch in enumerate(args):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        elif ch == "," and depth == 0:
            return args[:i], args[i + 1:]
    return args, ""


def scalar_dsts(inst):
    """The bare `sNN` names one instruction writes."""
    if not writes_first_operand(inst.op):
        return ()
    dst = _first_operand(inst.text.partition(" ")[2])[0]
    # `s_add_u64 s[88:89], ...` writes BOTH halves; matching only a bare `sNN` left the 64-bit
    # form -- which is how this codebase does pointer arithmetic -- outside the guard.
    rng = re.match(r"\s*s\[\s*(\d+)\s*:\s*(\d+)\s*\]", dst)
    if rng:
        return tuple("s%d" % i for i in range(int(rng.group(1)), int(rng.group(2)) + 1))
    one = re.match(r"\s*s(\d+)\b", dst)
    return ("s" + one.group(1),) if one else ()


def scalar_srcs(inst):
    """The operand text an instruction READS, with its destination removed.

    Symbolic names (`s[sgprArgType]`) are kept verbatim: they are as much a source identity as a
    numbered register, and dropping them would make two different reads look like one value."""
    args = inst.text.partition(" ")[2]
    return _first_operand(args)[1] if writes_first_operand(inst.op) else args


def _swap_targets(gir, fallback):
    """Every select key one swap moves, from its GIR tag; `(fallback,)` when it has none."""
    m = SWAP_RE.search(gir or "")
    if not m:
        return (fallback,) if fallback else ()
    prefix, index = ("desc:", SELECT_INDEX) if m.group(2) == "copy" else ("read:", 0)
    return tuple("%s%s%+d" % (prefix, name, index) for name in m.group(1).split("+"))


def _literal(text):
    """The immediate operand, hex or decimal.  `offset:` is an addressing field, not an operand."""
    body = OFFSET_RE.sub("", text)
    m = re.search(r"\b0x([0-9a-fA-F]+)\b", body)
    if m:
        return int(m.group(1), 16)
    m = re.search(r"(?:^|,)\s*(\d+)\s*(?=,|$)", body.split("//")[0].partition(" ")[2])
    return int(m.group(1)) if m else None


def loop_body(lines, head="label_LoopBeginL"):
    """`(first, last)` line indices of the rolled body: the label definition to its back edge."""
    start = None
    for i, line in enumerate(lines):
        m = LABEL_RE.match(line)
        if m and m.group(1) == head:
            start = i
            break
    if start is None:
        return None
    back = re.compile(r"\s*s_cbranch\w*\s+" + re.escape(head) + r"\b")
    end = next((i for i in range(start + 1, len(lines)) if back.match(lines[i])), None)
    return None if end is None else (start + 1, end)


def setup_new_tile(lines):
    """The line where a tile's own setup begins, or None.

    Everything above it loads kernel arguments and picks an ArgType; it branches a great deal and
    touches no LDS, so walking it multiplies paths without adding state."""
    for i, line in enumerate(lines):
        if "Begin setupNewTile" in line:
            return i
    return None


#: What moves the machine's state.  `barrier_signal` does not (`machine.py` steps over it), so a
#: span holding only one is inert however it is reached.
_MODELLED = ("tensor_load", "ds_load", "ds_store", "global_load", "global_store",
             "wmma", "barrier")


def _inert_skips(lines, lo, hi):
    """Branch lines whose skipped span the machine does not model, and so may be dropped.

    Only a FORWARD branch to a label inside the range qualifies, and only when nothing between
    carries a wait or an access -- otherwise the untaken path is a real trace difference."""
    labels = {}
    for i in range(lo, hi):
        m = LABEL_RE.match(lines[i])
        if m:
            labels[m.group(1)] = i
    out = set()
    for i in range(lo, hi):
        code = lines[i].split("//")[0]
        if not BRANCH_RE.match(code):
            continue
        target = labels.get(code.split()[-1].strip())
        if target is None or target <= i:
            continue
        span = range(i + 1, target)
        if any(WAIT_RE.match(lines[j].split("//")[0])
               or classify((lines[j].split("//")[0].split() or [""])[0]) in _MODELLED
               for j in span):
            continue
        out.update(span)
        out.add(i)
    return out


def decode_range(lines, lo, hi, allow_branches=False):
    """`[Inst | ('wait', counter, n, line)]` for `lines[lo:hi]`, in issue order.

    The machine runs straight-line code, so a control transfer inside the range means the trace
    would contain instructions that do not all execute -- including `s_wait_*`, which would retire
    loads that are really still outstanding.  That is a FALSE NEGATIVE, so it is refused.

    A FORWARD skip over instructions the machine does not model is the exception: both paths leave
    its state identical, so taking it or not cannot change any answer.  It is dropped, not refused.
    """
    inert = _inert_skips(lines, lo, hi)
    out = []
    for i in range(lo, hi):
        if i in inert:
            continue
        raw = lines[i]
        code = raw.split("//")[0]
        m = WAIT_RE.match(code)
        if m:
            out.append(("wait", m.group(1), int(m.group(2)), i + 1))
            continue
        d = DEPCTR_RE.match(code)
        if d:
            out.append(("wait", d.group(1), int(d.group(2)), i + 1))
            continue
        if BRANCH_RE.match(code) and not allow_branches and i not in inert:
            raise DecodeError(f"line {i + 1}: control transfer inside the executed range "
                              f"({code.strip()!r}); straight-line execution would run a path that "
                              f"does not, so no answer is available for this kernel")
        toks = code.split()
        if not toks:
            continue
        op = toks[0]
        kind = classify(op)
        _op, _, args = code.strip().partition(" ")
        dst, rest = _first_operand(args)
        g = GIR_TAG_RE.search(raw)
        gir = g.group(1) if g else None
        sync = SYNC_RE.search(raw)
        lds_tokens = frozenset(
            int(t) for t in (sync.group(1) or sync.group(2)).replace(",", " ").split()
        ) if sync else frozenset()
        tensors, addr_tensor = _tensors_of(gir, kind, code, i + 1)
        if kind in ("ds_load", "ds_store") and _width(op) is None:
            raise DecodeError(f"line {i + 1}: no access width for {op!r}; add its suffix to WIDTH "
                              f"rather than letting the access carry no byte range")
        desc_group = None
        if kind == "tensor_load":
            d = DESC_RE.search(code)
            if d is None:
                raise DecodeError(f"line {i + 1}: TDM load naming no descriptor register, so which "
                                  f"buffer it fills is unknown: {code.strip()!r}")
            desc_group = d.group(1)
        out.append(Inst(line=i + 1, op=op, kind=kind,
                        defs=_regs(dst) if kind in ("ds_load", "global_load") else frozenset(),
                        uses=_regs(rest) if kind in ("wmma", "valu") else frozenset(),
                        gir=gir, text=code.strip(), tensors=tensors, addr_tensor=addr_tensor,
                        desc_group=desc_group,
                        offset=int(OFFSET_RE.search(code).group(1)) if OFFSET_RE.search(code) else 0,
                        width=_width(op),
                        xor_target=select_target(dst) if kind == "xor" else None,
                        xor_targets=_swap_targets(gir, select_target(dst)) if kind == "xor" else (),
                        xor_mask=_literal(code) if kind == "xor" else None,
                        writes_select=select_target(dst) if writes_first_operand(op) else None,
                        addr_select=select_target(code),
                        lds_tokens=lds_tokens,
                        src_select=select_target(rest),
                        literal=_literal(code)))
    return out


def decode(path, head="label_LoopBeginL", allow_branches=False):
    """The rolled loop body of `path`."""
    lines = open(path, errors="replace").read().splitlines()
    span = loop_body(lines, head)
    if span is None:
        raise DecodeError(f"{path}: no rolled body for {head!r}")
    return decode_range(lines, *span, allow_branches=allow_branches)


def decode_prologue(path, head="label_LoopBeginL", allow_branches=True):
    """Everything before the loop head -- where the buffer parities and the ring are established.

    Branches are permitted here by default: the prologue's arms set up state the body then uses, and
    refusing every kernel with a prologue branch would leave nothing to check.  The cost is that the
    ENTRY parity is whatever this straight-line pass computes, which is why the machine reports the
    parity it started the body with."""
    lines = open(path, errors="replace").read().splitlines()
    span = loop_body(lines, head)
    if span is None:
        raise DecodeError(f"{path}: no rolled body for {head!r}")
    return decode_range(lines, 0, span[0] - 1, allow_branches=allow_branches)

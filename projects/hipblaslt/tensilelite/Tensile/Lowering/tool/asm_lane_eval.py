# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
LaneEval -- execute a straight-line slice of emitted gfx1250 assembly, once per LANE.
"""

from __future__ import annotations

import struct

MASK32 = 0xFFFFFFFF


class UnsupportedInstruction(RuntimeError):
    """An opcode in the slice that `LaneEval` does not model.

    Raised rather than skipped: skipping leaves a register stale and yields a WRONG address that
    still looks like an address.  The message names the instruction so the table can be grown
    deliberately."""


class LaneEval:
    """Per-lane machine state for one wavefront.

    VGPRs are `{index: [value per lane]}`, SGPRs are `{index_or_name: value}` -- scalars really are
    uniform, which is why the split matters: an address term coming from an SGPR cannot vary across
    lanes and so can never, by itself, make one instruction straddle a region boundary."""

    def __init__(self, nlanes: int = 32, serial_vgpr: int | None = None, wave: int = 0,
                 kernarg=None):
        self.n = nlanes
        self.v: dict[int, list[int]] = {}
        self.s: dict[object, int] = {}
        # THE KERNARG IMAGE, `{byte offset: dword}` -- the values `s_load_b*` brings in.
        #
        self.kernarg: dict[int, int] = dict(kernarg or {})
        self.vcc = [0] * nlanes
        self.scc = None
        # reciprocal implementation, swappable so `check` can perturb it and confirm
        # the hardware approximation cannot change the answer
        self.rcp = lambda one, x: one / x if x else float('inf')
        self.wave = wave
        if serial_vgpr is not None:
            # `vgprSerial` holds the workgroup-flat thread id, so lane L of wave W is W*n + L.
            self.v[serial_vgpr] = [wave * nlanes + i for i in range(nlanes)]

    _UN = {
        "s_ff1_i32_b32":   lambda a: (a & -a).bit_length() - 1 if a else 0xFFFFFFFF,
        "s_flbit_i32_b32": lambda a: 32 - a.bit_length() if a else 32,
        "s_brev_b32":      lambda a: int(format(a, "032b")[::-1], 2),
        "s_not_b32":       lambda a: ~a,
        "v_not_b32":       lambda a: ~a,
    }
    _SCC_CMP = {
        "s_cmp_eq_u32": lambda a, b: a == b, "s_cmp_eq_i32": lambda a, b: a == b,
        "s_cmp_lg_u32": lambda a, b: a != b, "s_cmp_lg_i32": lambda a, b: a != b,
        "s_cmp_lt_u32": lambda a, b: a < b,  "s_cmp_le_u32": lambda a, b: a <= b,
        "s_cmp_gt_u32": lambda a, b: a > b,  "s_cmp_ge_u32": lambda a, b: a >= b,
    }
    _BIN2 = {
        "s_min_u32":    lambda a, b: min(a, b),
        "s_max_u32":    lambda a, b: max(a, b),
        "s_mul_hi_u32": lambda a, b: (a * b) >> 32,
        "s_ashr_i32":   lambda a, b: a >> (b & 31),
        "s_andn2_b32":  lambda a, b: a & ~b,
        # bit-field mask: `width` ones, shifted left by `offset` -- the WGM/XCC remap builds its
        # modulo masks with this, so it sits squarely in the address slice.
        "s_bfm_b32":    lambda a, b: ((1 << (a & 31)) - 1) << (b & 31),
    }
    _BFE = ("s_bfe_u32", "v_bfe_u32")

    # ---- operand access -------------------------------------------------------------------
    def read(self, operand, lane: int) -> int:
        """One operand's value in `lane`.  An SGPR or literal is lane-invariant by construction.

        An operand the parser could not resolve is REFUSED here, not defaulted to zero: a zero
        address term is indistinguishable from a real one downstream, so it would turn a parse gap
        into a confident wrong answer.
        """
        if operand is None:
            raise UnsupportedInstruction("unparsed operand")
        kind, idx = operand
        if kind == "unknown":
            raise UnsupportedInstruction(f"unresolved operand {idx!r}")
        if kind == "v":
            if idx not in self.v:
                raise UnsupportedInstruction(f"read of unwritten VGPR v{idx} (a kernel input the "
                                             f"caller must seed, or a gap in the slice)")
            return self.v[idx][lane] & MASK32
        if kind == "s":
            if idx not in self.s:
                raise UnsupportedInstruction(f"read of unwritten SGPR s{idx} (a kernel input the "
                                             f"caller must seed, or a gap in the slice)")
            return self.s[idx] & MASK32
        if kind == "lit":
            return idx & MASK32
        raise UnsupportedInstruction(f"operand kind {kind!r}")

    def write(self, operand, values) -> None:
        kind, idx = operand
        if kind == "v":
            self.v[idx] = [x & MASK32 for x in values]
        elif kind == "s":
            self.s[idx] = values[0] & MASK32
        else:
            raise UnsupportedInstruction(f"cannot write operand kind {kind!r}")

    _BIN = {
        "v_add_nc_u32":   lambda a, b: a + b,
        "v_add_u32":      lambda a, b: a + b,
        "v_sub_nc_u32":   lambda a, b: a - b,
        "v_sub_u32":      lambda a, b: a - b,
        "v_and_b32":      lambda a, b: a & b,
        "v_or_b32":       lambda a, b: a | b,
        "v_xor_b32":      lambda a, b: a ^ b,
        "v_lshlrev_b32":  lambda a, b: b << (a & 31),      # REV: shift amount is src0
        "v_lshrrev_b32":  lambda a, b: b >> (a & 31),
        "v_mul_lo_u32":   lambda a, b: a * b,
        "v_mul_u32_u24":  lambda a, b: (a & 0xFFFFFF) * (b & 0xFFFFFF),
        "s_add_u32":      lambda a, b: a + b,
        "s_sub_u32":      lambda a, b: a - b,
        "s_and_b32":      lambda a, b: a & b,
        "s_or_b32":       lambda a, b: a | b,
        "s_xor_b32":      lambda a, b: a ^ b,
        "s_mul_i32":      lambda a, b: a * b,
        "s_lshl_b32":     lambda a, b: a << (b & 31),
        "s_lshr_b32":     lambda a, b: a >> (b & 31),
    }
    _CMP = {
        "v_cmp_eq_u32": lambda a, b: a == b,
        "v_cmp_ne_u32": lambda a, b: a != b,
        "v_cmp_lt_u32": lambda a, b: a < b,
        "v_cmp_le_u32": lambda a, b: a <= b,
        "v_cmp_gt_u32": lambda a, b: a > b,
        "v_cmp_ge_u32": lambda a, b: a >= b,
    }

    @staticmethod
    def _f32(bits):  return struct.unpack("<f", struct.pack("<I", bits & MASK32))[0]
    @staticmethod
    def _b32(x):     return struct.unpack("<I", struct.pack("<f", x))[0]
    @staticmethod
    def _f64(lo, hi):return struct.unpack("<d", struct.pack("<II", lo & MASK32, hi & MASK32))[0]
    @staticmethod
    def _b64(x):     return struct.unpack("<II", struct.pack("<d", x))

    def _stepf(self, op, dst, srcs) -> bool:
        F32 = {"v_cvt_f32_u32", "v_rcp_iflag_f32", "v_rcp_f32", "v_mul_f32", "v_cvt_u32_f32"}
        F64 = {"v_cvt_f64_u32", "v_rcp_f64", "v_mul_f64", "v_cvt_u32_f64", "v_add_f64",
               "v_fma_f64", "v_ldexp_f64"}
        if op not in F32 and op not in F64:
            return False
        k, di = dst
        if op in F32:
            out = []
            for l in range(self.n):
                a = self.read(srcs[0], l)
                if   op == "v_cvt_f32_u32":  out.append(self._b32(float(a)))
                elif op in ("v_rcp_iflag_f32", "v_rcp_f32"):
                    fa = self._f32(a); out.append(self._b32(self.rcp(1.0, fa)))
                elif op == "v_mul_f32":
                    out.append(self._b32(self._f32(a) * self._f32(self.read(srcs[1], l))))
                else:  # v_cvt_u32_f32
                    out.append(int(self._f32(a)) & MASK32)
            self.write(dst, out); return True
        def rd64(o, l):
            kk, ii = o
            return self._f64(self.read((kk, ii), l), self.read((kk, ii + 1), l))
        lo_out, hi_out = [], []
        for l in range(self.n):
            if op == "v_cvt_f64_u32":      r = float(self.read(srcs[0], l))
            elif op == "v_rcp_f64":        r = self.rcp(1.0, rd64(srcs[0], l))
            elif op == "v_mul_f64":        r = rd64(srcs[0], l) * rd64(srcs[1], l)
            elif op == "v_add_f64":        r = rd64(srcs[0], l) + rd64(srcs[1], l)
            elif op == "v_fma_f64":        r = rd64(srcs[0], l) * rd64(srcs[1], l) + rd64(srcs[2], l)
            elif op == "v_ldexp_f64":      r = rd64(srcs[0], l) * (2.0 ** self.read(srcs[1], l))
            else:                                            # v_cvt_u32_f64
                self.write(dst, [int(rd64(srcs[0], ll)) & MASK32 for ll in range(self.n)])
                return True
            b = self._b64(r); lo_out.append(b[0]); hi_out.append(b[1])
        self.write((k, di), lo_out); self.write((k, di + 1), hi_out)
        return True

    def _step64(self, op: str, dst, srcs) -> bool:
        """64-bit pair ops.  Handled separately because an operand is parsed as its FIRST register,
        so the high half is implicit at `idx + 1`; collapsing `s_mov_b64` to a 32-bit move would
        leave the high register holding a stale value that a later `s_add_u64` then folds into an
        address.  Returns True if `op` was one of these."""
        if op not in ("s_mov_b64", "v_mov_b64", "s_add_u64"):
            return False
        kind, di = dst
        if op in ("s_mov_b64", "v_mov_b64"):
            sk, si = srcs[0]
            for k in (0, 1):
                self.write((kind, di + k), [self.read((sk, si + k), l) for l in range(self.n)])
            return True
        # s_add_u64: full 64-bit add, then split back into the register pair
        (ak, ai), (bk, bi) = srcs[0], srcs[1]
        lo = self.read((ak, ai), 0) | (self.read((ak, ai + 1), 0) << 32)
        rhs = self.read((bk, bi), 0) | (self.read((bk, bi + 1), 0) << 32)
        tot = (lo + rhs) & 0xFFFFFFFFFFFFFFFF
        self.write((kind, di), [tot & MASK32])
        self.write((kind, di + 1), [(tot >> 32) & MASK32])
        return True

    def step(self, op: str, dst, srcs) -> None:
        """Execute one instruction across all lanes.

        Each family reports whether it recognised the opcode; an opcode no family
        claims is refused rather than skipped, because skipping leaves a register
        stale and yields a wrong address that still looks like an address.
        """
        for family in (self._step_wide, self._step_tables, self._step_bitfield,
                       self._step_scalar_load, self._step_address, self._step_lane):
            if family(op, dst, srcs):
                return
        raise UnsupportedInstruction(op)

    def _step_wide(self, op, dst, srcs) -> bool:
        """64-bit and float forms, which have their own tables."""
        if dst is not None and self._step64(op, dst, srcs):
            return True
        if dst is not None and self._stepf(op, dst, srcs):
            return True
        return False

    def _step_tables(self, op, dst, srcs) -> bool:
        """The plain table-driven forms: one unary, binary or compare function per opcode."""
        if op in self._UN:
            f = self._UN[op]
            self.write(dst, [f(self.read(srcs[0], l)) for l in range(self.n)])
            return True
        if op in self._BIN2:
            f = self._BIN2[op]
            self.write(dst, [f(self.read(srcs[0], l), self.read(srcs[1], l))
                             for l in range(self.n)])
            return True
        if op in self._SCC_CMP:
            # SCC is scalar; `dst` is the first SOURCE for a compare, so read both from the raw list
            a, b = (dst, srcs[0]) if len(srcs) == 1 else (srcs[0], srcs[1])
            self.scc = 1 if self._SCC_CMP[op](self.read(a, 0), self.read(b, 0)) else 0
            return True
        if op in ("v_mov_b32", "s_mov_b32"):
            self.write(dst, [self.read(srcs[0], l) for l in range(self.n)])
            return True
        if op in self._BIN:
            f = self._BIN[op]
            self.write(dst, [f(self.read(srcs[0], l), self.read(srcs[1], l))
                             for l in range(self.n)])
            return True
        if op in self._CMP:
            f = self._CMP[op]
            self.vcc = [1 if f(self.read(srcs[-2], l), self.read(srcs[-1], l)) else 0
                        for l in range(self.n)]
            return True
        return False

    def _step_bitfield(self, op, dst, srcs) -> bool:
        """Bit-field extract, bit test and scalar select."""
        if op in self._BFE:
            # src1 packs offset in [4:0] and width in [22:16]
            def _x(a, c):
                return (a >> (c & 31)) & ((1 << ((c >> 16) & 0x7F)) - 1)
            if op == "s_bfe_u32":
                self.write(dst, [_x(self.read(srcs[0], l), self.read(srcs[1], l))
                                 for l in range(self.n)])
            else:
                self.write(dst, [(self.read(srcs[0], l) >> (self.read(srcs[1], l) & 31))
                                 & ((1 << (self.read(srcs[2], l) & 31)) - 1)
                                 for l in range(self.n)])
            return True
        if op in ("s_bitcmp0_b32", "s_bitcmp1_b32"):
            a, b = (dst, srcs[0]) if len(srcs) == 1 else (srcs[0], srcs[1])
            bit = (self.read(a, 0) >> (self.read(b, 0) & 31)) & 1
            self.scc = bit if op.endswith("1_b32") else (1 - bit)
            return True
        if op == "s_cselect_b32":
            if self.scc is None:
                raise UnsupportedInstruction("s_cselect_b32 with no preceding SCC producer")
            pick = srcs[0] if self.scc else srcs[1]
            self.write(dst, [self.read(pick, 0)])
            return True
        return False

    def _step_scalar_load(self, op, dst, srcs) -> bool:
        """`s_load_bN` from a modelled constant buffer."""
        if op.startswith("s_load_b") and op[len("s_load_b"):].isdigit():
            ndw = int(op[len("s_load_b"):]) // 32
            base_off = 0
            if len(srcs) > 2 and srcs[2] is not None and srcs[2][0] == "lit":
                base_off = int(srcs[2][1])
            if dst is None or dst[0] != "s":
                raise UnsupportedInstruction(f"{op}: unparsed destination")
            for i in range(ndw):
                off = base_off + 4 * i
                if off not in self.kernarg:
                    raise UnsupportedInstruction(
                        f"{op} reads kernarg byte offset {off}, which the caller did not supply "
                        f"(pass `kernarg={{{off}: <value>}}`); refusing rather than reading zero")
                self.s[dst[1] + i] = self.kernarg[off] & MASK32
            return True
        return False

    def _step_address(self, op, dst, srcs) -> bool:
        """The fused shift/add forms an address term is built from."""
        if op in ("s_lshl1_add_u32", "s_lshl2_add_u32",
                  "s_lshl3_add_u32", "s_lshl4_add_u32"):  # (s0 << N) + s1, N in the mnemonic
            _n = int(op[len("s_lshl"):len("s_lshl") + 1])
            self.write(dst, [((self.read(srcs[0], l) << _n) + self.read(srcs[1], l))
                             for l in range(self.n)])
            return True
        if op == "v_lshl_add_u32":                       # (s0 << s1) + s2
            self.write(dst, [((self.read(srcs[0], l) << (self.read(srcs[1], l) & 31))
                              + self.read(srcs[2], l)) for l in range(self.n)])
            return True
        if op == "v_lshl_or_b32":                        # (s0 << s1) | s2
            self.write(dst, [((self.read(srcs[0], l) << (self.read(srcs[1], l) & 31))
                              | self.read(srcs[2], l)) for l in range(self.n)])
            return True
        if op == "v_add_lshl_u32":                       # (s0 + s1) << s2
            self.write(dst, [((self.read(srcs[0], l) + self.read(srcs[1], l))
                              << (self.read(srcs[2], l) & 31)) for l in range(self.n)])
            return True
        if op == "v_mad_u32_u24":
            self.write(dst, [((self.read(srcs[0], l) & 0xFFFFFF) * (self.read(srcs[1], l) & 0xFFFFFF)
                              + self.read(srcs[2], l)) for l in range(self.n)])
            return True
        return False

    def _step_lane(self, op, dst, srcs) -> bool:
        """Forms that read or select across lanes: readfirstlane, carry-out add, cndmask."""
        if op == "v_readfirstlane_b32":
            # Reads lane 0 into an SGPR.  Sound here ONLY because the slice is uniform-by-
            # construction at every site that uses it (wave id, workgroup id); if a future slice
            self.write(dst, [self.read(srcs[0], 0)])
            return True
        if op == "v_add_co_u32":
            # dst, carry-out, src0, src1 -- the carry operand is parsed out by the caller and the
            # sum is what an address term needs.  32-bit LDS addresses never use the carry.
            self.write(dst, [self.read(srcs[-2], l) + self.read(srcs[-1], l)
                             for l in range(self.n)])
            return True
        if op == "v_cndmask_b32":
            self.write(dst, [self.read(srcs[1], l) if self.vcc[l] else self.read(srcs[0], l)
                             for l in range(self.n)])
            return True
        return False


    # ---- convenience ----------------------------------------------------------------------
    def lanes_of(self, vgpr_index: int) -> list[int]:
        return list(self.v.get(vgpr_index, [0] * self.n))

# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx1250 MXScaleVgprInterleave characterization.

Target code: KernelWriter._mxsInterleavePlan / placeMXScaleInterleave /
remapMXScaleInterleave and the interleaved RegSets in macroAndSet.

v_wmma_scale costs one cycle when its two scale VGPRs share a VGPR bank
(index % 8). The interleave is on by default (KernelWriter.do["MXScaleVgprInterleave"]).
Where it applies, every MXS reference must resolve inside v0-v255 and no VGPR may be reached
through two different scale blocks, and each v_wmma_scale must still read the scales the same
ds_loads wrote as in the default layout. By default it never raises the VGPR count: a layout
that needs a larger scale region than the default is taken only when the accumulators keep their
place, and otherwise the kernel gets the layout with the fewest bank conflicts that fits the
default region (kernels already at the 1024 VGPR limit would not assemble otherwise). With the
parameter MXScaleVgprInterleaveExtraVgpr the region grows instead and every v_wmma_scale reads
its two scales from disjoint banks. With the switch off every
kernel keeps the default layout, whose scale banks collide.

The loop-model cases keep a different number of MXSA and MXSB buffers, and their drain names a
later MXSA buffer through an earlier one's symbol. The three-buffer case needs one VGPR past the
default span, and StinkyTofu recolours its VGPRs, so its scales are read as registers.

pytestmark = pytest.mark.unit. CPU-only; no GPU.
"""

import os
import re

import pytest

from config_harness import assert_assembles, emit_kernels_from_config
from tensilelite.KernelWriter import KernelWriter

pytestmark = pytest.mark.unit

_ARCH = "gfx1250"

_DESIGNED = os.path.join(os.path.dirname(__file__), "data", "test_data", "_designed", "gfx1250")

# (config, fallsBack, extraMatters).
#   fallsBack:    the bank-conflict-free layout needs more VGPRs than the default AND the longer
#                 region would move the accumulators, so the kernels get the fewest-conflict
#                 layout that fits the default region instead.
#   extraMatters: MXScaleVgprInterleaveExtraVgpr is not folded away, so True and False are two
#                 kernels.  These were one flag until the region learned to grow when the
#                 accumulators keep their place: a layout can then be taken without the
#                 parameter (three_mxsa_buffers), while Solution still forks on it because
#                 mxs_interleave_extra answers from the Solution state, which does not carry
#                 where the accumulators start.
_CONFIGS = [
    ("cluster_padding_npot_mt1.yaml", True, False),
    ("streamk_cluster_multicast.yaml", False, False),
    ("streamk_cluster_multicast_pgr2.yaml", False, False),
    ("streamk_tdm_prefetchgl2.yaml", False, False),
    ("streamk_tdm_prefetchgl2_f4.yaml", False, False),
    ("loopmodel_mxf8_scale_buffers.yaml", False, False),
    ("loopmodel_mxf8_three_mxsa_buffers.yaml", False, True),
]
_IDS = [name for name, _, _ in _CONFIGS]
_VGPRS = re.compile(r"\.amdhsa_next_free_vgpr\s+(\d+)")


_SET = re.compile(r"^\.set\s+(\w+),\s*(.+?)\s*$")
_RANGE = re.compile(r"v\[([^\]:]+)(?::([^\]]+))?\]")
_MXS = re.compile(r"vgprValuMXS([AB])_X(\d+)_I(\d+)(?:_P(\d+))?\b")
_OPERANDS = re.compile(r",\s*(?![^\[]*\])")


def _eval(expr, sym):
    text = re.sub(r"[A-Za-z_]\w*", lambda m: str(sym.get(m.group(0), "None")), expr)
    try:
        return int(eval(text))  # noqa: S307 - integer arithmetic on resolved .set values
    except (TypeError, SyntaxError, NameError):
        return None


def _range(operand, sym):
    m = _RANGE.match(operand.strip())
    if not m:
        return None
    lo = _eval(m.group(1), sym)
    hi = _eval(m.group(2), sym) if m.group(2) else lo
    return None if lo is None or hi is None else range(lo, hi + 1)


_MSB_BIAS = re.compile(r"\s*-\s*(?:256|512|768)\s*$")
_PLAIN = re.compile(r"v(\d+)$")


def _physical(operand, sym):
    """The VGPRs an operand names. A trailing -256, -512 or -768 is the s_set_vgpr_msb window the
    index is encoded in, not part of the index: v[856-768:871-768] is v856."""
    operand = operand.strip()
    m = _PLAIN.match(operand)
    if m:
        return range(int(m.group(1)), int(m.group(1)) + 1)
    m = _RANGE.match(operand)
    if not m:
        return None
    lo = _eval(_MSB_BIAS.sub("", m.group(1)), sym)
    hi = _eval(_MSB_BIAS.sub("", m.group(2)), sym) if m.group(2) else lo
    return None if lo is None or hi is None else range(lo, hi + 1)


def _feeds(src):
    """For each v_wmma_scale in program order, the ds_loads that last wrote its MXSA and MXSB
    scales, named by opcode and LDS offset -- which data the WMMA reads, whatever holds it."""
    sym, last, out = {}, {}, []
    for raw in src.splitlines():
        line = raw.split("//")[0].strip()
        m = _SET.match(line)
        if m:
            value = _eval(m.group(2), sym)
            if value is None:
                sym.pop(m.group(1), None)
            else:
                sym[m.group(1)] = value
            continue
        op, _, rest = line.partition(" ")
        if op.startswith("ds_load"):
            offset = re.search(r"\boffset:(\d+)", rest)
            for v in _physical(_OPERANDS.split(rest)[0], sym) or ():
                last[v] = "%s@%s" % (op, offset.group(1) if offset else 0)
        elif op.startswith("v_wmma_scale"):
            ops = _OPERANDS.split(rest)
            out.append(tuple(frozenset(last.get(v) for v in _physical(ops[i].split()[0], sym) or ())
                             for i in (5, 4)))
    return out


def _scan(src):
    """(wmma count, bank conflicts, MXS VGPRs >= 256, VGPRs shared by two blocks, piece symbols)."""
    sym, owner = {}, {}
    wmma = conflicts = high = shared = 0
    pieces = set()
    for raw in src.splitlines():
        line = raw.split("//")[0].strip()
        m = _SET.match(line)
        if m:
            value = _eval(m.group(2), sym)
            if value is None:
                sym.pop(m.group(1), None)
            else:
                sym[m.group(1)] = value
            continue
        if not line or line.startswith(".") or line.endswith(":") or " " not in line:
            continue
        ops = _OPERANDS.split(line.split(None, 1)[1])
        for op in ops:
            mm = _MXS.search(op)
            regs = _range(op.split()[0], sym) if mm else None
            if regs is None:
                continue
            if mm.group(4):
                pieces.add(mm.group(0))
            block = (mm.group(1), sym.get("vgprValuMXS%s_X%s_I%s" % mm.group(1, 2, 3)))
            for v in regs:
                high += v >= 256
                shared += owner.setdefault(v, block) != block
        if line.startswith("v_wmma_scale"):
            wmma += 1
            # A recoloured kernel names its scales as plain registers, not through the symbols.
            s0, s1 = _physical(ops[4].split()[0], sym), _physical(ops[5].split()[0], sym)
            high += sum(v >= 256 for regs in (s0, s1) if regs for v in regs)
            if s0 and s1 and {v % 8 for v in s0} & {v % 8 for v in s1}:
                conflicts += 1
    return wmma, conflicts, high, shared, pieces


def _emit(name):
    return emit_kernels_from_config(os.path.join(_DESIGNED, name), limit=8, arch=_ARCH)


@pytest.fixture(scope="module", params=_CONFIGS, ids=_IDS)
def kernels(request):
    name, fallsBack, _extraMatters = request.param
    return fallsBack, _emit(name)


@pytest.fixture(scope="module", params=_CONFIGS, ids=_IDS)
def kernels_on_off(request):
    name, fallsBack, _extraMatters = request.param
    on = _emit(name)
    init = KernelWriter.__init__

    def initOff(self, *args, **kwargs):
        init(self, *args, **kwargs)
        self.do["MXScaleVgprInterleave"] = False

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(KernelWriter, "__init__", initOff)
        off = _emit(name)
    return fallsBack, on, off


def _vgprs(src):
    return int(_VGPRS.search(src).group(1))


def test_mx_scale_vgpr_interleave_emits(kernels):
    out = kernels[1]
    assert out, "no kernels emitted"
    for base, src, err in out:
        assert err == 0, f"kernel {base!r} emitted with err={err}"


def test_mx_scale_vgpr_interleave_banks(kernels):
    fallsBack, out = kernels
    if fallsBack:
        pytest.skip("not conflict free in the default VGPR budget; see test_mx_scale_vgpr_interleave_tight_budget")
    for base, src, _err in out:
        wmma, conflicts, high, shared, _pieces = _scan(src)
        assert wmma > 0, f"{base}: no v_wmma_scale"
        assert conflicts == 0, f"{base}: {conflicts}/{wmma} v_wmma_scale read both scales from one bank"
        assert high == 0, f"{base}: MXS VGPR past v255"
        assert shared == 0, f"{base}: a VGPR is reached through two scale blocks"


def test_mx_scale_vgpr_interleave_assembles(kernels):
    for base, src, _err in kernels[1]:
        assert_assembles(src, base)


def test_mx_scale_vgpr_interleave_never_adds_vgprs(kernels_on_off):
    _fallsBack, on, off = kernels_on_off
    assert [b for b, _s, _e in off] == [b for b, _s, _e in on]
    for (base, src, _e), (_b, srcOff, _eOff) in zip(on, off):
        assert _vgprs(src) <= _vgprs(srcOff), f"{base}: the interleave raised the VGPR count"


def test_mx_scale_vgpr_interleave_keeps_each_wmmas_scales(kernels_on_off):
    # The interleave moves scales, never which ones a WMMA reads: every v_wmma_scale must take both
    # from the same ds_load as in the default layout. A bank count cannot see a reference remapped
    # onto the wrong register, and recolouring can resolve an undefined one without complaint.
    _fallsBack, on, off = kernels_on_off
    for (base, src, _e), (_b, srcOff, _eOff) in zip(on, off):
        feeds, feedsOff = _feeds(src), _feeds(srcOff)
        assert feeds and len(feeds) == len(feedsOff), f"{base}: v_wmma_scale count differs"
        wrong = [i for i, (a, b) in enumerate(zip(feeds, feedsOff)) if a != b]
        assert not wrong, (f"{base}: {len(wrong)}/{len(feeds)} v_wmma_scale read a scale another "
                           f"ds_load wrote, first #{wrong[0]}: {feeds[wrong[0]]} vs {feedsOff[wrong[0]]}")


def test_mx_scale_vgpr_interleave_tight_budget(kernels_on_off):
    # The conflict-free layout needs a larger scale region than the default: the kernel keeps
    # the default VGPR count and gets fewer scale-bank conflicts than the default layout.
    fallsBack, on, off = kernels_on_off
    if not fallsBack:
        pytest.skip("the conflict-free layout fits these kernels")
    fewer = 0
    for (base, src, _e), (_b, srcOff, _eOff) in zip(on, off):
        wmma, conflicts, high, shared, pieces = _scan(src)
        assert pieces, f"{base}: the interleave was not applied"
        assert high == 0 and shared == 0
        assert _vgprs(src) <= _vgprs(srcOff)
        offConflicts = _scan(srcOff)[1]
        assert conflicts <= offConflicts, f"{base}: more bank conflicts than the default layout"
        fewer += conflicts < offConflicts
    assert fewer >= 1, "no kernel has fewer scale-bank conflicts than the default layout"


def _emit_with_extra_vgpr(name, tmp_path_factory, values="true", limit=8):
    """The same designed config with MXScaleVgprInterleaveExtraVgpr: [values] forked in."""
    with open(os.path.join(_DESIGNED, name)) as f:
        text = f.read()
    text, n = re.subn(r"(ForkParameters:\n)(\s*)- ",
                      lambda m: f"{m.group(1)}{m.group(2)}- MXScaleVgprInterleaveExtraVgpr: [{values}]\n{m.group(2)}- ",
                      text, count=1)
    assert n == 1
    path = tmp_path_factory.mktemp("extra_vgprs") / name
    path.write_text(text)
    return emit_kernels_from_config(str(path), limit=limit, arch=_ARCH)


@pytest.fixture(scope="module", params=_CONFIGS, ids=_IDS)
def kernels_extra_requested(request, tmp_path_factory):
    name, _fallsBack, extraMatters = request.param
    return extraMatters, _emit(name), _emit_with_extra_vgpr(name, tmp_path_factory)


@pytest.fixture(scope="module", params=[c for c in _CONFIGS if c[2]], ids=[c[0] for c in _CONFIGS if c[2]])
def kernels_extra_vgprs(request, tmp_path_factory):
    name, _fallsBack, _extraMatters = request.param
    return _emit_with_extra_vgpr(name, tmp_path_factory)


def _without_random_labels(src):
    # The cluster barrier labels carry a random suffix that differs between two emissions.
    # The loop-counter gate names the same signal through an `LCL_` variant of that label.
    return re.sub(r"(skipCBPreSignal_(?:LCL_)?)[A-Za-z0-9]{16}", r"\1X", src)


def test_extra_vgpr_is_folded_unless_the_layout_needs_it(kernels_extra_requested):
    # Solution folds True into False when the conflict-free layout fits the default span: the kernel is
    # then identical to the one without the parameter. Where it does not fit, True changes the layout.
    # (The harness canonicalizes kernel names, so the sources are compared.)
    extraMatters, default, extra = kernels_extra_requested
    assert default and len(default) == len(extra)
    for (_, defaultSrc, _), (_, extraSrc, extraErr) in zip(default, extra):
        assert extraErr == 0
        assert (_without_random_labels(defaultSrc) != _without_random_labels(extraSrc)) == extraMatters


def test_extra_vgpr_on_a_non_mx_kernel_is_rejected(tmp_path_factory):
    # rich_gemm_sb has no MX scales: with the parameter set to True no solution survives.
    assert _emit_with_extra_vgpr("rich_gemm_sb.yaml", tmp_path_factory) == []


def test_mx_scale_vgpr_interleave_extra_vgprs_is_conflict_free(kernels_extra_vgprs):
    # With extra VGPRs allowed the region grows to hold the conflict-free layout.
    for base, src, err in kernels_extra_vgprs:
        assert err == 0, f"kernel {base!r} emitted with err={err}"
        wmma, conflicts, high, shared, _pieces = _scan(src)
        # Not `pieces`: a block only carries a _P<k> symbol when it is cut into more than one
        # piece, and a kernel whose blocks each hold one has none to show. What the extra VGPRs
        # buy is read off the v_wmma_scale operands below, which every kernel has.
        assert wmma > 0
        assert conflicts == 0, f"{base}: {conflicts}/{wmma} v_wmma_scale read both scales from one bank"
        assert high == 0 and shared == 0


def test_mx_scale_vgpr_interleave_switch_off(kernels_on_off):
    # Same kernels (the switch is not part of the name) in the default layout, where the
    # scale banks collide.
    fallsBack, on, off = kernels_on_off
    colliding = 0
    for base, src, err in off:
        assert err == 0, f"kernel {base!r} emitted with err={err}"
        _wmma, conflicts, _high, _shared, pieces = _scan(src)
        assert not pieces, f"{base}: interleave pieces with the switch off"
        colliding += conflicts > 0
    assert colliding >= 1, "no kernel has scale-bank conflicts in the default layout"

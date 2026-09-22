# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
"""UseLoopModel asymmetric TDMSplit -- gfx1250 characterization (CPU-only).

Pins the mixed-region split family: exactly one of TDMSplitA/TDMSplitB set, so a
Phi fuse group holds members with different region counts. Four of the five
defects this family exposed live BELOW GIR -- in KernelWriterAssembly and in
SolutionStructs/segment_interleave -- where the LoopModel digest goldens cannot
see them, and each was a kernel-wide question standing in for a per-operand one.

The emitted assembly carries a ``<GIR: ...>`` provenance tag on every
GIR-derived instruction, so the per-kernel tag histogram is a direct fingerprint
of the lowered plan: which movements copy in which generation, which descriptor
each swap and each global-read increment belongs to, and where the fences land.
Every one of the five defects changes that histogram.

Goldens are plain files under ``__goldens__/``, matching the sibling LoopModel
suite rather than this suite's syrupy protocol, so the whole LoopModel arm
regenerates with one switch. Regenerate with ``LOOPMODEL_GOLDEN_UPDATE=1``.

CPU-only: no GPU required.
"""

import collections
import json
import os
import pathlib
import re

import pytest

from config_harness import (
    assert_assembles,
    assert_real_gfx1250_kernels,
    emit_kernels_from_config,
)
from Tensile.Lowering.gir_tag import GIR_TAG_RE

pytestmark = pytest.mark.unit

_ARCH = "gfx1250"

_DESIGNED = os.path.join(os.path.dirname(__file__), "data", "test_data",
                         "_designed", _ARCH)

# family -> (designed config, expected asymmetric-shape count). The count is what
# survives Solution derivation: at LDSSI=1 the `[2,2]` segment map has no
# portSplitB layout, so `0/1` is rejected and only `1/0` reaches emit.
_FAMILIES = {
    "ldssi0": ("loopmodel_mixedsplit.yaml", 8),
    "ldssi1": ("loopmodel_mixedsplit_ldssi.yaml", 1),
}

GOLDENS = pathlib.Path(__file__).parent / "__goldens__"
UPDATING = os.environ.get("LOOPMODEL_GOLDEN_UPDATE") == "1"

# TDMFuse 0/1/2/3 x TDMSplitA/B in {0,1}. Solution derivation drops the shapes it
# rejects, so the emitted count is a characterized fact, not a computed one.
_LIMIT = 16

_TAG_RE = GIR_TAG_RE
# Tile/substep coordinates make a read tag unique per instruction; the histogram
# wants the op class, so they are folded away.
_COORD_RE = re.compile(r"\[[^\]]*\]|->X\d+")

_SWAP_RE = re.compile(r"^swap (.+) (copy|read)-hop$")
# `s_xor_b32 s[sgprtdmAGroup0+1], ...` -- the descriptor the swap rotates.
_XOR_DESC_RE = re.compile(r"s_xor_b32 s\[sgprtdm(\w+?)Group0\+1\]")

# Conservative SCC writers: everything that could break a Mark's compare -> select pair. The
# 64-bit adds are excluded because gfx1250 spells them `s_add_nc_u64` -- no carry, no SCC write
# (Gfx1250Instructions.def gives `ImplicitWriteSCC` to `s_add_u32`/`s_sub_u32` and not to `*_u64`).
_SCC_WRITE_RE = re.compile(
    r"^\s*s_(?!(?:add|sub)_u64)"
    r"(cmp|bitcmp|cselect|add|addc|sub|subb|and|or|xor|nand|nor|xnor|lshl|lshr"
    r"|ashr|abs|min|max|bfe|bfm|not|bcnt|quadmask|wqm|pack|setreg)")


def _check(name, produced, dump=str):
    """Compare against the stored golden, or write it when regenerating."""
    path = GOLDENS / name
    if UPDATING:
        path.parent.mkdir(exist_ok=True)
        path.write_text(dump(produced))
        pytest.skip("golden %s written" % name)
    assert path.exists(), "missing golden %s; regenerate with LOOPMODEL_GOLDEN_UPDATE=1" % name
    return path.read_text()


def _tag_histogram(src):
    """``{normalised GIR tag: count}`` for one kernel."""
    tags = (_COORD_RE.sub("", t).strip() for t in _TAG_RE.findall(src))
    return dict(sorted(collections.Counter(tags).items()))


def _copy_swap_units_by_descriptor(src):
    """``{descriptor: {unit, ...}}`` over the copy-hop buffer swaps.

    A copy-hop swap lowers to one ``s_xor`` on its descriptor's SGPR. Keyed on
    the movement rather than the descriptor, an asymmetric split draws two swaps
    for one descriptor -- and two identical XORs cancel, so the LDS double buffer
    never alternates.
    """
    by_desc = collections.defaultdict(set)
    for line in src.splitlines():
        m = _XOR_DESC_RE.search(line)
        tag = _TAG_RE.search(line)
        if not (m and tag):
            continue
        swap = _SWAP_RE.match(tag.group(1).strip())
        if swap and swap.group(2) == "copy":
            by_desc[m.group(1)].add(swap.group(1))
    return by_desc


def _arm(src):
    """The readable arm name -- ``f<TDMFuse> <TDMSplitA>/<TDMSplitB>``.

    The emitted basename is a hash, so the shape is recovered from the long
    kernel name the source carries.
    """
    m = re.search(r"_TDMF(\d)_", src)
    s = re.search(r"_TDMSA(\d)_TDMSB(\d)_", src)
    assert m and s, "kernel source carries no TDMFuse/TDMSplit name tags"
    return "f%s %s/%s" % (m.group(1), s.group(1), s.group(2))


_EMITTED = {}


@pytest.fixture(scope="module", params=sorted(_FAMILIES))
def family(request):
    """``(name, [(arm, base, src, err), ...])`` for one designed config."""
    name = request.param
    if name not in _EMITTED:
        config, _ = _FAMILIES[name]
        results = emit_kernels_from_config(os.path.join(_DESIGNED, config),
                                           limit=_LIMIT, arch=_ARCH)
        _EMITTED[name] = [(_arm(src), base, src, err) for base, src, err in results]
    return name, _EMITTED[name]


def test_loopmodel_mixedsplit_emits_assembly(family):
    """Every derived shape emits real, assemblable gfx1250 UseLoopModel assembly."""
    _name, emitted = family
    assert_real_gfx1250_kernels([(b, s, e) for _a, b, s, e in emitted])
    for arm, base, src, _err in emitted:
        assert_assembles(src, base)
        assert "_ULM1_" in src, f"Kernel {arm} is not a UseLoopModel kernel"


def test_the_asymmetric_family_is_covered(family):
    """The goldens below prove nothing if no asymmetric shape survives derivation."""
    name, emitted = family
    asymmetric = [a for a, _b, _s, _e in emitted if a.endswith(("1/0", "0/1"))]
    assert len(asymmetric) == _FAMILIES[name][1], (
        "%s: expected %d asymmetric-split kernels, got %d: %s"
        % (name, _FAMILIES[name][1], len(asymmetric), sorted(asymmetric)))


def test_copy_hop_swap_is_keyed_on_the_descriptor(family):
    """One descriptor, one copy-hop swap unit.

    Two units on one descriptor is the cancelling XOR pair: both rotate the same
    SGPR by the same constant, so the buffer stays put and each trip overwrites
    what the previous trip is reading.
    """
    _name, emitted = family
    seen = 0
    for arm, _base, src, _err in emitted:
        for desc, units in _copy_swap_units_by_descriptor(src).items():
            seen += 1
            assert len(units) == 1, (
                f"Kernel {arm} descriptor tdm{desc} carries {len(units)} copy-hop "
                f"swap units {sorted(units)}; two lower to the same s_xor and cancel"
            )
    assert seen, "no copy-hop swap found -- the check is vacuous"


def test_mark_lowering_is_branchless(family):
    """No Mark lowers to a branch.

    Every GIR Mark is a predicate on the wave, not control flow, and these sit in
    the steady body: the descriptor enable/null pair alone was a compare, a
    branch, a mov, a jump and three labels per generation.
    """
    _name, emitted = family
    for arm, _base, src, _err in emitted:
        for line in src.splitlines():
            if "s_branch" in line or "s_cbranch" in line:
                assert "<GIR:" not in line, f"Kernel {arm} lowers a Mark to a branch: {line.strip()}"


def test_mark_predicates_keep_their_scc(family):
    """A Mark's ``s_cselect`` reads the SCC its own compare set, so nothing between
    them may write SCC. The scheduler is free to interleave -- it just may not
    interleave an SALU that clobbers the condition."""
    _name, emitted = family
    pairs = 0
    for arm, _base, src, _err in emitted:
        lines = src.splitlines()
        opened = {}
        for i, line in enumerate(lines):
            m = _TAG_RE.search(line)
            if not m:
                continue
            tag = m.group(1)
            if "s_bitcmp" in line or (line.lstrip().startswith("s_cmp")):
                opened[tag] = i
            elif "s_cselect" in line and tag in opened:
                start = opened.pop(tag)
                pairs += 1
                clobber = [lines[j].strip() for j in range(start + 1, i)
                           if _SCC_WRITE_RE.match(lines[j])]
                assert not clobber, (
                    f"Kernel {arm}: <GIR: {tag}> selects on an SCC clobbered by "
                    f"{clobber[0][:70]!r}")
    assert pairs, "no Mark compare/select pair found -- the check is vacuous"


def test_loopmodel_mixedsplit_gir_tags_golden(family):
    """Golden: per-kernel ``<GIR: ...>`` tag histogram -- the lowered plan's
    fingerprint, sensitive to every movement, swap, increment and fence."""
    name, emitted = family
    produced = {arm: _tag_histogram(src) for arm, _base, src, _err in emitted}
    stored = _check("gir_tags_mixedsplit_%s.json" % name, produced,
                    dump=lambda d: json.dumps(d, indent=1, sort_keys=True) + "\n")
    expected = json.loads(stored)
    moved = {k: (expected.get(k), v) for k, v in produced.items() if expected.get(k) != v}
    assert not moved, "%d kernel(s) changed, e.g. %s" % (
        len(moved), list(moved.items())[:1])
    assert set(produced) == set(expected), "the emitted kernel set changed"


def test_loopmodel_mixedsplit_golden(family):
    """Golden: ``{arm: emitter return code}``. Keyed on the shape, not on the
    basename hash, so a rejected or newly-admitted shape shows as itself."""
    name, emitted = family
    produced = {arm: err for arm, _base, _src, err in emitted}
    stored = _check("digest_mixedsplit_%s.json" % name, produced,
                    dump=lambda d: json.dumps(d, indent=1, sort_keys=True) + "\n")
    assert produced == json.loads(stored)

# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx1250 companion tiles-per-buffer characterization.

Target code: LoopModel/adapter.py ``_set_register_depths``.

One buffer holds ``unit // vg`` tiles, and a carrier ring is that same cut made by the merge, so
an operand that already has one is not cut again. A companion's ring is then the larger of the
generations its parent keeps and the tiles that depth covers. Counting the parent's coverage in
whole units instead of buffers over-states it by ``vg``: MXSA takes 3 buffers where 2 span the
same tiles, its read lead comes out a generation early, and the frame counter flow derives its
waits against a pipeline deeper than the one emitted.

pytestmark = pytest.mark.unit. CPU-only; no GPU.
"""

import os
import re

import pytest

from config_harness import emit_kernels_from_config

pytestmark = pytest.mark.unit

_ARCH = "gfx1250"
_DESIGNED = os.path.join(os.path.dirname(__file__), "data", "test_data", "_designed", "gfx1250")
_CONFIG = "loopmodel_mxf8_companion_tiles_per_buffer.yaml"


@pytest.fixture(scope="module")
def kernels():
    return emit_kernels_from_config(os.path.join(_DESIGNED, _CONFIG), limit=8, arch=_ARCH)


def test_companion_ring_spans_its_parents_coverage(kernels):
    """Emits, and MXSA spans A's coverage in 2 buffers rather than its parent's 3."""
    assert kernels, "no kernel was emitted for %s" % _CONFIG
    for basename, source, err in kernels:
        assert err == 0, "%s did not emit (err=%s)" % (basename, err)
        buffers = sorted(set(re.findall(r"^\.set vgprValuMXSA_X(\d+)_I0,", source, re.M)))
        assert buffers == ["0", "1"], "MXSA took buffers %s, expected 2" % buffers

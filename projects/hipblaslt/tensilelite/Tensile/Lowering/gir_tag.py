# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
The `<GIR: ...>` assembly-comment grammar, for the emitter and every reader. Stdlib only: the tools
that parse emitted assembly must not need rocisa.
"""

from __future__ import annotations

import re

#: A body may contain `>`, but only as an arrow (`shared->reg`) where a word follows it, so the
#: CLOSING `>` is the one followed by whitespace or end of line. Stopping at the first `>` truncates.
GIR_TAG_RE = re.compile(r"<GIR: ([^<\n]+?)>(?=\s|$)")


def gir_tag(body):
    return "<GIR: %s>" % body


def sync_comment(tokens):
    """A memory-token set as `Components/TensorDataMover.issueLoad` spells it."""
    toks = sorted(tokens)
    return "sync LDS%u" % toks[0] if len(toks) == 1 else "sync LDS %s" % toks

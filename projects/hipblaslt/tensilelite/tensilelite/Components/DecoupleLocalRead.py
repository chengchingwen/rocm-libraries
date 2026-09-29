# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Per-operand `PrefetchLocalRead` and `ClusterLocalRead`, the way `DecouplePGR` does the global side.

A scalar names both operands; an `A`/`B` pair names them apart.  The two spellings mean the same
thing when the pair is equal, so an equal pair COLLAPSES to the scalar and its keys are dropped --
otherwise one schedule would answer to two names and dedupe would miss it.

`AUTO` (-1) asks for the value the operand's own position implies.  It is resolved here, not left
for a reader to interpret: every later rule sees a concrete number.
"""

#: `-1` on a per-operand key: derive it from the operand's position in the nest.
AUTO = -1


def _pair(ks, base):
    """`(decoupled, a, b)` for one base parameter, falling back to its scalar."""
    scalar = ks.get(base, 0)
    a, b = ks.get(base + "A"), ks.get(base + "B")
    if a is None and b is None:
        return False, scalar, scalar
    return True, scalar if a is None else a, scalar if b is None else b


def localReadLevels(ks):
    """`(decoupled, plrA, plrB)` -- the read-ahead depth each operand asks for."""
    return _pair(ks, "PrefetchLocalRead")


def clusterLevels(ks):
    """`(decoupled, clrA, clrB)` -- whether each operand gets a full register buffer."""
    return _pair(ks, "ClusterLocalRead")


def equalPairDegeneratesToScalar(ks, base):
    """True when both per-operand levels name the same real value, so the pair is the scalar."""
    decoupled, a, b = _pair(ks, base)
    return bool(decoupled and a == b and a != AUTO)


def collapseEqualPair(ks, base):
    """Drop an equal pair onto the scalar; return the value it pinned, or None."""
    if not equalPairDegeneratesToScalar(ks, base):
        return None
    _decoupled, value, _b = _pair(ks, base)
    for suffix in ("A", "B"):
        ks.pop(base + suffix, None)
    ks[base] = value
    return value

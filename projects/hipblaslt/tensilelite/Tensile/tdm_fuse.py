# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Semantics of the `TDMFuse` Solution parameter: which operands share one TDM descriptor.

A fuse GROUP is a set of operands driven by one descriptor, with the waves partitioned between its
members -- by parity when the group has two members and no explicit share, otherwise by contiguous
ranges. `Solution.py` resolves the parameter; this module says what its values mean, so the
scaffold, the cluster-load masks and the decoder all read one table.
"""

#: `TDMFuse` value -> the operand groups sharing a descriptor.
FUSE_GROUPS = {
    0: [],
    1: [["A", "B"], ["MXSA", "MXSB"]],
    2: [["A", "MXSA", "MXSB"]],
    3: [["B", "MXSA", "MXSB"]],
    4: [["MXSA", "MXSB"], ["A", "B"]],
    5: [["MXSA", "A"], ["B", "MXSB"]],
}

FUSE_NAME = {0: "none", 1: "AB+MX", 2: "A_MX", 3: "B_MX", 4: "MX_AB", 5: "paired"}

#: Wave share per group member; `None` means an even split, which `tdmWaveSelect` reads as parity.
FUSE_WAVE_SHARES = {
    0: [],
    1: [None, None],
    2: [[2, 1, 1]],
    3: [[2, 1, 1]],
    4: [None, None],
    5: [None, None],
}

EMITTABLE = (0, 1, 2, 3, 4, 5)

#: Groups that only exist on a microscaled kernel
_MX_MEMBERS = ("MXSA", "MXSB")


def wave_ranges(fuse, group_index, num_waves):
    """`[(member_index, first_wave, count)]` for one group, or None for the parity rule."""
    shares = FUSE_WAVE_SHARES.get(int(fuse), [])
    if group_index >= len(shares) or shares[group_index] is None:
        return None
    s = shares[group_index]
    total = sum(s)
    if num_waves % total:
        raise RuntimeError(
            f"TDMFuse={fuse} group {group_index} has wave shares {s} summing to {total}, which "
            f"does not divide NumWaves={num_waves} -- the partition would leave a wave unassigned.")
    unit = num_waves // total
    out, first = [], 0
    for i, share in enumerate(s):
        out.append((i, first, share * unit))
        first += share * unit
    return out


def fuse_groups(fuse, present):
    """The Phi groups for `fuse`, restricted to operands actually `present`."""
    out = []
    for group in FUSE_GROUPS.get(int(fuse), []):
        members = tuple(axis for axis in group if axis in present)
        if len(members) > 1:
            out.append(members)
    return out


def pairs_a_and_b_by_parity(fuse) -> bool:
    """Does `fuse` put A and B on ONE descriptor selected by wave parity?

    True only for a two-member `[A, B]` group with no explicit share -- the one shape whose member
    choice is the same `s_bitcmp1 WaveIdx, 0` the combined multicast mask uses.
    """
    shares = FUSE_WAVE_SHARES.get(int(fuse), [])
    for index, group in enumerate(FUSE_GROUPS.get(int(fuse), [])):
        if len(group) != 2 or (index < len(shares) and shares[index] is not None):
            continue
        if sorted(group) == ["A", "B"]:
            return True
    return False


def needs_mx(fuse):
    """Does this fuse mean nothing unless MX scales are present?"""
    return bool(fuse_groups(fuse, ("A", "B"))) is False and bool(
        fuse_groups(fuse, ("A", "B") + _MX_MEMBERS))


def crosses_data_and_scale(fuse):
    """Does any group mix a data operand with a scale one?"""
    for group in FUSE_GROUPS.get(int(fuse), []):
        mx = [axis for axis in group if axis in _MX_MEMBERS]
        if mx and len(mx) != len(group):
            return True
    return False


def max_group_size(fuse):
    """Largest group in `fuse` -- 3 for `A_MX`/`B_MX`, 2 for the rest, 0 for unfused."""
    return max((len(group) for group in FUSE_GROUPS.get(int(fuse), [])), default=0)

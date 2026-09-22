#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Build and render a LoopModel schedule without the Tensile adapter.

`test_a_hand_authored_theta_emits_the_SAME_LoopIR_as_the_adapted_one` proves this surface is
complete: a theta authored this way emits the LoopIR the adapter's does for the same kernel.
"""

from Tensile.LoopModel import (
    Axis, Fragment, Global, Operand, Shared, Theta, Trajectory, render_theta,
)


def staged(name, free_axis, broadcast_axes, copy_ahead=1, read_ahead=0):
    shared = Shared(ring_depth=2, offsets={"iter": copy_ahead})
    fragment = Fragment(
        broadcast_axes=set(broadcast_axes),
        fragment_elements=4,
        offsets={"K": read_ahead} if read_ahead else {})
    return Operand(
        name, free_axis,
        trajectory=Trajectory(Global(), shared, fragment),
        elem_bytes=2)


def gemm(order=("K", "M", "N"), pgr=1, plr=0, waves=1):
    extent = {"K": 2, "M": 2, "N": 2}
    operands = [
        staged("A", "M", {"N"}, pgr, plr),
        staged("B", "N", {"M"}, pgr, plr),
        Operand(
            "C", "M",
            trajectory=Trajectory(Fragment(broadcast_axes={"K"})),
            role="output"),
    ]
    for operand in operands[:2]:
        fragment = operand.trajectory.fragment
        fragment.offsets = {order[0]: plr} if plr else {}
    return Theta(
        operands=operands,
        ord=(Axis("iter", 0),) + tuple(Axis(a, extent[a]) for a in order),
        wave_count=waves)


def main():
    for label, theta in (("KMN, PGR1, PLR0", gemm()),
                         ("MKN, PGR1, PLR0", gemm(order=("M", "K", "N"))),
                         ("KMN, PGR2, PLR1", gemm(pgr=2, plr=1))):
        print("=" * 78)
        print(label)
        print(render_theta(theta))


if __name__ == "__main__":
    main()

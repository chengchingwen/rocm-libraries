# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from dataclasses import fields
import inspect
import subprocess
import sys

import pytest

import Tensile.LoopModel.loopir as loopir


def _transfer(local_tile, source, destination, where=loopir.TRUE, effect=None):
    return loopir.Transfer(
        local_tile,
        source,
        destination,
        loopir.Effect(
            local_tile if effect is None else effect,
            where,
        ),
    )


def _lower_theta(theta):
    from Tensile.LoopModel.checks import validate_loopir
    from Tensile.LoopModel.decoder import decode_theta
    from Tensile.Lowering.loopir_to_gir import lower_to_gir
    from Tensile.Lowering.gir.passes import pipeline as gir_pipeline
    from Tensile.Lowering.gir.passes import run_pipeline

    decoded = decode_theta(theta)
    assert not validate_loopir(decoded.program, decoded.profile)
    program = lower_to_gir(decoded.program, decoded.profile)
    run_pipeline(program, passes=gir_pipeline())
    return decoded, program


def test_loopir_import_is_theta_independent():
    script = """
import sys
import Tensile.LoopModel.loopir
forbidden = (
    'Tensile.LoopModel.theta',
    'Tensile.LoopModel.decoder',
    'Tensile.LoopModel.emit',
    'Tensile.LoopModel.ir',
    'Tensile.LoopModel.schedule',
)
loaded = tuple(name for name in forbidden if name in sys.modules)
raise SystemExit(repr(loaded) if loaded else 0)
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def _g3_program():
    wave = loopir.Axis("wave", 2)
    tile = loopir.Axis("tile", 3)
    a = loopir.Operand("A", (tile,))
    copy = loopir.CompletionClass("copy", loopir.WAVE)
    read = loopir.CompletionClass("read", loopir.WAVE)

    load0 = loopir.Movement(
        loopir.TENSOR_LOAD,
        (_transfer(
            a[tile],
            loopir.GLOBAL,
            loopir.SHARED,
            2 * tile + wave < 3,
            effect=a[wave, tile],
        ),),
        copy,
    )
    load1 = loopir.Movement(
        loopir.TENSOR_LOAD,
        (_transfer(
            a[tile],
            loopir.GLOBAL,
            loopir.SHARED,
            2 * tile + wave >= 3,
            effect=a[wave, tile],
        ),),
        copy,
    )
    reads = tuple(
        loopir.Movement(
            loopir.DS_LOAD,
            (_transfer(
                a[index],
                loopir.SHARED,
                loopir.REGISTER,
                effect=a[wave, index],
            ),),
            read,
        )
        for index in range(3)
    )
    program = loopir.Program(
        (),
        (tile,),
        (a,),
        (
            loopir.Region(loopir.STEADY),
            load0,
            load1,
            *reads,
        ),
        ((wave, loopir.WAVE),),
    )
    return program, wave, load0, load1, reads


def _raw_points(program, complete_operation, issue_operation):
    relation = next((
        dependence
        for dependence in loopir.dependences(program)
        if dependence.kind is loopir.RAW
        and dependence.complete.operation is complete_operation
        and dependence.issue.operation is issue_operation
        and dependence.complete.level is loopir.SHARED
        and dependence.issue.level is loopir.SHARED
    ), None)
    return loopir.dependence_points(relation) if relation is not None else ()


def test_g3_tensor_load_dependencies_are_exact_per_wave():
    program, wave, load0, load1, reads = _g3_program()

    assert {
        value for complete, _issue in _raw_points(program, load0, reads[0])
        for axis, value in complete.agents if axis is wave
    } == {0, 1}
    assert not _raw_points(program, load1, reads[0])

    assert {
        value for complete, _issue in _raw_points(program, load0, reads[1])
        for axis, value in complete.agents if axis is wave
    } == {0}
    assert {
        value for complete, _issue in _raw_points(program, load1, reads[1])
        for axis, value in complete.agents if axis is wave
    } == {1}

    assert not _raw_points(program, load0, reads[2])
    assert {
        value for complete, _issue in _raw_points(program, load1, reads[2])
        for axis, value in complete.agents if axis is wave
    } == {0, 1}


def test_dependence_repr_does_not_embed_program_subtrees():
    program, _wave, _load0, _load1, _reads = _g3_program()
    assert max(
        len(repr(dependence))
        for dependence in loopir.dependences(program)
    ) < 4096


def test_new_loopir_checker_rejects_reordered_phase_markers():
    from Tensile.LoopModel.checks import validate_loopir

    tile = loopir.Axis("tile", 1)
    operand = loopir.Operand("A", (tile,))
    move = loopir.Movement(
        loopir.TENSOR_LOAD,
        (_transfer(operand[tile], loopir.GLOBAL, loopir.SHARED),),
        loopir.CompletionClass("copy", loopir.WAVE),
    )
    program = loopir.Program(
        (),
        (tile,),
        (operand,),
        (
            loopir.Region(loopir.STEADY),
            move,
            loopir.Region(loopir.PROLOGUE),
            move,
        ),
    )
    assert any(
        "out of order" in error
        for error in validate_loopir(program)
    )


def test_loopir_checker_rejects_noncovering_tensorload_effects():
    from Tensile.LoopModel.checks import validate_loopir

    tile = loopir.Axis("tile", 2)
    operand = loopir.Operand("A", (tile,))
    copies = tuple(
        loopir.Movement(
            loopir.TENSOR_LOAD,
            (_transfer(
                operand[tile],
                loopir.GLOBAL,
                loopir.SHARED,
                where=tile == 0,
            ),),
            loopir.CompletionClass(f"copy{region}", loopir.WAVE),
        )
        for region in range(2)
    )
    program = loopir.Program((), (tile,), (operand,), copies)
    assert any(
        "not a disjoint cover" in error
        for error in validate_loopir(program)
    )


def test_wave_sensitive_effect_has_sixteen_points_but_local_tile_has_eight():
    wave = loopir.Axis("wave", 2)
    tile = loopir.Axis("tile", 8)
    a = loopir.Operand("A", (tile,))
    movement = loopir.Movement(
        loopir.TENSOR_LOAD,
        (_transfer(
            a[tile],
            loopir.GLOBAL,
            loopir.SHARED,
            effect=a[wave, tile],
        ),),
        loopir.CompletionClass("copy", loopir.WAVE),
    )
    program = loopir.Program(
        (), (tile,), (a,), (movement,), ((wave, loopir.WAVE),)
    )
    write = next(
        effect for effect in loopir.effects(program)
        if effect.kind is loopir.WRITE
    )
    assert len(loopir.effect_points(write)) == 16
    assert len({
        point.tile[-1] for point in loopir.effect_points(write)
    }) == 8
    assert len(movement.transfers[0].local_tile.indices) == 1
    assert len(movement.effect_for(0).tile.indices) == 2


def test_ds_load_reads_effect_tile_and_writes_wave_local_tile():
    wave = loopir.Axis("wave", 2)
    tile = loopir.Axis("tile", 8)
    a = loopir.Operand("A", (tile,))
    read = loopir.Movement(
        loopir.DS_LOAD,
        (_transfer(
            a[tile],
            loopir.SHARED,
            loopir.REGISTER,
            effect=a[wave, tile],
        ),),
        loopir.CompletionClass("read", loopir.WAVE),
    )
    program = loopir.Program(
        (), (tile,), (a,), (read,), ((wave, loopir.WAVE),)
    )
    source, destination = loopir.effects(program)
    assert source.kind is loopir.READ
    assert destination.kind is loopir.WRITE
    assert len(loopir.effect_points(source)) == 16
    assert len(loopir.effect_points(destination)) == 8
    assert loopir.effect_domain(source).agent_axes == ((wave, loopir.WAVE),)
    assert loopir.effect_domain(destination).agent_axes == ()


def test_declared_wave_is_not_implicitly_added_to_wave_invariant_effect():
    wave = loopir.Axis("wave", 4)
    tile = loopir.Axis("tile", 8)
    a = loopir.Operand("A", (tile,))
    movement = loopir.Movement(
        loopir.TENSOR_LOAD,
        (_transfer(a[tile], loopir.GLOBAL, loopir.SHARED),),
        loopir.CompletionClass("copy", loopir.WAVE),
    )
    program = loopir.Program(
        (), (tile,), (a,), (movement,), ((wave, loopir.WAVE),)
    )
    write = next(
        effect for effect in loopir.effects(program)
        if effect.kind is loopir.WRITE
    )
    assert loopir.effect_domain(write).agent_axes == ()
    assert len(loopir.effect_points(write)) == 8


def test_g3_finite_wave_tile_intersection_remains_exact_with_runtime_loop():
    trip = loopir.Param("T")
    iteration = loopir.Axis("iter", trip)
    wave = loopir.Axis("wave", 2)
    tile = loopir.Axis("tile", 3)
    a = loopir.Operand("A", (iteration, tile))
    copy = loopir.CompletionClass("copy", loopir.WAVE)
    read = loopir.CompletionClass("read", loopir.WAVE)
    loads = tuple(
        loopir.Movement(
            loopir.TENSOR_LOAD,
            (_transfer(
                a[iteration, tile],
                loopir.GLOBAL,
                loopir.SHARED,
                (2 * tile + wave < 3) if region == 0
                else (2 * tile + wave >= 3),
                effect=a[wave, iteration, tile],
            ),),
            copy,
        )
        for region in range(2)
    )
    reads = tuple(
        loopir.Movement(
            loopir.DS_LOAD,
            (_transfer(
                a[iteration, index],
                loopir.SHARED,
                loopir.REGISTER,
                effect=a[wave, iteration, index],
            ),),
            read,
        )
        for index in range(3)
    )
    program = loopir.Program(
        (trip,),
        (iteration, tile),
        (a,),
        (
            loopir.Region(loopir.STEADY),
            loopir.Loop(iteration, 0, trip, (*loads, *reads)),
        ),
        ((wave, loopir.WAVE),),
    )
    shared_raw = tuple(
        dependence for dependence in loopir.dependences(program)
        if dependence.kind is loopir.RAW
        and dependence.complete.level is loopir.SHARED
        and dependence.issue.level is loopir.SHARED
    )
    assert {
        (dependence.complete.operation, dependence.issue.operation)
        for dependence in shared_raw
    } == {
        (loads[0], reads[0]),
        (loads[0], reads[1]),
        (loads[1], reads[1]),
        (loads[1], reads[2]),
    }


def test_free_axis_is_a_bulk_local_effect_domain_not_an_issue_loop():
    program, _wave, load0, _load1, _reads = _g3_program()
    write = next(
        effect for effect in loopir.effects(program)
        if effect.operation is load0
        and effect.kind is loopir.WRITE
        and effect.level is loopir.SHARED
    )
    domain = loopir.effect_domain(write)
    assert tuple(axis.name for axis in domain.local_axes) == ("tile",)
    assert len(loopir.effect_points(write)) == 3


def test_dependence_is_minimal_and_kind_is_derived():
    assert tuple(field.name for field in fields(loopir.Effect)) == (
        "tile",
        "where",
    )
    assert tuple(field.name for field in fields(loopir.Dependence)) == (
        "complete",
        "issue",
        "domain",
    )
    program, _wave, load0, _load1, reads = _g3_program()
    assert all(
        isinstance(effect, loopir.EffectInstance)
        for effect in loopir.effects(program)
    )
    dependence = next(
        value for value in loopir.dependences(program)
        if value.complete.operation is load0
        and value.issue.operation is reads[0]
        and value.complete.level is loopir.SHARED
        and value.issue.level is loopir.SHARED
    )
    assert dependence.kind is loopir.RAW


def test_completion_wait_is_coalesced_across_ordered_serial_switch_cases():
    index = loopir.Axis("index", 2)
    tile = loopir.Axis("tile", 1)
    a = loopir.Operand("A", (tile,))
    copy_completion = loopir.CompletionClass("copy", loopir.WAVE)
    read_completion = loopir.CompletionClass("read", loopir.WAVE)
    copy = loopir.Movement(
        loopir.TENSOR_LOAD,
        (_transfer(a[0], loopir.GLOBAL, loopir.SHARED),),
        copy_completion,
    )
    reads = tuple(
        loopir.Movement(
            loopir.DS_LOAD,
            (_transfer(a[0], loopir.SHARED, loopir.REGISTER),),
            read_completion,
        )
        for _case in range(2)
    )
    program = loopir.Program(
        (),
        (index, tile),
        (a,),
        (
            loopir.Region(loopir.STEADY),
            copy,
            loopir.Loop(
                index,
                0,
                index.extent,
                (loopir.Switch(
                    index,
                    ((0, (reads[0],)), (1, (reads[1],))),
                ),),
            ),
        ),
    )

    required = tuple(
        dependence
        for dependence in loopir.required_dependences(program)
        if dependence.kind is loopir.RAW
        and dependence.complete.operation is copy
    )
    assert len(required) == 1
    assert required[0].issue.operation is reads[0]


def test_completion_wait_is_not_coalesced_across_conditional_arms():
    selector = loopir.Param("selector")
    index = loopir.Axis("index", 1)
    a = loopir.Operand("A", (index,))
    copy_completion = loopir.CompletionClass("copy", loopir.WAVE)
    read_completion = loopir.CompletionClass("read", loopir.WAVE)
    copy = loopir.Movement(
        loopir.TENSOR_LOAD,
        (_transfer(a[index], loopir.GLOBAL, loopir.SHARED),),
        copy_completion,
    )
    reads = tuple(
        loopir.Movement(
            loopir.DS_LOAD,
            (_transfer(a[index], loopir.SHARED, loopir.REGISTER),),
            read_completion,
        )
        for _arm in range(2)
    )
    program = loopir.Program(
        (selector,),
        (index,),
        (a,),
        (
            loopir.Region(loopir.STEADY),
            copy,
            loopir.Cond(selector == 0, (reads[0],), (reads[1],)),
        ),
    )

    required = tuple(
        dependence
        for dependence in loopir.required_dependences(program)
        if dependence.kind is loopir.RAW
        and dependence.complete.operation is copy
    )
    assert {dependence.issue.operation for dependence in required} == set(reads)


def test_program_bodies_hold_loopir_nodes_and_region_is_only_a_marker():
    program, _wave, _load0, _load1, _reads = _g3_program()
    assert all(isinstance(node, loopir.LoopIR) for node in program.body)
    marker = program.body[0]
    assert isinstance(marker, loopir.Region)
    assert tuple(field.name for field in fields(marker)) == ("tag",)


def test_movement_transfer_has_explicit_effect_and_local_tiles():
    _program, _wave, load0, _load1, _reads = _g3_program()
    transfer = load0.transfers[0]
    assert loopir.expr_equivalent(
        transfer.local_tile.indices[0],
        loopir.as_expr(_program.axes[0]),
    )
    assert load0.effect_for(0).tile.indices[0].free_axes() == frozenset((_wave,))
    assert not hasattr(transfer, "source_tile")
    assert not hasattr(transfer, "destination_tile")


def test_operation_api_has_checked_ds_load_and_implicit_serial_loops():
    tile = loopir.Axis("tile", 2)
    a = loopir.Operand("A", (tile,))
    b = loopir.Operand("B", (tile,))
    completion = loopir.CompletionClass("read", loopir.WAVE)
    assert issubclass(loopir.DS_LOAD, loopir.MovementOp)
    assert issubclass(loopir.WMMA, loopir.ComputeOp)

    with pytest.raises(loopir.IRTypeError, match="SHARED->REGISTER"):
        loopir.Movement(
            loopir.DS_LOAD,
            (_transfer(a[tile], loopir.GLOBAL, loopir.SHARED),),
            completion,
        )
    independent = loopir.Movement(
        loopir.TENSOR_LOAD,
        (
            _transfer(a[tile], loopir.GLOBAL, loopir.SHARED),
            _transfer(
                b[tile],
                loopir.GLOBAL,
                loopir.SHARED,
                tile == 0,
            ),
        ),
        completion,
    )
    assert len(independent.transfers) == 2

    common_effect = loopir.Effect(a[tile], tile >= 0)
    common = loopir.Movement(
        loopir.DS_LOAD,
        (loopir.Transfer(a[tile], loopir.SHARED, loopir.REGISTER),),
        completion,
        effect=common_effect,
    )
    assert common.effect_for(0) is common_effect
    assert common.tile_for_endpoint(0, source=True) is common_effect.tile
    assert loopir.expr_equivalent(
        common.tile_for_endpoint(0, source=False).indices[0],
        a[tile].indices[0],
    )

    with pytest.raises(loopir.IRTypeError, match="needs an Effect"):
        loopir.Movement(
            loopir.DS_LOAD,
            (loopir.Transfer(a[tile], loopir.SHARED, loopir.REGISTER),),
            completion,
        )
    with pytest.raises(loopir.IRTypeError, match="not both"):
        loopir.Movement(
            loopir.DS_LOAD,
            (_transfer(a[tile], loopir.SHARED, loopir.REGISTER),),
            completion,
            effect=loopir.Effect(a[tile]),
        )

    read = loopir.Movement(
        loopir.DS_LOAD,
        (_transfer(a[tile], loopir.SHARED, loopir.REGISTER),),
        completion,
    )
    loop = loopir.Loop(tile, (read,))
    assert not hasattr(loop, "binding")
    for removed in ("LOCAL_READ", "MATRIX", "ELEMENTWISE", "SERIAL"):
        assert not hasattr(loopir, removed)

    wave = loopir.Axis("wave", 2)
    with pytest.raises(loopir.IRTypeError, match="cannot be bound by Loop"):
        loopir.Program(
            (),
            (tile,),
            (a,),
            (loopir.Loop(wave, (read,)),),
            ((wave, loopir.WAVE),),
        )


def test_rewrite_is_a_pure_program_to_program_function():
    program, _wave, load0, _load1, _reads = _g3_program()
    rewritten = loopir.rewrite(
        program,
        lambda node: () if node is load0 else node,
    )
    assert rewritten is not program
    assert load0 in tuple(loopir.walk(program))
    assert load0 not in tuple(loopir.walk(rewritten))


def test_tensorload_partition_is_a_pure_generic_transformation():
    wave = loopir.Axis("wave", 2)
    tile = loopir.Axis("tile", 3)
    a = loopir.Operand("A", (tile,))
    copy = loopir.CompletionClass("copy", loopir.WAVE)
    load = loopir.Movement(
        loopir.TENSOR_LOAD,
        (_transfer(
            a[tile],
            loopir.GLOBAL,
            loopir.SHARED,
            effect=a[wave, tile],
        ),),
        copy,
    )
    program = loopir.Program(
        (),
        (tile,),
        (a,),
        (load,),
        ((wave, loopir.WAVE),),
    )
    partitioned = loopir.partition_movement(
        program,
        load,
        (2 * tile + wave < 3, 2 * tile + wave >= 3),
        (copy, copy),
    )
    parts = tuple(
        node for node in loopir.walk(partitioned)
        if isinstance(node, loopir.Movement)
    )
    assert len(parts) == 2
    assert tuple(loopir.walk(program))[-1] is load
    points = []
    for part in parts:
        write = next(
            effect for effect in loopir.effects(partitioned)
            if effect.operation is part
            and effect.kind is loopir.WRITE
            and effect.level is loopir.SHARED
        )
        points.append({
            (tuple(value for _axis, value in point.agents), point.tile)
            for point in loopir.effect_points(write)
        })
    assert not points[0] & points[1]
    assert len(points[0] | points[1]) == 6


def test_fuse_combines_transfers_but_not_domains():
    tile_a = loopir.Axis("tile_a", 2)
    tile_b = loopir.Axis("tile_b", 1)
    a = loopir.Operand("A", (tile_a,))
    b = loopir.Operand("B", (tile_b,))
    copy = loopir.CompletionClass("copy", loopir.WAVE)
    load_a = loopir.Movement(
        loopir.TENSOR_LOAD,
        (_transfer(a[tile_a], loopir.GLOBAL, loopir.SHARED),),
        copy,
    )
    load_b = loopir.Movement(
        loopir.TENSOR_LOAD,
        (_transfer(b[tile_b], loopir.GLOBAL, loopir.SHARED),),
        copy,
    )
    program = loopir.Program((), (tile_a, tile_b), (a, b), (load_a, load_b))
    fused = loopir.fuse_movements(program, (load_a, load_b), copy)
    operation = tuple(loopir.walk(fused))[0]
    assert isinstance(operation, loopir.Movement)
    assert tuple(
        transfer.local_tile.operand.name for transfer in operation.transfers
    ) == (
        "A",
        "B",
    )
    write_domains = tuple(
        tuple(axis.name for axis in loopir.effect_domain(effect).local_axes)
        for effect in loopir.effects(fused)
        if effect.kind is loopir.WRITE
    )
    assert write_domains == (("tile_a",), ("tile_b",))


def test_pipeline_is_a_pure_composition_of_ordinary_nodes_and_markers():
    trip = loopir.Param("T")
    iteration = loopir.Axis("iter", trip)
    tile = loopir.Axis("tile", 1)
    a = loopir.Operand("A", (iteration, tile))
    copy = loopir.CompletionClass("copy", loopir.WAVE)
    load = loopir.Movement(
        loopir.TENSOR_LOAD,
        (_transfer(a[iteration, tile], loopir.GLOBAL, loopir.SHARED),),
        copy,
    )
    read = loopir.Movement(
        loopir.DS_LOAD,
        (_transfer(a[iteration, tile], loopir.SHARED, loopir.REGISTER),),
        loopir.IMMEDIATE,
    )
    main = loopir.Loop(iteration, 0, trip, (load, read))
    program = loopir.Program((trip,), (iteration, tile), (a,), (main,))
    pipelined = loopir.pipeline(program, main, ((load, 1),))

    assert program.body == (main,)
    guard = pipelined.body[0]
    assert isinstance(guard, loopir.Cond)
    assert tuple(
        node.tag for node in guard.then_body if isinstance(node, loopir.Region)
    ) == (loopir.PROLOGUE, loopir.STEADY, loopir.DRAIN)
    assert isinstance(guard.else_body[0], loopir.Region)
    assert guard.else_body[0].tag is loopir.SHORT
    assert not hasattr(loopir, "Peel")


def test_removed_first_attempt_values_are_not_public():
    for name in (
        "Placement",
        "Ref",
        "AgentPoint",
        "AgentSet",
        "Wait",
        "Sync",
        "TileFootprint",
        "TileAccess",
        "EffectDomain",
        "EffectPoint",
        "ScheduleIR",
    ):
        assert not hasattr(loopir, name)


def test_lowering_directly_converts_order_without_schedule_policy():
    from Tensile.Lowering.loopir_to_gir import lower_to_gir
    from Tensile.Lowering.gir.nodes import Move

    program, _wave, load0, load1, reads = _g3_program()
    gir = lower_to_gir(program)
    operations = [
        operation
        for block in gir.blocks.values()
        for operation in block.body
    ]
    assert all(isinstance(operation, Move) for operation in operations)
    assert tuple(operation.unit for operation in operations) == (
        ("A",),
        ("A",),
        (),
        (),
        (),
    )

    source = inspect.getsource(__import__(
        "Tensile.Lowering.loopir_to_gir",
        fromlist=("lower_to_gir",),
    ))
    assert "LoopModel.traversal" not in source
    assert "LoopModel.schedule" not in source
    assert "TDMSplit" not in source
    assert "TDMFuse" not in source


def test_stored_effect_change_removes_derived_edge_and_lowered_dependency():
    from Tensile.Lowering.loopir_to_gir import lower_to_gir
    from Tensile.Lowering.gir.nodes import Move

    tile = loopir.Axis("tile", 2)
    a = loopir.Operand("A", (tile,))
    copy = loopir.Movement(
        loopir.TENSOR_LOAD,
        (_transfer(a[0], loopir.GLOBAL, loopir.SHARED),),
        loopir.CompletionClass("copy", loopir.WAVE),
    )
    read = loopir.Movement(
        loopir.DS_LOAD,
        (_transfer(a[0], loopir.SHARED, loopir.REGISTER),),
        loopir.CompletionClass("read", loopir.WAVE),
    )
    program = loopir.Program((), (tile,), (a,), (copy, read))
    assert any(
        edge.kind is loopir.RAW
        and edge.complete.operation is copy
        and edge.issue.operation is read
        for edge in loopir.dependences(program)
    )
    lowered = lower_to_gir(program)
    moves = tuple(
        operation
        for block in lowered.blocks.values()
        for operation in block.body
        if isinstance(operation, Move)
    )
    assert moves[1].deps

    unrelated = loopir.Movement(
        loopir.DS_LOAD,
        (_transfer(
            a[0],
            loopir.SHARED,
            loopir.REGISTER,
            effect=a[1],
        ),),
        loopir.CompletionClass("read", loopir.WAVE),
    )
    changed = loopir.Program((), (tile,), (a,), (copy, unrelated))
    assert not any(
        edge.kind is loopir.RAW
        and edge.complete.operation is copy
        and edge.issue.operation is unrelated
        for edge in loopir.dependences(changed)
    )
    changed_moves = tuple(
        operation
        for block in lower_to_gir(changed).blocks.values()
        for operation in block.body
        if isinstance(operation, Move)
    )
    assert not changed_moves[1].deps


def test_theta_pipeline_builds_valid_loopir_gir():
    from Tensile.LoopModel import adapter
    from Tensile.LoopModel.checks import validate_loopir
    from Tensile.LoopModel.decoder import decode_theta
    from Tensile.Lowering.loopir_to_gir import lower_to_gir
    from Tensile.Lowering.gir import check_plan
    from Tensile.Lowering.gir.passes import pipeline as gir_pipeline
    from Tensile.Lowering.gir.passes import run_pipeline

    params = {
        "MIWaveTile": [2, 2],
        "MatrixInstruction": [16, 16, 128, 1, 1, 2, 2, 2, 2],
        "DepthU": 128,
        "ElemBytes": 1,
        "PrefetchGlobalRead": 1,
        "PrefetchLocalRead": 0,
        "LoopOrder": "KMN",
    }
    theta = adapter.params_to_theta(params)
    decoded = decode_theta(theta)
    assert not validate_loopir(decoded.program, decoded.profile)
    actual = lower_to_gir(decoded.program, decoded.profile)
    run_pipeline(actual, passes=gir_pipeline())
    assert not check_plan(actual)


def test_wave_is_a_program_agent_not_a_loop_or_program_axis():
    from Tensile.LoopModel import adapter
    from Tensile.LoopModel.decoder import decode_theta

    decoded = decode_theta(adapter.params_to_theta({
        "MIWaveTile": [2, 2],
        "MatrixInstruction": [16, 16, 128, 1, 1, 2, 2, 2, 2],
        "DepthU": 128,
        "ElemBytes": 1,
        "PrefetchGlobalRead": 1,
        "PrefetchLocalRead": 0,
        "LoopOrder": "KMN",
        "NumWaves": 4,
    }))
    assert tuple(
        (axis.name, scope)
        for axis, scope in decoded.program.agents
    ) == (("wave", loopir.WAVE),)
    assert all(axis.name != "wave" for axis in decoded.program.axes)
    assert all(
        node.axis.name != "wave"
        for node in loopir.walk(decoded.program)
        if isinstance(node, loopir.Loop)
    )


def test_runtime_iter_is_projected_to_generation_phase_not_discarded():
    from Tensile.LoopModel import adapter
    from Tensile.LoopModel.decoder import decode_theta
    from Tensile.Lowering.loopir_to_gir import lower_to_gir
    from Tensile.Lowering.gir.nodes import Move

    params = {
        "MIWaveTile": [2, 2],
        "MatrixInstruction": [16, 16, 128, 1, 1, 2, 2, 2, 2],
        "DepthU": 128,
        "ElemBytes": 1,
        "PrefetchGlobalRead": 1,
        "PrefetchLocalRead": 0,
        "LoopOrder": "KMN",
    }
    decoded = decode_theta(adapter.params_to_theta(params))
    gir = lower_to_gir(decoded.program, decoded.profile)
    steady_copies = [
        operation
        for operation in gir.blocks["steady"].body
        if isinstance(operation, Move)
        and operation.dsts
        and operation.dsts[0].tile.space == "shared"
    ]
    assert steady_copies
    assert all(operation.dsts[0].gdelta == 1 for operation in steady_copies)
    assert all(
        axis != "iter"
        for operation in steady_copies
        for axis, _value in operation.dsts[0].tile.coord
    )


def test_post_finalize_read_retiming_requires_profile_refinalization():
    from Tensile.LoopModel import adapter
    from Tensile.LoopModel.decoder import decode_theta
    from Tensile.Lowering.loopir_to_gir import lower_to_gir

    decoded = decode_theta(adapter.params_to_theta({
        "MIWaveTile": [2, 2],
        "MatrixInstruction": [16, 16, 128, 1, 1, 2, 2, 2, 2],
        "DepthU": 128,
        "ElemBytes": 1,
        "PrefetchGlobalRead": 1,
        "PrefetchLocalRead": 1,
        "LoopOrder": "KMN",
    }))
    read = next(
        operation for operation in loopir.operations(decoded.program)
        if isinstance(operation, loopir.Movement)
        and operation.op is loopir.DS_LOAD
    )
    retimed = loopir.rewrite_operation(
        decoded.program,
        read,
        lambda operation: loopir.substitute_operation(
            operation,
            {
                axis: 0
                for axis in operation.transfers[0].local_tile.operand.axes
            },
        ),
    )
    with pytest.raises(RuntimeError, match="re-finalize"):
        lower_to_gir(retimed, decoded.profile)


@pytest.mark.parametrize("order", ("KMN", "KNM", "MKN", "MNK", "NKM", "NMK"))
def test_all_basic_loop_orders_lower_to_valid_gir(order):
    from Tensile.LoopModel import adapter

    params = {
        "MIWaveTile": [2, 2],
        "MatrixInstruction": [16, 16, 128, 1, 1, 2, 2, 2, 2],
        "DepthU": 128,
        "ElemBytes": 1,
        "PrefetchGlobalRead": 1,
        "PrefetchLocalRead": 0,
        "LoopOrder": order,
    }
    theta = adapter.params_to_theta(params)
    _decoded, actual = _lower_theta(theta)
    assert actual.blocks


def test_tdmsplit_tdmfuse_multiwave_program_lowers_to_valid_gir():
    from Tensile.LoopModel import adapter

    params = {
        "MIWaveTile": [2, 2],
        "MatrixInstruction": [16, 16, 128, 1, 1, 2, 2, 2, 2],
        "DepthU": 128,
        "ElemBytes": 1,
        "PrefetchGlobalRead": 1,
        "PrefetchLocalRead": 0,
        "LoopOrder": "KMN",
        "TDMSplit": [2, 2, 1, 1],
        "TDMSplitWaveRegions": [1, 1],
        "NumWaves": 4,
        "TDMFuse": 0,
    }
    theta = adapter.params_to_theta(params)
    _decoded, actual = _lower_theta(theta)
    assert actual.blocks


def test_decoder_wires_odd_read_groups_into_tensorload_effects():
    from Tensile.LoopModel import adapter
    from Tensile.LoopModel.decoder import _Decoder

    params = {
        "MIWaveTile": [6, 2],
        "MatrixInstruction": [16, 16, 128, 1, 1, 6, 2, 2, 1],
        "DepthU": 128,
        "ElemBytes": 1,
        "PrefetchGlobalRead": 1,
        "PrefetchLocalRead": 0,
        "LoopOrder": "KMN",
        "TDMSplit": [2, 1, 1, 1],
        "TDMSplitWaveRegions": [2, 1],
        "NumWaves": 2,
        "VectorWidthA": 2,
        "VectorWidthB": 2,
    }
    decoder = _Decoder(adapter.params_to_theta(params))
    program = loopir.Program(
        tuple(decoder.params),
        tuple(decoder.axes),
        tuple(decoder.operands.values()),
        decoder._copy_nodes(),
        decoder.agents,
    )
    writes = tuple(
        effect
        for effect in loopir.effects(program)
        if effect.kind is loopir.WRITE
        and effect.level is loopir.SHARED
        and effect.tile.operand.name == "A"
    )
    assert len(writes) == 2
    points = [
        {
            point.tile
            for point in loopir.effect_points(
                effect,
                {program.params[0]: 1},
            )
        }
        for effect in writes
    ]
    assert all(len(region) == 6 for region in points)
    assert points[0].isdisjoint(points[1])
    assert (0, 0, 0, 2) in points[0]
    assert (1, 0, 0, 2) in points[1]
    assert (0, 0, 1, 0) in points[0]
    assert (1, 0, 1, 0) in points[1]


def test_whole_vector_single_wave_keeps_every_tdmsplit_region_live():
    from Tensile.LoopModel import adapter
    from Tensile.LoopModel.checks import validate_loopir
    from Tensile.LoopModel.decoder import _Decoder

    params = {
        "MIWaveTile": [4, 4],
        "MatrixInstruction": [16, 16, 128, 1, 1, 4, 4, 1, 1],
        "DepthU": 256,
        "ElemBytes": 1,
        "PrefetchGlobalRead": 1,
        "PrefetchLocalRead": 0,
        "LoopOrder": "KMN",
        "TDMSplit": [2, 1, 2, 1],
        "NumWaves": 1,
        "VectorWidthA": 4,
        "VectorWidthB": 4,
    }
    decoder = _Decoder(adapter.params_to_theta(params))
    program = loopir.Program(
        tuple(decoder.params),
        tuple(decoder.axes),
        tuple(decoder.operands.values()),
        decoder._copy_nodes(),
        decoder.agents,
    )
    assert not validate_loopir(program)
    writes = tuple(
        effect
        for effect in loopir.effects(program)
        if effect.kind is loopir.WRITE
        and effect.level is loopir.SHARED
        and effect.tile.operand.name == "A"
    )
    assert len(writes) == 4
    assert all(
        loopir.effect_points(effect, {program.params[0]: 1})
        for effect in writes
    )
    finalized = _Decoder(adapter.params_to_theta(params)).build().program
    copies = {
        operation._identity
        for operation in loopir.operations(finalized)
        if isinstance(operation, loopir.Movement)
        and operation.op is loopir.TENSOR_LOAD
        and any(
            transfer.local_tile.operand.name == "A"
            for transfer in operation.transfers
        )
    }
    served = {
        dependence.complete.operation._identity
        for dependence in loopir.required_dependences(finalized)
        if dependence.kind is loopir.RAW
        and dependence.complete.tile.operand.name == "A"
        and isinstance(dependence.complete.operation, loopir.Movement)
        and dependence.complete.operation.op is loopir.TENSOR_LOAD
    }
    assert copies <= served


@pytest.mark.parametrize(
    "width,expected",
    (
        (8, {(0, 0): 0, (1, 0): 1}),
        (4, {(0, 0): 0, (1, 0): 0, (0, 1): 1, (1, 1): 1}),
    ),
)
def test_vector_width_changes_the_explicit_tdmsplit_effect(width, expected):
    from Tensile.LoopModel import adapter
    from Tensile.LoopModel.decoder import _Decoder

    decoder = _Decoder(adapter.params_to_theta({
        "MIWaveTile": [8, 2],
        "MatrixInstruction": [16, 16, 128, 1, 1, 8, 2, 2, 1],
        "DepthU": 128,
        "ElemBytes": 1,
        "PrefetchGlobalRead": 1,
        "PrefetchLocalRead": 0,
        "LoopOrder": "KMN",
        "TDMSplit": [2, 1, 1, 1],
        "NumWaves": 2,
        "VectorWidthA": width,
        "VectorWidthB": 2,
    }))
    program = loopir.Program(
        tuple(decoder.params),
        tuple(decoder.axes),
        tuple(decoder.operands.values()),
        decoder._copy_nodes(),
        decoder.agents,
    )
    copies = tuple(
        operation for operation in loopir.operations(program)
        if isinstance(operation, loopir.Movement)
        and operation.op is loopir.TENSOR_LOAD
        and operation.transfers[0].local_tile.operand.name == "A"
    )
    expression = copies[0].effect_for(0).where.lhs
    axes = {axis.name: axis for axis in (*program.axes, *(
        axis for axis, _scope in program.agents
    ))}
    for (wave, split), region in expected.items():
        assert expression.evaluate({
            axes["wave"]: wave,
            axes["iter"]: 0,
            axes["K_inner"]: 0,
            axes["M_split"]: split,
            axes["M_inner"]: 0,
        }) == region


def test_cached_loopir_text_keeps_target_prefetch_and_vgpr_header():
    from types import SimpleNamespace

    from Tensile.Components.LoopModel.Program import loopModelIrTextCached
    from Tensile.LoopModel import adapter
    from Tensile.LoopModel.decoder import decode_theta
    from Tensile.LoopModel.schedule import build_S

    theta = adapter.params_to_theta({
        "MIWaveTile": [2, 2],
        "MatrixInstruction": [16, 16, 128, 1, 1, 2, 2, 1, 1],
        "DepthU": 128,
        "ElemBytes": 1,
        "PrefetchGlobalRead": 1,
        "PrefetchLocalRead": 1,
        "LoopOrder": "KMN",
    })
    depths, _minimum = build_S(theta)
    decoded = decode_theta(theta)
    writer = SimpleNamespace(
        states=SimpleNamespace(kernelName="diagnostic"),
        _loopModelThetaCache=("diagnostic", (theta, depths)),
        _loopModelMainloopCache=("diagnostic", decoded.program),
    )
    text = loopModelIrTextCached(writer)
    assert "# LoopModel geometry" in text
    assert "operand role" in text
    assert "register_depths:" in text
    assert "vgpr read_mode" in text
    assert "derived_dependences:" in text
    assert "\nprogram\n" in text


def test_two_chunk_global_prefetch_lowers_to_valid_gir():
    from Tensile.Tests.unit.gir_fixtures import BF16_NT_KMN_PLR0, theta

    point = theta(dict(BF16_NT_KMN_PLR0))
    _decoded, actual = _lower_theta(point)
    assert actual.blocks


def test_nested_local_and_global_prefetch_lower_to_valid_gir():
    from Tensile.Tests.unit.gir_fixtures import BF16_NT_KMN, theta

    point = theta(dict(BF16_NT_KMN))
    _decoded, actual = _lower_theta(point)
    assert actual.blocks


@pytest.mark.parametrize("split,waves", ((False, 1), (True, 4)))
@pytest.mark.parametrize("order", ("KMN", "KNM", "MKN", "MNK", "NKM", "NMK"))
@pytest.mark.parametrize("pgr", (1, 2))
@pytest.mark.parametrize("plr", (0, 1))
def test_prefetch_split_and_loop_order_gir_matrix(
    split, waves, order, pgr, plr
):
    from Tensile.LoopModel import adapter

    params = {
        "MIWaveTile": [2, 2],
        "MatrixInstruction": [16, 16, 128, 1, 1, 2, 2, 2, 2],
        "DepthU": 128,
        "ElemBytes": 1,
        "LoopOrder": order,
        "PrefetchGlobalRead": pgr,
        "PrefetchLocalRead": plr,
        "NumWaves": waves,
    }
    if split:
        params.update({
            "TDMSplit": [2, 2, 1, 1],
            "TDMSplitWaveRegions": [1, 1],
            "TDMFuse": 0,
        })
    point = adapter.params_to_theta(params)
    _decoded, actual = _lower_theta(point)
    assert actual.blocks


@pytest.mark.parametrize(
    "ring_overrides,expected_copies",
    (
        ({"VgprGroupA": 2, "VgprAllocA": 3}, 3),
        ({
            "VgprGroupB": 1,
            "VgprAllocB": 6,
            "PrefetchLocalReadA": 1,
            "PrefetchLocalReadB": 3,
        }, 6),
    ),
)
def test_physical_register_ring_projection_is_materialized_in_loopir(
    ring_overrides, expected_copies
):
    from Tensile.LoopModel import adapter, traversal
    from Tensile.LoopModel.decoder import decode_theta
    from Tensile.LoopModel.schedule import build_S, resolve_coverage
    from Tensile.Lowering.loopir_to_gir import lower_to_gir

    params = {
        "MIWaveTile": [2, 2],
        "MatrixInstruction": [16, 16, 128, 1, 1, 2, 2, 2, 2],
        "DepthU": 128,
        "ElemBytes": 1,
        "LoopOrder": "KMN",
        "PrefetchGlobalRead": 2,
        "PrefetchLocalRead": 2,
        "NumWaves": 1,
        **ring_overrides,
    }
    point = adapter.params_to_theta(params)
    depths, _minimum = build_S(point)
    resolve_coverage(point, depths)
    copies = traversal.ring_trip_copies(point, depths)
    assert copies == expected_copies

    decoded = decode_theta(point)
    copy_steps = tuple(
        node for node in loopir.walk(decoded.program)
        if isinstance(node, loopir.Let)
        and node.axis.name == "_steady_copy"
    )
    assert len(copy_steps) == expected_copies
    assert tuple(step.value.constant_value for step in copy_steps) == tuple(
        range(expected_copies)
    )
    assert "loop_copies" not in inspect.signature(lower_to_gir).parameters
    _decoded, actual = _lower_theta(point)
    assert actual.blocks


def test_repeat_read_occurrences_drive_the_three_copy_ring_projection():
    from Tensile.LoopModel import traversal
    from Tensile.LoopModel.adapter import params_to_theta
    from Tensile.LoopModel.decoder import decode_theta
    from Tensile.LoopModel.schedule import build_S, resolve_coverage
    from Tensile.Lowering.loopir_to_gir import lower_to_gir
    from Tensile.Lowering.gir.emit_plan import plan_program
    from Tensile.Lowering.gir.passes import pipeline as gir_pipeline
    from Tensile.Lowering.gir.passes import run_pipeline
    from Tensile.Lowering.gir.verify_dataflow import check_plan

    params = {
        "MatrixInstruction": [16, 16, 128, 1, 1, 2, 2, 1, 1],
        "MIWaveTile": [2, 2],
        "DepthU": 256,
        "ElemBytes": 1,
        "PrefetchGlobalRead": 1,
        "PrefetchLocalRead": 2,
        "LoopOrder": "KNM",
        "TDMSplit": (2, 1, 1, 1),
        "VgprGroupA": 2,
        "VgprAllocA": 3,
        "SuppressNoLoadLoop": True,
        "LocalReadVectorWidth": 16,
    }
    theta = params_to_theta(params)
    decoded = decode_theta(theta)
    a_reads = tuple(
        operation
        for operation in loopir.operations(decoded.program)
        if isinstance(operation, loopir.Movement)
        and operation.op is loopir.DS_LOAD
        and operation.transfers[0].local_tile.operand.name == "A"
    )
    assert a_reads
    assert all(not hasattr(operation, "issue") for operation in a_reads)

    depths, _minimum = build_S(theta)
    resolve_coverage(theta, depths)
    copies = traversal.ring_trip_copies(theta, depths)
    assert copies == 3
    gir = lower_to_gir(
        decoded.program,
        decoded.profile,
    )
    run_pipeline(gir, passes=gir_pipeline())
    assert not check_plan(gir)
    issues = {
        action.at["issue"]
        for actions in plan_program(gir).values()
        for action in actions
        if action.kind == "read" and action.at["tc"] == "A"
    }
    assert {(("N_inner", 0),), (("N_inner", 1),)} <= issues


def test_ds_load_coverage_is_a_local_transfer_set():
    from Tensile.Lowering.loopir_to_gir import lower_to_gir

    iteration = loopir.Param("T")
    tile = loopir.Axis("tile", 4)
    operand = loopir.Operand("A", (tile,))
    read = loopir.Movement(
        loopir.DS_LOAD,
        (
            _transfer(
                operand[tile - tile % 2],
                loopir.SHARED,
                loopir.REGISTER,
            ),
            _transfer(
                operand[tile - tile % 2 + 1],
                loopir.SHARED,
                loopir.REGISTER,
            ),
        ),
        loopir.CompletionClass("read", loopir.WAVE),
    )
    program = loopir.Program(
        (iteration,),
        (tile,),
        (operand,),
        (
            loopir.Region(loopir.STEADY),
            loopir.Loop(
                tile,
                0,
                tile.extent,
                (loopir.Cond(tile % 2 == 0, (read,), ()),),
            ),
        ),
    )
    writes = tuple(
        effect for effect in loopir.effects(program)
        if effect.operation is read
        and effect.kind is loopir.WRITE
    )
    assert len(writes) == 2
    assert all(not loopir.effect_domain(write).local_axes for write in writes)
    assert {
        point.tile
        for write in writes
        for point in loopir.effect_points(write)
    } == {(0,), (1,), (2,), (3,)}
    lowered = lower_to_gir(program)
    assert lowered.meta.get("read_fold", {}) == {}
    assert lowered.meta.get("coverage_axes", {}) == {}


def test_mx_read_order_is_explicit_in_loopir_and_lowers_to_valid_gir():
    from Tensile.LoopModel import adapter
    from Tensile.LoopModel.decoder import decode_theta

    params = {
        "MIWaveTile": [2, 2],
        "MatrixInstruction": [16, 16, 128, 1, 1, 2, 2, 2, 2],
        "DepthU": 256,
        "ElemBytes": 1,
        "PrefetchGlobalRead": 2,
        "PrefetchLocalRead": 1,
        "LoopOrder": "MKN",
        "NumWaves": 4,
        "MXBlockA": 32,
        "MXBlockB": 32,
        "TDMFuse": 0,
    }
    theta = adapter.params_to_theta(params)
    decoded = decode_theta(theta)
    compute = next(
        operation for operation in loopir.operations(decoded.program)
        if isinstance(operation, loopir.Compute)
    )
    assert tuple(
        tile.operand.name for tile in compute.reads
    ) == ("A", "B", "MXSA", "MXSB")

    _decoded, actual = _lower_theta(theta)
    assert actual.blocks


def test_mixed_tdmsplit_fused_copy_covers_the_peer_axis_logically():
    from Tensile.LoopModel import adapter
    from Tensile.LoopModel.decoder import decode_theta

    params = {
        "MIWaveTile": [2, 2],
        "MatrixInstruction": [16, 16, 128, 1, 1, 2, 2, 2, 2],
        "DepthU": 256,
        "ElemBytes": 1,
        "PrefetchGlobalRead": 2,
        "PrefetchLocalRead": 1,
        "LoopOrder": "MKN",
        "NumWaves": 4,
        "MXBlockA": 32,
        "MXBlockB": 32,
        "TDMSplit": (1, 2, 2, 1),
        "TDMSplitWaveRegions": (1, 1),
        "TDMFuse": 0,
    }
    theta = adapter.params_to_theta(params)
    decoded = decode_theta(theta)
    fused = next(
        operation for operation in loopir.operations(decoded.program)
        if isinstance(operation, loopir.Movement)
        and operation.op is loopir.TENSOR_LOAD
        and tuple(dict.fromkeys(
            transfer.local_tile.operand.name
            for transfer in operation.transfers
        )) == ("A", "B")
    )
    b_transfers = tuple(
        transfer for transfer in fused.transfers
        if transfer.local_tile.operand.name == "B"
    )
    transfer_b = b_transfers[0]
    k_split = next(
        index for index, axis in enumerate(transfer_b.local_tile.operand.axes)
        if axis.name == "K_split"
    )
    assert {
        transfer.local_tile.indices[k_split].constant_value
        for transfer in b_transfers
    } == {0, 1}
    assert all("bulk" not in axis.name for axis in decoded.program.axes)

    _decoded, actual = _lower_theta(theta)
    assert actual.blocks


@pytest.mark.parametrize(
    "order,prefetch,split",
    (
        ("MKN", 1, None),
        ("KMN", 2, None),
        ("KMNMKN", 1, (2, 1, 1, 2)),
    ),
)
def test_target_coverage_inplace_read_order_lowers_to_valid_gir(
    order, prefetch, split
):
    from Tensile.LoopModel import adapter

    params = {
        "MIWaveTile": [2, 2],
        "MatrixInstruction": [16, 16, 128, 1, 1, 2, 2, 2, 2],
        "DepthU": 256,
        "ElemBytes": 1,
        "PrefetchGlobalRead": 2,
        "PrefetchLocalRead": prefetch,
        "LoopOrder": order,
        "NumWaves": 4,
        "MXBlockA": 32,
        "MXBlockB": 32,
        "TDMFuse": 0,
        "ReadVectorElems": {"MXSA": 8, "MXSB": 8},
        "ReadPhi": {"MXSA": 2, "MXSB": 2},
        "ReadRho": {"MXSA": 0, "MXSB": 0},
        "RegisterBudget": 992,
    }
    if split is not None:
        params.update({
            "TDMSplit": split,
            "TDMSplitWaveRegions": (1, 1),
        })
    theta = adapter.params_to_theta(params)
    _decoded, actual = _lower_theta(theta)
    assert actual.blocks


def test_fused_mixed_coverage_drain_order_lowers_to_valid_gir():
    from Tensile.LoopModel import adapter

    params = {
        "MIWaveTile": [2, 2],
        "MatrixInstruction": [16, 16, 128, 1, 1, 2, 2, 2, 2],
        "DepthU": 256,
        "ElemBytes": 1,
        "PrefetchGlobalRead": 2,
        "PrefetchLocalRead": 1,
        "LoopOrder": "MNKMNK",
        "NumWaves": 4,
        "MXBlockA": 32,
        "MXBlockB": 32,
        "TDMFuse": 2,
        "TDMSplit": (2, 2, 1, 1),
        "TDMSplitWaveRegions": (2, 2),
        "ReadVectorElems": {"MXSA": 4, "MXSB": 4},
        "ReadPhi": {"MXSA": 2, "MXSB": 2},
        "ReadRho": {"MXSA": 2, "MXSB": 2},
        "RegisterBudget": 992,
    }
    theta = adapter.params_to_theta(params)
    _decoded, actual = _lower_theta(theta)
    assert actual.blocks


def test_multi_axis_tensorload_regions_are_explicit_effect_partitions():
    from Tensile.LoopModel import adapter
    from Tensile.LoopModel.decoder import _Decoder

    params = {
        "MIWaveTile": [2, 2],
        "MatrixInstruction": [16, 16, 128, 1, 1, 2, 2, 2, 2],
        "DepthU": 256,
        "ElemBytes": 1,
        "PrefetchGlobalRead": 1,
        "PrefetchLocalRead": 0,
        "LoopOrder": "KMNMKN",
        "NumWaves": 4,
        "TDMSplit": (2, 2, 2, 2),
        "TDMSplitWaveRegions": (1, 1),
    }
    decoder = _Decoder(adapter.params_to_theta(params))
    body = decoder._copy_nodes()
    program = loopir.Program(
        tuple(decoder.params),
        tuple(decoder.axes),
        tuple(decoder.operands.values()),
        body,
        decoder.agents,
    )
    copies = tuple(
        operation for operation in loopir.operations(program)
        if isinstance(operation, loopir.Movement)
        and operation.op is loopir.TENSOR_LOAD
    )
    assert len(copies) == 4
    assert not any(
        isinstance(node, loopir.Switch)
        for node in loopir.walk(program)
    )
    for region, copy in enumerate(copies):
        assert {
            copy.effect_for(index).where.rhs.constant_value
            for index in range(len(copy.transfers))
        } == {region}


def test_repeat_read_axis_without_retiming_lowers_as_an_expression():
    from Tensile.LoopModel import adapter
    from Tensile.LoopModel.checks import validate_loopir
    from Tensile.LoopModel.decoder import decode_theta
    from Tensile.Lowering.loopir_to_gir import lower_to_gir

    decoded = decode_theta(adapter.params_to_theta({
        "MIWaveTile": [4, 4],
        "MatrixInstruction": [16, 16, 128, 1, 1, 4, 4, 2, 2],
        "DepthU": 256,
        "ElemBytes": 1,
        "PrefetchGlobalRead": 1,
        "PrefetchLocalRead": 0,
        "LoopOrder": "KMNMKN",
        "NumWaves": 4,
        "TDMSplit": (2, 2, 2, 2),
        "TDMSplitWaveRegions": (1, 1),
    }))
    assert not validate_loopir(decoded.program, decoded.profile)
    assert lower_to_gir(decoded.program, decoded.profile).blocks


def test_backward_same_iteration_relation_is_kept_conservatively():
    iteration = loopir.Axis("iteration", 2)
    operand = loopir.Operand("A", (iteration,))
    copy = loopir.Movement(
        loopir.TENSOR_LOAD,
        (_transfer(
            operand[iteration],
            loopir.GLOBAL,
            loopir.SHARED,
        ),),
        loopir.CompletionClass("copy", loopir.WAVE),
    )
    read = loopir.Movement(
        loopir.DS_LOAD,
        (_transfer(
            operand[iteration],
            loopir.SHARED,
            loopir.REGISTER,
        ),),
        loopir.CompletionClass("read", loopir.WAVE),
    )
    program = loopir.Program(
        (),
        (iteration,),
        (operand,),
        (loopir.Loop(
            iteration,
            0,
            iteration.extent,
            (read, copy),
        ),),
    )
    relation = next(
        dependence for dependence in loopir.dependences(program)
        if dependence.kind is loopir.RAW
        and dependence.complete.operation is copy
        and dependence.issue.operation is read
    )
    assert relation.domain.conservative


def test_no_load_loop_is_an_explicit_program_transform():
    from Tensile.LoopModel import adapter, traversal
    from Tensile.LoopModel.decoder import decode_theta
    from Tensile.LoopModel.schedule import build_S, resolve_coverage

    params = {
        "MIWaveTile": [2, 2],
        "MatrixInstruction": [16, 16, 128, 1, 1, 2, 2, 2, 2],
        "DepthU": 128,
        "ElemBytes": 1,
        "LoopOrder": "KMN",
        "PrefetchGlobalRead": 2,
        "PrefetchLocalRead": 2,
        "VgprGroupA": 2,
        "VgprAllocA": 3,
        "SuppressNoLoadLoop": True,
    }
    point = adapter.params_to_theta(params)
    depths, _minimum = build_S(point)
    resolve_coverage(point, depths)
    copies = traversal.ring_trip_copies(point, depths)
    decoded = decode_theta(point)
    assert loopir.DRAIN not in {
        region.tag for region in loopir.regions(decoded.program)
    }
    _decoded, actual = _lower_theta(point)
    assert actual.blocks

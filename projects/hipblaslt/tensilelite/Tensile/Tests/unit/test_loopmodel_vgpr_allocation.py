# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit

_KERNEL_WRITER_ASSEMBLY = Path(__file__).resolve().parents[2] / "KernelWriterAssembly.py"
_LOOPMODEL_REGISTERS = (Path(__file__).resolve().parents[2]
                        / "Components" / "LoopModel" / "Registers.py")


def _function(path, name, namespace=None):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    method = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    function = ast.FunctionDef(
        name=method.name,
        args=method.args,
        body=method.body,
        decorator_list=[],
        returns=method.returns,
        type_comment=method.type_comment,
    )
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = dict(namespace or {})
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[method.name]


def _apply_method():
    return _function(_LOOPMODEL_REGISTERS, "applyLoopModelValuRegs")


def _writer(counts):
    return SimpleNamespace(
        states=SimpleNamespace(
            a=SimpleNamespace(numVgprValu=11),
            b=SimpleNamespace(numVgprValu=12),
            mxsa=SimpleNamespace(numVgprValu=13),
            mxsb=SimpleNamespace(numVgprValu=14),
            loopModelValuVgprs=dict(counts),
        ),
    )


def _kernel(use_loop_model=True, mx=True):
    return {
        "UseLoopModel": use_loop_model,
        "ProblemType": {
            "MXBlockA": 32 if mx else 0,
            "MXBlockB": 32 if mx else 0,
        },
    }


def test_ulm1_overwrites_scaffold_value_ring_sizes():
    writer = _writer({"A": 32, "B": 48, "MXSA": 8, "MXSB": 16})

    _apply_method()(writer, _kernel())

    assert writer.states.a.numVgprValu == 32
    assert writer.states.b.numVgprValu == 48
    assert writer.states.mxsa.numVgprValu == 8
    assert writer.states.mxsb.numVgprValu == 16


def test_ulm0_keeps_scaffold_value_ring_sizes():
    writer = _writer({"A": 32, "B": 48})

    _apply_method()(writer, _kernel(use_loop_model=False, mx=False))

    assert writer.states.a.numVgprValu == 11
    assert writer.states.b.numVgprValu == 12


def test_ulm1_requires_a_count_for_every_present_operand():
    writer = _writer({"A": 32})

    with pytest.raises(RuntimeError, match="B"):
        _apply_method()(writer, _kernel(mx=False))


def test_ulm1_set_table_uses_each_operands_rotation_width():
    writer = SimpleNamespace(
        states=SimpleNamespace(loopModelRegBufferWidths={"A": 2, "B": 3})
    )
    count = _function(
        _KERNEL_WRITER_ASSEMBLY,
        "valuBufferCount",
        {"self": writer, "kernel": {"UseLoopModel": True}, "numBi": 7},
    )

    assert count("A") == 2
    assert count("B") == 3

    legacy = _function(
        _KERNEL_WRITER_ASSEMBLY,
        "valuBufferCount",
        {"self": writer, "kernel": {"UseLoopModel": False}, "numBi": 7},
    )
    assert legacy("A") == 7

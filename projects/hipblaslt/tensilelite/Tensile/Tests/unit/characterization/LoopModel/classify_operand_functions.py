# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Sort the (theta, op) functions into MODEL (intrinsic) vs DECISION (policy-dependent).

Regenerates the table in `Tensile/LoopModel/docs/ARCHITECTURE.md`. Run from the tensilelite root:
    python Tensile/Tests/unit/characterization/LoopModel/classify_operand_functions.py

The test from the plan: does the answer change if PLR or the register budget changes? Mechanically:
a function is a DECISION if it takes S / dr / group / W as a parameter, or transitively calls one
that does. Everything else is intrinsic to theta and belongs on the operand.
"""
import ast, collections, pathlib

PKG = pathlib.Path("Tensile/LoopModel")
POLICY_ARGS = {"S", "dr", "dr_g", "W", "group", "depth", "shift", "off"}

defs, calls, home = {}, collections.defaultdict(set), {}
for path in sorted(PKG.rglob("*.py")):
    mod = str(path.relative_to(PKG).with_suffix("")).replace("/", ".")
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        args = [a.arg for a in node.args.args]
        defs[node.name] = args
        home[node.name] = mod
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call):
                f = sub.func
                name = getattr(f, "id", None) or getattr(f, "attr", None)
                if name:
                    calls[node.name].add(name)

direct = {n for n, a in defs.items() if POLICY_ARGS & set(a)}
policy = set(direct)
for _ in range(12):                       # transitive closure
    grew = {n for n in defs if calls[n] & policy}
    if grew <= policy:
        break
    policy |= grew

operand_fns = [n for n, a in defs.items()
               if len(a) >= 2 and a[0] in ("theta", "th") and a[1] in ("op", "operand")]

buckets = collections.defaultdict(list)
for n in sorted(operand_fns):
    kind = "DECISION" if n in policy else "MODEL"
    buckets[(kind, home[n])].append(n)

for kind in ("MODEL", "DECISION"):
    total = sum(len(v) for (k, _), v in buckets.items() if k == kind)
    print("=== %s (%d) ===" % (kind, total))
    for (k, mod), names in sorted(buckets.items()):
        if k == kind:
            print("  %-12s %s" % (mod, " ".join(names)))

# LoopModel characterization

The safety net for the LoopModel refactor. Three suites:

| file | pins |
|---|---|
| `test_loopmodel_golden_char.py` | **what the decoder emits** — a rendered digest and a structural digest per configuration over the whole reachable matrix (2619 cells), plus full LoopIR/GIR/ledger text for six representative cells |
| `test_loopmodel_schedule_char.py` | **what the decoder decided** — the per-operand facts table and the decision table, as readable text, for five shapes with a failure history |
| `test_loopmodel_structure_char.py` | **the shape of the package** — a ratchet on deferred imports, private exports, duplicate helpers, format sites, function length and complexity, operand-function count, error kinds, comment and docstring length, and the core's freedom from the host project |

`matrix.py` builds the configurations; `check_goldens.py` is the fast parallel checker used between
edits (seconds, versus ~20 min for the serial pytest run).

## Running

```bash
# the gate
PYTHONPATH=.:build_tmp/tensilelite/rocisa:Tensile/Tests/unit:Tensile/Tests/unit/characterization/LoopModel \
    python -m pytest -q Tensile/Tests/unit/characterization/LoopModel

# the inner loop
PYTHONPATH=.:build_tmp/tensilelite/rocisa:Tensile/Tests/unit:Tensile/Tests/unit/characterization/LoopModel \
    python Tensile/Tests/unit/characterization/LoopModel/check_goldens.py

# regenerate after an INTENDED behaviour change (never to make a red test green)
LOOPMODEL_GOLDEN_UPDATE=1 PYTHONPATH=... python -m pytest -q <the golden file>
```

`build_tmp/tensilelite/rocisa` is on the path because `build_gir` reaches the rocisa-importing
lowering modules; without it the GIR half of every digest silently degrades to a refusal.

## The matrix

2619 cells = 9 loop orders × 291 configurations (1944 bf16, 324 mxf8, 180 bffuse, 171 mxmix).

* **bf16** — waves {1,4} × TDMSplitA/B {0,1,2}² × PLR {0,1,2} × PGR {1,2} × DepthU {64,128}
* **bffuse** — the bf16 shape with TDMFuse on
* **mxf8** — split {0,1,2} × PLR {0,1,2} × PGR {1,2} × DepthU {128,256}, 4 waves, fused TDM
* **mxmix** — mixed-input MX

Refusals are pinned as behaviour: a configuration that raises records its exception type and
message, because "this is rejected" is part of what the decoder does today.

## Two digests per cell

`digests_*.json` hashes the **rendered** text (LoopIR, GIR, ledger). It moves whenever wording
does, which makes it a readable cosmetic diff but a poor gate.

`semantic_*.json` hashes the **structure** — resolved depths, LoopIR nodes, obligations, and the
GIR program — through `matrix._semantic_value`, which drops display-only fields, normalizes
renamed metadata keys, and canonicalizes object identities. That is the gate: a refactor must move
it by zero cells.

**Identities must be canonicalized, not hashed raw.** GIR token facts carry `id(inst)`. Hashing
the address made the semantic digest differ between two runs of the *same* tree on every cell whose
GIR carries a token — 441 of 2619, all of them MX — which reads exactly like a regression and is
not one. `_IDENTITY_KEYS` replaces each address with its ordinal in first-visit order, which keeps
the producer/consumer pairing (the part that is a fact about the schedule) and drops the address
(the part that is not).

## Two departures from the suite protocol

1. **Plain golden files under `__goldens__/`, not syrupy snapshots.** `syrupy` is a declared
   dependency (`pyproject.toml`, `tox.ini`) but is absent from the bare interpreter used to drive
   the refactor, and a golden that cannot be run cannot guard anything. The files are JSON and
   text, so they diff in git without a plugin. Recorded in `DECISIONS.md`.
2. **Digests rather than full text for the 2619 cells.** Full renders would be ~580k lines of
   golden. The `_codegen` harness sets the precedent (an order-invariant digest rather than a full
   assembly hash); the six full-text cells exist so a digest change is still readable.

## The structure ratchet

`BASELINE` in the structure suite holds today's numbers. Every bound is `<=`, never `==`, so
improving a number never breaks the test and regressing one always does. Each refactor step lowers
one entry; a check that reaches 0 stays at 0.

The starting values, measured by AST at the beginning of the refactor, and where they stand:

| property | at the start | bound | now |
|---|---|---|---|
| deferred (function-level) imports | 27 | 0 | 0 |
| private names in an `__all__` | 9 | 0 | 0 |
| helpers defined in two modules | 5 | 0 | 0 |
| format sites outside `render.py` | 90 | 20 | 20 |
| longest function | 1264 (`emit.build_ir`) | 80 | 80 |
| free functions taking `(theta, operand)` | 69 | 82 | 106 |
| distinct exception types raised | 5 | 3 | 3 |
| longest comment run | 58 lines | 3 | 3 |
| longest function/class docstring | — | 9 | 9 |

`operand_functions` is the one bound still above its target, and it rose rather than fell. The
counter measures the opposite of what it was for: splitting a branchy function into named
per-operand steps raises it, and so does merging duplicated derivations. Lowering it honestly means
moving those answers onto the placements — the unfinished half of "put each fact on its owner" —
not deleting the names that made the arithmetic readable.

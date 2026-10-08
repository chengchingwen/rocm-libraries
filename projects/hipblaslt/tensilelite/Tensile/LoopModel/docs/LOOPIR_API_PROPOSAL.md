# LoopIR API: second design

Status: implemented.

This design keeps the useful part of the first attempt: a small immutable tree, logical axis
expressions, and pure schedule transformations. It removes the parts that turned into a second
scheduler beside LoopIR.

## Contract

LoopIR is the schedule. Theta generation constructs a simple logical program and derives the final
program by applying named `Program -> Program` functions. Lowering does not choose an order,
prefetch distance, split, fuse, guard, or dependency. It only:

1. translates the finalized tree to existing GIR;
2. builds the CFG, generation phis, and generation transfers implied by that tree;
3. projects target-owned physical register/LDS rings through `ThetaTargetProfile`; and
4. converts derived logical dependences to the existing GIR dependency payload.

Addresses, vector widths, instruction widths, and wave-tile address arithmetic are not LoopIR
values.

## Schema

```text
Program(
    params: tuple[Param, ...],
    axes: tuple[Axis, ...],
    operands: tuple[Operand, ...],
    body: tuple[LoopIR, ...],
    agents: tuple[tuple[Axis, Scope], ...],
)

LoopIR :=
    Movement | Compute | Loop | Cond | Let | Switch | Region

Movement(
    op: MovementOp,  # TENSOR_LOAD | DS_LOAD | GLOBAL_LOAD | STORE
    effect: Effect | absent,  # common semantic effect, when shared
    transfers: tuple[Transfer, ...],
    completion: CompletionClass | IMMEDIATE,
)

Transfer(
    local_tile: Tile,
    effect: Effect | absent,  # member-specific semantic effect
    source: GLOBAL | SHARED | REGISTER,
    destination: GLOBAL | SHARED | REGISTER,
)

Effect(tile: Tile, where: Pred = TRUE)

Compute(
    op: ComputeOp,  # WMMA
    reads: tuple[Tile, ...],
    writes: tuple[Tile, ...],
    completion: CompletionClass | IMMEDIATE,
)

Tile(operand: Operand, indices: tuple[Expr, ...])
Operand(name, axes: tuple[Axis, ...], role: INPUT | OUTPUT)

Loop(axis, lo, hi, body)
Cond(pred, then_body, else_body)
Let(axis, value, body)
Switch(expression, cases, otherwise)
Region(tag)
```

`Region` is a zero-runtime marker in a body sequence. It is not a body wrapper. There is no
`Body`, `Inst`, `ScheduleIR`, `Peel`, `Wait`, `Sync`, authored dependency edge, or special
`LoopCopy` node in LoopIR.

Each concrete `MovementOp` registers its legal source/destination pair and endpoint semantics.
`Movement` runs that checker immediately, so `Movement(DS_LOAD, ...)` cannot contain a
global-to-shared transfer. `ComputeOp` is the
corresponding extension point for compute instructions; `WMMA` is currently its only concrete
operation.

Every Transfer has an explicit wave-local `local_tile` and resolves exactly one semantic `Effect`,
either from its Movement or from the Transfer itself.  A wave-sensitive effect tile
explicitly adds wave while its local tile does not.  For example, WT8/WG2 has eight local tiles and
sixteen `(wave, local tile)` effect locations.

Movement endpoint semantics are checked by the operation:

```text
TENSOR_LOAD:  GLOBAL effect tile -> SHARED effect tile
DS_LOAD:      SHARED effect tile -> REGISTER local tile
GLOBAL_LOAD:  GLOBAL effect tile -> REGISTER local tile
STORE:        REGISTER local tile -> GLOBAL effect tile
```

This is how TensorLoad-to-DS_LOAD dependences retain full wave identity while WMMA consumes only
the eight wave-local register tiles.

## Logical axes and tiles

There is one `Axis` type. Meaning comes from use:

- a serial `Loop` binds operation instances;
- `Program.agents` binds axes that identify cooperating agents;
- an axis in an operand index identifies a logical tile;
- a Program agent appears in an effect tile only when explicitly requested.

One movement covering several logical tiles lists several `Transfer` values. Coverage is local to
that operation; it does not introduce another Program axis.

For example:

```python
copy_a = Movement(
    TENSOR_LOAD,
    transfers=(
        Transfer(
            local_tile=A[iteration, m],
            source=GLOBAL,
            destination=SHARED,
        ),
    ),
    completion=copy_completion,
    effect=Effect(A[iteration, wave, m]),
)
```

does not say where bytes live or how an address is formed. It says that this issue moves the
logical `A[iteration, m]` tile from global to shared.

Wave is declared in `Program.agents` when available. It is not a `Loop` and is not encoded as a
tile size or read width. Declaring it does not add it to every effect; the generator must name it
in an effect tile/predicate or specialize control with `Cond`/`Switch`.

## Scheduling by functional transformation

Manipulation keys are operation values and functions, never strings. Public transformations are
ordinary typed functions:

```python
Program -> Program
```

The core set includes:

- `append`
- `partition_movement`
- `fuse_movements`
- `rewrite_operation`
- `pipeline`
- `defer_operations`
- `prioritize_operations`
- region extension/replacement
- finite operation-run materialization

`compose(*transforms)` combines them left to right. Every transformation returns a new immutable
`Program` and preserves hidden logical operation identity where one operation gains multiple
structural occurrences.

Theta decoding uses the same API. It does not directly manufacture a private schedule object for
lowering to interpret.

Target projection facts are finalized against that exact Program. A later rewrite that changes a
read tile while preserving its operation identity must re-finalize the decoded schedule; lowering
rejects a stale profile rather than consulting Theta again.

## Logical read retiming

Read-ahead is represented by changing the read's logical tile expression and issue position in the
tree.

For a mixed-radix logical walk, let:

```text
position = Σ(axis_i / coverage_i) * stride_i
shifted  = position + logical_advance
```

The retimed tile coordinates are mixed-radix digits of `shifted`; carry beyond the logical span
increments `iteration`. A final drain guard is:

```text
position + logical_advance < logical_span
```

This handles the TDMSplit cases without address data. For example, with
`M_split=2, M_inner=1`, a one-tile advance maps:

```text
(iter, M_split=0, M_inner=0) -> (iter,   M_split=1, M_inner=0)
(iter, M_split=1, M_inner=0) -> (iter+1, M_split=0, M_inner=0)
```

An operand invariant over an enclosing axis may need two first-touch placements around a wrap.
That choice is represented as an ordinary `Cond` in LoopIR. The semantic tile relation remains
independent of the eventual LDS/register address.

## TensorLoad split and fuse

TDMSplit and TDMFuse are represented only on the TensorLoad side:

- each actual split TensorLoad instruction is one `Movement`;
- a fused TensorLoad is one `Movement` with multiple `Transfer` values;
- disjoint domain partitions use transfer predicates;
- local reads and computes continue to name their own logical tiles.

The `_split` axis remains a serial schedule axis and a logical tile coordinate where applicable,
so split compute partitions can be reordered. TensorLoad/DS_LOAD semantic effects explicitly
relate split, optional wave, and wave-local tiles. Thus `by_coord`,
`by_wave`, mixed, and cross-region reads are derived intersections of effect domains rather than
stored modes.

Dependency analysis has no TDMSplit/TDMFuse special case. It compares the effects of the resulting
tiles and predicates.

Each fused movement unit owns an identity-valued `CompletionClass`; per-region completion creates
a distinct class per region. A target may map multiple logical classes to the same hardware
counter spelling. `required_dependences` derives dominating completion waits from Program
structure without authored wait nodes or lowering-time drain repair.

## Effects and dependences

Semantic effect tiles are explicit Movement/Transfer semantics. Occurrence-bound effects and
dependencies are analyses, not authored edge authority:

```text
EffectInstance(
    occurrence: Occurrence,
    endpoint:
        MovementSource | MovementDestination |
        ComputeRead | ComputeWrite,
)

Dependence(
    complete: EffectInstance,
    issue: EffectInstance,
    domain: SymbolicRelation,
)
```

The endpoint determines read/write role, so RAW and WAR are derived properties. Physical
write-after-write reuse belongs to GIR's target-level buffer analysis; LoopIR does not derive a
logical WAW edge. A dependence has no authored kind, edge ID, fence, distance, or scheduling label.

The relation equates semantic tile expressions under both effects' loop, binding, predicate, local
bulk, and agent domains. Consequently split regions, fused issues, and per-wave partitions produce
per-tile dependencies rather than a unioned operand-wide wait.

When affine analysis cannot prove a loop-carried distance, the relation is retained as
conservative and consumers must wait. Only proven-empty relations are omitted.

All ordering is completion-to-issue:

```text
complete(effect A) -> issue(effect B)
```

Physical aliases and storage reuse are target projection facts. They remain outside LoopIR and are
handled by the existing GIR analyses and FrameContract.

## Validation and diagnostics

- Constructors enforce typed nodes, declared axes, and exactly one Effect owner.
- `validate_loopir` checks phase structure, effect domains, derived dependences,
  and the finalized target projection.
- GIR `check_plan` checks lowered dataflow, register/LDS use, and synchronization.
- The cached LoopIR dump retains the target geometry header, including each
  operand's prefetch, register/VGPR, and shared-buffer sizes.

## Lowering boundary

The lowerer may inspect only:

- finalized LoopIR structure;
- derived effects/dependences; and
- target projection methods supplied by `ThetaTargetProfile`.

It must not import Theta traversal or scheduling policy. Runtime `iter` expressions are preserved
and projected into GIR `abs_gen`/`gdelta`; they are never discarded.

Physical register-ring phase can require several existing GIR steady blocks. The frontend derives
the closure period from target projection facts and materializes `_steady_copy` Lets in Program
before lowering; lowering reads that explicit structure and receives no loop-copy count.

## Required invariants

1. The core `loopir` package imports no Theta, GIR, datatype, or target implementation.
2. Dependency edges are always derived.
3. No address quantity enters `Program`, `Tile`, `Effect`, or `Dependence`.
4. Every scheduling choice is visible as ordinary LoopIR structure or tile expressions before
   lowering.
5. Lowering the new Program through the unchanged GIR pipeline is equivalent to the legacy
   Theta/LoopIR path.
6. Comment-free assembly is byte-identical after normalizing only nondeterministic label suffixes.

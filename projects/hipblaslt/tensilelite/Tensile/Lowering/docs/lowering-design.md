<!-- Copyright Advanced Micro Devices, Inc., or its affiliates. -->
<!-- SPDX-License-Identifier: MIT -->

# TensileLite Lowering Design — LoopIR → GIR → rocisa

**Status:** IMPLEMENTED (R0–R3 + R1 emit). `UseLoopModel=True` now lowers θ-IR → GIR → rocisa
end-to-end; the old θ-IR walker (`Components/LoopModelLowering.py`) is **retired/deleted** and GIR
is the single source of truth for swap + GR-increment placement. Numerics pass on
`loopmodel_bf16_gfx1250.yaml` (4 layouts × PLR0/PLR1). This document is the **source of truth**
for the lowering; if it disagrees with the plan (`sharded-forging-hartmanis.md`), **this document
wins**. Still open: R4 (scaffold-anchor `Tag`/`guard_site`, GSU-on), `transfer_size`, and the §13
Phase-4/5 items (LoopIR srcs/dsts, 6-axis Coord, VgprPartition>1, wave-spec).

**Cross-references:** LoopModel (the θ decoder + LoopIR) — `Tensile/LoopModel/`; the LoopIR
node classes — `Tensile/LoopModel/ir.py`; the 6-mode iteration vocabulary —
`Tensile/LoopModel/translate.py`; the StinkyTofu Analysis/Pass model —
`/Workspace/mi450-dev/shared/stinkytofu/`.

---

## 0. Naming

| Term | Meaning |
|---|---|
| **LoopModel** | the component that produces the LoopIR from kernel params (θ decoder). |
| **LoopIR** | layer-1 IR: the rolled, address-free intra-workgroup schedule LoopModel emits. |
| **GIR** (GEMM IR) | layer-2 IR: the GEMM-algorithm dataflow DSL this document defines. |
| **Tile** | a logical operand fragment at an iteration coordinate, in a memory space. |
| **Gen** | a loop-carried *generation* — the SSA value for a staged buffer's version. |
| **Ref** | a *use* of a Tile inside an instruction: a Tile at a Gen(+δ) or at a register (group, slot), plus size. |
| **Move** | the data-movement verb: move a Tile one hop down the memory hierarchy. |
| **Mma** | the compute verb: one matrix-multiply-accumulate, with open `srcs`/`dsts`. |
| **Phase** | a software-pipeline region: `prologue` / `steady` / `drain{n}` / `tail`. |
| **Block** | one Phase as a basic block in the GIR CFG. |
| **Program** | the whole mainloop as one GIR CFG. |

Note the two distinct uses of "generation": the **LDS generation** (a `Gen`, loop-carried)
and the **register generation** (a rotation slot within a register group, concrete after
enumeration). §2.3 shows why only the first is a `Gen`.

---

## 1. Why two IRs (LoopIR and GIR)

### 1.1 They differ by *closure*, not amount of detail

- **LoopIR is a closed, legal *point*.** LoopModel produces exactly one canonical LoopIR
  per kernel, carrying its own legality guarantee (the obligation ledger is discharged). Its
  invariant is strong: *"this is a legal intra-workgroup schedule."*
- **GIR is an open, mutable *space*.** It is built to be rewritten by passes; intermediate
  states may be partial or not-yet-legal until a later pass repairs them. Its invariant is
  weak by design: *"a well-formed GEMM-dataflow graph."*

One object cannot be both (a fixed legal point vs a mutable optimization surface).
Therefore **the LoopIR→GIR lowering is where we trade the schedule's built-in legality for
manipulation freedom**, and take on preserving legality via GIR's verifier and analyses.

### 1.2 What GIR adds over LoopIR (formal), and how to think about it (developer view)

**Formal grounding (what GIR actually is).** GIR earns its existence on three concrete
counts, none of which is "more scope":
1. **Closure** (§1.1) — LoopIR is a proven-legal fixed point; GIR is the mutable optimization
   surface. One object cannot be both.
2. **Generation-SSA over a real CFG** — GIR makes the loop-carried buffer generation a
   first-class SSA value (phis + back-edge transfer) over an explicit control-flow graph.
   This is the load-bearing addition: it collapses the ~39 stage-conditional heuristics of the
   old walker into a single `gen_reaching` query (§6.1).
3. **Testability** — GIR (nodes, analyses, passes) imports no rocisa, so it is unit-testable
   standalone (§11); the correctness-critical dataflow is checked without a kernel build.

That is the honest "why two IRs." GIR does **not** structurally carry the reduction/workgroup/
wave layer: GSU/StreamK are located-by-analysis and realized by reused Components (§8.1),
`guard_site` cannot even locate StreamK (§6), VectorWidth is ejected (§8.3), and PersistentLoop
is a LoopModel change GIR merely absorbs (§8.4). The genuinely-new content GIR adds is the
mutable CFG + gen-SSA + the analysis/pass machinery — not a new scope.

**Developer mental model (how to work in GIR).** For a developer *using* GIR, the useful
picture is: **GIR is the place you express and manipulate GEMM algorithm in GEMM terms** —
tiles flowing global→shared→register via `Move`, consumed by `Mma`, over the (m,n,k)+iter
iteration space, with generations as the double-buffering. LoopIR is a **preset** fixing the
intra-workgroup schedule; you work *around* it via analyses and passes (buffering generations,
swap points, guard sites, and — via the promotion path §8.1 — potentially more of the
reduction/wg/wave layer later). In that sense GIR is a small **GEMM DSL**. This is the
*aspirational / extensibility* reading, deliberately weaker than "GIR carries the whole GEMM":
today most of the wg/wave layer is delegated to Components (§8.1), and the DSL grows toward it
only through the documented promotion path — it is not claimed to be there now.

The two readings are complementary: the formal three counts say *why the layer exists*; the
DSL view says *how to think about editing it*. Where they could conflict — "does GIR carry the
whole GEMM?" — the formal reading wins: it does not; it carries the intra-workgroup dataflow
plus query hooks, and grows by promotion.

### 1.3 Keep LoopIR frozen; let GIR grow

LoopIR is a small (~10-node), stable **contract**. GIR is TensileLite's working surface,
free to churn. LoopIR is also rolled (one node per op-class, trip-count-independent); GIR is
expanded (concrete instructions + a CFG). So LoopIR stays minimal and frozen; GIR grows.

### 1.4 The fill-in-the-blank pipeline, and the GIR↔L3 seam

`LoopIR → GIR → rocisa` is fill-in-the-blank: each layer is intentionally under-specified;
each later layer inserts what the earlier omitted. **A single concern (swap, address-shift,
token, stagger, GSU-guard) is SPLIT across GIR and L3 by role:**

- **GIR owns *what / where / why*** — the GEMM-semantic *fact*, produced by an Analysis
  ("the read Gen transitions v→v+1 at point P"; "this access's generation differs from the
  held one"; "the accumulation-consistent drain point is Block D"). No hardware in it.
- **L3 owns *how*** — expand that fact into instruction(s) per TensileLite setup + hardware
  cap (`v_xor` vs inline-const; byte math with `LdsPad`; one vs N vectorized loads; or
  *call the existing `Component.GSU` at the located point*). No whether/where decision.

So "address-shift is *decided in GIR, realized in L3*," never "address-shift is L3." L3
consumes GIR's semantic queries — the LLVM split (a pass decides "spill here"; target
lowering emits the `store`).

| Layer | Carries | Omits (filled later) |
|---|---|---|
| 1. **LoopIR** | rolled intra-workgroup schedule; hops; wmma; symbolic slots; dep skeleton | reduction/workgroup/wave layer; all L2/L3 detail |
| 2. **GIR** | GEMM-dataflow graph (6-axis) over a mutable CFG + gen-SSA + **semantic Analysis surface** + materialized anchors/tokens (wg/wave layer via query hooks, not structural — §1.2) | addresses, register colors, vector-instruction expansion, pointers, hw arithmetic |
| 3. **rocisa** (`Lowering/gir_to_rocisa.py` + Components) | physical color; addresses; vector expansion; pointer/swap realization; **calls the GSU Component at GIR-located points; StreamK stays scaffold-located (T2, §6)** | waitcnt, hazards (→L4) |
| 4. **StinkyTofu** | waitcnt; hazards; bank-conflict; regalloc; **instruction scheduling** | — |

**Rules.** **R-SEMANTIC:** GIR carries GEMM semantics, never machine realization (a swap is
a *Gen transition* fact, not a `v_xor`). **R-LEGAL:** GIR entry is legal (inherits the
preset); passes may create illegal intermediates but must restore `verify_gir` before the
pipeline ends. **R-ONCE:** KernelWriter builds the LoopIR once and lowers to GIR once; the
phase forks only emit from it.

---

## 2. Layer 1 — the LoopIR contract (input)

### 2.1 The 6-axis iteration vocabulary (NOT 3)

LoopIR's iteration space has **six** canonical modes, because each of K/M/N contributes a
**(split-region, inner-remainder)** pair — this is what expresses TDMSplit, DU-split, and
region interleaving:

```
0  K_split   (K/DU region)
1  K_inner   (inner K-reduction remainder / substep)      [reducible]
2  M_split   (M region)
3  M_inner   (in-region M fan — A's grouping mode)
4  N_split   (N region)
5  N_inner   (in-region N fan — B's grouping mode)
```

`LoopOrder` is a word over {K,M,N}: a 3-letter shortcut keeps each axis's (split,inner) pair
contiguous (`"KMN"` = the six in canonical order); a 6-letter word lets regions interleave.
A split of 1 drops that mode; all-1 degenerates to `[K_inner, M_inner, N_inner]` (the KMN
case in §2.3). The outer reduction trip `iter` (K//DU) is implicit above the six.

**GIR's coordinate system is these six modes + `iter`.** A design keyed only on (m,n,k)
cannot carry the split axes and is wrong.

### 2.2 LoopIR nodes (`ir.py`)

- **`Load(tokens, src, dst, kiter, coord, size_regs, size_bytes, part, n_parts)`** — a
  data-movement hop; `coord` over the 6 modes. global→shared = copy; shared→register = read;
  global→register = direct (DirectToVgpr).
- **`Mma(a, b, acc, kiter, coord, scales, block)`** — a fragment MAC; currently fixed
  fields `a`/`b`/`acc` plus `scales`. **Gap:** the fixed fields cannot express structured-
  sparse metadata or a fused `C` addend — LoopIR needs the same `srcs`/`dsts` generalization
  GIR uses (§2.4).
- **`Placement(space, slots, reg_off, byte_off, src_slot)`** — `slots = ((group, Expr), …)`,
  **one entry per register group** (register partitioning); `src_slot` = LDS source
  generation; `mod` = ring size S.
- **`Expr(var, mod, add, terms, carry)`**, `eval(env)`.
- **`Inst(op, placement, awaits)`**, **`Await(dep, counter, scope, kind)`** with
  `kind ∈ {RAW-residency, rotation-WAR, inplace-WAR}`.
- Structure: **`Loop(mode, extent, bodies)`**, **`Branch(mode, modulus, arms)`**,
  **`Peel(prologue|drain)`**, **`Cond(pred, then, els, label)`**.

LoopModel also assigns each operand a **ρ agent-role** and supports **register grouping**
via `VgprPartition` (a fragment's registers split into N groups — lo/hi or g0..g{N-1} — each
group holding a share of the fragment and rotating with its own width W; `Placement.slots`
carries one slot Expr per group). GIR's `Ref` must carry (group, slot) to represent this
(§4.2).

`emit_mainloop(theta)` → `{S, ir, ledger, ledger_empty, loop, tensor_per_kiter, …}`.
Invariants GIR relies on: **I-PHASES** (prologue Peel / steady Loop-in-Cond / drain Peel
present); **I-SLOT** (`src_slot`=LDS gen, `slots`=register gen per group, drain origin pinned
absolute); **I-AWAIT** (deps on `Inst.awaits`, carrying RAW/WAR `kind`).

### 2.3 Ground truth — one concrete steady trip (KMN, S=2, bf16, VgprPartition=1)

```
for K_inner in 0,1:
  for M_inner in 0,1:
    READ A   reg(group 0, slot=(K_inner+1)%2)   lds_gen=(iter+(K_inner+1)//2)%2
    for N_inner in 0,1:
      (M_inner==0) READ B  reg(group 0, slot=(K_inner+1)%2)  lds_gen=(iter+(K_inner+1)//2)%2
      MMA acc[M,N] += A*B  at (K_inner, M_inner, N_inner)
COPY A  global->shared  lds_gen=(iter+2)%2 = iter%2
COPY B  global->shared  lds_gen=iter%2
```

| quantity | expression | depends on | after inner enumeration |
|---|---|---|---|
| register slot (read dst / mma src) | `(K_inner+1)%2` | bounded inner index only | **concrete int** — not loop-carried |
| LDS generation (read src / copy dst) | `(iter+…)%2` | `iter` (+ inner carry) | **loop-carried** — a `Gen` |

Only the LDS generation is loop-carried → it is the one SSA value class (`Gen`). Register
slots are concrete attributes. With `VgprPartition>1` there are multiple groups, each with
its own slot and rotation width; still concrete after enumeration.

### 2.4 LoopIR change needed: `Mma` → `srcs`/`dsts`

LoopIR's `Mma` (and, for symmetry, its data-movement view) should adopt the same
`srcs`/`dsts` open-operand form as GIR (§4.3), replacing the fixed `a`/`b`/`acc` fields:

```python
# LoopIR Mma, generalized (coordinated LoopModel edit):
Mma(srcs, dsts, kiter, coord, block)   # srcs/dsts = typed operands; operand identity on each
```

Rationale: fixed fields cannot carry structured-sparse `meta` or a fused `C` addend, and the
`scales` side-field is a symptom of the same rigidity. Making LoopIR `srcs`/`dsts`:
- lets LoopModel express MX (scaleA/scaleB), sparse (meta), and fused (C) uniformly;
- makes `loopir_to_gir` (§5) a near-1:1 operand copy instead of field-mapping;
- keeps LoopIR otherwise frozen — this is the one coordinated LoopModel change this effort
  requires, tracked here because the doc is the source of truth.

Until this lands, sparse/fused are blocked at the LoopIR boundary (not at GIR). GIR's `Mma`
(§4.3) already has the target shape.

---

## 3. GIR overview — a GEMM-algorithm DSL in three pillars

1. **A GEMM-dataflow graph** — Tiles moved by **Move** and consumed by **Mma**, over the
   6-axis iteration **CFG**, with the loop-carried **Gen** as its one SSA value class (§4).
2. **A semantic Analysis surface** — the query API that makes GIR manipulable in GEMM terms
   and is the **contract L3 is written against** (§6).
3. **Thin Passes** — mutate the IR, mostly "run analysis, materialize its result" (§7).

Analysis vs Pass follows StinkyTofu exactly: **Analysis** = pure, cached, no mutation;
**Pass** = mutates, declares invalidated analyses; pass order is data in a pipeline file.

**GIR is a minimal CFG** — not a general control-flow graph. It has exactly the structure
the loop-carried Gen analysis needs: a `prologue` block, one `steady` block with a
back-edge to itself, the `drain{n}` blocks, and an optional `tail`. No arbitrary branching;
the peel-validity conditions are recorded as block attributes for L3, not emitted as GIR
branches (the KernelWriter scaffold owns real control flow — §10).

**What GIR is not:** not an instruction scheduler (StinkyTofu / L4); not a place to *choose*
a different intra-workgroup schedule (LoopIR fixed that).

---

## 4. GIR — the graph (nouns and verbs)

### 4.1 Iteration coordinate

A `Coord` maps the subset of the 6 canonical modes a node spans to concrete ints (A spans
K+M modes; B spans K+N; acc spans M+N; a scale spans its operand's modes; metadata spans the
sparse operand's modes). **No `iter` appears in a Coord** — the cross-trip relation lives in
the `Gen` phi, not in a coordinate.

### 4.2 Tile, Gen, Ref (the nouns)

```python
@dataclass(frozen=True)
class Tile:                 # a logical operand fragment — 'what data', no color/address
    operand: str            # 'A'|'B'|'C'|'acc'|'scaleA'|'scaleB'|'meta'|...  (open set)
    space:   str            # 'global'|'shared'|'register'
    coord:   dict           # subset of the 6 modes this fragment covers
    shape:   dict           # logical extent (element/reg counts) — SIZE, not address

@dataclass(frozen=True)
class Gen:                  # a loop-carried GENERATION — the ONE SSA value class in GIR
    id:   int
    ring: int               # ring size S (1 = in-place, never changes)
    # defined by a Phi in the steady header; back-edge transfer: v' = (v + adv) % ring

@dataclass(frozen=True)
class Ref:                  # a USE of a Tile inside an instruction
    tile:     Tile
    # -- shared-staged residence (loop-carried) --
    gen:      Gen | None = None   # which LDS generation
    gdelta:   int = 0             # concrete read-ahead offset from `gen` (0, 1, …)
    # -- register residence (concrete; VgprPartition-aware) --
    group:    int | None = None   # register group INDEX (0..VgprPartition-1); no name
    slot:     int | None = None   # concrete rotation slot within that group
    # -- size --
    size_regs:  int = 0
    size_bytes: int = 0
```

`Ref` answers register grouping: a register use names its **group index** and its **slot
within that group** (both concrete after enumeration). The group's rotation width W is not on
the Ref and is **not a GIR decision** — W is already decided by LoopModel and baked into the
LoopIR slot `Expr.mod` (`translate.py:_register_policy`); `reg_gen` reads it from there and
`reg_band` only *validates* it against computed liveness (B3, §6). The Ref is a *use*; W is a
LoopIR fact GIR consumes.

### 4.3 Move and Mma (the verbs) — unified `srcs`/`dsts`; semantics from `Tile.operand`

```python
@dataclass
class Move:                 # THE data-movement verb: one hop down the memory hierarchy
    srcs: tuple             # (Ref,)   e.g. a global tile
    dsts: tuple             # (Ref,)   e.g. a shared tile at a Gen (copy), or a register (read)
    deps: tuple = ()        # RAW + WAR edges carried from LoopIR awaits

@dataclass
class Mma:                  # THE compute verb — OPEN operand set via srcs/dsts
    srcs: tuple             # (Ref, …)  identity is Ref.tile.operand:
                            #   'A','B' (always); 'C' (fused A*B+C); 'scaleA','scaleB' (MX);
                            #   'meta' (structured-sparse)
    dsts: tuple             # (Ref,)    the accumulator ('acc')
    coord: dict             # over the 6 modes
    block: str = ""         # hardware wmma block group
    deps: tuple = ()
```

**No `role` map (question #4).** A `Mma` is just `srcs`/`dsts` lists of `Ref`; the operand
identity (A vs B vs scaleA vs meta vs C) is read from each `Ref.tile.operand`. This composes
every wmma variant with no schema change: plain = (A,B); MX = (A,B,scaleA,scaleB); sparse =
(A,B,meta); fused = (A,B,C). Each extra operand is just another `Tile` produced by its own
`Move` chain and appended to `srcs` — no new verb, no role vocabulary. `Move` and `Mma`
share the same `srcs`/`dsts` shape, so analyses/passes treat operands uniformly.

`Move`/`Mma` are the two *instruction* verbs. There is no `v_xor`/pointer/address node —
those are L3 realizations. But GIR does need one more body primitive to hold **semantic
points** (§4.4).

### 4.4 Mark — a semantic point, and the region-based insertion model

A `Mark` is a body element (like `Move`/`Mma`) carrying **no data movement or compute**, only
a **GEMM-semantic fact at a program point** for L3 to realize (`Mark(kind='swap',
{hop:read, gen_from:0, gen_to:1})` = the *fact* "the read generation changes 0→1 here," NOT a
`v_xor`; R-SEMANTIC).

```python
@dataclass
class Mark:
    kind: str      # 'swap' | 'gr_increment' | 'gsu_guard' | 'phase_boundary' | 'barrier'
    at:   dict     # kind-specific payload (schema in §4.7 G-MARK)
```

**But a `Mark` in the body is the OUTPUT of a three-layer process, not something a pass
drops wherever it likes.** A swap/increment/guard has a *legal placement range*, not a single
correct point: a read-hop swap `gen_from→gen_to` may sit anywhere **after the last access
using `gen_from`** and **before the first access using `gen_to`** — any point in that interval
is correct, and *which* point is a scheduling choice, not a semantic one. Baking a fixed body
index into the analysis is premature realization (the very mistake the old walker made).

So GIR mirrors StinkyTofu's waitcnt/liveness shape (analysis → plan → apply), three layers:

```python
@dataclass(frozen=True)
class Region:                  # a legal placement window, opaque ordinals over one block's body
    block: str
    after:  object            # anchor: place strictly AFTER this (a node, or block-entry)
    before: object            # anchor: place strictly BEFORE this (a node, or block-exit)
    def contains(self, pos): ...

@dataclass(frozen=True)
class PendingMark:             # an un-placed Mark: the fact + WHERE it may legally go
    mark:   Mark
    region: Region
```

1. **Analysis → a plan of `PendingMark`s (IR untouched).** e.g. `swap_regions(prog)` returns,
   per required swap, a `PendingMark(Mark('swap',…), Region(after=last-use-of-gen_from,
   before=first-use-of-gen_to))`. Plain data keyed by anchor, like StinkyTofu's
   `WaitInsertionPlan`; the *value* (the Mark) is separate from the *position* (the Region).
2. **Placement → picks a concrete point inside each `Region`** (default: earliest legal; a
   scheduler may prefer otherwise), producing an anchor-keyed placement. Still IR-untouched.
   This is StinkyTofu's `WaitPlanOptimizer.rewrite` role.
3. **Apply (the only IR mutator) → inserts the `Mark` before its resolved anchor**, walking
   the body in program order (StinkyTofu's `emitWaits` + `IRBuilder.createIR(insertBefore)`).

A `Region` spanning a block edge (e.g. `after` drain0's last gen_from read, `before` drain1's
first gen_to read) naturally expresses a cross-block swap — no single index needed.

**Two sources of `PendingMark`s — split by *what decides them*:**
1. **Native (LoopIR→GIR lowering)** — facts that follow directly from LoopIR **structure**,
   layout-independent: `phase_boundary` (prologue/steady/drain/tail) and `barrier` (where
   LoopIR has a sync). Their region is a point (the phase head), so they need no placement
   step. Emitted by the lowering.
2. **Analysis-produced** — facts requiring a **decision informed by tensilelite layout**:
   - `swap_regions` → swap `PendingMark`s where the reaching generation changes (§6/§7.1);
   - `gr_increment_regions` → increment `PendingMark`s: GIR marks only *where* (a PGR copy
     advances the global read each trip); the increment's *magnitude/wrap* (StaggerU WrapU,
     TDMSplit split-inc, Sparse, PGR pf-index, base stride) is realized entirely by L3's
     `tdmIncrementAB` from kernel state — no GIR tag (it would only duplicate kernel state);
   - `guard_site(kind)` → a `gsu_guard` `PendingMark` at the located region.

The **analysis never mutates the IR** — it returns `PendingMark`s (fixing the earlier draft
where "passes materialized Marks into the body" conflated analysis with mutation). Only the
single **apply** stage writes `Mark`s into `Block.body`.

Both kinds are the same primitive; L3 walks the body and realizes `Move`/`Mma`/`Mark`
uniformly (a `Mark` → the swap/increment/Component-call/label its `kind` names).

> Note: with `Mark`, swap and gr-increment *insertion* ARE GIR passes (they materialize
> semantic `Mark`s), but their *realization* (`v_xor`, `WrapU`/`SCSelectB32`) is still L3. No
> pass produces machine instructions — passes only produce semantic Marks (R-SEMANTIC).

### 4.5 Control-flow primitives (the CFG terminators)

GIR is a real CFG, so each `Block` ends in an explicit **terminator**, and predicates are
first-class. These are the control-flow primitives:

```python
@dataclass(frozen=True)
class Pred:                  # a symbolic predicate over trip-count symbols (no hardware)
    expr: str               # e.g. 'T > M' (enter steady) / 'T == 1' (single-trip -> drain)
    label: str = ""         # scaffold hint (e.g. 'toPGR1') — advisory, see below

@dataclass(frozen=True)
class Goto:                 # unconditional terminator
    target: str             # target Block label

@dataclass(frozen=True)
class CondGoto:             # conditional terminator (peel-validity guard AND the loop back-edge)
    pred:   Pred
    t_target: str           # taken block
    f_target: str           # fall-through block
```

There is **no separate loop-back primitive.** The steady back-edge is just a `CondGoto`
whose taken target is the loop header — `CondGoto(Pred('trip remaining'), t_target='steady',
f_target='drain0')`. "Is this a back-edge?" is a *graph property* (the target dominates the
source), answered by a `back_edges(prog)` analysis over the CFG, not by a distinct node type
(exactly as LLVM's `br` closes a loop or not with no special opcode). `gen_reaching` uses
`back_edges` to find which `CondGoto` carries the steady `xfer`.

**Recognized, not emitted (the reuse split).** GIR *carries* terminators so it is a genuine,
verifiable CFG and analyses see the back-edge and the peel-validity branch. But L3 does **not
emit** them: the KernelWriter scaffold (`openLoop`/`closeLoop`/the `toPGR1` ladder) already
emits the real branches and the trip-count compare. L3 recognizes each terminator and routes
the block under the matching scaffold label — the same "IR faithful / lowering suppresses"
split LoopIR's `Cond(pred,label)` already uses (LoopIR shows `Cond(pred='T > 2',
label='toPGR1', then=[Loop])`). `Pred.label` is the advisory hint that names the scaffold
branch; a from-scratch backend could instead emit `Pred.expr` directly.

So the terminators are **analysis-bearing structure**: they make the CFG well-formed and let
`gen_reaching` follow the back-edge, without GIR owning real control flow.

### 4.6 Block and Program (the minimal CFG)

```python
@dataclass
class Block:
    phase: str              # 'prologue' | 'steady' | 'drain{n}' | 'tail'
    loop:  bool = False     # steady: its CondGoto taken-target is itself (a back-edge)
    preds: tuple = (); succs: tuple = ()
    phis:  list = None      # Gen phis (loop header): v = φ(entry, v_next)
    xfers: list = None      # back-edge transfers: v_next = (v + adv) % ring
    body:  list = None      # [Move | Mma | Mark], program order, concrete coords
    term:  object = None    # the terminator: Goto | CondGoto
    role:  str = "all"      # wave role (ρ); 'all' until wave-specialization (§8.2)
    # (semantic anchors are Marks in `body`, §4.4 — not a side attribute dict)

@dataclass
class Program:
    blocks: list            # [prologue, steady(loop), drain0..N, tail?]  — the minimal CFG
    entry:  str             # entry block label
    tiles:  dict            # interned Tile table
    params: dict            # PRESETS analyses derive from (StaggerU, MI/MT, …); not raw data
    meta:   dict            # {S, roles, dtypes, …}
    def walk_rpo(self): ...  # reverse-postorder over the minimal CFG (follows terminators)
```

The minimal CFG is exactly: `prologue —Goto→ steady`; `steady —CondGoto(trip remaining →
steady, else → drain0)→` (the taken target `steady` is a back-edge, found by dominance, not
a special node); `drain0 → … → drainN —Goto→ (tail | end)`. The peel-validity `CondGoto`
(single-trip → skip steady → drain) sits at the prologue/steady seam, mirroring LoopIR's
`Cond`.

### 4.7 Verifier (`verify_gir`)

**G-GEN-SSA** (each Gen has exactly one def — a phi or a `GenXfer` — and **every Gen has a
use**, dominated through phis, incl. non-loop JOIN blocks like `drain0` §5.1 Fact 2. A Gen
with a def and no use within the modeled region is a **verifier error** — a dropped consumer
is the bug this check exists to catch. The one case that looked use-free — the drain
descriptor swap — is NOT modeled as a use-free Gen: its consumer (the tail copy) is out of
GIR's current scope, so that swap is **scaffold-owned**, not a GIR Gen transition, §5.1
Fact 3); **G-NOITER** (no `iter` in a Coord);
**G-SEMANTIC** (no physical field on any node); **G-CFG** (one prologue, `M` drains where
`M` = `_peel_depths` — NOT approximated as PGR-1, `decoder.py:198` — optional tail; **and, in
single-tile scope, exactly one steady-loop** — this "one loop" clause is the single-tile
assumption that §8.4's persistent loop relaxes to "≥1 loop"); **G-TERM** (every Block has a
terminator; every target exists; `preds`/`succs` agree; back-edges are found by dominance — a
`CondGoto` whose target dominates it. **Enumerate back-edges by dominance, not a single
`blk.loop` flag** (§8.4): in single-tile scope there is exactly one such back-edge (the
steady loop), but coding the check per-dominance-back-edge now makes the persistent case a
no-op relaxation later);
**G-OPERANDS** (every `Mma`/`Move` src resolves to a Ref whose producing `Move` exists —
MX/sparse operands not dangling); **G-MARK** (each `Mark.kind` is in the closed set below and
its `at` payload matches that kind's schema — no free-form kinds, no missing fields).

`Mark` payload schema (the closed near-term set; the verifier enforces per-kind fields so
`Mark` can't become a dumping ground):

| `kind` | source | required `at` fields |
|---|---|---|
| `phase_boundary` | native | `phase` |
| `barrier` | native | (none) |
| `swap` | pass (SwapInsert / Anchor for the drain copy-descriptor) | `hop∈{read,copy}`, `gen_from`, `gen_to`, **and the pointer name: `operand` when `hop=read`, `unit` when `hop=copy`** |
| `gr_increment` | pass (GrIncrement) | `unit` (placement only; magnitude is L3's `tdmIncrementGir`) |
| `gsu_guard` | pass (Anchor) | `variant` |

**`operand` vs `unit` — the pointer identity, and why it is hop-dependent.** A READ is per
op-class: A's read and B's read are separate instructions at separate addresses through separate
`LocalReadAddr` registers, so a read swap names an operand. A COPY is per **Φ movement** (§2.8
move 9): a fused group is ONE cooperative instruction over ONE descriptor, so the copy, the copy-
hop swap and the global-read advance all name the movement's **member tuple** — `('A',)` unfused,
`('A','B')` when multi-wave TDM aliases B's descriptor onto A's SGPRs and wave parity picks the
operand. `gir/refs.copy_unit` is the single derivation, and it CHECKS the members rotate in
lockstep rather than assuming it: they legitimately name different buffers (different `Gen.id` —
A's LDS region is not B's), and what makes one register able to serve them all is a shared ring
depth and offset.

Before this, three consumers (`swap_regions._hop_access`, `gr_increment._copy_of`,
`emit_plan._plan_move`) each took the FIRST shared dst Ref and named the movement by that member.
Under multi-wave the resulting instruction count is still correct — the descriptor really is
aliased, so one pointer is the right answer — which is exactly why it went unnoticed. A right count
from a wrong reason is not a modelled aliasing, and it left the analyses unable to tell a fused
movement from a lone-A one.

---

## 5. A concrete GIR example (simple bf16 GEMM, KMN, PGR2/PLR1, S=2, VgprPartition=1)

Textual dump of the `Program` for the §2.3 kernel (indices concrete; `v` is the steady Gen):

```
Program(entry=prologue, S=2, order=KMN)

Block prologue:
  Move  A.global{k=0}       -> A.shared[gen=0]          # fill buffer 0
  Move  B.global{k=0}       -> B.shared[gen=0]
  Move  A.shared[gen=0,δ=0]{m=*,k=0} -> A.reg(grp=0, slot=0){m=*,k=0}   # prime substep-0 regs
  Move  B.shared[gen=0,δ=0]{n=*,k=0} -> B.reg(grp=0, slot=0){n=*,k=0}
  term: CondGoto(pred=Pred('T > 2', label='toPGR1'), t_target=steady, f_target=drain0)

Block steady  loop=True  preds=(prologue,steady)  succs=(steady,drain0):
  phis:  v = φ(prologue: 0, steady: v_next)             # LDS generation, ring=2
  # MIWaveTile 2x2: m,n in {0,1}. acc(m,n)=A(m,k)*B(n,k): A shared across n, B across m.
  # A/B tiles at the same k differ by tile.coord (m for A, n for B); reg slot rotates on k.
  body:
    # --- substep K_inner=0: reads use generation v (this buffer) ---
    Move  A.shared[gen=v,δ=0]{m=0,k=0} -> A.reg(grp=0, slot=0){m=0,k=0}   # head reads
    Move  A.shared[gen=v,δ=0]{m=1,k=0} -> A.reg(grp=0, slot=0){m=1,k=0}
    Move  B.shared[gen=v,δ=0]{n=0,k=0} -> B.reg(grp=0, slot=0){n=0,k=0}
    Move  B.shared[gen=v,δ=0]{n=1,k=0} -> B.reg(grp=0, slot=0){n=1,k=0}
    Mma   srcs=(A.reg{m=0}, B.reg{n=0}) -> acc{m=0,n=0,k=0}
    Mma   srcs=(A.reg{m=0}, B.reg{n=1}) -> acc{m=0,n=1,k=0}
    Mma   srcs=(A.reg{m=1}, B.reg{n=0}) -> acc{m=1,n=0,k=0}
    Mma   srcs=(A.reg{m=1}, B.reg{n=1}) -> acc{m=1,n=1,k=0}
    Mark  swap {hop:read, gen_from:v, gen_to:v+1}          # placed by swap_regions→apply (§7.1)
    # --- substep K_inner=1: read-ahead uses generation v, δ=1 (next buffer), slot=1 ---
    Move  A.shared[gen=v,δ=1]{m=0,k=1} -> A.reg(grp=0, slot=1){m=0,k=1}
    Move  A.shared[gen=v,δ=1]{m=1,k=1} -> A.reg(grp=0, slot=1){m=1,k=1}
    Move  B.shared[gen=v,δ=1]{n=0,k=1} -> B.reg(grp=0, slot=1){n=0,k=1}
    Move  B.shared[gen=v,δ=1]{n=1,k=1} -> B.reg(grp=0, slot=1){n=1,k=1}
    Mma   srcs=(A.reg{m=0,k=1}, B.reg{n=0,k=1}) -> acc{m=0,n=0,k=1}
    Mma   srcs=(A.reg{m=0,k=1}, B.reg{n=1,k=1}) -> acc{m=0,n=1,k=1}
    Mma   srcs=(A.reg{m=1,k=1}, B.reg{n=0,k=1}) -> acc{m=1,n=0,k=1}
    Mma   srcs=(A.reg{m=1,k=1}, B.reg{n=1,k=1}) -> acc{m=1,n=1,k=1}
    # PGR copy (refill). With ring=2, gen=(v+2)%2==v: it rewrites the buffer being READ this
    # trip, so it carries a WAR dep on the reads that vacate it (rotation-WAR) — L4 must not
    # hoist the copy above them. This edge is load-bearing; shown explicitly here.
    Move  A.global{next} -> A.shared[gen=(v+2)%2]  deps=(WAR: A reads of gen v this trip)
    Move  B.global{next} -> B.shared[gen=(v+2)%2]  deps=(WAR: B reads of gen v this trip)
  xfers: v_next = (v + 1) % 2                            # back-edge: advance generation
  term: CondGoto(pred=Pred('trip remaining'), t_target=steady, f_target=drain0)  # back-edge (target dominates)

Block drain0  preds=(prologue, steady)  succs=(drain1):    # see §5.1 for the full drain
  ...
  term: Goto(drain1)
Block drain1  preds=(drain0)  succs=(end):
  ...
  term: Goto(end)
```

Every block ends in a terminator (§4.5); analyses follow them. L3 recognizes the
`CondGoto`s and routes to the scaffold's `toPGR1`/`LoopBeginL`/`LoopEndL` labels rather than
emitting the compare (the scaffold already does).

Reading it as GEMM: A/B tiles flow global→shared (Move copy) → register (Move read) →
consumed by Mma; double-buffering is the `Gen v` with ring 2; the read-ahead is `δ=1`; the
per-trip buffer advance is the back-edge `v_next=(v+1)%2`. No pointer, no `v_xor`, no
address — those are L3's realization of the Gen transitions this graph states. Also note the
refill copy carries a WAR dep on the read that vacates its buffer (§6 `dep_defuse`); the
`deps=` are elided above for brevity but shown in §5.1.

### 5.1 The drain, in full (M=2 from `_peel_depths`, `decoder.py:198`)

The drain is the Lemma-1 tail: `M` straight-line blocks (`M` = the peel depth from
`_peel_depths`, NOT approximated as PGR-1) that finish the in-flight generations without
issuing new global loads. For this kernel M=2. The drain is where the current walker's
ugliest heuristics live, so it is designed here explicitly — three facts that were previously
reconstructed:

```
Block drain0  preds=(prologue, steady)  succs=(drain1):
  phis:  v = φ(prologue: 0, steady: <steady back-edge exit gen>)   # JOIN block — see below
  body:
    # drain step 0 STILL reads (draining the pipe): reads use generation resolved from `v`.
    Move  A.shared[gen=v,δ=0]{m=*,k=0} -> A.reg(grp=0,slot=0)      # (WAR-free: pure read)
    Move  B.shared[gen=v,δ=0]{n=*,k=0} -> B.reg(grp=0,slot=0)
    Mma  ... (2x2 fan, k=0) ...
    Mark  swap {hop:read, gen_from:v, gen_to:v+1}                  # swap_regions→apply (gen changes)
    Move  A.shared[gen=v,δ=1]{m=*,k=1} -> A.reg(grp=0,slot=1)      # read-ahead of next gen
    Move  B.shared[gen=v,δ=1]{n=*,k=1} -> B.reg(grp=0,slot=1)
    Mma  ... (2x2 fan, k=1) ...
    Mark  gsu_guard {variant:nll}          # guard_site→apply: GSU==1 drain guard (§6)
    # (descriptor swap for the tail copy is SCAFFOLD-OWNED, not a GIR Mark — its consumer is
    #  the tail block, out of single-tile scope; §5.1 Fact 3. L3 reuses swapCopyCode here.)
  term: Goto(drain1)

Block drain1  preds=(drain0)  succs=(end):
  body:
    # drain step 1 tapers: a couple of reads then only Mma (no more reads to prefetch).
    Move  A.shared[gen=v',δ=0] -> ...   Move B... ; Mma ... (k=0)
    Mma  ... (k=1; the remaining accumulations, no reads) ...
  term: Goto(end)
```

The drain has three subtleties, each checked against the real ULM0 assembly for this kernel
(`z_ULM0_PLR1_SIA4_NT2`, K=256/DU=64). The `LoopEndL`→`toPGR1` drain block is:

```
label_LoopEndL:                                    # === the drain (NGLL) ===
  s_wait_tensorcnt 0
  ds_load_tr16_b128 vgprValuA_X0_… , vgprLocalReadAddrA  offset:0/1536/48/1584  # reads gen (X0)
  ds_load_tr16_b128 vgprValuB_X0_… , vgprLocalReadAddrB  offset:0/1536/48/1584
  s_cselect_b32 s38, sgprWrapUA, sgprGlobalReadIncsA     # GR-increment (StaggerU wrap)…
  s_add_u64     sgprtdmAGroup0+2, sgprtdmAGroup0+2, s38  #   …TDM addr += inc  (NO tensor_load here)
  s_cselect_b32 s38, sgprWrapUB, sgprGlobalReadIncsB
  s_add_u64     sgprtdmBGroup0+2, sgprtdmBGroup0+2, s38
  ds_load_tr16_b128 vgprValuA_X1_… , vgprLocalReadAddrA  offset:3072/4608/3120/4656  # read-ahead (X1)
  ds_load_tr16_b128 vgprValuB_X1_… , vgprLocalReadAddrB  offset:3072/…
  s_xor_b32 sgprtdmAGroup0+1, sgprtdmAGroup0+1, 0x4000   # DESCRIPTOR swap  (line 1639)
  s_xor_b32 sgprtdmBGroup0+1, sgprtdmBGroup0+1, 0x4000
  v_xor_b32 vgprLocalReadAddrA, 0x4000, vgprLocalReadAddrA  # read-pointer swap
  v_xor_b32 vgprLocalReadAddrB, 0x4000, vgprLocalReadAddrB
label_toPGR1:                                      # → OptNLL epilogue (stores D); no copy here
```

So the drain has **reads + GR-increment (`s_add tdm+=inc`) + descriptor swap (`s_xor tdm`) +
read-pointer swap (`v_xor`), and NO `tensor_load` in the drain block itself.** But — correcting
an earlier claim — there **is** a `tensor_load` later, in the **tail** block (`Tail global read
A/B`, and it reads the *same* `sgprtdmAGroup0` the drain's `s_xor` rotated). The tail is
conditionally skipped (`s_cbranch_scc1 label_SkipTailLoopL` when `numIter==0`), so for this
K%DU==0 size it does not execute — which is *why* the drain block shows no consumer — but for a
K%DU≠0 kernel the tail copy is the descriptor swap's real consumer. The shipping comment
`LoopModelLowering.py:387-390` ("so the later tail copy writes the correct buffer") is
therefore **correct**, not stale; my earlier "no consumer" reading looked only at the
(skipped) drain→toPGR1 span and missed the tail. This changes Fact 3 below.

**Fact 1 — read-hop drain generations are ABSOLUTE (T3 read-hop, proven).** The decoder pins
the drain origin: `_pin_iter` (`decoder.py:610-629`) wraps each drain step in
`Branch(mode="iter", arms={(t+1)%d: nodes})`, so each drain read's `src_slot=(iter+…)%S`
evaluates to a **step-constant independent of the runtime trip count T** (the walker confirms
this: `_hopBuffer` evaluates `src_slot` with `iter` bound by that Branch,
`LoopModelLowering.py:485-489`). So drain reads **do not consume the incoming loop-carried
`v` at all** — they read fixed parities. `gen_reaching` returns these constants; the
`swap_regions` analysis (§7.1) then bounds the drain0→drain1 swap by ordinary last-use/
first-use over these constants — no cross-block special case.

**Fact 2 — the drain0 JOIN needs a phi ONLY for the copy hop, and it is a real select, not a
max.** `drain0` has two predecessors: `steady` (long loop) and `prologue` (the peel-validity
`CondGoto` short-loop skip, `T ≤ M`). For the **read hop** the join is trivial — by Fact 1 the
drain reads are absolute constants that don't depend on which predecessor arrived, so there is
nothing to merge. For the **copy hop** the two predecessors *do* deliver different descriptor
generations — but this is resolved as a **derived analysis value, NOT a materialized Gen
def**. `gen_reaching` computes the drain-entry copy generation across the join (a real select,
runtime-selected on the taken edge / provably-equal — **NOT `max-over-preds`**; generations
are ring residues mod `S`, for which `max(0,1)` is meaningless) purely so the scaffold/L3 can
realize the descriptor swap at the correct parity. It does **not** mint a copy-hop `Gen` def
in the drain: since the drain issues no copy Move and the swap is scaffold-owned (Fact 3),
there is no in-GIR copy *use* in the drain, so creating a Gen def there would dangle under
strict G-GEN-SSA (§4.7). The steady copy `Gen` keeps its real uses (next-trip reads, RAW) and
terminates cleanly; the drain-entry value is an analysis result, not a new SSA value.
`resolve_phi_entry` therefore handles this non-loop join at **analysis** granularity (a
merged entry value), not by inserting a phi node. (The short-loop `T ≤ M` path is where the
two copy generations actually differ and is still unvalidated — §13.)

**Fact 3 — the drain descriptor swap's consumer is the TAIL copy (out of GIR's single-tile
scope), so the swap is SCAFFOLD-OWNED until the tail is modeled (B2 = Option B).** The `s_xor
sgprtdmAGroup0` in the drain rotates the descriptor so the **tail** `tensor_load` (`Tail global
read A/B`) writes the correct LDS bank — a real consumer, living in the **tail block**. Current
GIR scope is single-tile **without a modeled tail** (the tail loop is task #94, deferred; and it
is conditionally skipped for K%DU==0, which is why the drain→toPGR1 span alone shows no
consumer). So the copy-hop `Gen`'s use lives in a block GIR does not yet model. Rather than
invent a use-free Gen — which would let `gir_to_rocisa` drop the tail copy and still pass the
verifier, the exact bug the SSA check exists to catch — we treat this like StreamK (T2, §6):
- **the drain descriptor swap is SCAFFOLD-OWNED, not a GIR `Mark`.** GIR does not state it; L3
  reuses the existing `swapCopyCode` path (`LoopModelLowering.py:387-394`) at the drain, exactly
  as the current walker does. This is the honest treatment of an out-of-GIR consumer.
- the copy-hop `Gen` is threaded through GIR's modeled blocks with its `GenXfer`, but within
  the modeled region it terminates at the drain boundary; its onward transition (feeding the
  tail) is scaffold-realized. GIR carries the generation; it does not claim a swap whose
  consumer it cannot see.

**G-GEN-SSA stays STRICT: every Gen must have a use** — a missing consumer is a bug the
verifier must catch, so there is **no** global "use optional" relaxation. The copy-hop Gen is
not use-free; its use is the tail copy, currently outside GIR, so the drain descriptor swap is
scaffold-owned (above), not a Gen with a dangling def.

**Promotion path (B→A):** once GIR models a `tail` block (task #94), the tail copy becomes a
real `Move`, the copy-hop `Gen` gains an in-GIR use, and the drain descriptor swap becomes an
ordinary `PendingMark(swap, hop=copy)` — scaffold-owned → GIR-owned, no new mechanism (the §8.1
promotion pattern). Under PersistentLoop+PAP (§8.4) it resolves the same way: the drain then
contains the **next tile's copies** (θ paper §7.2: PAP = `retime` on the outer persistent
level), giving the copy-hop `Gen` an in-GIR use inside the drain via the outer loop-carried
phi. Either path (tail #94 or PAP) moves the swap from scaffold-owned to GIR-owned.

The order-driven (SIA4) correctness of the drain refill (when GIR does own it) hinges on the
σ_c "refill-after-vacating-read" rule and the **per-edge** discharge gate — θ paper §5.3.1
point 4 and §5.5 (coverage-granularity is insufficient for an order-driven backend; the
per-edge gate is required). Our `dep_defuse` WAR forwarding (§6) is that per-edge edge.

---

## 6. Analyses — the semantic query surface (the L3 contract)

Pure, cached, invalidated on mutation. This is what makes GIR a manipulable DSL and the API
L3 is written against.

```python
class AnalysisManager:                # StinkyTofu-shaped: lazy + cached + invalidated
    def get(self, analysis, prog):
        if (analysis, prog.version) not in self.cache:
            self.cache[key] = analysis.run(prog)
        return self.cache[key]
    def invalidate(self, names): ...   # a Pass calls this for what it changed
```

Two shapes of analysis (both pure, IR-untouched): **fact analyses** return per-node data;
**region analyses** return `PendingMark`s (a fact + its legal `Region`, §4.4) for the
placement+apply stages to realize.

| Analysis | Kind | Answers | Consumed by |
|---|---|---|---|
| `gen_reaching` | fact | concrete generation each access uses, incl. the back-edge transfer | `swap_regions`; L3 |
| `dep_defuse` | fact | RAW (copy→read→mma, per operand incl. scale/meta) + WAR (refill↔vacating read) edges | tokens; L3 WAR forwarding |
| `transfer_size` | fact | logical size each `Move` transfers | L3 vector expansion |
| `reg_band` | fact | per register group, live peak L / reload rate R → **validates** LoopIR's W | reg_gen; (W from `slot.mod`) |
| `preset_fact(name)` | fact | *(design pattern, not yet built)* derive a GEMM fact from a `Program.params` preset when a pass genuinely needs it; NOT used for gr_increment (its magnitude is L3's, §4.4) | future analyses |
| `swap_regions` | region | per required swap: `PendingMark(swap, Region(after last-use-of-gen_from, before first-use-of-gen_to))`, from `gen_reaching` | placement → apply |
| `gr_increment_regions` | region | per steady copy: `PendingMark(gr_increment, Region)` — placement only (magnitude is L3's `tdmIncrementAB`) | placement → apply |
| `guard_site(kind)` | region | `PendingMark(gsu_guard/…, Region)` at the intra-workgroup site (GSU==1 drain guard, PGR-last-iter) | placement → apply (L3 invokes Component) |

`guard_site` is the reuse crux for the anchors GIR **can** locate: **L3 asks GIR "where does
a GSU==1 drain guard go?"** GIR answers from the intra-workgroup dataflow (accumulation
consistency at a drain phase boundary); L3 then calls the tested `Component.GSU` there.

**`guard_site` does NOT cover StreamK (T2 scope limit).** StreamK partitions the K range
across *workgroups* — a grid-level fact that is, by construction, **not** in the
intra-workgroup dataflow GIR models (nothing in `Program` §4.6 can derive it). StreamK stays
located and realized entirely in the KernelWriter scaffold/Component, outside GIR. If a future
need requires GIR to reason about it, `Program.meta` would have to carry an explicit
workgroup-mapping fact first (a promotion, §8.1) — until then, do not pretend `guard_site` can
answer it.

**`reg_band` validates, it does not decide (B3).** The register rotation width `W` is already
decided by LoopModel: the policy is picked in `translate.py:_register_policy`
(unroll/floor/inplace) and the numeric modulus baked into the slot by `_rate_slot`
(`decoder.py`), read today by `_scanRegDepth`. The θ paper backs this ownership: **Lemma 3c**
(§2.6) proves the post-WAR live-peak floor `L` — hence `W ∈ [L,R]` — is a function of
`(ord, tile, off)` and the structural WAR alone, **σ-invariant and fixed at S-search time**
(i.e. decided in LoopModel, before any emission order). So W is a LoopIR fact GIR *consumes*,
not one GIR makes — deciding it in GIR would be a second authority that can disagree with the
decoder (violating R-LEGAL). `reg_gen` reads W from the slot `mod`; `reg_band` at most
cross-checks it against computed liveness and flags a mismatch.

### 6.1 How an analysis runs on GIR (worked: `gen_reaching`)

```python
def gen_reaching(prog) -> dict[node, int]:
    """Concrete generation each Move/Mma access uses. SSA reaching-value over the CFG:
    resolve each Ref.gen phi to the block-entry value (phi/select at a join, §5.1 Fact 2),
    add gdelta, mod ring; each back-edge's xfer gives the entry value of the next trip."""
    out = {}
    back = back_edges(prog)                             # by DOMINANCE, not a blk.loop flag (§4.7/§8.4)
    for blk in prog.walk_rpo():
        entry = resolve_phi_entry(blk)                 # per-hop Gen value on block entry (merges preds)
        for inst in blk.body:
            for ref in inst.srcs + inst.dsts:
                if ref.gen is not None:
                    out[(inst, ref)] = (entry[ref.gen] + ref.gdelta) % ref.gen.ring
        # one transfer per back-edge whose header is `blk` (single-tile: exactly one; persistent:
        # the K-loop AND the tile-loop, each with its own xfer — no single blk.loop/xfers[0])
        for be in back.headed_by(blk):
            out[(blk, be)] = (entry[be.gen] + be.xfer.adv) % be.xfer.ring
    return out
```

For §5: PLR1 reads are δ=0 then δ=1 → the value changes mid-body (one swap there); the
back-edge `(v+1)%2` returns to the entry parity of the *next* trip (balanced). PLR0 (no
read-ahead) → all δ=0, value constant in-body, changes only at the back-edge → one swap at
loop bottom. `1LDSBuffer` (ring 1) → `(v+1)%1==v`, never changes → no swap. One analysis,
all cases; L3 emits a swap exactly where this value changes.

---

## 7. Passes + pipeline (analysis → placement → apply)

Mirroring StinkyTofu's 3-layer split (§4.4): **region analyses** produce `PendingMark`s
(IR-untouched), a **placement** stage resolves each `Region` to a concrete anchor
(IR-untouched), and a single **apply** pass mutates the IR. Only apply writes to `Block.body`.

```python
class Pass:
    def run(self, prog, am): ...       # returns invalidated analysis names

# pipeline.py — pass order is DATA (the "backend" file)
PIPELINE = [
    RegGenPass(),                      # register (group,slot); W read from slot.mod (B3)
    TokensPass(),                      # stamp read tokens + WAR edges (dep_defuse)
    CollectPendingMarksPass(),         # run swap_regions + gr_increment_regions + guard_site
    PlacementPass(),                   # pick a concrete anchor in each PendingMark.region
    ApplyMarksPass(),                  # THE mutator: insert each Mark before its anchor
]                                      # (+ WaveSpecializePass later)
def run_pipeline(prog, ctx):
    am = AnalysisManager()
    for p in PIPELINE:
        am.invalidate(p.run(prog, am))
    verify_gir(prog)
    return prog
```

### 7.1 Region analysis (worked: `swap_regions`) — a GLOBAL edge-based dataflow

The physical LDS read pointer (`LocalReadAddr`) and the TDM write descriptor are **single
registers that flow through the entire mainloop** — the prologue positions them, steady rotates
them each trip, the drain/NLL continues.  So `swap_regions` is **one global reaching-definition
dataflow over the whole CFG** (prologue→steady→drain as one timeline), NOT a per-block pass.
This is the load-bearing reason GIR exists: an earlier per-block version re-seeded each block from
its own first access, which dropped the prologue→steady and steady→drain hand-offs, so the
prefetch/NLL swaps were wrong (PLR1 failed while PLR0 coincidentally survived).

The pointer's selected generation must **equal `gen_reaching` at every use**, and a swap is the
only op that changes it.  So minimal placement = insert a swap wherever the required generation
changes between two consecutive uses — computed per `(hop, operand)` as `need_in`/`need_out` per
block plus edge reconciliation:

```python
class SwapRegions(Analysis):
    def run(self, prog, am):
        reach = am.get(GenReaching(), prog)
        back  = am.get(BackEdges(), prog)
        pending = []
        for hop in ('read', 'copy'):            # copy hop: steady-owned (drain copy scaffold, Fact 3)
            for operand in operands(prog, hop):
                # need_in[B] = gen the FIRST hop access in B needs; need_out[B] = gen the LAST leaves.
                # INTRA-block: consecutive accesses whose gen differs -> a swap between them.
                # BACK-EDGE : treat as an edge — next-trip need_in = need_in advanced by the xfer;
                #             differs from need_out -> the per-trip loop-bottom swap.
                # FORWARD EDGE pred->succ: need_out[pred] != need_in[succ] -> a swap on that edge,
                #   placed by edge TYPE:
                #     into a loop header (prologue->steady) -> in the PREHEADER (end of pred), ONCE;
                #     straight-line (steady->drain0, draini->draini+1) -> at SUCC ENTRY;
                #     back-edge (steady->steady) -> the per-trip swap (below).
                pending += place_operand(prog, hop, operand, reach, back)
        return pending
```

Placement by swap **type** (a user directive, not just σ):
- **COPY / TDM write-descriptor swap** → place **early**: before the block's first copy Move, so
  the descriptor rotates before the next `tensor_load`.
- **READ / LDS-pointer swap** → place in the **middle** of its region: after the last read of
  `gen_from`, before the first read of `gen_to` (PLR1: mid-body at the read-ahead; PLR0: the
  region ends at the trip bottom `BLOCK_EXIT`, since the swapped gen is first used next trip).

Result for §5/§5.1, one global analysis, all cases:
- steady PLR1: read gen changes at the read-ahead → one mid-body swap per operand; the copy
  descriptor toggles on the back-edge → one early loop-bottom swap per operand.
- steady PLR0: no in-body read change; the read swap lands at the trip bottom; copy as above.
- **prologue→steady / steady→drain / drain{n}→drain{n+1}**: cross-edge generation changes fall
  out of `need_out[pred] != need_in[succ]` — the hand-offs the per-block version missed.
- drain copy descriptor (§5.1 Fact 3): its consumer (the tail copy) is out of single-tile scope,
  so the copy hop is steady-owned and the drain issues no copy Move — no `PendingMark`. (Becomes
  one once the tail (#94) or PAP (§8.4) is modeled.)
- `1LDSBuffer` (ring 1): `gen_reaching` constant → no `PendingMark`.

`gr_increment_regions` (§4.4) is the **same edge-based dataflow**: the global-read address is a
loop-carried pointer that advances on the steady back-edge; the increment is placed **early**
(before the first copy Move, like the TDM swap) and realized by `writer.tdmIncrementAB`.

### 7.2 Placement + apply

```python
class PlacementPass(Pass):
    def run(self, prog, am):
        for pm in prog.pending:                    # from CollectPendingMarksPass
            pm.anchor = earliest_legal(pm.region)  # default: earliest; a scheduler may override
        return []                                  # IR untouched
class ApplyMarksPass(Pass):                        # the ONLY IR mutator
    def run(self, prog, am):
        for pm in sorted(prog.pending, key=program_order):
            insert_before(pm.anchor, pm.mark)      # StinkyTofu emitWaits + createIR(insertBefore)
        return ["body"]
```

**Core passes recap:** `RegGenPass` (assign each register `Ref`'s (group index, slot); W is
READ from the LoopIR slot `mod`, not decided — `reg_band` validates only, B3; sole author of
register *color/slot*, not depth); `TokensPass` (stamp read tokens + WAR edges from
`dep_defuse`); `CollectPendingMarksPass` (runs `swap_regions`, `gr_increment_regions`,
`guard_site` — all region analyses producing `PendingMark`s; native `phase_boundary`/`barrier`
Marks were emitted by the lowering, §4.4); `PlacementPass`; `ApplyMarksPass` (the sole
mutator); *(future)* `WaveSpecializePass` (§8.2).

L3 realizes every applied `Mark` (swap → `v_xor`/inline-const; gr_increment →
`WrapU`/`SCSelectB32`; gsu_guard → Component call). No stage before apply mutates the IR, and
no stage emits machine instructions — analyses produce facts+regions, apply writes semantic
Marks, L3 realizes them (R-SEMANTIC).

---

## 8. Scope decisions

### 8.1 Rich query surface, NOT first-class GSU/StreamK — with a promotion path

GIR does not model split-K reduction / multi-buffer workspace / StreamK partials as
first-class nodes; that would force re-implementing the two heaviest Components and
generating them from GIR (the rewrite we avoid). They are **located by Analysis**
(`guard_site`) and **realized by the existing Components**.

**Promotion path** (so this is stage one of the ambitious version, not a detour): a
first-class construct is a query result materialized into a node. To promote (e.g. GSU as a
real GIR reduction-partition a pass can reason over), add the node **and** a pass that builds
it from the same Analysis the query already uses. Nothing built now is discarded.

### 8.2 Wave specialization (future pass; room reserved now)

Different waves run different streams (producer waves: global→shared Moves; consumer waves:
shared→register Moves + Mma). A genuine GIR dataflow-restructuring transform (partitions by
`Block.role`, adds cross-wave dependency edges, changes occupancy) — not L3 realization.
LoopModel's ρ role is its seed. Reserve `Block.role` + a cross-wave dep kind now; the pass is
future work, addable without reshaping §4.

### 8.3 Presets, not fixed data (StaggerU-like)

Some params enter GIR the way MI/MT/vector-element enter LoopModel: as **presets an analysis
derives facts from** (`preset_fact`), never as hardcoded low-level values. `Program.params`
holds the preset; `preset_fact` turns it into a GEMM fact; L3 realizes the arithmetic. The
pattern for every "L3 arithmetic driven by a param" (StaggerU, LdsPad, …).

**VectorWidthA/B is NOT even a GIR concern.** In TensileLite VW is a *lane↔element
distribution* (VW=2 = each lane holds 2 contiguous elements — a real sub-lane data layout),
not the schedule, not the dataflow, not the register *footprint* (VW=1 vs 2 hold the same
total registers, just distributed across lanes differently). It changes no Tile, Move, Mma,
Gen, or dep. So GIR stays **lane-distribution-agnostic** (same category as byte-address): VW
lives entirely in `Program.params` for L3 + the reused tensilelite address/layout machinery.
GIR never reasons about which lane holds which element. (If a future pass ever needs lane
layout — none identified, not even wave-spec, which partitions by *wave* not lane — the §8.1
promotion path adds it then.)

### 8.4 PersistentLoop + PrefetchAcrossPersistent (PAP) — a LoopModel change GIR absorbs for free

The θ paper §7.2 settles where this belongs: a **plain grid-stride / persistent kernel is
fully base θ** — the persistent loop is an **outer `time` level** (`resort` retags the
grid-tile coord from an agent to an outer looped mode, + `tile`), and **PAP is `retime` on
that persistent level** (issuing the next tile's copies during the current tile's drain — the
cross-level peel of Lemma 1). The reverse store trajectory (`register→shared→global`) is
`stage`; store-register reuse is the Lemma 2b cross-trajectory WAR. No new obligation kind.

Consequences for this effort:
- **It is a LoopModel emission change, not a GIR/scaffold injection.** LoopModel must emit the
  outer `time` level + its `retime`; then the LoopIR already contains the next-tile copies in
  the drain. Injecting PAP copies GIR-side would be reconstructing schedule outside the proven
  model — the disease we are removing.
- **GIR needs NO new node type, but the verifier/analysis DO relax from single-loop.** The
  node primitives already suffice: `Gen.ring`, `CondGoto`, and `Block.phis`/`xfers` are lists,
  so a second nested phi + `GenXfer` (the outer persistent loop's loop-carried `Gen`) is
  expressible with no new node. The spec is **already written to not hardcode single-loop** —
  not "no redesign," but the relaxation is pre-baked so persistent is a no-op:
  - **G-CFG** says "in single-tile scope, exactly one steady-loop" and **G-TERM** enumerates
    back-edges **by dominance** (§4.7) — so a persistent loop's second back-edge is admitted
    by relaxing the parenthetical single-tile clause to "≥1 loop," not by changing the check's
    mechanism.
  - `gen_reaching` (§6.1) already keys the transfer off `back_edges(prog)`/`back.headed_by(blk)`
    (one entry per back-edge), NOT a single `blk.loop`/`xfers[0]` — so a block inside both the
    K-loop and the tile-loop, carrying two loop-carried Gens of different rings, is handled with
    no analysis change.
  These are localized relaxations of one invariant clause, built *from* the existing
  list-valued primitives and dominance-based analysis — not a reshape of nodes or passes. Once
  relaxed, `gen_reaching` over
  the two-level CFG computes the cross-tile generation as it does the K-loop. It also resolves
  the §5.1 Fact 3 consumer question: under PAP the drain copy-hop `Gen` gains a **real in-GIR
  use** (the next tile's copies live in the drain), so the descriptor swap becomes GIR-owned
  (not scaffold-owned) via the outer phi.
  **Implication for R0:** build `verify.py`/`analysis.py` with back-edges enumerated by
  dominance (not a single `blk.loop`/`xfers[0]`), even for the single-loop case, so the
  persistent relaxation is a no-op later rather than a rewrite. This is the one place §8.4
  touches R0 design.
- **Out of base θ (deferred, named):** a **reducing epilogue** (split-K/atomic accumulate)
  needs the atomic-completion primitive (same bounded extension as GSU-MultipleBuffer); a
  **serpentine tile sweep** needs the `τ` traversal enrichment (non-monotone free-mode order).
  Neither is in the loopmodel test matrix.

**Verdict: deferrable.** PersistentLoop/PAP is not in the bf16 bring-up matrix. It needs **no
new GIR node type** (list-valued phis/xfers + CondGoto already suffice), a **LoopModel emission
enhancement** (the outer `time` level), and **localized relaxations of two verifier invariants
+ one analysis** (G-CFG/G-TERM single-loop → ≥1 loop; `gen_reaching` per-back-edge). The only
R0 obligation it imposes now: **enumerate back-edges by dominance, not a single `blk.loop`
flag** (above), so the later relaxation is a no-op. Build GIR single-tile now; add the outer
level in LoopModel when persistent kernels enter scope (task #113).

---

## 9. GIR → rocisa (L3) — the fact-consuming expander (as built)

L3 (`GirToRocisa`) **walks each block's emit plan and realizes `Move`/`Mma`/`Mark`**, consuming
GIR analysis facts and the `Program.params` presets, per setup+cap.  Each swap/increment Mark is
realized via the scaffold primitive for its SUBJECT (§4.7: `operand` on a read hop, `unit` on a
copy hop) — GIR owns the *placement*, the primitive owns the *instruction*:
- **`Mark(swap, hop='read')`** → `writer.localReadSwapOffsets(kernel, internalPointerSwap, tP)`
  (`v_xor LocalReadAddr`); deeper rings carry the specific `gen_from`/`gen_to` rotation.
- **`Mark(swap, hop='copy')`** → `writer.tdmSwapLdsOffset(kernel, tP)` (`s_xor` TDM descriptor),
  through the owning member's tP — a fused movement has one descriptor, so one swap.
- **`Mark(gr_increment)`** → `writer.tdmIncrementGir(kernel, tP, wrapLead, tPFused)`
  (`s_add tdm+=inc`; the StaggerU `WrapU`/`SCSelectB32` cselect, TDMSplit split-inc, Sparse are all
  derived inside the primitive from kernel state — the Mark carries no magnitude).
- **`Mark(gsu_guard)`** → call `Component.GSU.noLoadLoop` at that point (GSU==1 drain guard +
  accumulation save/restore preserved). (StreamK is not a GIR Mark — it is located and realized
  entirely in the scaffold, T2/§6.)
- **`Move`** → address (`gen_delta` + reused address-calc, `LdsPad`/layout) + vector
  expansion into N transfers at `VectorWidth`/`GlobalReadVectorWidth` (the lane distribution
  is pure L3, §8.3); token stamp (`ldsReadTokenIdx`/MemTokenData) + WAR forwarding from the
  Move's `deps` so L4 cannot hoist a refill above the vacating read.
- **operand emit** ← `Mma.srcs` (identity via `Ref.tile.operand`) → emit the right wmma
  variant: plain / MX-scaled / sparse / fused. Leaf emitters move here from the current
  walker (the one heuristic line `depth=self._regDepth` becomes "read the Ref's register
  depth from `reg_gen`").

Full L3 spec written after §4–§7 are confirmed.

---

## 10. KernelWriter integration (R-ONCE) — as built

```python
def _loopModelGirProgram(self, kernel):        # build ONCE per kernel, cached on the writer
    if self._loopModelGirCache matches kernel:  return cached
    theta = params_to_theta(kernel_to_params(kernel))
    return build_gir(theta, mainloop=emit_mainloop(theta))     # lower + run_pipeline ONCE

def _girEmitStage(self, kernel, tPA, tPB, phase, *, internalPointerSwap, copyByTc):
    prog = self._loopModelGirProgram(kernel)                   # shared across the 3 forks
    return GirToRocisa(self, kernel, tPA, tPB, prog).emit_block(
        prog, phase, tpByOperand={'A':tPA,'B':tPB, …MX…},
        internalPointerSwap=internalPointerSwap, copyByTc=copyByTc)
```

The three forks — **steady** (`_loopBody`), **drain** (`noLoadLoopBody`, phase `drain{step}`),
**prologue** — each call `_girEmitStage(...)`; the `Program` is built once and cached (R-ONCE).
GIR **owns swap + GR-increment placement** (its dataflow passes): L3 realizes each per-operand
swap/increment Mark via the scaffold's per-operand primitives (`tdmSwapLdsOffset` /
`localReadSwapOffsets` / `tdmIncrementAB`) — TensileLite issues **neither** on this path (the
scaffold's bundled swap/inc Modules are not consumed).  Only the copy *body* (`tensor_load`) and
the waits/sync scaffold are reused, positioned by GIR.  The KernelWriter scaffold still owns real
control flow (openLoop/closeLoop, the trip-count / `toPGR1` branches); GIR's minimal CFG + block
attrs are the facts L3 consumes.  `UseLoopModel=True` selects this path (no `==2` variant — the
walker is deleted).

---

## 11. Testability & migration

- `Lowering/gir/` (graph + analyses + passes) + `loopir_to_gir.py` import nothing from
  rocisa → **unit-testable standalone** (golden `Program`/analysis results per pass,
  verifier, gen-SSA queries). This + **extensibility** (sparse/MX/F32-emul; non-KMN +
  register groups #89; TDMSplit/MX #90; mxf8 #106; wave-spec later — all GEMM-dataflow
  additions, not special cases) justifies GIR as a real IR over a flat timeline.
  GIR unit tests: `Tensile/Tests/unit/test_gir.py` (GIR itself) + `test_gir_to_rocisa.py`
  (the pure emit-plan half), sharing `gir_fixtures.py`.
- `gir_to_rocisa.py` / `leaves.py` exercised in a real build (leaf emitters touch LDS-token
  state); the USER runs numerics.
- Migration is **complete for bf16**: the θ-IR walker is deleted and `UseLoopModel=True` is the
  GIR path unconditionally.  Bar met: err=0 on `loopmodel_bf16_gfx1250.yaml`
  (4 layouts × PLR0/PLR1).  GSU-on is R4; wider dtypes/orders are Phase 4/5.

---

## 12. Module layout

```
Tensile/Lowering/
  lowering-design.md          # this document
  loopir_to_gir.py            # layer 1->2 (the only place LoopIR Expr is read); build_gir (R-ONCE)
  gir_to_rocisa.py            # layer 2->3: GirToRocisa drives LeafEmitters from the emit plan;
                              #   realizes swap/gr_inc Marks via per-operand scaffold primitives
                              #   (tdmSwapLdsOffset / localReadSwapOffsets / tdmIncrementAB)
  leaves.py                   # LeafEmitters (per-tile wmma/ds_read realization) — L3, shared
  gir/
    nodes.py                  # Tile, Gen, Ref, Move, Mma, Mark, Pred, Goto, CondGoto, Block, Program
    region.py                 # Region, PendingMark (§4.4)
    analysis.py               # INFRA only: Analysis base class + AnalysisManager (lazy/cached)
    analyses/                 # one class per module (each an Analysis subclass)
      cfg.py                  #   Dominators, BackEdges (+ successors helper)
      gen_reaching.py         #   GenReaching (§6.1) -> Reaching result object (query API)
      swap_regions.py         #   SwapRegions (global edge-based dataflow, §7.1)
      dep_defuse.py           #   DepDefuseAnalysis (RAW/WAR edges from LoopIR awaits)
      reg_band.py             #   RegBandAnalysis (VALIDATE W, B3)
      gr_increment.py         #   GrIncrementRegions (placement only; §4.4)
    passes/                   # one class per module (each a Pass subclass)
      base.py                 #   Pass base class
      reg_gen.py  tokens.py  collect_pending.py  placement.py  apply_marks.py
      pipeline.py             #   pipeline() (the "backend" pass order) + run_pipeline()
    emit_plan.py              # plan_block: finalized GIR block -> ordered EmitActions (pure)
    verify.py                 # verify_gir (G-* invariants, incl. G-MARK payload schema)
    render.py                 # render_gir / gir_counts (debug)
```

(The whitepaper's earlier single-`analysis.py` sketch was split into `analyses/` + `passes/`
packages — one class per module — during implementation; `transfer_size` and `guard_site` are the
only §6-named analyses not yet built, deferred to R4/wider-vector work.)

---

## 13. Open items to pin next (in order)

1. **LoopIR `srcs`/`dsts` change** (§2.4): a coordinated LoopModel edit = a contract
   **version bump** (own the "frozen except this" contradiction explicitly). Land it FIRST so
   `loopir_to_gir` is a 1:1 copy, not a build-then-refactor-twice. Confirm scope + owner.
2. **drain0 JOIN — analysis value, NOT a phi def** (§5.1 Fact 2, T3): the two drain0
   predecessors (steady exit / prologue short-loop skip) leave the same read-hop gen (pinned
   drain origin, Fact 1) → nothing to merge; the copy-hop entry differs but is resolved by
   `gen_reaching`/`resolve_phi_entry` as a **derived merged value, minting no copy-hop `Gen`
   def in the drain** (the swap is scaffold-owned, so there is no in-GIR copy use to justify a
   def — Fact 3; a def would dangle under strict G-GEN-SSA §4.7). `resolve_phi_entry` handles
   this non-loop join at analysis granularity. Validate on the short-loop (`T ≤ M`) case.
3. **6-axis Coord** (§4.1): dict over the 6 modes — how Tiles declare their spanned subset
   (esp. TDMSplit region modes), made explicit on the Tile, not inferred from operand name.
4. **`Ref` register grouping** (§4.2): (group **index**, slot); W READ from LoopIR slot `mod`
   (B3), `reg_band` validates only. Verify on a `VgprPartition=2` A/B case (mixed-radix
   `Expr.terms`) before declaring resolved.
5. **`guard_site(kind)` granularity** (§6): one generic analysis keyed by `kind`, **minus
   StreamK** (T2 — not derivable from GIR; scaffold-located).
6. **Wave-spec reservation** (§8.2): reserve `role` + cross-wave dep now, pass later.
7. L3 full spec (§9) — after 1–6 settle.

### 13.1 R4 breakage scan — scaffold features vs the GIR-owned body

R4's real purpose (now that GIR owns prologue/steady/drain) is ensuring old scaffold
optimizations don't silently break against the GIR body.  Scan results:

- **SAFE-GATED** (rejected in `Solution.py`, can't reach GIR): non-gfx1250, non-WMMA, non-bf16
  (transitively blocks f8/f6/f4 packing, ConversionInst, MX, complex), SIA3.  **UseSubtileImpl**
  is a PERMANENT reject — subtile is a separate mainloop (its own `Components/Subtile` path); the
  two are mutually exclusive by design, not a "not-yet".
- **SAFE-HANDLED** (structural / order-independent): waits (`waitLWCode`/`syncCode`),
  `perIterLocalWrite`/`perIterGlobalRead` accumulate; swap + gr_inc GIR-owned (scaffold's bundled
  Modules suppressed); StaggerU WrapU inside `tdmIncrementAB`; open/closeLoop + trip-count
  branches wrap the body; the **GSU==1 noLoadLoop guard structurally NESTS around GIR**
  (`GSUOn.noLoadLoop` wraps `writer.noLoadLoop` → `noLoadLoopBody` → GIR), so no `gsu_guard` Mark
  is needed for it yet; InitCIterWmma (RegionClonePass); PGR≥3 drain-step mapping; 1LDSBuffer.
- **NEWLY GATED (R4)** — features that inject into / reshape the GIR body but the leaf emitters
  don't yet model, so they'd emit *silent wrong numerics*.  Now **rejected** in `Solution.py`
  (loud reject beats miscompile) until GIR models each:
  - **InnerUnroll>1** — the wmma leaf emits one wmma per (m,n,k) with `iui=0`; IU>1 would drop the
    `iui>0` WMMAs.
  - **DirectToVgpr** — the wmma leaf doesn't thread the DTV double-buffer `vregSetIdx` (would read
    the wrong register set).
  - **LocalSplitU>1** — GIR body is single-split.
  (The read leaf already raises on HalfPLR/Sparse — a loud crash, also safe.)
- **DEFERRED to task #94** (tail #94 / edge): a tail loop (`K%DepthU≠0`) runs the scaffold's
  `localReadDo`, but the prologue fork skips `localReadDo` so tail read-pointer state is stale;
  edge (`M%MT0≠0`/`N%MT1≠0`) has no GIR fork.  Not gated (the bring-up matrix uses exact tiles);
  gate or model when the tail/edge forks land.
- **`gsu_guard` / `Tag` scaffold-anchor contract**: DEFERRED until GSU support is actually added
  (the noLoadLoop guard nests structurally today; a `gsu_guard` Mark is only needed for guards GIR
  must *locate* itself, e.g. PGR-last-iter — none required yet).

Resolved this round:
- **T1 framing (§1.2).** Now leads with the formal grounding — **closure + gen-SSA-over-a-real-CFG
  + testability** — and casts the "GEMM DSL / whole-GEMM" view as a *weakened, developer-facing*
  mental model (how to work in GIR), explicitly not a claim that GIR structurally carries the
  wg/wave layer (it delegates that to Components / grows via the §8.1 promotion path).
- **PersistentLoop/PAP (§8.4).** Backed by θ paper §7.2: outer `time` level (`resort`) + `retime`;
  a LoopModel emission change GIR absorbs via a nested loop-carried Gen — deferrable, no GIR
  redesign (task #113). §5.1 Fact 3 conditioned on non-PAP; B3 now cites paper Lemma 3c.
- **Interface redesign (analysis→placement→apply).** A `Mark` is no longer dropped at a fixed
  point by a pass. **Region analyses** (`swap_regions`, `gr_increment_regions`, `guard_site`)
  return `PendingMark`s carrying a legal `Region` (§4.4); a **placement** stage picks the point;
  a single **apply** pass is the only IR mutator — mirroring StinkyTofu's waitcnt/liveness shape
  (analysis returns a range, a later stage picks the point, one emit mutates). Analyses never
  mutate the IR (§6/§7).
- **B2 drain copy swap** — final resolution: the descriptor swap's consumer is the **tail
  copy** (out of single-tile scope; the drain span alone showed none only because the tail is
  skipped for K%DU==0). So it is **scaffold-owned** until the tail (#94)/PAP (§8.4) is modeled;
  **G-GEN-SSA kept STRICT** (every Gen has a use). See the "Latest round" note below — this
  supersedes an intermediate "use-free Gen" draft.
- **B1** the swap boundary case is now the region seed over `walk_rpo` (drain0→drain1 +
  steady→drain0 fall out; §7.1). **T3** drain read-hop proven step-constant via `_pin_iter`;
  drain0 copy-hop join is a real select, **not max-over-preds** (§5.1 Fact 2). **B3** W from
  LoopIR `slot.mod`, `reg_band` validates only. **T2** StreamK dropped from `guard_site`. **T4**
  per-kind `Mark` schema (G-MARK) + gr_increment = structural-placement/layout-magnitude. Nits:
  drain count `M` from `_peel_depths`; §5 copy shows WAR `deps`. Earlier: groups index-not-name;
  VW out of GIR.

Latest round (linus 4th pass): **B2 re-settled against real asm** — the drain descriptor swap's
consumer IS the tail copy (comment `LoopModelLowering.py:387-390` correct, not stale; the tail
is conditionally skipped for K%DU==0, which is why the drain span alone showed none). So the
swap is **scaffold-owned** until the tail (task #94) or PAP (§8.4) is modeled (Option B);
**G-GEN-SSA kept STRICT** (every Gen has a use — no "use optional" relaxation), §5.1 Fact 3 /
§4.7. **§8.4 softened**: persistent needs no new node but DOES relax G-CFG/G-TERM single-loop +
`gen_reaching` per-back-edge; R0 must enumerate back-edges by dominance. §5 example: drain
copy-swap shown as scaffold-owned, not a GIR Mark.

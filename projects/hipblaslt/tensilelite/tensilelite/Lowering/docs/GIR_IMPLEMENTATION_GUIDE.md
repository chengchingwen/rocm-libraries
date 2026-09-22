# The GIR lowering — implementation guide

How `Tensile/Lowering/` actually works, module by module and analysis by analysis.

Peer of `Tensile/LoopModel/THETA_MODEL_GUIDE.md` (the layer above). Together they describe the whole
`UseLoopModel` path.

It also complements `lowering-design.md`, the design whitepaper from #107 — **that document says
what GIR is *for* and why it exists; this one says how the code works.** Where they disagree, the
code wins and this file should be corrected.

---

## 1. The idea, in plain terms

Skip the vocabulary. Here is the division of labour.

**LoopIR says what the schedule *is*.** Which operations exist, in what order, how far ahead each
runs. It is a tree, it is rolled (one line per kind of operation), and its buffer indices are
*formulas* — `(k+2) mod 2` — not numbers.

**GIR is where you work out the physical consequences.** Four questions, and they are all the same
kind of question:

- which of the two LDS buffers does *this* access actually touch?
- at which instruction does the LDS read pointer have to be flipped?
- at which instruction does the global read address have to advance?
- where do barriers go?

Every one of those is *"one physical register or resource must hold the right value at this
point"* — a **compiler dataflow problem**, not a scheduling problem. And you cannot ask "does this
value reach that point?" of a tree whose indices are symbolic formulas. So GIR is a **control-flow
graph with SSA**, the three pipeline stages are unrolled into real blocks, and every buffer
generation is an explicit named value with phis at the loop header. Then all four questions become
textbook fixpoints.

That is the whole reason there are two IRs. Not layering for its own sake: the questions genuinely
change form.

### The one pattern everything follows

```
    analysis  ──►  plan  ──►  placement  ──►  apply
   (IR frozen)             (IR frozen)      (the ONLY mutator)
```

**No analysis touches the IR.** Each returns a *plan*: "a pointer flip has to happen somewhere
between here and there". A separate stage picks a concrete point inside each range. One pass —
`ApplyMarksPass`, and only it — inserts.

Three layers rather than one, because **an insertion has a legal *range*, not a correct point.** A
read-pointer flip may sit anywhere after the last use of the old buffer and before the first use of
the new one. Baking an index into the analysis is premature realization, and it is the mistake the
first version of this code made.

### The recurring problem shape

Two of the analyses — the LDS read pointer and the TDM write descriptor — are literally the same
problem for a different register: **one register, a value it must hold at each access, and exactly
one kind of instruction that changes it.** That is the classic code-motion pair (a forward
availability fixpoint and a backward anticipation one), so both are solved by *one* module,
`value_placement.py`, and the rule cannot drift between them.

---

## 2. Layer map and module inventory

```
θ ──emit_mainloop──► LoopIR tree + ledger        (LoopModel)
        │  loopir_to_gir.lower_to_gir            unroll the 3 stages into a CFG, attach Gen SSA
        ▼
      GIR  ──run_pipeline──► finalized GIR       reg → tokens → collect → place → apply → scaffold
        │  gir.plan_block                        pure: CFG → an ordered list of EmitActions
        ▼
   gir_to_rocisa + leaves.py                     the thin rocisa adapter
        ▼
   KernelWriterAssembly
```

| module | lines | owns |
|---|---:|---|
| `loopir_to_gir.py` | 624 | the tree→CFG lowering; Gen SSA construction; `build_gir` (R-ONCE) |
| `gir_to_rocisa.py` | 500 | the thin rocisa adapter; fence + Mark realization |
| `gir/nodes.py` | 424 | the node set — `Tile`/`Gen`/`Ref`/`Move`/`Mma`/`Mark`/`Block`/`Program` |
| `leaves.py` | 334 | per-coordinate rocisa emitters (one WMMA tile, one LDS read) |
| `gir/analyses/lds_hazards.py` | 326 | RAW/WAR/WAW over the shared space, with trip distance |
| `gir/analyses/swap_regions.py` | 294 | where pointer flips go |
| `gir/verify.py` | 257 | the G-* well-formedness invariants |
| `gir/analyses/gr_increment.py` | 252 | where address advances go |
| `gir/analyses/value_placement.py` | 237 | the shared AVAIL/ANTIC fixpoint pair |
| `gir/emit_plan.py` | 202 | pure CFG → `EmitAction` list |
| `gir/analyses/fence_regions.py` | 174 | minimal fence cover over the hazards |
| `gir/render.py` | 370 | debug view: blocks/terms plus the θ-facts header (regions, Φ fuse, agent distribution), per-instruction dep KINDS, LDS tokens, §2.2 spans, and the short-arm verdict |
| `gir/passes/tokens.py` | 131 | stamp each LDS Move's completion token |
| `gir/analyses/gen_reaching.py` | 129 | which concrete generation each access uses |
| `gir/analyses/cfg.py` | 106 | dominators, back-edges |
| `gir/refs.py` | 102 | "which operand(s) does this Move's shared hop belong to" |
| `gir/passes/scaffold_map.py` | 82 | generic guard kinds → TensileLite label hints |
| `gir/region.py` | 67 | `Region` + `PendingMark` — the insertion model |
| `gir/analysis.py` | 59 | `Analysis` base + lazy cached manager |
| `gir/analyses/reg_band.py` | 59 | validate the register rotation width |
| `gir/passes/placement.py` | 56 | resolve each Region to an anchor |
| `gir/passes/pipeline.py` | 55 | the pass order, as data |
| `gir/passes/apply_marks.py` | 54 | the only IR mutator |
| `gir/analyses/dep_defuse.py` | 53 | index the edges carried from LoopIR |
| `gir/passes/reg_gen.py` | 36 | register slot authority (validation today) |
| `gir/passes/collect_pending.py` | 34 | run the region analyses, stash their plans |

---

## 3. The node set (`gir/nodes.py`)

### Nouns

| node | is |
|---|---|
| `Tile` | *what data* — an operand, a space, a semantic coordinate. Interned. |
| `Gen` | *which generation* — one buffer generation of a ring, an SSA value |
| `Ref` | *one access* — a `Tile` at a `Gen`, plus `gdelta` (trip offset) and, for peel refs, a concrete `abs_gen` |

### Verbs

`Move(srcs, dsts, deps, token)` — one hop along a trajectory. `Mma(srcs, dsts, coord)` — the compute
op. That is all. Everything else is control flow or a `Mark`.

`Move.deps` are the RAW/WAR edges **threaded from the LoopIR awaits by the lowering** — the
inherited half of §11. `Move.token` is the completion token, stamped later by `TokensPass`; `None`
until then, and a *fact*, not a count.

### `Mark` — a request for a scaffold-owned action

Closed set: `phase_boundary`, `fence`, `swap`, `gr_increment`, `gsu_guard`. Each carries an `at`
payload whose schema `verify` enforces per kind. Marks are what the analysis→apply pipeline inserts;
they are not data movement.

### Control flow

`Goto`, `CondGoto`, `CondChain`, `Return` — every terminator names its targets explicitly, and
`Block.preds`/`succs` must agree with them (checked). `GenPhi` at a block head and `GenXfer` on the
back-edge carry the generation across trips: **that is what makes the loop-carried relation a
first-class SSA fact instead of arithmetic hidden in a slot expression.**

`Block.phase` ∈ `prologue | steady | drain{n} | tail`, and the phase *is* the block label.

---

## 4. The lowering (`loopir_to_gir.py`)

`lower_to_gir(theta, mainloop, loop_copies)` walks the LoopIR **exactly once** and unrolls the three
stages into one CFG. It is the tree-walk half of the old walker with **none** of the physical-state
reconstruction — no pointer seeding, no swap heuristics, no token stepping. Those all became
analyses.

What it attaches:

- **one loop-carried `Gen` per shared-staged operand**, ring = that operand's LDS depth. Steady refs
  use it with a concrete `gdelta`; the steady header gets the phi and the back-edge the `GenXfer`.
- **peel refs carry a concrete `abs_gen`** instead of a `Gen` (§5.1 Fact 1): a prologue or drain step
  is pinned to a known chunk, so its generation is a number, not a phi-resolved value.
- the generic `Pred`s on terminators, label-free.
- `Program.meta` — the θ facts L3 and the analyses need (§11).

Two shape helpers worth knowing: `_flatten` walks the rolled tree into straight-line block bodies
under an environment binding the pinned loop counters, and `_drain_steps` splits the drain peel into
its individual `Bind` steps, each becoming its own block.

`build_gir(theta, mainloop, params, pipeline)` is the **R-ONCE** entry point: lower, then run the
pass pipeline, exactly once per kernel. `KernelWriter` caches the result; the three stage forks then
only *ask* for their block. That caching is the coupling the whole rearchitecture removed — each
fork used to rebuild the LoopIR.

---

## 5. The analysis framework (`gir/analysis.py`, `gir/region.py`)

An `Analysis` is **pure**: it reads a `Program` and returns a result object. `AnalysisManager`
caches by `(analysis, prog.version)`; any mutation bumps the version and invalidates.

Results are **objects with a query API**, never raw dicts — `reach.of(inst, ref)`,
`reach.xfer(block, gen)`. Callers never see the internal keying (which is identity-based, so the
result holds `prog` alive to keep it valid).

`Region` + `PendingMark` are the insertion model:

```python
Region(after=<the last access using the old value>,
       before=<the first access needing the new value>,
       policy=EARLIEST | MIDPOINT | PINNED)
PendingMark(mark=Mark(...), region=Region(...))
```

Anchors are **opaque node references, not indices** — deliberately, so that inserting one Mark
cannot invalidate another Mark's region.

---

## 6. The analyses

### 6.1 `cfg.py` — Dominators, BackEdges

Back-edges are found **by dominance** (an edge `src→target` where `target` dominates `src`), not by
a `blk.loop` flag. That is what makes the persistent-loop relaxation later a no-op rather than a
rewrite.

### 6.2 `gen_reaching.py` — GenReaching

SSA reaching-value over the CFG: resolve each `Ref.gen` phi to its block-entry value, add `gdelta`,
mod the ring. Each back-edge's `GenXfer` gives the next trip's entry value. A peel ref's `abs_gen`
resolves directly.

**This is the analysis everything else stands on** — tokens, hazards, and both pointer analyses all
ask it "which concrete buffer does this access touch".

### 6.3 `value_placement.py` — the shared fixpoint pair

The single implementation of "where do the defs of a one-register resource go":

- **AVAIL** (forward) — `avail_out[b]`: the value the register holds *leaving* `b` — the requirement
  of `b`'s last access, or a pass-through if `b` has none.
- **ANTIC** (backward) — the value the next access *demands*.

A def (a Mark) belongs exactly where supply ≠ demand. Both pointer analyses consume this, which is
the point: the rule cannot drift between them.

### 6.4 `swap_regions.py` — pointer flips

The physical LDS read pointer and the TDM write descriptor are **single registers** whose selected
generation must equal the generation every access names. A swap is the only thing that changes one,
so placement is a reaching-value problem. Its own docstring records why the previous version was
*not* a dataflow and what that cost (a double swap, and a wrong frame in the drain — #177).

### 6.5 `gr_increment.py` — address advances

Same shape, different register: each operand's global read address is a loop-carried pointer holding
the chunk the next copy will fetch. This file supplies only what is specific to that register —
which copies consume the descriptor, the absolute chunk each fetches, and how a chunk re-frames
across the back-edge. The fixpoint is `value_placement`'s.

### 6.6 `lds_hazards.py` — the independent hazard derivation

RAW/WAR/WAW over the shared space, with **trip distance**: an access at ring offset `g`, trip `v`,
advance `adv` touches `(base + v·adv + g) mod S`, so a producer and consumer collide at
`d = (g_p − g_c) mod S`. Cross-block pairs are decided conservatively (`distance=None`,
`cross_block=True`); `_provably_disjoint` prunes only peel↔peel pairs in absolute frames.

**Contains zero references to θ.** It derives the ordering facts from the CFG and the SSA alone —
see §11.

### 6.7 `fence_regions.py` — the minimal cover

Turns hazards into fence placements by **set cover**, not one-fence-per-hazard. `_admissible_slots`
computes, per hazard, which body positions would separate it; `_already_fenced` drops hazards an
existing fence covers (reachable-from-producer **and** dominates-consumer); a greedy cover picks the
rest. Keyed by index, because a `Hazard` holds a `Move` and is unhashable.

### 6.8 `dep_defuse.py` — the inherited edges

Indexes the `Move.deps` threaded from the LoopIR awaits into a queryable result, so the tokens pass
and L3's WAR-forwarding read one source of truth. **Pure — it only reads what the lowering put
there.** This is the *inherited* half of §11.

### 6.9 `reg_band.py` — validate, never decide

Returns the single register ring width seen per `(operand, group)`, and **raises** if a group's reads
disagree. W is a LoopModel decision (θ, Lemma 3c); deciding it here would be a second authority that
can disagree with the decoder.

---

## 7. The passes and the pipeline

```python
def pipeline():
    return [RegGenPass(),              # register slot authority (validation today)
            TokensPass(),              # stamp each LDS Move's completion token
            CollectPendingMarksPass(), # run the region analyses -> prog.pending
            PlacementPass(),           # resolve each Region to a concrete anchor
            ApplyMarksPass(),          # THE ONLY IR MUTATOR
            ScaffoldMapPass()]         # advisory TensileLite labels; last, independent
```

Pass order is **data**, in one file, modelled on StinkyTofu's backend file. There is a single
pipeline; a test that wants one pass in isolation runs that class directly (the analyses are pure)
rather than assembling a variant — so per-phase pipeline variants cannot accumulate.

**`CollectPendingMarksPass`** runs `SwapRegions`, `GrIncrementRegions` and `FenceRegions` and stashes
their plans. Fences needed no new machinery precisely because they are a region analysis like the
others — and their regions are `PINNED`, since the cover already chose the point.

**`PlacementPass`** — policy travels with the Region, because it differs by what the Mark mutates:

| policy | applies to | lands |
|---|---|---|
| `EARLIEST` | every TDM descriptor change — `gr_increment` **and** the copy-hop `swap` | immediately after `region.after`; every cycle before the load it feeds is overlap |
| `MIDPOINT` | the LDS read pointer | midway between the last use of the old generation and the first use of the new |
| `PINNED` | fences | where the cover put them |

**`ScaffoldMapPass`** is the *one* place generic guard kinds (`peel_validity`, `first_touch`,
`readahead_suppress`) become TensileLite label hints (`toPGR1`, `LoopEndL`, `NoGlobalLoadLoop_k`).
The label is **advisory**: a backend whose own loop scaffold already emits the compare routes each
arm to the matching label; a from-scratch backend lowers the predicate and ignores it. All
TensileLite naming is concentrated here, so skipping this pass is how you get a clean backend.

---

## 8. `verify_gir` — the shape gate

Runs after the pipeline. Raises on the first violation.

| invariant | catches |
|---|---|
| `G-TERM` | a missing terminator; a target that doesn't exist; declared preds/succs disagreeing with the actual CFG |
| `G-CFG` | more than one prologue or tail; a drain count ≠ the lowering's `M`; a short-loop record out of sync with `M`; more than one reduction-loop header (the tail's header is excluded **by phase**) |
| `G-GEN-SSA` | a `Gen` with two phi defs; a def with no use (a dropped consumer); a use with no def |
| `G-NOITER` | a chunk counter leaking into a coordinate — the cross-trip relation belongs in the phi |
| `G-SEMANTIC` | any physical attribute (vgpr, address, vector width, colour) attached to a node |
| `G-OPERANDS` | an `Mma` register source with no producing `Move` |
| `G-MARK` | an unknown Mark kind, or an `at` payload missing a field its kind's schema requires |

The `G-MARK` swap check is worth singling out: a **read**-hop swap must name an `operand`, a
**copy**-hop swap must name a `unit` (the Φ member tuple). Requiring the *right* one, not merely
"some name", is what stops a fused movement's other members from going unnoticed.

---

## 9. Layer 3 — GIR → rocisa

**`emit_plan.plan_block(prog, phase)`** is the **pure** half: it walks a finalized block body and
returns an ordered list of `EmitAction`s with every index, generation and token already resolved. It
imports nothing from rocisa, so the correctness-critical index/order logic is unit-testable
standalone.

**`gir_to_rocisa.py`** is the thin adapter: action → instruction. The order and all indices are
GIR's; this file only maps.

**`leaves.py`** holds the order-**agnostic** per-coordinate emitters — the rocisa for one WMMA tile
or one LDS read is identical regardless of loop order; only *when* it is emitted changes, and that
is the plan's business.

The concern split: **GIR decides what/where/why** (a swap is a generation transition at a point);
**L3 decides how** (that swap becomes `tdmSwapLdsOffset` or `localReadSwapOffsets`). GIR is the
single source of truth for swap and increment placement — TensileLite issues neither on this path.

---

## 10. What crosses the θ→GIR boundary

`Program.meta` carries, from θ: the reduction-chunk level name, reduction/free mode names, inner
order and extents, `per_region_completion`, `fuse_groups`, the movement units and their region
counts, and `agent_distributed` per operand. Plus `M` (the peel depth) and `short_loop`.

**GIR never re-derives a schedule decision.** It lowers what θ decided and then runs its own dataflow
over the result. The line is:

> θ decides *which op-class instances exist, at which level, in which order.*
> GIR decides *where the supporting machinery goes.*

Every pass respects it: not one creates or moves an op-class instance. They insert Marks, stamp
tokens, and validate.

---

## 11. Two sources of ordering information — and why both exist

```
inherited:    LoopIR awaits ──► Move.deps ──► DepDefuse ──► tokens pass + L3 WAR forwarding
independent:  CFG + Gen SSA ──► LdsHazards (zero θ refs) ──► FenceRegions ──► fences
```

Both are live, and the second exists because the first turned out not to be sufficient. That was
**#197**: the inherited path expressed ordering through memory tokens, a token named exactly one
buffer generation, and at one prefetch/read-ahead cell the copy and the read never named the same
one — so the backend derived **zero** barriers. The ledger was correct at LoopIR; the guarantee
simply did not survive the lowering.

The independent derivation (#204/#205) closed it, and it is what let GIR take barrier placement off
the scaffold entirely (#206).

**The gap that remains:** nothing re-runs the LoopIR ledger check on GIR, so for families where only
the inherited path exists — the register ring, the warm-up existence requirements — the lowering is
*trusted*. `decoder_proto/ledger_vs_gir.py` (#214) cross-checks the one family where both
derivations exist; it reports 26/26 agreement today and carries a negative control so an
always-passing run cannot be mistaken for a working check.

---

## 12. Invariants

1. **Analyses are pure.** No analysis mutates. `ApplyMarksPass` is the only mutator.
2. **A Mark has a range, not a point.** Analysis plans, placement chooses, apply inserts.
3. **Anchors are node references, not indices** — so Marks cannot invalidate each other's regions.
4. **One fixpoint implementation** for both pointer resources (`value_placement`).
5. **`GenReaching` is the single answer** to "which buffer does this touch"; tokens, hazards and
   both pointer analyses all ask it.
6. **W is validated, never decided** (`reg_band` raises on disagreement).
7. **All TensileLite naming is in `ScaffoldMapPass`.** Skipping it yields a clean backend.
8. **`build_gir` runs once per kernel** and is cached; the stage forks only read their block.

---

## 13. Known gaps

| gap | task |
|---|---|
| Ring-1 copy carries no WAR | #116-adjacent |
| `Move`'s docstring says "one hop **down** the memory hierarchy"; a store reverses it, and `G-OPERANDS` assumes a register Tile is produced by a `Move` (a store's source is the accumulator, produced by an `Mma`) | #116 |
| No `epilogue` block phase | #116 |
| Drain-multiplicity matrix + a separate tail-loop block | #148 |
| `KernelWriter` does not follow GIR branches (openLoop compares still emitted) | #150 |
| The short-loop arm is not lowered to GIR at all | #183 |
| Fused instance coord names only one member's region axis | #170 |
| `VgprPartition > 1` reaches θ but the group never reaches GIR | #118 |
| The lowering is trusted for every family except LDS (§11) | #214 covers LDS only |

# θ: put each fact on the object that owns it — plan (draft, for discussion)

> **STATUS: implemented.** `Trajectory`/`Global`/`Shared`/`Fragment` are the model,
> `DepthMap`/`SHARED_GROUP` are gone, the retime offsets sit on the placement each movement fills,
> and `ρ`/`resort`/`Φ`/`quantum` are spelled `AgentAssignment`/`axis_assignments`/
> `instances_per_instruction`/`coverage`. This file is kept as the *rationale* — the defects below
> are stated in the tense of the code before the change. For the as-built picture and the
> old-name → direct-name glossary, read `ARCHITECTURE.md`.
>
> Not done: the `operand_functions` count, which this plan expected to fall as facts moved onto
> placements, rose instead. Free functions `f(theta, operand, …)` are still the main way an
> operand's properties are asked for.

> *"there are too much class and field that carry the info it shouldn't. For example, the number
> lds buffer is on Operand, but it should be on the Hop that target shared, or, the Hop itself
> should be a full path that carry the list of transfer and mark that shared is a ring of N, also
> applied to register, so maybe Fragment should be a build block of Hop and need a Fragment parity
> of LDS. Also rho and resort and agent is something that should have clear buildup."*

Working document. §6 is the list of choices that need a call before code moves.

This supersedes the earlier "authoring verbs" draft, which treated the DSL as an additive surface
over the current data model. That was the wrong altitude: **θ is hard to author because the facts
are on the wrong objects**, and the verbs would only have papered over it. Authoring and
single-sourcing are now one piece of work.

---

## 1. The organizing idea is Definition 1, not taste

The paper already says where these facts live, and our layout is what diverges.

> **§2.2** — *"A buffer moves through the hierarchy along a **trajectory**: `global --[copy]-->
> shared --[read]--> register`. A length-`L` trajectory carries `L` independent pipelining offsets
> **and buffer depths** against `L` completion classes."*

> **Definition 1 (§2.6)** — *"For a placement `Λ` with **rotating set** `Q(Λ)` … the depth is a map
> `S_Λ : Q(Λ) → ℕ₊`. The units differ by space: for a **shared** placement `Q(Λ)` is a set of
> **modes**; for a **register** placement it is a set of **groups**."*

Three consequences, and each names a defect we already have:

1. **Depth is per *placement*, not per operand.** We put `lds_buffers` on `Operand`.
2. **The shared and register rotating sets are different shapes** — modes vs. groups. We have one
   `DepthMap` keyed `(operand, region, group)` with a magic `SHARED_GROUP = "shared"` string
   (9 call sites) that squeezes the shared placement into the register placement's key space. That
   sentinel *is* the divergence, made visible.
3. **`Fragment` is the register placement's rotating-set descriptor**, so it belongs to the hop that
   targets register — and the shared placement needs its own, which by Def 1 is *not* the same
   class (a set of modes, not a partition of a grouping mode's values).

`Hop` is already a placement descriptor in everything but name: it carries `split` (the shared
placement's region count), `quantum` (the movement fold), `phi_width`, `rho_span`, `vector_elems`.
Putting depth and the rotating set there finishes the job it already has.

---

## 2. Measured: how much of `Operand` is placement facts

Reads across `LoopModel/` + `Lowering/`, 2026-09-11.

| `Operand` field | reads | owner under Def 1 |
|---|---:|---|
| `fragment` | 64 | the **register** hop |
| `lds_buffers` | 12 | the **shared** hop |
| `frag_elems` | 6 | the register placement |
| `region_axes` | 4 | the shared placement |
| `wave_region_span` | 3 | the shared placement |
| `region_span` | 1 | the shared placement |
| `free_split` | **0** | dead — delete |
| `role` | 25 | the operand |
| `elem_bytes` | 5 | the operand |
| `free_mode` | 1 | the operand |

Six of twelve fields are placement facts; one is dead; `free_mode` has a single reader.

### The instruction count lands here too

A waitcnt analysis needs "how many instructions does this movement become". Today that number has
**two** homes in different units:

- `Lowering/lds_geometry.read_fragments()` — ds_reads per *act*, from `blockWidth`, `bpeDS`,
  `lrvw`, `inputPerThUnroll`, `matrixInstK`; assertion-checked against the register span and
  `MatrixInstK`. This is the real one: `leaves.py:543` iterates it.
- `LoopModel/traversal.plan_transfers()` — a per-*kiter, whole-operand* count from
  `Hop.vector_elems` / `frag_elems` / `elem_bytes` / `reg_bytes`. Disjoint inputs, different unit,
  one caller: `render.py:335`, a display table.

`read_fragments` computes a property of **the transfer**. If the hop is the transfer, the count has
an obvious single home and the second derivation goes away. The waitcnt prerequisite falls out of
this reorganization rather than needing its own.

*One caveat the count must carry:* an act can emit **zero** instructions — `emitLdsReadTile` returns
early when a folded read is not the group leader (`quantum.emit` false) and for the upper half-wave
under `tileSpanVW`. A count that ignores those over-counts outstanding ops, which yields a **larger**
`n`, which is a **weaker** wait. The suppression rules are part of the count, not a detail.

---

## 3. Target shape — placements are NODES, movements are the adjacency

§2.1 states the primitive: *"a **trajectory** is a buffer's **sequence of placements** linked by
movement operations."* The placement is the node; the movement is the edge, **derived** from two
consecutive nodes. So the path is a list of placements, not a list of `(src, dst)` edges:

```python
Operand("A", free_mode="M", role="input", elem_bytes=2,          # what the TENSOR is
        path=Hops([Global(),
                   Shared(ring=2, regions=..., quantum=..., vector_elems=...),
                   Fragment(groups=..., quantum=..., vector_elems=...)]))
```

- **The destination carries the movement that fills it.** θ already keys it this way:
  `off_map[(operand, role, level)]` has `role ∈ {copy, read}`, and role *is* the destination —
  `HOP_ROLE[(GLOBAL, REGISTER)] = READ` precisely because it fills register. So `vector_elems`,
  `quantum`, `phi_width`, `rho_span` and the instruction count sit on the placement being filled.
- **`Shared`** is the "Fragment parity of LDS" — Def 1's shared branch, a set of **modes** plus a
  depth — and the home for the region facts (`region_axes`, `region_span`, `wave_region_span`)
  currently loose on `Operand`.
- **`Fragment`** *is* the register placement (Def 1's register branch: **groups** plus a depth), so
  the name is already right; it just moves off `Operand` and onto the path.
- **`Global()`** carries nothing in the forward direction. The epilogue is
  `Hops([Fragment(), Shared(), Global()])` — direction falls out of position.

### What the edge encoding costs today (measured 2026-09-11)

| | count |
|---|---:|
| linear searches of the hop list | **31** |
| endpoint comparisons (`dst == REGISTER` 24, `dst == SHARED` 17, `src == SHARED` 13) | **54** |
| `hops[-1]` / `hops[0]` | 18 / 7 |

And **three near-duplicate helpers that all mean "the register-filling movement"**, with three
different predicates:

| helper | predicate |
|---|---|
| `traversal._local_read` | `not is_bulk and dst == REGISTER` |
| `traversal._register_read_hop` | `… and **src == SHARED**` |
| `emit._register_hop_coverage` | `… and **quantum is not None**` |

On a DirectToVgpr operand (`Global → Register`) the first returns the hop and the second returns
`None`. Three answers to one question, existing *only because the node is not addressable*. All
three become `path.fragment`.

Two further consequences:

- **DirectToVgpr becomes structural**: `Hops([Global(), Fragment()])`. The missing shared placement
  is the absence of a node, not a `dst == SHARED` search that silently yields `None`.
- **`path_direction` stops being able to fail.** It exists today to catch `dst(n) ≠ src(n+1)` — a
  disagreement only possible because the endpoints are stored twice. With nodes there is nothing to
  disagree; direction is the space ranks read along the sequence.

- **`Fragment`'s own cleanup travels with the move**: `parts` values are always `1` (only its
  *length* carries information), and `group_broadcast(group)` ignores its `group` argument while
  `_ring_broadcast` passes one.
- **`DepthMap` dissolves.** Each placement carries its own depth, so the `(operand, region, group)`
  key and the `SHARED_GROUP` sentinel both disappear.

---

## 4. Why this is byte-identical, and where it genuinely is not

A field relocation with every reader updated cannot move output — it is the same value read from a
different attribute. The gate is not there because a diff is expected; it is there because it turns
"reason about 64 call sites" into "know in 60 seconds", which is what makes this tractable at all.

Two places are *not* pure relocation and need their own commits:

1. **`lds_buffers()` holds a derivation in the same function as the field read** —
   `operand.lds_buffers or max(1, off(copy, chunk))`. Moving the field means editing that function,
   with the fallback one line away. Whether the fallback survives is a **decision** (§6 Q4), not a
   consequence of the move.
2. **`DepthMap.groups_of` builds from a set comprehension**, so its iteration order is
   hash-randomized. Both consumers are order-insensitive today (one sums, one formats an error), but
   splitting the map touches exactly this code, and the ledger already had to sort set iteration
   away once.

**The gate**: `/Workspace/wmma_dsl_baseline`, 8488 kernels over all ten `loopmodel_*.yaml` at
`7455caa3f55`. `python3 ref_asm/capture_loopmodel_asm.py check <sweep> <baseline>` — see
`ref_asm/LOOPMODEL_BASELINE.md`. Run it per commit, keep relocation and derivation commits
separate so a diff is attributable.

---

## 5. Steps

Each separately committable, each gated.

1. **Delete `free_split`** (0 readers). The smallest possible proof the gate works end to end.
2. **`path` becomes an object.** Wrap the hop list; no fields move yet. Pure addition.
3. **Move the register facts onto the register hop** — `fragment`, `frag_elems`. 70 readers, one
   mechanical change.
4. **Introduce `SharedRing` and move the shared facts onto the shared hop** — `lds_buffers`,
   `region_axes`, `region_span`, `wave_region_span`. The `lds_buffers` fallback stays *exactly* as
   it is in this step; changing it is step 8.
5. **Dissolve `DepthMap`** into the two placements' own depths; delete `SHARED_GROUP`.
6. **Give the hop its instruction count**, sourced from `read_fragments` plus the suppression rules;
   retire `plan_transfers` or make it call the same helper.
7. **`Fragment` cleanup** — `parts` to a group count, `group_broadcast` to honour its argument (this
   one *can* move output: `_ring_broadcast` is a consumer, so it is a derivation commit).
8. **Validation and the authoring verbs** — the earlier draft's content, now that each fact has one
   home to validate. Including the `lds_buffers` fallback decision and the `grouping_mode`
   existence check.

Steps 1–5 are relocation. 6–8 are derivations and each can legitimately show a diff.

---

## 6. Decisions needed

**Q1 — `Hop` as the placement descriptor, or a placement object hanging off it?**
`ir.Placement` already exists as an IR node, so that name is taken. Options: put the ring directly
on `Hop` (`Hop.ring`), or add `Hop.dst_placement`.
**Recommend: `Hop.ring`** — fewer objects, and `Hop` already carries placement facts.

**Q2 — Is `path` a first-class object, or still a plain list?**
Making it real gives θ's `path` field the home the tuple says it has. It also gives the trajectory
somewhere to answer "which hop targets shared".
**Recommend: yes**, as step 2 — it is pure addition and everything after it reads better.

**Q3 — ρ: per-mode or per-(mode, op-class)?**
`Resort` is *typed* per-mode (`axis, level, extent`, matching §3.3) but *carries* `role` and
`origin` — so the data is already per-op-class and the type disagrees. This is the question sent to
the paper author. It matters because the same "same-agent" predicate gates `inplace = 1`
(Lemma 3c(i)) and wave-vs-block `Await` scope.
**Recommend: make the type say what the data does** (per-op-class), and mark it as ours to revisit
if the author says ρ is genuinely per-mode.

**Q4 — Does the `lds_buffers` fallback survive?**
`unstated -> δ` is a derivation hiding in a field default. Under Def 1 the shared placement should
state its own depth, and an unstated one is arguably ill-formed rather than δ.
**Recommend: keep it in step 4, decide in step 8**, because removing it can change output and
deserves its own gated commit.

**Q5 — Does `frag_elems` move?**
It is the register fragment's element count, so by Def 1 it is a register-placement fact — but it is
also read as an element-type fact in places. Needs a look at all 6 readers before deciding.

---

## 7. Out of scope

- **The folder.** No module is added, moved, split or merged. This is a data-model change inside
  `theta.py` plus mechanical updates at the readers.
- **Behaviour**, except where §5 marks a step as a derivation commit, each agreed first.
- **The gaps in `theta-model-implementation-delta.md`** — the missing WAW family, the dead
  Precondition-Q check, the IR not carrying an agent. Real, and not this.
- **`Lowering/`**, beyond the five private LoopModel names it imports and the `read_fragments`
  hand-off in step 6.

# Fence / Hazard / Memory-Token — Implementation Report

Every analysis, pass, verifier, emitter and StinkyTofu component that participates in deciding
**where a barrier goes** and **what it names**, with pseudocode and rationale.

Scope: `Tensile/Lowering/gir/analyses/`, `Tensile/Lowering/gir/passes/`, `gir/verify.py`,
`gir_to_rocisa.py`, `Tensile/Lowering/tool/`, and the StinkyTofu passes that consume the result
(`/Workspace/mi450-dev/shared/stinkytofu/src/transforms/asm/`).

---

## 0. The pipeline, end to end

```
LoopIR ──lower_to_gir──▶ GIR Program
                             │
   LdsBufferIds  (storage identity -- depends on NOTHING)
        │
        ▼
     FrameMap  (rotation coordinate: which ring, per block)
        │
        ├────────────────────┬─────────────────────────────┐
        ▼                    ▼                             ▼
   FrameHazards         DependenceTokens              RegHazards
   (ordered pairs,       (obligation tokens)      (VGPR rotation twin)
    per frame)                │
        │                     ▼
        └──────────▶  FenceRegions(merge=True) ──▶ PendingMark(fence)
                      (WHERE; tier 1 RAW stamps,
                       tier 2 WAR/WAW rides)
        │                                          │
        │              HoistCopiesPass             ▼
        │              (issue points)        PlacementPass ─▶ ApplyMarksPass
        │                                                          │
        ▼                                                          ▼
   TokensPass  ──stamps Move.token / Move.token_ids──▶     Mark('fence') in body
        │                                                          │
        └──────────────▶ verify_gir (G-TOKEN, G-TOKEN-REUSE) ◀─────┘
                                     │
                                     ▼
                       gir_to_rocisa._emit_fence  →  rocisa SBarrier + MemTokenData
                                     │
                                     ▼
                   StinkyTofu: BuildImplicitDependency → WaitDataflow
                                     │
                                     ▼
                          s_wait_tensorcnt / dscnt / asynccnt  N
```

**The central fact.** A memory token is not decoration. StinkyTofu turns each token id into an
`RegType::LDS` pseudo-register: a producer gets it as a *def*, a consumer as a *use*, a barrier as
*both*. The emitted `s_wait_tensorcnt N` is then a pure consequence of the def-use chain plus queue
positions. So *what a fence names* determines *the emitted wait value*, and naming a token whose
reaching def is an in-flight TDM load drains the memory pipeline.

**The frame rule.** A fence's name and a Move's stamp are ONE fact, resolved through `tokens_for`
at `fm.cut(block)`. A second derivation of "which frame names this block" agrees in the steady block
and **inverts in a drain block**, which legitimately runs at both parities; the fence then names
A#0/B#0 while the reads it exists for are stamped A#1/B#1, and a barrier that defs 0 and 2 orders
nothing against a `ds_read` that uses 1. And a check that derives its own bar from the fence cannot
see it: **when a check cannot fail, it is derived from the thing it is checking.**

Two distinct namings exist and must not be confused:

| naming | built by | keyed on | consumed by |
|---|---|---|---|
| **storage id** — `Buffer(operand, region, ring)` | `LdsBufferIds` | physical LDS storage | hazard reasoning, G-TOKEN V1, G-TOKEN-REUSE |
| **obligation token** — dense int | `DependenceTokens` | hazard equivalence class | `Move.token_ids`, fence `at["tokens"]`, StinkyTofu |

---

## 1. `gir/analyses/cfg.py` — the graph substrate

### `successors(prog)`
```
for each block:
    keep only successor labels that exist in prog.blocks   # drops the 'end' sink
```

### `Dominators.run` → `{label: set(dominators)}`
Classic iterative fixpoint.
```
dom[entry] = {entry};  dom[n] = all_blocks for n != entry
repeat until stable:
    for n != entry:
        dom[n] = {n} | intersection(dom[p] for p in preds(n) if p analysed)
```
Used by `FenceRegions._fenced_in` to pick a block on every path between two ends, and by
`BackEdges`.

### `BackEdge` / `BackEdgeSet` / `BackEdges.run`
```
BackEdge  = (src, header, xfers)          # xfers = the src block's GenXfer list
BackEdges = { (lab, s) : s in succ(lab) and s dominates lab }
```
`xfers` is carried **per back-edge**, not `xfers[0]` — a persistent second loop is then admitted
with no analysis change. `leaving(label)` / `entering(label)` / `is_back_edge(src,target)` are the
query API; `GenReaching` uses `leaving` to apply the generation advance on the wrap.

---

## 2. `gir/analyses/frame_map.py` — which ring does a block stand at?

Replaces `gen_reaching.py`. A **frame** is the rotation coordinate; storage identity is
`LdsBufferIds`' job and does not depend on this.

### `Frame`
An immutable phase vector, `{gen id: phase}`, absent entries reading 0. `of(gen_id)`, `as_dict()`,
value equality and hashing.

### `_rings(prog)` → `{gen id: depth}`
Every Gen the program rotates through, from the phis, the xfers, and every `Ref.gen`.

### `_advance(frame, src, dst, is_back)`
```
for xf in src.xfers:   phases[xf.gen.id] = (phases[gen] + xf.adv) % xf.ring
if not is_back:
    for phi in dst.phis: phases[phi.gen.id] = phi.entry_val % phi.gen.ring
```
EVERY outgoing edge advances — the swap happens at the end of the body, before the branch decides,
so the trip that exits has advanced too. A header entry then takes its phi.

### `FrameMapping`
Forward dataflow over **sets** of phase vectors, union as the meet — never a collapse onto a
representative. `_classes` folds frames differing by a *uniform* rotation of every ring: those are
the same picture one trip on.

| method | answer |
|---|---|
| `frames(b)` | every frame `b` runs under |
| `entrances(b)` | one frame per **distinct entrance picture**; a back edge is not an entrance |
| `cut(b)` | **the** frame `b`'s stamps resolve in |
| `touches(ref, frame)` | the buffer ids `ref` names at `frame` |
| `generation(ref, frame)` | the generation, for `Move.token` |

> **`cut` is authoritative for anything that reaches the emitted contract.** `tokens_for(ref)`
> resolves through it and that is what lands in `Move.token_ids`. Any second derivation of "which
> frame names this block" agrees in `steady` and inverts in a drain block — see §7.4.

---

## 3. `gir/analyses/frame_hazards.py` — the ordered pairs that can collide

Replaces `lds_hazards.py`. Aliasing is a **concrete storage-id intersection**, not a
`(gdelta_a - gdelta_b) % ring` residue, and the trip gap is a property of the frame graph rather
than a number recovered from it.

### The record types
```
SharedTouch(block, pos, inst, ref, is_write, operand, regions)
Hazard(kind, producer, consumer, ring, gap, cross_agent, cross_block)
    kind : RAW (write→read) | WAR (read→write) | WAW (write→write)
    gap  : 0 = one execution of the pair; >=1 = a later instance of the consumer
    .same_trip        = gap == 0
    .in_program_order = same block and producer.pos < consumer.pos
```

### `FrameHazardSet`
Stores each edge **once** in `_edges`; `_instances` holds `(producer frame, consumer frame, edge
index)`. One edge can be live in several frames and `instances()` hands back the same object each
time. `needing_fence()` returns the cross-agent edges and is **the only place the fence/no-fence
decision is made** — a same-wave edge is discharged by program order plus the completion counter.

### `_walk(node, here, fm, state, reg, emit)` — the pairing rule
A killing reaching-defs over `(block, frame)` nodes; the meet is union at joins.
```
older = ((() if reads[s] else writes[s]) if not touch.is_write
         else set(reads[s]) or set(writes[s]))
...
if touch.is_write: writes[s], reads[s] = {(frame, id(touch))}, set()   # a WRITE kills
else:              reads[s].add((frame, id(touch)))                    # a read only accumulates
```
Both arms of `older` remove a barrier the emitted code does not need:

* **a read pairs with the last write only if nothing has read it yet** — visibility is monotone, so
  the fence that made a write visible to the first reader made it visible to every later one.
  Without this the second half of a read group split across the back edge opens its own barrier at
  the top of the loop for a write that is already visible;
* **a write pairs with the readers since the last write, and with that write only if there are
  none** — nothing refills a buffer it has not consumed, so a read sits between two writes and the
  RAW into it plus the WAR out of it already order them. The surviving WAW is the genuinely-unread
  overwrite (prologue → short arm).

The **entry**, not the instruction, is paired: the same instruction arriving over the back edge is a
later instance however identical the object is.

### `_disjoint_storage(a, b)`
Different operand ⇒ different tensor. Same operand ⇒ disjoint iff some region axis has
non-overlapping possible values. Sets, not equality, so an unpinned access is disjoint from nothing.

---

## 4. `gir/analyses/lds_buffers.py` — storage identity

### `Buffer`
```
Buffer(operand, region: tuple, ring: int)      # one LDS byte range
```
`region` is one value per region axis, `()` when the tile is not split; `ring` is which of the
operand's rotating buffers, in `[0, depth)`. **Depends on nothing** — every input is tile metadata
already on the `Ref`.

### `Geometry.regions_of(ref, is_write)`
Per region axis, the **set** of values this access may touch. A coord pins the axis or the access
touches every region; a *read* of an agent-relative operand pins nothing either, its region being a
fact about the wave rather than the coord.

### `LdsBufferIdSet`
`ids_for(ref)` is the may-set over the whole ring; `at(ref, ring)` narrows to one rotation position;
`depth_of(operand)`, `liveness(prog)`. Ids are dense and deterministic — sorted `buffer_key` order,
so the same Program always numbers the same.

---

## 5. `gir/analyses/token_liveness.py` — `BufferLiveness`

Unchanged: which buffers are live across each body index, from the write/read maps
`LdsBufferIds` collected.

---

## 6. `gir/analyses/dep_tokens.py` — obligation tokens

### `DependenceTokens.run`
Union-find over buffer ids. For every hazard instance, `tied = touches(producer.ref, fp) |
touches(consumer.ref, fc)` — **each end at its own frame** — and `classes.union(tied)`. Classes are
numbered by their lowest slot, so the numbering is dense and deterministic.

`per_ref[id(ref)]` is the token set of `fm.touches(ref, fm.cut(blk.label))`. This is what
`tokens_for` returns and what `TokensPass` stamps into `Move.token_ids`.

### `DependenceTokenSet.naming(edges, block)`
```
for h in edges where h.kind == RAW:
    for end in (h.consumer, h.producer):
        if end.block == block:  out |= tokens_for(end.ref); break
```
**Only RAW buys a wait** — the consumer must observe the producer's data, which is what a memory
token is for. And the name is resolved through `tokens_for` at the **fenced block**, the same
function that produces `Move.token_ids`, so the barrier and the accesses it orders always speak one
frame's names.

### `DependenceTokenSet.choose(prog, options, edges, repeating, preds)`
Pins the legal window to a concrete point, scored by `_in_flight` — the loads still outstanding
there, which is the wait the barrier would emit. Walks back over **real predecessor paths** (the
load a loop-top fence waits on was issued on the previous trip) to `_BACK_DEPTH = 2`, and takes the
smallest count any path gives, since a wait must be safe on every path.

---

## 7. `gir/analyses/fence_regions.py` — where the fence goes

Output is `[PendingMark(Mark('fence', {...}))]`; it mutates nothing. **Position only** — `TokensPass`
names them.

### `_admissible_slots(hazard, n_slots, block)`
The slots of `block` at which **one** fence discharges `hazard`. A fence sits at one of the edge's
OWN two ends, never on a block between them: an intermediate block lies on some path, and the
producer can re-reach the consumer around it via the back edge.
```
if hazard.cross_block:
    if block == producer.block:  return { s : s >  producer.pos }
    if block == consumer.block:  return { s : s <= consumer.pos }
    return {}
a, b = producer.pos, consumer.pos
if same_trip and in_program_order: return { s : a < s <= b }
return { s : s > a or s <= b }                       # the WRAP: two disjoint halves
```

### `_cover(hazards, n_slots, block)` → `{slot: [hazards]}`
Raises if some hazard admits no slot at all — the emitted order itself makes the edge
undischargeable, a schedule defect, not something to model around with a harmless fence.

**Two tiers, keyed on the hazard KIND:**
```
for stamps in (True, False):
    tier = { i : (hazards[i].kind == RAW) is stamps }
    if stamps: best = max(slot over the tier's windows)          # as LATE as legal
    else:      best = the latest already-chosen slot legal for some remaining edge,
                      else the slot covering the most of them
```
* **Tier 1, RAW** — the only kind that stamps the barrier, placed as late as its window allows: every
  slot it gives up is a load that has to have landed, while the loads issued after it stay in flight.
* **Tier 2, WAR / WAW** — needs a barrier between its ends but **no name on it**, so it *rides* a
  tier-1 barrier whenever one is legal. StinkyTofu makes every barrier a waitcnt point, so riding
  one orders it for free; naming a WAR's buffer would wait on the load issued just after the fence,
  which is how the wait gets pinned to 0.

### `_one_barrier(covers, n_slots, block)` and `merge`
The cover is **per buffer** — `(operand, region, ring)` — so a fence names only the storage it
orders. But a separate cover per buffer also opened a separate *barrier* per buffer, even where one
slot discharged them all. `_one_barrier` merges cover sets whose `_admissible_slots` intersections
still intersect; `_cover_slot` puts the merged barrier at the latest slot discharging every edge.

`FenceRegions(merge=True)` is the switch, mirroring `HoistCopiesPass(group=True)`, and folds into
`cache_key` so the two settings cache separately. Meaningful **only for tier 1** — a tier-2 edge
already rides a RAW barrier rather than opening its own.

| fixture `BF16_NT_KMN_FUSED_XAGENT`, PGR2/PLR1 | fences |
|---|---|
| `merge=True` | 6 — `{prologue: 2, steady: 1, short0: 1, short1: 1, drain0: 1}` |
| `merge=False` | 16 — `{prologue: 6, steady: 4, short0: 2, short1: 2, drain0: 2}` |

### `_windows` / `_runs` / `separating` / `separated`
`_windows` returns one `Region` per contiguous run of legal slots — a wrap's two halves are two
runs, so the freedom spans blocks and the naming layer spends it. A barrier belongs INSIDE the body;
the boundary slot is dropped unless it is the only one. `separated` treats an **empty** fence token
set as a full drain, which pins every edge — conservative, never wrong.

## 8. `gir/analyses/reg_hazards.py` — the register-side twin

Same shape as `LdsHazards`, over rotating VGPRs instead of LDS.
```
RegTouch(block, pos, inst, ref, is_write, operand, ring_pos=(group,slot), coord)
_free_coord(ref) = per non-K tile axis, the pinned value, or None if the axis is `covers`-spanned
_accesses(blk)   = every register touch with ref.slot is not None
                   (no slot ⇒ not on a ring ⇒ the accumulator ⇒ program order suffices)
_same_register(a,b) = MAY they touch one register (an unpinned axis matches every value)
```
Reuses `Hazard`, `RAW/WAR/WAW`, `_hazard_kind`, `_reachable` from `lds_hazards`. Consumed by
`HoistCopiesPass` when `reads=True`, so a lifted read cannot land on a register still live.

---

## 9. `gir/analyses/barrier_uniformity.py` — the liveness half

A fence at block or cluster scope is a **rendezvous**: the waves that arrive wait for the waves that
have not. A wave that skips it does not stall — the ones that did arrive do, forever.

```
PROC_SCOPES = ('block', 'cluster')
ROLE_ALL    = 'all'
_preds_of(term) = [pred] for CondGoto | [p for p,_ in arms] for CondChain | [] otherwise
                  # LoopBack carries a trip COUNT, not a comparison; Goto/Return test nothing

for each proc-scoped fence Mark:
    walk back to entry; report any block on the way that some waves can skip:
        reason='role'  : block.role != ROLE_ALL          (a subset runs it)
        reason='guard' : entered through a predicate reading a per-wave symbol
```
The ledger only proves the *ordering* obligations were discharged, so it is satisfied by a fence no
wave reaches. **Only this says the fence is reachable.** `barrier_uniformity_asm.py` asks the same
question of the emitted instruction stream.

---

## 10. The passes

### `passes/hoist_copies.py` — `HoistCopiesPass(group=True, reads=False)`
Chooses the **issue point** of each movement, which is what decides how many loads are in flight at
a fence.
```
_is_copy(inst) = Move with a shared dst          # global → shared
_is_read(inst) = Move with a register dst        # shared → register

_earliest(inst, label, *hazard_sets):
    return max(h.producer.pos + 1
               for h in all hazard sets
               if h.consumer.inst is inst and same block
                  and not cross_block and same_trip and in_program_order,
               default=0)

hoisted(blk, lds, reg, group, reads):
    at = { i : _earliest(copy_i) for each copy }
    if group and at:
        slot = max(at.values())
        at   = {i: slot for i in at} if slot < min(at) else {}
    if reads: at |= { i : _earliest(read_i, lds, reg) for each read }

    # INTRA-CLASS ORDER, per class (copies, then reads):
    floor = -1
    for i in class_indices:
        at[i] = max(at[i], floor)
        floor = i + 1 if at[i] >= i else at[i]

    at = { i:s for i,s in at.items() if s < i }         # only real lifts
    rebuild body inserting each lifted inst at its slot, sorted so ties keep original order
```
Only a **same-trip edge already in program order** constrains a movement: that is a reader of the
storage this movement overwrites, and it must stay behind that reader. An edge at a trip distance is
discharged by the trip boundary, not by position.

`group=True` sends the copies to **one** point (the latest of their individual earliests) so they
keep sharing a barrier; letting each chase its own floor straddles another operand's last read and
buys a second proc-scoped sync per trip. `reads=True` lifts the reads too, which lets a fence sit
past them instead of draining LDS at the trip top.

**Intra-class order is not free** and the clamp is subtle. Two copies to one ring must land in their
original sequence — no token downstream can express an order that was already wrong. The running
clamp is monotone in the *slot number*, but a movement landing on slot `i` is emitted **before** the
instruction sitting at index `i`. So a class member that does NOT move keeps index `i` and the floor
it leaves behind is `i + 1`, not `i`; clamping to `i` lets a later mover be emitted ahead of it.
Verified over both classes and all four `group`/`reads` settings on six fixtures: 214 checks, 0
inversions.

### `passes/tokens.py` — `TokensPass`
Stamps each LDS `Move` with **both** namings.
```
_lds_ref(inst, side)      = first shared ref among srcs / dsts
_unit_key(prog, operand)  = the key of prog.meta['unit_regions'] containing operand, else (operand,)
_completion_unit(...)     = that key if len>1 else the bare operand name
                            # so an unfused kernel's token stays ('lds','A',gen), unchanged
_region_key(prog, ref)    = () unless meta['per_region_completion'] AND the producing unit
                            actually emits per-region movements; else region_of(prog, ref)
                            # a read can only await at the granularity its PRODUCER offers
_buffer_generation(ref,g) = (int(g),)
_storage_ids(inst,tokens) = sorted union of tokens.tokens_for(r) over shared refs of inst

for each Move with a shared ref:
    inst.token     = ('lds', completion_unit, (gen,)) + region_key      # completion token
    inst.token_ids = _storage_ids(inst, tokens)                         # backend storage names
```
The two are stamped together because they are read together and drifting them apart is a defect
class. (Naming debt: `Move.token_ids` holds *dependence* tokens while three names say "buffer" —
task #407.)

### `passes/collect_pending.py` — `CollectPendingMarksPass`
Runs the region analyses and stashes their `PendingMark`s on `prog.pending`. IR-untouched.
```
default set = [SwapRegions, GrIncrementRegions, RegionIncrementRegions,
               FenceRegions, Gl2PrefetchRegions]
prog.pending = concat(am.get(a, prog) for a in analyses)
```

### `passes/placement.py` — `PlacementPass`
Resolves each `PendingMark`'s legal window to one concrete anchor. Mutates no IR.
```
lo = slot just past region.after          # first legal slot
hi = slot occupied by region.before       # one past the last legal slot
if hi < lo:  return (block, region.before)                 # malformed window: stated bound
idx = lo + (hi-lo)//2 if policy == MIDPOINT else lo
return (block, body[idx] if idx < len(body) else BLOCK_EXIT)
```
Policy travels with the Region because it differs by what the Mark mutates. A fence's window is
degenerate (`_region_at`), so `lo == idx` and the cover's choice is honoured exactly.

### `passes/apply_marks.py` — `ApplyMarksPass`
The only body mutator among the swap passes.
```
group placed marks by block
for each block: resolve anchors to indices, insert DESCENDING by index
                (so earlier insertions do not shift later anchors)
prog.pending = []
```

### `passes/split_prefetch_guard.py`
Creates the guarded prologue shape `prologue → prefetch_peel (guarded) → prologue_join`, plus
`steady`. This is what makes the PGR2 second copy conditional, and therefore what makes the queue
depth at loop entry path-dependent.

---

## 11. `gir/verify.py` — the invariants

### `_check_tokens` (G-TOKEN) — four sub-properties
```
V2: tokens.unresolved must be empty
    "an unnamed LDS access breaks StinkyTofu's all-or-none token rule for its whole basic block,
     so this is not a partial result"
V1: for every hazard, if the two ends are PROVABLY DISJOINT storage they must share NO id
    (a shared id orders them for no reason, and means the key no longer distinguishes)
V4: see _check_token_stamps
```
V3 is GONE. A fence with an EMPTY token set is a **full drain** — legal, conservative and
equivalent to naming every token; `memoryToken=None` lowers to a plain `SBarrier`. Over-draining is
a cost, not a hole, and `_check_token_reuse` is what catches a real one.

### `_check_token_stamps`
```
every shared access must have tokens_for(ref) non-empty
if any Move is stamped: tuple(inst.token_ids) == sorted union over its shared refs
    "the emitter reads the STAMP, so a stale or truncated one is a mis-named LDS access"

V4 — IN AN UNFENCED BLOCK THE TOKEN IS THE ONLY ORDERING:
fenced = { blocks touched by any needing_fence() edge }
for every hazard whose ends are both outside `fenced`:
    require tokens_for(producer.ref) & tokens_for(consumer.ref) != {}
```
With no barrier the def-use chain is all StinkyTofu has: two ends that share no token are free to be
reordered. The token need not be the buffer id and its producer need not be in this block — only
that the pair is chained.

### `_check_token_reuse` (G-TOKEN-REUSE) — and the bar it must NOT use

```
for h in hz.needing_fence():
    pins = set(dep.tokens_for(h.consumer.ref)) if h.kind == RAW else set()
    if not pins or separated(h, fences, n_of.get, pins): continue
    raise
```
**The bar is the CONSUMER'S STAMP** — what the backend actually carries — not what the fence claims.
A WAR/WAW needs a barrier between its ends and is checked with `separating()` alone, since only a
RAW is stamped.

> Deriving `pins` from the same function that *writes* the fence makes this check true by
> construction: it then asks "does the fence name what the fence names?", returns clean on a
> misnamed fence, and the fault it exists to catch becomes invisible. The standalone oracle
> `tool/fence_token_check.unseparated_edges` uses the identical bar for the identical reason. Both
> were once pointed at the fence's own source and both went silent on a live drain-block defect.

### the original text of `_check_token_reuse`
Two independent derivations must agree.
```
live   = tokens.liveness(prog)                       # from the CFG
racing = { id(h.consumer.ref) for h in hz.needing_fence() }   # from the hazards
for each block, walking the body in order:
    fenced |= fence.at['tokens']  at each fence Mark
    for each Move dst ref that is shared and in `racing`:
        for buf in tokens.buffers_for(ref):
            if buf in live.across(block, i) and not (dep.tokens_for(ref) & fenced):
                raise "refills {buf} against another wave's read, and no fence before it names
                       any of its tokens — the barrier orders every buffer except the one at risk"
```
`BufferLiveness` says the buffer still spans this point; `LdsHazards` says the refill races another
wave's read; a preceding fence must then name a token this refill carries.

### Other checks in the same file
`_check_gen_ssa`, `_check_noiter`, `_check_semantic`, `_check_phase_counts`, `_check_drain_count`,
`_check_short_arm_count`, `_check_loop_headers`, `_check_cfg`, `_check_walk`, `_check_emit`,
`_check_barrier_uniformity`, `_check_trip`, `_check_term`, `_check_operands`, `_check_marks`,
`check_register_slots`, `check_block_scope_covered`, `check_rotation_waw`.

### The third invariant, in the test suite
`Tensile/Tests/unit/test_gir.py::test_a_fence_names_every_buffer_LIVE_ACROSS_it_not_just_the_edges_that_placed_it`
— **a barrier is permeable per token**. It must name both ends of every edge it stands inside, not
only the edges the cover charged to it. A minimal-cover naming would be sound only if barriers were
opaque; they are not.

---

## 12. Emission — `gir_to_rocisa.py`

### `_fence_tokens(at)`
```
ids = at['tokens']
if not ids: raise "a fence with no tokens is a barrier that orders nothing (G-TOKEN should have
                   caught this)"
return sorted(int(t) for t in ids)
```

### `sync_comment(tokens)` (`gir_tag.py`)
```
"sync LDS%u" % t          if exactly one token
"sync LDS %s" % [t,...]   otherwise
```
Spelled exactly as `Components/TensorDataMover.issueLoad` spells it, so copy tags and fence tags are
greppable together.

### `_emit_fence(out, phase, at)`
```
tokens = _fence_tokens(at)
_tag(out, phase, 'fence', "<buffers> scope=<s> covers <n> edge(s) <kinds> <sync>")
code = writer._syncThreads(kernel,
          "GIR fence: order <buffers> (<kinds>) <sync>",
          memoryToken=tokens)
if code non-empty: _append_tag(code, "fence <buffers>"); _register_fence(code); out.add(code)
```
GIR decides *where* and *what it orders*; L3 owns the instruction. `_syncThreads` issues waitcnt +
barrier, so it has the strength the anti-dependence demands.

### `KernelWriterModules.syncThreads`
```
if NumThreads > WavefrontSize:
    if SeparateVscnt:                      add SWaitCnt(dscnt=0)      # extra navi wait
    elif SIA==2 or PGR>=2 or skipForce:    comment "Skip force waitcnt0"
    elif Waitcnt0Disabled:                 add SWaitCnt(dscnt=0,vlcnt=0,vscnt=0)
    barrier = SBarrier(comment)
    if memoryToken is not None: barrier.setMemToken(MemTokenData(memoryToken))
else:
    comment "Skip barrier: NumThreads=..."
```
`MemTokenData` (`rocisa/include/container.hpp:1479`) is a bare `std::vector<int>` — **there is no
per-token kind**. A fence cannot ask for "dscnt ordering only"; the token set is the entire lever.

---

## 13. StinkyTofu — from tokens to `s_wait_*`

### `conversion/rocisa/ToStinkyTofuUtils.cpp` / `AllHwMappings.cpp`
```
if (auto memToken = inst->getMemToken())
    stinkyInst->addModifier<MemTokenData>(MemTokenData{memToken->tokens});
```
The rocisa modifier is carried across into StinkyTofu's IR verbatim.

### `StinkyBuildImplicitDependencyPass.cpp`
```
addUniqueLdsDest(inst, t): append StinkyRegister(RegType::LDS, t, 1) to dests if absent
addUniqueLdsSrc (inst, t): same for srcs

for each instruction with MemTokenData (assert tokens non-empty):
    if isBarrier:            for t: addUniqueLdsDest(t); addUniqueLdsSrc(t)
    elif isTensorLoad or isDSWrite:              for t: addUniqueLdsDest(t)   # LDS producer
    elif isDSRead or isGlobalStoreAsyncFromLds:  for t: addUniqueLdsSrc(t)    # LDS consumer
    else: assert(false)
```
So the def-use chain builder sees
```
producer(def LDS[t]) → barrier(use+def LDS[t]) → consumer(use LDS[t])
```
which forces producers → barrier → consumers.

`processBarrier` returns false and skips a **non-all-wave split barrier** (workgroup-sync mode).

### `waitcnt/WaitDataflow.cpp` — `computeRequiredWaits(inst, state, rawNeedsWait, required[])`

The value on the emitted instruction. `required[c] = kUnused` initially; `tightenRequired` takes the
**min** across deps on one counter — that drains the closest-to-tail dep, the most permissive wait
that still satisfies it.

**(a) SSA RAW through the UD chain — this is where a tensorcnt value comes from:**
```
for src in inst->getSources():
    if isPhi(src): tightenRequired(c, phiCurrentQueueWait(src, c, state, seen)); continue
    c = classifyMemOp(src)
    if !rawNeedsWait[c](inst): continue          # e.g. tlcnt only drains at a barrier
    for q in state.queues[c]:
        n = q.countFrom(src)                     # position from the tail
        if n > 0: tightenRequired(c, n-1)
```
`n-1` **is** the emitted `N`: how many entries younger than the dep may remain outstanding. A
barrier that names token *t* has `LDS[t]` as a source, so its reaching def — the last `tensor_load`
into that token — is a dep, and `N` = number of loads issued after it. **One younger load in flight
⇒ `s_wait_tensorcnt 1`. None ⇒ 0.**

**(b) `scanDsAntiDeps(anchor, anchorTokens, barrierMode)` — WAR-on-LDS:**
```
for q in queues[CK_DS], for idx, op in q.ops:
    if !barrierMode and !(isDSRead(op) or isDSAtomic(op)): continue   # writer guards readers only
    if isOnSamePipeline(anchor, op): continue        # the DS FIFO orders same-pipeline pairs
    overlap = (op has no MemTokenData) || hasTokenOverlap(op.tokens, anchorTokens)
    if overlap: tightenRequired(CK_DS, clampWaitCount(qsize - idx - 1))
```
The SSA chain captures RAW but **not** anti-dependences, so an LDS writer must wait for prior LDS
readers on the same token and a barrier must wait for any prior DS op on a matching token.

**(c) `scanAsyncAntiDeps(anchorTokens)`** — the same for `asynccnt`
(`global_store_async_from_lds_*` is an LDS reader with no register dest).

**(d) Tensor-side conservative scan:**
```
if isTensorAnchor(inst) and inst has MemTokenData:
    for op in queues[CK_Tensor]:
        if op has NO MemTokenData: tightenRequired(CK_Tensor, qsize-idx-1)
```
Tagged overlaps are already covered by the SSA UD chain through the LDS pseudo-regs.

**(e) Conservative fallbacks — untagged means "cannot prove disjoint", so drain to 0:**
```
isTensorAnchor && no token && any tensor in flight            → required[CK_Tensor] = 0
(isLdsWriterAnchor||isBarrier) && no token && async in flight  → required[CK_Async]  = 0
isLdsWriterAnchor && no token && DS in flight && !isDSWrite    → required[CK_DS]     = 0
isBarrier && DS in flight && (barrier untagged || ANY queued DS op untagged)
                                                               → required[CK_DS]     = 0
```
The last one is why a single untagged `ds_read` anywhere in flight forces `s_wait_dscnt 0` at every
barrier — one missing token poisons the whole block.

**(f) Planned vs computed** (`applySpec`): take the **strictest (smallest)** of the planned and
computed values, so a relaxed planned value can never mask a tighter residual the consumer needs.

### `MemTokenConsistencyCheckPass.cpp`
```
isMemTokenCandidate = isTensorLoad | isDSWrite | isDSRead | isGlobalStoreAsyncFromLds
per basic block: if (some candidate has a token) and (some candidate has none) → error
```
This is the **all-or-none rule per basic block** that makes `LdsBufferIds.unresolved` fatal on our
side rather than a partial result.

### `LegalizationUtils.cpp`
When an instruction carrying `MemTokenData` is split or expanded (signal/wait pair, a load split into
`load1`/`load2`, a store into `store1`/…), the modifier is **copied onto every product**. A split
that dropped it would create an untagged op and trip (e)'s fallback.

### `InsertClusterBarrierPass.cpp` / `LoopRegionRemarkPass.cpp`
Cluster-scope barriers also carry tokens; the remark pass reports `-- no MemTokenData, non-movable`
for an instruction it therefore may not reorder.

### Barrier merging
StinkyTofu merges coincident/nearby barriers and **unions their tokens**:
```
s_barrier_signal -1   // merged barrier, sync LDS0, sync LDS1, sync LDS2, sync LDS3
```
Two GIR fences emitted at *different* slots can still merge. **Check the `merged barrier` comment in
the emitted `.s` before attributing a wait value to a GIR token set** — several apparently-inert
narrowings were inert for exactly this reason.

---

## 14. Verification tooling — `Tensile/Lowering/tool/`

### `fence_token_check.py`
Re-derives separation from the hazard edges and the fences **actually present in the final body**,
never from the cover that placed them.
```
fences_in(prog)    = { block: { index: frozenset(tokens) } }   over Mark('fence')
_positions(prog,t) = { id(ref): (block, index, frozenset(tokens_for(ref))) }  over the FINAL body
_paths(succ,src,dst) = every simple path, plus the self-loop when src==dst is reachable
for each cross-agent hazard:
    require some fence on every path between the ends whose tokens intersect BOTH endpoints' tokens
```
A fence index and an access index are on the same axis because both are read off the final body —
that is what makes the oracle independent of `FenceRegions`.

### `asm_wait_check.py`
Asks the same questions of the emitted `.s`. ⚠️ Known limitation (task #414): it models no LDS
address, so `wrong_generation` checks the *tag*, not the generation.

### `symexec/`
Symbolic executor modelling the counters, LDS and registers over a `(tensor, parity, bytes)` alias
domain. Perturb **inside** `loop_body`'s span or a live oracle looks dead.

Of the standalone asm oracles, three are gates (0 everywhere); `cross_agent_cover` fires on ULM0
itself and `unsynchronised_cross_wave` fires 2459× on hardware-validated code — neither
discriminates.

---

## 15. Worked example — MAF bf16, one trip

```
2760  s_wait_tensorcnt 0
2762  s_barrier_wait -1   // GIR fence: order A+B (WAR) sync LDS [0,1,2,3]
2764  tensor_load_to_lds ...            // sync LDS [0,2]   <GIR: copy A+B gen=0>
2843  s_barrier_wait -1   // GIR fence: order A+B (RAW) sync LDS [1,3]      (no wait)
```
The WAR fence's 64 covered edges split into two families:
```
24×  WAR  producer tokens [0] → consumer [0]    reads gen0 → copy gen0   (same generation, same trip)
 8×  WAR  producer tokens [1] → consumer [0]    reads gen1 → copy gen0   (the loop-carried wrap)
```
Tokens 1,3 are defined only in the prologue and are long retired, so naming them is free. Tokens 0,2
are defined by the in-loop copy itself, so a fence **before** that copy binds to the previous trip's
copy — that is the `s_wait_tensorcnt 0`. Dropping the same-generation contribution is refused by
G-TOKEN-REUSE; dropping the reader's side is refused by G-TOKEN; naming only the cover's own edges
is refused by the live-across test.

Contrast ULM0 (no GIR, 3× unrolled body, alternating `LDS0`/`LDS1`):
```
barrier sync LDS0 (no wait) | copy LDS0 | s_wait_tensorcnt 1 | barrier sync LDS1
```
Its copy never refills a token read in the same segment, so its pre-copy barrier costs nothing.

---

## 16. Known defects and live debt

| # | Where | Statement |
|---|---|---|
| #401 | `lds_hazards.Hazard.distance` | the field is the alias residue, not the trip gap — the name lies |
| #404 | `fence_regions._spanning` | a fence guarding a Φ-fused copy names one member, not the group |
| #407 | `passes/tokens.py` | `Move.token_ids` holds dependence tokens; three names say "buffer" |
| #396 | `fence_regions._admissible_slots` | cross-block `distance=None` is faithful for aliasing, costly for sync |
| #398 | `hoist_copies` | a read hoist must not raise a copy's floor |
| #414 | `tool/asm_wait_check.py` | models no LDS address; `wrong_generation` checks the tag |
| #409 | emitted asm | no `s_wait_dscnt 0` inside the main loop (in progress) |
| #411 | mxf8 | 46/263 hang: 4 cross-wave copy/read races in one barrier interval |

### The reverted optimisation stack
`960bbee6839` rolled `fence_regions.py` + `dep_tokens.py` back to `58a5ea7f0b9`, undoing five
commits — `7497e212cca`, `8ce120e4b52`, `526d3350a0e`, `aa628d5a034`, `ce5b47a1ae2`. Measured
trade recorded in that commit message:

* in-loop `s_wait_tensorcnt 0` per trip **1.00 → 0.00**, barriers **4.50 → 3.00**
* mxf8 cross-wave copy/read races **2609 → 2447**
* the 46/263 hardware-stalling solution **4 → 1**

Kept deliberately: the id/token split, `asm_wait_check`, the `gir_tag` grammar, and
`HoistCopiesPass(reads=True)`.

**Bisection of the wait value** (each measured by loading that commit's two files over an otherwise
current tree):

| state | ULM1 arms |
|---|---|
| `526d3350a0e` | 8× `[0]` |
| **`aa628d5a034`** — *cover per operand* | 6× `[0]`, **2× `[1 0]`** |
| `ce5b47a1ae2` — *name only ends already issued* | 6× `[0]`, 2× `[1 0]` |
| `960bbee6839` (revert) | 8× `[0]` |
| HEAD | 8× `[0]` |

`ce5b47a1ae2`'s rule — *"a fence names only the ends of its edges that have already issued; for a
WAR edge the consumer IS the next copy, and naming a copy that has not issued yet is pure loss"* —
is a positional filter in `_spanning`:
```
if e.block != block or e.pos < slot
```
Its own commit message states it does **not** move the tensorcnt (it buys barriers 13.50 → 12.00 and
dscnt waits 37.25 → 30.75). Applying it to HEAD alone was measured to change nothing.

**The open problem is not the fence rule.** It is making `aa628d5a034`'s per-operand cover coexist
with the race counts the revert recorded.

### Working-tree deviations from HEAD (uncommitted, under review)
`_admissible_slots` widened to give a cross-block hazard a producer-block window and an all-slots
window for blocks in between — this contradicts `a8763655463` and is the change most likely to be
wrong. Plus `_producing_half`, `_fills_shared`, `_wrap_drain` (unreachable), an `FR_DUMP` debug
print, and a `_class_key` rewrite. Measured inert on the wait value; the `dep_tokens` half is the
only one that repaired anything (519 test failures → green).

---

## 17. Invariant summary

1. **G-TOKEN V1** — provably disjoint storage must share no id.
2. **G-TOKEN V2** — no unresolved access; all-or-none per basic block is StinkyTofu's rule.
3. **G-TOKEN V3** — every fence names ≥1 token.
4. **G-TOKEN V4** — in an unfenced block, both ends of every hazard share a token.
5. **G-TOKEN-REUSE** — a racing refill is preceded by a fence naming one of its tokens.
6. **Live-across** — a barrier is permeable per token; it names both ends of every edge it spans.
7. **Barrier uniformity** — every wave of the scope reaches every proc-scoped fence.
8. **Cover totality** — every cross-agent hazard admits ≥1 slot, else the schedule is illegal.

Rules 1–5 are enforced in `verify.py`, 6 in the unit suite, 7 in `barrier_uniformity.py` + its asm
twin, 8 by the `RuntimeError` in `_cover`.

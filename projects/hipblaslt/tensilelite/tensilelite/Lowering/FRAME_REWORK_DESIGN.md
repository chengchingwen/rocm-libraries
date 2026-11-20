# Frame-based rework of storage / generation / hazard / token / fence

Directive: **storage identity is primitive and comes first; generation is a coordinate layered on
top of it; hazards are per-generation; tokens and fences are derived in ONE frame and verified
against the others — never taken as a union.**

---

## 1. What is wrong today

```
GenReaching ──▶ LdsBufferIds ──▶ DependenceTokens ──▶ FenceRegions
     │              (Buffer carries `generation`)
     └──▶ LdsHazards (symbolic gdelta arithmetic)
```

Three specific defects follow from that ordering:

1. **`Buffer(operand, region, generation)` conflates two things.** `generation` is a *logical label*
   in a rotation, not a byte range. Storage identity is being derived from a dataflow that should
   be layered on top of it, so `LdsBufferIds` cannot run until `GenReaching` has resolved a phase.

2. **`RELATIVE_BASE` invents a number.** When predecessors disagree, `_entry_of` assigns `0`, marks
   the block relative, and every consumer must remember to refuse (`_provably_disjoint`,
   `_pair`'s peel arm, `_one_absolute_frame`). The arbitrariness is inherited by successors. Two
   call sites currently gate on it; a third that forgets is a silent wrong answer.

3. **The union is the only reconciliation available.** `dep_tokens._class_key` unions the two ends'
   buffer ids precisely because *"the two ends carry the generation label of their own frame"*.
   That is the correct fix given the current model — and it is exactly the over-naming that makes a
   fence bind to an in-flight load and emit `s_wait_tensorcnt 0`. The model forces the imprecision.

`LdsHazards` half-escapes by working symbolically — `(gdelta_a − gdelta_b) mod ring` is
frame-independent because the base cancels — but that escape is unavailable to anything that needs
*identity* rather than *difference*, so `LdsBufferIds`, `DependenceTokens` and `FenceRegions` are all
stuck.

---

## 2. The new layering

```
LdsStorage        pure, no dataflow      (operand, region, slot) -> dense id
     │
FrameMap          dataflow               block -> frames it runs under
     │                                   (ref, frame) -> storage ids
FrameHazards      per (frame, trip gap)  concrete aliasing, no modular arithmetic
     │
DependenceTokens  derive @ f0, verify ∀f, widen only where forced
     │
FenceRegions      derive @ f0, verify ∀f, widen only where forced
```

### 2.1 `LdsStorage` — pure storage identity

A buffer is a **byte range**. Under an S-deep rotation an operand owns S disjoint ranges; which
*generation* currently occupies one is a separate question.

```
Slot   = (operand, region: tuple, slot: int)      # slot in [0, ring)
LdsStorage.run(prog):
    for each shared ref:
        regions = _regions_of(ref, ...)           # static: coord / covers / spans / extents
        ring    = ref.gen.ring if ref.gen else 1
        keys[id(ref)] = { Slot(operand, r, s) for r in cross_product(regions)
                                              for s in range(ring) }   # MAY-set over slots
    ids = dense numbering over sorted(all keys)
```

Depends on **nothing**. No `am.get`. Every input is tile metadata already on the Ref.

`keys[ref]` is the set of slots this ref *may* touch over the whole rotation; `FrameMap` narrows it
to the one it *does* touch in a given frame.

### 2.2 `FrameMap` — the map you described

> *"given one instruction in the loop, what buffer ids it is working on under what generation"*

A **frame** is a rotation phase: one value per ring identity. Frames advance on the back edge by the
`GenXfer.adv` vector, so the reachable frames form a cyclic orbit.

**One frame per block ENTRANCE, and every frame must be considered.** A frame is not a global orbit
index — it is the phase vector a block is entered *on*, and a block with several incoming edges has
several. So the frame set is a forward dataflow whose lattice is **sets of phase vectors**, and the
meet is **union**:

```
Frame = { ring_id : phase in [0, depth(ring_id)) }

phase_out(edge e from b) = { advance_e(F) for F in frames(b) }
                           # advance_e applies the GenXfer adv vector carried on that edge;
                           # a forward edge that advances nothing is the identity

frames(b) = { F0 }                                    if b is the entry block
          = union over incoming edges e of phase_out(e)   otherwise

fixpoint: iterate to stability -- guaranteed, phases are finite (mod depth per ring)
```

The back edge is just another incoming edge, so the steady block accumulates its orbit naturally
rather than by an lcm formula; the `skipPGR2` join carries one frame per arm; and the short-path
entry into `drain0` contributes its own frame beside the one arriving from `steady`.

```
FrameMap.run(prog):
    frames = fixpoint as above
    for blk, for F in frames(blk), for ref in blk:
        phase = (F[ref.gen] + ref.gdelta) % depth(ref.gen)
        touches[(id(ref), F)] = { storage id of (operand, region, phase) }

API:
    frames(block)        -> tuple[Frame]      # EVERY entrance phase, not a representative
    touches(ref, frame)  -> frozenset[int]    # storage ids, CONCRETE
    advance(frame, edge) -> Frame
```

**This is where `RELATIVE_BASE` dies, and the reason is the meet operator.** `_entry_of` today does:
*"if every predecessor agrees on a Gen's value take it, otherwise assign `RELATIVE_BASE = 0` and
flag the block relative"* — a **collapse** where the lattice calls for a **union**. Disagreeing
predecessors do not mean the phase is unknowable; they mean there are two of them, and both are
real. Replacing collapse with union removes the invented value entirely, and with it every
`is_relative` guard, `_provably_disjoint`'s second arm, `_one_absolute_frame`, and
`Unresolved("frame is relative")`.

It also explains the multi-entrance bugs directly: a block entered on two phases had one of them
silently overwritten by `0`, so exactly half its aliasing was invisible.

### 2.3 `FrameHazards` — per-generation hazards

Two accesses collide in a given frame pair iff their **concrete** storage id sets intersect. No
`_disjoint_storage` may-alias reasoning, no `(gdelta_a − gdelta_b) % ring`.

```
FrameHazards.run(prog):
    for t in range(period):
        for dt in range(period):                  # dt = trips between the two instances
            u = advance^dt(t)
            for (a, b) in candidate pairs:
                if touches(a.ref, t) & touches(b.ref, u):
                    kind = _hazard_kind(a, b)
                    emit FrameHazard(kind, a, b, at=t, gap=dt, cross_agent=...)

API:
    at(frame)          -> the hazards live in that frame
    all()              -> every (frame, hazard)
    needing_fence()    -> cross-agent subset, as today
```

`dt = 0` is the same trip; `dt >= 1` is the wrap. `Hazard.distance` — currently the alias residue
wearing a name that says trip gap (**#401**) — is replaced by an honest `(at, gap)` pair.

Cross-block pairs stop needing `_reachable` + `distance=None`: a cross-block pair is just a pair
whose two ends have frame sets from different blocks, and reachability decides which `(t, u)`
combinations are real.

### 2.4 `DependenceTokens` — construct frame-invariantly, verify per frame

An instruction carries **one** token stamp, used in every frame; the hazard set varies per frame.
So the stamp must be a frame-invariant object.

**The stamp IS derived in one frame; the check is what proves it.** A token is a LOGICAL name --
the only question a consumer asks is *"do these two ends share one?"* -- and the next frame is this
one with the ring slots PERMUTED.  A permutation is a bijection on names, so it relabels both ends
of a same-frame edge together and the answer is identical.  `tokens_for(ref)` is therefore
`touches(ref, representative(block))`, and `_assert_covers_every_frame` asserts that over every
instance rather than arguing it.

**Where one frame does not suffice, and why.** The relabelling argument needs both ends read at the
SAME frame.  For an edge whose ends run one rotation apart -- a wrap (`steady -> steady`, gap 1), or
a peel producer feeding the second trip (`prologue -> steady`, `A=0 -> A=1`) -- there is no common
frame, and no single-frame assignment exists: the same-frame edges inside the block already force
`offset(producer) == offset(consumer)` while the wrap forces them to differ by one.  That is a proof
of impossibility, not a tuning failure, and it restates two results already measured on this branch
(the token is frame-relative; cross-trip identity fails at any value).

So `_stamp_frames` names the representative frame PLUS any frame a cross-frame edge reads the access
at -- derived in one pass, never a repair loop.  Measured on MAF bf16 no-split `TDMFuse=1`: 200 of
330 shared refs carry exactly one token; the 130 that do not are exactly the ends of the wrap class.

**Verification never fails.** LoopIR already covers every hazard; GIR only *adjusts* placement and
naming under that rule. The per-frame check is therefore an **oracle, not a repair loop** — if it
reports an uncovered edge, the derivation is wrong (or LoopIR is), and that is a defect to fix, not
a case to widen.

That forces the construction to be right by symmetry rather than by iteration:

```
construct:
    classes = union-find over storage ids
    for h in FrameHazards.at(f0):                       # the representative frame
        union(touches(h.producer.ref, f0), touches(h.consumer.ref, f0))

assert:
    for t in all frames, for h in FrameHazards.at(t):
        assert tokens(h.producer.ref) & tokens(h.consumer.ref), \
               f"{h} uncovered in frame {t} -- the derivation is wrong, not this frame"
```

**Why one frame suffices.** Frame `t+1` is frame `t` with the ring slots permuted by the advance.
The classes are built over *storage ids reached through the frame map*, so applying the same
permutation to both ends of an edge maps a class onto a class. The stamp is invariant under the
rotation because the rotation is an automorphism of the hazard graph. The per-frame assertion is
what proves that automorphism actually holds for this program — including at the peel boundary,
where it is least obvious.

This is the operative difference from today: `_class_key` unconditionally unions both ends of every
edge because the two ends carry labels from *different* frames and it has no way to relate them.
Here they are related by construction, so no defensive union is needed.

### 2.5 `FenceRegions` — same shape

```
construct:
    cover = _cover(FrameHazards.at(f0) filtered to needing_fence(), ...)
    tokens_for_fence = tokens of the edges it is charged, PLUS every edge live across it
                       (invariant 6: a barrier is permeable per token)

assert:
    for t in all frames, for h in FrameHazards.at(t).needing_fence():
        assert separated(h, placed_fences, t)
```

`separated(h, fences, t)` is the existing `tool/fence_token_check` predicate, evaluated **per
frame** — and fixed per **#416** so that only a fence in the consumer's block at `s <= consumer.pos`
counts, never one in an intermediate block.

---

## 3. What this fixes

| today | after |
|---|---|
| `Buffer` mixes storage with a rotation label | `Slot` is a byte range; frame is a coordinate |
| `RELATIVE_BASE` invents a phase, consumers must remember to refuse | frames are enumerated; no arbitrary value exists |
| `_class_key` unions both ends unconditionally | union-find closes only what another frame forces |
| `_one_absolute_frame` gates the fence naming | every frame is absolute; the gate is deleted |
| `Hazard.distance` is the alias residue (#401) | `(at, gap)` — frame and trip gap, separately |
| `_provably_disjoint`, `is_relative` guards ×3 | deleted |
| cross-block `distance=None` (#396) | real `(t, u)` frame pairs |
| a green oracle that shares the change's premise (#416) | per-frame separation, consumer-block only |

It also removes the reason the reverted stack existed: those five commits were all attempts to claw
precision back from a union that the *model* forced. With per-frame derivation the precision is
there by construction, so `aa628d5a034`'s per-operand cover stops being a special case.

---

## 3b. What landed (2026-09-07)

| module | state |
|---|---|
| `analyses/lds_storage.py` | NEW. `Slot(operand, region, slot)`, `Geometry.regions_of`, dense ids. Depends on nothing. |
| `analyses/frame_map.py` | NEW. `Frame`, per-entrance fixpoint, `touches` / `frames` / `reaches` / `representative`. |
| `analyses/frame_hazards.py` | NEW. Pairing by concrete storage-id intersection over the frame graph; `(at, gap)`. |
| `analyses/dep_tokens.py` | REWRITTEN on union-find over storage ids. |
| `analyses/fence_regions.py` | REWRITTEN. Ends-only placement, per-operand cover, `separating` / `separated` / `fences_of`, `_assert_covered`. |
| `analyses/lds_buffers.py` | Thin adapter: `LdsStorage` x `FrameMap.representative`. |
| `analyses/gen_reaching.py` | DELETED. `RELATIVE_BASE`, `is_relative`, `Reaching` gone with it. |
| `analyses/lds_hazards.py` | DELETED. `_reachable` moved to `cfg.reachable`; `_provably_disjoint`, `Unresolved` gone. |
| `verify.py` | `_check_token_reuse` re-expressed on the shared `separated`; the block-local scan is gone. |
| `tool/fence_token_check.py` | Same: one separation predicate, `#416` fixed, `_paths` deleted. |
| `passes/hoist_copies.py` | Rule 3 intra-class clamp added. |
| `passes/tokens.py` | Reads `FrameMap.generation` at the representative frame. |

**Suites**: `test_gir` + `test_loopmodel` + `test_tdm_split` = 1154 passed, 2 xfailed.

**Emitted asm, MAF bf16 (10 kernels, CPU-only generation, HEAD `c552b9ff83e` vs this tree).** Same
10/12 solutions. In-loop sync is UNCHANGED on both arms -- ULM0 `s_wait_tensorcnt [1,1,1]` with 18
barriers, ULM1 `[0]` with 6. What changed is the naming and the fence shape: a fused copy names
`sync LDS [0, 4]` where HEAD named `[0, 1, 6, 7]` (the per-slot refinement), and the prologue's one
`fence A+B` becomes a `fence B` and a `fence A`. Body length moves by 1-2 instructions.

**No new refusals anywhere.** `loopmodel_mxf8` (28 blocks), `loopmodel_bf16` (24) and
`loopmodel_f8` (6) report solution counts IDENTICAL to HEAD at both gates. The `LoopModel`
characterization suite fails the same 24 tests on both trees (they need the real conftest).

---

## 4. Migration order

Each step keeps the emitted GIR + asm byte-identical except where stated, and the suites green.

- **Step 0 — goldens.** `Lowering` has no characterization suite (only `LoopModel` does). Capture
  rendered GIR + fence marks + emitted `.s` over the MAF and mxf8 matrices first. Without this the
  rework is unfalsifiable.
- **Step 1 — fix #416.** The separation oracle must stop accepting an intermediate-block fence.
  Do this *before* anything else or every later step is checked by a hole.
- **Step 2 — `LdsStorage`.** Add it; make `LdsBufferIds` a thin adapter over it so nothing else
  moves. Byte-identical.
- **Step 3 — `FrameMap`.** Add alongside `GenReaching`; assert on every config that
  `FrameMap.touches(ref, f)` agrees with `GenReaching.of(ref)` wherever the latter is *not*
  relative. That equivalence check is the proof the map is right.
- **Step 4 — `FrameHazards`.** Add alongside `LdsHazards`; assert the edge sets agree modulo the
  `(at, gap)` vs `distance` encoding. Expect genuine disagreements where `is_relative` forced an
  `Unresolved` — those are the wins, and each is reported, not absorbed.
- **Step 5 — `DependenceTokens`** on derive+verify. **First step that may change output.**
- **Step 6 — `FenceRegions`** on derive+verify. Second step that may change output.
- **Step 7 — delete** `GenReaching`, `LdsHazards`, `RELATIVE_BASE`, `is_relative`,
  `_provably_disjoint`, `_one_absolute_frame`, `Unresolved`'s frame arms.
- **Step 8 — hardware run**, MAF + mxf8.

---

## 4b. Order independence — `HoistCopiesPass` must not be a premise

The design must produce a correct result **whether or not the hoist has run**, and must not encode
the hoisted shape anywhere. Two rules make that hold.

**Rule 1 — every position-derived fact is recomputed after any body mutation.** Positions (`pos`,
fence slot, `same_trip`, `in_program_order`) are properties of the *current* body, not of the
program. `HoistCopiesPass` calls `prog.bump()`, so `FrameMap` / `FrameHazards` / `DependenceTokens` /
`FenceRegions` must all be invalidated and re-derived after it. Nothing in the new layers may cache
a slot index across a mutation.

Note what changes when a copy moves: a same-trip edge whose producer was *before* the copy and is
now *after* it flips `in_program_order`, which flips `_admissible_slots` from the `(a, b]` interval
to the wrap union. That is a genuine change of obligation, not a perturbation — which is exactly why
the derivation must follow the final order rather than predict it.

**Rule 2 — the verifier is a pure function of `(body, frames)`.** `verify(body, frames, stamps,
fences) -> [uncovered]` takes the body as given and answers whether the chosen tokens and fence
positions cover every hazard in every frame. It does not know what order the body "should" be in.
That single property buys three things:

- the design works with the hoist off (fewer in-flight copies, more fences, still correct);
- it works with the hoist on (grouped copies, fewer fences, still correct);
- it works with a *future* scheduler, because moving the hoist earlier or later in the pipeline only
  changes the input to the same function.

**Ordering constraint that does hold.** `HoistCopiesPass` decides issue points and therefore how
many loads are in flight at a fence — the thing the emitted `s_wait_tensorcnt` counts. So it must
run **before** token stamping and fence placement, which is where it sits today
(`Hoist → Tokens → CollectPending(FenceRegions) → Placement → ApplyMarks`). The new layers keep that
position. The point is that they must not *depend* on it: a config with the hoist disabled must go
through the same derive-and-verify and come out correct, just with a different fence count.

**Rule 3 — the hoist must preserve INTRA-CLASS ordering.** Copies keep their relative order among
copies; reads keep theirs among reads. Cross-class interleaving is free, intra-class inversion is
not: two copies to the same ring in the wrong order write the buffers in the wrong sequence, and no
token or fence downstream can express that the order was already wrong.

Today's `hoisted()` preserves it only by accident. `group=True` collapses every copy onto one slot,
so their original order survives the `sorted(at)` insertion — but with `group=False`, or for the
reads under `reads=True`, `_earliest` is evaluated per instruction and a later movement can be given
a strictly earlier floor than an earlier one, inverting them. The fix is a running clamp per class,
in original program order:

```
for cls in (copies, reads):
    prev = -1
    for i in sorted(indices of cls):
        at[i] = max(at[i], prev)     # never land before a same-class movement that precedes you
        prev  = at[i]
```

Monotone by construction, and it degrades to today's behaviour whenever the floors were already
monotone — which is why the current code has not been caught by it.

**Consequence for `_earliest`.** `HoistCopiesPass._earliest` reads `LdsHazards`/`RegHazards` and
keeps a movement behind same-trip in-order readers of the storage it overwrites. Under the new model
it reads `FrameHazards` instead, and the floor becomes the max over **every frame**: a copy may only
rise above a reader if it does so in all frames, since the emitted order is shared. That is stricter
than today and is the correct reading — today's version computes the floor in whatever frame the
symbolic arithmetic happened to land on.

---

## 5. Settled

1. **Buffer id = `(operand, region, ring)`, every component from LoopIR.** No GIR dataflow
   contributes to storage identity; the whole set is enumerable statically before any analysis runs.
   The frame then selects which member of the ring an access touches.

2. **Frames are per ring.** A "ring" is one rotating buffer set as LoopIR declares it — an operand's
   LDS ring, and an MX scale's ring, are different rings and may have different depths. A frame is
   therefore a **phase vector**, one entry per ring identity, not a single global index:

   ```
   Frame = { ring_id : phase in [0, depth(ring_id)) }
   advance(F) = { r : (F[r] + adv[r]) % depth(r) }
   ```

   A single scalar index over the lcm orbit is a valid *encoding* of that vector, and collapses to
   the vector when every ring has the same depth and advances by 1 — the common case. Keep the
   vector as the model so a ring with a different depth (`depth=3` scale ring beside a `depth=2`
   parent) needs no special case; encode it as a scalar index only for iteration.

3. **Verification cannot fail; it is an oracle, not a repair loop.** LoopIR already covers every
   hazard and GIR only adjusts placement and naming under that rule, so a token or fence is
   *constructed* to satisfy the hazard. A per-frame check that reports an uncovered edge means the
   derivation is wrong — a defect to fix, never a trigger to widen. §2.4/§2.5 assert; they do not
   iterate.

4. **The representative frame comes from the peel and the trip index.** The peel fixes the phase the
   first steady trip enters on; trip `t` runs at `advance^t(F_after_peel)`. So `f0 = F_after_peel`,
   and the frame set of the steady block is that orbit — not an arbitrary "frame 0".

5. **A frame is per block ENTRANCE, and every one is considered.** `frames(b)` is the union over
   incoming edges of the advanced phase of the source — a fixpoint, not a formula. A join carries
   one frame per arm; the short path into `drain0` contributes its own beside `steady`'s. Deriving
   on `f0` is allowed; **checking is over the whole set**, and nothing may be assumed about a frame
   because "the loop usually enters on it".

# SGPR state in the TensileLite mainloop — what exists, who owns it, and what an own path owes

**Scope.** Every SGPR the mainloop phases set up or mutate: loop counters, stagger/wrap, global-read
increments, buffer descriptors, LDS addresses, swap state, and the TDM descriptors. **StreamK is
excluded** (we do not use it) — `skGrid`, `skTiles`, `CtaIdx`, `CtaEnd`, `ItersPerTile`,
`TotalItems`, `MagicNumber/ShiftItersPerTile` and everything in `Components/StreamK.py` are out.
Pure-epilogue state (`SrdD`, `SrdC`, `SrdBias`, `SrdE`, `SrdScale*`, `Alpha`, `Beta`) is noted but
not detailed — it belongs to #116, not to the mainloop split.

**Why this exists.** #232: when GIR emits its own path (a kept short arm, or a block created by
critical-edge splitting) that path is not just the instructions GIR generated. TensileLite's phases
set up and maintain shared state around the region, and an own path that ignores it is broken, not
smaller. This report is the enumeration that task demands, done once, from the code.

Derived by enumerating all 155 `defineSgpr(...)` sites and scanning def/use with the enclosing
function across `KernelWriterAssembly.py`, `KernelWriter.py` and `Components/*.py` (StreamK
excluded). Counts below are from that scan.

---

## 0. The headline: we already interoperate, through four primitives

`Tensile/Lowering/gir_to_rocisa.py` writes **no SGPR directly**. Its entire interaction with machine
state is four scaffold calls (plus one query):

| call | what it moves |
|---|---|
| `writer.tdmIncrementGir(...)` | the TDM global-read descriptor (address advance + StaggerU wrap) |
| `writer.tdmSwapLdsOffset(...)` | the TDM LDS destination offset (double-buffer toggle) |
| `writer.localReadSwapOffsets(...)` | the LDS read pointer (`LocalReadAddr` / `LDSBufferReadInc`) |
| `writer._syncThreads(...)` | the barrier |
| `writer.isTdmWaveSeparated(...)` | *query only* — no state |

That is the entire surface. **The rule to preserve: GIR states WHERE a state change happens; the
scaffold owns HOW.** An own path should extend this list, not start writing SGPRs — see §6.

---

## 1. Loop counters

### `LoopCounter<Char>` (`loopCounterName()`, `KernelWriterAssembly.py:324`)
*writes 6 / reads 11.* One per summation index; the unroll one is the mainloop's.

- **Init** (`calculateLoopNumIter`): `LoopCounterL = SizesSum >> log2(DepthU)`, i.e. **T = K / DU**,
  then GSU-adjusted via `GSU.calculateLoopNumIter`.
- **Mutation** (`closeLoop`): **counts DOWN** — `SSubU32(loopCounter, loopCounter, 1)` — and exits
  on `counter <= endCounter` with `endCounter = kernel["PrefetchGlobalRead"]`, i.e. **T − M trips**.
- **Read by**: `declareStaggerParms`, `openLoop`'s entry ladder, `closeLoop`, `openSumAtLeastUnroll`.

> **This is the divergence #231 is about.** GIR modelled the loop as an up-counter comparison
> (`iter < T−M`, from 0, step +1). The machine counts down to a limit. Same trip count, different
> arithmetic. GIR now states `LoopBack(trips=T−M)` and leaves the counter representation to
> emission — which is what makes an own path's branch *lowerable* rather than a translation between
> two conventions.

### `OrigLoopCounter` (`KernelWriter.py:9717`)
*writes 4 / reads 10.* **Overloaded, and the overload is a trap.**

- In the **main loop**: a copy of the initial `LoopCounter` (`calculateLoopNumIter`) — i.e. T. Used
  by `declareStaggerParms` to decide `StaggerU = 0` when the trip count is too small, and by the
  NLL/HalfPLR paths (`SCmpEQU32(OrigLoopCounter, 0)` — "was the main loop executed at all?").
- In the **tail loop**: **repurposed** — set to 0 and *incremented* to count localRead increments
  (`KernelWriterAssembly.py:7703`, "repurpose to count each localRead increment").

An own path that runs between those two régimes must know which meaning is live. Anything reading
`OrigLoopCounter` after the tail loop has started is reading a different quantity.

### `LSUTailLoopOffset` — LocalSplitU tail bookkeeping. Out of scope while LSU is unsupported.

---

## 2. Stagger and wrap — the most position-sensitive state

### `StaggerU` (kernarg, `KernelWriter.py:9691`) *writes 4 / reads 4*
Packed field, unpacked in `declareStaggerParms`: `& 0xFF` = the stagger value, `& 0x1F00` = stride
shift, `& 0xE000` = mapping mode. **At `PrefetchGlobalRead >= 3` only**, it is force-zeroed when
`OrigLoopCounter <= PGR-1` (`KernelWriterAssembly.py:6339-6342`), so the stagger is silently
disabled on short trips *for PGR3+*. PGR1/PGR2 have no such clause.

### `StaggerUIter` (`KernelWriterAssembly.py:786`) *writes 6 / reads 19*
The iteration at which the global-read address **wraps** back to the tensor base.

- **Set up** in `declareStaggerParms`, then adjusted in `calculateStagger`:
  *"Subtract (PGR-1); StaggerUIter now contains target iteration to wrap"*, and
  *"StaggerUIter += PGR only if StaggerU > 0"*.
- **Read by**: `calculateStagger`, `removeStagger`, `globalReadIncrement`, **`tdmIncrementAB`,
  `tdmIncrementABWaveSperated`, `tdmIncrementGir`** — i.e. every address-advance path, ours
  included.

### `WrapU{A,B,MXSA,MXSB,Metadata}` (`:792`) *writes 3 / reads 28*
Bytes to add to the SRD to reset the address from the last iteration back to the base. Written only
in `calculateStagger`; read by every increment path and by `removeStagger`.

> **Relevance to us:** `tdmIncrementGir` already consumes `StaggerUIter` and `WrapU*`. This is the
> "wrap lead" #188 derived rather than calibrated. An own path that advances an address **must** go
> through the same primitive or it will desynchronize the wrap.

### `StaggerUIterDTV` — PGR≥2 + one-sided DTV only. Out of scope (DTV is out of scope).

---

## 3. Global-read increments

### `GlobalReadIncs{A,B,MXSA,MXSB,Metadata}` (`:805`) *writes 7 / reads 17*
Per-tensor byte stride added to the SRD per unroll iteration. Written by `graIncrements` /
`graIncrementsCommon` / `graIncrementMask`; read by `globalReadIncrement`, `calculateStagger`,
`calculateLoopNumIter`, and the TDM increment paths.

Note `calculateLoopNumIter`'s tail path does
`SCSelectB32(tmp, 0, GlobalReadIncsA)` on *"completely skipped unroll loop?"* — the increment is
conditionally forced to zero based on `OrigLoopCounter == 0`. **A path that skips the main loop
changes what the tail's addressing does.** Directly relevant to the `T < M` arm.

### `GL2PrefetchInc{A,B,...}` (`:903`) *writes 1 / reads 10*
The PrefetchGL2 third pointer (#216). Written by `setIncrement`; read mostly in `_loopBody`.

---

## 4. Buffer descriptors and limits

### `Srd{A,B,MXSA,MXSB,Metadata}` (`:754`) *writes 75 / reads 61*
4-SGPR aligned buffer descriptors. Set up in `computeLoadSrd`; advanced by `incrementSrd`; the tail
re-points them via `setTailSrd`. **Only defined when the tensor is NOT on TDM**
(`if not kernel["enableTDMA"]`) — on our gfx1250 TDM path A/B have no SRD at all; the TDM descriptor
groups replace them.

### `ShadowLimit{A,B,...}` (`:769`) *writes 9 / reads 21*
64-bit shadow of the buffer limit, needed when the real limit would overflow 32 bits
(`use64bShadowLimit`). Maintained alongside the SRD by `incrementSrd` / `setTailSrd`;
`restoreShadowLimitSrd` puts it back. `checkpointDescriptorState` / `restoreDescriptorState` exist
to save and restore this pair — **evidence that the scaffold already treats descriptor state as
something a divergent path must checkpoint.**

---

## 5. LDS addressing and swap state

| SGPR | site | writes/reads | meaning |
|---|---|---|---|
| `LocalWriteAddr{A,B,…}` | `KernelWriter.py:9857` | 10 / 16 | LDS write address; `lwaFirstOffset` sets, `localWriteSwapXOR` toggles, `localWriteResetOffsets` restores |
| `LDSBufferReadInc` | `:823` | 5 / 3 | read-pointer increment for the ≥3-buffer case (`lraAddressesInitFor3LDSBlk`, `localReadSwapOffsets`) |
| `LDSBufferWriteInc` | `:824` | 4 / 4 | write-pointer counterpart (`localWriteAddRound`) |
| `SwapCommon` | `KernelWriter.py:9869` | 4 / 3 | shared double-buffer selector (`localWriteSwapCommon`); read by `directToLdsM0Update` |
| `Swap{A,B,MXSA,MXSB,Metadata}` | `:9872` | 0 / 0 | **declared but never referenced** — see §7 |

`papDtlSaveLdsBank` / `papDtlRestoreLdsBank` save and restore `LocalWriteAddr`, `LDSBufferReadInc`,
`LDSBufferWriteInc` and `SwapCommon` together — again, the scaffold's own precedent for
checkpointing a phase's state across a divergent path.

---

## 6. The TDM descriptors (our actual load path)

Declared in `defineTdmSgprs` (`:930-1016`). Field semantics are in the declaration comments:

| SGPR | size | meaning |
|---|---|---|
| `tdmAGroup0` / `tdmBGroup0` | 4 | descriptor group 0 |
| `tdmAGroup1` / `tdmBGroup1` | 8 | descriptor group 1 |
| `tdmAGroup2` | 4 | *"iter_count / lds_inc / global_inc when iterate_enable=1"*; Group 3 is aliased to it because the 4-operand form needs a valid name |
| `tdmMXS{A,B}Group{0,1}` | 4/8 | MX-scale descriptors |
| `tdmMetadataGroup{0,1}` | 4/8 | sparse metadata |
| `tdmABIncs` | 1 | shared A/B increment when descriptors are **aliased** (multi-wave) |
| `tdmMXSAMXSBIncs` | 1 | the MX counterpart |
| `tdm{A,B}{Global,Lds}SplitIncs` | 1 each | single-wave TDMSplit per-tensor increments |
| `tdmLdsAddr{A,B}` | 1 | LDS address tracking for aliased descriptors with subtile double-buffering |
| `tdmLdsSwapMask{A,B}` | 1 | the double-buffer toggle mask |

**Two facts that constrain any own path:**

1. **Aliasing is conditional.** *"Alias B descriptor onto A for multi-wave to reduce SGPR
   pressure"* — at `NumWaves ≥ 2` B's descriptor lives in A's SGPRs. So "increment B" is not a
   fixed register; it depends on `NumWaves` and on subtile mode. This is exactly why `copy_unit`
   (#190) names a Φ movement by its **member tuple** rather than one operand.
2. **Multi-wave TDMSplit persists nothing.** *"recomputes the LDS/global split increments … 
   transiently at point of use (`_tdmSplitMultiWaveInc`); nothing is persisted here."* A path that
   re-emits a split load must recompute, not read.

---

## 7. Findings worth acting on

**F1 — `Swap{A,B,MXSA,MXSB,Metadata}` are declared and never used.** 0 writes, 0 reads across the
whole writer; only `SwapCommon` is live. Either dead SGPR allocation (wasted budget, and SGPR
pressure is why B is aliased onto A in the first place) or an unfinished per-tensor swap. Worth a
separate look; not ours to delete blind.

**F2 — `OrigLoopCounter` means two different things** (§1). Any own path spanning the main→tail
boundary must know which. This is the single most likely source of a subtle bug in a split path.

**F3 — the scaffold already has checkpoint/restore precedent** for exactly the state a divergent
path disturbs: `checkpointDescriptorState`/`restoreDescriptorState` (SRD + ShadowLimit) and
`papDtlSaveLdsBank`/`papDtlRestoreLdsBank` (LDS addresses + swap). **If an own path needs its own
state, these are the mechanism to reuse rather than reinvent** — and their existence says the
scaffold authors already hit this problem on the PAP path.

**F4 — at PGR3+ the stagger is silently disabled on short trips**, and only there:
`if PrefetchGlobalRead >= 3: SCMovB32(StaggerU, 0)` when `OrigLoopCounter <= PGR-1`
(`KernelWriterAssembly.py:6339`). So the `T < M` arm runs in a *different addressing régime* than
the steady path — but **only for PGR3+**, which is exactly the configuration set our fold analysis
already reports as SPLIT. Two independent reasons the PGR3 arms cannot be folded, arrived at from
opposite directions (def-use pairing vs addressing mode), which is mutual corroboration rather
than coincidence. PGR1/PGR2 have no such clause, so the régime is uniform there.

*(An earlier draft of this line stated the disable unconditionally. It is guarded by PGR>=3; the
guarded form is the one that matters, because it lines up with the SPLIT set.)*

**F5 — the GIR boundary is already correct and should be preserved.** Four primitives, zero direct
SGPR writes (§0). #232's per-component decision has a strong default: **interoperate via a
primitive**, and add a primitive when one is missing, rather than emitting SGPR arithmetic in
`gir_to_rocisa`.

---

## 8. The per-component decision (#232 step 2)

Step 1 was the enumeration above. This is the answer it forces, one row per component: does an own
path **emit** it, **interoperate** through a scaffold primitive, or **refuse** because we cannot
supply it correctly? "Refuse" is a real answer, not a placeholder — see §9.

| component | §  | decision | why |
|---|---|---|---|
| loop counter `LoopCounterL` | 1 | **interoperate** | GIR states a trip count (#231); the counter's direction, init and limit are the phase's arithmetic. Two authorities for one register is how #228 happened |
| `OrigLoopCounter` | 1 | **refuse to touch** | overloaded across the main→tail boundary (F2). We neither read nor write it; a path that needs it must first make the two meanings distinct |
| `StaggerUIter`, `WrapU*` | 2 | **interoperate** | already consumed inside `tdmIncrementGir`. The wrap lead is derived (#188), not calibrated, but the *application* stays in the primitive |
| `StaggerU` zeroing at PGR3+ | 2 | **refuse to reproduce** | F4: the short arm's addressing régime differs from the steady one. An own short path that re-emits loads without reproducing this is wrong — and PGR3 is exactly the SPLIT set, so this is live, not hypothetical |
| `GlobalReadIncs*` | 3 | **interoperate** | written by `graIncrements*` before the region; we only trigger the advance |
| `GL2PrefetchInc*` | 3 | **interoperate (primitive missing)** | #216: the third pointer exists in the scaffold but no GIR primitive advances it. Adding the primitive is the work; emitting SGPR arithmetic is not |
| `Srd*`, `ShadowLimit*` | 4 | **not applicable on our path** | TDM replaces the SRD for A/B (`if not enableTDMA`). Applies again only if a non-TDM tensor enters ULM |
| LDS write addr / `SwapCommon` | 5 | **interoperate** | `tdmSwapLdsOffset` / `localReadSwapOffsets`. GIR says WHERE, the scaffold says HOW |
| TDM descriptor groups | 6 | **interoperate** | aliasing is conditional on `NumWaves`, so "increment B" is not a fixed register — the reason `copy_unit` (#190) names a movement by member tuple |
| multi-wave TDMSplit incs | 6 | **recompute at use** | the scaffold persists nothing (`_tdmSplitMultiWaveInc`); a re-emitting path must recompute. Blocked on #212 |
| the barrier | 0 | **interoperate** | `_syncThreads`. We own the memory token and the copy order only — never barrier placement |
| the branch itself | — | **emit (pending #150)** | the one component GIR is meant to own. Today the scaffold emits it and GIR labels it advisorily (`ScaffoldMapPass`) |

**The shape of the table is the finding.** Eleven of twelve rows are *interoperate* or *refuse*; one
is emit, and it is the one not done yet. That is F5 restated as a rule: an own path is a path that
**chooses its control flow**, not one that reimplements machine state. Any future row that reads
"emit" for a state component should be treated as a design smell and argued explicitly.

---

## 9. Making an unsupplied component LOUD (#232 step 3)

A refusal only helps if it is audible. The measured failure mode was the opposite:

> `FenceRegions` places real `fence` marks into the kept `T < M` arm — **2 at PGR1, 6 at PGR3** on
> the two multi-agent fixtures (`FUSED_XAGENT`, `UNFUSED_MULTIAGENT`) — and nothing emits them.
> `_girEmitStage` requests blocks by PHASE NAME and never asks for `short{i}`. Single-agent
> fixtures drop nothing; the 2/6 are exactly the cross-agent Φ discharge.

This is inert *today* — the arm is not emitted at all and TensileLite's own `toPGR1` path serves
`T < M` — but it is indistinguishable from a forgotten block, and it becomes a live hazard the
moment #150 or #222 makes those blocks emittable.

The mechanism, in three parts:

1. **`Block.model_only`** (`gir/nodes.py`) — a block GIR reasons over that NO backend emits.
2. **`FoldShortPathPass`** sets it on a kept arm and records the *reason* in
   `meta['short_loop']['model_only']`. It cannot record the count: it runs FIRST, and the marks
   arrive later.
3. **`RecordUnemittedPass`** (near the end of the pipeline) counts the discarded marks and their
   kinds; **`verify_gir`'s G-EMIT** holds the two against each other and fails if a block is flagged
   without a recorded reason, or if the count is stale because something added a mark afterwards.

G-EMIT does not forbid the drop — TensileLite legitimately owns that path. **It forbids the drop
being invisible.** It is also self-testing: the first run of the check reported *6 real marks
against a recorded 0*, which is what moved the counting to its own end-of-pipeline pass.

What G-EMIT does **not** do, stated so it is not read as more than it is: it does not re-prove the
fold. Once `FoldShortPathPass` deletes an arm there is nothing left to re-derive legality from; the
proof lives in `ShortPathFold` and the pass carries the obligations forward. G-EMIT checks the
*kept* side only.

---

## 10. What this means for the open tasks

- **#232** (what an own path owes): the enumeration is §§1-6. The per-component answer is
  overwhelmingly *interoperate*, not *re-emit* — the counter (§1), the wrap (§2), the descriptors
  (§4, §6) are all shared mutable state with existing owners. The realistic list of things an own
  path must arrange: enter with the counter in the régime the phase expects; go through
  `tdmIncrementGir`/`tdmSwapLdsOffset`/`localReadSwapOffsets` for every state change; and if it
  needs private state, use the checkpoint/restore pair (F3).
- **#231** (trip count): §1 is the evidence. The machine counts down; GIR now states a count.
- **#222 / #150**: F4 says the short path's addressing régime differs from the steady path's, which
  is an additional reason the two arms are not interchangeable beyond the def-use pairing analysis.
- **#216** (PrefetchGL2): §3 shows `GL2PrefetchInc*` already exists and is read in `_loopBody` —
  the third pointer has scaffold support to interoperate with.

---

*Method: `defineSgpr` enumeration + a def/use scanner resolving each reference to its enclosing
function, over `KernelWriterAssembly.py`, `KernelWriter.py`, `Components/*.py` minus StreamK.
Counts are reference counts, not dynamic frequencies. Semantics are quoted from the code's own
comments where available; where inferred, that is stated.*

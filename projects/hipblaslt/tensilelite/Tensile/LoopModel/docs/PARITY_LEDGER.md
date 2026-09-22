# ULM parity ledger — every config ULM0 accepts and ULM1 refuses

**The bar (#259/#265).** Not "the yaml matrix is green" — a yaml is a subset of what ULM0 accepts,
so a green matrix can never establish parity. The bar is *ULM1 emits everything ULM0 emits*, and
the honest way to hold it is to enumerate the refusals and justify each one.

A ULM1 refusal is silent in the worst way: the solution stops appearing in `Actual Solutions`, the
ULM0 twin still generates and passes, and the matrix reads green with the arm under test quietly
absent. That is the `PrintWinnersOnly` trap, and it is why `KernelWriter.py:3786` makes the GIR
semantic gate a loud `ValueError` rather than a `Solution.py` reject.

ULM1 can refuse in exactly two places:

| | where | shape |
|---|---|---|
| **STATIC** | a `reject(...)` in `Solution.py` guarded on `UseLoopModel` | finite, enumerable — this table |
| **DYNAMIC** | a raise on `translate → θ → build_gir → check_plan` | swept; see §3 |

## 1. The static ledger

Three classes, and the distinction is the whole point of the document:

- **SCOPE** — ULM will never do this. A deliberate non-goal, not a gap. Do not plan around it lifting.
- **DEFERRED** — a real gap with a named owner. Every one of these is a parity debt.
- **DERIVED** — a correctness precondition that is genuinely true on the ULM path and genuinely not
  on the scaffold path, so the asymmetry is the model being right rather than ULM being narrower.

| # | `Solution.py` | condition | class | why |
|---|---|---|---|---|
| 1 | 787 | `ISA != (12,5,0)` | SCOPE | the feature is gfx1250-only |
| 2 | 791 | `not EnableMatrixInstruction` | SCOPE | WMMA path only |
| 3 | 806 | `MacDataType` not bf16 / 8-bit float | DEFERRED | fp6 (0.75 B) and fp4 (0.5 B) each have their own scaffold local-read branch with a different register packing; the ULM read leaf's fragment table is checked against neither |
| 4 | 822 | `MXBlockA != MXBlockB` | DERIVED | the WMMA carries ONE `block` modifier for both scale operands, so there is no faithful emit — an ISA fact, not a ULM one |
| 5 | 827 | one-sided MX | DEFERRED | the absent side needs the scaffold's `ValuMXSDummy`, whose width follows the present side's block |
| 6 | 842 | `_ScheduleIterAlg != 0` (SIA3) | SCOPE | SIA3 is the hand-tuned schedule; θ **is** the schedule, so it is never wanted |
| 7 | 852 | `UseSubtileImpl` | SCOPE | subtile is a separate mainloop; mutually exclusive by design |
| 8 | 860 | `InnerUnroll != 1` | DEFERRED | the GIR wmma leaf emits one wmma per `(m,n,k)` with `iui=0` |
| 9 | 865 | `DirectToVgprA/B` | SCOPE | pre-TDM staging route; gfx1250 does not use it |
| 10 | 871 | `LocalSplitU > 1` | DEFERRED | the GIR body is single-split |
| 11 | 894 | `DirectToLds` | SCOPE | TDM supersedes it on gfx1250 |
| 12 | 902 | `TDMInst != 3` | SCOPE | the non-TDM local-write path adds a write step GIR does not model (the write side is #116) and GIR owns LDS fence placement |
| 13 | 1078 | `VectorWidth` does not fit one TDMSplit region | DERIVED | VW sets the wave distribution over the tile; under a split it must be measured against the tiles ONE REGION holds (#237) |
| 14 | 5000 | `UnrollMajorLDS=0 and not LDSTrInst` | DEFERRED | the read leaf implements the transpose and general unroll-major reads only (#91); a tile-major layout is neither |
| 15 | 5093 | DU-half TDMSplit without `NoTailLoop` | DEFERRED | the DU split cuts the shared K axis the tail also shrinks, and the tail's `tensor_load` is scaffold-owned (#94) — two authors of one descriptor |
| 16 | 5793 | `PrefetchGlobalRead > NumLdsBlk` | DERIVED* | §2.4 `S ≥ δ`: chunk `NumLdsBlk` lands on chunk 0's buffer before it is read. See the caveat below |

**Not in the table, deliberately:** `Solution.py:748` rejects `LoopOrder != KMN` *without*
`UseLoopModel`. That is the inverse direction — a ULM1-only **capability** — and by construction can
never be a parity gap.

**Also not a gate:** the MX TileSpan half-wave scale layout (`Solution.py:3020`) has **no rejection
at all**. The geometry is derived from `VectorWidth`, the read leaf models both halves, and
`emitWmmaTile` asks the scaffold's own `mxsTileSpanScaleSel` for the (register, selector) pair, so
the load and consume sides cannot disagree. #265's subject is supported, not gated.

### The one caveat worth flagging — entry 16

Entry 16 is scoped to `UseLoopModel` on the argument that the scaffold path is *genuinely different*:
without TDM the global read lands in G2L **registers** and the LDS write is a separate serialized
step, so `PrefetchGlobalRead` is the depth of the register prefetch and not the LDS ring's δ.
`1LDSBuffer=1` with `PGR=2` is a shipped non-TDM configuration for exactly that reason.

That argument is sound for a **non-TDM** scaffold kernel. It does not obviously cover a **ULM0 kernel
with `TDMInst=3`**, which writes LDS directly and therefore has the same two-depths-are-one-quantity
property that makes the ULM1 case wrong. If such a kernel is reachable, the asymmetry is not ULM
being conservative — it is the scaffold missing a precondition ULM enforces.

This is recorded rather than acted on: it is a *scaffold* question, not a ULM parity gap, and
changing a shipped non-ULM gate is out of scope for #259. Filed as the open question below.

## 2. What the table means for #259 / #265

Nine entries are SCOPE or DERIVED — permanent, and correct. **Five are DEFERRED**, and those five
are the actual remaining parity debt:

| entry | owner |
|---|---|
| 3 fp6 / fp4 inputs | no task yet |
| 5 one-sided MX | no task yet |
| 8 `InnerUnroll > 1` | Phase 4/5 (#90) |
| 10 `LocalSplitU > 1` | no task yet |
| 14 tile-major LDS layout | #91 follow-on |
| 15 DU-split + tail | #94 |

So MX parity (#259) and MX TileSpan (#265) are **not blocked by anything MX-specific**. Entry 5
(one-sided MX) is the only MX entry, and one-sided MX is not what either task is about.

## 3. The dynamic half — MEASURED 2026-08-25

`Solution.py` acceptance is necessary but not sufficient — a config can pass every gate and still
raise on the decode/lower path, which is the failure mode #297 reported. Swept at the **product
layer** (`bridge.kernel_to_params`), because a dropped Solution key is invisible below it.

**8794 configs** — 4 dtypes (bf16, fp8, mxf8 at MXBlock 32 and 16) × wave-tile `{2×2, 4×4, 2×8,
8×7}` × wave-group `{1×1, 2×1, 2×2}` × PGR `{1,2,3}` × PLR `{0,1,2}` × all 11 loop orders ×
`TDMSplitA/B` `{0,1,2}²` × VectorWidth `{1,2}` — through
`bridge → params_to_theta → build_gir → check_plan + walk_violations`:

```
swept 8794    clean 6364    refused 2430    distinct refusal signatures: 1
```

**One** signature, and it is entirely `PrefetchGlobalRead >= 3`:

```
AssertionError: SwapRegions: block 'drain1' is a JOIN whose predecessors leave
                the read pointer for 'A' at different generations
```

Bisected: PGR ≤ 2 clean, PGR ≥ 3 always asserts — independent of PLR, NumLdsBlk, dtype, loop order,
split and shape. Removing the PGR3 configs leaves exactly 6364, the clean count, so **every config
outside that region passes**.

**Over the reachable space the dynamic half is 100% clean**, because PGR ≥ 3 is unreachable under
UseLoopModel (§5). That is the result #259 and #265 were waiting for, and it is the first evidence
about them that is not matrix evidence.

Standing test: `Tensile/Tests/unit/test_parity_sweep.py`.

## 4. Open question (not a #259 blocker)

**Is a `TDMInst=3`, `1LDSBuffer=1`, `PGR>=2` kernel reachable on the ULM0 arm?** If yes, entry 16's
precondition applies to it too and the scaffold does not enforce it. Answering needs the ULM0 arm's
own solution enumeration, not a ULM question.

## 5. A LATENT trap: PGR ≥ 3

`Solution.py:6324` makes `PrefetchGlobalRead >= 3` require **both**:

| requirement | ULM's answer |
|---|---|
| `DirectToLdsA and DirectToLdsB` | rejected — ledger entry 11 (SCOPE) |
| `_ScheduleIterAlg == 3` | rejected — ledger entry 6 (SCOPE) |

Two *independent* SCOPE gates, so the region is doubly unreachable and the assertion above is
latent, not live. `gir_to_rocisa.py:498,512` already says as much in prose.

**But #104 is marked done as "PGR>=3 drain-step mapping for UseLoopModel".** The mapping is gated
out and, measured, does not survive `SwapRegions` — `drain1` exists only once the peel depth exceeds
2, so this is the drain-multiplicity case still open as **#148**. Recorded here so the two are not
believed independently: if either SCOPE gate is ever lifted, PGR ≥ 3 fails immediately and #148 is
the work, not #104.

`test_pgr3_is_unreachable_under_ulm` pins both halves — the two `Solution.py` requirements and the
two ULM refusals — with `test_the_decoder_really_does_assert_at_pgr3` as its non-vacuity guard. A
decoder limitation held up by someone else's reject should not be rediscovered by a hardware run.

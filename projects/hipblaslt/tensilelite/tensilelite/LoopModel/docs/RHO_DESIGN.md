# ρ as the single agent authority — design note

**Status:** **R1–R4 LANDED 2026-08-25.** ρ is the single agent authority; the hops are projections
of it. Emit-neutral throughout: 34/34 kernels byte-identical to the pre-ρ build.
**Paper:** the type and both open questions were CONFIRMED by paper rev 2026-08-25 — see §8.
**Scope:** `Theta.rho` becomes the one place agent facts are stated; `agent_served_modes()`,
`Hop.rho_span` and `Hop.region_agent_relative` become derivations of it.
**Supersedes the framing of:** #165 ("ρ: optional narrowing selector"), which assumed ρ's only
remaining job was a *narrowing* claim. That framing is wrong, and §1 says why.

**Correction (2026-09-17):** the wave-level TDMSplit resort described below was itself a model
error and has been removed. TDMSplit factorizes a coordinate axis
(`WaveTile = split × inner`); it is never traversed by waves instead of by the coordinate.
Within LoopModel, `wave_region_span`, derived from VW/wave layout/LDSSegmentInterleave, now
affects only the read's possible storage-region aliases and therefore its wait/barrier edges
(LDSSegmentInterleave still affects physical LDS addressing in the lowering layer).
`Hop.region_agent_relative` records that alias fact directly; it is not a projection of ρ.
ρ remains authoritative for genuine agent transforms such as sub-wave carrier distribution and
copy-agent sharing. The remainder of this note records the earlier migration and should be read
with this correction.

---

## 1. The finding: our ρ is not the paper's ρ

The paper types ρ twice, and both are structured:

> **§2.8 line 213** — The field↔move map: … `ρ`←`resort`(2)+`specialize`(8)
> **§2.8 move 2** — `resort(mode → agent)` — assign a looped mode to a hardware agent
> (parallelize) instead of traversing it.
> **§2.8 move 8** — `specialize(agent-mode, roles)` — partition agents into producer/consumer roles.
> **§2.9 line 244** — `ρ : agent assignment + role partition`

So the paper's ρ is **a map from looped mode to agent level**, plus a role partition over agents.

`theta.py` types it as something else entirely:

```python
rho: dict = field(default_factory=dict)   # ρ: op-class -> agent selector
#   "The value is opaque and only ever compared for equality — the model is address-opaque and
#    must not learn what a 'wave' is; two members with DIFFERENT selectors are on different
#    agents, and that is the whole content `agent_distributed` needs."
```

**That mistyping is the reason it has no readers.** An equality-only token can answer exactly one
question — *"are these two op-classes on different agents?"* — and `theta.agents > 1` already
answers it more cheaply and without a map. So ρ was built to answer a question that did not need
it, and the questions that *did* need it could not be phrased against its type. `translate.py:863`
sets `rho = {}` and nothing reads it; the only two other mentions in the tree are docstrings saying
so.

Meanwhile ρ's real content is live, in three per-hop places that never mention ρ:

| what | where set | what it is, in ρ terms |
|---|---|---|
| `Hop.region_agent_relative` | `translate.py:588`, `_waveA < aMT` | the region mode was `resort`ed to the **wave** level |
| `Hop.rho_span` | `KernelWriter.py:3722` → `ReadRho` → `translate.py:600` | a lane-mode tile was `resort`ed to the **half-wave** level, span 2 |
| `theta.agents` | `NumWaves` | the *cardinality* of the wave-level resort, with the map thrown away |

Three authorities for one field, none of them named ρ. This is the "two places computing one fact"
shape the review deck already calls the dominant bug class here.

---

## 2. Proposed type

```python
# ρ (§2.8 move 2 `resort` + move 8 `specialize`; §2.9 "agent assignment + role partition").
#
# resort: mode name -> the AGENT LEVEL that traverses it instead of the coordinate.
#         Levels are ORDERED coarse->fine and are the ONLY hardware vocabulary in the model:
#         the decoder compares and orders them, and never learns what one is made of.
Level = ("block", "wave", "subwave")      # ordered; extend at the fine end

@dataclass(frozen=True)
class Resort:
    mode:   str      # a looped mode of `tile`
    level:  str      # one of Level
    extent: int      # how many agents at `level` the mode is spread over (== mode extent)

@dataclass
class Rho:
    resort: tuple = ()      # tuple[Resort], one per resorted mode
    roles:  dict  = None    # `specialize` half — OUT OF SCOPE, see §5
```

`Theta.rho: Rho`. The value stops being opaque; what stays opaque is what a level *is*. The model
still never learns what a wave is — it learns only that `subwave` is finer than `wave`, which is
the fact every derivation below actually needs.

### 2.1 ρ is SUPPLIED, exactly as `S` is — and that is what keeps θ address-free

ρ is a **preset over θ**: a value from outside that θ reasons over but does not choose. That is the
established pattern in this decoder, and `translate.py:576` already names it for the movement
quantum:

> THE MOVEMENT QUANTUM IS SUPPLIED (§2.2), never derived here. … the bridge computes it where the
> target facts live and hands it over already in GEMM terms. **Same shape as `S`: a value from
> outside that θ reasons over but does not choose.**

So `getMxsTileSpanInfo` **stays at the target**, where `asmCaps` lives. The bridge hands over resort
entries in GEMM terms; `translate` assembles them into `Rho` alongside the wave-level entries it
derives from the tile/wave params it can already see. Nothing about the agent partition moves into
the model.

**This is the point of the whole change, not a concession.** With ρ mistyped, every consumer that
needed an agent fact had to reach for an *address-shaped* one and smuggle it in as a per-hop scalar:
`wave_region_span < split` is a lane-layout computation, and `getMxsTileSpanInfo` is an
address-layout query. Both are facts θ is forbidden to hold (§2.1, address-opacity), and both ended
up inside `Hop` because ρ could not carry them. **Set ρ up correctly and no address is needed
anywhere in θ** — the agent partition is stated once, as a preset, and the three derivations in §3
are pure arithmetic over modes and levels. The mistype did not merely leave a field unread; it
pushed address-flavoured facts into the model to compensate.

---

## 3. The three derivations

| consumer | today | derived from ρ |
|---|---|---|
| `agent_served_modes()` | union of `region_modes` over hops with `region_agent_relative` | `{r.mode for r in rho.coarser_than("wave")}` — **wave-and-coarser, not the whole map**; see below |
| `Hop.region_agent_relative` | `_waveA < aMT` | the hop's region mode is in `domain(resort)` at level `wave` or coarser |
| `Hop.rho_span` | `ReadRho`, `partner = 2 if span else 0` | `extent` of the `subwave`-level resort whose mode the hop's carrier group spans; `0` if none |
| `theta.agents` | `NumWaves` | **left alone in step 1** — see §5 |

`agent_served_modes()` becomes a one-liner and, more importantly, becomes *definitionally* right:
the paper defines an agent-served axis as one `resort` sent to an agent, and that is now literally
what the function returns. Today it rediscovers the same set from a per-hop boolean whose own
derivation (`wave_region_span < split`) is a different sentence that happens to agree.

**But it is wave-and-coarser, NOT every resort entry — this note's first draft had it wrong and R1
caught it.** Including sub-wave entries put `M_inner` into the agent-served set, which would have
subtracted a live tile axis from `reduction_modes()` and dropped it from the inner nest: #245's
failure, inverted. §2.2 (paper rev 2026-08-25) supplies the principle the measurement found — a
sub-wave axis is "a **data-distribution** axis …, **not** a synchronization scope … an operand
*may* carry a sub-wave axis in its presence", and §5.2's scope ladder starts at `wave`. An axis an
operand may still carry in its presence is by definition not one the agents traverse *instead of*
the coordinate.

---

## 4. What this dissolves

**The hierarchy hazard is not real once ρ is typed correctly.** Reviewing the flat-integer version
of this change, the fatal objection was that the two live sources sit at *different levels of the
agent hierarchy* — `region_agent_relative` is a wave-level fact, `rho_span` a half-wave one — so
collapsing them to one number would make a bf16+TDMSplit kernel report a span, drive
`derive_quantum` into its DISTRIBUTED branch, collapse the slot, and **silently turn a wide load
into a broadcast**.

`Resort.level` removes that: the two are different *entries*, not different readings of one number.
Each derivation filters by level, so a wave-level resort can never be mistaken for a sub-wave span.
The paper had this all along (§3.2 speaks of "a block-level resort" and "a wave-level resort" in one
sentence); we lost it by typing ρ as a token.

**It also gives §2.4's same-agent predicate a real home.** The paper is explicit that this must read
ρ:

> "*Same-agent*" is a rho-PREDICATE (Lemma 3c(i)), NOT "no specialize" … the decoder must read rho
> to evaluate same-agent (§2.4, §4.6)

Today `agent_distributed()` answers it with `self.agents > 1` and a comment conceding ρ is where the
finer claim would live. With ρ typed, that becomes a query over the resort map rather than a global
count — and the "narrowing" of #165 stops being a new feature and becomes an ordinary read.

---

## 5. Scope — what is deliberately left out

1. **ρ being supplied is the design (§2.1), not a shortfall.** Listed here only to head off the
   opposite mistake: do *not* try to derive the agent partition inside θ. That would pull
   `getMxsTileSpanInfo`'s address-layout query into the model and break the very opacity §2.1 buys.
   The win is **one authority instead of three**, stated as a preset — the same relationship `S`
   and `Φ` already have to the target.
2. **`theta.agents` stays as it is in step 1.** Deriving it as `∏ extent` over wave-and-coarser
   resorts requires the *wave* modes (`MIWaveGroup`) to be first-class modes in `tile`, which they
   are not today. Making them so is a larger change with its own risk. Step 1 keeps `agents` and
   adds a **consistency assertion** (`agents == ∏ extent` when any wave-level resort exists), so the
   duplicate is *checked* rather than silently tolerated. Removing it is a follow-up.
3. **The `specialize` half is not implemented.** No shipping configuration partitions agents into
   producer/consumer roles. `Rho.roles` exists as the named slot so the field matches the paper, and
   `#165` is re-scoped to "populate `roles` when a producer/consumer split ships" — a smaller and
   better-defined task than it is today.
4. **Ordering.** `rho = {}` is at `translate.py:863`; `_read_hop` runs at ~596. ρ must be built
   before the hops. Mechanical, but it constrains ρ's derivation to depend on nothing built between.

---

## 6. Migration

| step | change | risk |
|---|---|---|
| R1 | **DONE.** `Rho`/`Resort`/`AGENT_LEVELS` in `theta.py`; `translate` builds it; `rho_consistency()` checks it against all three live authorities; two fixtures in `test_loopmodel.py`. **1728/1728** realizable sweep configs consistent, 1014 unit tests pass, **34/34 kernels byte-identical** to the pre-ρ build. | none — realized: pure addition |
| R2 | **DONE.** `agent_served_modes()` returns `ρ.served_modes()` (wave-and-coarser). The pre-ρ hop scan was kept under its own name so check 1 stayed non-vacuous, plus a negative control. | low |
| R3 | **DONE.** ρ moved above `_read_hop`; both hop fields are now projections of it. `getMxsTileSpanInfo` untouched. `_agent_rel` deleted — `_wave_span` (the cardinality) supersedes the boolean. ρ is keyed BY MODE, so A and MXSA sharing `aRegions` is one resort, not two. §7 acceptance passed. | realized: byte-identical |
| R4 | **DONE.** Checks 1+2 retired (they became `x == x` once the hops derived from ρ); the independent oracle moved to `test_loopmodel` as a params-side reimplementation. Deck frame `020_01`, `THETA_MODEL_GUIDE` §4.5, and three `IMPLEMENTATION_REPORT` sections corrected; #165 re-scoped to `ρ.roles`. | none |

R1 is the whole safety argument: it makes the new derivation *state* its agreement with the old one
across the full matrix before anything depends on it, so R3 is a switch-over between two things
already proven equal rather than a rewrite.

---

## 7. Acceptance test

Pure Python, no kernel build, and it is the test that distinguishes every way §4's hazard could come
back:

```
bf16 + TDMSplit   (wave-level resort, no subwave)  =>  |{slot}| == Φ      (wide load)
MX   + TileSpan   (subwave resort, extent 2)       =>  |{slot}| == Φ / ρ  (broadcast)
```

plus the existing invariants: `derive_quantum` bit-identical over every (Φ, ρ) pair we construct
(the differential that is 20/20 today), `agent_served_modes()` set-equal to the current result on
every config in the three shipping matrices, and `reduction_modes()` unchanged — that last one is
what #245 fixed and what a reclassification would break.

Hardware is not required to accept R1–R2. R3 needs the f8 and mxf8 emit-diffs to stay byte-identical
before it can be called done.

### 7.1 Hardware result (2026-08-25, R1–R4 in tree)

Both runs were built after the last code edit of the ρ work (`theta.py` 02:48, `translate.py` 02:54;
runs 03:02 and 03:07), so they exercise the retyped ρ, not the pre-R1 tree.

| yaml | solutions | rows | verdict |
|---|---|---|---|
| `loopmodel_mxf8_gfx1250` (TileSpan x TDMSplit block) | 34 / 34 after KernelWriter | 68 | **68 PASSED, 0 FAILED**, `clientExit=0` |
| `loopmodel_mxf8_gfx1250` (9 blocks) | 528 / 528 after KernelWriter | 928 | **928 PASSED, 0 FAILED**, 9× `clientExit=0` |

`NumElementsToValidate: -1` in both, so PASSED is a full element-wise check against the reference —
an absolute verdict, not a ULM0-vs-ULM1 comparison. No solution is silently absent: per block,
distinct kernels appearing in the results equals the generated count, in all nine.

This also closes the outstanding hardware item for #216: the split yaml contains **15 distinct
ULM1 × PrefetchGL2 kernels** (30 rows, `_PGL1_` ∧ `_ULM1_`), all passing — the same 15 cells the
CPU-only differential predicted.

Coverage inside the two runs: 8 ULM0 / 60 ULM1 rows (split yaml), 276 / 652 (broad matrix); all four
`TDMSplitA/B` combinations; `_PGL0_` and `_PGL1_` at 34 rows each.

---

## 8. ANSWERED by paper rev 2026-08-25 (Q36)

**(a) The type — confirmed verbatim.** §2.9/§3.3: `ρ.resort` is "a set of `(mode, agent-level,
extent)`". §2.8 L215 adds the diagnosis this note argued for, including that ρ is a **searched
preset** and that opacity governs a placement's *content*, "**not** … the mode→level map".

**(b) `extent` — answered: the proper-subset fold is unreachable.** §2.2: "A carrier group spans
the **full** agent extent `ρ` resorts the mode to — never a proper subset … `Φ` divides `ρ`'s full
extent **by construction** … declining it loses no legal merge." Precondition U forbids the second
realization; Precondition Q rejects the uncovered remainder. Our `derive_quantum` already declines
it, so nothing changes — the case is now known *unreachable* rather than merely *unimplemented*.

**(c) A third point the update settled, which R1 had only measured.** §2.2: a sub-wave axis is "a
**data-distribution** axis …, **not** a synchronization scope … no `Await` is ever scoped to it".
That is the reason `served_modes()` reads wave-and-coarser only. R1 found it empirically first (a
sub-wave entry leaked `M_inner` into the agent-served set, which would have re-introduced #245's
failure inverted); the paper now supplies the principle.

### Superseded — the original question text

`Resort.extent` is read as *"how many agents at this level the mode is spread over"*, and
`derive_quantum`'s DISTRIBUTED branch additionally requires `Φ ∣ ρ` — a carrier group must divide
evenly over the sub-agents. Is `extent` always the full extent of the resorted mode, or can a
carrier group span a *proper subset* of the agents the mode was resorted to? Today's code cannot
express the subset case (it declines the fold, returning `None`), which is safe; the question is
whether §2.2 intends it to be expressible. **This does not block R1–R4** — every shipping
configuration is the full-extent case.

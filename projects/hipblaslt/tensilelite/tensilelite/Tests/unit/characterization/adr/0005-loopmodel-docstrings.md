# ADR 0005 — LoopModel reasoning extracted from docstrings

Status: accepted, 2026-08-29. Every docstring in `LoopModel/` that ran to twelve lines or
more mixed the function's contract with the reasoning behind it and, usually, the miscompile
that produced the reasoning. The code now keeps the contract -- the first paragraph, which is
what it always was -- and cites this file.

Extracted VERBATIM. Nothing here is a paraphrase, because a paraphrase of a hardware-measured
fact is how the fact gets quietly lost.


## `theta.py`

### `Mode`

One tiled traversal axis: a name and its extent.  `theta.ord` is a list of these — the
FULL permutation of looped time modes, outer→inner, INCLUDING the outer reduction /
persistent levels (§2.4 point 1, §5.3.1 `levels = [iter] + [substep, m, n, …]`).

A level is OUTER or INNER by its `extent` alone — ONE field, no bool and no enumerated
level-name/kind.  `extent` is the mode's STATIC trip count, and `0` is the sentinel for "trip
not statically known" — the intrinsic fact that tells the two rings of §6.1 apart and is
exactly the STATIC-vs-DYNAMIC loop distinction:
  * `extent > 0` (INNER / static) — an intra-chunk mode (the inner K substep, the M/N fan):
                   its trip is a searched `tile` factor, a compile-time constant, so the loop
                   is FULLY UNROLLED (no back-edge).  The §2.6 per-chunk register machinery,
                   the read nest, and geometry/sizing iterate EXACTLY these (`inner_modes()`);
                   the register ring rotates WITHIN the unrolled body.
  * `extent == 0` (OUTER / dynamic) — a pipelined level whose trip is a PROBLEM DIMENSION,
                   "the output of the nest, not an input coordinate" (§2.1), "not a searched
                   coordinate" (§5.2): a real RUNTIME loop with a back-edge, needing the
                   software-pipeline peel (prologue/drain) + the `Cond` short-loop guard.  The
                   reduction chunk `iter` and any persistent `gtile` are these; the runtime trip
                   symbol (`K//DU`, tiles-per-block) is a problem fact emit/translate owns,
                   identified by the mode's NAME — the Mode need not store it.  The shared
                   multibuffer rotates ACROSS this back-edge (§6.1 reduction-chunk ring).

So a persistent level, an inner-substep pipeline, and the outer-reduction level are all just `ord` modes
told apart by `extent==0` — adding one is a new mode with `extent=0`, NOT a new decoder branch.
Making `iter` a real `ord` mode (was an implicit synthesized loop) is what lets `off`/`S` be
per-(op-class × level) and the read-ahead a clean retime, instead of faking the substep→reduction chunk
crossing with a presence-mode carry that leaked the fan into the shared reduction chunk.

### `Fragment`

The register-buffer structure of an operand's VGPR placement.

An operand's tile-fan is delivered from rotating VGPR BUFFERS.  The fan is partitioned
into register-reuse GROUPS (e.g. the lower-M-half and higher-M-half of A), and each group
cycles through its own number of buffers = its rotation WIDTH `W`.  The register footprint
`[lo:4, hi:2]` is a two-WIDTH, same-RATE point (§2.6, Lemma 3b): both halves reload the
same (R=4) with the same floor (L=2) but run at different widths (lo W=R, hi W=1 via an
in-place refill).  `[4,2]` is DERIVED — derive (R,L) per group, pick W by policy.  See the
L/R/W explanation in `group_width` and THETA_MODEL_GUIDE.md §6.4.

broadcast_axes — the loop axes this operand is CONSTANT over (does NOT depend on).  A[m,k]
               is constant over N, so `{"n"}`; B[n,k] over M.  This single fact drives
               presence (§2.1), reload rate, and the live-range — the model's structural
               primitive.  (Was per-register-name `nullspace`; it was always uniform across
               a fragment's slots, so it is one set.)
parts        — the register split.  The NORMAL form is an INT N = split into N EQUAL groups
               (`parts=2` = lo/hi, `parts=4` = one group per in-region tile) — expanded
               internally to `(1,)*N`.  A TUPLE form stays available for explicit/uneven slot
               counts: `(1,)` = one group (no split); `(1, 1)` = two equal groups (1 slot
               each); `(2,)` = one group holding 2 slots.  `len(parts)` = number of groups;
               each entry = that group's share of the fragment's registers.
labels       — OPTIONAL pretty names per group for readable output (e.g. ("lo","hi")); the
               default is g0/g1/…  Purely cosmetic — the group is identified by its label,
               and `group_policy` / the depth map key on it.
grouping_mode — the tile MODE whose VALUES the register groups partition (the in-region fan
               the fragment is consumed along; e.g. "min" for A, "nin" for B).  A register
               group IS a subset of this mode's values (§2.8 move 1), not a register-file
               split.  ASSUMPTION (current): EQUAL partition — `len(parts)` divides the
               mode's extent.  None on a non-split fragment (fan stays in the rate).
group_policy — the ONE way to set each group's rotation width `W` (the searched S coord;
               §2.6: W is searched, its band [L,R] is derived).  A dict `label -> choice`
               plus an optional "*" default.  Choices (see `group_width` for L/R/W):
                 'unroll'  → W = R   (band ceiling; a buffer per value, most VGPR)
                 'overlap' → W = L_pf (the PREFETCH-OVERLAP peak — §2.6 names it `overlap`,
                             NOT `floor`, because L_pf is the *naive* floor; the operative
                             floor is L_war, which `inplace` takes)
                 'inplace' → W = 1   (single buffer, in-place refill — legal below L)
                 int (e.g. 3) → an EXPLICIT W, validated into [L,R]; interior search point.
               None ⇒ every group defaults to 'unroll' (W=R).

### `Hop`

One leg of an operand's staging trajectory (path).  Carries transfer granularity and its
MOVER KIND.  (The retime `δ` is NOT here — it is the first-class `theta.off_map`, §2.4/§2.9,
read via `off_at(op, role, level)`; a hop is one op-class edge, and `off` is indexed by
(op-class × level) where the op-class is this operand paired with this hop's `role`.)

kind  — how this leg moves data:
          'tdm'  — a BULK, whole-tile cooperative mover (tensor_load_to_lds): one
                   instruction moves a whole tile, and `split` cuts it into that many
                   storage-disjoint region parts — the region-split knob (a `tile` on the copy hop, §2.8 move 1).
          'load' — a PER-LANE transfer (ds_read, buffer_load, DirectToVgpr): each lane
                   moves its own share; `split` does not apply.
        Default (None) is INFERRED: a global→shared leg is 'tdm' (the bulk path); any
        other leg ('shared→register' ds_read, 'global→register' DTV) is 'load'.  A
        non-bulk global→shared leg (async copy / DirectToshared) is expressed by setting
        kind='load' explicitly.
split — for a 'tdm' leg, the number of region parts the tile is split into (1 = whole).
        Ignored for a 'load' leg.
quantum — THE MOVEMENT QUANTUM (§2.2, Precondition Q): an `ir.QuantumMap`, i.e. TWO
        EXPRESSIONS over the operand's tile coordinate —

            carrier(t)  which instruction carries tile t   (tiles sharing it are ONE movement)
            slot(t)     which position inside that instruction's destination holds t

        An EXPRESSION, not a schema, because the two hardware mechanisms differ only in the
        RANGE of `slot` and so need no tag: a wide load gives each tile its own slot (N tiles ->
        N registers), a sub-agent broadcast maps partners onto the SAME slot (N tiles -> 1
        register), and a composition of the two is a composition of the expressions.  Every
        consumer EVALUATES; none interprets.  `registers = |{slot(t)}|` over the carrier group,
        `coordinates served = the carrier preimage`.

        SUPPLIED, NEVER DERIVED HERE.  The merge follows from the ADDRESS LAYOUT and the
        emitter's folding, and θ is address-opaque (§2.1), so the expression is built OUTSIDE
        (the bridge, from target facts) and θ only evaluates it — the same standing as `S`.
        `None` is the identity merge: one tile per instruction, always admissible, the default.

        WHAT θ DOES WITH IT is structural: the merged coordinates are ONE presence point, hence
        one step and one generation (§2.2), so the pair is inseparable by construction and there
        is nothing to test.  Four attempts to have θ DERIVE the merge (scan the free modes for a
        contiguous-looking axis; test whether the pair shares a generation and narrow if not;
        refuse a TDMSplit region crossing; `tile`-split an axis to create the pair) were each an
        address assumption in disguise — see `Lowering/gir/quantum.py`.

### `Operand`

One tensor trajectory in generic geometry.

free_mode   — the ord Mode name this operand fans over (its M or N tiles).
hops        — the trajectory (path): list[Hop], global→…→register.  Each hop carries its
              own mover `kind` ('tdm' bulk vs 'load' per-lane) and TDM `split`; there is no
              separate operand-level bulk_src/split (they belonged on the hop).
fragment    — the register-axis structure (reuse-group partition + broadcast_axes).
frag_elems  — elements one lane holds per (free-tile, substep) fragment.
elem_bytes  — bytes per element.
lds_buffers — S_shared for THIS operand: how many shared (LDS) buffers its copy rotates.
              MEANINGFUL ONLY for an operand with a shared placement; a one-hop
              global->register (DTV) operand has none and carries `0` = not applicable.
              §2.6 makes the depth a per-placement map and §2.4 only bounds it (`S ≥ δ`), so
              the count is a per-operand field, not a global one — the paper's `S_shared(B)=1`
              beside a deeper `A` is exactly this.  `1` is the in-place refill (a single slot
              reused every chunk under the §2.4 `S=δ` WAR).  `0` means "not stated" and the
              depth falls back to `δ`, which is only correct when the target's own buffer
              count happens to equal `δ` — see `_shared_depth` for why that fallback is a
              trap and must not be relied on by a caller that knows the real count.
region_modes — mode names that ENUMERATE storage-disjoint sibling fragments (the
              region split's parts).  "Two-map reading" (§2.6): broadcast FOR THE RATE (a group's R
              is per-fragment) but indexed FOR THE GRID (real M/N data-tiles); the split
              multiplies the NUMBER of groups, not any group's rate.
role        — the op-class's data role (§2.1, PAPER-CONFIRMED Q18 §2.1 line 80):
              'input'  — a loaded operand (A/B/scale): has an input trajectory (`hops`),
                         present on its free modes + the reduction modes it contracts over.
              'output' — the ACCUMULATOR op-class: present on the FREE/output modes ONLY,
                         reduction modes ABSENT (`fragment.broadcast_axes` = the reduction
                         modes).  Register-resident, zero-init, NO input trajectory in the
                         mainloop (`hops=[]`); the epilogue's `stage` splices its reverse
                         register→(shared)→global path onto this SAME op-class later.  Its
                         presence is where `reduction_modes` is DEFINED (§2.1 line 82:
                         `reduction_modes = inner_modes \ pres(output)`).

### `Resort`

One `resort(mode → agent)` (§2.8 move 2): a looped mode traversed by the AGENTS instead of
by the coordinate.

`extent` is how many agents at `level` the mode is spread over — the cardinality the movement
quantum needs (§2.2's ρ) and the thing an equality-only selector token cannot supply.  `origin`
is provenance only: the (op-class, hop-role) this entry was derived from, so a dump or a
consistency failure can name where it came from.  Nothing keys on it.

`role` DOES key.  A READ entry is mode-unique — "a mode is resorted to ONE agent level", which
is why two operands sharing a region axis are ONE entry and `translate` raises on a second
answer.  A COPY entry is PER-OP-CLASS and deliberately NOT mode-unique: the members of a Φ
group share a free axis but take DIFFERENT shares of the agents (`A_MX` gives A two waves and
each scale one), so `M_inner` legitimately carries one copy entry per member.  Keeping them
apart by role is what lets ρ hold both without the read side's uniqueness invariant — the one
thing that made a second, per-operand field look necessary — having to be weakened.

### `Rho`

ρ = agent assignment + role partition (§2.9 line 244).

`resort` is the §2.8 move-2 half: which looped modes the agents traverse, at which level, over
how many agents.  `roles` is the move-8 `specialize` half — producer/consumer agent
partitioning — which NO shipping configuration uses and which is therefore empty; see
`RHO_DESIGN.md` §5.

WHY THIS IS NOT AN OPAQUE SELECTOR MAP, which is what this field used to be.  Typed as
`op-class -> token` compared only for equality, ρ can answer exactly one question — "are these
two op-classes on different agents?" — which a plain agent COUNT already answers.  So it was
born answering a question that did not need it, had zero readers for the whole implementation,
and its real content re-appeared as three unrelated per-hop scalars.  Worse, because ρ could
not carry the partition, each of those three had to be computed from an ADDRESS-shaped fact (a
lane-layout comparison, an address-layout query) and smuggled into `Hop` — pushing exactly the
facts §2.1 forbids into the model.  Typed as the paper types it, the partition is one supplied
preset and every derivation is arithmetic over modes and levels, with no address in θ at all.

SUPPLIED, LIKE `S` AND THE MOVEMENT QUANTUM.  ρ is a value from outside that θ reasons over but
does not choose; the target computes it where the target facts live and hands it over in GEMM
terms.  Do not try to derive the agent partition inside θ — that would pull the address-layout
query back in and break the opacity this buys.

### `DepthMap`

`S_Λ : Q(Λ) → ℕ₊` (Definition 1) — depth per PLACEMENT Λ, per rotating coordinate.

The key is `(op, region, group)`:

  op      the op-class the placement belongs to
  region  WHICH placement — the storage-disjoint part index, or `None` for the whole tile.
          A region-split tile (§2.8 move 1) is TWO placements of one operand, and Definition 1
          indexes `S` by the placement, not by the operand.
  group   the rotating coordinate within it: a register reuse group, or `SHARED_GROUP` for the
          LDS ring (the two are different rows of the one map, §2.6).

THE REGION DIMENSION IS WHY THIS IS KEYED IN THREE PARTS (#212).  It used to be `(op, group)`,
which silently asserts one placement per operand per space.  That held while nothing split a
tile, and it is precisely what blocks TDMSplit (#217): A's two storage-disjoint LDS halves have
to be able to carry different depths, and under a two-part key they cannot even be named.

`get` WITHOUT a region does NOT silently pick one.  If per-region rows exist and DISAGREE it
raises, because collapsing a per-placement fact to per-operand is the bug this key exists to
prevent — a caller asking a whole-tile question of a split tile must name the region.  When the
rows agree (today's equal-partition case) the collapse is well-defined and it answers.

### `readahead_level_of`

`(level_name, level_extent, region_span)` — the ONE derivation of the level a local
read-ahead lives on, from an `ord`'s inner modes, the region-mode names, and (for a
per-operand answer) the op-class's broadcast axes.

`PrefetchLocalRead` IS AN OUTER-LANGUAGE PARAMETER, and this is its translation into θ's
(§2.4's `off`): PLR is the minimum **wmma-region** read-ahead, so `off(read, ·)` sits on the
axis one region step advances — the OUTERMOST mode of the region's inner nest — and the region
is everything inner to it.

    KMN     -> level K, region M*N
    NKM     -> level N, region K*M
    KMNKMN  -> level K_inner, region M_inner*N_inner  ("per region": the outer K/M/N split
               triple enumerates regions and the read-ahead advances inside one)

THE LEVEL IS PER OP-CLASS; THE REGION IS NOT.  A wmma region is a shared notion, but the axis
an operand reads ahead ALONG is the outermost axis THAT OPERAND VARIES ON — pass its
`broadcast_axes` and they drop out of the nest.  Under `MKN` that gives A the level `M_inner`
(A varies on M and K) and B the level `K_inner` (B is constant over M, varying on K and N).
Omitting the broadcast exclusion put `off(B, read, M_inner) = 1` on an operand that never
revisits `M`, and the read-ahead could not discharge: MEASURED, 8 undischarged
`readahead-residency B@prologue ... -> B@wmma` obligations, `P7` of the §5.3.1 structural gate,
229 unit failures and 70/288 sweep configs — every one of them a non-K-outermost order, which
is exactly the set where an operand's outermost varying axis differs from the ord's.

IT WAS THE INNERMOST REDUCTION AXIS, which is right for `KMN` alone and wrong everywhere else.
Two measured consequences (2026-08-28, MIWaveTile[4,4]):
  * `DepthU=64` has NO reduction inner mode at all — the ord is `M_inner(4) N_inner(4)` — so
    the level did not exist, no `off` entry was written, and `dr` came out 0 for EVERY
    `PLR` 0..3.  The read-ahead silently vanished on every DU=64 kernel.
  * `MKN` (`M_inner K_inner N_inner`) put the offset on `K_inner`, one level INSIDE the axis a
    region step actually advances.

A LEADING AXIS WITH NO VALUES IS SKIPPED.  A DU `TDMSplit` re-factors the reduction as
`K_split x K_inner` and `K_inner` can collapse to extent 1 while keeping its `ord` position
(measured: `KMNKMN` + `TDMSplit=[2,2,2,2]` gives `K_inner(1) M_inner(2) N_inner(2)`).  An
extent-1 axis cannot carry an advance — shifting along it is the identity — so the level moves
inward to the first mode that has values, and the region is what remains inside THAT.  This is
the same guard `placement._substep_mode` already carried, and for the same measured reason:
priming one step where the body consumes two left `wmma u=1` reading a generation nothing had
loaded (192/1152 at the K-innermost orders).  Consequence to be aware of: for such a degenerate
nest the region is smaller than the full `M*N` the un-collapsed ord would give.

A REGION MODE IS NOT PART OF THE NEST — it ENUMERATES regions, so it is dropped wherever it
sits rather than used as a boundary.  Taking the last region mode's ORD POSITION as the
boundary instead is wrong for an interleaved split: a one-sided `TDMSplit` gives
`K_inner(2) M_split(2) M_inner(2) N_inner(4)`, where `M_split` is nested INSIDE `K_inner`, and
slicing after it moved the level from `K_inner` to `M_inner` — measured, that broke 84 of 288
sweep configs with `LoopIR structural gate failed (§5.3.1)`.  Dropping region modes gives
`K_inner` there and `K_inner` for `KMNKMN` alike.

Returns None when the nest is empty (no inner modes, or every inner mode is a region mode) —
there is then no intra-region axis to read ahead along, and `off(read, ·)` is simply absent.

### `readahead_level`

`readahead_level_of` for a built θ.  With `op`, the PER-OP-CLASS level (its broadcast axes
dropped — the axis that operand actually reads ahead along); without, the ord-level answer.

Region names are the union over op-classes either way, because the region nest is a property
of the `ord` every op-class is emitted into, not of one operand.

EVERY REGION MODE IS DROPPED, reduction-axis ones included, because a region mode ENUMERATES
regions and the read-ahead advances INSIDE one ("`KMNKMN` -> `M_inner*N_inner` per region": the
outer split triple is the enumerator).  A DU split is the case that tests this — it re-factors
the reduction as `K_split x K_inner`, and leaving `K_split` in the nest makes it the level, so
`PLR=1` asks for a whole `K_split` chunk of look-ahead.  MEASURED at `MIWaveTile[8,8]`,
`TDMSplit=[1,1,2,2]`: `W` 2 -> 8 and `foot(A)` 128 -> 512, breaking #311's proved invariant that
a region split cannot change the register footprint (it partitions the same tiles; it cannot
create data).  With `K_split` dropped the level is `K_inner` and the footprint is 128 across
all four splits.

(An earlier cut kept reduction-axis regions, reasoning from `_rate_broadcast`'s "A REDUCTION
REGION MODE STAYS IN THE RATE".  That qualifier is about the RATE — which counts a rotation's
generations, and a DU split genuinely is a rotation there — not about which axis a read-ahead
advances along.  The MX failure that motivated it is fixed by the presence rule below, which
subsumes it: what had gone wrong was the level landing on a SPANNED axis, not on `K_split`.)

THE PER-OP NEST IS ITS READ HOP'S *PRESENCE*, not merely "inner modes minus broadcast axes".
Presence is the per-hop fact (§2.1/§2.8) and it drops two things at once: the operand's
broadcast axes, AND any axis the read's own instruction SPANS (§2.2's absorption — a spanned
axis is not a presence point of that hop, so there is nothing to advance along).  Using
broadcast alone put the MX scale rings' read-ahead on `M_inner`, which is exactly the axis
their 2-tile load spans: measured, 4 undischarged `readahead-residency MXSA/MXSB@prologue`
obligations on the shipping mxf8 path.  With presence the level falls through to `K_inner`,
which is what the obligations name.

### `group_unit_tiles`

THE UNIT: fan tiles ONE BUFFER of this group holds, derived from `ord` AND the split.

This is the quantity every register count must agree on, and the one place it is defined.
`R` is "buffers needed to cover the copy region" and `|part|` is "tiles per buffer"; a
footprint `W x |part|` is only meaningful when the `W` counts BUFFERS of exactly the
`|part|` the same expression multiplies by.  Mixing them double-counts the fan.

MEASURED (kw6: `MKN`, `MIWaveTile[4,4]`, `partsA=2`, `unroll`).  `R` counted the fan in
TILES (8 per operand) while `|part|` counted it in RESIDENT tiles (2), so `foot(A)` came out
`W(4) x |part|(2) x frag(16) x 2 groups = 256` for a wave holding `4 tiles x 16 = 64`
registers of A — 4x the data where PLR1 needs 2x.  GIR reported the same fact at emit:
`USE-BEFORE-DEF ... reads A buffer X3`, because a 4-deep ring has no 4th generation to fill.

The unit is BOTH `ord`- and split-derived, which is what makes `R` legitimately `ord`-variant
without inflating the footprint:

    KMN  MIWaveTile[4,4]   resident 4 of 4  -> unit 4 tiles, R = K(2)          foot 2x4x16
    MKN  MIWaveTile[4,4]   resident 2 of 4  -> unit 2 tiles, R = K(2) x 2 = 4  foot 4x2x16

Same 128 registers — the order changes the ring DEPTH and the buffer SIZE reciprocally, not
the amount of data, which is the invariant a footprint must have.  A `TDMSplit` moves it the
same way, through `resident_free_tiles`.

### `group_fan_reloads`

Buffer-generations the FAN contributes to `R` = owned tiles / tiles-per-buffer.

`1` means the fan NAMES (one buffer covers everything the group owns, so advancing the fan
selects a different register — not a reload); `>1` means it RELOADS and belongs in the rate.
Both `_rate_broadcast` and `group_rate` read this so they cannot disagree about which.

A PARTIAL fold CANCELS here: a `q`-wide fold divides the owned tile count and the resident
window by the same `q`, so the ratio — and therefore `R` — is unchanged.  That is why this
takes no fold argument, and why a partly-folded fan cannot silently enter the rate in TILES
while the rotation counts GENERATIONS (#316).

FULL ABSORPTION DOES NOT CANCEL, AND IS HANDLED EXPLICITLY.  When one instruction spans the
whole fan (`q == N`, §2.2's degenerate case) the axis LEAVES the read hop's presence, so it
contributes no presence point and therefore no generation — the fan is delivered entire, once.
The ratio cannot see that on its own: `resident_free_tiles` walks the free-tile index sequence,
which is not folded, so `owned` stays `N` while `unit` drops to 1 and the fan reads as
RELOADING `N` times.  MEASURED on the shipping mxf8 path at `DepthU=128` (where `K_inner`
collapses out of `ord` entirely, leaving `M_inner(2) N_inner(2)`): `MXSA`'s presence is EMPTY —
its one instruction covers both M tiles — yet `R` came out 2, so `W` was 2 and the ring needed
a rotating coordinate that does not exist.  The gate caught it as
`P3/P5: ('MXSA',) placement references UNBOUND mode 'M_inner'`: the read is hoisted above the
`M_inner` loop (its presence no longer contains it) while its placement still named it.

Symmetric across the operands — it is whichever scale ring's free axis is OUTERMOST in `ord`
(`MXSA` under `MKN`/`MNK`, `MXSB` under the N-outer orders).

### `_rate_broadcast`

The axes a register group is CONSTANT over for its RATE/live-peak — the "two-map reading"
(§2.6).  A register group is a *partition of `grouping_mode`'s values* (§2.8 canonical form):
sibling values of that mode select a DIFFERENT part (a different storage-disjoint fragment),
so the grouping mode does NOT reload a single group — it is constant FOR THE RATE.  The rate
is thus over the HELD modes only (substep etc.).  So this set is the group's broadcast axes
PLUS its `grouping_mode` PLUS the operand's FREE `region_modes` (all spatial tile-count, not
reload).  A fragment with no `grouping_mode` keeps its fan IN the rate — there the fan IS the
rotation.  The same modes stay non-broadcast for the GRID role (computed on the data map),
which is why roles and rate are separate computations.

A REDUCTION REGION MODE STAYS IN THE RATE, and the FREE qualifier above is the whole of it.
The exclusion argument is about a storage partition: sibling values of a free region axis name
a DIFFERENT fragment, so the split multiplies the NUMBER of groups (`resident_free_tiles`) and
leaves each group's band [L, R] alone.  A region on the REDUCTION axis is not a partition of
anything — the two regions are successive SUBSTEPS of one tile, held in the same group, live at
the same time because they feed different mma instances of one iteration.  Excluding it makes
the group constant over the only axis it actually reloads on.

MEASURED 2026-08-18, TN + `TDMSplitA/B = 2` (the DU half): `K_split` extent 2 with `K_inner`
collapsed to 1, so `group_rate` R went 2 -> 1, W -> 1, and every read got `slot = 0`.  The
emitted asm loaded region 0 and region 1 into the SAME `ValuA_X0_I0+0/+4` and both mma's read
whichever landed last — 1152/1152 failures with a correct descriptor and correct addresses.
The unsplit kernel of the same shape has R=2 from `K_inner` alone; re-factoring `DepthU` as
`K_split x K_inner` must not change the product, and this is what keeps it invariant.

### `group_live_peak`

L — the live peak, the derived FLOOR (§2.6 table row 2).  Peak simultaneously-live
generations, a cyclic interval-overlap where a generation `(iteration, value)` has a read
interval in step-index space.  Computed on 3 unrolled iterations, peak over the steady one.

TWO peaks (§2.6, the "post-WAR live peak" paragraph):
  * PREFETCH-OVERLAP (post_war=False): interval `[first_read − off, last_read]` — the read is
    prefetched `off` iterations early, so a generation overlaps the next generation's
    read-ahead.  This is what free double-buffering (`unroll`) needs; it can exceed R.
  * POST-WAR (post_war=True, the FLOOR Lemma 3b measures against): interval `[first_read,
    last_read]` — NO `−off` extension.  An in-place refill SERIALIZES the next fill behind the
    last read (the §2.4 S=δ WAR), so it does not overlap across the boundary.  What remains is
    only the WITHIN-iteration overlap of DISTINCT values: SEQUENTIAL runs (value t fully read
    before value t+1's first read — mxfp8 hi: substep0 then substep1) don't overlap → L=1, so
    a single buffer refilled in place suffices; INTERLEAVED values (k0,k1,k0,k1 — s36 deep-A
    held across an inner fan) do overlap → L=2, so W=1 is the simultaneous-input race Lemma 3b
    forbids and the floor is 2.

The order-aware register `strategy` (translate.py) needs the post-WAR floor: it drops a group
to W=1 only where sequentially refillable (post-WAR L=1) and clamps a deep/interleaved group
up to its true L instead of an illegal W=1.  `off` (prefetch depth) is thus NOT part of the
floor — it is what `unroll` (the prefetch-overlap peak) buys above the floor.

### `_reg_read_off`

The register read-prefetch depth `off(read, ·)` = the PLR read-ahead, read from the
first-class off_map (§2.4, spec §70: δ lives in off_map, not on the hop).  Keyed on the
READ-AHEAD LEVEL (`readahead_level`, #321) — the outermost mode of the region's inner nest,
which is the innermost reduction mode only when the reduction is outermost.  0 for an op-class
with no input trajectory (the output/accumulator) or no intra-region axis.

THE LEVEL IS THIS OP-CLASS'S, so `op` MUST be passed.  Asking for the ord-level answer instead
reads the offset off the wrong key and silently returns 0 for any op-class whose own level
differs — and this quantity feeds `resident_free_tiles`, so a spurious 0 shrinks the resident
window, which shrinks `group_unit_tiles`, which inflates `group_fan_reloads` and hence `R` and
`W`.  MEASURED on the shipping mxf8 path at `MKN`: `off(MXSA)` came back 0 (its level is
`K_inner`, the ord's is `M_inner`), resident 2 -> 1, unit 2 -> 1, `R` 2 -> 4 and `W` 2 -> 4, so
the scale ring was four deep where its parent `A` was two.  The short `T < M` arm then had
slots 2 and 3 to prime and did not, aborting codegen with `the short arm cannot stand alone:
4 consumer(s) ... read a location the arm never writes`.

### `_assignment_width`

Narrowest `W` in `[lo, hi]` whose rotation MAP admits the requested read-ahead, else `lo`.

The band `[L_war, R]` is a CAPACITY interval; this is the ASSIGNMENT side of the same question
(see `placement.register_reuse_verdict`).  A `W` that holds enough generations can still write
`(P + shift) mod W` onto a slot whose value is consumed later, and the narrowest width that
avoids that is not derivable from the peaks — the shift is `ord`-dependent, which is §2.6 Q37's
"computed by the walk" seen from the width side.

`lo` on no answer, deliberately: the width never widens speculatively, and a depth the map
cannot take is refused downstream rather than emitted into a wider ring that still collides.

IMPORTED LOCALLY because `placement` imports `theta` at module scope.  This is the one direction
that has to be deferred; the alternative is moving the walk plus `_presence`/`_ord_strides`/
`_rate_modes` into this module, which puts the read-placement machinery in the θ object.

### `group_width`

W — the rotation width = the value S stores, searched in the band [L, R] (§2.6 row 3).
L and R are DERIVED; W is the searched coordinate, taken from the fragment's SINGLE knob
`group_policy` (via Fragment.policy_of):
    'unroll'   → W = R   (band ceiling; flagship lo)
    'overlap'  → W = L_pf (the prefetch-overlap peak; §2.6 calls it `overlap`, not `floor`)
    'pipeline' → W = max(L_pf, min(2, R))  — `overlap` plus a MEASURED emitter floor of 2.
                 THE DERIVED DEFAULT (#315).  See the branch below: `L_pf` collapses to 1 at
                 dr=0 and the emitter's actual def-use order does not survive a single-buffer
                 register ring (171/268 PLR0 cells failed numerically), and no checker sees it.
    int W      → an EXPLICIT width an autotuner searched — validated into [L, R];
                 a request outside the band is REJECTED, not silently used.
Explicit-W is how TensileLite (the autotuner) passes an interior point L<W<R.
    'inplace' → W = the POST-WAR floor (§2.6): 1 where the group is sequentially refillable
                (a §2.4 S=δ register WAR — mxfp8 hi, X01 alone per region), else CLAMPED up to
                the post-WAR L for an interleaved/deep group where W=1 would race (Lemma 3b).

TWO floors (§2.6): `floor` policy uses the PREFETCH-OVERLAP peak (the width that double-buffers
to hide read latency); the band's `W ≥ L` lower bound and `inplace` use the POST-WAR peak (the
realizable floor, L ≤ R).  R is the ceiling (`unroll`).

### `rho_consistency`

R1's safety argument (#303): does ρ agree with the three live agent authorities?

Returns a list of human-readable disagreements — EMPTY means the new derivation and the old
ones say the same thing on this point.  REPORT-ONLY BY DESIGN, the same shape as the #203
fence harness: ρ is built and checked before anything reads it, so the R3 switch-over is
between two things already proven equal rather than a rewrite hoping to land clean.  Raising
here instead would make a brand-new derivation able to fail a kernel that works today.

Pure, cheap, no rocisa: safe to call on every decode and from the test sweep.

WHAT IS STILL CHECKED, AND WHAT R3 RETIRED.

  3. AGENT COUNT (live) — the product of the wave-and-coarser extents must DIVIDE
     `theta.agents`.  Divisibility, NOT equality: ρ resorts the modes `tile` actually has, and
     the wave GRID is not one of them (`MIWaveGroup` waves are not first-class modes —
     RHO_DESIGN.md §5.2), so one operand split two ways over four waves gives product 2 against
     `agents` 4, which is correct.  What is never legal is a product that does not divide
     `agents`: that says ρ spreads a mode over more agents than exist.  This check stays
     because `theta.agents` is still an INDEPENDENT authority (it comes from the wave count,
     not from ρ).

  1/2. SERVED MODES and SUB-AGENT SPAN — RETIRED BY R3, HAVING SUCCEEDED.  Through R2 these
     compared ρ against a pre-ρ hop derivation and were green on 1728/1728 realizable configs,
     which is what licensed the switch-over.  R3 made `Hop.rho_span` and
     `Hop.region_agent_relative` projections OF ρ, so both comparisons became `x == x`.  A
     tautological check is worse than no check — it reports success it cannot fail to report —
     so they were removed rather than kept for the green tick.  The independent comparison now
     lives in `test_loopmodel`, which recomputes the agent facts from the raw params.

### `_shared_depth`

S_shared: the shared ring BUFFER depth (per reduction chunk) (§2.4/§2.6).

The coupling is `S ≥ δ` and **`S = δ` is the tight, feasible floor** (§2.4 line 95, §3.3
"S=δ legal"): with `δ`-deep prefetch the copy for iteration `iter+δ` REUSES the slot iteration `iter`
occupies, ordered after `iter`'s last read by the refill-after-read WAR (a same-agent
rotation, discharged dually by the read's own selector, §4.1 line 207).  So a same-wave
kernel double-buffers at off=2 → **S_shared = off = 2** (a `%2` ring), NOT off+1.  (off+1
is only needed for a CROSS-AGENT rotation — a spare slot, §2.4 line 101 `S ≥ δ+1` — which
mxfp8 is not.)  The `off+1` is the GENERATION count — how many iterations are live at the peak —
which the register-side L/W split (§2.6) distinguishes from the buffer WIDTH; on the shared
ring the S=δ WAR makes the buffer count δ even though δ+1 generations coexist.

**The depth is the operand's OWN buffer count (`op.lds_buffers`), not `δ`.**  §2.4 only BOUNDS
the depth (`S ≥ δ`, with `S = δ` *feasible*); §3.3 searches `S` in `[L, budget]`.  It never says
`S = δ`.  Which value the kernel actually has is a target fact — the number of shared buffers
its LDS allocation reserves — so θ must READ it, exactly as the register side reads the width.

Deriving `S = δ` instead was a real bug, and a quiet one: `δ` and the buffer count COINCIDE at
the common `PGR=2` double-buffered setting, so every test agreed.  At `PGR=1` against the same
double-buffered LDS they diverge — θ said 1, the kernel had 2 — and a depth-1 ring has no
rotation, so the decoder emitted NO `swap` at all while the kernel's addressing, its second
`LdsOffset` block and its token flip all still alternated.  Every copy then overwrote the buffer
being read.  (Same shape as the register-side `n_s == LoopIters` coincidence, #122.)

**No `S ≥ δ` gate here, deliberately.**  §2.4's coupling is per PLACEMENT, and on this target
`off(copy, chunk) = PrefetchGlobalRead` is NOT the LDS placement's δ: the global read lands in
G2L *registers* and the LDS write is a separate, serialized step.  `1LDSBuffer=1` with `PGR=2`
is a shipped configuration (`Solution.py`: `numLdsBlk = 1 if 1LDSBuffer==1 else 2`, independent
of PGR), i.e. a real kernel with `S_shared=1 < δ=2`.  Asserting `S ≥ δ` against PGR rejects it.
That is also the second reason the old `S = δ` derivation was wrong — not only did it read a
quantity the kernel does not use for this ring, it read one that is not even an upper bound on
it.  A genuine `S ≥ δ` check belongs against the LOCAL-WRITE offset once the write side is
modelled (#116), not against PGR.

`1` is the in-place refill — one slot reused every chunk under the §2.4 `S=δ` WAR, which is how
the paper's `S_shared(B)=1` beside a deeper `A` arises.  `0` means the caller did not state a
count and we fall back to `δ`; that fallback is retained only for hand-built θ in tests, and is
the trap described above for anything that knows the real number.

### `served_modes`

The modes the AGENTS traverse INSTEAD OF the coordinate — §2.8 move 2's domain, as the
axis CLASSIFICATION consumers mean it.  `Theta.agent_served_modes()` rediscovers this same
set from a per-hop boolean; the two are asserted equal by `rho_consistency`.

WAVE-AND-COARSER ONLY, AND THE SUB-WAVE EXCLUSION IS THE WHOLE SUBTLETY.  A wave-level
resort takes the axis out of the read/compute nest entirely: the wave works inside its own
region and nothing in the nest varies over it, which is why `reduction_modes()` must
subtract it (#245 — leaving it in put an unbound mode in a slot Expr).  A SUB-WAVE resort
does not do that.  It DISTRIBUTES tiles of an axis the coordinate still enumerates: the
carrier group merges `Φ` tiles and spreads them over `ρ` sub-agents, so the axis survives
as the group index (§2.2's fold) and only its intra-group coordinate leaves.

§2.2 (paper rev 2026-08-25) states the boundary directly: a sub-wave axis is "a
DATA-DISTRIBUTION axis (it settles which tile a sub-agent carries and which are co-folded),
NOT a synchronization scope … an operand MAY carry a sub-wave axis in its presence or fold
it into a carrier group, but no `Await` is ever scoped to it (§5.2's scope ladder starts at
`wave`)."  An axis an operand may still carry in its presence is by definition not one the
agents traverse INSTEAD of the coordinate.

MEASURED before the paper confirmed it: with sub-wave entries included, `M_inner` appeared
in `served_modes()` while the old derivation correctly omitted it — which would have
reclassified a live tile axis and re-introduced #245's failure from the other direction.

### `agent_distributed`

Are this op-class's shared-staged hazards CROSS-AGENT (§2.4)?

True iff more than one agent touches the buffer.  §2.4's `S=δ` discharge is
`complete(read) ⤳ issue(copy)` via program-order-on-issue, which relates events on ONE
agent; the moment a byte's writer and its reader are different agents that composition does
not exist and the edge needs a proc-scoped selector.  In a GEMM every agent reads the whole
staged tile, so 2+ cooperating agents is already enough — which is exactly the shipping
rule: as soon as two waves cooperate, the barrier is needed.

Φ IS NOT PART OF THIS, and an earlier version wrongly made it the whole test ("the members
of a fused group disagree about their carrier").  Fusion decides whether A and B ride one
instruction; it says nothing about who carries what.  A LONE operand cooperatively loaded by
four waves is cross-agent with no fuse anywhere, and that predicate returned False for it —
no fence, a race — which only stayed unreachable because the bridge welded fusion to
multi-wave.  ρ remains the place a future *narrower* claim would live (this movement is
confined to one agent after all); it is not needed to answer this one.

### `movement_units`

The COPY MOVEMENTS this θ emits, as `[(unit_key, members, n_regions)]` — the ONE
authority for "which cooperative movements exist and how many region instances each has".

A movement is a Φ group of global→shared op-classes merged into one cooperative
instruction (§2.8 move 9), or a lone op-class when unfused.  `n_regions` is the
region-index pairing count (§4.3): the shared split when EVERY member is split the same
way (the clean AB case → AB0/AB1), else 1 — a mixed-split group has no region-`j`
instance for its unsplit member, so it moves whole.

Both the emitter (which builds the movements) and the lowering (which must know a
movement's completion granularity to key its token) need this; deriving it twice invites
the two to disagree.

### `reduction_chunk_mode`

The outer level the REDUCTION CHUNK advances on — the INNERMOST outer level, since
`ord` is outer→inner and any coarser outer level (a persistent tile level, §5.3.2) sits
ABOVE the chunk.  None for a legacy θ with no outer level.

This is the ONE axis all of the following are keyed on, and they must never disagree:
  - the §6.1 shared multibuffer ring depth `S_shared` (`_shared_depth`),
  - the shared-read `src_slot` rotation (`_read_placement`),
  - the steady `Loop` `build_ir` emits and the `Bind` its peels pin.
They used to index `outer_modes()` directly and split 3-vs-2 between `[0]` and `[-1]` —
identical while there is exactly one outer level, silently divergent the moment a second
(coarser) outer level exists (#113).  Route every such lookup through here.

THE PEEL IS NO LONGER ON THAT LIST (§5.3.1, 2026-08-14).  `peel_depths` returns a
`PeelDepths` whose `M_ℓ` is per LEVEL, because the paper's construction peels *every* level
with `M_ℓ > 0` — and names keying it here as the defect: "a decoder that keys peel depth
only on the innermost outer level silently drops every coarser-level `off`."  What the peel
still uses this accessor for is `PeelDepths.M`/`copy_off()`, i.e. "the depth AT the chunk",
which is the one level `build_ir` can currently emit a region for; a coarser level's peel
is refused, not dropped (`emit._check_peel_is_emittable`), until #113 supplies its Loop.

### `off_at`

The retime offset `off(op-class, level)` (§2.4 `off : (op-class × level) → ℤ`, a
first-class θ field, §2.9).

The OP-CLASS is `(opname, role)` — one operand's movement along one hop — with `role` from
the COPY/READ/STORE vocabulary (`HOP_ROLE`; use `hop.role` when you hold the hop, or
`off_of` below).  Absent entry → 0 (that op-class is not pipelined at that level).

Nested pipelining is just more map entries: `off(copy, iter)` and an outer
`off(copy, gtile)` (persistent, §5.3.2) coexist as two keys, needing no decoder change.
The role is what makes the CROSS-LEVEL case expressible too: `off(read, iter)` — a read
carrying an outer-level offset, which §5.3.1's boundary term relocates across the
enclosing boundary — is a distinct key from `off(copy, iter)`, where an operand-only key
would have conflated the two.

### `presence_modes`

The inner Modes an op-class VARIES over (its presence set, §2.1) = the inner modes minus
its broadcast (constant-over) axes.  One structural primitive; no name tests.

PRESENCE IS A PER-HOP FACT (§2.8 move 1, §5.3.1 line 583).  Pass the `hop` to get THAT
op-class's presence: a BULK copy spans the inner axes in one instruction, so its presence is
just its `region_modes` (empty when unsplit — the copy sits at the reduction chunk), while a
per-lane hop varies over everything the data does.  See `geometry.hop_broadcast`.

`hop=None` answers for the OPERAND's data — the access/read presence.  That is what the
derived families below want (`reduction_modes` is defined off the OUTPUT op-class's data
presence, `free_modes` off the operand's), so it stays the default rather than becoming a
required argument at 9 call sites that all mean the same hop.

A PER-HOP MODE MAY CARRY A NARROWER EXTENT THAN THE `ord` MODE OF THE SAME NAME.  §2.2's
movement quantum FOLDS an axis of extent `N` into `N/q` carrier groups, and the coordinates
inside one group are ONE PRESENCE POINT.  A fully-folded axis (`q = N`) drops out via
`hop_broadcast`; a PARTIALLY folded one stays with extent `N/q` — it still varies, over the
GROUP INDEX.  This is the single point at which the fold is applied (`geometry.hop_fold`):
every consumer that counts presence points — the read-ahead shift, its flat position, the
NLL drain-suppression bound `P + shift >= n_pres` — then inherits it for free, which is
exactly what §5.3.1 means by "no separate tile-vs-point conversion".

SCOPED TO THE HOP, AND THAT IS WHY THE FOOTPRINT DOES NOT MOVE.  §2.8 move 1 says presence
"applies per hop independently", so the fold narrows `presence_modes(op, hop)` and leaves
`presence_modes(op)` — the operand's DATA presence — at raw extents.  That is not a
conservative dodge, it is the right split: a `Φ`-merge changes how many INSTRUCTIONS reach
the registers, not how many REGISTERS the data occupies (a 2-wide `ds_read` still lands two
tiles in two registers).  `presence_tiles`/`free_tiles`/`sizing` all read the hop-less form
and so are invariant under `Φ` BY CONSTRUCTION — which is exactly what was measured when
this fold was first applied half-way (footprint unchanged at 512).  Reading the fold out of
the data presence instead would halve the footprint and under-allocate the register file.

THE COORDINATE STAYS A TILE INDEX — the group is named by its LEADER, `g·q`.  The nest is
unrolled over `ord` extents (`inner_modes`, untouched here) and the read is issued once per
group by `emit._first_touch_modes`' `mode % q == 0` guard, so the acts land at
`0, q, 2q, …`.  Folding the extent here is therefore a change to how many presence POINTS
the read-ahead counts, not to the coordinate space: `_readahead_shift`'s `n_pres` and the
NLL bound `P + shift >= n_pres` become point counts, and `emit._shifted_coord` rescales the
folded mode's carry back to its leader tile.  `refs.covered_coords` expands that leader
in place over the group, which is the same rule seen from the consumer side — one
derivation of "the group is named by `g·q`", read by the shift, the guard and the
expansion.

### `agent_served_modes`

The THIRD kind of inner axis: traversed by the AGENTS, not by the coordinate (§2.8 move
2 `resort`).  #245.

The model classified inner axes TWO ways -- FREE (in the output's presence) or REDUCTION
(`inner \ pres(output)`, by subtraction).  An agent-served region axis is NEITHER: the
wave reads, computes and accumulates entirely inside its own region, so nothing in the
read/compute nest varies over it, while the BULK COPY still walks its regions.  Deriving
"reduction" by subtraction therefore MISCLASSIFIES it the moment presence drops it, and
every consumer of `reduction_names()` inherits that -- measured: `_rate_broadcast`
(theta.py:770) excludes a FREE region mode from the register ring but keeps a REDUCTION
one, so the reclassified axis stayed in the rotation and the slot Expr named a mode no Loop
binds (`P3/P5 ... UNBOUND M_split`).

Naming the class explicitly is the fix: `reduction_modes` subtracts it below, so the
two-way split becomes three-way in ONE place instead of being patched at each consumer.

R2 (#303): THIS NOW READS ρ.  An agent-served axis is, by §2.8 move 2's own definition, a
mode `resort` sent to an agent level — so the answer is `ρ.resort` projected to
wave-and-coarser, not a fact rediscovered from a per-hop boolean.  §2.2 is why the
projection stops at `wave`: a sub-wave axis is a data-distribution axis, not a
synchronization scope, and an operand may still carry it in its presence.

R3 RETIRED THE SECOND DERIVATION.  Through R2 a pre-ρ hop scan survived beside this so
`rho_consistency` had something independent to compare against.  Now that `Hop`'s agent
fields are themselves projections of ρ (`translate._read_hop`), that scan would be reading
ρ back out — `x == x` — so it was deleted rather than left as green theatre.  The
independent second opinion moved to `test_loopmodel`, which recomputes the agent facts
from the raw params and asserts ρ matches; that is a genuine reimplementation, which an
in-library copy of a derivation can no longer be once ρ is the only authority.


## `emit.py`

### `_check_peel_is_emittable`

Refuse a θ whose peel `build_ir` cannot express — instead of emitting a DIFFERENT schedule.

§5.3.1 (2026-08-14) makes the peel PER-LEVEL: every level with `M_ℓ > 0` gets its own
prologue/drain, and an offset at a level OUTER to an op-class's own home level relocates that
op-class's leading δ inner-nest instances across the enclosing boundary (Lemma 1's cross-level
boundary term).  `build_ir` peels exactly ONE level — the reduction chunk — because a coarser
level's prologue/drain has to sit inside an enclosing `Loop` for that level, and that Loop is
#113 (persistent / grid-stride).

Emitting the chunk-only peel anyway would produce a schedule that ignores the coarser `off`
**and still reports an empty ledger** — the exact silent-drop the paper names ("a decoder that
keys peel depth only on the innermost outer level silently drops every coarser-level `off`"),
and the one failure mode the empty-ledger certificate exists to rule out.  So: refuse, loudly,
with the derived facts in the message.

Scoped to OUTER levels.  An INNER level's `M_ℓ` is real (`off(read, substep)` gives the substep
level one) but it needs no prologue/drain REGION: an inner level is statically unrolled, so its
peel is absorbed into the unroll and realized by the read-ahead family (`preloaded_tiles`
primes it, `_readahead_shift` sustains it, the drain's `readahead_suppress` guard ends it).
Only an OUTER level needs a region this emitter cannot build.

Unreachable from the parameter path — `translate` sets only `off(copy, chunk)` and
`off(read, substep)`, each of which is that op-class's OWN home level.  It fires for a
hand-built θ (and will stop firing, level by level, as #113 lands).

### `_scope_of`

The PROC-SCOPE the discharge of obligation `ob` must reach (§2.4, §5.2's scope ladder).

Default `wave`: the reader and the refiller are the same agent, so the `S = δ` WAR's middle
edge is program-order-on-issue and no selector spanning agents is needed — the σ_c copies-last
order IS the discharge.

`block` when the movement is agent-distributed (§2.4 route b, Lemma 7's `fuse` clause).  Then
the buffer is refilled by only the sub-set of agents that wrote it and read by ALL of them, so
program order relates nothing across the two and the WAR is live: §2.4 requires "a proc-scoped
selector or `S ≥ δ+1`", the selector's scope reaching the agents involved — for cross-wave
reader/refiller inside one block, BLOCK scope.  The paper is also specific about its strength:
it must be a MEMORY-ORDERING barrier (ordering the read's completion before the refill's
write); a bare execution-only barrier leaves the WAR live.  That requirement travels with the
scope to the backend.

THE RAW RESIDENCY IS ALSO BLOCK-SCOPED WHEN THE MOVEMENT IS AGENT-DISTRIBUTED (corrected
2026-08-22, #276).  This bullet used to read:

    "a RAW residency's consumer awaits a completion CLASS, which is not an agent-relative fact
     — the counter says the bytes landed regardless of which agent moved them"

and that is FALSE for a PER-WAVE counter, which is the only kind gfx1250 has.  `s_wait_loadcnt`
retires THIS wave's outstanding loads; it says nothing about another wave's.  Under a
cooperative fill each agent issues its own part of the slab and then every agent reads the WHOLE
tile, so a reader waiting its own counter has no edge to the writer that actually filled the
bytes it is about to read.  The paper is explicit — §5.4 line 700 lists this obligation as
    "L2 ... — cross-agent, BLOCK SCOPE, quantified within one generation"
with its discharge "the proc-scoped selector (§4.6) at block scope", and §4.6/§5.4 (lines 339,
735) state that a per-agent counter cannot cover a cross-agent edge at all.

  * a `rotation-WAR` is the `S ≥ δ+1` case, and §2.4 offers the selector and the spare slot as
    ALTERNATIVE remedies ("a proc-scoped selector **or** carry a spare slot"): the rotation
    already holds a slot nobody is reading this trip, so the missing cross-agent edge is
    supplied by the depth, not by a barrier.  Widening its scope too would emit a barrier per
    trip for an obligation the depth has already discharged.

WHAT THIS DOES NOT DO, STATED SO THE FIX IS NOT MISTAKEN FOR MORE THAN IT IS.  `scope` is
currently INERT: its only consumer is the diagnostic tag string at `gir_to_rocisa.py:295`
(`"%s scope=%s covers %s edge(s)"`), and `_syncThreads` at :298 never receives it.  So this
correction changes what the MODEL believes, and provably not one emitted byte.  The barrier that
actually makes these kernels correct today comes from `LdsHazards` -> `FenceRegions`, which
probes the LDS ring rotation — a NARROWER mechanism than the obligation it happens to cover.
Making `scope` load-bearing is the next step of #276 and must be measured against #199's
over-synchronization budget before it lands.

### `_region_mode_of`

The inner mode that ENUMERATES this fused group's storage regions, or None if the group
is unsplit.  Taken from the group's representative member; §2.8 move 9 allows members to be
tiled on DIFFERENT region axes (the pairing is by index, not by a shared mode), so a
heterogeneous group labels by one axis — tracked in #169(c).

Read off the COPY HOP'S OWN PRESENCE (§5.3.1 line 583), not by reaching for
`op.region_modes`: a bulk copy is present exactly on the axes it does NOT span in one
instruction, which for a split tile is the region mode and for an unsplit one is nothing.
Same answer, but now the copy's level comes from the presence set the paper says places it,
so a future hop with a different bulk-ness needs no new branch here.

`_bulk_hop`, NOT `hops[0]`: a direct-to-register (DTV) operand's first hop is
`global→register`, which is the READ.  Asking it returns the data presence and would name
a reduction axis as if it were a region axis.

### `_enum_axes`

The axes that ENUMERATE this movement's regions — the REPRESENTATIVE member's own, not
the union over members.

The two are different questions and conflating them was wrong in both directions:

  FUSED GROUP (§2.8 move 9).  Members are paired BY INDEX — region j of A moves with
  region j of B — so A's `M_split` and B's `N_split` are ONE index, not a product.  Taking
  the union makes the movement sit under both loops and emits `|M|x|N|` placements for a
  `|M|`-region movement.
  ONE OPERAND SPLIT ON TWO AXES (MT+DU: `M_split` x `K_split`).  Here the axes genuinely
  multiply, and one member carries both — so the representative's own tuple is the product
  and both loops must enclose it.

Union-over-members cannot tell these apart because it loses which member contributed what.
The representative's own tuple answers exactly the enumeration question; `_group_region_modes`
(the union) stays, but only for PRESENCE — so a fused movement is not guarded by its peer's
axis.

### `_group_region_modes`

EVERY region axis this movement spans — the union over its members, not one member's.

A Φ-FUSED group is one instruction moving SEVERAL operands (§2.8 move 9), so its copy is
present on each member's own region axis: an A+B fuse over a split tile spans `M_split`
AND `N_split`.  `_region_mode_of` deliberately returns just the representative (the level
it is FILED under); this is the set that says what it is INVARIANT over, which is a
different question and the one first-touch asks.

Getting that wrong is not cosmetic.  `_nest_copies` guards a group on the enclosing levels
it does not span; with a fused group filed under `M_split` and judged "invariant over
N_split" by name inequality, its single cooperative load would be first-touch-guarded to
`N_split == 0` and the regions paired with the other N values would never be issued.

### `_group_guard`

`Cond` predicates restricting a group-restricted read to the coordinates it OWNS.

A register group is a SUBSET OF THE GROUPING MODE'S VALUES (§2.8 move 1), not a slice of
the register file: at `MIWaveTile[4,4]` with `VgprPartitionA=2`, `lo` holds
`M_inner ∈ {0,1}` and `hi` holds `{2,3}`.  The read that fills a group therefore belongs
only at that group's coordinates — but the read sits INSIDE the grouping mode's `Loop`, so
without a guard both halves are issued at all four tile indices.  Measured before this:
every `M_inner` carried both a group-0 and a group-1 destination, `check_plan` reported
"operand A reads source coordinate(s) … MORE THAN ONCE per trip", and the surplus reads
filled registers no wmma ever names.

The index is `grouping_mode // width` — `Expr.carry`'s floor division, the same field the
read-ahead rollover uses — so the guard is one structured `Pred` per bound, read by field
like every other predicate here.  Groups are an EQUAL partition (`Fragment.grouping_mode`'s
stated assumption, enforced in `translate`), which is what makes the division exact.

Returns () when the read needs no guard: an unsplit operand, or a subset that happens to
be every group.  A NON-CONTIGUOUS subset would need a disjunction, which `Pred` cannot
express — `_read_groups` only ever produces contiguous runs (policy order is group 0
`unroll`, the rest `inplace`), so that case is a decoder defect and says so rather than
silently guarding on the wrong range.

### `loop_shape`

Decide, from θ alone, whether the ord yields ONE uniform steady body (cleanly
loopable) or needs a CONDITION-BRANCH / multi-body loop path.

Criterion: each role's inner modes must be CONTIGUOUS in ord.  A blocked nest that
splits a free-axis role AROUND the reduction axis (e.g. ord roles [m, n, k, n] — the
N fan interrupted by a K step) cannot be expressed as one uniform steady body: the
K-loop boundary falls mid-sweep, so the body before and after the K step differ.  Per
§5.2/§5.1 such an ord needs a loop node with MULTIPLE steady bodies (Loop.bodies>1), and
the whole-tree ledger gate (§5.5) makes the multi-body case sound.

STATUS: `emit_mainloop` CONSULTS this and REJECTS a clean=False ord (raises) — it is no longer
a detector nobody reads.  `build_ir` still emits one body per Loop, so a non-contiguous role
would otherwise yield a single uniform body with an EMPTY ledger: a silently wrong schedule,
not a visible failure.  Splitting the steady body into `Loop.bodies>1` (§5.1) is the tracked
feature (#102); until it lands such an ord is not emittable rather than wrongly emitted.
Reachable in practice — LoopOrder 'MKNMKN' and the interleaved 'KMKNMN' are clean=False at
a wave tile of [4,4] (they read clean at [2,2] only because the split/inner pair collapses to
extent 1 and drops out of ord).

Returns {"clean": bool, "roles": [...], "split_roles": [...]}.  clean=True → one body;
clean=False → the named roles are non-contiguous and need a multi-body loop path (not yet
emitted).

### `_nest_copies`

Order the chunk's copies by the ORD, not by operand — one nest, first-touch merged.

WHAT WAS WRONG.  Each op-class got its OWN `Loop(region_mode)` and they were appended as
siblings, so the emitted sequence was grouped per operand: `A0 A1 B0 B1`.  The ord for a
KMN split kernel is `[K_inner, M_split, N_split]`, and traversing it gives `A0 B0 B1 A1` —
which is exactly the order the READS come out in, because they are placed by `build_level`
at their presence level inside the shared nest.  Two different orders over one ord is not
a schedule; σ is supposed to be a single traversal.

INVISIBLE WITHOUT A SPLIT, which is why it survived: unsplit, every op-class has exactly
ONE copy, so "grouped by operand" and "ord order" are the same sequence `A, B`.  They can
only diverge once an op-class owns two or more copies — precisely what a region split
creates.

The nest is built ONLY over the modes that are some copy's level (here `M_split`,
`N_split`), never over the full ord: a level no copy is present on would wrap everything in
a trip-N loop with every body first-touch-guarded to iteration 0 — the same instructions,
needless structure.  A group is invariant over every OTHER group's level (a region mode
belongs to one op-class), so its guard set is just the enclosing levels.

The copies stay ONE BLOCK.  This reorders WITHIN the block and does not move copies into
the read/wmma nest, so §5.5's copies-first / copies-last placement — and the ledger built
on it — is untouched.

### `_read_groups`

How many INSTRUCTIONS this operand's read is split into, and which register groups each
one fills.  Returns a tuple of group-tuples (a single `None` entry = the unsplit read).

§5.3.1 pt5 gives each group exactly TWO self-consistent shapes and no third: PREFETCH
(`dr_g >= 1`) is hoisted ahead of its consuming wmma; INPLACE (`dr_g = 0`) is not hoisted
at all — it issues in place, one line before its own wmma.  One `Inst` carries one advance,
so an operand holding groups of BOTH shapes cannot be read by one instruction: the read
splits, and σ_c defers the INPLACE half past the consumers (the register-ring analogue of
the §5.3.1-pt4 copy freeze).  A uniform-shape operand stays whole.

The split key is the DERIVED `dr_g`, not `W_g > 1`.  Width is only one of the two ways a
group reaches INPLACE — the other is §2.6's term (ii) (a broadcast fan outer to the whole
presence set), which zeroes `dr_g` at any width.  Keying on width alone left such a group
in the hoisted instruction while its own advance was 0.

### `_first_touch_modes`

Enclosing inner modes the op is INVARIANT over — non-presence modes at an ord position
OUTER to the op's read level.  Each becomes a first-touch `mode == 0` guard.

Returns `(name, step)`: `step = 0` is the ordinary invariant guard `mode == 0` (issue once
over the whole extent); `step = q > 0` is §2.2's PARTIAL FOLD `mode % q == 0` — issue once
per CARRIER GROUP, `N/q` times over the axis.

THE FOLD NEEDS NO NEW MODE, FIELD OR MOVE — it is this guard's modulus.  §2.2: "a `q`-wide
quantum folds the axis into `N/q` carrier groups … the folded axis keeps `N/q` present
coordinates — the group index", and the quantum is "not a new field … both already in `θ`".
Full absorption (`q = N`) is the degenerate case where `mode % N == 0` IS `mode == 0`, which
is why one guard expresses both: a fully-spanned axis leaves presence and lands here as an
invariant mode, a partly-spanned one stays in presence and lands here with its modulus.

### `_read_anchor`

`{invariant mode: value}` at which `op`'s read-ahead must fire — the ANCHOR (#331).

`depth` is THIS NEST'S read-ahead depth, not θ's `dr`.  The degenerate (`T < M`) tree passes
0 — Lemma 1: a fully-peeled level resets every `off` inner to it — and a depth-0 read is not
hoisted at all, so it must stay at FIRST touch.  Reading θ's `dr` here instead left the
short arm's reads anchored to the last pass of the broadcast axis while their shift was 0,
so the arm's `N_inner = 0` consumers had no producer: "4 consumer(s) in it read a location
the arm never writes".  The anchor and the shift are one decision and must be taken at one
depth.

`{}` means the ordinary first-touch position (fire at value 0).  A non-empty answer defers
the read to a LATER pass of a mode it is invariant over, which is the only position at
which the refill follows the last consumer of the name it overwrites.

WHY THIS IS THE FIRST-TOUCH GUARD'S VALUE AND NOT A NEW EMISSION PATH.  A read invariant
over `m` is already emitted exactly once across `m`, by `Cond(m == 0)`.  The refill has to
happen once too — just at the OTHER end of `m`, after the last wmma that consumes the old
generation.  So the deferral is the same guard with a different constant, and no read is
added, dropped or duplicated.  Under `NKM` that turns A's `N_inner == 0` into
`N_inner == 1`: same single read, last pass instead of first.

The value comes from `placement.reload_positions`, which reads the SAME name-event walk as
`register_reuse_verdict` — so "may I defer" and "defer to where" cannot disagree.


## `geometry.py`

### `derive_quantum`

§2.2's carrier group as `Φ` applied to `ρ` — the `ir.QuantumMap` DERIVED, not supplied.

"The quantum is **not a new field**: a carrier group is exactly a `Φ` fuse group (§2.8 move 9 —
movement instances merged into one cooperative instruction) over the tiles that `ρ` (§2.8 move
2) has placed on the agents that one instruction spans … `θ` **names the members**."  `Φ` is
`hop.phi_width`, `ρ` is `hop.rho_span`, and this composes them into the two Exprs everything
downstream evaluates (`carrier(t)` = which instruction carries tile `t`, `slot(t)` = its
position inside it).

THE TWO SHAPES §2.2 NAMES, AND NOTHING ELSE:
  * CONTIGUOUS (`rho_span == 0`) — ρ lays consecutive tiles on the spanned agents, so the groups
    are the blocks `{0..Φ−1}, {Φ..2Φ−1}, …` and `carrier = t // Φ`.
  * DISTRIBUTED (`rho_span > 0`) — ρ spreads the group across a SUB-AGENT span (§2.2's sub-wave
    partition, a `tile` of the lane mode; the MX TileSpan half-wave).  The upper sub-agent folds
    onto the lower (the broadcast), then Φ folds each run, and the surviving groups are numbered
    CONTIGUOUSLY: `carrier = blk·(ρ/Φ) + (t mod ρ)//Φ` with `blk = t // 2ρ`.

LOAD-VS-BROADCAST IS NOT A FLAG.  A wide load gives each tile its own `slot`; a broadcast maps
the partner onto the leader's; whoever counts `|{slot}|` sees the difference.  So the two shapes
differ only in `carrier`, and `slot = t mod Φ` serves both — under a partner fold `Φ | ρ`, so
partners agree on `slot` and SHARE the register, which is the broadcast, expressed not flagged.

`carrier` IS THE DENSE INSTRUCTION INDEX, and that is what makes it the register index too: one
instruction writes one run of registers, so the run's base is its ordinal.  Returning the LEADER
TILE instead (`Φ·(t//Φ)`) is an equally valid group label and is off by a factor of `Φ` the
moment anyone multiplies it by a register stride.

None (the identity merge) when there is nothing to merge — and that is SILENCE, not a map:
a supplied `carrier(t) = t` means "θ decided: merge nothing", which stops the leaf folding tiles
that really do share one instruction.  Silence means "θ said nothing; keep your own fold".

### `quantum_factors`

`{axis: factor}` — how many CONSECUTIVE coordinates of each axis one instruction of a
per-lane hop carries (§2.2, Precondition Q).  The full derivation; `quantum_axes` is its
fully-spanned view.

§2.2: "a `q`-wide quantum FOLDS the axis into `N/q` carrier groups, so the intra-group
coordinate leaves presence (the `q` co-carried tiles are one presence point) while the group
index stays; the axis is fully absent only when one instruction spans it whole (`q = N`)."  So
the unit is a **factor of an axis, not the axis**: `factor == extent` is the full absorption the
old set-valued form could express, and `1 < factor < extent` is the ORDINARY PARTIAL CASE — an
extent-8 axis under a 2-tile load leaves 4 group indices present, not 0 and not 8.

DERIVED FROM THE SUPPLIED MAP, not declared alongside it, and not computed as `gcd(q, extent)`.
`QuantumMap.carrier(t)` says which instruction carries tile `t`, and a factor `f` is admissible
exactly when every block of `f` consecutive coordinates of the axis rides ONE carrier.  Reading
it off the map rather than off `q` is what keeps the two shapes of §2.2 membership — the
CONTIGUOUS fold (`carrier = t // q`) and the DISTRIBUTED sub-agent fold (the TileSpan partner
map, whose members are scattered across half-waves) — on one derivation: `gcd(q, extent)` is
the right number for the first and is not even the relevant quantity for the second.

THE REGULARITY CLAMP FALLS OUT rather than being imposed.  §2.2 requires `f ∣ N` ("a carrier
group is a uniform fold — `N/f` equal groups of `f`") and says a decoder "clamps `q` to
`gcd(q, N)` … never emitting a ragged final group".  Only divisors of `extent` are tried here,
so a ragged fold is never returned; on the contiguous map the largest admissible divisor IS
`gcd(q, extent)` (a block of `f` sits inside one `q`-run iff `f ∣ q`, and it tiles the axis iff
`f ∣ N`), which is the paper's clamp, derived.  This is why `MIWaveTile [8,7]`'s odd axis keeps
single-tile loads: no divisor of 7 above 1 divides a 2- or 4-wide fold.

THE INNERMOST-FIRST RULE ALSO FALLS OUT.  A load covering `tilePerRead` adjacent tiles fills
whichever axis its run reaches: at `MIWaveTile 2` with an MT split `M_inner` is degenerate, so a
2-tile load fills `M_split`; at `MIWaveTile 4` the same load fills `M_inner` and `M_split` stays
whole, because tiles 0 and 2 then have different carriers.  Both come out of the same test.

### `hop_broadcast`

The intra-iteration axes ONE INSTANCE OF THIS HOP does not vary over — presence PER HOP
(§2.1 line 71, §2.8 move 1 "applies per hop independently", §5.3.1 line 583).

`broadcast_axes` above is a DATA fact: the axes the operand's values are constant over.  That
is the right answer for a per-lane hop, where one instruction moves one lane's share and the
hop therefore varies over everything the data does.  It is the WRONG answer for a BULK hop.

§2.1 line 71 defines bulk-ness AS presence: "an inner axis a coarse copy SPANS IN ONE
INSTRUCTION".  A `tensor_load_to_lds` moves the whole slab in one go, so it does not vary over
the inner axes at all — they are its broadcast set, whatever the data depends on — and the copy
therefore sits at the reduction-chunk level.  Split the tile into `split` storage-disjoint
region parts and exactly one axis comes BACK into the hop's presence: the region mode, because
now there is one instruction PER REGION (§2.8 move 1) and the copy sits one level deeper.

So bulk presence is `region_modes`, and non-bulk presence is the data broadcast.  This is the
fact §5.3.1 line 583 says a decoder "MUST NOT HARDCODE ... to the slab level": before this,
`emit.copy_insts` reproduced the same answer by testing `op.split` and reaching for
`op.region_modes` itself, which is the shortcut that warning names.  Same answer, derived.

### `hop_fold`

`{axis: factor}` for the axes this hop's quantum folds only PARTIALLY (`1 < factor < extent`).

THE SECOND HALF OF `hop_broadcast`, and the two partition the quantum's effect on presence:
an axis one instruction spans WHOLE leaves presence (it is in `hop_broadcast`'s set); an axis it
spans in PART stays, with `extent // factor` group indices instead of `extent` coordinates
(§2.2 — "the folded axis keeps `N/q` present coordinates, the group index").

WHY THIS IS THE ONLY PLACE THE FOLD IS APPLIED.  §5.3.1's steady advance is explicit that the
traversal "is over PRESENCE POINTS, not raw tiles: a movement-quantum fold is applied when
`pres(p)` is built (§2.2) … and the shift, the position `P`, and the drain-suppression bound
`P + shift ≥ n_pres` all count in presence points, with `n_pres` the presence-point count `N/q`.
Because the fold is done once, at presence, the advance and its bound inherit it WITH NO
SEPARATE TILE-VS-POINT CONVERSION."  So `_readahead_shift`, `_ord_strides` and the NLL
suppression need no fold-awareness of their own — they read whatever extents presence reports,
which is exactly what makes them correct here and is why folding them separately would be the
two-derivations-of-one-rule bug this codebase keeps hitting.

Bulk hops fold nothing: a bulk copy's inner axes are already wholly absent (`hop_broadcast`).


## `ledger.py`

### `Endpoint`

One end of an obligation — WHICH op-class, at WHICH placement or site, at WHICH coordinate.

This was a formatted STRING (`"A@shared"`, `"A:g0:refill"`, `"A@prologue:K_inner0,M_split0"`)
that four call sites re-split to recover the fields, with three different split rules.  The
encoding was also load-bearing for the per-region π (#168), which had to APPEND a field to it.
A record is read by field; adding a qualifier is a field, not another suffix to parse.

op    — the op-class name (a θ operand).
at    — WHERE this end sits: a `Space` for a residency (global/shared/register), a rate-group
        label for a rotation WAR ('shared' = the shared ring, else a register group), or a
        SITE for a coordinate-keyed obligation ('prologue' / 'wmma').
role  — 'lastread' | 'refill' for a WAR endpoint; '' otherwise.
coord — ((mode, value), …) when the obligation is keyed to one coordinate (Lemma 3d's
        read-ahead residency); () otherwise.
gen   — for an `Await`'s dep only: the qualifier MODE NAMES that pick which generation /
        instance is awaited, outermost first — e.g. ('iter',) for the chunk the consumer
        reads, ('iter', 'M_split') under the per-region π (§5.2 line 368: "X names a producer
        INSTANCE or op-class-generation").  Empty on a ledger entry, which stays symbolic
        per-op-class (§4.1).

`op` is an op-class name, OR — on the producer end of a Φ-fused movement — the MEMBER TUPLE of
that movement.  A fused group is one cooperative instruction with one completion (§2.8 move 9,
§5.2), so "A's copy completed" is not an event that exists; the only event is the movement's.
A consumer naming a single member awaits a completion the hardware never signals separately and
is free to issue before the movement lands.  An UNFUSED movement keeps the bare string, so
every single-movement kernel is byte-identical to before.

### `build_ledger`

One RAW-residency per hop + one WAR per S rate-group.

WAR LABELING (paper §4.1 line 207 labeling note): the rotation/in-place WAR is discharged by
the VACATING READ's selector composed with placing the refill after it — `complete(read) ⤳
issue(refill) ⤳ complete(refill)` — so one selector (the read's) discharges both the read
residency and the WAR.  The obligation is therefore tagged with the READ's completion class,
NOT the physical refiller's:
  - a SHARED (shared) buffer's WAR: the read that vacates it is the shared→register read → its
    counter is that read hop's class (DS), even though the physical refiller is the
    global→shared copy (class C_copy).  (Pre-paper-update this was mislabeled C_copy.)
  - a REGISTER rate-group's WAR: the vacating read is the operand's final →register hop; the
    refiller is that same hop, so read-class == refill-class (DS/VMEM) — unchanged.
The emitted schedule is unaffected (both read and copy are awaited at the right positions);
only the ledger LABEL changes, to match the paper's discharge accounting.

### `discharge_once`

Drop an `Await` that a preceding instruction in the same execution path already established.

An obligation is ONE thing, so it is discharged ONCE.  `complete(P) ⤳ issue(C)` is satisfied
for every consumer after the first, because within a straight-line path the later consumers are
transitively ordered behind the first one's boundary.  Re-stating the Await on each consumer
turns one boundary into N discharge sites — the prologue read-ahead emitted four identical
`await A+B@global` lines for one cooperative copy — which is not what the ledger owes and gives
a count-deriving backend N partial waits instead of one.

Scoped to COPY-PRODUCED residencies (`RAW-residency` out of a copy hop, and `crossing-RAW`).
The rotation WARs are left alone: their producer is a per-consumer vacating read, so they are
genuinely distinct obligations rather than one boundary restated.

A boundary dies when its PRODUCER re-issues, so the copy that refills the buffer re-establishes
it; and a `Loop` body is entered with a fresh set, because a boundary established before the
loop does not hold on the second trip.

### `check_ledger_discharged`

§5.5 empty-ledger gate, PER-EDGE (paper line 427: "each producer→consumer obligation is
covered by an `Await` at the consuming instance" — and the per-edge gate CAN reject a discharge
mis-placed onto the wrong instance, which coarse counter-coverage cannot).  Two ways an
obligation is VIOLATED:
  (a) COVERAGE — no `Await` on its completion class anywhere in the tree; or
  (b) ORDER — a WAR consumer (a refill) is emitted WITHOUT its producer (the vacating read of
      the same buffer) appearing BEFORE it in the SAME region body.  This is the σ_c check:
      within one emitted trip the refill must follow the consumer whose buffer it overwrites,
      so an ORDER-DRIVEN backend (§5.4/§5.5: one that discards the markers and re-derives sync
      from emitted memory-token def-use order) derives the WAR, not an inverted RAW.  Checked
      on BOTH rings, because the paper's insufficiency argument is about the ring, not the
      space:
        * SHARED — a refill COPY must follow EVERY shared→register read that vacates it, the
          §2.7-line-185 `∀ r ∈ reads(gen t)` (so: the LAST read, not merely SOME read — the
          peel-skeleton `all_copies + steady_nest` order and the copy-interleaved-into-the-
          read-stream order are both rejected).
        * REGISTER — an `inplace-WAR` refill READ (its group's rotation width is 1, so it
          overwrites the slot in use) must follow a wmma consuming that group.  This arm used
          to be missing: the filter was `consumer.at == "shared"`, but register-ring endpoints
          carry `at` = the GROUP label, so every register WAR was silently coverage-only —
          exactly the §5.5 hole ("still has an Await on the counter class and passes coverage,
          yet encodes the wrong dependency"), one ring down.  An `Await` cannot substitute
          here: it waits on ALREADY-ISSUED reads, and the consuming wmma has not issued.
      A `rotation-WAR` read (W>1) is NOT order-constrained — it writes slot `(k+dr) mod W`
      while the consumer reads `k mod W`, so hoisting it touches no live register.
Returns the list of violated obligations; empty ⇒ emittable with correct σ_c.


## `ir.py`

### `QuantumMap`

How an op-class's TILES merge into the instructions that move them (§2.2, Precondition Q).

TWO EXPRESSIONS OVER THE TILE COORDINATE, in the same `Expr` vocabulary `Placement` already uses
for rotation slots — because this is the same kind of fact: a derived index, evaluated, never
interpreted.

    carrier(t)   WHICH instruction carries tile `t`.  Tiles sharing a carrier are ONE movement:
                 one completion, one presence point, one generation (§2.2).
    slot(t)      WHICH position inside that instruction's destination holds `t`.

LOAD vs BROADCAST IS DERIVED, NOT DECLARED — which is the whole reason this is an expression
and not a schema.  The two mechanisms differ only in the RANGE of `slot` over a carrier group:

    wide load, 2 tiles      carrier = t // 2      slot = t % 2      2 slots -> 2 registers
    sub-agent broadcast     carrier = t - stride*((t % 2*stride) // stride)
                            slot    = t % stride                    partners SHARE a slot
                                                                    -> 1 register
    composed                carrier/slot compose; still two Exprs, no new case

So `registers written = |{slot(t) : t in the carrier group}|` and `coordinates served =
the carrier preimage`.  Nothing has to be tagged "this one is a broadcast": a schema of
`(axis, factor, kind)` tuples has to be *interpreted* differently per kind, and every consumer
then re-implements the interpretation.  Here every consumer evaluates.

IT IS SUPPLIED, NEVER DERIVED IN θ.  The merge is a consequence of the ADDRESS LAYOUT and the
emitter's folding, and θ is address-opaque (§2.1) — so the expression is computed OUTSIDE (the
bridge, from target facts) and θ only evaluates it.  Same standing as `S`: a value from outside
that the model reasons over but does not choose.  `None` means the identity merge — one tile per
instruction, always admissible, and the default everywhere.

### `Load`

A named transfer op: move `tokens` from `src` space to `dst` space.

The completion counter is DERIVED from the (src,dst) hop, not stored — add a row
to HOP_COUNTER and this one op supports the new source/dest/counter with no other
change.  `tokens` with len>1 is a fused load (one hardware instruction carrying
several operands); the counter still counts it once.  `kiter` is the K-loop
iteration (one reduction chunk) this op feeds — the issue coordinate.

`coord` = ((axis,val),...) is the semantic position this transfer fills — for a
ds_read that is the (free-tile, k-substep) fragment, e.g. (("m",0),("k",0)); for a
bulk tile load it is just the split part.  `size_regs`/`size_bytes` are what ONE
instruction moves (the load size).  The logical DESTINATION (buffer/rotation slot,
register group) lives on the owning `Inst.placement`, not here.

### `_term`

One linear term, normalized to `(mode, coef, div)` = `coef · (env[mode] // div)`.

A term is written `(mode, coef)` — `div = 1`, the ordinary case and every term this IR built
before §2.2's fold was applied — or `(mode, coef, div)`.

WHY A DIVISOR EXISTS AT ALL (#312).  A term's variable is a LOOP INDEX, and the nest is
unrolled over `ord` extents; but the quantities these expressions compute — the flat presence
position, the rotation slot — count PRESENCE POINTS (§2.2, §5.3.1, Lemma 3e), and a `q`-folded
axis has `N/q` of them.  A folded mode's contribution is therefore `coef · (var // q)`, which
no `(mode, coef)` pair can express.  The divisor is the ONE mechanism for that conversion, so
every builder of a linear form over loop indices — `_shifted_coord`, `_shifted_rate_slot`,
`_rate_slot`, `_read_placement`'s `src_slot` — states the fold the same way.  (An earlier
version cleared denominators instead, scaling the whole form by `Π q`; that is exact inside a
floor-div but WRONG inside a bare `% mod` — the rotation slot — so it could not serve
`_rate_slot`, and keeping both would have been two encodings of one fact.)

### `Pred`

A structured predicate: compare an index `Expr` against an integer.  `lhs op rhs`.

REUSES `Expr` (the mixed-radix/carry index formula) as the left side — no new formula
language — so both a loop-count guard and a residue pin are the SAME structured object a
consumer reads by FIELD (`lhs`/`op`/`rhs`), never by parsing a string.  This replaces the old
free-form `Cond.pred` string that forced the lowering to parse ("lots of special rules").

Examples:
  peel-validity `T ≥ M`     : Pred(Expr(var="T"), ">=", M)
  drain-step pin `iter%2==1` : Pred(Expr(var="iter", mod=2), "==", 1)
  first-touch `M_inner==0`   : Pred(Expr(var="M_inner"), "==", 0)

`rhs` is either an `int` (the common guard/pin case, e.g. `T >= 2`) OR another `Expr` (a
symbolic bound, e.g. the steady back-edge `iter < T − M` where the bound depends on the runtime
trip `T`) — both read by field, no string.

`eval(env)` concretizes both sides and applies `op` via the single `_PRED_OPS` table.  If EITHER
side references a runtime symbol absent from `env` (e.g. "T", a problem dimension), `eval`
returns None (the predicate is runtime, not statically decidable) — a consumer lowers it to a
real branch.

### `Placement`

The LOGICAL storage an instance reads/writes (§5.2) — NOT the physical register color.

space  — global | shared | register (from `path`).
slots  — PER register group, the rotation slot as a SYMBOLIC `Expr` (index % S): a tuple of
         (group-label, Expr) pairs.  A two-half register fragment carries both — e.g.
         (("lo", min%2), ("hi", 0)) = lo rotates a 2-ring, hi in-place — the X00→X20 /
         X01-in-place rotation, kept symbolic (rolled).  Shared/global: one (("", Expr)).
src_slot — for a shared→register READ: the SOURCE shared-buffer generation as a symbolic
         `Expr` (`iter % S_shared`).  This is the read's CONSUME side, distinct from `slots`
         (the register destination).  `mod>1` ⇒ the shared buffer double-buffers and swaps at
         the reduction-chunk loop boundary; `mod<=1` ⇒ single in-place shared buffer (no swap).  None
         when the read has no shared source (e.g. DTV global→register).  This carries the shared
         generation into the IR so a consumer selects the right shared half without re-deriving
         it — the buffer info the copy already had on its own `slots`, now on the read too.
reg_off/byte_off — offset of this fragment within its buffer.

### `Await`

A NAMED-dependency discharge point (§5.4): the value `dep` on completion class
`counter` must be complete, within proc-scope `scope`, before the owning instance issues.
`dep` names a producer op-class-generation, NOT a count — the σ-invariant object of
Prop 7.  `count` is the §5.4 LOWERING (Await → selector residual) and is LEFT UNSET (-1) by
this decoder, which targets a COUNT-INSERTING backend (§5.4 bullet 2): that backend's own
counter-insertion pass re-derives the number from the emitted dataflow, and Prop 7 guarantees
the named form covers whatever σ it commits.  The slot exists for a raw / counter-based
backend to fill (§5.2: "an initially-empty count slot ... unset in the built IR ... the number
is a cached derivation, never authored"); that lowering is deferred work, so nothing writes
this field today.
`kind` is the ledger hazard class of the discharged obligation (§5.5): "RAW-residency" (true
data dep — the consumer needs the producer's value) or "rotation-WAR"/"inplace-WAR" (anti-dep —
the writer must wait for prior readers).  A *-WAR await means its owning instance must be ISSUED
AFTER those readers (§2.4 line 99), which the σ_c issue-order refinement uses to place refill
copies last.  `kind` mirrors the `Obligation.kind` the Await materializes — one source of truth.

### `Loop`

A real loop node of the rolled nest: `for <mode> in range(extent)`.  `body` is a
sequence of child `Loop`/`Branch`/`Inst`/`Peel` nodes.  `role` is the ρ agent role
(single-role default).

MULTI-BODY (§5.1, the non-contiguous-role case): `bodies` may hold several distinct steady
sub-bodies "taken in program order", each running over its OWN SUB-RANGE of the trip —
`body_ranges[i]` is the half-open `(lo, hi)` for `bodies[i]`.  That range is what makes the
multi-body form mean something: without it, N bodies run back-to-back every iteration and are
indistinguishable from one concatenated body.  With it, the emitter can PEEL a first-touch
guard into the loop structure — `for m: if m==0: R; Y` becomes `for m in [0,1): R; Y` then
`for m in [1,trip): Y` — so the predicate is not evaluated per iteration (§5.1's "the body
before the boundary differs from the body after").

`body_ranges` empty ⇒ every body spans the whole trip (the ordinary single-body loop).  Read
the pairing through `ranged_bodies()`, never by zipping the two lists at a call site.

### `Cond`

A first-class conditional over the reduction trip count — the peel's VALIDITY guard
(Lemma 1 hypothesis `T_ℓ ≥ off(·,ℓ)`).  The software pipeline's steady region is only
reachable when the loop is long enough to fill+drain the pipe; a too-short loop takes a
degenerate path (skip the steady body, go straight to the drain / no-load-loop).  Making this
a NODE (not an implicit assumption) lets a backend lower the boundary correctly from the IR
alone.

`pred` — a structured `Pred(lhs: Expr, op, rhs)` (NOT a string): the guard read by FIELD, e.g.
  `Pred(Expr(var="T"), ">=", M)` (steady region runs) or a drain-step pin
  `Pred(Expr(var="iter", mod=2), "==", 1)`.  A consumer reads `pred.lhs`/`pred.op`/`pred.rhs`
  structurally — no string parsing, clean derivability.
`then`/`els` — body-sequences for the true / false arms (either may be empty).
`kind` — the GENERIC (backend-agnostic, paper-vocabulary) meaning of this guard: one of
  `peel_validity` (Lemma 1 `T ≥ M`), `readahead_suppress` (NLL: the read-ahead substep would
  cross the reduction bound), `first_touch` (§5.1: a read invariant over an enclosing mode,
  issued once).  The pure CORE emits ONLY `kind` — NO TensileLite scaffold names.  The
  LoopIR→TensileLite-scaffold mapping (which `kind` routes to which openLoop/NoGlobalLoadLoop
  label) lives entirely in the GIR scaffold pass, which SETS `label` from `kind`.
`label` — an advisory scaffold-branch HINT (e.g. "toPGR1"), populated DOWNSTREAM by the GIR
  scaffold pass, NOT by the core.  A consumer whose own loop-count scaffold already emits this
  compare routes each arm to the matching label region instead of re-emitting the branch; a
  from-scratch backend ignores it and lowers `pred`.

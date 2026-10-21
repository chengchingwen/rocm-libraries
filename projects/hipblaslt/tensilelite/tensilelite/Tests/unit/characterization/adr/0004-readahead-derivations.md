# ADR 0004 — the read-ahead derivations: why each is shaped the way it is

Status: accepted. Extracted verbatim from `LoopModel/placement.py` docstrings, 2026-08-29,
when the package's comments were cut to the 1-3 line contract the code needs. Nothing here
is new; every paragraph was already in the source and each was paid for with a hardware run
or a measured miscompile. The code now carries the contract and cites this file for the why.


## How to use this file

Each section is the reasoning that used to sit in that function's docstring. The code now states
the CONTRACT in a line or three and cites `ADR 0004`; when you need to know *why* a derivation is
shaped the way it is -- and most of these are shaped by a specific miscompile -- it is here.

## `_readahead_shift`

The steady read-ahead advance for `op`, in ORD-POSITION units over `op`'s presence modes.

Returns `(shift, strides, n_pres, ext)` — the advance, the per-presence-mode ord strides, their
product, and each mode's extent — or `(0, …)` when the op does not read ahead.

`dr` is the REQUESTED depth; the advance is built from the DERIVED `dr_g`
(`_readahead_depth`, from the §2.6 floor), so a `dr_g = 0` group returns advance 0 — Shape B,
matching the empty prologue `preloaded_tiles` gives it from the same derivation.

§5.3.1 point 5 splits `ord` at the reduction substep: `[OUTER… | substep(n_s) | INNER…]`.  The
steady read-ahead must sustain exactly what the prologue primed, and Lemma 3d's prologue is
`{first coord of pres∩OUTER} × {leading min(dr,n_s) substeps, or ALL n_s if INNER=∅} × {pres∩INNER}`.
So the advance in substeps is that same substep factor, and converting it to ord-position units
is a multiply by the reduction mode's stride WITHIN THIS OPERAND'S PRESENCE traversal:

    substep_factor = n_s if INNER == ∅ else min(dr, n_s)
    shift          = substep_factor · stride_pres(reduction mode)

Both halves matter, and the decoder had neither:
  * INNER=∅ (a K-innermost order like `…M·N·K`) makes the factor the FULL `n_s`, `dr`-independent
    — §5.3.1's "steady read-ahead discharges at TILE granularity, pre-issuing the next tile's K".
    Advancing by `dr` there re-reads the current tile's other substep, which the prologue has
    already loaded: paper line 494's "leaf 0's read-ahead targets `(m0,n1)`, NOT `(m0,n0,k1)` …
    no second producer and no double-issue" — the double-issue we were emitting.
  * `stride_pres` is what carries the advance OUT of the reduction mode and into the free modes.
    Bumping the reduction coord alone wraps inside the tile and never reaches the next tile, so
    every tile after the first got its leading substep one wmma late.
For a K-outer order (`…K·M·N`) the reduction mode is outermost in the presence traversal, the
factor is `min(dr,n_s)`, and this reduces to the previous `dr`-bump exactly — KMN/KNM are
unchanged by construction.

THE SHIFT WALKS THE GROUP'S *RELOAD* MODES, NOT `pres(p)` (§5.3.1 pt 5, paper rev 2026-08-21).
The steady-advance rule is explicit that the flat traversal is "over `p`'s flat traversal of its
own group's RELOAD modes — the modes `R` counts (the group's `Q(Λ)` rotation axes, §2.6), *not*
every mode in `pres(p)`", and that a SIBLING-ENUMERATING mode — "a region split on a FREE axis,
the `{lo}{hi}` partition, or a reduction axis realized as CONCURRENT SIBLINGS rather than a
rotation" — "selects a different group's band, not a later generation of this group, so the
shift MUST NOT carry through it ... a shift that walked past it would target the neighbouring
sibling's register — a slot no future generation of the advancing group occupies (this is the
wrong-region read a free-region-split `ord` produces when the reduction interleaves INWARD of
the split)."

THE LINE IS FREE-vs-REDUCTION, AND THE PAPER DRAWS IT: "Reduction axes, and a genuinely-reloaded
free fan (cross-tile prefetch, `dr_g ≤ W_g−1`), ARE reload modes and stay in the traversal."  So
a DU (reduction-axis) split stays — it is a rotation, and #254's two-axis `K_split × K_inner`
register rotation depends on it — while an MT (free-axis) region split leaves the radix.  An
earlier attempt excluded `op.region_modes` wholesale and regressed exactly there (12 failures,
all on the two-reduction-rate-mode rotation), which is what pinned the scoping.

PRESENCE IS NOT THE TRAVERSAL, and the same passage keeps the two uses apart: a free-axis region
split "is in `pres(p)` and thus contributes to the prologue" by its `ord` position, but "is NOT a
reload mode the steady SHIFT advances through ... a free-region sibling BOUNDS the shift, which
then rolls the chunk rather than crossing into the neighbour."  So only this function's radix
narrows; `preloaded_tiles` keeps the full presence set, and the overflow that used to land in
the neighbour region now rolls the reduction-chunk coordinate — a shared-copy residency the
chunk-crossing rule already models (§5.3.1's `off(copy, iter) ≥ r`).

MEASURED 2026-08-21, the failure this repairs.  fp8 `MT64x64x256` `MI 16x16x128`
`MIWaveTile[2,2]` `MIWaveGroup[2,2]` `VW=2` PGR1/PLR1 `TDMSplitA/B=1`, six-axis orders:
    KMN     A: shift=2  strides={M_inner:1, M_split:1, K_inner:2}   <- reduction has the TOP stride
    KMKNMN  A: shift=1  strides={M_inner:1, K_inner:1, M_split:2}   <- reduction at stride 1
Under `KMN` the advance cannot reach `M_split`; under every six-letter word the interleave puts
the reduction INWARD of the split, so the first advance carries into the region axis.  Hardware
(`build_loop_model_f8.log`, solutions 15/17/19 of 39): FAILED at `LOKMKNMN`/`LOKMNKMN`/`LOKMNMNK`
x PLR1, PASSED at PLR0 and at KMN.  The mismatch is one wave-tile of A — D rows 48,50,52,54 wrong
(`VW=2` interleaves the two wave-tiles at row granularity, so even rows are wave 1's tile 0),
rows 0-47 and every odd row correct, N axis wholly clean.


## `readahead_reach`

`r` — the ENVELOPE over read op-classes of the DERIVED chunk reach (§5.3.1 pt5).

THE ONE derivation of "how many whole reduction CHUNKS the substep read-ahead crosses into",
shared by every consumer that counts chunks: the chunk-level peel `M` (`peel_depths`) and the
drain's read-ahead suppression window (`emit`).  It is built from the per-group DERIVED depth
`dr_g` (`prefetch_steps_for` — the §2.6 floor `W_g >= L`), NOT from the requested
`off(read, substep)`.

ENVELOPE, NOT "the" depth, and the distinction is load-bearing: this takes `max` over groups
while the per-instruction `_readahead_depth` takes `min`.  Both are right for their consumer —
a PEEL DEPTH and a SUPPRESSION WINDOW must cover the deepest group (too small drops reads a
later step needs, which is the very defect below), while a per-instruction shift must not
exceed the shallowest group's rotation.  Do not "unify" them into one number.

THE TWO ARE NOT THE SAME NUMBER, and using the request is the defect this function exists to
remove.  `dr_g = max(0, min(dr, W_g − 1))`, zeroed outright by the two structural vetoes
(`_broadcast_fan > 1`, `_sibling_outer_to_reduction`) — so a request deeper than the rotation
can hold reaches no further than the rotation does.  Everything ELSE in the read-ahead family
already derives: `preloaded_tiles` primes `dr_g` coords, `_readahead_shift` advances by
`dr_g`, the drain guard's predicate is that same advance's own bound, and `build_ledger`'s
`readahead-residency`/`crossing-RAW` rows are derived from both.  The chunk-counting consumers
were the only two reading the raw request, and both then describe a read-ahead the steady body
never issues.

MEASURED 2026-08-21 — bf16 NT KMN MI[16,16,32] MIWaveTile[2,2] DU64 (n_s=2) PGR2 SIA4, with
`ClusterLocalRead: 0` so `Solution.py:6096` does not reset PLR and the request reaches θ:
  * PLR=3.  `dr_g` is clamped to 1 (`W_g = 2`), so the steady advance and the guard predicate
    are the PLR1 ones — but the raw `ceil(dr/n_s) = 2` widened the DRAIN SUPPRESSION WINDOW
    from 1 step to 2, so the NGLL step (t = M−2) also guarded away the read-ahead reads that
    the LAST drain step's leading `wmma`s consume.  Emitted `ds_read` count: 62 at PLR1, 62 at
    PLR2, **54** at PLR3 — eight local reads silently gone against an unchanged 36 `v_wmma`,
    with `ledger_empty = True` and no warning.  With this function: 62, matching PLR1/PLR2.
  * PLR>=5.  The same raw form drove the chunk peel `M` to 3/4/5 (PLR 5/8/10) while
    `off(copy, iter)` stayed 2 — extra unrolled drain steps and a raised `T > M` short-arm
    threshold for a pipeline depth that does not exist.  With this function `M` stays 2.
Neither shape is reachable at the default `ClusterLocalRead = 1`, where `Solution.py:6096`
forces `PLR := 0` once `PLR >= LoopIters`; `HalfPLR != 0` also sets `ClusterLocalRead = 0`
(`Solution.py:3103`), so the user does not have to spell it to reach here.

NOT A CLOSED-FORM REWRITE OF THE OLD LINE, and the tempting claim that it is would be wrong.
It is inviting to say `((span−1) + dr_g·fan)//span` "reduces to `ceil(dr_g / n_s)`, the same
integer with the request swapped for the realization".  Three DIFFERENT denominators are in play
and that sentence silently unifies them: `peel_depths` used the `max` reduction-inner extent,
`emit` used `_reduction_rate_mode`'s extent, and `chunk_reach` uses the INNERMOST reduction-inner
extent.  They coincide on 100% of the shipping matrix but differ in FORM on 47% of it, so the
equality is a measured coincidence, not an identity.

Likewise "new <= old always, since `dr_g <= dr` and ceil is monotone" IS NOT A PROOF — two
ceilings with different denominators are not comparable that way.  What is true is MEASURED:
over every θ in all three shipping yamls the two agree except where §2.6 clamps
(69,108/69,108 `peel_depths` calls and 11,518/11,518 drain builds identical; f8 152/152 and
mxf8 404/404 kernels byte-identical).  One unrealized WIDENING case exists on paper —
`n_s = 1, s_sub = 2, dr = dr_g = 2` gives old 1, new 2 — which `build_S` pins away today
(`W_g = 2` in every logged case) but which nobody has excluded in general.

`S = None` derives the depth map itself (`build_S`); pass it when the caller already has one.


## `reload_positions`

Per-name ANCHORS for a read-ahead of `shift` — where each refill must be emitted (#331).

Returns `{name: coord}` mapping a rotation name to the `ord` coordinate its refill must FOLLOW,
or `None` when no such placement exists (which is exactly `register_reuse_verdict == clobber`).

THE RULE, and it is arithmetic rather than policy.  In a body of `T` steps, a name written at
step `p` and last consumed at step `u` is live for `(T - p) + u` steps once the read-ahead
makes its consumer the NEXT trip.  One generation is live at a time iff that is `< T`, i.e.

    p > u          — the refill follows the LAST consumer of the name it overwrites

So the anchor is the coordinate of that last consumer.  MEASURED (NKM, A):

    name (K0,M0) last consumed at 11 -> anchor 12      currently emitted at 1
    name (K0,M1)                  12 ->        13                          3
    name (K1,M0)                  14 ->        15                          6
    name (K1,M1)                  15 ->        16                          8

The anchors are STAGGERED through the broadcast axis's last pass, not bunched at the body end:
each load issues as early as it legally can, which is the maximum slack before its use next
trip.  Cost is nil — same names, same `W`, no unrolling; only the position moves.

WHY THE COORDINATE AND NOT THE STEP INDEX: `steps[u % T]` is a coordinate over `ord`, which is
what `Inst.anchor` carries and what survives the rolled nest.  See `ir.Inst.anchor`.

SAME WALK AS THE VERDICT (`_register_timeline`), deliberately: "is there a legal position" and "which
position is it" are one question asked twice, and answering them separately is what produced a
verdict of `safe` over an unmoved read (the 2026-08-28 miscompile).

AND THE ANSWER IS CHECKED AGAINST THE SCHEDULE IT CREATES (#332).  The anchor is read off the
UNANCHORED walk, but emitting it MOVES EVERY NAME'S REFILL — one `Inst` carries all of them, so
deferring the read to the last pass of the invariant mode re-times the whole set, and the
positions the anchor was derived from no longer exist.  The moved schedule can be worse than the
one it replaces: MEASURED, `MNK` + `TDMSplitA=1, TDMSplitB=2`, A at `shift=1` — unanchored the
refills sit at `N_inner=0` and slot 1's last consumer is at `N_inner=1, K_splitB=1`, so the walk
anchors at `N_inner=1`; but at that position the `K_splitB=0` leaf refills slot 1 and the
`K_splitB=1` leaf of the SAME pass then consumes the old generation.  The emitted plan showed it
as `[(13,'use'), (15,'refill'), (17,'use')]`, hardware showed it as 96 miscomparing cells, and
every layer in between was green because nothing re-walked.  So the anchor must be a FIXPOINT:
propose it, re-walk with `at` set, and refuse the depth if the move does not actually rescue
it.


## `hop_quantum_axes`

The free axes ONE instruction of `hop` may span — Precondition Q (§2.2), decided.

Two bounds, and the tighter wins:

  TARGET   `quantum_tile_cap` — the instruction's payload in tiles.  A wide load cannot span
           more tiles than it moves.
  ORDER    the axes must not be time-separated: "an axis may sit *inside* one hop's quantum
           **only if `ord` and `off` do not separate its coordinates in time** — i.e. only if
           every coordinate the quantum spans is scheduled at the same generation" (§2.2).

The candidates are taken as a SUFFIX of the operand's free presence modes in ord order —
innermost first — because a quantum is a `tile` on the hop's data axis (§2.8 move 1) and a
`tile` appends sub-modes at the inner end; a non-suffix set would name a strided gather no
single instruction issues.  Each candidate is tested against the decoder's own generation map,
and the widest generation-uniform one wins.  Narrowing is the §2.2 repair, never a rejection:
the empty tuple is always admissible and simply means one tile per instruction.

THE UNIT IS A **FACTOR** OF AN AXIS, NOT THE AXIS.  §2.2 says the quantum is set by a `tile` on
the hop's data axis, and a `tile` SPLITS: an axis of extent 4 under a 2-tile load merges its
coordinates in PAIRS and leaves two instructions, it does not collapse to one.  So the result is
`((axis, factor), ...)` — how many consecutive coordinates of that axis one instruction spans —
and `factor < extent` is the ordinary partial case, not an error.  (The half-wave MX TileSpan is
this same shape one level up: it merges two tiles into one load, so four tiles leave two loads.)
An earlier version returned bare axis NAMES and required `Π extent <= cap`, which silently
refused to merge at all whenever the axis was wider than the load — right at `MIWaveTile [2,2]`
by coincidence, wrong at [4,4] and at the [8,7] acceptance shape.

Only the INNERMOST candidate axis may take a partial factor; an outer axis is either spanned
whole or not at all, because one instruction covers a CONTIGUOUS run of the flat traversal and a
partial outer axis would stride over the inner one.

MEASURED (2026-08-19, MXBlock 32 / MatrixInstK 128 / MIWaveTile [2,2] / 4 waves / PGR1 PLR1):
    KMN     every operand absorbs its free axis   -- one ds_load_b64 per scale, legal
    KMNMNK  MXSB and B absorb theirs; MXSA and A absorb NOTHING
and the asymmetry is §2.6 term (ii): `MXSB`'s broadcast mode `M_inner` is outer to ALL of
`pres(MXSB)`, so its fan forces `dr_g = 0` and its read-ahead shift is 0 — both its tiles read
one generation.  `MXSA`'s broadcast mode `N_inner` is NOT outer to all of `pres(MXSA)`, so it
keeps a shift of 2 and its two tiles land in different reduction chunks.  Merging those two
into one `ds_load_b64` is the bug this function exists to stop: one instruction, one LDS base
pointer, two generations.


## `reloads_whole_set`

Is `op` invariant over a mode OUTER to its own read-ahead level?

That is the shape whose every register name is consumed once per pass of the invariant mode and
is therefore live for the WHOLE trip — A under `NKM`/`NMK`, B under `MKN`/`MNK`.  It is exactly
the population §2.6's assignment walk refuses a sub-trip advance for, and the reason is
structural rather than a width shortfall: see `prefetch_distance_for`.

The partner operand is NOT this shape even though it is also broadcast: its invariant mode is
INNER to its level, so its values die within the pass and an ordinary partial advance is safe.
That asymmetry is the whole content of "PLR works on KMN but not NKM".

OVER `reload_modes`, NOT `_presence` — AND THAT IS WHY IT IS NOT `_broadcast_fan > 1` (#332).

The two are the same predicate asked about two different spaces, because they answer two
different questions:

  `_broadcast_fan`  — CAPACITY, §2.6 term (ii): how many generations are live at once.  That is
                      a question about NAMES, so it is asked over `_presence`.  Under `M·N·K`,
                      A is re-read at each `n` into the same name WITH THE SAME VALUE, so no
                      second generation exists and `fan(A) = 1` is right.
  this              — THE ADVANCE UNIT: how far one look-ahead may step.  The shift moves
                      through `reload_modes` and nothing else (§5.3.1 pt5 — a free-axis region
                      sibling BOUNDS it), so the mode that has to be outer is outer to the
                      RELOAD set.  If it is, then every name is consumed once per pass of that
                      mode and no SUB-traversal advance can be legal; the unit is a whole
                      reload traversal.

`reload ⊆ pres`, so this is strictly WEAKER and every operand the fan flags is still flagged —
the only configurations that newly qualify are those with a free-axis region split, where the
two sets actually differ.

MEASURED, and this is what overturned the earlier reading that the presence form was "the right
one physically".  `MNK` + `TDMSplitA=1, TDMSplitB=2`, A: `pres = {M_split, M_inner, K_splitB,
K_inner}` but `reload = {M_inner, K_splitB, K_inner}`, and `N_inner` is outer to all three.
Under the presence form A took a SUBSTEP advance, whose refill lands mid-pass on a name the
same pass still consumes — 96 miscomparing cells on gfx1250.  The alternative repair, refusing
the depth, is a PLR downgrade: `dr_g = 0` against a requested `PLR = 1`.  The whole-traversal
advance is the one answer that is both correct and honours the parameter, because at
`shift = n_reload` the shifted coordinate EQUALS the leaf's own, so each refill lands exactly
on the name it just finished consuming.


## `_broadcast_fan`

`fan(p)` — §2.6 contributor (ii), in its DECODER-EVALUABLE form.

§2.6 gives the live-peak floor per operand as a closed form with two contributors,
`L(p) = (dr_g + 1) + [1 if term (ii) fires]`, and states term (ii)'s test as a pure
presence/`ord` predicate:

    term (ii) fires for `p` iff there is a mode `q ∉ pres(p)` of extent > 1 that is outer,
    in `ord`, to EVERY mode of `pres(p)`.

We return the PRODUCT of those qualifying modes' extents — `fan(p)`, the number of times `p`'s
whole register-name space is re-traversed before any one of its values dies — because §2.6
also fixes the units: compare per NAME (`names(p) = Π extent over pres(p)`), not per rotation
ring.  `fan(p) > 1` ⇒ term (ii) fires ⇒ `L(p) > W_g` for every `dr_g ≥ 1`, so Lemma 3b forces
`dr_g = 0` — at ANY `n_s`.  `fan(p) == 1` ⇒ no term (ii).

The "outer to EVERY mode" is load-bearing and is what separates the two operands of a 3-axis
order; the weaker "broadcast over some mode outer to the reduction" reading zeroes BOTH.  Under
`ord = M·N·K`:
  A: pres {M,K}, broadcast over N — but N sits BETWEEN M and K, so it is not outer to
     `M ∈ pres(A)`.  fan(A) = 1.  (A is re-read at each `n` into the same name, but with the
     SAME value: harmless.)
  B: pres {N,K}, broadcast over M, and M is outer to BOTH N and K.  fan(B) = extent(M).
     B's whole name space is re-traversed once per `m`, so a read-ahead write lands on a name
     still consumed later in the fan — for ANY advance distance, unrescuable by ordering
     (the simultaneous-input race of Lemma 3b, not a WAR).
KMN/KNM: the broadcast mode is interleaved with the presence modes, never outer to all of them,
so neither operand qualifies and both keep read-ahead.

The extent > 1 test is part of the predicate (a broadcast mode of extent 1 re-traverses
nothing, so it cannot force a second live generation), and it is a GUARD, not a live branch on
the parameter path: `translate` drops degenerate modes, so a θ built from parameters never
carries an extent-1 inner mode and the clause never fires there.  It is kept because
`inner_modes()` itself makes no such promise — a hand-built θ (the multi-group work, #118) can
carry one — and `test_broadcast_fan_ignores_a_degenerate_mode` covers it directly, since no
parameter config can.

§2.6 records that this predicate agrees with a direct live-range walk on the joint
`(free-tile, rotation-slot)` name space; `test_loopmodel` keeps that walk as an oracle.


## `_shifted_coord`

The read-ahead-shifted coordinate along presence mode `m`: `((P + shift) // stride(m)) % ext(m)`
over the flat presence position `P = Σ stride(m')·idx(m')`.

This is the re-decomposition that makes the advance carry between modes: overflow out of the
reduction mode lands in the next free-mode tile, and overflow past `n_pres` is the reduction-chunk
crossing `src_slot` already models.

A SIBLING-ENUMERATING MODE IS OUTSIDE THE TRAVERSAL, so its coordinate is UNSHIFTED (§5.3.1 pt 5).
`_readahead_shift` drops free-axis region siblings from `strides`, because "a free-region sibling
BOUNDS the shift, which then rolls the chunk rather than crossing into the neighbour."  The
sibling this read belongs to is fixed by its grid coordinate; the advance happens INSIDE it.
IT KEEPS ITS OWN COORDINATE, which is NOT the same as the caller's `None`.  `emit.py`'s
`(m, _shifted_coord(...)) if shift else (m, None)` uses `None` for the WHOLE-OP no-shift case,
where the walker supplies every coordinate itself; returning `None` for a single mode inside a
shifted op instead DROPS that coordinate, and the region-1 reads vanish (measured: `MKN`
`split=[2,2,1,1]` PLR1, "wmma reads A buffer X0 tile 1, which no read in scope has filled").
The identity `Expr(var=m)` is the right answer: the sibling index is whatever the enclosing loop
says, carried through unchanged.

A FOLDED MODE IS ADVANCED IN POINTS AND EMITTED AS A TILE (§2.2, #312).  `strides`/`ext` count
PRESENCE POINTS — a `q`-folded axis contributes `N/q` carrier groups, not `N` tiles — so the
expression above lands on a GROUP INDEX `g`, while the coordinate every consumer reads (the
unrolled nest, `refs.covered_coords`, the wmma's source) is a TILE.  The group is named by its
LEADER, so the tile is `g·q`: one multiply, and the same naming `emit._first_touch_modes`'
`mode % q == 0` guard uses when it picks which act issues and `covered_coords` uses when it
expands the leader over its group.  `q = 1` (every unfolded mode, which is every mode of every
kernel that sets no `ReadPhi`) leaves the `carry` form below untouched, so this is inert
wherever there is no fold.

THE FLAT POSITION IS BUILT BY `_pos_terms`, which is where the OTHER half of the same
tile-vs-point mismatch lives: the loop VARIABLE this expression reads is a tile index too.

THE SCALE GOES OUTSIDE THE `mod`, NOT INSIDE.  `((P+shift)//stride) % ext` must reduce over the
GROUP count `ext = N/q` and only then scale — reducing over `N` after scaling would let `g·q`
reach `N·q`.  `Expr.digits` is exactly that shape (`coef · ((pos // div) mod ext)`) with a
single field, which is why no new `Expr` form is needed.


## `_readahead_depth`

`dr_g` for a READ INSTRUCTION — the min over the register groups that instruction fills.

One `Inst` writes one advance, so a read covering groups of differing depth must take the
shallowest (`emit._read_groups` splits the read when the depths differ, so this min is normally
over a uniform-depth subset).

A FREE SIBLING OUTER TO THE REDUCTION FORCES SHAPE B (`dr_g = 0`), §2.6 + §5.3.1 pt 5.  A
cross-tile read-ahead needs two things and this configuration supplies neither.  (i) The advance
cannot reach the next tile: the sibling is a storage-disjoint group, so the shift "must not carry
through it" — it "BOUNDS the shift, which then rolls the chunk" (`reload_modes`).  (ii) The
prologue cannot prime the other sibling either: Lemma 3d's outer factor is "{first coord of
pres ∩ OUTER}", and line 593 says a region outer to the reduction "contributes its first
coordinate only (region 0 alone in the read-ahead prologue)".  So sibling 1's leading substep is
primed by nothing and advanced to by nothing — precisely §5.3.1's "a group is either
read-ahead-pipelined or in-place, never a mixture", and the in-place answer is the available one:
`dr_g = 0`, empty prologue, refill under the `S=δ` register WAR.

This is the same SHAPE of veto `_broadcast_fan` already applies for §2.6 term (ii)
(`fan(p) > 1 ⇒ dr_g = 0`) — a structural property of `(ord, tile)` that forbids the hoist —
and it is why the exclusion in `reload_modes` is not sufficient alone.

MEASURED 2026-08-21 over the 168-cell order x split x prefetch sweep (`check_plan`):
    exclusion off            0 failing   (but the emitted six-axis kernels are wrong on hardware)
    exclusion only          16 failing   (`MKN`/`NKM` x outer free split x PLR1 — sibling 1's
                                          leading substep read by nothing: "wmma(idx0=1,u=0)
                                          reads A buffer X0 tile 1, which no read in scope has
                                          filled")
    exclusion + this rule    0 failing
When the sibling is INNER to the reduction (the six-letter words) the hoist stays legal and only
the shift narrows, which is the case this whole change exists to repair.

The veto itself lives in `prefetch_steps_for` — the per-group derivation this function mins
over, and the one `preloaded_tiles` also reads — so prologue and steady cannot disagree
about the shape.


## `reload_modes`

The modes `op`'s register group RELOADS over — the traversal the steady read-ahead shift
walks, and the ONE derivation every consumer of that shift must share (§5.3.1 pt 5).

NOT the same set as `_presence`, though it is *currently* equal to it — see "STATUS" below.
§5.3.1 pt 5 states the shift is taken "over `p`'s flat traversal of its own group's RELOAD modes
— the modes `R` counts (the group's `Q(Λ)` rotation axes, §2.6), *not* every mode in `pres(p)`",
and that a SIBLING-ENUMERATING mode ("a region split on a FREE axis, the `{lo}{hi}` partition, or
a reduction axis realized as CONCURRENT SIBLINGS rather than a rotation") "selects a different
group's band, not a later generation of this group, so the shift MUST NOT carry through it".
Reduction axes and a genuinely-reloaded free fan DO stay — the paper says so in the same
sentence, which is what keeps a DU split's `K_split × K_inner` rotation (#254) intact.

WHY THIS IS A FUNCTION AND NOT AN INLINE FILTER.  `_ord_strides` renumbers whatever mode set it
is handed, so narrowing the set at ONE call site silently changes the radix that site decomposes
on while the others keep the old one.  `emit.py` states the invariant that breaks: "the same
`shift` drives the register slot and the shared `src_slot` in `_read_placement`, so the three
cannot drift."  Three sites each calling `_presence` independently cannot honour that by
construction — they honour it only if someone keeps them equal by hand, which is exactly what
failed when the exclusion was tried inline (`MKN split=[2,2,1,1]` PLR1: "wmma reads A buffer X0
tile 1, which no read in scope has filled", and `NKM` symmetrically on B).  One function is the
fix for that class of bug, independent of what the function returns.

NOT THE TREE LEVEL.  `placement_level` keeps using `_presence`: §5.3.1 places an op-class at
`level(p) = innermost mode of pres(p)`, and a free region sibling is genuinely present (line 593:
it "is in `pres(p)` and thus contributes to the prologue" by its `ord` position).  The two uses
are distinct — presence for the prologue and the tree, reload for the shift — and only the
latter drops siblings.

THE OTHER HALF IS `_readahead_depth`.  Excluding the sibling here is necessary but not
sufficient: when the sibling is OUTER to the reduction the group cannot read ahead at all, and
`_readahead_depth` returns 0 for exactly that case.  Enabling this alone regressed 16 of the 168
sweep cells (`MKN`/`NKM` x an outer free split x PLR1); with both, the sweep is clean.  See
`_free_siblings` and the Shape-B note in `_readahead_depth`.


## `_shifted_rate_slot`

The rotation slot of the READ-AHEAD-SHIFTED position: the group's mixed-radix RATE INDEX
evaluated at `P + shift` instead of at `P`, mod `W` (#254).

`_rate_slot` is `Σ radix_m · idx(m)  mod W` over the group's rate modes.  The read-ahead reads
the value at flat presence position `P + shift`, so its slot is that same index evaluated
there — which means re-decomposing `P + shift` into per-mode DIGITS,
`digit_m = ((P + shift) // stride(m)) mod ext(m)`, and recombining them in the same radix.

IT USED TO BE ONE DIGIT.  The old form substituted the single reduction rate mode's shifted
coord for the whole slot and REFUSED when a group had more than one, because a sum of
floor-divided coords had no `Expr` shape.  `Expr.digits` is that shape, so the general form is
now just the general form: every rate mode contributes its digit at its own radix.

Two properties worth stating because they are what make this safe to apply unconditionally:
  * ONE rate mode reduces to exactly the old expression whenever `W == ext(m)` — which is
    every `unroll`-policy group, i.e. every reachable config (`W = R = ext(m)` there).  Where
    they differ (`W < ext(m)`) the digit form is the correct one: the slot is `rate_index mod
    W`, and `rate_index` is the digit, not the raw quotient.
  * The digits are taken over ALL rate modes, matching `_rate_slot`.  Substituting only the
    reduction mode's digit silently dropped a free rate mode's contribution for any fragment
    that keeps its fan in the rate (no `grouping_mode`).

Returns None when the group has no rate mode inside this operand's presence traversal, so the
caller keeps its unshifted slot.

THE RADIX IS IN PRESENCE POINTS, LIKE THE POSITION IT DECOMPOSES (#312).  A `q`-folded axis
contributes `N/q` GENERATIONS to the ring, not `N` — the `q` co-carried tiles arrive in one
instruction and are therefore one generation by construction (§2.2) — so the digit's extent is
the folded one — which is what `_rate_modes` now reports — and the flat position is built by
the shared `_pos_terms`, whose per-term divisor converts each tile-valued loop index to the
presence point it names.


## `_rate_modes`

The inner modes ONE register group cycles distinct values over — its RATE modes (§2.6):
non-`_rate_broadcast` and of extent > 1, in ord order (outer→inner).  The rotation slot is the
mixed-radix index over THESE modes mod W — NOT the single innermost-presence mode (a
sibling-selector like `min` is nullspace-for-rate and must not drive the ring).

AN EXTENT-1 MODE IS NULLSPACE FOR EVERYTHING — it has one value, so the group is constant over
it by definition, and dropping it cannot change either the mixed-radix index (its coord is
always 0, its radix 1) or `group_rate` (a factor of 1).  It is dropped anyway because
`_reduction_rate_mode` takes the INNERMOST reduction rate mode, and a degenerate one shadows
the real substep axis: under a DU `TDMSplit`, `DepthU` re-factors as `K_split(2) x K_inner(1)`
and the innermost reduction mode becomes the extent-1 one.  The read-ahead would then shift
along an axis with no values, and the several-reduction-rate-modes guard below would fire on a
pair that is really one axis.

THE EXTENT IS IN PRESENCE POINTS, NOT TILES (§2.2, Lemma 3e).  A `q`-folded axis contributes
`N/q` GENERATIONS to the ring, not `N`: by Precondition Q the `q` tiles a carrier group
co-carries share one step, so they are ONE generation by construction, and Lemma 3e's walk
"enumerates presence points, never raw tiles".  A rotation cycling `N` names over `N/q`
generations would park each value in a slot no consumer names.  So the extent comes from
`hop_extents` (the folded view); the loop VARIABLE stays a tile index and `_pos_terms`'
per-term divisor converts it — the same tile-vs-point split as everywhere else (#312).

UNREACHABLE TODAY, AND THE INVARIANT IS WORTH NAMING because it holds of the BRIDGE, not of θ.
A fold candidate is a free, non-broadcast, non-reduction inner axis
(`geometry._quantum_candidate_axes`); `_rate_broadcast` removes the `grouping_mode` and the
free `region_modes`; and `translate.reg_fragment` happens to make every free inner axis one or
the other (`grouping_mode = M_inner` whenever its extent > 1, `M_split` a region mode), so no
folded axis reaches this list — measured over 2240 configs, 0 overlaps.  That is two
independent facts of one bridge, not a theorem about θ, which is why the arithmetic here is
written for the general case rather than asserting the accident.
`test_a_folded_axis_that_is_a_rate_mode_counts_presence_points` pins both halves.


## `preloaded_tiles`

The register-reads `op`'s read-ahead prologue must pre-issue (Lemma 3d general construction,
§5.3.1 pt5).  Returns a list of coord dicts {inner-mode-name: value} over `op`'s presence.

CLOSED FORM — three independent factors, NO leaf simulation (each factor intersected with
`pres(op)` so a broadcast axis drops out).  Split `ord` at the reduction-substep mode `ik`
(INNERMOST reduction inner mode, extent `n_s`): `ord = [OUTER… | substep(n_s) | INNER…]`
  prologue(p) = { first coord of pres(p) ∩ OUTER }
              × { leading min(dr_g, n_s) substeps, or ALL n_s if the ord INNER block is empty }
              × { pres(p) ∩ INNER }
- `dr` counts REDUCTION-SUBSTEPS, clamped at `n_s`; INNER-block empty → full n_s (dr-independent).
- PER-OPERAND-PRESENCE: a broadcast axis (mode ∉ pres) drops out; A and B may differ in arity.
- OUTER factor = the single first tile.

PER-GROUP read-ahead depth (§5.3.1 pt5, "Heterogeneous register width").  The depth is
`dr_g = prefetch_steps_for(...)` — the SAME quantity the steady advance uses — and the
paper's net rule is that the two are equal by construction:

    prologue depth = steady advance = `dr_g` for every group; round up to the first tile's
    full K only when `dr_g ≥ 1` and INNER = ∅; a `dr_g = 0` group prologues NOTHING and
    refills in place.

So a Shape-B group (`dr_g = 0`, whether from `W_g = 1` or from the §2.6 term-(ii) fan)
contributes no coord here, and its steady advance is 0 — one shape, not a mixture.  Pairing a
Shape-A prologue with a Shape-B steady is the named failure: the pipe is primed and then never
re-primed.  Because `build_ledger` derives the `readahead-residency` obligations from THIS
function and `emit.fill_reads` emits from it, the two cannot disagree — suppressing one side
alone is what previously left obligations undischarged.

`S` supplies the per-group widths and is REQUIRED for that derivation; `S=None` (a caller with
no depth map) falls back to the requested `dr` for the whole operand.  dr=0 → empty.

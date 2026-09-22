# ADR 0008 — LoopModel docstrings and comments, verbatim

The code states the claim in one line (Constraint 2, tightened to a 5–10% prose budget).
Everything that used to sit beside it is here, unedited, keyed by where it was.

## `Tensile/LoopModel/__init__.py <module>` (module)

```
LoopModel — the θ schedule-model decoder, ported into TensileLite (was `spacetime` in the
decoder_proto repo).

The CORE (ir, theta, placement, emit, ledger, sizing, render, geometry, latalg) has NO
TensileLite / AMD vocabulary.  The `adapter` package maps TensileLite parameters onto θ;
nothing in the core imports it.  Build a θ directly, or via `adapter.params_to_theta`.

The named example configurations are NOT here: they are test fixtures, consumed only by the unit
tests, and live with them in `Tensile/Tests/unit/loopmodel_scenarios.py` beside `gir_fixtures.py`.
They were in this package as `scenarios.py` and had no library consumer.

θ = (tile, path, ord, off, S, ρ, Φ).  S is DERIVED from (ord, register-axis tile);
the reuse scheme is the fragment's rate-group partition (a `tile` value), not a flag.
```

## `Tensile/LoopModel/__init__.py:1` (comment)

```
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
```

## `Tensile/LoopModel/adapter/__init__.py <module>` (module)

```
The adapter: TensileLite Solution -> theta. The only place AMD/TensileLite vocabulary appears.

Two hops, in this order:

    kernel dict  --solution.py-->  parameters  --build.py-->  theta

`solution.py` knows TensileLite key names (`MIWaveTile`, `TDMSplitA`, `PrefetchLocalRead`, ...) and
nothing about the schedule model. `build.py` knows the schedule model and nothing about TensileLite.
`loop_order.py` turns the LoopOrder word into the canonical `ord`; `fragments.py` counts elements;
`target.py` holds the two hardware constants.

Was two modules, `bridge.py` and `translate.py`, which split one job along no particular line.
```

## `Tensile/LoopModel/adapter/__init__.py:1` (comment)

```
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
```

## `Tensile/LoopModel/adapter/build.py.params_to_theta` (function)

```
Translate a TensileLite-param dict into a CANONICAL θ point.

    Builds: ord as Modes — the OUTER reduction-chunk level (first-class, extent 0) plus the inner
    reduction substep and the free tiles; each operand a Trajectory of
    Hops carrying per-hop retime delta; each fragment's register-axis rate-group `tile`
    (the reuse coordinate) + nullspace signature (broadcast over the other operand's free
    axis, plus reuse over the substep).  S is then DERIVED, not flagged.
```

## `Tensile/LoopModel/adapter/build.py._place_read_offsets` (function)

```
Write `off(read, <level>) = PLR` for each requesting op-class, at ITS OWN read-ahead level.

    THE TRANSLATION OF AN OUTER PARAMETER INTO θ (#321).  `PrefetchLocalRead` is stated in
    outer-language terms — the minimum wmma-REGION read-ahead — and θ has no such notion; what θ
    has is `off` on a level.  The level is `theta.readahead_level(th, op)`: the outermost mode of
    that op-class's read-hop PRESENCE, free region modes dropped.

    IT IS DONE AFTER THE `Theta` IS BUILT because the level is a presence fact, and presence is a
    per-hop derivation that needs the assembled θ (`presence_modes`, §2.2's spanned-axis
    absorption).  Computing it from the raw `ord` beforehand can only approximate presence, and the
    approximation is what put the MX scale rings' read-ahead on the axis their load spans.
```

## `Tensile/LoopModel/adapter/build.py._fit_register_budget` (function)

```
DERIVE THE REGISTER PARTITION from the budget (#315/#318): how many fan tiles stay at the
    derived width, and how many fall to their floor.

    The width is derived (`'pipeline'`, §2.6) but the PARTITION is not derivable from correctness —
    §2.6 makes `W` a *searched* coordinate in `[L_war, budget]`, so what decides how many tiles are
    deep is the budget and nothing else.  This is that search, and it is the only place a partition
    is created.

    THE SHAPE IS THE ONE THE ALLOCATOR CAN ACTUALLY REALIZE, which is why it is two parts and why
    the deep part is the LOW tiles.  With `{lo: W, hi: L_war}` GIR names buffer `X0` for every tile
    and `X1..X{W-1}` only for `lo`'s, so the emitted blocks are ragged — `X0` spans the whole fan,
    the rest span `d` tiles — and the register cost is `frag · (fan + d·(W−1))` rather than
    `frag · fan · W`.  MEASURED: at `MIWaveTile[8,8]`, `X0` holds tiles 0-7 and `X1` holds 0-3.
    That ragged block is exactly what ULM0's `HalfPLR` means: `valuBlocks = 1.5` is `d = fan/2`.

    THE SEARCH IS LARGEST-d-THAT-FITS, per the objective: take the most prefetch the budget allows,
    and bank nothing beyond it.  `d = fan` is the unpartitioned kernel and is tried first, so a
    kernel that already fits is left alone and no partition is created — a partition that saves
    nothing would still cost the `reg_group` guards at emit.

    `budget` is None for every caller that does not hold a register ceiling (unit tests, the
    `OutputLoopIR` dump), and then this is a no-op — the same standing `ReadVectorElems` has.
```

## `Tensile/LoopModel/adapter/build.py:1` (comment)

```
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
```

## `Tensile/LoopModel/adapter/build.py:79` (comment)

```
# NOTE (MI shape): the default MI 16×16×64 matches the PAPER's base GEMM (paper line 52) and gives
# substep = DepthU/MI_K = 128/64 = 2 (the a/b that makes the flagship two-rate). The gfx1250
# CATALOG fp16 opcode is actually 16×16×32 (8 VGPRs/frag); K=64 fp16 is not a real single opcode.  (ADR 0007 #15)
```

## `Tensile/LoopModel/adapter/build.py:95` (comment)

```
# PER-OPERAND overrides. -1 = "use the whole-kernel preset" (PrefetchGlobalRead /
# PrefetchLocalRead / NumLdsBlk respectively), which is the historical single-value behaviour.  (ADR 0007 #14)
```

## `Tensile/LoopModel/adapter/build.py:102` (comment)

```
# {operand: elements ONE shared->register instruction moves per lane} — the movement quantum's
# DOMAIN (§2.2), which is the target's instruction selection and so has no Solution key; the
# bridge fills it from the writer.  (ADR 0007 #12)
```

## `Tensile/LoopModel/adapter/build.py:106` (comment)

```
# {operand: ir.QuantumMap} — the §2.2 movement quantum, SUPPLIED. Two Exprs over the tile
# coordinate (`carrier`, `slot`); load-vs-broadcast is the RANGE of `slot`, not a tag.  (ADR 0007 #11)
```

## `Tensile/LoopModel/adapter/build.py:164` (comment)

```
# Prefetch depths are the copy (PGR) / read (PLR) retime offsets — Lemma 1's peel is general in
# off (any depth), so these are NOT capped: dg=PGR, dl=PLR verbatim. (An earlier min(_,2) cap
# was an artificial limit that broke PGR>=3; removed.  (ADR 0007 #10)
```

## `Tensile/LoopModel/adapter/build.py:187` (comment)

```
# REGIONS ONE WAVE OCCUPIES (#245).  Sizes the READ's tile axis and decides whether the read
# varies over the region axis at all; the COPY keeps `aMT`/`bMT` either way.  Absent (a
# hand-built param dict) = the split factor, i.e. the previous behaviour exactly.
```

## `Tensile/LoopModel/adapter/build.py:193` (comment)

```
# DU (shared K): ONE axis at the FINEST split; a coarser operand is present on it at a
# stride.  See ADR 0006.
```

## `Tensile/LoopModel/adapter/build.py:218` (comment)

```
# canonical-mode name sets for role/region/grouping binding (only >1 modes survive in ord;
# naming a dropped mode is harmless — the core intersects with the present inner modes).
```

## `Tensile/LoopModel/adapter/build.py:244` (comment)

```
# §2.2's CARRIER GROUP IN θ'S OWN VOCABULARY. `ReadPhi` is Φ (movement instances merged into
# one cooperative instruction) and `ReadRho` is ρ (the sub-agent span they are distributed over
# — the sub-wave partition, a `tile` of the lane mode).  (ADR 0007 #7)
```

## `Tensile/LoopModel/adapter/build.py:319` (comment)

```
# Bind `grouping_mode` to the in-region fan (when it exists) so the fan folds into
# |part|×region and NOT into the rate — this is what prevents the fan double-count (§5.2)
# for BOTH A and B.  Absent fan (extent 1) => no grouping mode, as before.
```

## `Tensile/LoopModel/adapter/build.py:332` (comment)

```
# copy-hop region split = HOW MANY regions this operand's copy moves, which is its own count on
# each region axis — `aDU`, not the shared axis extent.  (ADR 0007 #5)
```

## `Tensile/LoopModel/adapter/build.py:344` (comment)

```
# The ALLOCATION bound, which is not symmetric with the register side. VGPR buffers we can grow
# to match θ (#122); LDS we cannot — `Solution.py` sizes the LDS block from NumLdsBlk and
# checks it against MaxLDS, so a ring DEEPER than it reserved writes past the allocation.  (ADR 0007 #3)
```

## `Tensile/LoopModel/adapter/build.py:361` (comment)

```
# `free_split` is `aMT`/`bMT` -- the FREE-axis region count -- carried as a number so the
# sibling bound (§5.3.1 pt 5) has an anchor that survives a mode reassignment (#245).  It is
# NOT `splitA`/`splitB`, which are `mSplit * aDU` and fold the reduction axis in.
```

## `Tensile/LoopModel/adapter/build.py:422` (comment)

```
# A fused group pairs its members BY REGION INDEX (§4.3), which is only well-defined when
# every member has the same region count — a mixed-split group (e.g. {MXSA(1), A(2)}) has
# no region-1 instance of MXSA to pair with A's.  (ADR 0007 #2)
```

## `Tensile/LoopModel/adapter/build.py:461` (comment)

```
# the role comes off the HOP the offset retimes, so the key cannot drift from the
# trajectory it describes (`Hop.role`, theta.HOP_ROLE).
```

## `Tensile/LoopModel/adapter/build.py:467` (comment)

```
# THE LEVEL NEEDS THE BUILT θ (it is the hop's PRESENCE, §2.2's absorption included),
# so the entry is deferred to `_place_read_offsets` below rather than guessed here.
```

## `Tensile/LoopModel/adapter/build.py:584` (comment)

```
# SCORE WHAT THE ALLOCATOR EMITS, not the model's ragged footprint — see
# `geometry.operand_emitted_regs`.  (ADR 0007 #1)
```

## `Tensile/LoopModel/adapter/fragments.py._per_operand` (function)

```
Resolve a per-operand override `<base><operand>` against the whole-kernel `preset`.

    `-1` (or absent) means "not overridden" and yields the preset, so the historical single-value
    behaviour is the default and a caller that sets nothing sees no change.  The operand name is
    the theta op-class name ('A'/'B'); an op-class with no `<base><name>` key — the MX scales, the
    accumulator — simply takes the preset.
```

## `Tensile/LoopModel/adapter/fragments.py._frag_elems` (function)

```
Elements one lane holds for one MI-tile fragment = MIInputPerThread.

    GROUNDED (gfx1250 isa 12, wave32): MIInputPerThreadA = MI_M · MI_K · MatrixInstB / 32
    (MatrixInstruction.py:129-131).  For A pass free_mi=MI_M, for B free_mi=MI_N.
    gfx1250 uses this MFMA-style formula (the RDNA `= MI_K` override is isa 10/11 only).
    VGPRs = ceil(elems · elem_bytes / 4); e.g. 16×16×32 fp16 → 16 elems → 8 regs;
    16×16×128 fp8 A → 64 elems → 16 regs (WMMA guide §3/§8, verified vs the source).
```

## `Tensile/LoopModel/adapter/fragments.py._acc_frag_elems` (function)

```
Elements one lane holds of the ACCUMULATOR fragment = `MI_M · MI_N · MatrixInstB / 32`.

    THE ACCUMULATOR IS K-INDEPENDENT, and that is the whole point of a separate helper.  An input
    fragment is `MI_M · MI_K · B / 32` (`_frag_elems`) because a lane holds a slice of the
    reduction; the accumulator holds a slice of the OUTPUT TILE, which has no `K` extent at all —
    a 16x16 fp32 tile on Wave32 is `256/32 = 8` VGPRs per lane whether `MI_K` is 16 or 128.

    MEASURED DEFECT this replaces (2026-08-26): `accElems` was `_frag_elems(k, mi[0])`, i.e. the
    INPUT formula, so C's footprint scaled linearly with `MI_K` — 2x over at `MI 16x16x32`, **8x
    over at `MI 16x16x128`**, the shipping mxf8 shape (reported 3584 registers for an accumulator
    that occupies 448).  The line's own comment already said "the output tile a lane accumulates
    (M_inner*N_inner fragment)"; only the code disagreed.  Nothing is emitted from this — the
    accumulator has no hop and `sizing` uses it for FOOTPRINT REPORTING only — so no kernel
    changes, but every register-budget statement built on it was wrong, which is what made the
    `d`-dial budget solve unanswerable.
```

## `Tensile/LoopModel/adapter/fragments.py._mx_frag_elems` (function)

```
MX scale elements per lane = MIInputPerThread(data) // MXBlock · duplicateFactor,
    duplicateFactor = 32 // MI_M (=2 for MI_M=16 on gfx1250) (MatrixInstruction.py:143).
    e.g. SCALE MXBlock=32, 16×16×128: 64//32 · 2 = 4 elems → 1 VGPR (guide §6.1).
```

## `Tensile/LoopModel/adapter/fragments.py._coalesce` (function)

```
Consecutive substeps sharing one loaded register group ~ numReadsIterCoalesced.
    Mapping: how many substeps the local-read vector width spans over the compute
    width.  (LRVW/VW = 4/2 = 2 for the default two-rate kernel.)
```

## `Tensile/LoopModel/adapter/fragments.py:1` (comment)

```
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
```

## `Tensile/LoopModel/adapter/loop_order.py.canonical_loop_order` (function)

```
The CANONICAL SPELLING of a LoopOrder value: the 3-letter shortcut when the word keeps each
    axis's (split, inner) pair contiguous, the 6-letter word otherwise.

    `_loop_order_word` expands `KMN` to `KKMMNN`, so those two are ONE traversal with two
    spellings.  `LoopOrder` is a kernel-name parameter, so leaving both spellable produces two
    kernel names — `LOKMN` and `LOKKMMNN` — for a byte-identical kernel: a duplicate that the
    benchmark reports as two data points and the dedupe cannot merge.  Collapsing the doubled
    spelling here is what makes the accepted set (every word the decoder parses) and the emitted
    set (one name per traversal) agree, instead of narrowing the parameter to keep them apart.

    An INTERLEAVED word has no 3-letter equivalent and is returned unchanged; it is also the only
    form that reaches the §5.1 multi-body peel, and `Solution.py` rejects it when no split axis is
    live (there it collapses to the 3-letter order its INNER modes spell, which is the same
    duplicate problem arrived at from the other side).

    Raises the same `ValueError` as `_loop_order_word` for anything that is not a legal word, so
    validation has one authority.
```

## `Tensile/LoopModel/adapter/loop_order.py._canonical_ord` (function)

```
Build the 6 canonical Modes and emit them in the LoopOrder word order, dropping extent-1
    modes — EXCEPT an inner mode whose paired split mode is live.

    Returns (ord_modes, names, extents); names maps the canonical role -> Mode so operand
    construction can bind free/region/grouping modes by role.

    WHY A SPLIT OPERAND'S INNER MODE SURVIVES DEGENERACY.  `X_inner` is the TILE axis and
    `X_split` is the STORAGE-REGION axis; they are different things that merely coincide in extent
    when `fan == split`.  Dropping the degenerate one does not simplify the model, it MERGES two
    roles onto one mode: at MIWaveTile 2 with a 2-way split, `M_inner` has extent 1, and with it
    gone `M_split` became simultaneously the region axis and the only free-tile axis.  Every
    consumer that asks "which tile" and every consumer that asks "which region" then reads the
    same coordinate, so a region-split tile at that shape cannot state the two facts separately —
    which is why MIWaveTile [2,2] behaved differently from [4,4], where `M_inner` survives at
    extent 2 and the two stay distinct.

    A trip-1 loop level is free: it emits its body once and contributes a radix-1 digit, so every
    flat index is arithmetically unchanged; only the ROLE separation is gained.

    SCOPED TO THE SPLIT, and that scoping is load-bearing rather than conservative.  With no live
    `X_split` there is no second role to merge away — an unsplit operand simply has no region axis
    — so keeping the degenerate inner mode buys nothing and is not free after all: it measurably
    changes emission.  Those kernels pass today, so the condition is "my split partner is live",
    not "I am an inner mode".

    WHAT THE CHANGE ACTUALLY IS — MEASURED, not argued (96-config sweep over MIWaveTile x order x
    PGR x PLR x split, `_keep` forced to keep every degenerate mode, comparing the UNROLLED act
    sequence with its concrete slots):

        84 / 96   byte-identical (op, operand, register slot, lds slot) sequence
        12 / 96   differ, and in ALL TWELVE the multiset is the same — a pure REORDERING of acts
         0 / 96   any change to the per-(op, operand) slot multiset

    So the slot VALUES are invariant, exactly as the paper's Theorem 6 says: the trip-1 level emits
    its body once and contributes a radix-1 digit, so every flat index is arithmetically unchanged
    (the LDS slot EXPRESSION grows terms like `M_split*2`, but each such mode is pinned at 0).
    What moves is the emission ORDER — the interleave of the operands' reads, and where the §5.1
    first-touch guards materialize, because a kept extent-1 mode becomes a real enclosing level for
    the invariance test.  An earlier version of this comment claimed register slots were rewritten;
    they are not, and that claim was never measured.  The residual gap between "does not affect the
    tree" and "reorders 12 of 96" is Q34.
```

## `Tensile/LoopModel/adapter/loop_order.py.inner_ord_of` (function)

```
The INNER (intra-chunk) modes of an ord list — those with a concrete (>0) extent.  A tiny
    translate-local helper mirroring Theta.inner_modes (translate builds `ord` before it has a
    Theta to call).
```

## `Tensile/LoopModel/adapter/loop_order.py:1` (comment)

```
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
```

## `Tensile/LoopModel/adapter/loop_order.py:15` (comment)

```
# The canonical 6-mode vocabulary + LoopOrder word encoding (§A/§D).
# ===========================================================================
```

## `Tensile/LoopModel/adapter/loop_order.py:22` (comment)

```
#: TILE axis -> its STORAGE-REGION axis.  An extent-1 inner mode survives ONLY when its paired
#: split mode is live — see `_canonical_ord`.
```

## `Tensile/LoopModel/adapter/loop_order.py:123` (comment)

```
# THE K REGION AXIS IS THREE MODES, NOT ONE: `K_split` (the GCD of the operands' DU factors,
# SHARED) times `K_splitA` and `K_splitB` (each operand's own residual, `own // gcd`).  (ADR 0007 #18)
```

## `Tensile/LoopModel/adapter/loop_order.py:133` (comment)

```
# canonical Mode NAMES kept generic-but-role-tagged so geometry's role test still works:
# the physical mode name is the role name; extents drive m/n/k classification via broadcast.
```

## `Tensile/LoopModel/adapter/loop_order.py:144` (comment)

```
# `_word_to_modes` speaks the 6-letter K/M/N word, so the K region factorization is expanded
# HERE: wherever the word places K's region axis, emit the shared link then the two residuals.  (ADR 0007 #17)
```

## `Tensile/LoopModel/adapter/loop_order.py:151` (comment)

```
# SET the OUTER DepthU reduction level `iter` (extent=0 → dynamic/runtime trip = K//DU), as the
# OUTERMOST entry of `ord` (§2.4 point 1, §5.3.1 `levels = [iter] + [substep, m, n, …]`).  (ADR 0007 #16)
```

## `Tensile/LoopModel/adapter/solution.py.kernel_to_params` (function)

```
Translate a TensileLite Solution dict into the decoder's TensileLite-param dict.

    Only the keys the decoder addon reads are set; the addon fills the rest from its own
    DEFAULTS.  Keys that exist verbatim on a real kernel are copied; the few that differ
    (MIWaveTile from MIWaveTileA/B, ElemBytes from the DataType) are derived here.

    `target` — facts that are NOT in the Solution because they are the TARGET's, resolved on the
    writer (instruction selection, derived `lrvw*` state).  §2.2 makes this a real distinction
    rather than a plumbing wart: a hop's MOVEMENT QUANTUM is a θ field, but the *domain* it is
    chosen from is what the ISA offers — exactly as `S` is θ's while the register budget bounding
    it is the target's.  So the caller that holds the writer passes those widths in; a caller that
    does not (a unit test, the `OutputLoopIR` dump) omits them and every hop keeps the safe
    one-tile quantum, which is a narrower load and the same schedule.  Recognized keys:

        ReadVectorElems : {operand: elements one shared->register instruction moves per lane}
        RegisterBudget  : the VGPR ceiling theta's register partition is solved against (#318).
                          Same standing as the rest of `target`: the paper's own words are that
                          "`S` is theta's while the REGISTER BUDGET bounding it is the target's",
                          so it is supplied by the caller that holds the writer and omitted by one
                          that does not (a unit test, the `OutputLoopIR` dump) -- and omitted it is
                          a no-op, leaving every fan unpartitioned.
    
```

## `Tensile/LoopModel/adapter/solution.py._read_quantum` (function)

```
{operand: ((axis, factor), ...)} — which tile coordinates ONE read instruction covers.

    THIS IS THE OUTSIDE COMPUTATION, and it belongs outside on purpose.  Which tiles a single
    instruction covers is a consequence of the LDS ADDRESS LAYOUT and the emitter's folding
    behaviour; θ is address-opaque (§2.1) and cannot see either.  So the layout+fold is resolved
    here and crosses into θ already translated into GEMM semantics — *which tile coordinates pair*
    — exactly as `S` is a per-placement map computed outside and reasoned over inside (§2.6).

    Once inside, θ's use of it is purely structural: the named axis is ABSENT from the read hop's
    presence, so the paired coordinates are ONE presence point and therefore one step and one
    generation (§2.2).  The pair is inseparable by construction — `ord` orders modes, and no
    reordering can get inside a single presence point — so there is nothing for θ to verify.

    EMPTY IS ALWAYS ADMISSIBLE and is the default: one tile per instruction, the narrow form.  A
    caller with no target facts (a unit test, the `OutputLoopIR` dump) gets it and emits a correct,
    narrower kernel.  `target["ReadQuantum"]` overrides for a caller that has resolved the fold
    itself (the writer, which knows `tilePerRead` and the TileSpan layout).

    NOT YET DERIVED HERE.  The scale read's fold is `tilePerRead = blockWidth*BPR // mxUnit` tiles,
    and expressing WHICH tile coordinates that pairs needs the operand's mode factorization, which
    is θ's side of the boundary — so it is passed in rather than guessed.  Deriving it from the
    Solution alone is the next step and must not be faked: four attempts to have θ infer the
    pairing (scan the free modes, test generations, refuse on a region crossing, `tile`-split to
    create the pair) were each an address assumption in disguise.
```

## `Tensile/LoopModel/adapter/solution.py._read_vector_elems` (function)

```
{operand: elements one shared→register instruction moves per lane} — §2.2's quantum domain.

    ONLY the SCALE operands are named.  A data operand's read moves `LocalReadVectorWidth`, which
    `translate` already has; a scale's moves its own width, and inheriting the data one is what made
    the scale read's instruction width unknowable, which is the DOMAIN the outside computation
    needs when it resolves the fold into a quantum.

    WHY THIS READS THE SOLUTION AND NOT THE WRITER.  The authoritative number is
    `writer.states.lrvwUnrollMXSA`, but that is assigned in `_initKernel` *after* the first
    `_loopModelTheta` call (`loopModelRegBuffers`, KernelWriter.py ~7498, vs ~7620) — and θ is built
    ONCE and cached for both consumers, so reading it there would silently give the vgpr sizing a
    different θ than the emitter gets.  The two Solution keys below are what that assignment itself
    reads, and they are DERIVED onto the kernel dict long before codegen
    (`Solution.py` ~2917/2936), so this is the same fact one step earlier in the chain, not a second
    derivation of it.  `target` overrides when a caller does hold the resolved width.

    THE ROUNDING IS NOT MODELLED, DELIBERATELY.  `selectMemoryInstruction` may round the width DOWN
    to an available `ds_load`, which would leave θ authorizing a wider quantum than L3 emits — the
    unsafe direction.  That is backstopped at the emit boundary rather than guessed at here: the
    §2.2 enforcement check (`Lowering/gir/quantum.py`) compares what L3 merges against the span θ
    declared and refuses a widening, so a rounding surprise is a loud failure, never a silent one.
```

## `Tensile/LoopModel/adapter/solution.py.kernel_to_ir_text` (function)

```
Return the rolled θ-IR (render_stream) for `kernel`, plus the θ translation notes and
    the ledger status.  On ANY failure returns a `# LoopModel IR: <reason>` comment so the
    caller can write it without risk to codegen.
```

## `Tensile/LoopModel/adapter/solution.py:1` (comment)

```
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
```

## `Tensile/LoopModel/adapter/solution.py:15` (comment)

```
# The decoder's render_* helpers use a few unicode glyphs (theta, Sigma, times, middle
# dot, subscripts).  Transliterate to ASCII so the dumped .txt is a plain text file
# (unicode makes `file` report "binary" and trips some editors/diff tools).
```

## `Tensile/LoopModel/adapter/solution.py:46` (comment)

```
# LDS double-buffer count — the REAL kernel-derived value (2=double, 1=single/in-place),
# not an assumption; drives in_place_shared in translate.py.
```

## `Tensile/LoopModel/adapter/solution.py:58` (comment)

```
# rho / agents (§2.4): how many agents cooperate on one tile.  Its own input, not derived
# from the fuse -- a lone operand loaded by several waves is cross-agent with no fuse
# present.  (ADR 0007 #21)
```

## `Tensile/LoopModel/adapter/solution.py:64` (comment)

```
# How many of those regions ONE WAVE occupies (#245): the copy writes every region, a
# wave reaches only the ones its own free-axis offset lands in.  (ADR 0007 #20)
```

## `Tensile/LoopModel/adapter/solution.py:75` (comment)

```
# THE MOVEMENT QUANTUM (§2.2) — computed HERE, where the layout is knowable, and handed to
# θ in GEMM terms.  See `_read_quantum`.
```

## `Tensile/LoopModel/adapter/solution.py:78` (comment)

```
# §2.2's carrier group as Φ and ρ — the two numbers θ derives the `QuantumMap` from
# (`geometry.derive_quantum`).  Pass-through like `ReadVectorElems`: they are TARGET facts
# (how many instances one instruction merges, and the sub-agent span they spread over),
```

## `Tensile/LoopModel/adapter/solution.py:111` (comment)

```
# MIWaveTile is stored per-operand (MIWaveTileA/B) on a derived Solution; the decoder
# wants the [WT0, WT1] pair.
```

## `Tensile/LoopModel/adapter/solution.py:183` (comment)

```
# `lrvwUnrollMXS*`: the scale's own unroll-major read width, else 1 element (the
# tile-major branch reads one scale per instruction).
```

## `Tensile/LoopModel/adapter/target.py:1` (comment)

```
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
```

## `Tensile/LoopModel/emit/__init__.py <module>` (module)

```
emit_can (paper §5.3) — the θ → rolled region-tree IR driver.

Clean-architecture split of the decoder: this module is the ORCHESTRATOR (emit_can steps
8/9a/9b).  It builds the rolled θ-nest (`build_ir`, step 8 + 9a with σ_c), attaches the named
`Await` discharge points (step 9b), and packages the result (`emit_mainloop`).  It
consumes the pure derivations split into sibling modules:
  - ledger.py     — build_ledger / check_ledger_discharged / walk_insts (the §4 ledger)
  - placement.py  — presence / rate-slot / _read_placement / peel_depths (§2.1/§2.6)
  - sigma_c.py    — move_reloads_after_last_use (the §5.3.1-pt4 refill-after-read freeze)

The numeric wait COUNT is deliberately NOT lowered here.  §5.4 admits two sound backends; this
decoder targets the second — a COUNT-INSERTING backend, whose own counter-insertion pass re-derives
every count from the emitted dataflow, so a count resolved here would only be overwritten.  Per
§5.2 the `Await.count` slot is therefore left UNSET (-1) in the built IR and the named dependency
is the whole output; Proposition 7 guarantees the named form covers every legal σ.  The
counting-rule lowering a raw / counter-based backend would need is deferred work — it lived here
until 2026-08 and was removed because nothing consumed it and it had silently gone stale against
the §5.3.1 tree restructure.

No TensileLite / AMD vocabulary; stdlib + LoopModel core only.
```

## `Tensile/LoopModel/emit/__init__.py.build_ir` (function)

```
The rolled theta-nest IR (paper 5.2/5.3): one Loop per ord inner mode, one Inst per
    op-class at its presence level, slots symbolic. Nothing is enumerated -- the renderer does
    the partial unroll. See ADR 0005.
```

## `Tensile/LoopModel/emit/__init__.py.loop_shape` (function)

```
Decide, from θ alone, whether the ord yields ONE uniform steady body (cleanly
    loopable) or needs a CONDITION-BRANCH / multi-body loop path.

    See ADR 0005.
    
```

## `Tensile/LoopModel/emit/__init__.py._Emitter` (class)

```
Builds the rolled theta-nest IR. One instance per (theta, S).

    The shared state every mixin reads is set up here; each mixin owns one side of the
    schedule and can be read without the other two.
    
```

## `Tensile/LoopModel/emit/__init__.py:1` (comment)

```
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
```

## `Tensile/LoopModel/emit/__init__.py:46` (comment)

```
# The issuing AGENT for every instance in a single-role kernel (§5.2: ρ → `Inst.a` and `Loop.role`;
# "an unspecialized kernel is the single-role special case"). ONE name instead of a literal
# repeated at each emit site.  (ADR 0007 #25)
```

## `Tensile/LoopModel/emit/__init__.py:70` (comment)

```
# PRESENCE-DERIVED partition key (no m/n/k literal): each inner mode's key is the SET of
# operands present on it (a reduction mode → all inputs; a free mode → its own operand subset).
# Contiguity of that key in ord is the multi-body test — general for any operand count.
```

## `Tensile/LoopModel/emit/__init__.py:91` (comment)

```
# PRECONDITION Q (§2.2). The quantum is SUPPLIED on the hop (translate <- bridge <- the
# target's layout+fold); this only surfaces it. θ does not derive it: which tiles one
# instruction covers is an address fact and θ is address-opaque (§2.1).  (ADR 0007 #24)
```

## `Tensile/LoopModel/emit/__init__.py:97` (comment)

```
# ONE obligation, ONE discharge: drop an Await a preceding instruction on the same path
# already established (ledger.discharge_once).  Runs BEFORE the gate, so the gate checks the
# boundary that is actually emitted rather than a redundant restatement of it.
```

## `Tensile/LoopModel/emit/__init__.py:101` (comment)

```
# No wait-count lowering: each Await keeps count=-1 (unset), the named dependency IS the
# output (§5.2/§5.4, count-inserting backend).  The ledger gate below therefore sees the TRUE
# Await.kind values — nothing rewrites them between build and check.
```

## `Tensile/LoopModel/emit/__init__.py:108` (comment)

```
# NAME THE θ THAT FAILED. This raise runs inside a joblib worker during a whole-matrix
# build, so the traceback alone says "a kernel" and nothing about WHICH — on the 2026-08-21
# mxf8 run that turned a one-line defect into a guess-and-rebuild loop.  (ADR 0007 #22)
```

## `Tensile/LoopModel/emit/copies.py._nest_copy_runs` (function)

```
Fold each maximal run of copies in an ordered body into one ord-ordered nest.

        EVERY consumer of `copy_insts` must call this, not just the steady one.  The region `Loop`
        used to be built inside `copy_insts`, so it came for free everywhere; moving it here (so
        σ_c can still see individual movements) means a caller that forgets it emits copies with
        NO enclosing region loop — and a copy with no region loop carries no region coordinate,
        which is how the prologue silently went from `A0 A1 B0 B1` to four coordinate-less copies.
        Nesting is idempotent-safe: an already-nested node is not in `_copy_meta`, so a second
        pass leaves it alone.
```

## `Tensile/LoopModel/emit/copies.py._site_awaits` (function)

```
Build the Await tuple for the (opname, role) consume site from its ledger obligations.
        `dep` carries the producer name (σ-invariant, §5.4); counter/kind come from the obligation.
        kiter_note, when given, suffixes the RAW dep (the generation the consumer reads); under the
        per-region π a region note is appended too (see `_region_note`).

        `groups` (default: all) restricts the REGISTER-RING WARs to the given groups: a read split
        per group (differing rotation widths) must carry only the WAR of the slot IT overwrites,
        else each half would wait on the other's vacating read.  Obligations that are not
        register-group WARs (the RAW residency, the shared ring) are carried by every split.
```

## `Tensile/LoopModel/emit/copies.py._nest_copies` (function)

```
Order the chunk's copies by the ORD, not by operand — one nest, first-touch merged.

        See ADR 0005.
        
```

## `Tensile/LoopModel/emit/copies.py:1` (comment)

```
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
```

## `Tensile/LoopModel/emit/copies.py:28` (comment)

```
# The issuing AGENT for every instance in a single-role kernel (§5.2: ρ → `Inst.a` and `Loop.role`;
# "an unspecialized kernel is the single-role special case"). ONE name instead of a literal
# repeated at each emit site.  (ADR 0007 #31)
```

## `Tensile/LoopModel/emit/copies.py:78` (comment)

```
# A Φ-fused movement's obligations are keyed by the MOVEMENT, because one cooperative
# instruction owes one thing (ledger). Every member consumes that same object, so a
# member's consume site resolves to its own per-op-class obligations PLUS the movement's.  (ADR 0007 #30)
```

## `Tensile/LoopModel/emit/copies.py:89` (comment)

```
# the dep is the producer ENDPOINT plus the generation qualifiers that pick which
# instance: the chunk the consumer reads, and (per-region π) its region.
```

## `Tensile/LoopModel/emit/copies.py:106` (comment)

```
# per-movement bytes: each member contributes its OWN per-region share (tile / its own
# split), so a region movement of a split member is tile/split, an unsplit member whole.
```

## `Tensile/LoopModel/emit/copies.py:124` (comment)

```
# `part` is the intra-tile transfer index of ONE movement (1 here — we do not model
# sub-tile transfer fan-out on the copy).  It is NOT the region: the region is `coord`,
# supplied by the enclosing region Loop.
```

## `Tensile/LoopModel/emit/copies.py:141` (comment)

```
# the buffer slot is REGION-INVARIANT (all regions of a chunk land in the same ring
# generation), so it is computed once for the rolled copy, not per region.
```

## `Tensile/LoopModel/emit/copies.py:153` (comment)

```
# drain step under Bind(iter=T−M+t): use the STEADY symbolic slot (keyed on iter,
# which Bind defines to T−M+t), so the drain shares the one residue timeline.
# deepest operand drops FIRST — copy present iff off_p ≤ M−1−t (§5.3.1 point 2).
```

## `Tensile/LoopModel/emit/copies.py:163` (comment)

```
# PROLOGUE copies fill EMPTY (not-yet-read) buffers → no vacating read → no WAR.
# STEADY/DRAIN copies refill a slot the current reduction chunk was read from → WAR
# needed.  (ADR 0007 #28)
```

## `Tensile/LoopModel/emit/copies.py:184` (comment)

```
# The COPY nests over EVERY ord level -- including the agent-served ones the read/compute
# nest dropped.  Both the sort order and the extent map need the full list.
```

## `Tensile/LoopModel/emit/copies.py:189` (comment)

```
# levels come from the ENUMERATION axes: every axis a movement is enumerated by must
# exist as an enclosing loop.  (ADR 0007 #27)
```

## `Tensile/LoopModel/emit/helpers.py._check_peel_is_emittable` (function)

```
Refuse a θ whose peel `build_ir` cannot express — instead of emitting a DIFFERENT schedule.

    See ADR 0005.
    
```

## `Tensile/LoopModel/emit/helpers.py._scope_of` (function)

```
The PROC-SCOPE the discharge of obligation `ob` must reach (§2.4, §5.2's scope ladder).

    See ADR 0005.
    
```

## `Tensile/LoopModel/emit/helpers.py._reduction_inner_mode` (function)

```
`(name, extent)` of the ord-level READ-AHEAD LEVEL, used at its one call site purely as a
    SENTINEL — "does this θ have an intra-region axis at all", i.e. can any read-ahead exist.

    IT IS NOT THE AXIS THE DRAIN'S NLL GUARD TESTS, despite what this used to say.  That guard is
    `Pred(Expr(position_terms(...), add=shift) < n_pres)` and is built entirely from the OPERAND's own
    `_readahead_shift` — flat presence positions, per op-class — because the crossing bound is the
    advance's own bound (see the comment at the call site).  Since #321 each op-class also has its
    OWN level, so no single axis could serve here even in principle; the ord-level answer is
    sufficient for a "is there one" test and nothing more is read from it.
```

## `Tensile/LoopModel/emit/helpers.py._read_hop_quantum` (function)

```
The `ir.QuantumMap` of `op`'s shared->register read, or None for the identity merge.

    Passed through from the hop verbatim.  θ does not build it: the merge follows from the address
    layout and the emitter's fold, both invisible here (§2.1), so it is supplied by the bridge and
    only evaluated downstream (`QuantumMap.group` for the coordinates one instruction serves,
    `.regs` for the registers it writes).
```

## `Tensile/LoopModel/emit/helpers.py._resolve_ft` (function)

```
Rewrite a subtree, RESOLVING every first-touch guard on `mode` to a known outcome:
    `taken=True` splices the guard's body in (this is the mode's first iteration), `taken=False`
    drops it.  Everything else is copied structurally.

    The guard sits at the READ's level, which is INNER to the loop it tests, so resolving it
    rewrites the whole subtree below that loop — that is exactly why the two sub-bodies differ
    and why this cannot be a local edit at the guard site.
```

## `Tensile/LoopModel/emit/helpers.py._peel_first_touch` (function)

```
§5.1 multi-body: HOIST first-touch guards on THIS loop's mode into the loop STRUCTURE.

    A read invariant over `mode` is issued once, under `Cond(mode == 0)` placed at the READ's
    level (inner to this loop).  Splitting the trip at that boundary gives two sub-bodies over
    disjoint ranges — `[0,1)` with the read, `[1,trip)` without — and the predicate disappears:
    "the body before the boundary differs from the body after", with the guard in the structure
    rather than evaluated every iteration.

    Skipped when the trip is symbolic (cannot split statically) or 1 (no "rest" half); the
    guard then stays, which is correct, just costlier.
```

## `Tensile/LoopModel/emit/helpers.py._region_note` (function)

```
Under the per-region π (`theta.per_region_completion`), the region mode whose value
    identifies WHICH movement instance a consumer awaits — else None.

    §5.2 line 368 lets an `Await`'s `X` name "a producer INSTANCE or op-class-generation", and
    §4.3 says a region split yields "two fused movements with TWO COMPLETIONS", so under this π
    a read of region j must name region j and leave the others in flight.  The read is ROLLED
    at its presence level, so the note is the region MODE NAME and resolves to a value when the
    tree is unrolled.  The LEDGER entry stays symbolic per-op-class (§4.1, O(1) safety check) —
    only the emitted discharge point is instance-named.
```

## `Tensile/LoopModel/emit/helpers.py._region_mode_of` (function)

```
The inner mode that ENUMERATES this fused group's storage regions, or None if the group
    is unsplit.  Taken from the group's representative member; §2.8 move 9 allows members to be
    tiled on DIFFERENT region axes (the pairing is by index, not by a shared mode), so a
    heterogeneous group labels by one axis — tracked in #169(c).

    See ADR 0005.
    
```

## `Tensile/LoopModel/emit/helpers.py._enum_axes` (function)

```
The axes that ENUMERATE this movement's regions — the REPRESENTATIVE member's own, not
    the union over members.

    See ADR 0005.
    
```

## `Tensile/LoopModel/emit/helpers.py._group_region_modes` (function)

```
EVERY region axis this movement spans — the union over its members, not one member's.

    See ADR 0005.
    
```

## `Tensile/LoopModel/emit/helpers.py._group_guard` (function)

```
`Cond` predicates restricting a group-restricted read to the coordinates it OWNS.

    See ADR 0005.
    
```

## `Tensile/LoopModel/emit/helpers.py._axis_stride` (function)

```
How many values of axis `g` this movement skips between issues.

    `nreg` is the movement's TOTAL region count and `spans` the axes it is split on, so its
    count on `g` alone is `nreg` divided by the other spanned axes' extents.  A movement not
    spanning `g` at all has count 1 and therefore stride `ext[g]` — the first-touch case.

    REFUSES an ambiguous deficit: if the movement is coarser than the axis on MORE than one of
    its spanned axes, `nreg` does not say which axis lost the factor, and picking one is the
    silent-wrong-answer class.  No current shape does this (an operand is full on all but at
    most one axis), so this is a guard, not a branch anyone takes.
```

## `Tensile/LoopModel/emit/helpers.py._bulk_hop` (function)

```
`op`'s BULK (cooperative, whole-tile) hop, or None if it has none.

    The copy hop, identified by its MOVER KIND rather than by position: a DTV operand's
    `hops[0]` is `global→register`, a per-lane READ, and asking it a copy-placement question
    gets a confident wrong answer (it names a reduction axis where a region axis belongs).
```

## `Tensile/LoopModel/emit/helpers.py:1` (comment)

```
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
```

## `Tensile/LoopModel/emit/helpers.py:25` (comment)

```
# The issuing AGENT for every instance in a single-role kernel (§5.2: ρ → `Inst.a` and `Loop.role`;
# "an unspecialized kernel is the single-role special case"). ONE name instead of a literal
# repeated at each emit site.  (ADR 0007 #33)
```

## `Tensile/LoopModel/emit/helpers.py:185` (comment)

```
# the COPY HOP's presence set names the region axis, and is empty when the tile is not
# split (§5.3.1 line 583) — so the `op.split <= 1` test and the region_modes scan are the
# same one question, asked of the hop instead of reconstructed from two fields.
```

## `Tensile/LoopModel/emit/helpers.py:290` (comment)

```
# a *-WAR's consumer is the REFILLER: the shared ring is refilled by the copy, a register
# group by the read itself.
```

## `Tensile/LoopModel/emit/helpers.py:305` (comment)

```
# "reads" = every INPUT operand that ends in a register fragment (two-hop shared→reg AND one-
# hop DTV global→reg); each is emitted at its presence level. A DTV operand has no separate
# copy (its single hop lands in registers) and awaits on the VMEM (load) counter.  (ADR 0007 #32)
```

## `Tensile/LoopModel/emit/helpers.py:309` (comment)

```
# a "scale" operand = an INPUT read beyond the two matmul inputs (an extra elementwise/
# broadcast operand, §7.1) — presence-derived, no "MX" name test.
```

## `Tensile/LoopModel/emit/nest.py.build_level` (function)

```
Rolled region node for level `li..`.  Body = reads whose read_level is this mode
        (hoisted above their inner fan), then the child level (or the leaf wmma).

        `read_shift` overrides the read hop's retime depth for this nest; `None` = θ's `dr`.  The
        one caller that overrides is the DEGENERATE (short-loop) tree, which passes 0 — Lemma 1:
        a fully-peeled level resets to 0 every `off` at levels INNER to it.
```

## `Tensile/LoopModel/emit/nest.py._drain_step` (function)

```
One drain step (paper §5.3.1 `i = T−M+t`): the ROLLED read/wmma nest (reduction stays a
        `for k` Loop) plus the staggered refill copies, ALL in the STEADY symbolic form (slots keyed
        on `iter`), wrapped in `Bind(iter = T−M+t)` so every `iter`-symbolic slot/coord resolves to
        this drain chunk's DEFINED value.  `suppress_ahead` guards the OOB read-ahead on the NLL steps.

        This replaces the old free-`iter` `Cond(iter%d==residue)` pin — that pin only existed
        because `iter` was undefined outside the steady loop.  `Bind` DEFINES `iter` for the step
        (the continuous chunk-issue timeline, one residue chain prologue→steady→drain), so the drain
        copies/reads use the SAME steady `Expr(var=iter, …)` placement — no special `T−Δ` slot.
```

## `Tensile/LoopModel/emit/nest.py:1` (comment)

```
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
```

## `Tensile/LoopModel/emit/nest.py:27` (comment)

```
# The issuing AGENT for every instance in a single-role kernel (§5.2: ρ → `Inst.a` and `Loop.role`;
# "an unspecialized kernel is the single-role special case"). ONE name instead of a literal
# repeated at each emit site.  (ADR 0007 #38)
```

## `Tensile/LoopModel/emit/nest.py:43` (comment)

```
# AWAITS from the ledger (step 9b): the wmma consumes each read operand's REGISTER residency
# (X@register RAW).  One Await per read operand, derived from its obligation (dscnt/vmem).
```

## `Tensile/LoopModel/emit/nest.py:48` (comment)

```
# REGISTER PLACEMENT the wmma CONSUMES, PER READ OPERAND — the name-determining tile coord
# + rotation the source vgpr carries (§2.6). The wmma consumes the UN-shifted rotation slot
# (`_read_placement` with dr=0): the value a PRIOR iteration's read-ahead deposited.  (ADR 0007 #37)
```

## `Tensile/LoopModel/emit/nest.py:69` (comment)

```
# NLL suppression (drain last steps): a read that VARIES over the reduction substep
# carries the read-ahead-shifted coord `(k+dr)%n_s`; on the AHEAD substeps (k ≥ n_s−dr)
# that read crosses into the OUT-OF-BOUNDS next reduction chunk.  (ADR 0007 #36)
```

## `Tensile/LoopModel/emit/nest.py:77` (comment)

```
# register-group ownership: this read fills ONE group, so it belongs only at the
# grouping mode's values that group owns (#118).  Inside the readahead guard and
# outside first-touch, matching how those two already nest.
```

## `Tensile/LoopModel/emit/nest.py:82` (comment)

```
# first-touch: wrap in `mode == <anchor>` for each enclosing invariant mode (outermost
# first, so the outermost guard is the outermost node).  (ADR 0007 #35)
```

## `Tensile/LoopModel/emit/nest.py:92` (comment)

```
# The wmma sits at `leaf_mode` -- the deepest level the accumulator varies over -- and the
# nest continues below it for any level the output does not traverse.
```

## `Tensile/LoopModel/emit/nest.py:109` (comment)

```
# HOISTED ABOVE THE WHOLE INNER NEST, so the grouping mode is not a loop variable
# here and `_group_guard`'s predicate would reference a name that is not in scope.  (ADR 0007 #34)
```

## `Tensile/LoopModel/emit/nest.py:132` (comment)

```
# deepest operand drops FIRST: a copy is present iff off_p ≤ M−1−t.  The steady copy_insts
# emits every copy; drop the drained ones by off (§5.3.1 point 2, no OOB copy_p(T)).
```

## `Tensile/LoopModel/emit/reads.py._read_groups` (function)

```
How many INSTRUCTIONS this operand's read is split into, and which register groups each
        one fills.  Returns a tuple of group-tuples (a single `None` entry = the unsplit read).

        See ADR 0005.
        
```

## `Tensile/LoopModel/emit/reads.py.read_inst` (function)

```
`shift` = the PLR read-ahead applied to THIS read's reduction coord (default dr).  The
        prologue's DIRECT fill passes shift=0 (unshifted: it reads exactly substeps [0,dr)).
        `groups` restricts the read to a subset of the operand's register groups (a split read —
        see `_read_groups`); the coord and size are the fragment's either way.
```

## `Tensile/LoopModel/emit/reads.py._first_touch_modes` (function)

```
Enclosing inner modes the op is INVARIANT over — non-presence modes at an ord position
        OUTER to the op's read level.  Each becomes a first-touch `mode == 0` guard.

        See ADR 0005.
        
```

## `Tensile/LoopModel/emit/reads.py._read_anchor` (function)

```
`{invariant mode: value}` at which `op`'s read-ahead must fire — the ANCHOR (#331).

        See ADR 0005.
        
```

## `Tensile/LoopModel/emit/reads.py._fill_read_inst` (function)

```
One DIRECT (unshifted) prologue read of `op` at the concrete presence-coord `coord_dict`
        (a Lemma-3d prologue member).  Every presence mode is PINNED to its concrete value — no
        read-ahead shift (the prologue pre-issues the true data), no rolled fan.

        The placement slot is EVALUATED against the pinned coord (`Placement.at`) so the register
        buffer + shared src are CONCRETE ints, not the unbound symbolic Expr `buf(K_inner)%2` — a
        prologue read at a fixed coord must show a resolvable buffer + its register group.
```

## `Tensile/LoopModel/emit/reads.py.fill_reads` (function)

```
PROLOGUE read-ahead fill (Lemma 3d, §5.3.1 pt5): pre-issue exactly the first-trip reads
        whose residency `O_r` the steady read-ahead cannot discharge in-body — the closed-form
        three-factor `preloaded_tiles` set, PER read op-class (per-operand-presence, so a
        broadcast operand and its partner may differ in arity).  DIRECT (unshifted) into the
        register gens the steady body's first trip consumes; no wmma.
```

## `Tensile/LoopModel/emit/reads.py:1` (comment)

```
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
```

## `Tensile/LoopModel/emit/reads.py:27` (comment)

```
# The issuing AGENT for every instance in a single-role kernel (§5.2: ρ → `Inst.a` and `Loop.role`;
# "an unspecialized kernel is the single-role special case"). ONE name instead of a literal
# repeated at each emit site.  (ADR 0007 #45)
```

## `Tensile/LoopModel/emit/reads.py:62` (comment)

```
# The read's awaits come from the ledger at the site (op, "read"): its RAW residency on
# whatever filled the buffer, plus any in-place refill WAR.  (ADR 0007 #44)
```

## `Tensile/LoopModel/emit/reads.py:67` (comment)

```
# The fold converts the shifted PRESENCE POINT back to the TILE that names its carrier
# group (§2.2); `read_fold` is the one derivation, shared with `_first_touch_modes`.
```

## `Tensile/LoopModel/emit/reads.py:73` (comment)

```
# `advance=shift` records the §5.3.1 pt5 SHAPE on the node: 0 = INPLACE (in place, one line
# before its own wmma), non-zero = PREFETCH (hoisted).  σ_c defers only a hoisted refill.
```

## `Tensile/LoopModel/emit/reads.py:91` (comment)

```
# A PARTIALLY FOLDED axis is at or inner to the read level (it is still in presence), so it
# is not in the slice above; it is added here with its modulus.  (ADR 0007 #42)
```

## `Tensile/LoopModel/emit/reads.py:132` (comment)

```
# PROLOGUE fill: the registers are EMPTY (nothing has read them yet), so there is no
# vacating read and the rotation/in-place WAR does not apply — the same reasoning
# `_copy_inst(war=False)` already applies to the prologue ramp copies.  (ADR 0007 #40)
```

## `Tensile/LoopModel/emit/reads.py:150` (comment)

```
# `preloaded_tiles` derives the per-group `dr_g` itself (§5.3.1 pt5: prologue depth =
# steady advance = dr_g), so an INPLACE group fills NOTHING here and its steady advance
# is 0 — one shape, consistently.  (ADR 0007 #39)
```

## `Tensile/LoopModel/fuse.py <module>` (module)

```
The Φ fuse GROUPING — which operands share one aliased TDM descriptor.

ONE TABLE, read by everything that needs it.  Before this module the grouping lived in
`translate.py` while the scaffold spelled its own pairs out in three places
(`KernelWriterAssembly.defineTdmSgprs`'s RegSet chains, `KernelWriter`'s
`initTDMDescriptorWaveSeparated` calls, `gir_to_rocisa._fused_tp`'s literal `pairs` list).  Four
derivations of one fact is the exact shape of the recurring bug in this subsystem: they agree on
the shipping case and diverge on the new one.

WHAT A GROUP MEANS PHYSICALLY.  `KernelWriterAssembly.defineTdmSgprs` gives ONE member of the group
real descriptor SGPRs (`tdm<tc>Group0` = 4, `Group1` = 8) and `RegSet`s the others onto it, so the
group is one physical descriptor.  `initTDMDescriptorWaveSeparated` then fills it per wave, the
wave index selecting which member THIS wave moves.  The aliasing IS the fuse (see
`ValidParameters`' TDMFuse note): with one physical descriptor there is only one thing to advance.

TWO WAVE SELECTORS.  The shipping groupings (0/1/4) use PARITY — `SBitcmp1B32(waveIdx, 0)` picks
the member and `wCompId = waveIdx >> 1` picks the slice over `NumWaves // 2` cooperating waves,
which is why that path carries a power-of-two assert.  The MX groupings (2/3/5) use CONTIGUOUS
RANGES from `FUSE_WAVE_SHARES`, so a group can partition UNEVENLY — `A_MX` gives A two waves and
each scale one, because A is ~32x the bytes of a scale and an equal split would idle half the
waves.  `KernelWriterAssembly.tdmWaveSelect` emits both; the range form needs no power-of-two.
```

## `Tensile/LoopModel/fuse.py.wave_ranges` (function)

```
`[(member_index, first_wave, count)]` for one group, or None for the parity rule.

    Contiguous, in `FUSE_GROUPS` member order, sized by `FUSE_WAVE_SHARES`.  Returns None when the
    group uses the historical parity selector so the caller emits that instead of a range compare.

    Raises when the shares do not partition `num_waves` exactly — an uneven partition would leave
    a wave with no member (or two), and silently rounding is how a wave ends up moving nothing.
    
```

## `Tensile/LoopModel/fuse.py.fuse_groups` (function)

```
The Φ groups for `fuse`, restricted to operands actually `present`.

    `present` is any container of operand names.  A group that loses members to the restriction
    and is left with fewer than two is not a fuse at all and is dropped — so a non-MX kernel at
    `TDMFuse=1` yields just `[('A','B')]`, exactly as before this table moved.
    
```

## `Tensile/LoopModel/fuse.py.needs_mx` (function)

```
Is `fuse` MEANINGLESS without MX scales — i.e. does it lose every group without them?

    Asked by RESTRICTING the table, not by testing membership.  `1` names `MXSA`/`MXSB` and still
    works perfectly on a bf16 kernel, because its `A`/`B` group survives the restriction; a
    membership test would wrongly call it MX-only and reject every bf16 fuse.  `2`, `3` and `5`
    genuinely collapse to nothing (their surviving members are singletons, which is not a fuse).
    
```

## `Tensile/LoopModel/fuse.py.crosses_data_and_scale` (function)

```
Does any group mix a DATA operand with a SCALE one?

    True for `paired` (5) and for `A_MX`/`B_MX` (2, 3).  Such a group cannot reuse the scaffold's
    historical A/B and MXSA/MXSB descriptor pairing — it re-points which operands alias onto which
    — so it is the flag the SGPR allocation keys on.
    
```

## `Tensile/LoopModel/fuse.py:1` (comment)

```
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
```

## `Tensile/LoopModel/fuse.py:27` (comment)

```
# operand names are θ op-class names, which coincide with the scaffold's `tensorChar`
# ('A', 'B', 'MXSA', 'MXSB') — the two vocabularies agree here, so no translation is needed.
```

## `Tensile/LoopModel/fuse.py:36` (comment)

```
# 6 (`all4`) — DISCRIMINATOR, not a shipping grouping. Every operand in ONE group at one wave
# each, so it is the only point with a 3+-member group and NO unfused operand.  (ADR 0007 #47)
```

## `Tensile/LoopModel/geometry.py <module>` (module)

```
Geometry — the ONE module that owns the layout arithmetic.

Everything numeric the decoder used to FIT by hand — the wmma grid, the per-region block
size, a fragment's register width, and which axes an operand is broadcast (constant) over —
is DERIVED here: tile-counts, a broadcast (stride-0) test, and a register-packing count.

This module uses `latalg` (plain integer layout arithmetic), NOT PyCuTe.  The loop-schedule
core is address-free (the schedule-model paper's central claim), so the only "layout" facts
needed are integer tile-counts / broadcast / packing — a dozen lines of arithmetic, no
spatial-layout library.  (In the decoder_proto repo a `latalg.selftest_vs_pycute()`
differential test confirmed each op matched PyCuTe against the vendored `Cute/` oracle; that
oracle is not shipped in TensileLite, so the self-test is dropped in this port.)  The rest of
the core (`theta.py`/`geometry.py`/`decoder.py`) imports only the plain-int/plain-set results
below and never sees a layout object.

Inputs are STRUCTURAL θ facts, not fitted numbers:
  * ord mode extents          — the tile counts (the tile extent ÷ the compute-atom shape).
  * which modes an operand indexes vs is constant over — inherent to being A[m,k] / B[n,k]
    (carried as the fragment broadcast_axes; the numeric consequences are derived).
  * a fragment's element footprint (frag_elems, elem_bytes) + reg_bytes — the packing.
  * the operand's rate-group partition count — the authored MT-half split (from PLR).
```

## `Tensile/LoopModel/geometry.py.axis_tiles` (function)

```
Number of compute-atom tiles along one axis = size of logical_divide(tile, atom) in
    ELEMENT space.  This is how a Mode extent should be OBTAINED (tile_axis ÷ atom_axis)
    rather than typed by hand: e.g. wave M = 64 elems, MI_M = 16 → 4 M-tiles; N = 128 / 16
    → 8; reduction-substep = 256 / 128 → 2.  The quotient's outer extent (n_tiles) is the
    per-wave tile count feeding ord.
```

## `Tensile/LoopModel/geometry.py.broadcast_axes` (function)

```
The intra-iteration loop axes this operand is CONSTANT over (does NOT depend on).  A[m,k] is
    constant over N, B[n,k] over M — read straight off the fragment's `broadcast_axes`, kept to
    the axes actually present in this ord.  The model's structural primitive (§2.1): it drives
    presence, reload rate, and the live-range.  One source of truth for `theta.presence` and the
    read placement.  (Formerly called `nullspace` — the stride-0 axes of the data map.)
```

## `Tensile/LoopModel/geometry.py._quantum_candidate_axes` (function)

```
`[(name, extent), ...]` in ord order (outer→inner) — the axes a per-lane hop's quantum may
    fold, with the flat-index strides they carry.

    NON-CIRCULAR ON PURPOSE.  The candidates come from `broadcast_axes` (a DATA fact) and the
    reduction set, never from `presence_modes` — `hop_broadcast` is what presence is built from, so
    consulting presence here would close a loop.
```

## `Tensile/LoopModel/geometry.py.derive_quantum` (function)

```
§2.2's carrier group as `Φ` applied to `ρ` — the `ir.QuantumMap` DERIVED, not supplied.

    See ADR 0005.
    
```

## `Tensile/LoopModel/geometry.py.hop_quantum` (function)

```
This hop's `QuantumMap`: the explicit override if one was supplied, else `Φ∘ρ` derived.

    The override exists for a target whose fold is not expressible as (Φ, ρ); using it is a DECLARED
    deviation from §2.2, not the default path.
```

## `Tensile/LoopModel/geometry.py.quantum_factors` (function)

```
`{axis: factor}` — how many CONSECUTIVE coordinates of each axis one instruction of a
    per-lane hop carries (§2.2, Precondition Q).  The full derivation; `quantum_axes` is its
    fully-spanned view.

    See ADR 0005.
    
```

## `Tensile/LoopModel/geometry.py.quantum_axes` (function)

```
The intra-iteration axes ONE INSTRUCTION of a per-lane hop spans WHOLE (§2.2).

    The `factor == extent` view of `quantum_factors` — the degenerate `q = N` case in which the axis
    leaves the hop's presence entirely.  A PARTIALLY folded axis is deliberately absent from this
    set: it keeps `extent // factor` group indices in presence, so subtracting it here would drop
    coordinates that still vary.  Callers that need the partial case read `quantum_factors`.
```

## `Tensile/LoopModel/geometry.py.hop_broadcast` (function)

```
The intra-iteration axes ONE INSTANCE OF THIS HOP does not vary over — presence PER HOP
    (§2.1 line 71, §2.8 move 1 "applies per hop independently", §5.3.1 line 583).

    See ADR 0005.
    
```

## `Tensile/LoopModel/geometry.py.hop_fold` (function)

```
`{axis: factor}` for the axes this hop's quantum folds only PARTIALLY (`1 < factor < extent`).

    See ADR 0005.
    
```

## `Tensile/LoopModel/geometry.py.free_tiles` (function)

```
Tiles the operand fans over = product of extents of its OWN free presence modes
    (`theta.free_modes(op)` = presence \ reduction, §2.6).  Presence-derived — works for any
    operand (inputs, scales, output) with no name/role test.  Reads each Mode's own extent — no
    name→Mode re-lookup (that scanner existed only because the presence family was string-only).
```

## `Tensile/LoopModel/geometry.py.k_tiles` (function)

```
Reduction substeps per kiter = product of extents of the REDUCTION modes
    (`theta.reduction_modes()` = inner \ pres(output), §2.1 line 82).  No 'k'-role literal.

    A THETA-LEVEL quantity: how many reduction substeps the kiter has.  It is NOT how many an
    OPERAND traverses — an op-class broadcast over the reduction traverses none.  Use
    `presence_tiles` for a per-operand count; multiplying an operand's footprint by this is the
    over-count #212 measured (C = 4 x 2 x 16 = 128 registers for a tile that holds 4 x 16 = 64).
```

## `Tensile/LoopModel/geometry.py.reduction_tiles` (function)

```
Reduction substeps THIS operand traverses = product of extents of its presence modes that
    are reduction modes.  `k_tiles` for an operand present on the reduction (A, B); `1` for one
    broadcast over it (the accumulator).  The peer of `free_tiles` — together they partition the
    operand's presence, which is what `presence_tiles` multiplies out.
```

## `Tensile/LoopModel/geometry.py.presence_tiles` (function)

```
Distinct tiles of `op` one kiter touches = product of extents of its WHOLE presence set
    (§2.1).  `free_tiles(op) * reduction_tiles(op)` by construction, since presence partitions into
    free and reduction modes.  This is the presence-derived replacement for `free_tiles * k_tiles`,
    which silently assumed every operand is present on the reduction (#212).
```

## `Tensile/LoopModel/geometry.py.wmma_grid` (function)

```
(free-tiles of input-0, free-tiles of input-1, k_tiles) for one kiter — the matmul grid
    from the two matrix INPUT op-classes (§7.1 "matmul-primitive": two register fragments in).
    Inputs are identified by role=='input' with a hop, in θ order — no 'A'/'B' literal.  A
    kernel with a different input count is out of the matmul-primitive scope (§7.1).
```

## `Tensile/LoopModel/geometry.py.frag_regs` (function)

```
Registers one fragment occupies per lane, DERIVED by recasting the per-lane element
    layout into register-sized groups and taking its coshape.  frag_elems elements of
    elem_bytes each, packed into reg_bytes registers → recast scale = reg_bytes/elem_bytes
    elements per register.  e.g. 64 fp8 (1B) into 4B regs → scale 4 → 16 regs; 16 fp16
    (2B) → scale 2 → 8 regs.  Falls back to a byte ceiling when elem_bytes ∤ reg_bytes.
```

## `Tensile/LoopModel/geometry.py.readahead_level_of` (function)

```
`(level_name, level_extent, region_span)` — the ONE derivation of the level a local
    read-ahead lives on, from an `ord`'s inner modes, the region-mode names, and (for a
    per-operand answer) the op-class's broadcast axes.

    See ADR 0005.
    
```

## `Tensile/LoopModel/geometry.py.readahead_level` (function)

```
`readahead_level_of` for a built θ.  With `op`, the PER-OP-CLASS level (its broadcast axes
    dropped — the axis that operand actually reads ahead along); without, the ord-level answer.

    See ADR 0005.
    
```

## `Tensile/LoopModel/geometry.py.group_owned_tiles` (function)

```
Fan tiles ONE register group OWNS — its share of `grouping_mode`'s values (§2.8 move 1).

    The equal-partition `|part|` of §6.2, in TILES.  1 for a fragment with no `grouping_mode`
    (there the fan is not partitioned into groups at all; it stays in the rate).
```

## `Tensile/LoopModel/geometry.py.group_unit_tiles` (function)

```
THE UNIT: fan tiles ONE BUFFER of this group holds, derived from `ord` AND the split.

    See ADR 0005.
    
```

## `Tensile/LoopModel/geometry.py.group_fan_reloads` (function)

```
Buffer-generations the FAN contributes to `R` = owned tiles / tiles-per-buffer.

    See ADR 0005.
    
```

## `Tensile/LoopModel/geometry.py._rate_broadcast` (function)

```
The axes a register group is CONSTANT over for its RATE/live-peak — the "two-map reading"
    (§2.6).  A register group is a *partition of `grouping_mode`'s values* (§2.8 canonical form):
    sibling values of that mode select a DIFFERENT part (a different storage-disjoint fragment),
    so the grouping mode does NOT reload a single group — it is constant FOR THE RATE.  The rate
    is thus over the HELD modes only (substep etc.).  So this set is the group's broadcast axes
    PLUS its `grouping_mode` PLUS the operand's FREE `region_modes` (all spatial tile-count, not
    reload).  A fragment with no `grouping_mode` keeps its fan IN the rate — there the fan IS the
    rotation.  The same modes stay non-broadcast for the GRID role (computed on the data map),
    which is why roles and rate are separate computations.

    See ADR 0005.
    
```

## `Tensile/LoopModel/geometry.py._group_value_seq` (function)

```
The distinct value id a rate-group holds at each step = projection of the step coord
    onto the modes the group is NOT nullspace over (§2.6), region modes excluded (per-group,
    per-fragment reading).
```

## `Tensile/LoopModel/geometry.py.group_rate` (function)

```
R — reload rate = # distinct values ONE register group of ONE fragment holds per reduction chunk
    = product of extents of the modes its intra-fragment read time-map is nonzero over
    (non-`nullspace`, region modes excluded).  From (ord, tile).  §2.6 table row 1 + line 333:
    a fragment split r-ways has the SAME per-fragment band [L,R] as unsplit — the split
    multiplies the NUMBER of groups (resident_free_tiles), not the rate.  So the flagship's
    lo/hi are R=4 whether or not M is region-split; whole-fan R=8 is the over-count to avoid.
```

## `Tensile/LoopModel/geometry.py.group_live_peak` (function)

```
L — the live peak, the derived FLOOR (§2.6 table row 2).  Peak simultaneously-live
    generations, a cyclic interval-overlap where a generation `(iteration, value)` has a read
    interval in step-index space.  Computed on 3 unrolled iterations, peak over the steady one.

    See ADR 0005.
    
```

## `Tensile/LoopModel/geometry.py._reg_read_off` (function)

```
The register read-prefetch depth `off(read, ·)` = the PLR read-ahead, read from the
    first-class off_map (§2.4, spec §70: δ lives in off_map, not on the hop).  Keyed on the
    READ-AHEAD LEVEL (`readahead_level`, #321) — the outermost mode of the region's inner nest,
    which is the innermost reduction mode only when the reduction is outermost.  0 for an op-class
    with no input trajectory (the output/accumulator) or no intra-region axis.

    See ADR 0005.
    
```

## `Tensile/LoopModel/geometry.py._free_tile_index_seq` (function)

```
The operand's OWN free-axis tile index at each step = the mixed-radix combine of THIS
    operand's free presence modes (`theta.free_names(op)` = presence \ reduction, §2.6), in ord
    order.  The identity whose live-peak is the resident tile count (how many distinct free tiles
    are simultaneously held under this ord).  Presence-derived — no operand-name / role test.
```

## `Tensile/LoopModel/geometry.py.resident_free_tiles` (function)

```
The number of the operand's free-axis tiles SIMULTANEOUSLY RESIDENT under `ord` — the
    live-peak of its free-tile index over the traversal (§2.6 line 279/285-288/302).  When
    the operand's free axis is INNERMOST it turns over cleanly: only a window of tiles is live
    at once (min footprint, more ds_read traffic).  When the free axis is OUTER and the other
    operand's axis sweeps inside, every free tile is revisited each inner pass, so all are
    resident (full fan).  So the footprint SHRINKS or GROWS with the loop order — this is
    feature #4's residency half.  Block reuse (holding fewer tiles resident, reloading more
    often) is expressed as a smaller rotation width W via the register partition + width policy (§2.6) —
    there is no separate residency cap; the derived live-peak is the reuse the ord allows.
    Reuses the same interval-overlap machinery as group_live_peak, on the free-tile index
    instead of a rate-group's value.
```

## `Tensile/LoopModel/geometry.py.lds_buffers` (function)

```
S_shared: the shared ring BUFFER depth (per reduction chunk) (§2.4/§2.6).

    See ADR 0005.
    
```

## `Tensile/LoopModel/geometry.py.operand_generations` (function)

```
Σ over REGISTER rate-groups of the group's depth (generation count) — from derive_S.
    Excludes the 'shared' entry, which is the shared ring depth (per reduction chunk), a DIFFERENT placement
    that must not be summed into register generations (§6.2 keeps placements separate).
```

## `Tensile/LoopModel/geometry.py.group_regs` (function)

```
Register width of ONE reuse group = its share of the fragment's registers.
    A fragment of `frag_regs` registers split into `total_slots` slots, this group holding
    `size_of(group)` of them, occupies frag_regs·size_of/total_slots registers.
```

## `Tensile/LoopModel/geometry.py.operand_footprint_regs` (function)

```
Physical VGPRs for this operand.

    GROUPING-MODE operand (register groups = a partition of `grouping_mode`'s values, the
    paper's canonical form §2.8 move 1) — the paper's footprint (§5.2), no double-count
    (§5.1 trace: mxfp8 A=192, B=256):

        footprint = region_count × Σ_part( W_part × |part| ) × frag_regs

    where W_part = the rotation width S stores (held modes only — the grouping mode's values
    are tile-count, excluded from the rate by _rate_broadcast), |part| = the # of grouping-mode
    values in that part (per-part, so uneven partitions work uniformly), region_count =
    region_modes extent.  The fan lives in exactly ONE factor (|part| × region_count), never
    also in W — this is what prevents the double-count.  ASSUMPTION (current): EQUAL partition
    — every part holds `extent/n_parts` values (validated divisible); the per-part `|part|`
    shape is already the paper's, so uneven partitions are a one-field extension (a group
    carrying its own value-subset size), not a formula change.

    NON-grouping operand (flagship explicit-W, §3.6, blocked) — unchanged: Σ(depth × group
    width) × resident_free_tiles (feature #4 residency, §2.6 line 279).
```

## `Tensile/LoopModel/geometry.py.operand_emitted_regs` (function)

```
VGPRs the ALLOCATOR will actually reserve for this operand — `region x max_g W_g x |fan| x
    frag_regs`.  The model's own footprint (`operand_footprint_regs`) is the RAGGED
    `Sum_part(W_part x |part|)`; this is the flat shape the emitter realizes.

    THE TWO DIFFER ONLY FOR A PARTITIONED FRAGMENT, and there the emitter does not deliver the
    ragged saving: `KernelWriter.loopModelRegBuffers` collapses each operand to `max W_g`, and
    `numVgprBuffer` is then the max over operands, with a buffer-major / fan-minor `.set` table.
    So a `{lo: W, hi: L_war}` partition reserves exactly what the unpartitioned kernel does (#320).

    THE BUDGET SEARCH MUST SCORE THIS, not the model footprint, or it "fits" a budget by choosing a
    partition that changes nothing.  Measured on the shipping mxf8 path (`DepthU=512`, `KMN`,
    `PrefetchLocalRead=2`, budget 224): the search reported `304 -> 202 regs` and partitioned every
    operand two ways; the allocation was unchanged, and the partition then broke the `T < M` short
    arm, which does not prime a partitioned ring — 16 unserved consumers at register group 1,
    aborting codegen.  Scoring the emitted shape makes the ladder find no candidate that helps, so
    it leaves the kernel unpartitioned and says so, and the overflow is reported by the late
    budget refusal where the total is real.

    This makes the partition path DORMANT rather than deleted: it becomes useful the moment the
    allocator is tile-major, which is the other half of #320.
```

## `Tensile/LoopModel/geometry.py._region_count` (function)

```
Regions the operand's REGISTERS are replicated over — the WAVE's region span, not the
    TILE's (#311).

    §6.2's `region_count × Σ_part(W × |part|)` reads `|part|` as a PER-REGION tile count.  Our
    `M_inner` is not per-region: `translate` sets `mInner = fanM // wave_region_span` (#245), so a
    wave sitting inside one region keeps the WHOLE `MIWaveTile` on `M_inner`.  The factor that
    reconstructs the wave's fan is therefore the divisor `translate` applied — `wave_region_span`
    — and it is 1 exactly when the wave spans one region.
    Using the TILE's region count instead counted the same tiles twice: measured `MIWaveTile[8,8]`
    with `TDMSplit=[2,2,1,1]` gave `M_split(2) × M_inner(8)` = 16 modelled tiles against the wave's
    8, doubling `foot(A)` 512 -> 1024.  A region split partitions the SAME tiles into
    storage-disjoint regions; it cannot create data, so the footprint must not move.
```

## `Tensile/LoopModel/geometry.py.operand_buffer_regs` (function)

```
Registers this operand loads per kiter = PRESENCE tiles × frag_regs (§2.1).

    Was `free_tiles × k_tiles × frag_regs`, which multiplies in the theta-level reduction extent
    unconditionally and so over-counts any op-class BROADCAST over the reduction by exactly that
    factor (#212: the accumulator came out 4 x 2 x 16 = 128 registers for a tile holding 64).
    `presence_tiles` is the same product for an operand present on the reduction — A and B are
    unchanged — and drops the factor for one that is not, with no role or name test.
```

## `Tensile/LoopModel/geometry.py.plan_transfers` (function)

```
Instructions this hop issues per kiter.  A 'tdm' bulk hop issues `hop.split` (one
    cooperative instruction per region part).  A 'load' per-lane hop issues buffer-regs /
    load-size.
```

## `Tensile/LoopModel/geometry.py:1` (comment)

```
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
```

## `Tensile/LoopModel/geometry.py:41` (comment)

```
# Tile counts — tile extent ÷ compute atom, in element space (the honest extent source).
# ===========================================================================
```

## `Tensile/LoopModel/geometry.py:55` (comment)

```
# Per-operand access layout — the honest spatial object.
# ===========================================================================
```

## `Tensile/LoopModel/geometry.py:69` (comment)

```
# An AGENT-SERVED region axis is broadcast for every data consumer (#245): the wave reads,
# computes and accumulates inside its own region. One subtraction point, so the read coord,
# free_modes, register naming, the #288 wmma projection and the footprint cannot disagree.  (ADR 0007 #55)
```

## `Tensile/LoopModel/geometry.py:99` (comment)

```
# carrier = t // Φ either way: Φ tiles ride one instruction, and the instruction's ordinal IS
# the register base.  ρ changes only WHERE those Φ tiles live, hence only the SLOT.
```

## `Tensile/LoopModel/geometry.py:104` (comment)

```
# The group does not divide evenly over the sub-agents, so some agent would carry a
# fraction of a tile.  No `slot` expresses that; say nothing rather than emit one that
# is right for neither (§2.2 names only the even sub-wave partition).
```

## `Tensile/LoopModel/geometry.py:108` (comment)

```
# DISTRIBUTED (§2.2): the Φ tiles are spread over `rho` sub-agents, so `rho` of them share
# ONE register and the lane decides which — a BROADCAST, expressed by collapsing the slot
# rather than flagged.  (ADR 0007 #54)
```

## `Tensile/LoopModel/geometry.py:136` (comment)

```
# The operand's flat tile index, ord order with the innermost axis fastest — the same
# flattening `tile_flat` uses, so `carrier` is evaluated on the coordinate it was written for.
```

## `Tensile/LoopModel/geometry.py:180` (comment)

```
# §2.2: the axes ONE instruction SPANS are absent from the hop's presence, exactly as a
# bulk copy's inner axes are.  (ADR 0007 #53)
```

## `Tensile/LoopModel/geometry.py:204` (comment)

```
# Free / reduction extents / wmma grid — PRESENCE-derived, no operand-name/role literals.
# ===========================================================================
```

## `Tensile/LoopModel/geometry.py:262` (comment)

```
# Fragment register width — via latalg.recast_count (integer ceil), not a layout op.
# ===========================================================================
```

## `Tensile/LoopModel/geometry.py:280` (comment)

```
# Register groups: what one group owns, how fast it turns over, how much is live
# ===========================================================================
```

## `Tensile/LoopModel/geometry.py:370` (comment)

```
# PARTIALLY FOLDED: the axis KEEPS `extent // q` presence points, so the fan contributes
# that many generations, not `extent`.  (ADR 0007 #51)
```

## `Tensile/LoopModel/geometry.py:548` (comment)

```
# No shared placement — a one-hop global->register (DTV) operand (§5.3.1). There is no LDS
# ring to have a depth, so any number here would be a fiction.  (ADR 0007 #49)
```

## `Tensile/LoopModel/geometry.py:563` (comment)

```
# Sizing: how many instructions a hop costs, and how many registers it occupies
# ===========================================================================
```

## `Tensile/LoopModel/geometry.py:583` (comment)

```
# Register footprint — generations × width × tiles  (the product, §6.2).
# ===========================================================================
```

## `Tensile/LoopModel/geometry.py:622` (comment)

```
# the ACCUMULATOR: one live value per output tile, held across the WHOLE reduction — no
# rotating ring (§6.1), no read-ahead residency.  (ADR 0007 #48)
```

## `Tensile/LoopModel/geometry.py:697` (comment)

```
# Instruction counts.
# ===========================================================================
```

## `Tensile/LoopModel/geometry.py:733` (comment)

```
# The compute grid.
# ===========================================================================
```

## `Tensile/LoopModel/ir.py <module>` (module)

```
IR vocabulary — the totally-ordered instruction stream the decoder emits.

This module is the decoder CORE: it contains NO TensileLite / AMD-specific concepts,
only a generic async-copy + compute + completion-counter machine.  It is the artifact
a consumer (e.g. a code generator) decodes into a real loop.

Every op is a plain dataclass so the stream serializes trivially.  The logical
destination (space, rotation slot, register group) lives on `Inst.placement` as a
structured `Placement`, so a consumer reads fields instead of parsing a string.

── The IR is ADDRESS-FREE (and therefore already "collapsed") ──────────────────────
This IR carries no physical address / thread / lane / instruction-width information.  Each
op-class appears ONCE, at its presence level, with a SYMBOLIC rotation slot (`Placement`,
an `Expr` like `buf(min)%2`) — the paper's rolled region-tree (§5.2), size O(#op-classes ×
#levels), independent of trip count.  A consumer (TensileLite) DECODES this into concrete
per-thread instructions: it lowers the logical slot to a physical VGPR color, expands a bulk
tile into its per-lane transfers of a given vector width, and materializes the loop.  The
model deliberately treats addresses as opaque (§2.1) — the spatial layout algebra (CuTe-style
addresses, swizzles, per-lane fragment layouts) is the *companion enrichment layer* (§7), NOT
part of θ.

WHY THERE IS NO `--collapse` / no per-instance Inst.  An earlier (layout-algebra) version of
this IR enumerated ONE Inst per thread/address instance — a `ds_read` fanned into many
concrete loads carrying distinct addresses — and a `--collapse` renderer folded runs of
identical-`semantic` ops into `(inst) ×N`.  Trimming the layout algebra removed that
enumeration: the rolled IR IS the collapsed form, so there is nothing left to fold.  The
dead `Dim`/`Slot`/`Pad` address-slot classes and the collapse machinery were removed with it.

HOW TO RE-ADD ADDRESS SUPPORT (if a future kernel needs the enrichment layer).  This is an
ADDITIVE layer (the paper proves it changes no result, §7); do NOT fold it into the core:
  1. Add a `layout`/`addr` field on `Placement` (or a parallel `AddrPlacement`) carrying the
     per-lane element→(register, byte) map + swizzle — the CuTe layout the core treats as
     opaque.  Leave `Placement.space`/`slots`/`group` (the logical slot) exactly as-is.
  2. Add an OPTIONAL renderer/backend pass that EXPANDS a rolled Inst into per-lane/per-
     transfer instances by walking that layout — reviving the old per-instance stream as a
     downstream VIEW, never the stored IR.  `Load.semantic`/`Mma.semantic` (kept below) are
     the fold key for re-collapsing such an expanded view.
  3. Instruction width (elements/transfer) is a lowering of the hop's `vector_elems` (already
     on `Hop`) × the layout — compute it in that pass, not in θ.
Keep the invariant: θ + this IR stay address-free; addresses live in the enrichment pass and
in TensileLite's decoder.
```

## `Tensile/LoopModel/ir.py.Counter` (class)

```
COMPLETION CLASSES (§2.7): "a nameable set of operations a synchronization primitive can
    wait on".  Named in the paper's own notation — `C_<movement>`, after §5.2's `C_mma` — because
    the class is a MODEL object; which hardware counter realizes it is the backend's business.

    These carried the gfx1250 wait mnemonics as their values (`"s_wait_dscnt"`, …), which put
    instruction names in the backend-agnostic core.  The mnemonic was matched by nothing: the only
    consumer was the renderer, which stripped the `s_wait_`/`cnt` boilerplate back off to display
    the class.  A backend targeting a counter machine maps these to its own instructions.
```

## `Tensile/LoopModel/ir.py.QuantumMap` (class)

```
How an op-class's TILES merge into the instructions that move them (§2.2, Precondition Q).

    See ADR 0005.
    
```

## `Tensile/LoopModel/ir.py.Load` (class)

```
A named transfer op: move `tokens` from `src` space to `dst` space.

    See ADR 0005.
    
```

## `Tensile/LoopModel/ir.py._term` (function)

```
One linear term, normalized to `(mode, coef, div)` = `coef · (env[mode] // div)`.

    See ADR 0005.
    
```

## `Tensile/LoopModel/ir.py.Expr` (class)

```
A tiny symbolic index expression over loop-mode names.  Two shapes the rolled IR needs:
    (a) single-var `(var + add) % mod` — a shifted coord or a one-mode rotation slot; and
    (b) MIXED-RADIX `(Σ coef·(var // div) + add) % mod` via `terms` — a rotation slot whose group
    cycles over SEVERAL rate modes (§2.6: the rotation width W ranges over the product of the
    group's rate modes, not one mode).  `terms` is a tuple of `(mode, coef)` or `(mode, coef, div)`
    (see `_term`: `div > 1` converts a tile-valued loop index to the §2.2 presence point it names);
    when set it supersedes `var`.  `eval(env)` concretizes when the renderer unrolls.
```

## `Tensile/LoopModel/ir.py.Pred` (class)

```
A structured predicate: compare an index `Expr` against an integer.  `lhs op rhs`.

    See ADR 0005.
    
```

## `Tensile/LoopModel/ir.py.Placement` (class)

```
The LOGICAL storage an instance reads/writes (§5.2) — NOT the physical register color.

    See ADR 0005.
    
```

## `Tensile/LoopModel/ir.py.is_war` (function)

```
True iff `kind` is an ANTI-dependency (the writer waits for prior readers, §2.4 line 99).

    Raises on an unknown kind rather than answering False: "not in my list" and "is a true dep"
    are different facts, and only one of them is safe to schedule on.
```

## `Tensile/LoopModel/ir.py.Await` (class)

```
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

    See ADR 0005.
    
```

## `Tensile/LoopModel/ir.py.Inst` (class)

```
One op-class instance at its presence level (§5.2): a `Load` or `Mma` payload, its
    logical `placement` (symbolic slots), issuing `agent`, and named `awaits`.  ONE Inst per
    op-class — the loop nest around it supplies the iterations; nothing is enumerated.
```

## `Tensile/LoopModel/ir.py.Loop` (class)

```
A real loop node of the rolled nest: `for <mode> in range(extent)`.  `body` is a
    sequence of child `Loop`/`Branch`/`Inst`/`Peel` nodes.  `role` is the ρ agent role
    (single-role default).

    See ADR 0005.
    
```

## `Tensile/LoopModel/ir.py.Branch` (class)

```
Explicit register-rotation control (user decision): a switch on `selector = mode % S`,
    one `arm` per residue.  `arms` is a dict {residue-int: body-sequence} — arm r is the ops
    that run when `mode % S == r` (arm 0 uses buf0, arm 1 buf1, …).  Makes the rotation a
    first-class control structure, not just an index expression.
```

## `Tensile/LoopModel/ir.py.Bind` (class)

```
Bind an OUTER loop induction (`mode`, e.g. `iter`) to a concrete/symbolic VALUE over a
    straight-line sub-body (§5.3.1 peel steps).  A prologue/drain step is ONE reduction-chunk of
    the peeled `iter` loop pinned to its issue index: prologue step `j` → `iter = j−M`, drain step
    `t` → `iter = T−M+t` (the paper's `i = T−M+t`).  `value` is an `Expr` (concrete for prologue,
    symbolic `T`-relative for drain), so every slot/coord under the step sees a DEFINED `iter` —
    replacing the old free-`iter` `Cond(iter%d==residue)` pin (which referenced an `iter` no scope
    defined).  A walker pushes `env[mode] = value` (or carries it symbolically) when descending.
```

## `Tensile/LoopModel/ir.py.Peel` (class)

```
A prologue/drain peel node (Lemma 1): the hoisted first-`k` (`kind='prologue'`) or
    last-`k` (`kind='drain'`) iterations of loop `mode`, a straight-line body (no back-edge),
    contents fixed by `off`.  May hold instances hoisted from an enclosed level (cross-level
    boundary term).
```

## `Tensile/LoopModel/ir.py.Cond` (class)

```
A first-class conditional over the reduction trip count — the peel's VALIDITY guard
    (Lemma 1 hypothesis `T_ℓ ≥ off(·,ℓ)`).  The software pipeline's steady region is only
    reachable when the loop is long enough to fill+drain the pipe; a too-short loop takes a
    degenerate path (skip the steady body, go straight to the drain / no-load-loop).  Making this
    a NODE (not an implicit assumption) lets a backend lower the boundary correctly from the IR
    alone.

    See ADR 0005.
    
```

## `Tensile/LoopModel/ir.py.group` (function)

```
The tiles sharing `t`'s carrier — the coordinates ONE instruction serves (its preimage).

        The preimage, not a product of spanned axes: it is total over any carrier expression,
        including ones whose group is not a contiguous run (the sub-agent partner sits `stride`
        away, not adjacent).
```

## `Tensile/LoopModel/ir.py.free_vars` (function)

```
The mode names this expression references — its free variables under an env.

        ONE authority: `eval` substitutes 0 for an unknown name rather than failing, so "is this
        expression resolvable here?" cannot be asked of `eval` and must be asked of the variable
        set.  `Bind.bound_env` and the structural validator both need it.
```

## `Tensile/LoopModel/ir.py.at` (function)

```
Concrete Placement with each slot Expr evaluated to its buffer VALUE but PRESERVING the
        ring MODULUS (`mod`).  A pinned prologue/drain read must show a concrete buffer index yet
        still carry the rotation width `W` a consumer (GIR reg_band) reads from `slot.mod` — so we
        emit `Expr(add=value, mod=W)`, not a bare `cst(value)` (mod=0) which would drop the width.
```

## `Tensile/LoopModel/ir.py.ranged_bodies` (function)

```
`[(lo, hi, nodes)]` — each sub-body with the half-open iteration range it runs over.

        THE authority for the range pairing: a consumer that UNROLLS (the renderer, the GIR
        lowering) must walk each body over its own range, while a consumer that only wants the
        instructions can keep iterating `bodies`.  `hi` is None when the trip is symbolic (an
        outer runtime loop), meaning "to the end".
```

## `Tensile/LoopModel/ir.py.body` (function)

```
The SINGLE body — valid only on a single-body loop.  Raises on a multi-body loop
        (§5.1) rather than silently returning the first and dropping the rest, which is how a
        `bodies[0]` convenience turns a multi-body schedule into a wrong one.  Multi-body-aware
        code uses `bodies` (all instructions) or `ranged_bodies()` (with each body's range).
```

## `Tensile/LoopModel/ir.py.bound_env` (function)

```
`env` with `mode` bound to `value` IFF `value` is statically resolvable in `env`
        (prologue `j−M` = a constant → binds a concrete int).  If `value` references a RUNTIME
        symbol absent from `env` (drain `T−M+t`, `T` a problem dimension), leave `mode` UNBOUND —
        the body stays symbolic and the GIR lowering resolves the drain residue RELATIVE to the
        steady loop (Step 2), never pinning a wrong concrete generation.
```

## `Tensile/LoopModel/ir.py:1` (comment)

```
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
```

## `Tensile/LoopModel/ir.py:53` (comment)

```
# Spaces and completion counters.
# ===========================================================================
```

## `Tensile/LoopModel/ir.py:96` (comment)

```
# The movement quantum — how tiles merge into one instruction (§2.2)
# ===========================================================================
```

## `Tensile/LoopModel/ir.py:146` (comment)

```
# The instruction ops.
# ===========================================================================
```

## `Tensile/LoopModel/ir.py:198` (comment)

```
# Rolled θ-nest IR (paper Chapter 5 "emit IR").  See ADR 0006.
# ===========================================================================
```

## `Tensile/LoopModel/ir.py:326` (comment)

```
# `free_vars` is the ONE authority on which names an Expr reads (its own docstring says
# so); re-listing them here is what left `digits`-only expressions looking constant and
# would now also have to learn the term divisor.  Same set, one derivation.
```

## `Tensile/LoopModel/ir.py:373` (comment)

```
# `keep_unresolved` is OPT-IN and OFF for the codegen path, which must always get a
# concrete Placement.  (ADR 0007 #58)
```

## `Tensile/LoopModel/ir.py:467` (comment)

```
# display extent (§5.2 line 339, Q19 ADOPTED). For an `outer`
# (pipelined, runtime) loop `trip` is the continuation PREDICATE the
# backend lowers to a back-edge (e.g.  (ADR 0007 #57)
```

## `Tensile/LoopModel/ir.py:474` (comment)

```
# runtime-trip, peeled level (the shared-ring / reduction axis); False
# = a statically-unrolled inner mode.  (ADR 0007 #56)
```

## `Tensile/LoopModel/ir.py:570` (comment)

```
# construction (Lemma 1 peels the pipelined loop) — the structural
# marker for walkers, consistent with Loop/Branch.
```

## `Tensile/LoopModel/ir.py:595` (comment)

```
# walker uses (consistent with Loop/Branch/Peel `outer`).  False for
# the peel-validity guard (an ordinary conditional, not a step split).
```

## `Tensile/LoopModel/latalg.py <module>` (module)

```
latalg — the minimal layout arithmetic the decoder core needs, with NO CuTe/PyCuTe dependency.

The loop-schedule core is address-free (the schedule-model paper's central claim): the only
"layout" facts it uses are integer tile-counts, a broadcast (stride-0) test, and a per-lane
register-packing count.  `geometry.py` used to get these from PyCuTe (`logical_divide`,
`_nullspace`, `recast`/`_coshape`); this module replaces those three calls with plain integer
arithmetic so a TensileLite-mergeable build carries no spatial-layout library.

Each function documents the PyCuTe op it stands in for.  (The prototype's
`selftest_vs_pycute()` differential test against the vendored `Cute/` oracle is dropped in
this in-tree port — the oracle is not shipped with TensileLite; the trim was validated in the
decoder_proto repo.)

Scope: every use in `geometry.py` is over CONTIGUOUS 1-D layouts (a single extent with unit
stride), or a product of independent extents.  We implement exactly that surface; nothing here
composes strides, reads an address, or handles the general nested CuTe layout.
```

## `Tensile/LoopModel/latalg.py.tile_split` (function)

```
Stands in for `logical_divide(Layout((n,),(1,)), Layout((t,),(1,))).shape` on a
    contiguous 1-D layout.  Returns (per_tile, n_tiles).  For the divisible case the decoder
    always feeds (n % t == 0), PyCuTe yields (t, n//t): the inner extent is the tile size, the
    outer extent is the number of tiles.  We match that, and for the non-divisible case fall
    back to (min(t,n), ceil(n/t)) — but callers guard divisibility, so only the exact case is
    load-bearing.
```

## `Tensile/LoopModel/latalg.py.recast_count` (function)

```
Stands in for `recast(Layout((elems,),(1,)), scale).size` = ceil(elems / scale): packing
    `elems` sub-units into groups of `scale` (e.g. elements per register).  PyCuTe's recast
    rounds up (recast((4,),4)=1, recast((1,),4)=1, recast((16,),2)=8).
```

## `Tensile/LoopModel/latalg.py:1` (comment)

```
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
```

## `Tensile/LoopModel/ledger.py <module>` (module)

```
The obligation ledger (paper §4) — hazards that MUST be discharged, and the empty-ledger gate.

Clean-architecture split of the decoder (modeled on GIR): this module owns the §4 ledger and
the §5.5 whole-tree discharge check.  No TensileLite / AMD vocabulary; stdlib + LoopModel core
only.  `build_ir` (the emit_can driver, `emit.py`) consumes `build_ledger`.

  build_ledger              one RAW-residency per hop + one WAR per S rate-group (§4.1)
  check_ledger_discharged   the per-edge empty-ledger gate over the whole tree (§5.5)
  walk_insts / all_awaits / _region_bodies   tree traversal helpers the gate uses
```

## `Tensile/LoopModel/ledger.py.Endpoint` (class)

```
One end of an obligation — WHICH op-class, at WHICH placement or site, at WHICH coordinate.

    See ADR 0005.
    
```

## `Tensile/LoopModel/ledger.py.movement_name` (function)

```
The PRODUCER identity of `opname`'s global→shared movement (§2.8 move 9, Φ).

    A Φ-fused group is ONE cooperative instruction that completes once, so every member's data
    becomes resident at the same instant and there is no per-member completion to await.  The
    producer of a copy-hop obligation is therefore the MOVEMENT, and a consumer of any member
    awaits that one event.

    Returns the bare op-class name for a lone movement — so an unfused kernel's obligations,
    `Await` deps and IR dumps are byte-identical to before — and the member tuple for a fused one.
    `theta.movement_units()` is the single authority for what the movements are; `TokensPass`
    already keys the completion token off the same thing, and they must not drift.
```

## `Tensile/LoopModel/ledger.py.build_ledger` (function)

```
One RAW-residency per hop + one WAR per S rate-group.

    See ADR 0005.
    
```

## `Tensile/LoopModel/ledger.py.discharge_once` (function)

```
Drop an `Await` that a preceding instruction in the same execution path already established.

    See ADR 0005.
    
```

## `Tensile/LoopModel/ledger.py.walk_insts` (function)

```
Pre-order walk of the rolled θ-nest yielding every `Inst` (one per op-class per presence
    level — NOT enumerated).  Recurses Loop bodies, Branch arms, Peel bodies, and Cond arms.
    `tree` is the node list build_ir returns (or a nested body).
```

## `Tensile/LoopModel/ledger.py._region_bodies` (function)

```
The instruction lists that form ONE emitted TRIP each — the σ_c ordering unit.  A trip is a
    straight-line body once its INNER reduction/fan loops (K_inner/M_inner/N_inner) are unrolled
    inline by the walker; the WAR (refill) must be preceded WITHIN its trip by the read that vacates
    the buffer it refills.  The TRIP boundaries are the OUTER `iter` structures: the steady
    `Loop(mode='iter')` body is one trip, the prologue Peel body is one trip, and each drain step
    (a `Bind(chunk = T-M+t)` under the drain Peel) is one trip.  Everything else nested inside a
    trip (the inner Loops/Branches/Conds) is flattened INTO that trip, in program order.
```

## `Tensile/LoopModel/ledger.py.check_ledger_discharged` (function)

```
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

    See ADR 0005.
    
```

## `Tensile/LoopModel/ledger.py.render` (function)

```
Display form.  Reads like the old string so dumps stay familiar, but it is DERIVED
        from the fields — nothing parses it back.  A fused movement renders `A+B`, matching the
        `<GIR: … A+B>` tags in the emitted assembly.
```

## `Tensile/LoopModel/ledger.py:1` (comment)

```
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
```

## `Tensile/LoopModel/ledger.py:125` (comment)

```
# Same Φ rule as the copy-hop residency above, and for the same reason on BOTH ends:
# one cooperative copy, one crossing obligation, matched by every member's crossing
# read.  Two per-member obligations would be two accounts against one instruction.
```

## `Tensile/LoopModel/ledger.py:132` (comment)

```
# Read-ahead residency O_r (Lemma 3d): a first-trip read the steady body cannot discharge
# must be pre-issued in the prologue, else the ledger is non-empty.  (ADR 0007 #65)
```

## `Tensile/LoopModel/ledger.py:169` (comment)

```
# each trip re-runs the body; a boundary from outside does not survive into trip 2,
# and the body's own copies invalidate it anyway.
```

## `Tensile/LoopModel/ledger.py:250` (comment)

```
# `inline_outer` = descend the OUTER-level Branch that wraps a DRAIN step (the peel's
# `Bind` binds each drain step's reduction chunk residue via a Branch(outer=True)).  (ADR 0007 #63)
```

## `Tensile/LoopModel/ledger.py:277` (comment)

```
# each drain step is `Bind(iter=T−M+t, body=...)`; flat() inlines the Bind
# body so the σ_c order gate inspects that trip's instructions.
```

## `Tensile/LoopModel/ledger.py:309` (comment)

```
# `len(body)` for an operand with NO read in this trip: past every index, so the
# copy-first peel skeleton stays a violation exactly as it was under the existential.
```

## `Tensile/LoopModel/ledger.py:317` (comment)

```
# (b3) order, CROSSING RAW: the copy must be emitted BEFORE this operand's reads in the steady
# body, because those reads consume the chunk it fills (§5.3.1 pt5).  (ADR 0007 #61)
```

## `Tensile/LoopModel/ledger.py:385` (comment)

```
# (c) READ-AHEAD RESIDENCY (Lemma 3d): each `readahead-residency` obligation requires its coord
# (a FIELD on the producer endpoint) to be actually emitted as a register read in the PROLOGUE
# body.  (ADR 0007 #59)
```

## `Tensile/LoopModel/ledger.py:441` (comment)

```
# (b) order: per region, every WAR-refill copy must be preceded by the vacating read. ONLY the
# `S=δ` in-place refill is order-constrained to follow its reads (§5.3.1 point 4 is scoped to
# "an S=delta refill copy").  (ADR 0007 #62)
```

## `Tensile/LoopModel/operand_view.py <module>` (module)

```
What an operand IS: the facts derivable from theta alone, with no policy in them.

The companion of `schedule.OperandPlan` -- these answers do not change with PLR or the register
budget, and every one of them does. A view, not a move: theta owns the operands, so an operand
cannot reach its own theta.
```

## `Tensile/LoopModel/operand_view.py:1` (comment)

```
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
```

## `Tensile/LoopModel/prefetch.py <module>` (module)

```
How far ahead each load runs, and everything that follows from it.

This is the half that needs `S`, the register depth map decided in `schedule.py`:
prefetch steps per group, the peel depth around them, what the prologue must preload,
and where each instruction is placed.
```

## `Tensile/LoopModel/prefetch.py.readahead_reach` (function)

```
How many whole reduction chunks the prefetch crosses into -- the MAX over groups.

    The peel depth and the drain suppression window both need the deepest group; the
    per-instruction `_readahead_depth` takes the MIN and must not be unified with this.
```

## `Tensile/LoopModel/prefetch.py.copy_must_be_first` (function)

```
Must `op`'s shared copy be emitted BEFORE this trip's reads? (§5.3.1 pt5.)

    The copy in trip `iter` fills chunk `iter + off(copy, iter)`, whose slot is
    `(iter+off) mod S_shared`.  This trip READS chunks `iter … iter+r`.  Which relation the copy
    has with those reads is decided by WHICH of them its slot lands on:

        off == r   the copy fills the very chunk the read-ahead crosses into -> RAW.
                   The crossing reads consume it, so the copy must come FIRST.
        off != r   the copy's slot collides with a chunk read EARLIER in the trip (or with none)
                   -> WAR, or nothing.  The copy must come LAST.

    Worked, at S_shared = 2, r = 1 — the two shipping points, and they go opposite ways:

        PGR1  off=1  slot (iter+1)%2   trip reads iter, iter+1   -> hits the CROSSING read  -> first
        PGR2  off=2  slot (iter+2)%2   trip reads iter, iter+1   -> hits the CURRENT read   -> last
                     = iter%2

    An earlier version of this returned `S_shared > r`, which is True for both and so emitted
    PGR2 copy-first: the copy then overwrote the buffer the trip was still consuming.  That is a
    WAR violation, and it miscompared on hardware at `T-M >= 1` steady trips (K=256 for PGR2,
    where M=2; K=128 already sufficed for PGR1, where M=1).  The paper's "when the shared depth
    equals the reach" describes when copy-first fails *within* the `off == r` branch — it is not a
    standalone predicate for choosing the branch.
```

## `Tensile/LoopModel/prefetch.py.chunk_crossing_violations` (function)

```
The §5.3.1 pt5 coupling `off(copy, iter) >= r + 1`, per operand — [] when θ is emittable.

    The shared-ring companion to the register floor `L(p)` (§2.6): `L` sizes the register buffering
    the read-ahead needs, `off(copy, iter) >= r+1` sizes the SHARED buffering it needs.  A chunk `c`
    is copied in steady trip `c - off(copy, iter)` and copies sit AFTER reads in the steady body
    (§5.3.1 point 4), so the copy of chunk `iter+r` precedes the read that wants it only if it was
    issued in an EARLIER trip.

    Violating it is not a tuning wart: the first steady trip reads a shared buffer never written,
    and every later trip reads the copy issued after it in the same body.  The paper says the
    ledger catches this by construction; this closed form lets the decoder prune the invalid
    `(dr, off(copy))` combinations up front instead of discovering them at ledger-walk time.

    Returns [(operand, dr_g, r, off_copy, required)] for each operand that violates it.
```

## `Tensile/LoopModel/prefetch.py.PeelDepths` (class)

```
The peel depth `M_ℓ` **per level** (§5.3.1, 2026-08-14 update) — ONE source of truth for the
    peel construction in `build_ir`.

    The paper's construction is explicit that `M` is a VECTOR, not an integer: *"For EACH level ℓ
    that carries an offset, `M_ℓ = max_p off(p, ℓ)`. A level with `M_ℓ = 0` needs no peel; a level
    with `M_ℓ > 0` gets its own prologue/drain of depth `M_ℓ`."*  It also names the defect this
    replaces: *"a decoder that keys peel depth only on the innermost outer level silently drops
    every coarser-level `off`."*  That was exactly this function, which read
    `off_at(op, reduction_chunk_name())` and returned a scalar.

    Fields:
      per_level — {level_name: M_ℓ}, only levels with M_ℓ > 0, in ord order (outer→inner).
      offs      — {(op_name, role, level_name): δ}, the per-op-class offsets each M_ℓ is the max
                  of.  Per-op-class, because §5.3.1 pt1-2's peel is STAGGERED by each op-class's
                  own δ, not by the level's max.
      dr        — the substep read-ahead depth (`max off(read, substep)`), kept separate because
                  the whole read-ahead family (`preloaded_tiles`, `_readahead_shift`,
                  `chunks_crossed`) is expressed in SUBSTEPS, not in whole chunks.
      chunk     — the reduction-chunk level's name, or None for a legacy θ with no outer level.
      reach     — `r`, the DERIVED chunk reach of that read-ahead (`readahead_reach`): how many
                  whole reduction chunks the advance the emitter ACTUALLY issues crosses into.
                  Distinct from `dr`, which is the REQUEST; the two differ whenever §2.6 clamps a
                  group (`dr_g = min(dr, W_g−1)`, or 0 under a structural veto).  Every consumer
                  that counts CHUNKS — the chunk-level peel below, the drain's read-ahead
                  suppression window (`emit`) — must read this, never `dr`.
      deferrals — {(op_name, role, level_name): δ} for δ < 0: the NEGATIVE half of the ℤ-valued
                  retime matrix, i.e. DEFER (issue |δ| later) rather than prefetch.  Kept apart
                  from `offs` and never folded into `M_ℓ`, because §5.3.1's construction is written
                  for δ > 0 and gives a negative δ the wrong presence (see `peel_depths`).
                  Non-empty ⇒ `build_ir` refuses.  Author question Q29.
    
```

## `Tensile/LoopModel/prefetch.py.peel_depths` (function)

```
Build the per-level `PeelDepths` for `theta` (§2.4/§5.3.1).

    `M_ℓ` is the max over **every op-class carrying an offset at `ℓ`** — the paper's updated
    wording is "copies AND a slab-level read offset (e.g. `off(read,iter)`), not copies only", which
    the (op-class, level) key from #209 finally makes expressible.  At the reduction chunk the
    substep read-ahead also contributes, converted to whole chunks (`ceil(dr / n_substeps)`):
    that is a *reach*, not an offset at the chunk, so it is added to the chunk's max rather than
    entered into `offs`.

    Every δ is read through `off_at(op, role, level)`; the level NAMES are derived from θ (the
    outer levels from `outer_modes()`, the substep from the innermost reduction inner mode) — no
    literals.
```

## `Tensile/LoopModel/prefetch.py.BoundaryHoist` (class)

```
One instance of §5.3.1's CROSS-LEVEL BOUNDARY TERM (Lemma 1).

    *"When `off(p, ℓ_outer) = δ > 0` for an op-class `p` whose HOME level is INNER to `ℓ_outer`,
    the outer offset hoists `p`'s leading δ instances of the inner nest OUT of the current
    `ℓ_outer` iteration and INTO the DRAIN of the PREVIOUS `ℓ_outer` iteration (equivalently, at
    the first outer iteration, into the outer PROLOGUE).  This is a determinate relocation — the δ
    instances are exactly `[inner-coord 0 .. δ)` of the current outer tile."*

    op/role — the op-class whose instances relocate.
    level   — `ℓ_outer`, the level whose offset does the hoisting.  It must be an OUTER
              (runtime-trip) level: the relocation target is "the DRAIN of the PREVIOUS `ℓ_outer`
              iteration", and only an outer level HAS a prologue/drain region to relocate into.
              An offset at an INNER (statically unrolled) level is not a boundary term — the inner
              nest has no back-edge, so there is no boundary to cross; that case is the ordinary
              §5.3.1 pt5 read-ahead (`off(read, substep)`), whose reach into the next chunk is
              already modelled by `chunks_crossed` + `preloaded_tiles` and whose "peel" is
              absorbed into the unroll.
    home    — the op-class's own emission level, strictly INNER to `level` (that is the test).
              `home == level` is the ordinary staggered peel, not a relocation.
    delta   — δ, how many leading inner-nest instances move.
    coords  — those instances, as concrete presence-coord dicts in ord order.
    
```

## `Tensile/LoopModel/prefetch.py._leading_inner_coords` (function)

```
The relocated instances — the paper's *"exactly `[inner-coord 0 .. δ)` of the current outer
    tile"*.  The inner coordinate is the HOME level's own index (that is the level whose instances
    the outer offset relocates), so the set is simply its leading δ values.
```

## `Tensile/LoopModel/prefetch.py.boundary_hoists` (function)

```
Every cross-level boundary term `theta` carries (see `BoundaryHoist`), outer level first.

    THREE conditions, all necessary:
      1. `level` is an OUTER level — only an outer level has the prologue/drain region the
         instances relocate INTO.  An offset at an inner (unrolled) level crosses no boundary; that
         is the ordinary §5.3.1 pt5 read-ahead, handled by `chunks_crossed`/`preloaded_tiles`.
      2. `home` is strictly INNER to `level` in ord order.  `home == level` is the ordinary
         staggered peel (a copy prefetching its own chunk level), not a relocation.
      3. `home` is itself an OUTER (looped) level.  This is the discriminator that separates the
         paper's canonical case from ordinary prefetch, and it is worth spelling out because the
         two look alike under conditions 1-2 alone:

           * home OUTER (`off(copy, gtile)`, home `iter`) — the op-class has a whole ENCLOSED
             ITERATION SPACE per `gtile` trip, and δ of those iterations' instances move into the
             previous trip's drain.  This is §5.3.2's next-tile prefetch: a relocation.
           * home INNER (`off(copy, iter)` for a REGION-SPLIT copy, home `M_split`) — the op-class
             has a fixed unrolled fan per trip and every member of it stays in the trip; δ shifts
             which CHUNK the instances fetch, not which trip they are issued in.  That is
             `PrefetchGlobalRead`, the ordinary staggered peel, and reading it as a relocation
             flags every shipping region-split kernel.

    Empty for every θ the parameter path builds today: `translate` sets `off(copy, chunk)` — whose
    home is the chunk itself, or a region mode that is inner and unrolled — and
    `off(read, substep)`, whose level is inner.  It becomes non-empty the moment a coarser outer
    level exists (§5.3.2's persistent `gtile`, #113).
```

## `Tensile/LoopModel/prefetch.py.preloaded_tiles` (function)

```
The coords loaded before the loop, so the first trip has data.

    Prologue depth == steady advance == the DERIVED depth, per group: a group that does
    not prefetch preloads nothing. Priming one while the other refills in place is the
    named failure -- the pipe filled once and never refilled. ADR 0004.
```

## `Tensile/LoopModel/prefetch.py.prefetch_steps_for` (function)

```
`dr_g` — the per-group read-ahead depth, DERIVED from the §2.6 floor (`W_g ≥ L`).

    §2.6:  `L(p) = (dr_g + 1) + [1 if term (ii) fires]`, and Lemma 3b forces `W_g ≥ L`.  Solving
    for the deepest admissible look-ahead, with term (ii) read off `broadcast_width`:

        dr_g  =  0                        if fan(p) > 1     # L = dr_g + 2 > W_g for any dr_g ≥ 1
                 min(dr, W_g − 1)         otherwise         # L = dr_g + 1 ≤ W_g  (§5.3.1 line 525)

    `dr` is the requested depth `off(read, substep)`; `W_g = S(op, group)` is the searched width.

    This is ONE quantity, and §5.3.1's Shape A / Shape B rule makes it the ONLY one: *"prologue
    depth = steady advance = `dr_g` for every group; round up to the first tile's full K only when
    `dr_g ≥ 1` and INNER = ∅; a `dr_g = 0` group prologues nothing and refills in place."*  So
    `preloaded_tiles` and `_readahead_shift` must both call this, and a group is either

      Shape A (`dr_g ≥ 1`): read hoisted ahead of its consuming wmma, prologue non-empty
                            (full first-tile K when INNER = ∅), steady advance non-zero;
      Shape B (`dr_g = 0`): read NOT hoisted — issued in place one line before its own wmma with a
                            completion wait between — prologue EMPTY, steady advance 0.

    never a mixture.  The failure mode the paper calls out by name is a Shape-A prologue paired
    with a Shape-B steady: the pipe is primed and then never re-primed.  (That was this decoder's
    state until this function existed: the fill ran at the global `dr` while the steady ran at 0.)
```

## `Tensile/LoopModel/prefetch.py.prefetch_refusal` (function)

```
Why `prefetch_steps_for` returned less than the REQUESTED `dr` — or None if it did not.

    PLR IS AN INPUT AND `W` IS DERIVED, so a request the derivation cannot honour is a fact about
    the kernel that must be REPORTED, never absorbed.  `prefetch_steps_for` returns one integer
    and that integer is all any caller sees, so a `dr_g` of 0 is indistinguishable from a `dr` of 0
    — the schedule reads as "PLR was not asked for" when what happened is "PLR was asked for and
    silently dropped".  `render`'s `readmode` line says as much in its own note: "a printed dr_g of
    0 does not by itself say which one fired".  This says which one.

    Returns `(requested, derived, reason, detail)`:
      'band-empty'  the band `[L_war, R]` cannot hold the read-ahead generation at all.  §2.6 caps
                    `W <= R` and `group_rate` counts the distinct values of ONE trip, so an operand
                    whose values stay live across the whole trip needs `(dr+1) x B` names against a
                    ceiling of `B` — no width in the band admits it.  MEASURED on `NKM`, A
                    (broadcast over `N_inner`, the OUTERMOST inner mode, so every A value is reused
                    in the `N_inner=1` pass): `L=(1+1)*2=4` against `R=2`, clobber at W=1,2,3 alike.
                    B escapes because its broadcast axis `M_inner` is INNER to its level, so its
                    values die before the refill and the same `W=2` is safe.
      'capacity'    Lemma 3b's `W >= L`: the live peak at depth `d` exceeds this group's width.
      'assignment'  the width holds the values but the SHIFTED name collides with a live one
                    (`register_reuse_verdict` == clobber) — capacity is necessary, not sufficient.

    Pure; asks the same two questions `prefetch_steps_for`'s walk asks, in the same order, so the
    two cannot disagree about why a depth was refused.
```

## `Tensile/LoopModel/prefetch.py.prefetch_refusals` (function)

```
Every (op, group) whose PLR request the derivation could not honour.

    The whole-θ form of `prefetch_refusal`.  Empty tuple means PLR is honoured everywhere, which
    is the only state in which the emitted schedule matches the requested one.
```

## `Tensile/LoopModel/prefetch.py._readahead_depth` (function)

```
The prefetch depth for one read INSTRUCTION -- the MIN over the groups it fills.

    One Inst carries one advance. Compare `readahead_reach`, which takes the MAX for a
    different consumer. ADR 0004.
```

## `Tensile/LoopModel/prefetch.py._readahead_shift` (function)

```
The steady prefetch advance for `op`, as `(distance, strides, n_pres, extents)`.

    Built from the DERIVED depth, so a group at depth 0 gets distance 0. ADR 0004.
```

## `Tensile/LoopModel/prefetch.py.loads_in_place` (function)

```
True when the read-ahead refill writes the very name its own leaf's `wmma` is reading — the
    §2.6 `inplace` case, a WRITE-AFTER-READ, NOT a capacity violation.

    The advance lands on name `(P + shift) mod names(p)`; when `shift ≡ 0 (mod names(p))` that is
    name `P`, the one the co-located `wmma` consumes.  §2.6 (line 152) and Lemma 3c say the
    post-WAR peak of such a group is `1` and §5.3.1 pt4 MANDATES the rescuing order — emit the
    refill AFTER the vacating read — so the correct response is a σ_c deferral, not a smaller
    `dr_g`.  (Demoting instead is what cost WaveTile [1,1] its read-ahead: with a degenerate free
    fan `names(p)` collapses to `n_s` and the INNER=∅ advance of `n_s` wraps exactly, every time.)

    The ledger turns this into the `inplace-WAR` obligation whose await `sigma_c.move_reloads_after_last_use`
    keys on; `W_g == 1` is the other, already-handled way to land in the same case.

    THE SAME-LEAF CASE IS THE WHOLE RULE, and generalising it to "any later leaf of the trip" is a
    MISCOMPILE (measured 2026-08-28, reverted).  The reasoning that fails: §5.3.1 pt4 mandates the
    refill after the vacating read, so a consume at any later leaf ought to be rescuable by σ_c.
    But `move_reloads_after_last_use` defers WITHIN EACH NEST LEVEL, and an operand broadcast over an OUTER mode
    has its surviving consume in the next pass of that outer loop — rescuing it needs the read
    hoisted OUT of the loop, which is loop-invariant motion, not a deferral.  `shift % n_pres == 0`
    is exactly the case σ_c CAN rescue, which is why it is the condition.
```

## `Tensile/LoopModel/prefetch.py._read_placement` (function)

```
Symbolic register Placement for op's read: per group, slot = the mixed-radix rate-mode
    index mod W (`ring_slot`).  lo/hi carry their own Expr (lo rotates its rate ring, hi
    in-place=0) — the X00→X20 / X01 rotation kept ROLLED, and now over the CORRECT axis (the
    group's rate modes), not the innermost presence mode.

    `groups` (default: all) RESTRICTS the placement to a subset of the fragment's register
    groups.  An operand whose groups have DIFFERENT widths is read by more than one instruction
    (§2.6: the rotation width is per group, so the WAR that bounds how early a group may be
    refilled is per group too) — each carrying only the slots it writes.  The group LABEL is kept
    even for a single-group subset, so a split read still names which half it fills.

    READ-AHEAD (`dr` = the read hop's retime depth, the read hop's `delta`, = retime(read,
    K_inner, dr), §2.4): the read fetches the element `dr` K_inner-iterations AHEAD in the
    compute order (§2.1 `step = ⟨reduction-index, ord-strides⟩ − off`).  On the flat inner
    traversal (§2.5) that is a shift of `shift = dr · stride_ord(K_inner)` on the position
    `P = Σ stride_ord(m)·idx(m)`, so the read at `P` fetches `P+shift` — landing in reduction chunk
    `iter + (P+shift)//N_inner`.

    SHARED SOURCE BUFFER (`src_slot`): the ORD-DEPENDENT presence rollover `(reduction chunk + (P+shift)//
    N_pres) mod S_shared` over the read's PRESENCE modes — DERIVED from θ (reduction chunk from
    `outer_modes()`, strides from `inner_modes()` order, dr from `off`).  See the block at the
    src_slot code below for the full derivation and why reduction-only is wrong.

    REGISTER SLOT is on the same-axis `+dr` shift: a real VGPR *name* (a 1-1 index→name map with
    W=R today), so read and consuming wmma stay consistent for any order; only when W<R (register
    rotation, a register partition >1, task #118) must it also take the flat-ord shift — deferred there.
    dr=0 ⇒ no read-ahead.
```

## `Tensile/LoopModel/prefetch.py.resolve_quantum` (function)

```
Report the SUPPLIED per-hop quantum — `{operand: ((axis, factor), ...)}`.

    A reporting/validation step, not a derivation: `translate` fills `Hop.quantum` from the
    param dict (which the bridge fills from the target's layout+fold), and this only surfaces it
    for logging and tests.  `S` is accepted and unused so the decode call site reads the same as
    the other per-placement steps.
```

## `Tensile/LoopModel/prefetch.py._generation_map` (function)

```
`coord -> (shared generation, register slots)` for every point of `op`'s read presence.

    "GENERATION" HERE IS THE MULTIBUFFER INDEX — which of the `S_shared` ring buffers, i.e. WHICH
    REDUCTION CHUNK's data — and it is NOT the TDMSplit storage region.  Two region coordinates of
    one operand live in the SAME ring buffer at different offsets; what makes two coordinates differ
    here is the read-ahead carry `(P + shift) // N_pres` rolling one of them into the next chunk.
    The two were called "generation" interchangeably in an earlier round of this work and that
    conflation is worth naming: a region difference is harmless to a merge, a chunk difference is
    not.

    The generation is `_read_placement`'s own `src_slot` — `(chunk + (P + shift) // N_pres) mod
    S_shared` — evaluated per coordinate, NOT a second derivation of it.  That is the whole point:
    Precondition Q must be decided against the generation the decoder will actually assign, or the
    guard and the thing it guards can drift.  The register slots ride along because a merge must
    also not span two `Valu*_X*` bases, which has no encoding.
```

## `Tensile/LoopModel/prefetch.py.hop_quantum_axes` (function)

```
The axes one instruction of this hop covers, and by what factor (the movement quantum).

    Supplied by the adapter, never derived here -- theta is address-opaque. ADR 0004.
```

## `Tensile/LoopModel/prefetch.py.M` (function)

```
`M` at the REDUCTION CHUNK — the one level `build_ir` currently peels.  A coarser outer
        level's peel needs the `Loop(gtile)` that does not exist yet (#113); `peeled_levels()`
        reports every level that WANTS a peel, so the two cannot silently disagree.
```

## `Tensile/LoopModel/prefetch.py.copy_off` (function)

```
`{op_name: δ}` for the COPY op-classes at the reduction chunk — the staggered-peel input
        `build_ir` walks (`copy_p` present iff `off_p >= M-t` in the ramp, `off_p <= M-1-t` in the
        drain).
```

## `Tensile/LoopModel/prefetch.py:1` (comment)

```
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
```

## `Tensile/LoopModel/prefetch.py:119` (comment)

```
# §5.3.1 pt5 (rewritten): the coupling is `off(copy,iter) >= r`; the `+1` is demanded
# ONLY in the `S=δ`-tight copy-last case, where copy-first would overwrite a slot the
# current trip still reads.  (ADR 0007 #80)
```

## `Tensile/LoopModel/prefetch.py:213` (comment)

```
# THE SENSE IS PER OP-CLASS, THE ROLE IS PER HOP (#212 axis 3). Everything below reads
# `hop.role`, which is keyed on (src, dst) alone and therefore cannot tell a forward READ
# from a store round trip's read-BACK.  (ADR 0007 #79)
```

## `Tensile/LoopModel/prefetch.py:238` (comment)

```
# the substep read-ahead's REACH into whole chunks (§5.3.1: M is the max over all op-classes,
# with the read's SUBSTEP off converted to whole iters).
```

## `Tensile/LoopModel/prefetch.py:245` (comment)

```
# THE REACH IS DERIVED, NOT REQUESTED. `readahead_reach` runs the §2.6 clamp per group before
# converting to chunks; the old form here was `ceil(dr / n_substeps)` on the RAW request, which
# grew `M` around a read-ahead the steady body never issues (see `readahead_reach`).  (ADR 0007 #78)
```

## `Tensile/LoopModel/prefetch.py:347` (comment)

```
# LEMMA 3d's OUTER FACTOR IS A PRESENCE POINT, NOT A TILE (§5.3.1). "the first presence
# point is the whole first carrier group, all `q` of its `ρ`/`Φ` members, not tile 0 alone.  (ADR 0007 #77)
```

## `Tensile/LoopModel/prefetch.py:429` (comment)

```
# single-group: one depth for the whole operand, still DERIVED (a single-group fragment can be
# Shape B too — that is exactly the §2.6 term-(ii) operand of a K-innermost order, which has no
# width split at all.  (ADR 0007 #76)
```

## `Tensile/LoopModel/prefetch.py:478` (comment)

```
# CAPACITY FIRST (cheap, Lemma 3b's `W >= L`), THEN ASSIGNMENT (the actual admission test).
# Both are needed and neither implies the other: the count can fit while the modular name
# map still writes over a value consumed later.  See `register_reuse_verdict`.
```

## `Tensile/LoopModel/prefetch.py:579` (comment)

```
# THE RELOAD TRAVERSAL, not the presence set — see `reload_modes` for why this is a shared
# function rather than an inline filter, and for the status of the free-sibling exclusion.
```

## `Tensile/LoopModel/prefetch.py:582` (comment)

```
# PRESENCE POINTS, NOT TILES (§2.2 / §5.3.1). A quantum-folded axis contributes `N/q` group
# indices to this traversal, so the radix, `n_pres` and the drain-suppression bound `P + shift
# >= n_pres` all count carrier groups.  (ADR 0007 #69)
```

## `Tensile/LoopModel/prefetch.py:587` (comment)

```
# THE SAME LEVEL `prefetch_distance_for` ADVANCES ALONG (#321) — this guard and that conversion
# are two halves of one rule, so they must test the same axis.  (ADR 0007 #68)
```

## `Tensile/LoopModel/prefetch.py:594` (comment)

```
# INNER = the ord modes strictly INNER to the reduction substep — a property of `ord`, not of
# this operand (the paper splits the whole order, then intersects each factor with presence).  (ADR 0007 #67)
```

## `Tensile/LoopModel/prefetch.py:666` (comment)

```
# SHARED SOURCE BUFFER (`src_slot`): which LDS multibuffer generation the read's data
# lives in.  See ADR 0006.
```

## `Tensile/LoopModel/prefetch.py:671` (comment)

```
# the shared ring rotates over the reduction chunk.  DERIVED from θ — no literal, and the
# SAME accessor peel_depths/lds_buffers use, so the three can never disagree.
```

## `Tensile/LoopModel/prefetch.py:675` (comment)

```
# ONE shift for the whole read: the coord and the slot must advance together.
# See ADR 0006.
```

## `Tensile/LoopModel/presence.py.prefetch_axis_name` (function)

```
The level `off(read, ·)` lives on — `theta.readahead_level`, the OUTERMOST mode of the
    region's inner nest (#321).  Was the innermost REDUCTION mode, which is the same axis only
    when the reduction is outermost (`KMN`); see `readahead_level_of` for the measurements.
```

## `Tensile/LoopModel/presence.py.op_class_level` (function)

```
The LEVEL an op-class is emitted at — its HOME level (§5.3.1 line 523, "placed by ITS OWN
    presence set").  The second input the cross-level boundary term needs: an offset at a level
    OUTER to an op-class's home is what relocates its instances across the enclosing boundary.

      read  — the innermost mode of its presence set; the reduction chunk when presence is empty.
      copy  — the region mode when the movement is region-split (one level deeper, per-region),
              else the reduction chunk (a COARSE copy has empty inner presence).
      store — the innermost of its presence set, like a read (#116; no store op-class exists yet).

    Returns None for an op-class with no level (a legacy θ carrying no outer mode).
```

## `Tensile/LoopModel/presence.py.group_value_span` (function)

```
The `grouping_mode` VALUE SUBSET a register group owns (§2.6: a group is a partition of the
    grouping tile-mode's values).  Returns (mode_name, [values]) or (None, None) if the fragment is
    not group-split.  EQUAL partition (current): group i owns [i·part_vals, (i+1)·part_vals).
```

## `Tensile/LoopModel/presence.py.varying_axes` (function)

```
The inner modes op's shared→register READ varies over (its presence set, §2.1) — the
    non-broadcast inner modes.  The read is emitted at the INNERMOST of these (its loop level).

    NOTE presence is a PER-HOP fact (§2.8 move 1 "applies per hop independently"), and this
    helper answers it for the READ hop only.  A COARSE copy is present on no inner mode and sits
    at the reduction-chunk level; a REGION-SPLIT copy has the region mode in ITS presence set and
    sits one level deeper, per-region (§5.3.1 line 523) — `emit.copy_insts` places it there by
    rolling it under a `Loop(region_mode)`.  Do not read this as "every copy sits at the chunk
    level"; that shortcut is exactly what §5.3.1 line 523 warns against.

    THE MOVEMENT QUANTUM IS SUBTRACTED HERE (§2.2, Precondition Q).  An axis the read's single
    instruction SPANS is absent from that hop's presence — "the coordinates inside one quantum are
    one presence point" — so the read varies once per INSTRUCTION, not once per tile, and the read's
    tree level, coord, rate modes and act count all follow with no second rule.

    THIS IS HALF OF A PAIR AND MUST NOT BE APPLIED ALONE.  Dropping the axis here removes it from
    `Load.coord`; `ir.Load.quantum` then states how many coordinates the one instruction fills, and
    `loopir_to_gir._quantum_coords` re-inserts the axis to produce that many register `Ref`s — one
    instruction, several definitions.  Apply the subtraction without the expansion and every
    non-leader consumer loses its reaching def (measured: 34 tests, "32 consumer(s) read a location
    the arm never writes"); apply the expansion without the subtraction and the axis appears twice
    in one coord (measured: 492 tests).  `resolve_quantum` must have run first; while the axes are
    empty this is the historical per-tile answer exactly.
```

## `Tensile/LoopModel/presence.py.reload_modes` (function)

```
The modes the prefetch advances through: presence minus the free-axis region siblings.

    A sibling is a storage-disjoint group, so the shift must not carry through it; it
    BOUNDS the shift, which then rolls the chunk. ADR 0004.
```

## `Tensile/LoopModel/presence.py.sibling_free_axes` (function)

```
`op`'s region modes that ENUMERATE storage-disjoint siblings rather than reload — the
    free-axis ones (§5.3.1 pt 5).

    A REDUCTION-axis region split is NOT one of these: the paper keeps it in the traversal
    ("Reduction axes, and a genuinely-reloaded free fan, ARE reload modes and stay"), because its
    two regions are successive SUBSTEPS of one tile held in one group — the same reasoning
    `theta._rate_broadcast` already applies when it excludes only the FREE region modes from the
    rate.  Dropping reduction siblings too breaks the two-axis `K_split x K_inner` rotation (#254):
    measured 12 unit failures, all on that rotation.

    ANCHORED ON THE SPLIT, NOT ON THE MODE NAME (#245, 2026-08-22).  `region_modes` answers this
    correctly today, but it answers by NAMING a mode, so it is only as durable as that mode's
    presence in `ord`.  A region axis whose extent collapses to 1 is dropped by
    `_canonical_ord._keep`, which empties `region_modes` and SILENTLY RELEASES this veto -- and
    with it #274's `dr_g = 0`, so the read-ahead carry returns.  MEASURED when that happened: at
    `KMNMNK x TDMSplit1` MXSA's four acts split across blocks, drain0 emitted ZERO scale loads and
    the prologue fetched a tile a stage early; re-imposing the veto on the renamed axis took the
    violations 2 -> 0.  The split is a fact about STORAGE, so the veto keys on `op.free_split`.

    THE FALLBACK IS UNREACHABLE ON THE SHIPPING TREE and is here for the reassignment #245 needs:
    while the region mode survives in `ord`, `region_modes` is non-empty and the first branch
    returns exactly what it always did -- byte-identical emission, verified.  Only when a split
    operand has NO surviving free region mode does the second branch name the mode that now carries
    its free tiles instead, so the bound follows the TILES rather than the label.
```

## `Tensile/LoopModel/presence.py._sibling_outer_to_reduction` (function)

```
Is one of `op`'s free siblings OUTER, in `ord`, to the reduction it rotates over?

    This is the configuration in which the group cannot read ahead — see `_readahead_depth`.
```

## `Tensile/LoopModel/presence.py.axis_is_live` (function)

```
Does this ord block contain any axis with more than one value?

    "INNER = ∅" in §5.3.1 pt5 is a statement about the TRAVERSAL, not about the mode list: a block
    of extent-1 modes is traversed once, exactly as an empty block is, so it must take the same
    branch.  Testing `not inner_block` instead answers a question about how θ happened to FACTOR
    the axes.
```

## `Tensile/LoopModel/presence.py.prefetch_axis_mode` (function)

```
The reduction SUBSTEP axis: the innermost reduction inner mode WITH MORE THAN ONE VALUE.

    The peel (`_readahead_coords`) and the steady advance (`_readahead_shift`) both split `ord` at
    this axis, and both used to take the innermost reduction mode outright.  That is the same axis
    whenever the reduction is one mode — but a DU `TDMSplit` re-factors it as `K_split x K_inner`,
    and `K_inner` can collapse to extent 1 while remaining INNERMOST in ord.  The split then
    happened at an axis with no values.

    MEASURED 2026-08-18, `TDMSplitA/B = 2` + `PrefetchLocalRead=1` under the K-INNERMOST orders
    (`LOMNK`, `LONMK`) — 192 of 1152, and exactly those two words.  `n_s` came out 1, so the
    read-ahead prologue primed ONE substep where the steady body consumes two: the first
    `wmma u=1` read a register generation nothing had loaded.  K-outer orders were unaffected
    because there the factor is `min(dr, n_s)` and `dr = 1` clamped it to the same 1 either way —
    which is why the defect selected on the loop order and on nothing else.

    NOW THE READ-AHEAD LEVEL (#321), because that is the axis the advance walks and this docstring's
    own invariant is that the peel and the steady advance split `ord` at ONE place.  They stopped
    agreeing when the advance moved: at `DepthU=64` there is no reduction inner mode at all, so this
    returned None, `preloaded_tiles` bailed at its `sub is None` guard, and the prologue was
    empty while the steady body shifted by 1 — a steady advance with nothing priming it, which is
    the Lemma 3d violation this function exists to prevent.  The ">1 values" fallback above is not
    lost: `readahead_level_of` carries it (a collapsed leading axis is skipped there).
```

## `Tensile/LoopModel/presence.py.ring_axes` (function)

```
The inner modes one register group cycles distinct values over, in ord order.

    Extents are in PRESENCE POINTS, not tiles: a folded axis contributes N/q generations.
    Extent-1 modes are dropped as nullspace. ADR 0004.
```

## `Tensile/LoopModel/presence.py.ring_slot` (function)

```
The rotation slot Expr for a group of width `d`: the MIXED-RADIX index over the group's
    rate modes (innermost = fastest, coef 1; next = ×extent; …) taken mod `d`.  d≤1 → cst(0)
    (in-place / single buffer, no rotation).

    THE RADIX IS IN PRESENCE POINTS AND SO IS THE DIGIT (§2.2, #312).  `ring_axes` now reports
    the folded extent `N/q`, so the coefficient chain counts generations; the loop variable is
    still a tile index, so a folded mode contributes `var // q` — the per-term divisor
    `position_terms` supplies.  This is the UNSHIFTED peer of `_shifted_rate_slot`, and the two must
    agree digit for digit or the read-ahead's producing slot and the wmma's consuming slot name
    different registers; sharing the radix source and the divisor is what makes that hold rather
    than something to re-check.
```

## `Tensile/LoopModel/presence.py._reduction_rate_mode` (function)

```
The single innermost REDUCTION rate mode (role 'k') of `op`'s register group, as
    (name, extent), or None.  This is the substep axis the PLR read-ahead shifts + wraps into
    the reduction-chunk (`iter`) loop.  Phase 3 has exactly one (K_inner); a group with several reduction
    rate modes would need a mixed-radix flat shift (deferred — guarded at the call site).
```

## `Tensile/LoopModel/presence.py.axis_strides` (function)

```
`stride_ord(m)` for a set of inner modes: the product of extents of the modes INNER to
    `m` in `ord`, restricted to `modes` (§2.5 — `ord` is a permutation, so the traversal over
    that mode subset is a monotone mixed-radix counter; the innermost has stride 1).  Returns
    (strides, N) where N = Π extents over `modes`.

    `modes` defaults to ALL inner modes, but a read's reduction chunk position must be computed over its
    PRESENCE modes ONLY (§2.1): a broadcast mode carries identical data across its values, so it
    must NOT advance the reduction chunk (over-counting it inflates the flat position and bounces the buffer
    once its extent > 1 — the WT>1 failure).  Pass the read's presence-mode names to exclude
    broadcast modes.

    `extents` overrides a mode's extent for this traversal.  The one caller that passes it is the
    read-ahead shift, and what it passes is the READ HOP's presence extents: §2.2's movement quantum
    folds an axis of extent `N` into `N/q` carrier groups, so a partially-folded axis contributes
    `N/q` PRESENCE POINTS to the traversal, not `N` tiles (§5.3.1 — "the shift, the position `P`,
    and the drain-suppression bound `P + shift >= n_pres` all count in presence points").  The fold
    itself is derived once, in `geometry.hop_fold`, and arrives here through
    `theta.presence_modes(op, hop)`; this parameter is how it reaches the radix, NOT a second place
    that decides it.
```

## `Tensile/LoopModel/presence.py.hop_extents` (function)

```
`{mode: extent}` as the operand's READ HOP sees them — its presence extents (§2.2).

    Equal to the `ord` extents except on an axis the hop's quantum folds only in part, where it is
    `N/q` (the carrier-group count).  ONE derivation: `theta.presence_modes(op, hop)` applies
    `geometry.hop_fold`, and everything that counts presence points reads the result from here.
```

## `Tensile/LoopModel/presence.py.read_fold` (function)

```
`{mode: q}` — the PARTIAL quantum factor of the operand's read hop (§2.2), or `{}`.

    THE COMPANION OF `hop_extents`, and the reason both live here.  `hop_extents` gives the axis in
    PRESENCE POINTS (`N/q` carrier groups) and this gives the `q` that converts a point back to the
    TILE that names it — the group's leader, `g·q`.  Every consumer needs exactly one of the two:
    the read-ahead arithmetic counts points (`_readahead_shift`, `axis_strides`, the NLL bound),
    while anything that emits or guards a COORDINATE needs the tile (`_shifted_coord`'s scale,
    `emit._first_touch_modes`' `mode % q == 0` modulus).

    ONE DERIVATION, TWO READERS.  `emit._first_touch_modes` used to re-find the read hop and call
    `geometry.hop_fold` itself; with the shift now also needing `q`, that would be two places
    deriving one fact — the failure shape this codebase hits most (see
    [[pattern-two-derivations-one-rule]]).  They agree on the unfolded case and would diverge
    exactly where it matters.
```

## `Tensile/LoopModel/presence.py.broadcast_width` (function)

```
How many times `op`'s whole register-name space is re-traversed before any value dies.

    CAPACITY (section 2.6 term ii), asked over presence. The advance-unit question is
    `reloads_whole_set`, over reload modes. ADR 0004.
```

## `Tensile/LoopModel/presence.py.position_terms` (function)

```
`(terms, L)` for the FLAT PRESENCE POSITION `P = Σ stride(m)·point(m)`, as an `Expr` linear
    form over the LOOP VARIABLES.

    THE UNITS DO NOT MATCH, AND THAT IS THE WHOLE PROBLEM (#312).  `strides` counts PRESENCE POINTS
    (`axis_strides` over `hop_extents`, so a `q`-folded axis contributes `N/q`), but the loop
    variable the expression reads is a TILE index — the nest is unrolled over `ord` extents, which
    a per-operand fold cannot narrow (the same `M_inner` loop feeds A's folded read AND the wmma's
    unfolded `M` index).  So mode `m`'s contribution is `stride(m) · (var(m) // q_m)`, which is
    exactly `ir.Expr`'s per-term divisor (see `ir._term`).

    ONE BUILDER, FOUR READERS.  `_shifted_coord`, `_shifted_rate_slot`, `ring_slot` and
    `_read_placement`'s `src_slot` carry all built this same linear form; three of them had it
    inline and the fourth was missed for a full cycle (the `src_slot` carry, whose mixed units
    crossed reduction chunks `q`-fold early).  They call this instead.
```

## `Tensile/LoopModel/presence.py._shifted_coord` (function)

```
`op`'s read coord with the prefetch applied, per presence mode.

    The advance carries OUT of the reduction axis into the free modes: leaf 0's prefetch
    targets the next tile, not the next substep of this one. ADR 0004.
```

## `Tensile/LoopModel/presence.py._shifted_rate_slot` (function)

```
The rotation slot the prefetched read writes, as an Expr.

    Shares its radix and divisor with `ring_slot`, or the producing and consuming slots
    would name different registers. ADR 0004.
```

## `Tensile/LoopModel/presence.py.quantum_tile_cap` (function)

```
How many of `op`'s free tiles ONE instruction of `hop` could carry, from the TARGET.

    `hop.vector_elems` is the instruction's element payload per lane and `op.frag_elems` is one
    (free-tile, substep) fragment's, so their ratio is the tile count — the same two numbers
    `geometry.load_regs`/`plan_transfers` already divide, read here for the axis question instead of
    the instruction-count one.  This is the *domain* the target offers, exactly as the register
    budget is the domain `S` is searched in (§2.6); which of it `ord` actually permits is
    `hop_quantum_axes` below.  A hop that states no width (`vector_elems == 0`) means "whole
    fragment", i.e. one tile.
```

## `Tensile/LoopModel/presence.py.is_uniform_over` (function)

```
Is the generation map constant WITHIN each quantum block of `axes`?

    `axes` is `((name, factor), ...)`.  A coordinate's block along a partially-spanned axis is
    `v // factor`, so two coordinates share an instruction iff they agree on every un-spanned axis
    AND on the block index of every spanned one.  With `factor == extent` the block index is
    always 0 and this degenerates to "the axis is ignored", which is the fully-absorbed case.

    This is Precondition Q's "no quantum straddles a generation boundary" read literally off the
    decoder's own map — and reading it per BLOCK rather than per axis is what lets an extent-4 axis
    merge in pairs instead of being refused outright.
```

## `Tensile/LoopModel/presence.py:1` (comment)

```
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
```

## `Tensile/LoopModel/presence.py:89` (comment)

```
# ONE DERIVATION, asked of the read hop. `geometry.hop_broadcast` already answers "what does
# ONE INSTANCE of this hop not vary over", and since 2026-08-20 that includes the axes the
# instruction SPANS (`geometry.quantum_axes`).  (ADR 0007 #75)
```

## `Tensile/LoopModel/presence.py:227` (comment)

```
# radix chain over the rate modes, innermost fastest — the same order `_shifted_rate_slot`
# builds its digit fields in, read from the same `ring_axes` extents.
```

## `Tensile/LoopModel/registers.py <module>` (module)

```
The register ring: when each name is loaded, when it is read, and how wide the ring
must be to keep those apart.

`_register_timeline` replays the traversal; `_uses_before_load` and
`_overwrites_live_register` are the two ways a width can be wrong; `reload_positions`
moves a reload until neither fires.
```

## `Tensile/LoopModel/registers.py.chunks_crossed` (function)

```
`r` — how many reduction CHUNKS ahead `op`'s substep read-ahead reaches (§5.3.1 pt5,
    "The chunk-crossing coupling").

        fan  = product of extent over pres(op) modes INNER to the reduction substep
        span = n_s * fan                                    (distinct reads in one chunk)
        r    = ((span - 1) + dr_g * fan) // span

    The substep read-ahead is NOT confined to its own chunk: once the reduction substep is outer to
    the operand's inner free fan (or the fan is exhausted), the top `shift` positions have nowhere
    sideways to carry and roll the chunk coordinate.  The paper is explicit that "the crossing is
    the rule, not a special case" — so `src_slot`'s carry out of the presence traversal into `iter`
    is CORRECT, and what the crossing costs is shared buffering, not a different slot formula.

    `dr_g = 0` (a Shape-B in-place group) does no look-ahead and so reaches `r = 0`.
```

## `Tensile/LoopModel/registers.py.prefetch_distance_for` (function)

```
The FLAT-POSITION advance a per-group depth `dr_g` produces — the ONE conversion (§5.3.1 pt5).

    `dr_g` counts REDUCTION SUBSTEPS; every consumer of the advance (the shifted coordinate, the
    rate slot, the NLL bound, and the admission walk) counts FLAT PRESENCE POSITIONS.  The
    conversion is NOT `dr_g x stride`: `substep_factor` is the FULL `n_s` when the ord INNER block
    is empty and `min(dr_g, n_s)` otherwise, and the strides are over FOLDED extents
    (`hop_extents`, §2.2) because the traversal counts presence points, not tiles.

    EXTRACTED so `_readahead_shift` and `prefetch_steps_for`'s admission test cannot disagree.
    They did: the first cut of the admission test used a bare `d x stride` on UNFOLDED extents,
    which reported a clobber for a `ReadPhi>1` fold and silently zeroed `dr_g` — the two-derivations
    bug on the very quantity the admission test exists to police.
```

## `Tensile/LoopModel/registers.py.reloads_whole_set` (function)

```
Is every one of `op`'s registers live across a full outer pass, so all must reload at once?

    Asked over `reload_modes`, NOT presence: this is the ADVANCE UNIT, while
    `broadcast_width` is CAPACITY over presence. Same shape, two spaces, two questions.
    Collapsing them cost 96 miscomparing cells. ADR 0004.
```

## `Tensile/LoopModel/registers.py.RegisterTimeline` (class)

```
The read/write event trace of one group's rotation names — the ONE walk (#331).

    `events` maps a register NAME to `[(t, 'r'|'w', value)]` over three chunks of the `ord`
    traversal; `steps` is the traversal itself and `T` its length, so `steps[t % T]` is the
    coordinate of event `t`.  Both the ASSIGNMENT VERDICT and the ANCHOR read this, because they
    are two questions about one trace: "does the refill land on a live name" and "after which leaf
    does it stop landing on one".  Deriving them separately is what let a verdict say `safe` while
    the placement stayed put — the 2026-08-28 miscompile.
```

## `Tensile/LoopModel/registers.py._register_timeline` (function)

```
Build the `RegisterTimeline` for `(op, group)` at rotation width `W` and advance `shift`.

    `at` is the ANCHOR — `{invariant mode: value}`, the first-touch guard the emitter will put on
    this read.  `None`/`{}` walks the UNANCHORED schedule (the read fires at each name's first
    touch, guard value 0), which is the schedule `reload_positions` derives an anchor FROM.  A
    non-empty `at` walks the schedule that anchor PRODUCES: the write is issued only at steps whose
    coordinate matches the guard, so every name's refill moves with the Inst.

    BOTH WALKS ARE NEEDED AND THEY ARE NOT THE SAME SCHEDULE (#332).  Deriving an anchor from the
    unanchored walk and emitting it without re-walking is a miscompile: MEASURED, `MNK` +
    `TDMSplitA=1, TDMSplitB=2`, A at shift 1 — unanchored the refills sit at `N_inner=0`, the walk
    picks anchor `N_inner=1`, and at THAT position the `K_splitB=0` leaf refills the slot the
    `K_splitB=1` leaf is still about to consume.  The anchor has to be a FIXPOINT of the walk that
    chose it, which is what `reload_positions` now checks.

    `None` when the operand has no presence traversal or does not advance — there is no rotation to
    trace, and every caller's answer in that case is the trivial one.
```

## `Tensile/LoopModel/registers.py._uses_before_load` (function)

```
`{name: [(t, wanted, held)]}` — every STEADY-chunk consume that finds the wrong generation
    in its slot.  A straight simulation of the walk: replay the events in order, carry the last
    value written to each name, and flag a read whose wanted value is not the one held.

    THIS IS THE COMPLETE TEST AND `_overwrites_live_register` IS NOT (#332).  A conflict is "this write is
    premature" — it looks BACKWARD from a write to a consumer of the value it destroys.  That misses
    the dual failure, "this read is starved": a slot that is simply never refreshed before its
    consumer.  The two coincide on the DEFAULT schedule, where the only way to starve a read is to
    have overwritten it, which is why the conflict scan carried the model this far.  They come apart
    the moment the read MOVES: anchoring `MNK`+MT-split's A at `N_inner=1` makes the `K_splitB=0`
    leaf refill slot 1 with the CURRENT chunk (a redundant rewrite, no conflict at all), and the
    next trip's `K_splitB=1` leaf then consumes a generation nobody ever loaded.  No write was
    premature; a read was starved.  The GIR detector saw it, the model did not.

    At equal `t` the read sorts before the write (`'r' < 'w'`), which is the emitted order for an
    in-place refill: the leaf consumes the old value, then overwrites it.  So a §2.6 in-place WAR is
    not stale, and only a genuinely missing generation is reported.
```

## `Tensile/LoopModel/registers.py._overwrites_live_register` (function)

```
`{name: [last consume, ...]}` for every steady-chunk write that lands on a name whose
    previous value is STILL CONSUMED at or after the write — the raw finding both the verdict and
    the anchor read.

    ONE SCAN, TWO QUESTIONS.  `register_reuse_verdict` asks "is this position wrong" and
    `reload_positions` asks "which position would be right"; before this they each re-implemented
    the scan and drifted apart — the verdict returned `clobber` for a name whose anchor was
    non-`None`, and `prefetch_steps_for` reads the anchor, so the operand was admitted at a
    depth the verdict had already rejected.  Same events in, same conflicts out.
```

## `Tensile/LoopModel/registers.py.register_reuse_verdict` (function)

```
`safe` | `inplace` | `clobber` — the ASSIGNMENT test for a read-ahead of `shift` (§2.6).

    `at` scores the ANCHORED schedule instead of the default one (see `_register_timeline`).

    THE ADMISSION TEST IS ASSIGNMENT, NOT CAPACITY, and conflating them is what made
    `max{d : L(d) <= W}` admit a clobber.  `group_live_peak` answers "how many generations are
    simultaneously live" — Lemma 3b's `W >= L`, a COUNT.  This answers "does the write LAND on a
    slot whose value is still needed" — the modular name map.  At `W = 2, L = 2` the count fits and
    the map can still collide, so `L <= W` is NECESSARY, NOT SUFFICIENT.  Measured before this
    existed: `WT[1,1]/split PLR1 MKN B:g0 advance 1 clobbers a live register` while `L(1)=2 <= W=2`.

    THE WALK.  Over three chunks of the `ord` traversal, at each step the read writes name
    `(P+shift) mod W` and the leaf's wmma consumes name `P`.  A write onto a name whose previous
    value still has a consume STRICTLY LATER is the unrescuable Lemma 3b race (`clobber`); one whose
    only surviving consume is at the SAME leaf is the rescuable §2.6 in-place WAR (`inplace`).  Only
    the STEADY (second) chunk is judged, so the prologue's partial fill is not mistaken for a race.

    WHY THIS IS A WALK AND NOT A FORMULA: §2.6's Q37 says term (ii)'s broadcast-hold is a PRODUCT
    over the pinned rotation modes, "not a `+1`", and that how it combines with a depth-`dr`
    read-ahead "is not given a closed form: it is `ord`-dependent and computed by the walk".  The
    two constants this replaces (`fan > 1 -> 0`, and `L = dr_g + 2`) were that missing closed form
    guessed, then frozen at the one width we ship.

    ONE AUTHORITY, INDEPENDENT ORACLE ELSEWHERE — the same split `agent_served_modes` uses: the
    library holds this, and the checks that keep it honest are the ones that walk the EMITTED acts
    (`check_plan` / `verify_dataflow` on the lowered GIR), which is a genuinely different derivation
    rather than a copy of this one.
```

## `Tensile/LoopModel/registers.py.reload_positions` (function)

```
`{register name: coord}` -- the loop position each reload must follow, or None if there
    is no legal one (which is the `clobber` verdict).

    Checked as a FIXPOINT: anchoring moves every name's reload, so the proposal is
    re-walked against the schedule it creates. ADR 0004.
```

## `Tensile/LoopModel/registers.py:1` (comment)

```
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
```

## `Tensile/LoopModel/registers.py:103` (comment)

```
# AN EXTENT-1 MODE IS NULLSPACE, the same rule `ring_axes` states: it has one value, so the
# traversal neither repeats under it nor advances through it, and it cannot make an outer mode
# "not outer".  (ADR 0007 #72)
```

## `Tensile/LoopModel/registers.py:177` (comment)

```
# THE ANCHOR IS A GUARD ON THE ISSUING STEP, so it filters WRITES only — the wmmas keep
# consuming at every step regardless of where the refill was moved to.
```

## `Tensile/LoopModel/registers.py:279` (comment)

```
# STARVATION IS THE AUTHORITY (#332). `_uses_before_load` is the complete correctness test —
# every steady consume finds the generation it wants — and it strictly contains the premature-
# write case below, which cannot starve a read without also leaving one stale.  (ADR 0007 #71)
```

## `Tensile/LoopModel/registers.py:308` (comment)

```
# THE ANCHOR MUST LIE IN THE SAME TRIP AS THE REFILL.  A consume that survives the back
# edge cannot be preceded by any in-body position, so there is NO legal placement and
# the answer is the clobber the verdict already gives.  This is the admission test.
```

## `Tensile/LoopModel/registers.py:324` (comment)

```
# THE FIXPOINT. `at` is the guard the emitter will actually put on this read
# (`emit._read_anchor` builds the same dict from this return value).  (ADR 0007 #70)
```

## `Tensile/LoopModel/render.py <module>` (module)

```
Rendering — human-readable DEBUG VIEWS of the IR (not the IR itself).

The DECODER core carries no TensileLite / AMD vocabulary; THIS FILE IS THE ONE EXEMPTION, and the
header used to claim otherwise.  `_VERB`/`_SPACE` and `_mma_line` spell the abstract hops as
`tdm` / `ds_read` / `vmem_ld` / `vgpr` / `wmma` — gfx1250 mnemonics — on every rendered line, on
purpose: the dump is for humans reading gfx1250 assembly beside it.  The claim mattered because it
is exactly the invariant #163 enforces elsewhere, so stating it here made this file look audited
against a rule it deliberately breaks.  Three stream views (render_stream picks):
  render_ir(ir)        — the ROLLED θ-nest: loop headers, one Inst per op-class (the stored IR)
  render_unrolled(ir)  — expand the leaves into the interleaved read;wmma stream
  render_ir_raw(ir)    — raw dataclass dump, for debugging the emitter
The `--unroll` / `--raw` names below are `render_stream` KEYWORD ARGUMENTS, not CLI flags: the only
command-line switch is `--output-loop-ir`, a bare store_true with no view selector.  They were
written as flags here and in the rolled dump's own header line, which told every reader to pass an
option that does not exist.
plus render_geometry(theta) — the derived sizing table (load sizes, mma grid) so every number
can be checked by hand.  (No collapse view — the rolled IR is already collapsed; see
render_stream + ir.py "Re-adding address support".)
```

## `Tensile/LoopModel/render.py._loop_header` (function)

```
The loop header text.  OUTER (runtime, pipelined) loop: `for <iter> while (<back-edge pred>)`
    — names the induction AND shows the structured back-edge predicate.  INNER (static) loop:
    `for <mode> in range(N)`.  `node.trip` is a structured `Pred` for the outer loop (read by
    field, no string) or an int repeat count for an inner loop.
```

## `Tensile/LoopModel/render.py._subbody_header` (function)

```
The header for ONE sub-body of a §5.1 multi-body loop, over its OWN half-open range.

    A multi-body loop is a SEQUENCE of sub-loops, not one loop that runs every body per trip:
    `ranged_bodies()` pairs body `i` with `[lo_i, hi_i)`, and both the unrolled view and the GIR
    lowering walk it that way.  Printing one `for m in range(trip):` header above all the bodies
    (what this used to do) therefore reads as `trip × (bodyA + bodyB)` — it over-counts the level
    by the number of bodies, and when the ranges are singletons (the first-touch peel `[0,1)`,
    `[1,trip)` at trip=2) the level is FULLY UNROLLED and there is no loop left to head at all.

    So the header comes from the sub-body's own range: a singleton range is a BINDING of the mode
    to that value (no loop), anything wider is a real loop over the sub-range.
```

## `Tensile/LoopModel/render.py._coord_of` (function)

```
The op's coord as text: symbolic mode names in the rolled view, concrete indices in the
    unrolled view (env supplies mode → value).  `op.coord` is ((mode, None|val|Expr), …).  An
    Expr value is a SHIFTED coord (the PLR read-ahead: substep k renders/evals as (k+dr)%S).
```

## `Tensile/LoopModel/render.py._shape_str` (function)

```
The §2.2 MOVEMENT QUANTUM and the §5.3.1 pt5 read-ahead ADVANCE, neither of which was
    printed anywhere.

    `quantum` decides how many TILES one instruction carries.  With it invisible, a read that moves
    a 2-tile quantum renders exactly like a 1-tile read — same coord, same `size_regs` (which is
    the per-FRAGMENT width, not the movement's) — so the one line that would have shown the merge
    shows nothing.  That is the fact `covered_coords` re-inserts on the GIR side as `spans`, and
    every #264/#267 investigation had to reconstruct it from the checker instead of reading it.

    `advance` is the PREFETCH/INPLACE shape: 0 means the read issues in place one line before its
    own consumer, non-zero means it is hoisted.  `render_geometry` prints the per-group `dr_g` that
    DERIVES it, but the per-instruction answer — which is what `sigma_c` actually branches on — was
    only inferable from where the line happened to sit.
```

## `Tensile/LoopModel/render.py._await_line` (function)

```
A named-dependency discharge point (§5.4): the NAMED dep is the primitive; the lowered
    count (if resolved) shows in parens as a debug aid.

    `kind` IS PRINTED, because it is what decides ISSUE ORDER and not merely how the wait reads.
    A `*-WAR` await means the owning instance must be issued AFTER the readers it names (§2.4 line
    99) — that is the σ_c refinement that places refill copies last.  Rendering every await
    identically made a correctly-ordered body and a wrongly-ordered one produce the same dump, and
    the kinds are a CLOSED set (`ir.py` RAW_KINDS / WAR_KINDS, #276 S1), so an unrecognised one is
    a defect and is shown as such rather than being quietly abbreviated to its dep name.
```

## `Tensile/LoopModel/render.py.render_stream` (function)

```
The ONE IR view both entry points call.  raw > unroll > rolled.  `unroll` expands the
    WaveTile/region leaves into the interleaved read;wmma stream (a VIEW of the rolled IR).

    There is no `collapse` view: the rolled IR is ALREADY the collapsed form (one Inst per
    op-class, §5.2).  A `--collapse` folder existed in the layout-algebra era, when the IR
    enumerated one Inst per thread/address instance and identical ones were folded to `(inst)
    ×N`; trimming the layout algebra (addresses are opaque, §2.1/§7) removed that enumeration,
    so there is nothing left to fold.  See the "Re-adding address support" note in ir.py.
```

## `Tensile/LoopModel/render.py.render_ir` (function)

```
Print the ROLLED θ-nest: `for <mode> in range(N):` loop headers, each op-class ONE line at
    its presence level with its symbolic slot, and named Await points.  Nothing enumerated — this
    is the exact-θ IR.

    TWO THINGS THIS USED TO ADVERTISE AND NEVER EMITS.  The slot example was written
    `vgpr[lo=buf(min)%2]`; the string `vgpr[` appears in no dump the renderer produces (a register
    destination renders through `_reg_slots` as `register[buf...]`).  And "Branch arms for register
    rotation" describes a node the emitter no longer builds — `Branch` handling is kept below for
    an IR that still contains one, but no `build_ir` output does.  Both sent a reader looking for
    output that cannot appear, which reads as a broken renderer rather than a stale docstring.
```

## `Tensile/LoopModel/render.py.render_unrolled` (function)

```
Expand the rolled nest into the concrete interleaved stream: walk the loop nest with a
    live environment {mode: index}, and at each innermost leaf emit that step's fresh reads
    THEN its wmma (consume-then-refill), evaluating the symbolic slot Exprs to concrete buffers.
    Enumeration lives ONLY here; a step's read and wmma stay adjacent (correct unroll).
```

## `Tensile/LoopModel/render.py._region_and_agent_lines` (function)

```
The REGION and AGENT facts, which no other line of this table carries (#287).

    Three θ fields decide how a split operand's reads and copies must be read, and all three were
    unprintable:

      `Operand.free_split`  — the ord-INDEPENDENT anchor for the §5.3.1 pt5 sibling bound.  It
          exists precisely because the sibling veto used to be recovered from `region_modes`, which
          drops out of `ord` in the degenerate shapes; a reader could not tell the two apart.
      `Hop.region_agent_relative` — PER HOP, and that is the whole point: the region is the AGENT's
          on the read hop and the COORDINATE's on the copy hop of the SAME operand.  Printing one
          per-operand answer would re-introduce the conflation this field exists to remove, so the
          hops are listed separately whenever they disagree.
      `theta.agents`        — how many agents cooperate on one tile.  With it unprinted, a
          single-wave dump and a four-wave dump of the same shape are indistinguishable, and the
          cross-agent obligations only the latter has look like noise.

    Cost on record: `translate.py:519` justified "the read's region needs ρ" with a checker message
    about region coverage.  Nothing in any dump said the operand was agent-relative, so the wrong
    premise stood for a day.  Emitted only when there is something to say.
```

## `Tensile/LoopModel/render.py.render_schedule` (function)

```
One row per (operand, register group): the whole policy, checkable by hand.

    `ahead = 0` beside a non-empty `refused` is a request the derivation could not meet -- the
    kernel is named for a prefetch it does not perform.
    
```

## `Tensile/LoopModel/render.py:1` (comment)

```
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
```

## `Tensile/LoopModel/render.py:38` (comment)

```
# Per-op formatting.
# ===========================================================================
```

## `Tensile/LoopModel/render.py:80` (comment)

```
# A Cond's MEANING is its GENERIC `kind` (backend-agnostic paper vocabulary; the core sets this,
# NOT a TensileLite scaffold name).  The renderer annotates each guard with what it IS, so the dump
# reads as standalone GEMM pseudocode without opening GIR or knowing any scaffold labels.
```

## `Tensile/LoopModel/render.py:112` (comment)

```
# `{ax}=(formula)` — the DATA index this read accesses along `ax`, a read-ahead SHIFT of
# the loop var (e.g. `K_inner=(K_inner+1)%2`).  The `=` disambiguates it from an array
# subscript (the old `K_inner[(K_inner+1)%2]` read like an array-of-array).
```

## `Tensile/LoopModel/render.py:172` (comment)

```
# the wmma CONSUMES a per-operand register residence (a {operand: Placement} dict, same slots the
# producing ds_read deposited).  Show each source's register group+slot so the dump names the
# vgpr the wmma reads — matching the ds_read dest (user point #1: wmma source must carry the group).
```

## `Tensile/LoopModel/render.py:182` (comment)

```
# THE OPERAND NAMES COME OFF THE NODE. This line hardcoded `C[...] += A*B`, so `Mma.a`, `Mma.b`
# and `Mma.acc` were unverifiable from every view except --raw — and a decoder that named the
# wrong operand rendered identically to one that named the right one.  (ADR 0007 #84)
```

## `Tensile/LoopModel/render.py:217` (comment)

```
# Raw dump — the actual IR dataclasses, no interpretation by the renderer.
# ===========================================================================
```

## `Tensile/LoopModel/render.py:276` (comment)

```
# ROLLED view — the IR as it is stored: a loop nest, each op-class once (§5.2).
# ===========================================================================
```

## `Tensile/LoopModel/render.py:300` (comment)

```
# §5.1 multi-body: a SEQUENCE of sub-loops, each over its own half-open range —
# so each gets its OWN header (a binding when the range is a single value), and
# there is no enclosing `for` to nest them under.  See _subbody_header.
```

## `Tensile/LoopModel/render.py:307` (comment)

```
# Branch is a register-ROTATION selector, NOT a guard: `rotate buf = mode % S`,
# each arm labelled by the buffer residue it selects (§6.1 register ring).
```

## `Tensile/LoopModel/render.py:337` (comment)

```
# UNROLLED view — expand the leaves, interleave read;wmma (a VIEW, §5.3 9a).
# ===========================================================================
```

## `Tensile/LoopModel/render.py:366` (comment)

```
# PINNED-residue branch (a peel-unrolled reduction substep): the arm key IS the
# mode's concrete value — bind it into env so nested coords/slots evaluate to it.
```

## `Tensile/LoopModel/render.py:375` (comment)

```
# a peel-step iter binding: push the bound value into env (concrete for prologue;
# a symbolic `T−M+t` drain value stays UNBOUND — shown, but the body's `iter`-slots
# render symbolically since T is runtime).
```

## `Tensile/LoopModel/render.py:381` (comment)

```
# honor a RESOLVED guard in this concrete env (first-touch `mode==0`, drain NLL `k
# < n_s−dr`) — the unrolled stream shows only the arm that actually executes.  (ADR 0007 #83)
```

## `Tensile/LoopModel/render.py:394` (comment)

```
# a Load carries a single Placement (evaluate it to concrete slots for this env); a
# wmma carries a PER-OPERAND {operand: Placement} DICT (consume-side reg residence) —
# pass the dict through so _mma_line evaluates each source's slots against env.
```

## `Tensile/LoopModel/render.py:417` (comment)

```
# Geometry table — every derived number, checkable by hand.
# ===========================================================================
```

## `Tensile/LoopModel/render.py:469` (comment)

```
# THE DOWNGRADE, NAMED. `readmode` prints `dr_g` and this line says whether that `dr_g` is the
# one that was ASKED FOR. PLR is an input parameter and `W` is derived, so a `dr_g` below `dr`
# is the derivation overriding the request — never a silent outcome.  (ADR 0007 #82)
```

## `Tensile/LoopModel/render.py:486` (comment)

```
# LEVELS: the full ord permutation, outer→inner, marking the OUTER pipelined levels (extent
# symbolic/0 → the reduction chunk / persist) vs the INNER unrolled modes (concrete extent).
```

## `Tensile/LoopModel/render.py:496` (comment)

```
# IN ORD ORDER, like every other list in this block.  `sorted()` over a set printed
# ['K_inner','K_split'] directly under a `levels(ord)` line reading K_split · K_inner — two
# orderings of the same modes, one line apart, with nothing saying they differ.
```

## `Tensile/LoopModel/render.py:502` (comment)

```
# off: the first-class retime matrix {(op, role, level): δ} (§2.4) — per (op-class × level),
# where the op-class is (operand, hop role).  Rendered `A:copy@iter=2`.
```

## `Tensile/LoopModel/render.py:555` (comment)

```
# THE ASYMMETRY, SPELLED OUT.  This is #170/#245's whole shape: agent on one hop,
# coordinate on the other, in one operand.  Collapsing it to a single answer is the bug.
```

## `Tensile/LoopModel/render.py:563` (comment)

```
# THE REGION COUNT, not just the axes. Without it `MXSA region axes ('M_split',)` reads as
# a split operand while `MXSA.split == 1` -- ONE storage region -- and the GIR dump, which
# gates on `operand_regions > 1`, correctly omits MXSA entirely.  (ADR 0007 #81)
```

## `Tensile/LoopModel/render.py:574` (comment)

```
# The decision table (`schedule.Schedule`) — what the decoder decided, per operand.
# ===========================================================================
```

## `Tensile/LoopModel/render.py:596` (comment)

```
#: The wording of every adapter note, keyed by its `Note.kind`. The compute path records the
#: numbers; this is the only place the sentence exists.
```

## `Tensile/LoopModel/schedule.py <module>` (module)

```
Every derived per-operand DECISION, in one place.

theta says what the kernel is; this says what we decided to do about it. If you are asking "what
changes when PrefetchLocalRead goes 1 -> 2", or "why does this operand get two register buffers and
that one four", the answer is an `OperandPlan` and you can print the whole table with
`render.render_schedule`.

ONE DERIVATION. The quantities below were computed at 14 separate call sites, each re-deriving them
from `(theta, S, depth)`. They are pure, so the answers agreed -- until one site passed a slightly
different argument, which is the shape of most of this decoder's history
(#310/#311/#312/#316/#321/#325/#331/#332). Going through here means two sites cannot disagree,
because there is only one site.

Memoized on `(operand, group, depth)` rather than fixed to one depth: the degenerate short-loop tree
legitimately asks at depth 0 while the steady body asks at theta's own, and collapsing those would
be a behaviour change rather than a refactor.
```

## `Tensile/LoopModel/schedule.py._assignment_width` (function)

```
Narrowest `W` in `[lo, hi]` whose rotation MAP admits the requested read-ahead, else `lo`.

    See ADR 0005.
    
```

## `Tensile/LoopModel/schedule.py.group_width` (function)

```
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

    See ADR 0005.
    
```

## `Tensile/LoopModel/schedule.py.derive_S` (function)

```
The per-tile-group buffer-ring depth map S.  Each group's depth is its rotation width
    `W` = group_width (the searched S coordinate from the fragment's group_policy, clamped
    into the derived band [L,R]).  No authored literals — the whole map derives.  Groups are
    keyed by their LABEL (Fragment.groups()).
```

## `Tensile/LoopModel/schedule.py.build_S` (function)

```
The full depth map: derived register floors PLUS the off-driven cross-iteration shared
    depth.  Register groups are keyed (op, group); the shared ring is keyed (op, 'shared').
    Returns (S, floor) where floor is the pure derived register floor (for reporting).
```

## `Tensile/LoopModel/schedule.py:1` (comment)

```
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
```

## `Tensile/LoopModel/schedule.py:40` (comment)

```
#: which loop axis the lookahead is counted along -- derived per operand from ord, NOT fixed
#: to K (#321: at DepthU=64 there is no K inner axis at all and the prefetch went silently inert)
```

## `Tensile/LoopModel/schedule.py:45` (comment)

```
#: the same lookahead as an offset in this operand's own load order, which is what the address
#: arithmetic and the drain bound consume. Equal to `prefetch_steps` only when the axis is
#: innermost; every place that used one meaning the other has been a bug.
```

## `Tensile/LoopModel/schedule.py:119` (comment)

```
# The register depth map S -- the first decision everything downstream reads
# ===========================================================================
```

## `Tensile/LoopModel/schedule.py:126` (comment)

```
# A DTV OPERAND HAS NO SHARED PLACEMENT (`global->register` only), so the substep read-ahead
# this widens for does not exist and the off lookup is meaningless.  Guard on the hop.
```

## `Tensile/LoopModel/schedule.py:130` (comment)

```
# THE REQUEST IS `off(READ, substep)` — WHAT PLR SET — not `prefetch_depth`, which returns
# `S_shared`, the LDS ring size (a DIFFERENT quantity; its own docstring says so).  (ADR 0007 #87)
```

## `Tensile/LoopModel/schedule.py:135` (comment)

```
# SIZE FOR THE REQUESTED DEPTH — `PLR` IS AN INPUT, `W` IS DERIVED. The user asks for a read-
# ahead depth; the ring width is ours to choose.  (ADR 0007 #86)
```

## `Tensile/LoopModel/schedule.py:191` (comment)

```
# reduction — no mainloop rotating ring (§6.1); its footprint is the
# separate accumulator term (§7), not a multibuffer depth entry.
```

## `Tensile/LoopModel/sigma_c.py <module>` (module)

```
σ_c ordering (paper §5.3.1 point 4 / emit_can step 9a): the refill-after-read freeze.

Clean-architecture split of the decoder: this module owns the issue-order refinements the decode
is REQUIRED to make, so an order-driven backend derives each WAR rather than an inverted forward
RAW.  ONE rule, applied at BOTH ring levels — a refill must be emitted AFTER the consumers that
vacate the buffer it overwrites:

  * SHARED ring — an `S=δ` refill COPY goes after the reads that vacate its buffer.
  * REGISTER ring — a refill READ into a group whose rotation width is 1 goes after the wmmas
    that vacate its single slot.  A W>1 group needs no deferral: read-ahead writes slot
    `(k+dr) mod W`, which is not the slot `k mod W` the current wmma reads, so the hoisted read
    and the consumer never touch the same register.  At W==1 those coincide, so hoisting the
    refill would clobber the operand the current wmma still needs — an inversion no `Await` can
    repair (an Await waits only on ALREADY-ISSUED reads, and the consuming wmma has not issued).

Both are the same §2.4 line 99 WAR, one per placement.  No TensileLite / AMD vocabulary; stdlib +
LoopModel core only.
```

## `Tensile/LoopModel/sigma_c.py._movement` (function)

```
The movement `Inst` node `n` denotes, looking THROUGH a rolled region loop and through a
    single-armed guard.

    A region-split copy is ONE op-class placed at its region level (§5.3.1 line 523), so the tree
    node is `Loop(region_mode)` over a single copy `Inst` — still one movement op-class, and σ_c
    must defer it as a unit.  Matching only a bare `Inst` here would silently stop deferring split
    copies the moment they became rolled.  A read is likewise wrapped in its first-touch /
    read-ahead-suppress `Cond` (single `then`, empty `els`), which does not change WHICH movement
    the node denotes — only whether it is issued this pass.
```

## `Tensile/LoopModel/sigma_c.py._rebuilt` (function)

```
`n` with σ_c applied to each of its sub-bodies (the freeze holds at EVERY level, not just
    the region body: an in-place register refill sits at its operand's read level, inside the
    nest).
```

## `Tensile/LoopModel/sigma_c.py.move_reloads_after_last_use` (function)

```
Apply σ_c to a straight-line body, and recursively to every sub-body within it.

    Two deferrals, both the §2.4 line 99 WAR "refill last", one per placement ring:

      (1) SHARED — a WAR-refill COPY (Inst with dst=SHARED carrying a *-WAR await) must be emitted
          AFTER the reads that vacate the buffer it refills.  The peel skeleton lists copies FIRST
          (line 394); this moves each to the END of the body, after the read/wmma nest.  Copies
          WITHOUT a WAR await (prologue fill: empty buffer) stay put.
      (2) REGISTER — a HOISTED refill READ carrying an `inplace-WAR` (its advance lands on the very
          slot in use) must be emitted AFTER the wmmas that vacate it.  Its body position is the
          read's own level, so this lands it after that level's inner nest — exactly where the
          consumers are.  A `rotation-WAR` read (W>1, non-colliding slot) is NOT deferred: it
          writes a different slot and keeps its read-ahead hoist.

          `HOISTED` is load-bearing, and the `inplace-WAR` await alone does not imply it.  §5.3.1
          pt5 gives a group two shapes: PREFETCH (`dr_g >= 1`) hoists the read ahead of its consumer,
          INPLACE (`dr_g = 0`) does not hoist at all — "issued in place one line before its own
          wmma".  An INPLACE read is ALREADY correctly placed: the generation it overwrites is the
          PREVIOUS trip's, whose consumer is behind it across the back edge, so the WAR is
          loop-carried and there is nothing to defer.  Deferring it anyway moves it past THIS
          trip's consumer, which then reads a register nothing filled — and because an INPLACE group
          also prologues nothing (same rule), no fill ever repairs it.  That is the paper's named
          failure "a PREFETCH prologue paired with an INPLACE steady", in mirror image.
          MEASURED 2026-08-18: `dr_g = max(0, min(dr, W-1))`, so `W == 1` forces `dr_g = 0` — every
          W==1 read is INPLACE, and keying the deferral on the await alone deferred exactly the
          reads that must not move.  216/216 `n_s == 1` configs failed `check_register_dataflow`
          with USE-BEFORE-DEF (0/432 at `n_s >= 2`), and all 16 such MXFP8 kernels miscompared on
          hardware.  The other route to `inplace-WAR` — `loads_in_place`, a wrapping advance
          at W>1 (#173) — requires a non-zero shift, so it is PREFETCH and still defers.

    Order within the result is stable, and (2) precedes (1): the register refill is inner to the
    shared refill, so the shared copy stays last.
```

## `Tensile/LoopModel/sigma_c.py:1` (comment)

```
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
```

## `Tensile/LoopModel/sigma_c.py:119` (comment)

```
# A copy whose chunk THIS trip's read-ahead crosses into (`off == r`, §5.3.1 pt5) is a RAW
# producer for those reads and must LEAD them — never deferred.  (ADR 0007 #88)
```

## `Tensile/LoopModel/theta.py <module>` (module)

```
θ — the canonical Spacetime-layout point the decoder consumes.

Follows the paper's tuple θ = (tile, path, ord, off, S, ρ, Φ):
  tile  — the mode set (created by splitting), INCLUDING the register-axis rate-group
          partition on each fragment (the searched reuse coordinate, §2.8 move 1).
  path  — each operand's Trajectory: a list of Hops (global→shared→register), each hop
          carrying its own retime `delta` and (derived) depth.
  ord   — the outer→inner order of the `time` modes (§2.5).
  off   — per-(hop,level) retime offsets, stored on the Hops as `delta`.
  S     — the depth map, DERIVED from (ord, register-axis tile, off); see derive_S.
  ρ     — agent assignment + role partition (§2.9): `resort` maps a looped mode to an AGENT
          LEVEL with an extent; `roles` is the `specialize` partition.  A MAP, not a token.
  Φ     — fusion partition (which loads merge into one cooperative instruction).

(No `φ`: the paper's core has no address-relabeling field — addresses are opaque, §2.1/§4.3.
The swizzle/stride algebra is the trimmed companion layout-algebra layer, not θ.)

Decoder CORE: no TensileLite / AMD vocabulary.  An addon (translate.py) fills θ from a
specific generator's params.  S is derived, not flagged — the reuse scheme is a `tile`
value (the fragment rate-group partition), validated to reproduce the flagship [lo:4,hi:2].
```

## `Tensile/LoopModel/theta.py.Mode` (class)

```
One tiled traversal axis: a name and its extent.  `theta.ord` is a list of these — the
    FULL permutation of looped time modes, outer→inner, INCLUDING the outer reduction /
    persistent levels (§2.4 point 1, §5.3.1 `levels = [iter] + [substep, m, n, …]`).

    See ADR 0005.
    
```

## `Tensile/LoopModel/theta.py.Fragment` (class)

```
The register-buffer structure of an operand's VGPR placement.

    See ADR 0005.
    
```

## `Tensile/LoopModel/theta.py.trajectory_sense` (function)

```
FORWARD (toward the compute) or REVERSE (away from it) for one op-class.

    Taken from `op.is_output`, which is the structural fact: an accumulator's trajectory is the
    reflection of an input's.  CROSS-CHECKED against the hops when there are any — a reverse sense
    with forward-running hops (or the other way) means the operand's role and its path disagree,
    which is exactly the confusion this function exists to end, so it raises instead of picking.
```

## `Tensile/LoopModel/theta.py.Hop` (class)

```
One leg of an operand's staging trajectory (path).  Carries transfer granularity and its
    MOVER KIND.  (The retime `δ` is NOT here — it is the first-class `theta.off_map`, §2.4/§2.9,
    read via `off_at(op, role, level)`; a hop is one op-class edge, and `off` is indexed by
    (op-class × level) where the op-class is this operand paired with this hop's `role`.)

    See ADR 0005.
    
```

## `Tensile/LoopModel/theta.py.Operand` (class)

```
One tensor trajectory in generic geometry.

    See ADR 0005.
    
```

## `Tensile/LoopModel/theta.py.Resort` (class)

```
One `resort(mode → agent)` (§2.8 move 2): a looped mode traversed by the AGENTS instead of
    by the coordinate.

    See ADR 0005.
    
```

## `Tensile/LoopModel/theta.py.Rho` (class)

```
ρ = agent assignment + role partition (§2.9 line 244).

    See ADR 0005.
    
```

## `Tensile/LoopModel/theta.py.Note` (class)

```
One thing the adapter recorded about this theta, for the `--output-loop-ir` dump.

    A RECORD, NOT A SENTENCE: `kind` names the message and `values` carries the numbers behind it.
    `render.NOTE_TEXT[kind]` is the only place the wording lives, so the compute path never builds
    a display string it usually discards.
    
```

## `Tensile/LoopModel/theta.py.DepthMap` (class)

```
`S_Λ : Q(Λ) → ℕ₊` (Definition 1) — depth per PLACEMENT Λ, per rotating coordinate.

    See ADR 0005.
    
```

## `Tensile/LoopModel/theta.py.is_outer` (function)

```
True iff this is an OUTER (dynamic / runtime-trip) pipelined level — the peel and
        shared-ring axis (§5.3.1/§6.1).  `extent==0` ⇔ trip not statically known.  The single
        intrinsic outer/inner test; `outer_modes()`/`inner_modes()` are just this filter.
```

## `Tensile/LoopModel/theta.py.group_broadcast` (function)

```
The axes a group is constant over.  Uniform across a fragment's slots, so this is
        just the fragment's `broadcast_axes` (the `group` arg is accepted for call-site
        symmetry and ignored).
```

## `Tensile/LoopModel/theta.py.policy_of` (function)

```
This group's W-in-[L,R] choice: an explicit int, 'unroll' (W=R), 'overlap' (W=L_pf),
        or 'inplace' (W=L_war).  Per-label entry first, then the fragment's "*" default, then 'unroll'.
        The single source of W; reproduces the two-width `[lo:4,hi:2]` (lo='unroll',
        hi='inplace') and lets an autotuner pass an explicit interior W.
```

## `Tensile/LoopModel/theta.py.at` (function)

```
The resort entries at any of `levels`, for one hop ROLE (default `read`).

        ROLE-SCOPED BY DEFAULT, and that is what makes the copy half additive.  Every caller that
        predates the copy entries is asking the classification question — which modes the agents
        traverse instead of the coordinate — so they keep seeing exactly the entries they saw
        before, including `span_over`'s "two resorted modes would be a product" guard, which would
        otherwise fire on a mode carrying both a read and a copy entry.
```

## `Tensile/LoopModel/theta.py.copy_agents` (function)

```
How many agents cooperate on `opname`'s global→shared COPY — the Φ group's wave share.

        ρ IS THE SOURCE FOR THIS.  It used to live in a per-operand `Operand.copy_agents` scalar
        that `translate` filled and NOTHING read; the fact belongs here, beside the read-side
        partition it is the sibling of, so there is one agent assignment rather than two shapes of
        one.  Per-op-class rather than per-mode because a Φ group's members share a free axis and
        take different shares of it — see `Resort.role`.

        `default` (1, the unspecialized single-agent answer) when ρ carries no copy entry, which is
        every single-wave kernel and anything built without the target's wave facts.
```

## `Tensile/LoopModel/theta.py.served_modes` (function)

```
The modes the AGENTS traverse INSTEAD OF the coordinate — §2.8 move 2's domain, as the
        axis CLASSIFICATION consumers mean it.  `Theta.agent_served_modes()` rediscovers this same
        set from a per-hop boolean; the two are asserted equal by `rho_consistency`.

        See ADR 0005.
        
```

## `Tensile/LoopModel/theta.py.span_over` (function)

```
The `level`-level agent span of whichever resorted mode lies in `modes` — ρ as the
        movement quantum reads it (§2.2: `Φ` over the tiles ρ places on the spanned agents).

        0 when no resorted mode is involved, which is the CONTIGUOUS shape: consecutive tiles on
        consecutive agents, `carrier = t // Φ`, one slot per tile.  A non-zero span is the
        DISTRIBUTED shape, and it is what collapses the slot into a broadcast.  Returns the entry's
        extent, never a count of entries — two resorted modes at one level would be a product, and
        that case raises rather than guessing.
        
```

## `Tensile/LoopModel/theta.py.agent_distributed` (function)

```
Are this op-class's shared-staged hazards CROSS-AGENT (§2.4)?

        See ADR 0005.
        
```

## `Tensile/LoopModel/theta.py.movement_units` (function)

```
The COPY MOVEMENTS this θ emits, as `[(unit_key, members, n_regions)]` — the ONE
        authority for "which cooperative movements exist and how many region instances each has".

        See ADR 0005.
        
```

## `Tensile/LoopModel/theta.py.levels` (function)

```
The FULL level permutation, outer→inner (§5.3.1 `levels = [iter] + [substep, m, n, …]`)
        — every looped time mode including the outer pipelined levels.  The emit nest and the peel
        skeleton walk these; `levels[0]` is the outermost level (the one peeled separately).
```

## `Tensile/LoopModel/theta.py.outer_modes` (function)

```
The OUTER pipelined levels (a problem-dimension trip, `Mode.is_outer`), in ord order —
        outermost first, since `ord` is already outer→inner.  These are the peel axes and the
        shared/shared-multibuffer axis (§6.1 reduction-chunk ring, §5.3.2 persistent level): the persistent
        `gtile` then the reduction chunk `iter`.  Empty for a legacy θ that leaves `iter` implicit.
        Callers wanting *the reduction chunk* must use `reduction_chunk_mode()`, NOT an index.
```

## `Tensile/LoopModel/theta.py.reduction_chunk_mode` (function)

```
The outer level the REDUCTION CHUNK advances on — the INNERMOST outer level, since
        `ord` is outer→inner and any coarser outer level (a persistent tile level, §5.3.2) sits
        ABOVE the chunk.  None for a legacy θ with no outer level.

        See ADR 0005.
        
```

## `Tensile/LoopModel/theta.py.inner_modes` (function)

```
The INTRA-reduction chunk modes only (the inner K substep + the M/N fan), in ord order.  The §2.6
        per-chunk register machinery (reload-rate / live-peak / width), the read nest, and the
        geometry/sizing layer walk EXACTLY these — the register ring rotates over them (§6.1).
        Excludes the outer pipelined levels so their symbolic trip never enters a rate/pressure
        product.  For a legacy θ (no outer mode) this is the whole `ord`, unchanged.
```

## `Tensile/LoopModel/theta.py.off_at` (function)

```
The retime offset `off(op-class, level)` (§2.4 `off : (op-class × level) → ℤ`, a
        first-class θ field, §2.9).

        See ADR 0005.
        
```

## `Tensile/LoopModel/theta.py.presence_modes` (function)

```
The inner Modes an op-class VARIES over (its presence set, §2.1) = the inner modes minus
        its broadcast (constant-over) axes.  One structural primitive; no name tests.

        See ADR 0005.
        
```

## `Tensile/LoopModel/theta.py.agent_served_modes` (function)

```
The THIRD kind of inner axis: traversed by the AGENTS, not by the coordinate (§2.8 move
        2 `resort`).  #245.

        See ADR 0005.
        
```

## `Tensile/LoopModel/theta.py.reduction_modes` (function)

```
`reduction_modes = {inner modes} \ pres(output)` (§2.1 line 82, PAPER-CONFIRMED Q18):
        the inner Modes ABSENT from every output op-class's presence — the contraction axes, in
        ord order.  General for batched/einsum (a batch mode is present on the output, a reduction
        mode is not, even though both index every input).  No `theta.op("A")`, no "present on all
        inputs" heuristic.  Empty if θ carries no output op-class (a malformed mainloop θ —
        translate always constructs the accumulator).
```

## `Tensile/LoopModel/theta.py.free_modes` (function)

```
`op`'s FREE presence Modes = presence minus the reduction modes (§2.6).  An operand's
        free-tile index / resident-tile live-peak iterate THESE — its own output subscripts, no
        operand-name test.
```

## `Tensile/LoopModel/theta.py.tensor_instrs_per_kiter` (function)

```
DISTINCT global->shared instructions per kiter (one shared counter tracks all).
        Fusion reduces the count — a fused group is ONE instruction PER REGION: a region-split
        group ({A,B} split-2) emits one movement per region (AB0, AB1), while an unsplit group
        ({MXSA,MXSB}) emits one (§2.8 move 9, region-indexed fusion).
```

## `Tensile/LoopModel/theta.py.regions_of` (function)

```
The region indices `op`'s `group` has rows for, in order.  `[None]` = one whole-tile
        placement.  This is the enumeration a per-placement consumer iterates instead of assuming
        one placement per operand.
```

## `Tensile/LoopModel/theta.py.groups_of` (function)

```
{group: depth} for one placement of `op`.  With `region=None` this folds every region's
        rows together, so it answers for an unsplit operand and for a split one whose regions agree
        — via `get`, which raises rather than pick when they do not.
```

## `Tensile/LoopModel/theta.py:1` (comment)

```
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
```

## `Tensile/LoopModel/theta.py:37` (comment)

```
# tile — modes and the register-axis rate-group partition
# ===========================================================================
```

## `Tensile/LoopModel/theta.py:114` (comment)

```
# hop (src, dst) -> op-class role.  A one-hop DTV trajectory (global→register) IS the read; the
# reverse-trajectory legs are the epilogue store (§5.3.2 (1), #116) and are listed so a store
# op-class has a role the moment `stage` splices its path on.
```

## `Tensile/LoopModel/theta.py:208` (comment)

```
# : (`copy_agents` WAS HERE. How many agents cooperate on this op-class's copy is a ρ fact and
# : now lives there -- `Rho.copy_agents(name)`, beside the read-side partition it is the
# sibling : of.  (ADR 0007 #95)
```

## `Tensile/LoopModel/theta.py:241` (comment)

```
# ρ — agent assignment + role partition (§2.8 moves 2 & 8, §2.9 line 244)
# ===========================================================================
```

## `Tensile/LoopModel/theta.py:247` (comment)

```
#: The finest level a synchronization scope may name (§5.2's ladder floor).  Anything finer is a
#: distribution fact only.
```

## `Tensile/LoopModel/theta.py:252` (comment)

```
# : A resort entry's HOP ROLE. `read` is the classification half — which modes the agents traverse
# : INSTEAD of the coordinate, so the read/compute nest must not enumerate them. `copy` is the :
# MOVEMENT half — how many agents cooperate on one op-class's global→shared copy, i.e.  (ADR 0007 #94)
```

## `Tensile/LoopModel/theta.py:368` (comment)

```
# INCLUDING the outer reduction level `iter` (extent==0) and any
# persistent level (also extent==0).  Project with outer_modes()
# / inner_modes(); `levels()` is the whole list (§5.3.1).
```

## `Tensile/LoopModel/theta.py:380` (comment)

```
# off: {(op_name, role, level_name): δ} — §2.4's retime matrix over (op-class × level), where
# the op-class is (operand, hop role) and `role` is one of theta.COPY/READ/STORE.  Read via
# `off_at(op, role, level)`; never index this dict directly.
```

## `Tensile/LoopModel/theta.py:386` (comment)

```
# (no `reduction_modes` field: the contraction axes are DERIVED by `reduction_names()` from the
# accumulator's presence, §2.1 line 82.  (ADR 0007 #92)
```

## `Tensile/LoopModel/theta.py:524` (comment)

```
# AN AGENT-SERVED AXIS IS NOT A REDUCTION. `inner \\ pres(output)` is a SUBTRACTION, so
# anything the output stops varying over falls in here by default -- including an axis that
# left the output's presence because the AGENTS traverse it, not because it is contracted.  (ADR 0007 #90)
```

## `Tensile/LoopModel/theta.py:551` (comment)

```
# (mode_roles() removed: the core no longer labels modes m/n/k — §7.1 index-agnostic.  Roles
# are derived where needed from presence + reduction_names(); consumers project coords onto
# free_names(op) / reduction_names().  loop_shape uses the operand-presence set as its key.)
```

## `Tensile/LoopModel/validate.py <module>` (module)

```
validate_loopir — the PERMANENT structural gate that the emitted LoopIR is proper, standalone,
pipelined-GEMM pseudocode (§5.3.1).  It reads the rolled θ-nest ALONE (no GIR) and asserts the
shape a human debugging the dump would check by eye, so a regression that makes the IR nonsense is
caught at build, not by a failing GPU run.

Returns a list of human-readable defect strings (EMPTY = valid).  Pure/read-only; no mutation.

The properties (from the reference spec + the paper §5.3.1 skeleton):
  1. Root is ONE peel-validity `Cond(T >= M)`; `then = [Prologue-Peel, steady Loop(outer), Drain-Peel]`
     (in that order); `els` present (the short-loop).  (M==0 degenerate: bare steady, no Cond.)
  2. No copy/read/wmma Inst appears OUTSIDE the root Cond's arms — nothing runs before the guard.
  3. `iter` is BOUND in every peeled phase (prologue/drain/short via a `Bind`); NO free `iter`
     reference sits directly under a Peel without a Bind.
  4. Drain = M straight-line `Bind` steps (one per chunk); NO `Cond(iter%d)` residue pin, NO
     Branch-cartesian.
  5. Every register-dest read + every wmma-source placement carries a register GROUP; a read at a
     PINNED (Bind-concrete) coord resolves to a CONCRETE (int) slot — no unbound-symbol slot.
  6. In the steady body a wmma is preceded by its operands' reads (read-ahead leads); the σ_c refill
     copies are the LAST ops of the steady body.
  7. The obligation ledger is empty (`check_ledger_discharged == []`).
  8. Core `Cond`s carry only GENERIC `kind`s (peel_validity / first_touch / readahead_suppress /
     short_step_validity) and NO scaffold `label` — scaffold naming is the GIR pass's job.
  9. Every short-loop (`els`) step past the first is guarded by `T > t`: that arm runs when T < M,
     so an unguarded step t>=1 would copy a nonexistent reduction chunk.
```

## `Tensile/LoopModel/validate.py._unbound_placement_refs` (function)

```
Every (Inst, mode) where the Inst's placement references a mode NO enclosing binder binds.

    This is the assertion P3 and P5 both rest on, and it is one check, not two: P3 ("no FREE `iter`
    reference sits under a Peel without a Bind") and P5's second half ("a read at a pinned coord
    resolves to a CONCRETE slot — no unbound-symbol slot") are the same defect seen from two
    angles.  The historical bug was a prologue read rendering `register[buf(K_inner)%2]` with
    `K_inner` bound by nothing.

    Binders are the nodes that introduce a mode into scope: `Loop` (its index), `Bind` (the peel's
    pinned induction — whether it resolves to a concrete int or stays symbolic in the runtime trip,
    it is DECLARED), and `Branch` (its residue selector).  The trip symbol is a problem dimension,
    not a mode, so it is never "unbound".
```

## `Tensile/LoopModel/validate.py.validate_loopir` (function)

```
Return a list of structural-defect strings for the rolled IR `ir` (EMPTY = valid GEMM
    pseudocode).  `S` (the depth map) is only needed for the ledger check; derived if omitted.
    `undischarged` may carry an already-computed `check_ledger_discharged` result (emit_mainloop
    has one) so the gate does not redo it.
```

## `Tensile/LoopModel/validate.py.rho_consistency` (function)

```
R1's safety argument (#303): does ρ agree with the three live agent authorities?

    See ADR 0005.
    
```

## `Tensile/LoopModel/validate.py:1` (comment)

```
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
```

## `Tensile/LoopModel/validate.py:38` (comment)

```
#: `reg_group` restricts a group-restricted read to the grouping-mode values that register group
#: OWNS (§2.8 move 1, #118) — a structural fact about the register partition, in the same class as
#: `first_touch`: a predicate over loop modes only, no scaffold vocabulary.
```

## `Tensile/LoopModel/validate.py:139` (comment)

```
# STRICT `T > M`. The paper's §5.2 skeleton writes `>=` and its NB blesses either, but the
# NB's reasoning ("at T = M ... its steady loop has trip count 0, a header that never
# executes") presumes a PRE-TESTED header.  (ADR 0007 #102)
```

## `Tensile/LoopModel/validate.py:158` (comment)

```
# M==0 degenerate: a bare steady loop is only valid when there is genuinely NO peel — a
# prologue/drain sitting at top level (outside a guard) is defect P1 (must be Cond-wrapped).
```

## `Tensile/LoopModel/validate.py:196` (comment)

```
# P3 (the real assertion) + P5 (its concrete-slot half): NO placement may reference a mode that
# no enclosing Loop/Bind/Branch binds. Both properties reduce to this one check — see
# `_unbound_placement_refs`.  (ADR 0007 #101)
```

## `Tensile/LoopModel/validate.py:228` (comment)

```
# every sub-body must satisfy σ_c independently — a multi-body steady loop (§5.1) has one
# trip per body, so checking only `bodies[0]` would leave the rest unchecked.
```

## `Tensile/LoopModel/validate.py:231` (comment)

```
# the refill copies are the σ_c-ordered tail: the LAST top-level nodes of the body that
# are shared-dest Loads (copies). Assert no read/wmma follows the first copy.  (ADR 0007 #100)
```

## `Tensile/LoopModel/validate.py:234` (comment)

```
# only an `S=δ` refill is required to be last (§5.3.1 point 4 is scoped to "an
# S=delta refill copy"); a rotation-WAR copy is emitted FIRST so this trip's
# chunk-crossing reads can consume it (§5.3.1 pt5), and P6 must not flag that.
```


# Pass 2 — the rest of the docstring, and the standalone comments

## `Tensile/LoopModel/__init__.py:1`

```
# Copyright Advanced Micro Devices, Inc., or its affiliates.
```

## `Tensile/LoopModel/adapter/__init__.py:1`

```
# Copyright Advanced Micro Devices, Inc., or its affiliates.
```

## `Tensile/LoopModel/adapter/build.py.Splits`

```
How TDMSplit cuts each operand's tile: per-operand MT, shared DU, and how many regions
    one wave occupies.
    
```

## `Tensile/LoopModel/adapter/build.py.Ord`

```
The loop nest this LoopOrder gives: the six modes in order, each operand's regions, and
    the modes each operand is broadcast over.
    
```

## `Tensile/LoopModel/adapter/build.py._build_fragments`

```
Each operand's register fragment: its width policy, its region labels, and the MX scale
    ring that rides with it.
    
```

## `Tensile/LoopModel/adapter/build.py._place_read_offsets`

```
Write `off(read, <level>) = PLR` for each requesting op-class, at its own read-ahead
    level.
    
```

## `Tensile/LoopModel/adapter/build.py._fit_register_budget`

```
derive the register partition from the budget/: how many fan tiles stay at the derived
    width, and how many fall to their floor.
    
```

## `Tensile/LoopModel/adapter/build.py:1`

```
# Copyright Advanced Micro Devices, Inc., or its affiliates.
```

## `Tensile/LoopModel/adapter/build.py:78`

```
# note (MI shape): the default MI 16×16×64 matches the paper's base GEMM and gives substep =
```

## `Tensile/LoopModel/adapter/build.py:79`

```
# DepthU/MI_K = 128/64 = 2 (the a/b that makes the flagship two-rate).
```

## `Tensile/LoopModel/adapter/build.py:93`

```
# PER-operand overrides.
```

## `Tensile/LoopModel/adapter/build.py:97`

```
# The MX scales get the same per-operand knobs the data operands have.
```

## `Tensile/LoopModel/adapter/build.py:99`

```
# {operand: elements ONE shared->register instruction moves per lane} — the movement quantum's
```

## `Tensile/LoopModel/adapter/build.py:100`

```
# domain which is the target's instruction selection and so has no Solution key; the bridge
```

## `Tensile/LoopModel/adapter/build.py:102`

```
# {operand: ir.QuantumMap} — the movement quantum, supplied.
```

## `Tensile/LoopModel/adapter/build.py:104`

```
# pi (internal, see PARAM_HELP): default matches the backend today.
```

## `Tensile/LoopModel/adapter/build.py:161`

```
# Prefetch depths are the copy (PGR) / read (PLR) retime offsets —'s peel is general in off
```

## `Tensile/LoopModel/adapter/build.py:162`

```
# (any depth), so these are NOT capped: dg=PGR, dl=PLR verbatim.
```

## `Tensile/LoopModel/adapter/build.py:175`

```
# POSITIONAL LIST right-padded with 1s: per-operand/per-axis split, 1 = no split. Legacy bool:
```

## `Tensile/LoopModel/adapter/build.py:176`

```
# True → [2,2] (split both MT), False → [1,1].  (ADR 0007 #9)
```

## `Tensile/LoopModel/adapter/build.py:183`

```
# regions ONE wave occupies.
```

## `Tensile/LoopModel/adapter/build.py:187`

```
# DU (shared K): ONE axis at the finest split; a coarser operand is present on it at a stride.
```

## `Tensile/LoopModel/adapter/build.py:211`

```
# canonical-mode name sets for role/region/grouping binding (only >1 modes survive in ord;
```

## `Tensile/LoopModel/adapter/build.py:212`

```
# naming a dropped mode is harmless — the core intersects with the present inner modes).
```

## `Tensile/LoopModel/adapter/build.py:215`

```
# the read'S region is the agent'S.
```

## `Tensile/LoopModel/adapter/build.py:228`

```
# the read hop'S payload is PER operand, and `lrvw` is only the data operands' answer.
```

## `Tensile/LoopModel/adapter/build.py:231`

```
# the movement quantum is supplied never derived here.
```

## `Tensile/LoopModel/adapter/build.py:234`

```
# A wave that does NOT cover every region makes the read'S region agent-relative.
```

## `Tensile/LoopModel/adapter/build.py:237`

```
# 's carrier group in θ'S own vocabulary.
```

## `Tensile/LoopModel/adapter/build.py:278`

```
# derived from ρ, not from `ReadRho` / a per-hop boolean R3).
```

## `Tensile/LoopModel/adapter/build.py:294`

```
# retime δ is NOT on the hop — it is theta.off_map (built below), read via off_at.
```

## `Tensile/LoopModel/adapter/build.py:297`

```
# 2-hop: a bulk 'tdm' global→shared copy (carrying its region split) + a per-lane ds_read.
```

## `Tensile/LoopModel/adapter/build.py:311`

```
# Bind `grouping_mode` to the in-region fan (when it exists) so the fan folds into
```

## `Tensile/LoopModel/adapter/build.py:312`

```
# |part|×region and NOT into the rate — this is what prevents the fan double-count for both
```

## `Tensile/LoopModel/adapter/build.py:323`

```
# copy-hop region split = how many regions this operand's copy moves, which is its own count on
```

## `Tensile/LoopModel/adapter/build.py:324`

```
# each region axis — `aDU`, not the shared axis extent.
```

## `Tensile/LoopModel/adapter/build.py:328`

```
# S_shared PER operand: the depth is a map, not a scalar).
```

## `Tensile/LoopModel/adapter/build.py:332`

```
# the MX scale rings are their own, preset to their parent'S.
```

## `Tensile/LoopModel/adapter/build.py:335`

```
# The allocation bound, which is not symmetric with the register side.
```

## `Tensile/LoopModel/adapter/build.py:350`

```
# `free_split` is `aMT`/`bMT` -- the free-axis region count -- carried as a number so the
```

## `Tensile/LoopModel/adapter/build.py:351`

```
# sibling bound has an anchor that survives a mode reassignment.
```

## `Tensile/LoopModel/adapter/build.py:361`

```
# C — the output/accumulator op-class line 80, paper-confirmed Q18).
```

## `Tensile/LoopModel/adapter/build.py:368`

```
# MX scales follow their parent: same grouping_mode + region_modes + floor policy so.
```

## `Tensile/LoopModel/adapter/build.py:371`

```
# the scale ring uses its parent'S fragment builder, not a second inline `Fragment`.
```

## `Tensile/LoopModel/adapter/build.py:410`

```
# A fused group pairs its members by region index which is only well-defined when every
```

## `Tensile/LoopModel/adapter/build.py:411`

```
# member has the same region count — a mixed-split group (e.g.
```

## `Tensile/LoopModel/adapter/build.py:443`

```
# the read-ahead level, one derivation shared with the core.
```

## `Tensile/LoopModel/adapter/build.py:448`

```
# the role comes off the hop the offset retimes, so the key cannot drift from the
```

## `Tensile/LoopModel/adapter/build.py:449`

```
# trajectory it describes (`Hop.role`, theta.HOP_ROLE).
```

## `Tensile/LoopModel/adapter/build.py:454`

```
# the level needs the built θ (it is the hop's presence,'s absorption included), so the
```

## `Tensile/LoopModel/adapter/build.py:455`

```
# entry is deferred to `_place_read_offsets` below rather than guessed here.
```

## `Tensile/LoopModel/adapter/build.py:458`

```
# π: region completion granularity (internal toggle; False matches the backend today).
```

## `Tensile/LoopModel/adapter/build.py:538`

```
# score what the allocator emits, not the model's ragged footprint — see
```

## `Tensile/LoopModel/adapter/build.py:539`

```
# `geometry.operand_emitted_regs`.
```

## `Tensile/LoopModel/adapter/build.py:550`

```
# try the deep count downward; the first that fits is the largest, i.e.
```

## `Tensile/LoopModel/adapter/fragments.py._mx_frag_elems`

```
MX scale elements per lane = MIInputPerThread(data) // MXBlock · duplicateFactor,
    duplicateFactor = 32 // MI_M (=2 for MI_M=16 on gfx1250) (MatrixInstruction.py:143).
    
```

## `Tensile/LoopModel/adapter/fragments.py:1`

```
# Copyright Advanced Micro Devices, Inc., or its affiliates.
```

## `Tensile/LoopModel/adapter/loop_order.py.canonical_loop_order`

```
The canonical spelling of a LoopOrder value: the 3-letter shortcut when the word keeps
    each axis's (split, inner) pair contiguous, the 6-letter word otherwise.
    
```

## `Tensile/LoopModel/adapter/loop_order.py._word_to_modes`

```
Map a 6-letter axis word to the ordered list of canonical mode-role names (outer->inner):
    first occurrence of a letter -> f'{axis}_split', second -> f'{axis}_inner'.
    
```

## `Tensile/LoopModel/adapter/loop_order.py._canonical_ord`

```
Build the 6 canonical Modes and emit them in the LoopOrder word order, dropping extent-1
    modes — except an inner mode whose paired split mode is live.
    
```

## `Tensile/LoopModel/adapter/loop_order.py:1`

```
# Copyright Advanced Micro Devices, Inc., or its affiliates.
```

## `Tensile/LoopModel/adapter/loop_order.py:11`

```
# The canonical 6-mode vocabulary + LoopOrder word encoding/.
```

## `Tensile/LoopModel/adapter/loop_order.py:13`

```
# Each of K/M/N contributes a (split-region, inner-remainder) pair.
```

## `Tensile/LoopModel/adapter/loop_order.py:22`

```
# LoopOrder is a WORD over the axis letters {K, M, N} — memorable, no lookup table.  See ADR 0006.
```

## `Tensile/LoopModel/adapter/loop_order.py:65`

```
# the K region axis is three modes, NOT ONE: `K_split` (the gcd of the operands' DU factors,
```

## `Tensile/LoopModel/adapter/loop_order.py:66`

```
# shared) times `K_splitA` and `K_splitB` (each operand's own residual, `own // gcd`).
```

## `Tensile/LoopModel/adapter/loop_order.py:75`

```
# canonical Mode names kept generic-but-role-tagged so geometry's role test still works: the
```

## `Tensile/LoopModel/adapter/loop_order.py:76`

```
# physical mode name is the role name; extents drive m/n/k classification via broadcast.
```

## `Tensile/LoopModel/adapter/loop_order.py:82`

```
# a degenerate tile axis survives iff its own region axis is live — see the docstring
```

## `Tensile/LoopModel/adapter/loop_order.py:86`

```
# `_word_to_modes` speaks the 6-letter K/M/N word, so the K region factorization is expanded
```

## `Tensile/LoopModel/adapter/loop_order.py:87`

```
# here: wherever the word places K's region axis, emit the shared link then the two residuals.
```

## `Tensile/LoopModel/adapter/loop_order.py:93`

```
# SET the outer DepthU reduction level `iter` (extent=0 → dynamic/runtime trip = K//DU), as the
```

## `Tensile/LoopModel/adapter/loop_order.py:94`

```
# outermost entry of `ord` point 1, `levels = [iter] + [substep, m, n, …]`).
```

## `Tensile/LoopModel/adapter/solution.py.kernel_to_ir_text`

```
Return the rolled θ-IR (render_stream) for `kernel`, plus the θ translation notes and the
    ledger status.
    
```

## `Tensile/LoopModel/adapter/solution.py:1`

```
# Copyright Advanced Micro Devices, Inc., or its affiliates.
```

## `Tensile/LoopModel/adapter/solution.py:12`

```
# The decoder's render_* helpers use a few unicode glyphs (theta, Sigma, times, middle dot,
```

## `Tensile/LoopModel/adapter/solution.py:13`

```
# subscripts).
```

## `Tensile/LoopModel/adapter/solution.py:42`

```
# LDS double-buffer count — the real kernel-derived value (2=double, 1=single/in-place),
```

## `Tensile/LoopModel/adapter/solution.py:43`

```
# not an assumption; drives in_place_shared in translate.py.
```

## `Tensile/LoopModel/adapter/solution.py:52`

```
# Φ move 9) — the fuse, derived, never requested.
```

## `Tensile/LoopModel/adapter/solution.py:54`

```
# rho / agents: how many agents cooperate on one tile.
```

## `Tensile/LoopModel/adapter/solution.py:56`

```
# region split move 1) — passed through, NOT defaulted off.
```

## `Tensile/LoopModel/adapter/solution.py:58`

```
# How many of those regions ONE wave occupies: the copy writes every region, a wave reaches
```

## `Tensile/LoopModel/adapter/solution.py:59`

```
# only the ones its own free-axis offset lands in.
```

## `Tensile/LoopModel/adapter/solution.py:62`

```
# microscaling — passed through for the same reason `TDMSplit` is.
```

## `Tensile/LoopModel/adapter/solution.py:65`

```
# phase-1 scope: default schedule, no new search axes.
```

## `Tensile/LoopModel/adapter/solution.py:67`

```
# the read hop'S instruction payload, per operand — the movement quantum's domain.
```

## `Tensile/LoopModel/adapter/solution.py:69`

```
# the movement quantum — computed here, where the layout is knowable, and handed to θ in
```

## `Tensile/LoopModel/adapter/solution.py:70`

```
# GEMM terms.
```

## `Tensile/LoopModel/adapter/solution.py:72`

```
# 's carrier group as Φ and ρ — the two numbers θ derives the `QuantumMap` from
```

## `Tensile/LoopModel/adapter/solution.py:73`

```
# (`geometry.derive_quantum`).
```

## `Tensile/LoopModel/adapter/solution.py:83`

```
# MIWaveTile is stored per-operand (MIWaveTileA/B) on a derived Solution; the decoder wants the
```

## `Tensile/LoopModel/adapter/solution.py:84`

```
# [WT0, WT1] pair.
```

## `Tensile/LoopModel/adapter/solution.py:89`

```
# bytes/element from the compute input datatype (numBytes = numRegisters*4).
```

## `Tensile/LoopModel/adapter/solution.py:112`

```
# `lrvwUnrollMXS*`: the scale's own unroll-major read width, else 1 element (the tile-major
```

## `Tensile/LoopModel/adapter/solution.py:113`

```
# branch reads one scale per instruction).
```

## `Tensile/LoopModel/adapter/solution.py:124`

```
# any remaining non-ascii -> '?' so the file is guaranteed plain ascii text
```

## `Tensile/LoopModel/adapter/target.py:1`

```
# Copyright Advanced Micro Devices, Inc., or its affiliates.
```

## `Tensile/LoopModel/emit/__init__.py.build_ir`

```
The rolled theta-nest IR.2/5.3): one Loop per ord inner mode, one Inst per op-class at
    its presence level, slots symbolic.
    
```

## `Tensile/LoopModel/emit/__init__.py.loop_shape`

```
Decide, from θ alone, whether the ord yields ONE uniform steady body (cleanly loopable)
    or needs a condition-branch / multi-body loop path.
    
```

## `Tensile/LoopModel/emit/__init__.py:1`

```
# Copyright Advanced Micro Devices, Inc., or its affiliates.
```

## `Tensile/LoopModel/emit/__init__.py:25`

```
# The issuing agent for every instance in a single-role kernel: ρ → `Inst.a` and `Loop.role`; "an
```

## `Tensile/LoopModel/emit/__init__.py:26`

```
# unspecialized kernel is the single-role special case").
```

## `Tensile/LoopModel/emit/__init__.py:46`

```
# presence-derived partition key (no m/n/k literal): each inner mode's key is the SET of
```

## `Tensile/LoopModel/emit/__init__.py:47`

```
# operands present on it (a reduction mode → all inputs; a free mode → its own operand subset).
```

## `Tensile/LoopModel/emit/__init__.py:58`

```
# render the keys as sorted operand-name tuples for a readable report (not branch logic).
```

## `Tensile/LoopModel/emit/__init__.py:63`

```
# `ord` cleanliness.
```

## `Tensile/LoopModel/emit/__init__.py:66`

```
# precondition Q.
```

## `Tensile/LoopModel/emit/__init__.py:70`

```
# ONE obligation, ONE discharge: drop an Await a preceding instruction on the same path already
```

## `Tensile/LoopModel/emit/__init__.py:71`

```
# established (ledger.discharge_once).
```

## `Tensile/LoopModel/emit/__init__.py:73`

```
# No wait-count lowering: each Await keeps count=-1 (unset), the named dependency is the
```

## `Tensile/LoopModel/emit/__init__.py:74`

```
# output/ count-inserting backend).
```

## `Tensile/LoopModel/emit/__init__.py:76`

```
# the structural gate.
```

## `Tensile/LoopModel/emit/__init__.py:79`

```
# name the θ that failed.
```

## `Tensile/LoopModel/emit/__init__.py:117`

```
# A read-ahead residency gets no Await of its own.
```

## `Tensile/LoopModel/emit/__init__.py:147`

```
# trailing drain steps that suppress the read-ahead = the derived chunk reach `r`.
```

## `Tensile/LoopModel/emit/__init__.py:158`

```
# no σ_c here — and that absence is the point, not an omission.
```

## `Tensile/LoopModel/emit/__init__.py:170`

```
# strict `T > M`, not `T >= M`.
```

## `Tensile/LoopModel/emit/copies.py._site_awaits`

```
Build the Await tuple for the (opname, role) consume site from its ledger
        obligations.
        
```

## `Tensile/LoopModel/emit/copies.py:1`

```
# Copyright Advanced Micro Devices, Inc., or its affiliates.
```

## `Tensile/LoopModel/emit/copies.py:25`

```
# The issuing agent for every instance in a single-role kernel: ρ → `Inst.a` and `Loop.role`; "an
```

## `Tensile/LoopModel/emit/copies.py:26`

```
# unspecialized kernel is the single-role special case").
```

## `Tensile/LoopModel/emit/copies.py:36`

```
# σ_c first, nesting second, and the order is not stylistic.
```

## `Tensile/LoopModel/emit/copies.py:60`

```
# A Φ-fused movement's obligations are keyed by the movement, because one cooperative
```

## `Tensile/LoopModel/emit/copies.py:61`

```
# instruction owes one thing (ledger).
```

## `Tensile/LoopModel/emit/copies.py:70`

```
# the dep is the producer endpoint plus the generation qualifiers that pick which
```

## `Tensile/LoopModel/emit/copies.py:71`

```
# instance: the chunk the consumer reads, and (per-region π) its region.
```

## `Tensile/LoopModel/emit/copies.py:82`

```
# ONE movement instance = the region-`region` group of this fused set (AB0 = {A0,B0}).
```

## `Tensile/LoopModel/emit/copies.py:85`

```
# a token is the θ op-class name, always — never a synthesized string.
```

## `Tensile/LoopModel/emit/copies.py:87`

```
# per-movement bytes: each member contributes its own per-region share (tile / its own
```

## `Tensile/LoopModel/emit/copies.py:88`

```
# split), so a region movement of a split member is tile/split, an unsplit member whole.
```

## `Tensile/LoopModel/emit/copies.py:91`

```
# the region is a coordinate, left symbolic because the enclosing `Loop(region_mode)`.
```

## `Tensile/LoopModel/emit/copies.py:94`

```
# awaits from the ledger (step 9b): the copy's consume site is (op, "copy") — its RAW.
```

## `Tensile/LoopModel/emit/copies.py:105`

```
# `part` is the intra-tile transfer index of ONE movement (1 here — we do not model sub-
```

## `Tensile/LoopModel/emit/copies.py:106`

```
# tile transfer fan-out on the copy).
```

## `Tensile/LoopModel/emit/copies.py:121`

```
# the buffer slot is region-invariant (all regions of a chunk land in the same ring
```

## `Tensile/LoopModel/emit/copies.py:122`

```
# generation), so it is computed once for the rolled copy, not per region.
```

## `Tensile/LoopModel/emit/copies.py:130`

```
# The fully-peeled (short-loop) arm.
```

## `Tensile/LoopModel/emit/copies.py:133`

```
# drain step under Bind(iter=T−M+t): use the steady symbolic slot (keyed on iter,
```

## `Tensile/LoopModel/emit/copies.py:134`

```
# which Bind defines to T−M+t), so the drain shares the one residue timeline.
```

## `Tensile/LoopModel/emit/copies.py:142`

```
# prologue copies fill empty (not-yet-read) buffers → no vacating read → no WAR.
```

## `Tensile/LoopModel/emit/copies.py:146`

```
# A region-split copy is ONE op-class at its region level line 523: "the.
```

## `Tensile/LoopModel/emit/copies.py:150`

```
# filed at the innermost axis it is enumerated by, so all of them enclose it
```

## `Tensile/LoopModel/emit/copies.py:158`

```
# The copy nests over every ord level -- including the agent-served ones the read/compute
```

## `Tensile/LoopModel/emit/copies.py:159`

```
# nest dropped.
```

## `Tensile/LoopModel/emit/copies.py:163`

```
# levels come from the enumeration axes: every axis a movement is enumerated by must
```

## `Tensile/LoopModel/emit/copies.py:164`

```
# exist as an enclosing loop.
```

## `Tensile/LoopModel/emit/copies.py:178`

```
# no `n == ext[m]` check.
```

## `Tensile/LoopModel/emit/copies.py:180`

```
# presence on an enclosing axis is A stride, and first-touch is its degenerate.
```

## `Tensile/LoopModel/emit/copies.py:190`

```
# NOT `_peel_first_touch` here, unlike the reads.
```

## `Tensile/LoopModel/emit/helpers.py._check_peel_is_emittable`

```
Refuse a θ whose peel `build_ir` cannot express — instead of emitting a different
    schedule.
    
```

## `Tensile/LoopModel/emit/helpers.py._reduction_inner_mode`

```
`(name, extent)` of the ord-level read-ahead level, used at its one call site purely as a
    sentinel — "does this θ have an intra-region axis at all", i.e.
    
```

## `Tensile/LoopModel/emit/helpers.py._resolve_ft`

```
Rewrite a subtree, resolving every first-touch guard on `mode` to a known outcome:
    `taken=True` splices the guard's body in (this is the mode's first iteration),
    `taken=False` drops it.
    
```

## `Tensile/LoopModel/emit/helpers.py._region_note`

```
Under the per-region π (`theta.per_region_completion`), the region mode whose value
    identifies which movement instance a consumer awaits — else None.
    
```

## `Tensile/LoopModel/emit/helpers.py._region_mode_of`

```
The inner mode that enumerates this fused group's storage regions, or None if the group
    is unsplit.
    
```

## `Tensile/LoopModel/emit/helpers.py._enum_axes`

```
The axes that enumerate this movement's regions — the representative member's own, not
    the union over members.
    
```

## `Tensile/LoopModel/emit/helpers.py:1`

```
# Copyright Advanced Micro Devices, Inc., or its affiliates.
```

## `Tensile/LoopModel/emit/helpers.py:25`

```
# The issuing agent for every instance in a single-role kernel: ρ → `Inst.a` and `Loop.role`; "an
```

## `Tensile/LoopModel/emit/helpers.py:26`

```
# unspecialized kernel is the single-role special case").
```

## `Tensile/LoopModel/emit/helpers.py:45`

```
# A read `off` at an outer level is accepted, deepens the peel, AND is applied to nothing.
```

## `Tensile/LoopModel/emit/helpers.py:65`

```
# A negative `off` is ill-typed.
```

## `Tensile/LoopModel/emit/helpers.py:151`

```
# the copy hop's presence set names the region axis, and is empty when the tile is not split
```

## `Tensile/LoopModel/emit/helpers.py:152`

```
# line 583) — so the `op.split <= 1` test and the region_modes scan are the same one question,
```

## `Tensile/LoopModel/emit/helpers.py:232`

```
# X@shared is consumed by the read; X@register by the WMMA.
```

## `Tensile/LoopModel/emit/helpers.py:234`

```
# a *-WAR's consumer is the refiller: the shared ring is refilled by the copy, a register group
```

## `Tensile/LoopModel/emit/helpers.py:235`

```
# by the read itself.
```

## `Tensile/LoopModel/emit/helpers.py:245`

```
# "reads" = every input operand that ends in a register fragment (two-hop shared→reg AND one-
```

## `Tensile/LoopModel/emit/helpers.py:246`

```
# hop DTV global→reg); each is emitted at its presence level.
```

## `Tensile/LoopModel/emit/helpers.py:248`

```
# a "scale" operand = an input read beyond the two matmul inputs (an extra elementwise/
```

## `Tensile/LoopModel/emit/helpers.py:249`

```
# broadcast operand, — presence-derived, no "MX" name test.
```

## `Tensile/LoopModel/emit/helpers.py:253`

```
# THIS USED TO RE-DERIVE THE RULE AND GET THE EMPTY CASE WRONG.  See ADR 0006.
```

## `Tensile/LoopModel/emit/helpers.py:255`

```
# the WMMA is placed by its own presence, not by the nest's depth.
```

## `Tensile/LoopModel/emit/helpers.py:262`

```
# COARSE copy has empty inner presence → the reduction-chunk level; a REGION-SPLIT copy.  See ADR 0006.
```

## `Tensile/LoopModel/emit/nest.py._drain_step`

```
One drain step `i = T−M+t`): the rolled read/wmma nest (reduction stays a `for k`
        Loop) plus the staggered refill copies, all in the steady symbolic form (slots keyed
        on `iter`), wrapped in `Bind(iter = T−M+t)` so every `iter`-symbolic slot/coord
        
```

## `Tensile/LoopModel/emit/nest.py:1`

```
# Copyright Advanced Micro Devices, Inc., or its affiliates.
```

## `Tensile/LoopModel/emit/nest.py:25`

```
# The issuing agent for every instance in a single-role kernel: ρ → `Inst.a` and `Loop.role`; "an
```

## `Tensile/LoopModel/emit/nest.py:26`

```
# unspecialized kernel is the single-role special case").
```

## `Tensile/LoopModel/emit/nest.py:36`

```
# symbolic leaf wmma: coord is the loop indices, block from region math (symbolic mi/ni)
```

## `Tensile/LoopModel/emit/nest.py:40`

```
# awaits from the ledger (step 9b): the wmma consumes each read operand's register
```

## `Tensile/LoopModel/emit/nest.py:41`

```
# residency (X@register RAW).
```

## `Tensile/LoopModel/emit/nest.py:45`

```
# register placement the wmma consumes, PER read operand — the name-determining tile coord
```

## `Tensile/LoopModel/emit/nest.py:46`

```
# + rotation the source vgpr carries.
```

## `Tensile/LoopModel/emit/nest.py:55`

```
# an anchored refill is emitted after this level'S consumers.
```

## `Tensile/LoopModel/emit/nest.py:60`

```
# nll suppression (drain last steps): a read that varies over the reduction substep
```

## `Tensile/LoopModel/emit/nest.py:61`

```
# carries the read-ahead-shifted coord `(k+dr)%n_s`; on the ahead substeps (k ≥ n_s−dr)
```

## `Tensile/LoopModel/emit/nest.py:64`

```
# A read-ahead read is out of bounds on the last peel step exactly when its.
```

## `Tensile/LoopModel/emit/nest.py:67`

```
# register-group ownership: this read fills ONE group, so it belongs only at the
```

## `Tensile/LoopModel/emit/nest.py:68`

```
# grouping mode's values that group owns.
```

## `Tensile/LoopModel/emit/nest.py:71`

```
# first-touch: wrap in `mode == <anchor>` for each enclosing invariant mode (outermost
```

## `Tensile/LoopModel/emit/nest.py:72`

```
# first, so the outermost guard is the outermost node).
```

## `Tensile/LoopModel/emit/nest.py:81`

```
# The wmma sits at `leaf_mode` -- the deepest level the accumulator varies over -- and the
```

## `Tensile/LoopModel/emit/nest.py:82`

```
# nest continues below it for any level the output does not traverse.
```

## `Tensile/LoopModel/emit/nest.py:87`

```
# ...then the anchored refills, after every consumer this level contains.
```

## `Tensile/LoopModel/emit/nest.py:91`

```
# A read whose presence the quantum emptied sits at the chunk level, and `build_level` has.
```

## `Tensile/LoopModel/emit/nest.py:98`

```
# hoisted above the whole inner nest, so the grouping mode is not a loop variable
```

## `Tensile/LoopModel/emit/nest.py:99`

```
# here and `_group_guard`'s predicate would reference a name that is not in scope.
```

## `Tensile/LoopModel/emit/nest.py:114`

```
# drain chunk index i = T − M + t (paper), as a symbolic Expr over the trip symbol T.
```

## `Tensile/LoopModel/emit/nest.py:116`

```
# deepest operand drops first: a copy is present iff off_p ≤ M−1−t.
```

## `Tensile/LoopModel/emit/reads.py._read_groups`

```
How many instructions this operand's read is split into, and which register groups
        each one fills.
        
```

## `Tensile/LoopModel/emit/reads.py._first_touch_modes`

```
Enclosing inner modes the op is invariant over — non-presence modes at an ord
        position outer to the op's read level.
        
```

## `Tensile/LoopModel/emit/reads.py._fill_read_inst`

```
One direct (unshifted) prologue read of `op` at the concrete presence-coord
        `coord_dict` (a Lemma-3d prologue member).
        
```

## `Tensile/LoopModel/emit/reads.py.fill_reads`

```
prologue read-ahead fill: pre-issue exactly the first-trip reads whose residency
        `O_r` the steady read-ahead cannot discharge in-body — the closed-form three-factor
        `preloaded_tiles` set, PER read op-class (per-operand-presence, so a broadcast
        
```

## `Tensile/LoopModel/emit/reads.py:1`

```
# Copyright Advanced Micro Devices, Inc., or its affiliates.
```

## `Tensile/LoopModel/emit/reads.py:25`

```
# The issuing agent for every instance in a single-role kernel: ρ → `Inst.a` and `Loop.role`; "an
```

## `Tensile/LoopModel/emit/reads.py:26`

```
# unspecialized kernel is the single-role special case").
```

## `Tensile/LoopModel/emit/reads.py:54`

```
# The read's awaits come from the ledger at the site (op, "read"): its RAW residency on
```

## `Tensile/LoopModel/emit/reads.py:55`

```
# whatever filled the buffer, plus any in-place refill WAR.
```

## `Tensile/LoopModel/emit/reads.py:57`

```
# PLR read-ahead: the read at flat presence position P fetches P+shift
```

## `Tensile/LoopModel/emit/reads.py:59`

```
# The fold converts the shifted presence point back to the tile that names its carrier
```

## `Tensile/LoopModel/emit/reads.py:60`

```
# group `read_fold` is the one derivation, shared with `_first_touch_modes`.
```

## `Tensile/LoopModel/emit/reads.py:65`

```
# `advance=shift` records the shape on the node: 0 = inplace (in place, one line before its
```

## `Tensile/LoopModel/emit/reads.py:66`

```
# own wmma), non-zero = prefetch (hoisted).
```

## `Tensile/LoopModel/emit/reads.py:76`

```
# A chunk-level read has no enclosing inner mode.
```

## `Tensile/LoopModel/emit/reads.py:81`

```
# A partially folded axis is at or inner to the read level (it is still in presence), so it
```

## `Tensile/LoopModel/emit/reads.py:82`

```
# is not in the slice above; it is added here with its modulus.
```

## `Tensile/LoopModel/emit/reads.py:103`

```
# ONE read, ONE position.
```

## `Tensile/LoopModel/emit/reads.py:115`

```
# prologue fill: the registers are empty (nothing has read them yet), so there is no
```

## `Tensile/LoopModel/emit/reads.py:116`

```
# vacating read and the rotation/in-place WAR does not apply — the same reasoning
```

## `Tensile/LoopModel/emit/reads.py:131`

```
# `preloaded_tiles` derives the per-group `dr_g` itself: prologue depth = steady
```

## `Tensile/LoopModel/emit/reads.py:132`

```
# advance = dr_g), so an inplace group fills nothing here and its steady advance is 0 —
```

## `Tensile/LoopModel/fuse.py:1`

```
# Copyright Advanced Micro Devices, Inc., or its affiliates.
```

## `Tensile/LoopModel/fuse.py:5`

```
# operand names are θ op-class names, which coincide with the scaffold's `tensorChar` ('A', 'B',
```

## `Tensile/LoopModel/fuse.py:6`

```
# 'mxsa', 'mxsb') — the two vocabularies agree here, so no translation is needed.
```

## `Tensile/LoopModel/fuse.py:14`

```
# 6 (`all4`) — discriminator, not a shipping grouping.
```

## `Tensile/LoopModel/fuse.py:16`

```
# 7 (`mix31`) — discriminator.
```

## `Tensile/LoopModel/fuse.py:23`

```
# : wave shares per group member — how many waves of the group's slice each member gets, as.
```

## `Tensile/LoopModel/fuse.py:35`

```
# why `5` is parity AND NOT `[[1,1],[1,1]]`.
```

## `Tensile/LoopModel/fuse.py:56`

```
# : The groupings the scaffold can realize today.
```

## `Tensile/LoopModel/geometry.py.axis_tiles`

```
Number of compute-atom tiles along one axis = size of logical_divide(tile, atom) in
    element space.
    
```

## `Tensile/LoopModel/geometry.py._quantum_candidate_axes`

```
`[(name, extent),...]` in ord order (outer→inner) — the axes a per-lane hop's quantum may
    fold, with the flat-index strides they carry.
    
```

## `Tensile/LoopModel/geometry.py.quantum_factors`

```
`{axis: factor}` — how many consecutive coordinates of each axis one instruction of a
    per-lane hop carries Precondition Q).
    
```

## `Tensile/LoopModel/geometry.py.hop_broadcast`

```
The intra-iteration axes ONE instance of this hop does not vary over — presence PER hop
    line 71, move 1 "applies per hop independently", line 583).
    
```

## `Tensile/LoopModel/geometry.py.hop_fold`

```
`{axis: factor}` for the axes this hop's quantum folds only partially (`1 < factor <
    extent`).
    
```

## `Tensile/LoopModel/geometry.py.free_tiles`

```
Tiles the operand fans over = product of extents of its own free presence modes
    (`theta.free_modes(op)` = presence minus reduction,.
    
```

## `Tensile/LoopModel/geometry.py.k_tiles`

```
Reduction substeps per kiter = product of extents of the reduction modes
    (`theta.reduction_modes()` = inner minus pres(output), line 82).
    
```

## `Tensile/LoopModel/geometry.py.reduction_tiles`

```
Reduction substeps this operand traverses = product of extents of its presence modes that
    are reduction modes.
    
```

## `Tensile/LoopModel/geometry.py.wmma_grid`

```
(free-tiles of input-0, free-tiles of input-1, k_tiles) for one kiter — the matmul grid
    from the two matrix input op-classes "matmul-primitive": two register fragments in).
    
```

## `Tensile/LoopModel/geometry.py.frag_regs`

```
Registers one fragment occupies per lane, derived by recasting the per-lane element
    layout into register-sized groups and taking its coshape.
    
```

## `Tensile/LoopModel/geometry.py.readahead_level_of`

```
`(level_name, level_extent, region_span)` — the ONE derivation of the level a local read-
    ahead lives on, from an `ord`'s inner modes, the region-mode names, and (for a per-
    operand answer) the op-class's broadcast axes.
    
```

## `Tensile/LoopModel/geometry.py._rate_broadcast`

```
The axes a register group is constant over for its rate/live-peak — the "two-map
    reading".
    
```

## `Tensile/LoopModel/geometry.py._group_value_seq`

```
The distinct value id a rate-group holds at each step = projection of the step coord onto
    the modes the group is NOT nullspace over region modes excluded (per-group, per-fragment
    reading).
    
```

## `Tensile/LoopModel/geometry.py.group_rate`

```
R — reload rate = # distinct values ONE register group of ONE fragment holds per
    reduction chunk = product of extents of the modes its intra-fragment read time-map is
    nonzero over (non-`nullspace`, region modes excluded).
    
```

## `Tensile/LoopModel/geometry.py._reg_read_off`

```
The register read-prefetch depth `off(read, ·)` = the PLR read-ahead, read from the
    first-class off_map: δ lives in off_map, not on the hop).
    
```

## `Tensile/LoopModel/geometry.py._free_tile_index_seq`

```
The operand's own free-axis tile index at each step = the mixed-radix combine of this
    operand's free presence modes (`theta.free_names(op)` = presence minus reduction, in ord
    order.
    
```

## `Tensile/LoopModel/geometry.py.resident_free_tiles`

```
The number of the operand's free-axis tiles simultaneously resident under `ord` — the
    live-peak of its free-tile index over the traversal line 279/285-288/302).
    
```

## `Tensile/LoopModel/geometry.py.operand_emitted_regs`

```
VGPRs the allocator will actually reserve for this operand — `region x max_g W_g x |fan|
    x frag_regs`.
    
```

## `Tensile/LoopModel/geometry.py._region_count`

```
Regions the operand's registers are replicated over — the wave's region span, not the
    tile's.
    
```

## `Tensile/LoopModel/geometry.py.operand_tile_bytes`

```
shared bytes for one operand's whole tile across the wave — presence-derived, same fix
    and same reason as `operand_buffer_regs` above.
    
```

## `Tensile/LoopModel/geometry.py:1`

```
# Copyright Advanced Micro Devices, Inc., or its affiliates.
```

## `Tensile/LoopModel/geometry.py:19`

```
# Tile counts — tile extent ÷ compute atom, in element space (the honest extent source).
```

## `Tensile/LoopModel/geometry.py:24`

```
# tile count = latalg.n_tiles (was logical_divide(...).shape[1]).
```

## `Tensile/LoopModel/geometry.py:30`

```
# Per-operand access layout — the honest spatial object.
```

## `Tensile/LoopModel/geometry.py:39`

```
# An agent-served region axis is broadcast for every data consumer: the wave reads, computes
```

## `Tensile/LoopModel/geometry.py:40`

```
# and accumulates inside its own region.
```

## `Tensile/LoopModel/geometry.py:62`

```
# carrier = t // Φ either way: Φ tiles ride one instruction, and the instruction's ordinal is
```

## `Tensile/LoopModel/geometry.py:63`

```
# the register base.
```

## `Tensile/LoopModel/geometry.py:67`

```
# The group does not divide evenly over the sub-agents, so some agent would carry a
```

## `Tensile/LoopModel/geometry.py:68`

```
# fraction of a tile.
```

## `Tensile/LoopModel/geometry.py:70`

```
# distributed: the Φ tiles are spread over `rho` sub-agents, so `rho` of them share ONE
```

## `Tensile/LoopModel/geometry.py:71`

```
# register and the lane decides which — a broadcast, expressed by collapsing the slot
```

## `Tensile/LoopModel/geometry.py:91`

```
# The operand's flat tile index, ord order with the innermost axis fastest — the same
```

## `Tensile/LoopModel/geometry.py:92`

```
# flattening `tile_flat` uses, so `carrier` is evaluated on the coordinate it was written for.
```

## `Tensile/LoopModel/geometry.py:106`

```
# every coordinate rides the carrier of its own group's leader
```

## `Tensile/LoopModel/geometry.py:128`

```
# : the axes ONE instruction spans are absent from the hop's presence, exactly as a bulk
```

## `Tensile/LoopModel/geometry.py:129`

```
# copy's inner axes are.
```

## `Tensile/LoopModel/geometry.py:131`

```
# The region axis enters the hop's presence only when the tile is actually split.
```

## `Tensile/LoopModel/geometry.py:151`

```
# Free / reduction extents / wmma grid — presence-derived, no operand-name/role literals.
```

## `Tensile/LoopModel/geometry.py:198`

```
# Fragment register width — via latalg.recast_count (integer ceil), not a layout op.
```

## `Tensile/LoopModel/geometry.py:213`

```
# Register groups: what one group owns, how fast it turns over, how much is live
```

## `Tensile/LoopModel/geometry.py:231`

```
# A collapsed region-leading axis means the read-ahead crosses regions.
```

## `Tensile/LoopModel/geometry.py:288`

```
# partially folded: the axis keeps `extent // q` presence points, so the fan contributes
```

## `Tensile/LoopModel/geometry.py:289`

```
# that many generations, not `extent`.
```

## `Tensile/LoopModel/geometry.py:304`

```
# name-vs-reload is an `ord` fact, NOT A structural ONE.
```

## `Tensile/LoopModel/geometry.py:325`

```
# the rate counts generations, so A folded axis contributes `N/q`, NOT `N`.
```

## `Tensile/LoopModel/geometry.py:337`

```
# the fan contributes buffers, NOT tiles.
```

## `Tensile/LoopModel/geometry.py:348`

```
# `off` overrides, so a caller can ask "what would the peak be at depth d?" instead of only.
```

## `Tensile/LoopModel/geometry.py:351`

```
# the joint (free-tile, rotation-slot) name space not one collapsed ring.
```

## `Tensile/LoopModel/geometry.py:354`

```
# the free tile modes the rate projection drops — these name storage, they do not reload it
```

## `Tensile/LoopModel/geometry.py:365`

```
# `off` is in reduction substeps, the name'S traversal is in positions.
```

## `Tensile/LoopModel/geometry.py:435`

```
# No shared placement — a one-hop global->register (DTV) operand.
```

## `Tensile/LoopModel/geometry.py:449`

```
# Sizing: how many instructions a hop costs, and how many registers it occupies
```

## `Tensile/LoopModel/geometry.py:469`

```
# Register footprint — generations × width × tiles (the product,.
```

## `Tensile/LoopModel/geometry.py:485`

```
# the accumulator: one live value per output tile, held across the whole reduction — no
```

## `Tensile/LoopModel/geometry.py:486`

```
# rotating ring no read-ahead residency.
```

## `Tensile/LoopModel/geometry.py:493`

```
# |part| = # grouping-mode values in each part.
```

## `Tensile/LoopModel/geometry.py:499`

```
# `|part|` is A residency, NOT an extent — the ord half of (feature #4).
```

## `Tensile/LoopModel/geometry.py:501`

```
# footprint = region × Σ_part( W_part × |part| ) × frag_regs (paper's 4 orthogonal factors)
```

## `Tensile/LoopModel/geometry.py:534`

```
# Instruction counts.
```

## `Tensile/LoopModel/geometry.py:562`

```
# The compute grid.
```

## `Tensile/LoopModel/ir.py.Counter`

```
completion classes: "a nameable set of operations a synchronization primitive can wait
    on".
    
```

## `Tensile/LoopModel/ir.py.Await`

```
A named-dependency discharge point: the value `dep` on completion class `counter` must be
    complete, within proc-scope `scope`, before the owning instance issues.
    
```

## `Tensile/LoopModel/ir.py.Inst`

```
One op-class instance at its presence level: a `Load` or `Mma` payload, its logical
    `placement` (symbolic slots), issuing `agent`, and named `awaits`.
    
```

## `Tensile/LoopModel/ir.py.Branch`

```
Explicit register-rotation control (user decision): a switch on `selector = mode % S`,
    one `arm` per residue.
    
```

## `Tensile/LoopModel/ir.py.Peel`

```
A prologue/drain peel node: the hoisted first-`k` (`kind='prologue'`) or last-`k`
    (`kind='drain'`) iterations of loop `mode`, a straight-line body (no back-edge), contents
    fixed by `off`.
    
```

## `Tensile/LoopModel/ir.py.Cond`

```
A first-class conditional over the reduction trip count — the peel's validity guard
    hypothesis `T_ℓ ≥ off(·,ℓ)`).
    
```

## `Tensile/LoopModel/ir.py.group`

```
The tiles sharing `t`'s carrier — the coordinates ONE instruction serves (its
        preimage).
        
```

## `Tensile/LoopModel/ir.py.semantic`

```
Identity for folding a run of equal-semantic ops: ops equal here differ only by
        index/offset.
        
```

## `Tensile/LoopModel/ir.py._runtime`

```
True iff Expr `e` references a symbol absent from `env` (a runtime problem
        dimension).
        
```

## `Tensile/LoopModel/ir.py.at`

```
Concrete Placement with each slot Expr evaluated to its buffer value but preserving
        the ring modulus (`mod`).
        
```

## `Tensile/LoopModel/ir.py.bound_env`

```
`env` with `mode` bound to `value` iff `value` is statically resolvable in `env`
        (prologue `j−M` = a constant → binds a concrete int).
        
```

## `Tensile/LoopModel/ir.py:1`

```
# Copyright Advanced Micro Devices, Inc., or its affiliates.
```

## `Tensile/LoopModel/ir.py:11`

```
# Spaces and completion counters.
```

## `Tensile/LoopModel/ir.py:37`

```
# The single extension point: a (src, dst) hop maps to the completion class that tracks it.
```

## `Tensile/LoopModel/ir.py:48`

```
# The movement quantum — how tiles merge into one instruction
```

## `Tensile/LoopModel/ir.py:92`

```
# The instruction ops.
```

## `Tensile/LoopModel/ir.py:105`

```
# shape: the read-ahead advance this transfer carries, in flat presence positions.
```

## `Tensile/LoopModel/ir.py:107`

```
# movement quantum: an `ir.QuantumMap` — `carrier(t)` (which instruction carries tile `t`).
```

## `Tensile/LoopModel/ir.py:130`

```
# "A0(k0)*B0(k0)" — the 8-wmma group issued together
```

## `Tensile/LoopModel/ir.py:134`

```
# ops fold together iff they belong to the same wmma block (the hardware group)
```

## `Tensile/LoopModel/ir.py:140`

```
# Rolled θ-nest IRChapter 5 "emit IR").
```

## `Tensile/LoopModel/ir.py:165`

```
# (terms, add, ((coef, div, ext), …)) →.
```

## `Tensile/LoopModel/ir.py:167`

```
# + (add + Σ coef·env[m]) // div over carry_terms=((mode,coef),…).
```

## `Tensile/LoopModel/ir.py:229`

```
# generic comparison dispatch — one table, evaluated the same way everywhere (no per-call ifs).
```

## `Tensile/LoopModel/ir.py:253`

```
# `free_vars` is the ONE authority on which names an Expr reads (its own docstring says
```

## `Tensile/LoopModel/ir.py:254`

```
# so); re-listing them here is what left `digits`-only expressions looking constant and
```

## `Tensile/LoopModel/ir.py:295`

```
# `keep_unresolved` is opt-in and off for the codegen path, which must always get a
```

## `Tensile/LoopModel/ir.py:296`

```
# concrete Placement.
```

## `Tensile/LoopModel/ir.py:308`

```
# see ADR 0006.  See ADR 0006.
```

## `Tensile/LoopModel/ir.py:358`

```
# : `anchor` — position, never value.
```

## `Tensile/LoopModel/ir.py:366`

```
# display extent line 339, Q19 adopted).
```

## `Tensile/LoopModel/ir.py:371`

```
# runtime-trip, peeled level (the shared-ring / reduction axis); False
```

## `Tensile/LoopModel/ir.py:372`

```
# = a statically-unrolled inner mode.
```

## `Tensile/LoopModel/ir.py:408`

```
# outer-level residue binding, the peel's Bind) — the structural
```

## `Tensile/LoopModel/ir.py:409`

```
# marker, not a name.
```

## `Tensile/LoopModel/ir.py:451`

```
# construction peels the pipelined loop) — the structural marker for
```

## `Tensile/LoopModel/ir.py:452`

```
# walkers, consistent with Loop/Branch.
```

## `Tensile/LoopModel/ir.py:470`

```
# walker uses (consistent with Loop/Branch/Peel `outer`).
```

## `Tensile/LoopModel/latalg.py.tile_split`

```
Stands in for `logical_divide(Layout((n,),(1,)), Layout((t,),(1,))).shape` on a
    contiguous 1-D layout.
    
```

## `Tensile/LoopModel/latalg.py.recast_count`

```
Stands in for `recast(Layout((elems,),(1,)), scale).size` = ceil(elems / scale): packing
    `elems` sub-units into groups of `scale` (e.g.
    
```

## `Tensile/LoopModel/latalg.py:1`

```
# Copyright Advanced Micro Devices, Inc., or its affiliates.
```

## `Tensile/LoopModel/ledger.py.Endpoint`

```
One end of an obligation — which op-class, at which placement or site, at which
    coordinate.
    
```

## `Tensile/LoopModel/ledger.py.discharge_once`

```
Drop an `Await` that a preceding instruction in the same execution path already
    established.
    
```

## `Tensile/LoopModel/ledger.py.walk_insts`

```
Pre-order walk of the rolled θ-nest yielding every `Inst` (one per op-class per presence
    level — NOT enumerated).
    
```

## `Tensile/LoopModel/ledger.py.check_ledger_discharged`

```
empty-ledger gate, PER-edge: "each producer→consumer obligation is covered by an `Await`
    at the consuming instance" — and the per-edge gate can reject a discharge mis-placed onto
    the wrong instance, which coarse counter-coverage cannot).
    
```

## `Tensile/LoopModel/ledger.py:1`

```
# Copyright Advanced Micro Devices, Inc., or its affiliates.
```

## `Tensile/LoopModel/ledger.py:66`

```
# A global→shared hop under Φ is ONE obligation for the whole movement — both.
```

## `Tensile/LoopModel/ledger.py:72`

```
# ONE obligation PER placement.
```

## `Tensile/LoopModel/ledger.py:75`

```
# in-place vs rotation.
```

## `Tensile/LoopModel/ledger.py:78`

```
# : the shared ring satisfies `S >= δ`, and at `S == δ` the refill is in place —.
```

## `Tensile/LoopModel/ledger.py:86`

```
# chunk-crossing residency "The chunk-crossing coupling").
```

## `Tensile/LoopModel/ledger.py:99`

```
# Same Φ rule as the copy-hop residency above, and for the same reason on both ends:
```

## `Tensile/LoopModel/ledger.py:100`

```
# one cooperative copy, one crossing obligation, matched by every member's crossing
```

## `Tensile/LoopModel/ledger.py:105`

```
# Read-ahead residency O_r: a first-trip read the steady body cannot discharge must be pre-
```

## `Tensile/LoopModel/ledger.py:106`

```
# issued in the prologue, else the ledger is non-empty.
```

## `Tensile/LoopModel/ledger.py:119`

```
# deterministic order.
```

## `Tensile/LoopModel/ledger.py:141`

```
# each trip re-runs the body; a boundary from outside does not survive into trip 2,
```

## `Tensile/LoopModel/ledger.py:142`

```
# and the body's own copies invalidate it anyway.
```

## `Tensile/LoopModel/ledger.py:216`

```
# `inline_outer` = descend the outer-level Branch that wraps a drain step (the peel's
```

## `Tensile/LoopModel/ledger.py:217`

```
# `Bind` binds each drain step's reduction chunk residue via a Branch(outer=True)).
```

## `Tensile/LoopModel/ledger.py:232`

```
# a nested outer Loop/Branch or a Peel is a separate trip — not inlined; handled by
```

## `Tensile/LoopModel/ledger.py:233`

```
# visit
```

## `Tensile/LoopModel/ledger.py:244`

```
# each drain step is `Bind(iter=T−M+t, body=...)`; flat() inlines the Bind
```

## `Tensile/LoopModel/ledger.py:245`

```
# body so the σ_c order gate inspects that trip's instructions.
```

## `Tensile/LoopModel/ledger.py:263`

```
# quantified over all reads line 185: `∀ r ∈ reads(gen t), ∀ w ∈ writes(gen t+S):. See.
```

## `Tensile/LoopModel/ledger.py:276`

```
# `len(body)` for an operand with no read in this trip: past every index, so the copy-
```

## `Tensile/LoopModel/ledger.py:277`

```
# first peel skeleton stays a violation exactly as it was under the existential.
```

## `Tensile/LoopModel/ledger.py:284`

```
# (b3) order, crossing RAW: the copy must be emitted before this operand's reads in the steady
```

## `Tensile/LoopModel/ledger.py:285`

```
# body, because those reads consume the chunk it fills.
```

## `Tensile/LoopModel/ledger.py:312`

```
# (b2) order, register ring: a hoisted in-place refill read must follow a wmma consuming its.
```

## `Tensile/LoopModel/ledger.py:318`

```
# A placement blanks the label of A single-group operand (`placement._read_placement`:.
```

## `Tensile/LoopModel/ledger.py:346`

```
# A refill consumes the mark.
```

## `Tensile/LoopModel/ledger.py:352`

```
# (c) read-ahead residency: each `readahead-residency` obligation requires its coord (a field
```

## `Tensile/LoopModel/ledger.py:353`

```
# on the producer endpoint) to be actually emitted as a register read in the prologue body.
```

## `Tensile/LoopModel/ledger.py:367`

```
# same structured shape the obligation carries: sorted ((mode, value), …)
```

## `Tensile/LoopModel/ledger.py:382`

```
# (b) order: per region, every WAR-refill copy must be preceded by the vacating read.
```

## `Tensile/LoopModel/operand_view.py:1`

```
# Copyright Advanced Micro Devices, Inc., or its affiliates.
```

## `Tensile/LoopModel/prefetch.py.PeelDepths`

```
The peel depth `M_ℓ` **per level** 2026-08-14 update) — ONE source of truth for the peel
    construction in `build_ir`.
    
```

## `Tensile/LoopModel/prefetch.py._leading_inner_coords`

```
The relocated instances — the paper's *"exactly `[inner-coord 0.. δ)` of the current
    outer tile"*.
    
```

## `Tensile/LoopModel/prefetch.py.preloads_for_group`

```
Build the three-factor set for one group at depth `dr_g`, restricting the grouping mode
    (if this fragment is group-split) to `gm_vals`.
    
```

## `Tensile/LoopModel/prefetch.py.loads_in_place`

```
True when the read-ahead refill writes the very name its own leaf's `wmma` is reading —
    the `inplace` case, a write-after-read, NOT a capacity violation.
    
```

## `Tensile/LoopModel/prefetch.py._read_placement`

```
Symbolic register Placement for op's read: per group, slot = the mixed-radix rate-mode
    index mod W (`ring_slot`).
    
```

## `Tensile/LoopModel/prefetch.py.copy_off`

```
`{op_name: δ}` for the copy op-classes at the reduction chunk — the staggered-peel
        input `build_ir` walks (`copy_p` present iff `off_p >= M-t` in the ramp, `off_p <=
        M-1-t` in the drain).
        
```

## `Tensile/LoopModel/prefetch.py:1`

```
# Copyright Advanced Micro Devices, Inc., or its affiliates.
```

## `Tensile/LoopModel/prefetch.py:77`

```
# (rewritten): the coupling is `off(copy,iter) >= r`; the `+1` is demanded only in the
```

## `Tensile/LoopModel/prefetch.py:78`

```
# `S=δ`-tight copy-last case, where copy-first would overwrite a slot the current trip
```

## `Tensile/LoopModel/prefetch.py:124`

```
# the read-ahead level is PER op-class so it is resolved inside the `dr` fold below
```

## `Tensile/LoopModel/prefetch.py:126`

```
# every (op-class, level) offset θ carries, as a per-op-class map — the staggered peel's input.
```

## `Tensile/LoopModel/prefetch.py:129`

```
# the sense is PER op-class, the role is PER hop axis 3).
```

## `Tensile/LoopModel/prefetch.py:145`

```
# M_ℓ = max_p off(p, ℓ), per level, in ord order.
```

## `Tensile/LoopModel/prefetch.py:152`

```
# the substep read-ahead's reach into whole chunks: M is the max over all op-classes, with the
```

## `Tensile/LoopModel/prefetch.py:153`

```
# read's substep off converted to whole iters).
```

## `Tensile/LoopModel/prefetch.py:159`

```
# the reach is derived, NOT requested.
```

## `Tensile/LoopModel/prefetch.py:163`

```
# keep ord order (dict insertion above already follows it, except for the chunk bump)
```

## `Tensile/LoopModel/prefetch.py:215`

```
# lemma 3d's outer factor is A presence point, NOT A tile.
```

## `Tensile/LoopModel/prefetch.py:220`

```
# A free sibling in `outer` is enumerated, NOT pinned line 644).
```

## `Tensile/LoopModel/prefetch.py:227`

```
# per-mode value ranges for the inner factor; the grouping mode is restricted to gm_vals.
```

## `Tensile/LoopModel/prefetch.py:234`

```
# the grouping mode may be outer (not inner); the loop below restricts it there per value.
```

## `Tensile/LoopModel/prefetch.py:260`

```
# A broadcast-outer operand primes its whole name SET.
```

## `Tensile/LoopModel/prefetch.py:292`

```
# single-group: one depth for the whole operand, still derived (a single-group fragment can be
```

## `Tensile/LoopModel/prefetch.py:293`

```
# Shape B too — that is exactly the term-(ii) operand of a K-innermost order, which has no
```

## `Tensile/LoopModel/prefetch.py:297`

```
# width-split: union per group, each at its own derived dr_g over its grouping-mode values.
```

## `Tensile/LoopModel/prefetch.py:313`

```
# solved against the walk, NOT guessed Q37).
```

## `Tensile/LoopModel/prefetch.py:318`

```
# capacity first (cheap,'s `W >= L`), then assignment (the actual admission test).
```

## `Tensile/LoopModel/prefetch.py:321`

```
# the assignment verdict is taken over every group of the operand, not just this one.
```

## `Tensile/LoopModel/prefetch.py:323`

```
# admission is "is there A legal position", NOT "is the default position legal".
```

## `Tensile/LoopModel/prefetch.py:386`

```
# the reload traversal, not the presence set — see `reload_modes` for why this is a shared
```

## `Tensile/LoopModel/prefetch.py:387`

```
# function rather than an inline filter, and for the status of the free-sibling exclusion.
```

## `Tensile/LoopModel/prefetch.py:389`

```
# presence points, NOT tiles /.
```

## `Tensile/LoopModel/prefetch.py:392`

```
# the same level `prefetch_distance_for` advances along — this guard and that conversion are
```

## `Tensile/LoopModel/prefetch.py:393`

```
# two halves of one rule, so they must test the same axis.
```

## `Tensile/LoopModel/prefetch.py:399`

```
# inner = the ord modes strictly inner to the reduction substep — a property of `ord`, not of
```

## `Tensile/LoopModel/prefetch.py:400`

```
# this operand (thesplits the whole order, then intersects each factor with presence).
```

## `Tensile/LoopModel/prefetch.py:425`

```
# read-ahead slot = the rate index of the shifted coord, not the unshifted slot plus `dr`.
```

## `Tensile/LoopModel/prefetch.py:430`

```
# shared source buffer (`src_slot`): which LDS multibuffer generation the read's data lives in.
```

## `Tensile/LoopModel/prefetch.py:434`

```
# the shared ring rotates over the reduction chunk.
```

## `Tensile/LoopModel/prefetch.py:437`

```
# ONE shift for the whole read: the coord and the slot must advance together.
```

## `Tensile/LoopModel/prefetch.py:486`

```
# the inner axis contributes the largest divisor of its extent that still fits the load
```

## `Tensile/LoopModel/prefetch.py:495`

```
# an outer axis can only be added once the inner one is spanned whole
```

## `Tensile/LoopModel/presence.py.prefetch_axis_name`

```
The level `off(read, ·)` lives on — `theta.readahead_level`, the outermost mode of the
    region's inner nest.
    
```

## `Tensile/LoopModel/presence.py.op_class_level`

```
The level an op-class is emitted at — its home level line 523, "placed by its own
    presence set").
    
```

## `Tensile/LoopModel/presence.py.group_value_span`

```
The `grouping_mode` value subset a register group owns: a group is a partition of the
    grouping tile-mode's values).
    
```

## `Tensile/LoopModel/presence.py.varying_axes`

```
The inner modes op's shared→register read varies over (its presence set, — the non-
    broadcast inner modes.
    
```

## `Tensile/LoopModel/presence.py.sibling_free_axes`

```
`op`'s region modes that enumerate storage-disjoint siblings rather than reload — the
    free-axis ones.
    
```

## `Tensile/LoopModel/presence.py.ring_slot`

```
The rotation slot Expr for a group of width `d`: the mixed-radix index over the group's
    rate modes (innermost = fastest, coef 1; next = ×extent; …) taken mod `d`.
    
```

## `Tensile/LoopModel/presence.py._reduction_rate_mode`

```
The single innermost reduction rate mode (role 'k') of `op`'s register group, as (name,
    extent), or None.
    
```

## `Tensile/LoopModel/presence.py.axis_strides`

```
`stride_ord(m)` for a set of inner modes: the product of extents of the modes inner to
    `m` in `ord`, restricted to `modes` — `ord` is a permutation, so the traversal over that
    mode subset is a monotone mixed-radix counter; the innermost has stride 1).
    
```

## `Tensile/LoopModel/presence.py.position_terms`

```
`(terms, L)` for the flat presence position `P = Σ stride(m)·point(m)`, as an `Expr`
    linear form over the loop variables.
    
```

## `Tensile/LoopModel/presence.py:1`

```
# Copyright Advanced Micro Devices, Inc., or its affiliates.
```

## `Tensile/LoopModel/presence.py:59`

```
# ONE derivation, asked of the read hop.
```

## `Tensile/LoopModel/presence.py:78`

```
# the `{lo}{hi}` partition is A sibling enumerator too, and theists it in the same.
```

## `Tensile/LoopModel/presence.py:85`

```
# The split is live but its mode is gone: the free tiles moved to another axis.
```

## `Tensile/LoopModel/presence.py:132`

```
# radix chain over the rate modes, innermost fastest — the same order `_shifted_rate_slot`
```

## `Tensile/LoopModel/presence.py:133`

```
# builds its digit fields in, read from the same `ring_axes` extents.
```

## `Tensile/LoopModel/registers.py.chunks_crossed`

```
`r` — how many reduction chunks ahead `op`'s substep read-ahead reaches "The chunk-
    crossing coupling").
    
```

## `Tensile/LoopModel/registers.py.reloads_whole_set`

```
Is every one of `op`'s registers live across a full outer pass, so all must reload at
    once?
    
```

## `Tensile/LoopModel/registers.py._uses_before_load`

```
`{name: [(t, wanted, held)]}` — every steady-chunk consume that finds the wrong
    generation in its slot.
    
```

## `Tensile/LoopModel/registers.py._overwrites_live_register`

```
`{name: [last consume,...]}` for every steady-chunk write that lands on a name whose
    previous value is still consumed at or after the write — the raw finding both the verdict
    and the anchor read.
    
```

## `Tensile/LoopModel/registers.py.reload_positions`

```
`{register name: coord}` -- the loop position each reload must follow, or None if there
    is no legal one (which is the `clobber` verdict).
    
```

## `Tensile/LoopModel/registers.py:1`

```
# Copyright Advanced Micro Devices, Inc., or its affiliates.
```

## `Tensile/LoopModel/registers.py:48`

```
# the reload traversal, NOT the presence SET — the same set `_readahead_shift` walks.
```

## `Tensile/LoopModel/registers.py:53`

```
# the shift advances along the read-ahead level — the axis `off(read, ·)` is written on.
```

## `Tensile/LoopModel/registers.py:57`

```
# A broadcast-outer operand reads ahead by A whole trip, NOT by A substep.
```

## `Tensile/LoopModel/registers.py:75`

```
# an extent-1 mode is nullspace, the same rule `ring_axes` states: it has one value, so the
```

## `Tensile/LoopModel/registers.py:76`

```
# traversal neither repeats under it nor advances through it, and it cannot make an outer mode
```

## `Tensile/LoopModel/registers.py:97`

```
# two spaces, ONE walk.
```

## `Tensile/LoopModel/registers.py:125`

```
# the anchor is A guard on the issuing step, so it filters writes only — the wmmas keep
```

## `Tensile/LoopModel/registers.py:126`

```
# consuming at every step regardless of where the refill was moved to.
```

## `Tensile/LoopModel/registers.py:182`

```
# starvation is the authority.
```

## `Tensile/LoopModel/registers.py:188`

```
# A later consume is an unrescuable race at this position, and `sigma_c` is NOT a way.
```

## `Tensile/LoopModel/registers.py:207`

```
# the anchor must lie in the same trip as the refill.
```

## `Tensile/LoopModel/registers.py:211`

```
# ONE Inst has ONE position.
```

## `Tensile/LoopModel/registers.py:221`

```
# the fixpoint.
```

## `Tensile/LoopModel/render.py._coord_of`

```
The op's coord as text: symbolic mode names in the rolled view, concrete indices in the
    unrolled view (env supplies mode → value).
    
```

## `Tensile/LoopModel/render.py._reg_slots`

```
The register residence of a wmma source Placement: `[buf..]` (single group) or
    `[lo=buf..,hi=buf..]` (width-split fragment) — the group labels + slots, no shared
    `src_slot`.
    
```

## `Tensile/LoopModel/render.py._await_line`

```
A named-dependency discharge point: the named dep is the primitive; the lowered count (if
    resolved) shows in parens as a debug aid.
    
```

## `Tensile/LoopModel/render.py.render_ir`

```
Print the rolled θ-nest: `for <mode> in range(N):` loop headers, each op-class ONE line
    at its presence level with its symbolic slot, and named Await points.
    
```

## `Tensile/LoopModel/render.py.render_unrolled`

```
Expand the rolled nest into the concrete interleaved stream: walk the loop nest with a
    live environment {mode: index}, and at each innermost leaf emit that step's fresh reads
    then its wmma (consume-then-refill), evaluating the symbolic slot Exprs to concrete
    
```

## `Tensile/LoopModel/render.py:1`

```
# Copyright Advanced Micro Devices, Inc., or its affiliates.
```

## `Tensile/LoopModel/render.py:19`

```
# Per-op formatting.
```

## `Tensile/LoopModel/render.py:47`

```
# A Cond's meaning is its generic `kind` (backend-agnosticvocabulary; the core sets this, NOT a
```

## `Tensile/LoopModel/render.py:48`

```
# TensileLite scaffold name).
```

## `Tensile/LoopModel/render.py:76`

```
# `{ax}=(formula)` — the data index this read accesses along `ax`, a read-ahead shift
```

## `Tensile/LoopModel/render.py:77`

```
# of the loop var (e.g.
```

## `Tensile/LoopModel/render.py:108`

```
# a copy's coord must be shown, not just its op-class name.
```

## `Tensile/LoopModel/render.py:123`

```
# the wmma consumes a per-operand register residence (a {operand: Placement} dict, same slots
```

## `Tensile/LoopModel/render.py:124`

```
# the producing ds_read deposited).
```

## `Tensile/LoopModel/render.py:132`

```
# the operand names come off the node.
```

## `Tensile/LoopModel/render.py:161`

```
# Raw dump — the actual IR dataclasses, no interpretation by the renderer.
```

## `Tensile/LoopModel/render.py:211`

```
# rolled view — the IR as it is stored: a loop nest, each op-class once.
```

## `Tensile/LoopModel/render.py:227`

```
# multi-body: a sequence of sub-loops, each over its own half-open range — so
```

## `Tensile/LoopModel/render.py:228`

```
# each gets its own header (a binding when the range is a single value), and
```

## `Tensile/LoopModel/render.py:233`

```
# Branch is a register-rotation selector, NOT a guard: `rotate buf = mode % S`,
```

## `Tensile/LoopModel/render.py:234`

```
# each arm labelled by the buffer residue it selects register ring).
```

## `Tensile/LoopModel/render.py:263`

```
# unrolled view — expand the leaves, interleave read;wmma (a view, 9a).
```

## `Tensile/LoopModel/render.py:276`

```
# outer runtime loop: show one representative iteration
```

## `Tensile/LoopModel/render.py:281`

```
# each sub-body over its own range multi-body)
```

## `Tensile/LoopModel/render.py:291`

```
# pinned-residue branch (a peel-unrolled reduction substep): the arm key is the
```

## `Tensile/LoopModel/render.py:292`

```
# mode's concrete value — bind it into env so nested coords/slots evaluate to
```

## `Tensile/LoopModel/render.py:300`

```
# a peel-step iter binding: push the bound value into env (concrete for prologue; a
```

## `Tensile/LoopModel/render.py:301`

```
# symbolic `T−M+t` drain value stays unbound — shown, but the body's `iter`-slots
```

## `Tensile/LoopModel/render.py:305`

```
# honor a resolved guard in this concrete env (first-touch `mode==0`, drain nll `k
```

## `Tensile/LoopModel/render.py:306`

```
# < n_s−dr`) — the unrolled stream shows only the arm that actually executes.
```

## `Tensile/LoopModel/render.py:318`

```
# a Load carries a single Placement (evaluate it to concrete slots for this env); a
```

## `Tensile/LoopModel/render.py:319`

```
# wmma carries a PER-operand {operand: Placement} dict (consume-side reg residence)
```

## `Tensile/LoopModel/render.py:321`

```
# pin only what this env can resolve.
```

## `Tensile/LoopModel/render.py:340`

```
# Geometry table — every derived number, checkable by hand.
```

## `Tensile/LoopModel/render.py:345`

```
# three columns do NOT mean what their headings suggest, and each was measured misleading:
```

## `Tensile/LoopModel/render.py:365`

```
# register-ring widths W per rate-group — the register multibuffer, distinct from S_lds
```

## `Tensile/LoopModel/render.py:391`

```
# the downgrade, named.
```

## `Tensile/LoopModel/render.py:406`

```
# levels: the full ord permutation, outer→inner, marking the outer pipelined levels (extent
```

## `Tensile/LoopModel/render.py:407`

```
# symbolic/0 → the reduction chunk / persist) vs the inner unrolled modes (concrete extent).
```

## `Tensile/LoopModel/render.py:416`

```
# in ord order, like every other list in this block.
```

## `Tensile/LoopModel/render.py:420`

```
# off: the first-class retime matrix {(op, role, level): δ} — per (op-class × level), where the
```

## `Tensile/LoopModel/render.py:421`

```
# op-class is (operand, hop role).
```

## `Tensile/LoopModel/render.py:429`

```
# the legend must reproduce the table, and this one did not.
```

## `Tensile/LoopModel/render.py:434`

```
# shape, per register group.
```

## `Tensile/LoopModel/render.py:455`

```
# the asymmetry, spelled out.
```

## `Tensile/LoopModel/render.py:462`

```
# the region count, not just the axes.
```

## `Tensile/LoopModel/render.py:471`

```
# The decision table (`schedule.Schedule`) — what the decoder decided, per operand.
```

## `Tensile/LoopModel/schedule.py._assignment_width`

```
Narrowest `W` in `[lo, hi]` whose rotation map admits the requested read-ahead, else
    `lo`.
    
```

## `Tensile/LoopModel/schedule.py.build_S`

```
The full depth map: derived register floors plus the off-driven cross-iteration shared
    depth.
    
```

## `Tensile/LoopModel/schedule.py:1`

```
# Copyright Advanced Micro Devices, Inc., or its affiliates.
```

## `Tensile/LoopModel/schedule.py:70`

```
# one Inst has one position, so the per-name anchors collapse to the guard they share
```

## `Tensile/LoopModel/schedule.py:102`

```
# The register depth map S -- the first decision everything downstream reads
```

## `Tensile/LoopModel/schedule.py:108`

```
# A DTV operand has no shared placement (`global->register` only), so the substep read-ahead
```

## `Tensile/LoopModel/schedule.py:109`

```
# this widens for does not exist and the off lookup is meaningless.
```

## `Tensile/LoopModel/schedule.py:112`

```
# the request is `off(read, substep)` — what PLR SET — not `prefetch_depth`, which returns
```

## `Tensile/LoopModel/schedule.py:113`

```
# `S_shared`, the LDS ring size (a different quantity; its own docstring says so).
```

## `Tensile/LoopModel/schedule.py:117`

```
# size for the requested depth — `PLR` is an input, `W` is derived.
```

## `Tensile/LoopModel/schedule.py:133`

```
# clamp, do NOT reject.
```

## `Tensile/LoopModel/schedule.py:135`

```
# `'floor'` is the old spelling of `'overlap'`, accepted, NOT rejected.
```

## `Tensile/LoopModel/schedule.py:137`

```
# `'pipeline'` = `max(L_pf, min(2, R))` — `L_pf` with a second floor of 2, and that floor
```

## `Tensile/LoopModel/schedule.py:138`

```
# is.
```

## `Tensile/LoopModel/schedule.py:140`

```
# the width must satisfy the assignment constraint too, not just the capacity peaks.
```

## `Tensile/LoopModel/schedule.py:153`

```
# reduction — no mainloop rotating ring its footprint is the separate
```

## `Tensile/LoopModel/schedule.py:154`

```
# accumulator term not a multibuffer depth entry.
```

## `Tensile/LoopModel/sigma_c.py._movement`

```
The movement `Inst` node `n` denotes, looking through a rolled region loop and through a
    single-armed guard.
    
```

## `Tensile/LoopModel/sigma_c.py._rebuilt`

```
`n` with σ_c applied to each of its sub-bodies (the freeze holds at every level, not just
    the region body: an in-place register refill sits at its operand's read level, inside the
    nest).
    
```

## `Tensile/LoopModel/sigma_c.py:1`

```
# Copyright Advanced Micro Devices, Inc., or its affiliates.
```

## `Tensile/LoopModel/sigma_c.py:56`

```
# an anchored read is deferred too, whatever its WAR class.
```

## `Tensile/LoopModel/sigma_c.py:65`

```
# A copy whose chunk this trip's read-ahead crosses into (`off == r`, is a RAW producer for
```

## `Tensile/LoopModel/sigma_c.py:66`

```
# those reads and must lead them — never deferred.
```

## `Tensile/LoopModel/theta.py.Resort`

```
One `resort(mode → agent)` move 2): a looped mode traversed by the agents instead of by
    the coordinate.
    
```

## `Tensile/LoopModel/theta.py.is_outer`

```
True iff this is an outer (dynamic / runtime-trip) pipelined level — the peel and
        shared-ring axis/.
        
```

## `Tensile/LoopModel/theta.py.groups`

```
The reuse-group labels (the group is identified by its label; its slot count is
        `size_of(label)`).
        
```

## `Tensile/LoopModel/theta.py.policy_of`

```
This group's W-in-[L,R] choice: an explicit int, 'unroll' (W=R), 'overlap' (W=L_pf),
        or 'inplace' (W=L_war).
        
```

## `Tensile/LoopModel/theta.py.opclass`

```
The op-class this entry belongs to, from `origin` — meaningful for a copy entry,
        which is per-op-class by construction.
        
```

## `Tensile/LoopModel/theta.py.copy_agents`

```
How many agents cooperate on `opname`'s global→shared copy — the Φ group's wave
        share.
        
```

## `Tensile/LoopModel/theta.py.served_modes`

```
The modes the agents traverse instead of the coordinate — move 2's domain, as the
        axis classification consumers mean it.
        
```

## `Tensile/LoopModel/theta.py.span_over`

```
The `level`-level agent span of whichever resorted mode lies in `modes` — ρ as the
        movement quantum reads it: `Φ` over the tiles ρ places on the spanned agents).
        
```

## `Tensile/LoopModel/theta.py.movement_units`

```
The copy movements this θ emits, as `[(unit_key, members, n_regions)]` — the ONE
        authority for "which cooperative movements exist and how many region instances each
        has".
        
```

## `Tensile/LoopModel/theta.py.levels`

```
The full level permutation, outer→inner `levels = [iter] + [substep, m, n, …]`) —
        every looped time mode including the outer pipelined levels.
        
```

## `Tensile/LoopModel/theta.py.outer_modes`

```
The outer pipelined levels (a problem-dimension trip, `Mode.is_outer`), in ord order
        — outermost first, since `ord` is already outer→inner.
        
```

## `Tensile/LoopModel/theta.py.reduction_chunk_mode`

```
The outer level the reduction chunk advances on — the innermost outer level, since
        `ord` is outer→inner and any coarser outer level (a persistent tile level, sits above
        the chunk.
        
```

## `Tensile/LoopModel/theta.py.inner_modes`

```
The intra-reduction chunk modes only (the inner K substep + the M/N fan), in ord
        order.
        
```

## `Tensile/LoopModel/theta.py.off_at`

```
The retime offset `off(op-class, level)` `off: (op-class × level) → ℤ`, a first-class
        θ field,.
        
```

## `Tensile/LoopModel/theta.py.off_of`

```
`off_at` for the op-class this `hop` of `op` defines — the form to use when the
        caller already holds the hop, so the role is read off the trajectory rather than
        restated.
        
```

## `Tensile/LoopModel/theta.py.presence_modes`

```
The inner Modes an op-class varies over (its presence set, = the inner modes minus
        its broadcast (constant-over) axes.
        
```

## `Tensile/LoopModel/theta.py.agent_served_modes`

```
The third kind of inner axis: traversed by the agents, not by the coordinate move 2
        `resort`).
        
```

## `Tensile/LoopModel/theta.py.reduction_modes`

```
`reduction_modes = inner modes minus pres(output)` line 82, paper-confirmed Q18): the
        inner Modes absent from every output op-class's presence — the contraction axes, in
        ord order.
        
```

## `Tensile/LoopModel/theta.py:1`

```
# Copyright Advanced Micro Devices, Inc., or its affiliates.
```

## `Tensile/LoopModel/theta.py:16`

```
# tile — modes and the register-axis rate-group partition
```

## `Tensile/LoopModel/theta.py:42`

```
# `parts` as an int N = split into N equal single-slot groups (the normal param form).
```

## `Tensile/LoopModel/theta.py:81`

```
# vocabulary (§2.4 `off : (op-class × level) → ℤ`).  (ADR 0007 #97)
```

## `Tensile/LoopModel/theta.py:83`

```
# hop (src, dst) -> op-class role.
```

## `Tensile/LoopModel/theta.py:95`

```
# per OP-CLASS, and NOT the same thing as the hop role (#212 axis 3).  (ADR 0007 #96)
```

## `Tensile/LoopModel/theta.py:97`

```
# how far along the memory hierarchy a space sits; a hop that decreases rank runs toward compute.
```

## `Tensile/LoopModel/theta.py:122`

```
# 's carrier group, in the paper's own vocabulary: "a carrier group is exactly a `Φ` fuse.
```

## `Tensile/LoopModel/theta.py:125`

```
# : is this hop'S storage region A property of the agent rather than of the coordinate?
```

## `Tensile/LoopModel/theta.py:160`

```
# : (`copy_agents` was here.
```

## `Tensile/LoopModel/theta.py:162`

```
# : How many storage-disjoint sibling regions this operand's free axis is cut into (1 = uncut).
```

## `Tensile/LoopModel/theta.py:164`

```
# : How many of the operand's free regions **one agent** spans /.
```

## `Tensile/LoopModel/theta.py:190`

```
# ρ — agent assignment + role partition moves 2 & 8, line 244)
```

## `Tensile/LoopModel/theta.py:193`

```
# : The agent hierarchy, ordered coarse -> fine.
```

## `Tensile/LoopModel/theta.py:200`

```
# : A resort entry's hop role.
```

## `Tensile/LoopModel/theta.py:286`

```
# including the outer reduction level `iter` (extent==0) and any
```

## `Tensile/LoopModel/theta.py:287`

```
# persistent level (also extent==0).
```

## `Tensile/LoopModel/theta.py:293`

```
# ρ move 2 `resort` + move 8 `specialize`; line 244 "agent assignment + role.
```

## `Tensile/LoopModel/theta.py:295`

```
# How many agents cooperate on one tile.
```

## `Tensile/LoopModel/theta.py:297`

```
# off: {(op_name, role, level_name): δ} —'s retime matrix over (op-class × level), where the
```

## `Tensile/LoopModel/theta.py:298`

```
# op-class is (operand, hop role) and `role` is one of theta.copy/read/store.
```

## `Tensile/LoopModel/theta.py:300`

```
# π move 7, "discharge policy — which class names each obligation"): the completion.
```

## `Tensile/LoopModel/theta.py:302`

```
# (no `reduction_modes` field: the contraction axes are derived by `reduction_names()` from the
```

## `Tensile/LoopModel/theta.py:303`

```
# accumulator's presence, line 82.
```

## `Tensile/LoopModel/theta.py:392`

```
# `Mode` is canonical here; the `*_names()` variants below are for the call sites that
```

## `Tensile/LoopModel/theta.py:393`

```
# want strings (set algebra, and the GIR name lists).  (ADR 0007 #91)
```

## `Tensile/LoopModel/theta.py:424`

```
# an agent-served axis is NOT A reduction.
```

## `Tensile/LoopModel/theta.py:447`

```
# (mode_roles() removed: the core no longer labels modes m/n/k — index-agnostic.
```

## `Tensile/LoopModel/theta.py:493`

```
# a named region with no row of its own falls back to the whole-tile placement
```

## `Tensile/LoopModel/validate.py._placement_exprs`

```
Every symbolic `Expr` an `Inst`'s placement carries: each register-group rotation slot,
    the shared source-buffer generation, and (for a wmma) the same over its per-operand
    placements.
    
```

## `Tensile/LoopModel/validate.py._unbound_placement_refs`

```
Every (Inst, mode) where the Inst's placement references a mode no enclosing binder
    binds.
    
```

## `Tensile/LoopModel/validate.py.validate_loopir`

```
Return a list of structural-defect strings for the rolled IR `ir` (empty = valid GEMM
    pseudocode).
    
```

## `Tensile/LoopModel/validate.py._end`

```
`op@at[:role][ coord]` — the Endpoint's fields, read as fields (they were a
            formatted string once, and four call sites re-split it three different ways).
            
```

## `Tensile/LoopModel/validate.py:1`

```
# Copyright Advanced Micro Devices, Inc., or its affiliates.
```

## `Tensile/LoopModel/validate.py:20`

```
# the runtime trip count is a problem dimension not a mode any binder introduces.
```

## `Tensile/LoopModel/validate.py:108`

```
# strict `T > M`.
```

## `Tensile/LoopModel/validate.py:125`

```
# M==0 degenerate: a bare steady loop is only valid when there is genuinely no peel — a
```

## `Tensile/LoopModel/validate.py:126`

```
# prologue/drain sitting at top level (outside a guard) is defect P1 (must be Cond-
```

## `Tensile/LoopModel/validate.py:134`

```
# every Inst must be reachable through the root Cond (or the bare steady loop); a top-level
```

## `Tensile/LoopModel/validate.py:135`

```
# Inst sibling would run unconditionally before the guard.
```

## `Tensile/LoopModel/validate.py:153`

```
# no Cond that pins an iter residue (iter%d == r) — the old inplace artifact
```

## `Tensile/LoopModel/validate.py:163`

```
# P3 (the real assertion) + P5 (its concrete-slot half): no placement may reference a mode that
```

## `Tensile/LoopModel/validate.py:164`

```
# no enclosing Loop/Bind/Branch binds.
```

## `Tensile/LoopModel/validate.py:194`

```
# every sub-body must satisfy σ_c independently — a multi-body steady loop has one trip per
```

## `Tensile/LoopModel/validate.py:195`

```
# body, so checking only `bodies[0]` would leave the rest unchecked.
```

## `Tensile/LoopModel/validate.py:197`

```
# the refill copies are the σ_c-ordered tail: the last top-level nodes of the body that
```

## `Tensile/LoopModel/validate.py:198`

```
# are shared-dest Loads (copies).
```

## `Tensile/LoopModel/validate.py:200`

```
# only an `S=δ` refill is required to be last point 4 is scoped to "an S=delta
```

## `Tensile/LoopModel/validate.py:201`

```
# refill copy"); a rotation-WAR copy is emitted first so this trip's chunk-crossing
```

## `Tensile/LoopModel/validate.py:210`

```
# a read/wmma nest after a refill copy breaks σ_c (refill must be last)
```

## `Tensile/LoopModel/validate.py:245`

```
# name the obligations, not just the count.
```

## `Tensile/LoopModel/validate.py:276`

```
# The `els` arm runs when T < M, so the short trip [0,T) is STRICTLY shorter than its M peeled
```

## `Tensile/LoopModel/validate.py:277`

```
# steps and T is runtime.  (ADR 0007 #98)
```

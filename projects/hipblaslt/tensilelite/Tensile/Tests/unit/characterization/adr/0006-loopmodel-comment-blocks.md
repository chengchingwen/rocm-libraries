# ADR 0006 — LoopModel reasoning extracted from inline comment blocks

Status: accepted, 2026-08-29. Companion to ADR 0005, which did the same for docstrings.

Every run of eight or more `#` lines in `LoopModel/` was an argument, not a comment: the
claim, then the reasoning, then the miscompile that produced the reasoning. The code keeps
the claim and cites this file. Extracted VERBATIM.


## `adapter/build.py`, line 166

```
    # DU (shared K): ONE axis at the FINEST split, and a coarser operand is present on it at a
    # STRIDE.  Require one to divide the other (user's restriction) — else reject, because then
    # no single axis contains both presence patterns.
    #
    # THE AXIS IS THE MAX, NOT THE MIN.  It used to be `min(A_DU, B_DU)` with the deeper operand's
    # leftover factor `res = max/min` folded into `op.split` and given no mode — so at [1,1,4,2] A
    # had FOUR regions as instructions (`sizing` issues `op.split`) and TWO as coordinates (the
    # axis extent).  Two of A's four loads were then unaddressable, un-tokened, and unprovable
    # disjoint, because every consumer that REASONS about a region works off the coordinate while
    # only the instruction count works off the number.
    #
    # THE K REGION AXIS FACTORS AS `gcd x A's residual x B's residual`, and each operand's regions
    # are the product of the modes IT owns.  A SINGLE mode of extent `max` does not work, and the
    # reason is precise rather than aesthetic.
    #
    # The DU axis is genuinely shared, so the traversal has to visit every distinct (A-region,
    # B-region) pair — `lcm(aDU, bDU)` values.  The question is not how many but HOW THEY ARE
    # NAMED.  Naming them with one mode makes the COARSER operand present at a STRIDE: its region
    # is `coord // (lcm/own)`, a division no consumer performs.  `region_of` reads coords straight
    # off `Tile.coord`, so that operand's region came back partly `None`, `_flat` returned `None`,
    # and `_split_copies` DROPPED its walk silently — both its loads landing on region 0 (#170).
    # `walk_violations` skipped it on the same rule, so the shape was invisible to the verifier
    # meant to catch it.
    #
    # Factoring instead as `gcd x own residuals` removes the division: A's region is
    # `K_split x K_splitA` and B's is `K_split x K_splitB`, both plain mixed-radix products of
    # pinned coordinates, and neither operand is ever strided-present.  An operand is simply ABSENT
    # from the OTHER's residual — ordinary presence, which the model already handles everywhere —
    # and that absence IS the first-touch guard the stride was trying to express, now stated as a
    # mode membership instead of an arithmetic condition the readers cannot see.
    #
    # NO DIVISIBILITY REQUIREMENT.  An earlier form used a two-mode chain `gcd -> lcm/gcd` with
    # each operand taking a PREFIX, which needs `aDU | bDU` (only then is the coarser count a
    # prefix product).  Per-operand residuals drop that: `aDU=2, bDU=3` gives
    # `K_splitA(2) x K_splitB(3)`, total 6, A owning 2 and B owning 3.  The guard that used to
    # reject coprime factors is gone with it.
```


## `adapter/build.py`, line 211

```
    # --- canonical 6-mode ord, ordered by the LoopOrder word (§A/§D) ----------------------
    # THE READ'S TILE AXIS IS THE WAVE'S TILES (#245).  `fanM` is `MIWaveTile` -- what ONE WAVE
    # holds -- so dividing it by the TILE's region count empties `M_inner` whenever a wave sits
    # inside one region, and the wave's tiles then have nowhere to live but `M_split`, which is
    # the COPY's axis.  That is the conflation: at `MIWaveTile2 / nsplit2` it gave
    # `M_split(2) x M_inner(1)`, so wave 0's second tile -- physically in region 0, since
    # `LraTileAssignment`'s `strideWave` puts a whole wave inside one region -- was coordinated as
    # region 1 and carried region 1's memory token.
    # Divide by the regions THE WAVE spans, not the tile's region count (#245): at
    # `wave_region_span == 1` the whole of `MIWaveTile` lands on `M_inner` and `M_split` is left to
    # the copy.  Identical to the old expression whenever the wave spans every region.
```


## `adapter/build.py`, line 234

```
    # THE READ'S REGION IS THE AGENT'S.  #245, blocked on #170 -- NOT on ρ.
    #
    # `LraTileAssignment` fixes the wave's free-axis offset at `strideWave = numTileInInst *
    # matrixInstT * VectorWidth`, so at `MIWaveGroup[2,2]` (strideWave 32, region 32 rows) wave
    # `w` lies ENTIRELY inside region `w`: the region is constant across everything one wave
    # reads, and labelling the wave's second tile `region 1` -- which `t // tiles_per_region`
    # does -- gives it the wrong memory token.
    #
    # THIS BLOCK USED TO CITE A MEASUREMENT THAT DID NOT SAY WHAT IT CLAIMED.  Dropping `M_split`
    # from the read's broadcast set produced, on 2026-08-21:
    #     steady: operand A has 2 storage region(s) [0, 1] but its reads name [0]
    #     steady: operand B has 2 storage region(s) [0, 1] but its reads name [0]
    # and that was read as "one program reads one region while its copy writes two", hence as
    # needing ρ (#165) to say whose region it is.  Both steps were wrong.  Those lines came from
    # `verify_dataflow.check_region_coverage`, which asked a COORDINATE question about an AGENT
    # fact: wave `w` naming region `w` ALONE is the correct answer, and the union over agents does
    # cover both regions.  The check now exempts under-coverage on an agent-relative operand
    # (2026-08-22) and the same shape reports clean.  ρ was never the missing piece -- #276
    # refuted that premise outright, and ρ still has zero readers.
    #
    # WHAT ACTUALLY BLOCKS THE REPAIR is the COPY side, and it is #170.  Moving the region off the
    # coordinate is right for the read and wrong for the copy: the copy must still walk both
    # regions, and a Φ-fused group tiled on different region axes then has no region coordinate to
    # walk with (`gir_to_rocisa.py:573` refuses; 16 f8 kernels hit it).  `Hop.region_agent_relative`
    # is per-hop, `ord` is global, which is why no single edit here can express the asymmetry.
    # So the region stays on the read until #170 lands; the wrong token is the lesser, visible
    # defect.
    # EACH OPERAND'S OWN CUTS: its free axis, the SHARED K link, and its OWN K residual — never
    # the peer's.  The product of the listed extents is EXACTLY its region count by construction
    # (`gcd x own/gcd == own`), which is the invariant
    # `test_the_listed_region_axes_multiply_to_EXACTLY_the_operands_region_count` pins.  Listing an
    # axis an operand does not cut is what made a 2-region operand's reads name regions 0..3
    # (measured 2026-08-18 on `[aMT,bMT,aDU,bDU] = [1,1,2,4]`), and listing the shared axis for an
    # operand that does not cut it at all is what put a region displacement into a contiguous
    # image (the asymmetric-DU defect).
```


## `adapter/build.py`, line 276

```
    # THE READ HOP'S PAYLOAD IS PER OPERAND, and `lrvw` is only the DATA operands' answer.  A scale
    # tensor's shared→register instruction moves its own width (`blockWidth * BPR` bytes against a
    # scale `mxUnit`, which is what makes one `ds_load` cover several tiles), and that number is the
    # target's instruction selection, not a Solution key — so it arrives through
    # `bridge.kernel_to_params(..., target=)` as `ReadVectorElems`.  Falling back to `lrvw` for a
    # scale is what left the scale read's true instruction width unstated, which is the DOMAIN the
    # outside computation reads when it resolves the fold.  The fallback is the SAFE direction (a narrower quantum is always
    # admissible, §2.2) and because a hand-built param dict has no target to ask.
```


## `adapter/build.py`, line 293

```
    # A WAVE THAT DOES NOT COVER EVERY REGION MAKES THE READ'S REGION AGENT-RELATIVE (#245).
    # `_waveA`/`_waveB` count the regions ONE wave occupies; when that is fewer than the operand's
    # regions, two agents run the same read instruction against different regions, so the
    # coordinate names one of them and the access reaches all of them.
    #
    # HOW MANY WAVES the axis is spread over — `aMT` regions, `_waveA` of them per wave.  This is
    # ρ's `extent` for the wave-level resort (§2.8 move 2), and it SUPERSEDES the boolean
    # `_agent_rel = {A: _waveA < aMT, ...}` that used to live here: the boolean is exactly
    # `_wave_span > 1`, and an equality-only fact is what left ρ unable to state the partition in
    # the first place (#303 / RHO_DESIGN.md §1).  Keep the cardinality; derive the predicate.
```


## `adapter/build.py`, line 314

```
    # --- ρ: the AGENT ASSIGNMENT, built BEFORE the hops that derive from it (#303 R3) ----------
    #
    # §2.8 move 2 / §2.9: `ρ.resort` is a set of `(mode, agent-level, extent)`.  It is assembled
    # here, from the target facts above, and from here on it is the ONLY statement of which agents
    # traverse what — `Hop.rho_span` and `Hop.region_agent_relative` are read back OUT of it just
    # below, rather than being a second copy of the same facts.  R1/R2 proved the two agree on
    # 1728/1728 realizable configs before this switch-over.
    #
    # KEYED BY MODE, AND THAT IS THE MODEL, NOT AN OPTIMISATION.  A mode is resorted to ONE agent
    # level; two operands sharing a region axis (A and MXSA both carry `aRegions`) is ONE resort of
    # that axis, not two.  Emitting one entry per operand would put `M_split` in twice and make
    # `Rho.span_over` raise on its own duplicate.
    #
    # The ingredient table is (regions, free tile axis, wave span) per op-class; MXSA/MXSB inherit
    # their parent's, exactly as `region_modes`/`free_split` do at the Operand construction below.
```


## `adapter/build.py`, line 386

```
    # --- register width: DERIVED, not a parameter (#315).
    #
    # `W = L_pf`, the §2.6 PREFETCH-OVERLAP peak, for every operand.  This is the width at which
    # the pipeline's read-ahead is by construction covered: `L_pf = (dr_g + 1) · B`, the count of
    # generations simultaneously live.  Anything above it — up to the band ceiling `R` — buys only
    # extra hoisting freedom, and pays for it in occupancy one register at a time.  So `L_pf` is
    # the width to take unless something asks for more, and nothing does.
    #
    # WHAT THIS REPLACES.  `VgprPartitionA/B` + `VgprReuseStrategyA/B` were LoopModel-internal
    # knobs (never in `ValidParameters.py`, never set by a YAML) and at the shipping
    # `partition == 1` the strategy was IGNORED — the policy was hardcoded `unroll`, i.e. `W = R`,
    # the band CEILING.  Measured (#314): bf16 `MT256x256x256` PLR1 KMN has a band of
    # `L_war 64 / L_pf 128 / R 512` registers per operand, and we shipped the 512.  The knobs also
    # could not express the one shape that matters — ULM0's `HalfPLR` (`valuBlocksA = 1.5`) — and
    # their equal-partition rule (`n ∣ fan`) rejected every partition at the shipping mxf8
    # `MIWaveTile[8,7]` (fan 7).  A parameter that cannot reach the point it names is the #118
    # defect class, so it is gone rather than repaired.
    #
    # THE CLAMP IS WHAT MAKES `L_pf` SAFE, and it is not applied here: `theta.group_width` resolves
    # `'overlap'` to `max(L_pf, L_war)`, and `L_war` is the proved floor (Lemma 3b — below it two
    # simultaneously-live generations share one register, the unrescuable simultaneous-input race).
    # So a PINNED operand — one whose broadcast axis is OUTER to the reduction in `ord`, §2.6 term
    # (ii), which is `MNK`/`NMK` for both operands and one operand each under `MKN`/`NKM` — has
    # `L_war = extent(K)` and clamps straight back up to the ceiling by itself.  No special case
    # here, and no path around the clamp.
    #
    # The GROUP machinery stays (`Fragment.parts`, `group_policy`, `emit._read_groups`,
    # `_group_guard`): only the PARAMETERS are gone.  Per-group widths — a genuine mixed `d`, e.g.
    # HalfPLR's 1.5 blocks — are a separate change, because they need TILE-MAJOR register naming
    # (`macroAndSet` lays the `.set` table out buffer-major/fan-minor and `loopModelRegBuffers`
    # returns one `max W_g` per operand, so a mixed width costs the same as a uniform one at emit).
```


## `adapter/build.py`, line 439

```
    # S_shared PER OPERAND (§2.6: the depth is a map, not a scalar).  `NumLdsBlk` (the real
    # kernel-derived count, Solution.py — 2 = double-buffered, 1 = single in-place) is the PRESET;
    # `LDSBufferA`/`LDSBufferB` override it per operand so the two can be searched apart.  This is
    # a READ of a target fact, never an assumption: the depth is how many buffers the LDS
    # allocation actually reserves, and deriving it from `δ` instead is what broke PGR=1 (see
    # `theta._shared_depth`).  A DTV operand has NO shared placement at all (its trajectory is the
    # one-hop global->register of §5.3.1), so it gets 0 = "not applicable" — NOT 1.  Writing 1
    # would assert that it rotates a single LDS buffer, which is a different and false claim; the
    # field is simply meaningless without a shared hop, and `_shared_depth` refuses to answer for
    # such an operand rather than returning a number nobody should use.
```


## `adapter/build.py`, line 485

```
    # C — the OUTPUT/accumulator op-class (§2.1 line 80, PAPER-CONFIRMED Q18).  Present on the
    # FREE/output modes ONLY; the REDUCTION (K) modes are its broadcast axes (it holds one live
    # value per output tile, constant across the reduction).  NO input trajectory in the mainloop
    # (`hops=[]`) — the epilogue's `stage` splices its reverse register→(shared)→global path onto
    # this SAME op-class later (#116).  Its presence is where the reduction axes are DEFINED:
    # `reduction_names() = inner_modes \\ pres(C)`, general for batched/einsum.  The K modes are the
    # reduction axes of the canonical vocabulary; C is broadcast over whichever survive in ord
    # (extent>1).  frag_elems = the output tile a lane accumulates (M_inner·N_inner fragment) —
    # not load-bearing for presence, sized so footprint reporting includes the acc.
    #
    # EVERY K REGION MODE IS A REDUCTION AXIS.  Omitting one here does not make it "not a region
    # axis" — it makes it a FREE axis, because this set is the DEFINITION of the reduction modes.
    # Measured 2026-08-18: with the residual missing, an asymmetric DU split at gcd=1 (where the
    # only live K region mode IS a residual) turned the reduction into a free mode and every wmma
    # read a register no read had filled.
```


## `adapter/build.py`, line 506

```
    # MX scales follow their PARENT (§5.1): same grouping_mode + region_modes + floor policy so
    # their M/N fan is a DATA-tile (folded into |part|×region), not a whole-fan register rotation.
    #
    # THE SCALES KEEP `'unroll'` (W = R) WHILE A/B ARE DERIVED (#315) — and the reason is
    # UNVERIFIED, not proven.  Read this before changing it in either direction.
    #
    # WHAT IS NOT THE REASON.  On 2026-08-26 I moved the scales onto the derived width, mxf8
    # `DepthU 512` failed codegen with 24 OVERWRITE-BEFORE-USE reports, and I concluded the scale
    # ring must span every substep.  THAT CONCLUSION WAS WRONG: the reports were a CHECKER defect.
    # `verify_dataflow` modelled a wmma's inputs as the two matmul GRID axes, so scale registers
    # were written and never marked consumed — invisible at `W = R` (each substep owns a slot, so
    # a slot is only rewritten with the SAME source and the guard suppresses it) and a phantom
    # violation as soon as the ring narrowed.  The emitted order was provably fine throughout
    # (`read k -> wmma u=k -> read k+2`).  Fixed: `emit_plan._plan_mma` now carries its DERIVED
    # `(operand, group, slot, tile)` source list on the act and the checker reads it.  With that
    # fixed, mxf8 `DepthU 512` is CLEAN at the derived width.
    #
    # WHY THEY STILL KEEP `'unroll'`.  Only because the narrow scale ring has never RUN.  The one
    # hardware fact about scale widths is the 2026-08-18 measurement recorded above — 171/268 PLR0
    # cells failed numerically at `W = 1` — and the derived `'pipeline'` floors at 2, so it does
    # not obviously reproduce that.  But the checker has just been shown blind to exactly this
    # dataflow, so its clean verdict is weak evidence, and register rings are not position-checked
    # at all (#172).  `unroll` is the status quo the last green mxf8 run used.
    #
    # TO CHANGE IT: set `'pipeline'` here and do an mxf8 hardware run across DepthU 128/256/512.
    # It is a real register saving (MXSA 4 -> 2 at DU512) and the model says it is safe; it just
    # has not been measured, and this is not a thing to decide from a passing checker.
    # AND THEIR OWN `lds_buffers` / `off`, PRESET TO THE PARENT'S — the part that was missing.
    # A scale gets the same per-operand knobs a data operand has (`LDSBufferMXSA`,
    # `PrefetchGlobalReadMXSA`, `PrefetchLocalReadMXSA`), each defaulting to its parent's resolved
    # value, so the depths CAN be searched apart while the shipped kernel is described correctly.
    #
    # Leaving `lds_buffers` unstated is not neutral: `theta._shared_depth` then falls back to
    # `S = δ`, which its own docstring calls the trap "for anything that knows the real number",
    # and here it is.  At PGR=2 the fallback COINCIDES with the double-buffered count and
    # everything agrees; at PGR=1 the scales get S=1 against A/B's S=2.  S=1 is the in-place
    # refill, so the scale copies are classified as `S=δ` WAR refills (emitted LAST) in the same
    # steady body where A/B's chunk-crossing copies must be emitted FIRST — one body, two
    # incompatible σ_c orders, which surfaces as the P6/P7 structural gate rather than as a wrong
    # number.  MEASURED 2026-08-18: MX + PGR1 + PLR1 was the only failing cell of the four
    # PGR x PLR, and PGR2 passed for exactly that coincidence.
    # THE SCALES KEEP THE PARENT'S `region_modes` EVEN THOUGH `split_of` GIVES THEM ONE REGION.
    # That looks like over-listing against the invariant above, and it is not safe to "correct".
    # Both ways of correcting it were tried on hardware, 2026-08-19, and both LOST kernels:
    #
    #   region_modes -> ()          MXSA's rate went R=2 -> R=4, because `M_split` then entered the
    #                               rate instead of being broadcast over it (§2.6: the split
    #                               multiplies the NUMBER of groups, never a group's R).  That
    #                               re-widths the rotation and moves every scale register's slot.
    #   gate LDS storage on the     MXSA/MXSB dropped from 4 LDS buffers to 2 (the region-1 pair
    #   region COUNT instead        stopped being named at all).  The extra tokens are NOT dead
    #                               weight: removing them lost ordering the kernel relies on.
    #
    # Both took `mxf8` from 6 failures to 32, including `LOKMN` + `TDMSplitA/B=1`, which was green.
    # So the tuple is load-bearing in BOTH maps here — the rate needs the axis broadcast, and LDS
    # storage needs the buffers it implies — and the invariant's "product == own region count"
    # simply does not extend to an operand that is indexed on an axis it does not itself partition.
    # Leave it alone until that is modelled properly rather than trimmed at one consumer.
```


## `adapter/build.py`, line 566

```
        # THE SCALE RING USES ITS PARENT'S FRAGMENT BUILDER, not a second inline `Fragment`.
        # `reg_fragment`'s `gm` rule is character-for-character what the local `gmA` was, so the
        # only thing the duplicate construction contributed was a DIFFERENT WIDTH POLICY: it set no
        # `group_policy`, and `Fragment.policy_of` falls back to `'unroll'` (W = R) while #315 moved
        # A and B to the derived `'pipeline'`.  That went unnoticed while `R` was ord-invariant at 2
        # — `unroll` and `pipeline` agree there — and became visible when `R` became ord-variant
        # (#319/#321): at `MKN` with `PrefetchLocalRead=0`, `R = 4`, so the scale ring came out
        # W = 4 against its parent's 2, and the short `T < M` arm had slots 2 and 3 to prime.
        # MEASURED: aborted the mxf8 build on MKN/MNK/NKM/NMK x PLR0 with `the short arm cannot
        # stand alone: 4 consumer(s) ... read a location the arm never writes`.
```


## `adapter/build.py`, line 636

```
    # --- ρ: NOTHING POPULATES IT ---------------------------------------------------------
    # ρ (§2.8 moves 2+8, `resort`/`specialize`) says WHICH AGENT CARRIES WHICH op-class.  No
    # TensileLite parameter expresses that today, so it stays empty and no consumer reads it.
    #
    # `WaveSeparateGlobalRead` is NOT it, and was mapped here twice by mistake — the name invites
    # it.  All it does is rewrite `LSC`/`LSP` (Solution.setGlobalLoadTileDimClassic): the load-tile
    # DECOMPOSITION, i.e. which slice of the macro-tile each thread reads.  Every wave still loads
    # part of A and part of B either way, so the CARRIER SET is unchanged — that is `tile`/hop
    # geometry, not an agent assignment.  A faithful mapping for it would target the copy hop's
    # decomposition, which the decoder does not model.
    #
    # ρ is built ABOVE, before `_read_hop`, because the hops now DERIVE from it (#303 R3).
    # It used to be assembled here, after the operands, which was only possible while
    # nothing read it (R1).  See `_rho_ingredients`.
```


## `adapter/build.py`, line 651

```
    # --- off: the first-class retime matrix {(op_name, role, level_name): δ} (§2.4/§2.9) ------
    # `off` is per (OP-CLASS × LEVEL), and an op-class is (operand, hop) — so the key carries the
    # hop's ROLE (`theta.COPY`/`READ`/`STORE`, from `Hop.role`) as well as the operand.  Translate
    # names the levels it built:
    #   off(copy, <outer level>)   = PGR (dg) — the global→shared copy prefetches whole DepthU chunks.
    #   off(read, <substep level>) = PLR (dl) — the shared→register read prefetches K substeps.
    # The outer level is ord_[0] (the DepthU reduction `iter`); the substep is the INNERMOST
    # reduction inner mode present in ord (the K axis the read-ahead shifts along).  Both names are
    # DERIVED from the modes just built — the core reads them back via off_at(op, role, level),
    # never by literal.  δ=0 entries are omitted (absent ⇒ 0).
    #
    # The ROLE is not decoration: §5.3.1's cross-level boundary term needs `off(read, <outer>)` —
    # a read carrying an OUTER-level offset — which under the old (operand, level) key was the SAME
    # key as `off(copy, <outer>)`.  The two are different op-classes and must stay distinguishable.
```


## `adapter/build.py`, line 666

```
    # THE READ-AHEAD LEVEL, one derivation shared with the core (#321).  `PrefetchLocalRead` is an
    # OUTER-language parameter — the minimum wmma-REGION read-ahead — and this is where it becomes
    # θ's `off`: the offset sits on the axis one region step advances (the outermost mode of the
    # region's inner nest), not on the innermost reduction axis.  The old expression matched a
    # hardcoded K-name list, so it also could not see a reduction axis under any other spelling.
    # PER OPERAND (§2.4: `off` is a matrix over op-class x level — per-op-class BY DEFINITION, and
    # §5.3.1 pt1-2's staggered peel exists exactly to carry a heterogeneous `off`, worked there
    # with off_A=1, off_B=2).  `PrefetchGlobalRead` / `PrefetchLocalRead` are the PRESETS;
    # `...A`/`...B` override per operand.  An operand not named in the map has δ=0 (absent ⇒ 0).
```


## `adapter/build.py`, line 694

```
    # PER-MEMBER AGENT SHARES (Φ) -> ρ.  `agents` below is ONE scalar and so asserts that every
    # member of a group takes an equal share; that is true for the shipping groupings and FALSE for
    # `A_MX`/`B_MX`, whose `FUSE_WAVE_SHARES` are [2,1,1].
    #
    # THESE GO IN ρ, NOT IN A SECOND PER-OPERAND FIELD.  ρ is the agent assignment; the copy share
    # is an agent fact about an op-class, so it is a `Resort` with `role="copy"` keyed by op-class
    # in `origin`.  The read entries above keep their mode-uniqueness invariant untouched (`_resort`
    # still raises on a second answer for one mode) because the two roles never share a key — which
    # is the whole reason a separate `Operand.copy_agents` looked necessary and was not.
    #
    # The mode is the op-class's own free axis, the same ingredient the read entries use, so a dump
    # can line the two up; what distinguishes the members of a group is the EXTENT, which is
    # exactly the asymmetry a scalar cannot hold.
```


## `adapter/loop_order.py`, line 16

```
# Each of K/M/N contributes a (split-region, inner-remainder) pair.  The FIXED canonical
# index order (0..5) is [K_split, K_inner, M_split, M_inner, N_split, N_inner]:
#   0 K_split = K_Split                (K/DU region)
#   1 K_inner = substep / K_Split      (inner K-reduction remainder — reorderable)
#   2 M_split = M_Split                (M region)
#   3 M_inner = WT0 / M_Split          (in-region M fan — A's grouping mode)
#   4 N_split = N_Split                (N region)
#   5 N_inner = WT1 / N_Split          (in-region N fan — B's grouping mode)
# EVERY axis is (split OUTER, inner INNER): a TDM region's whole fan completes before the
# region index advances, so no two regions' TDM buffers overlap in the default order — this
# holds for the K/DU region too (a DU-split kernel finishes K-region 0 before K-region 1).
# A split of 1 makes its region mode extent-1 → it drops out of ord; all-1 degenerates to
# the 3-loop KMN [K_inner, M_inner, N_inner] = the old [substep, mtile, ntile].
```


## `adapter/loop_order.py`, line 38

```
# LoopOrder is a WORD over the axis letters {K, M, N} — memorable, no lookup table.
#
#   * 3-letter shortcut ("KMN", "KNM", ...): each axis's (split, inner) pair kept
#     CONTIGUOUS in that order.  "KMN" = [K_split,K_inner, M_split,M_inner, N_split,N_inner]
#     (the canonical default).  The 6 shortcuts are the 6 whole-axis orders.
#   * 6-letter word ("KMKNMN", ...): each letter appears TWICE; FIRST occurrence = that
#     axis's _split, SECOND = its _inner.  Lets regions interleave (a split's fan does NOT
#     complete before another axis advances) — e.g. "KMKNMN" =
#     [K_split, M_split, K_inner, N_split, M_inner, N_inner].
#
# Because first-occ is ALWAYS the split, this encoding structurally EXCLUDES inner->split
# orders (a region indexed inside its own fan) — exactly the rare/unwanted tail.  A 3-letter
# word expands to the contiguous 6-letter form (doubling each letter): "KMN" -> "KKMMNN".
```


## `adapter/solution.py`, line 84

```
        # Φ (§2.8 move 9) — the FUSE, derived, never requested.  Multi-wave TDM aliases B's
        # descriptor onto A's SGPRs (`tdmAGroup0`) and issues ONE cooperative `tensor_load_to_lds`
        # that serves both operands, wave parity selecting which.  That is not a knob on the
        # scaffold side: `KernelWriterAssembly.isTdmWaveSeparated` is exactly the condition below,
        # and when it holds the fusion HAPPENS.  So θ must be told, or it would model two
        # independent movements against one aliased descriptor.
        #
        # `TDMFuse` IS a Solution parameter now (2026-08-18), resolved to a plain 0/1 in
        # `Solution.py` where `enableTDM{A,B}` and `NumWaves` are live; `-1` reproduces the old
        # condition exactly.  The CONDITION is read, never re-derived — that is how θ and the
        # scaffold drifted before.
        #
        # THE VALUE IS PASSED VERBATIM, and the MX pairing lives in `translate.FUSE_GROUPS[1]`
        # rather than in a mapping here.  A wave-separated MX kernel aliases TWO descriptors — A/B
        # and MXSA/MXSB — so θ's group `1` names both pairs, and the operand filter drops the MX
        # group on a kernel that has no scales.  Putting it there rather than here keeps ONE
        # definition of what the fuse means, reachable identically from a real Solution and from a
        # hand-built param dict (a unit test setting `TDMFuse: 1` gets the shipping pairing, not a
        # bridge-only variant of it).
```


## `adapter/solution.py`, line 109

```
        # REGION SPLIT (§2.8 move 1) — passed through, NOT defaulted off.  `TDMSplit` cuts each
        # operand's tile into storage-disjoint regions, and the scaffold acts on it unconditionally:
        # `globalReadDo` issues one `tensor_load` per region whether or not θ knows.  So dropping
        # the key here does not model an unsplit kernel — it models a DIFFERENT kernel than the one
        # being generated, and the two layers then disagree about how many loads exist and how far
        # the descriptor moved.  That is exactly what happened: the key was absent, θ derived one
        # region, `emitCopyTile` took its unsplit branch, and the region walk GIR was supposed to
        # own stayed inside `globalReadDo` with nothing left to cancel it (#236).
        #
        # THE SOLUTION KNOB IS NARROWER THAN theta's.  `TDMSplitA`/`TDMSplitB` name ONE axis per
        # operand at factor 2; θ's `[A_MT, B_MT, A_DU, B_DU]` takes any factor with both axes live
        # at once.  That gap is deliberate — the model is exercised ahead of L3, which does not yet
        # walk a 2-D region grid — so the mapping happens HERE and `translate.py` keeps the general
        # form.  `split_factors` is the same reader `Solution.py`'s gates use, so a kernel that was
        # accepted cannot be modelled as a different one.
```


## `adapter/solution.py`, line 134

```
        # MICROSCALING (§5.1) — passed through for the SAME reason `TDMSplit` is.  A non-zero
        # `MXBlock{A,B}` means the kernel has a scale TENSOR: `translate.py` gives it its own
        # operand (`MXSA`/`MXSB`) with its own trajectory, register ring, copies and fences,
        # following its parent's grouping mode and region modes.  Drop the key and θ models a
        # kernel with two operands where the emitted one has four — the scale loads exist either
        # way, so what is lost is not "MX support" but the AGREEMENT between the model and the
        # instruction stream about how many movements are in flight.
        #
        # It lives under ProblemType, not at kernel top level, which is why it needs naming here
        # at all — `kernel.get("MXBlockA")` answers None on a real Solution and reads as "no MX".
```


## `emit.py`, line 66

```
    # A READ `off` AT AN OUTER LEVEL IS ACCEPTED, DEEPENS THE PEEL, AND IS APPLIED TO NOTHING.
    # §2.4 (L106/L118) makes `off` a matrix over (op-class x LEVEL) for EVERY op-class, and §5.3.1
    # (L510) says the staggered peel's input is "copies AND a slab-level read offset (e.g.
    # `off(read,iter)`), not copies only".  We honour neither: `peel_depths` folds the entry into
    # `M_l` (so the guard, the prologue depth, the steady bound and the drain step count all move)
    # while the only thing `build_ir` staggers over is `PeelDepths.copy_off()`, which filters
    # `role == COPY and lvl == chunk` (placement.py:233-234) — so every read `Inst` keeps its
    # UNSHIFTED slot and `src_slot`.  Half-applied: neither the requested θ nor a refusal, and the
    # ledger certifies EMPTY.  That is verbatim the silent drop Lemma 1 names ("a decoder that keys
    # peel depth only on the innermost outer level silently drops every coarser-level `off`") and
    # the one failure the empty-ledger certificate exists to rule out.  Found by the 2026-08-21
    # full-paper audit (D1).
    #
    # WHY THE CHECK IS HERE AND NOT IN `boundary_hoists`.  This IS a cross-level boundary term by
    # the paper's definition (an `off` at a level outer to the op-class's home), and
    # `boundary_hoists` drops it on its third condition `if home not in outer` — but that condition
    # is load-bearing for the REGION-SPLIT COPY, whose home is a region mode (an inner mode) while
    # its `off` sits at the chunk: that pairing is the ordinary copy prefetch and is on the shipping
    # TDMSplit path.  Loosening the shared predicate would refuse those live kernels.  The
    # distinction is per-role — a copy's ordinary prefetch level IS the chunk, a read's is the
    # substep — so the narrow refusal belongs here, where it can name the role.
    #
    # Unreachable from the parameter path: `translate` sets only `off(copy, chunk)` and
    # `off(read, substep)` (translate.py:839-843).  Latent, not shipping — refused so it cannot
    # become shipping silently.
```


## `emit.py`, line 110

```
        # A NEGATIVE `off` is ILL-TYPED.  §2.4 (2026-08-15) types the retime matrix
        # `off : (op-class x level) -> N` — "a non-negative depth throughout" — and `retime` now
        # SETS `off(p,l) := d` rather than subtracting, so no move reaches a negative entry either.
        #
        # DEFERRAL — the thing a store wants, issuing |d| iterations LATER — is NOT a negative
        # offset.  It is a POSITIVE `off` on a REVERSE-TRAJECTORY op-class: Lemma 1's reflected
        # peel, "steady [0, T-d) with the trailing d instances hoisted forward into the successor
        # iteration, prologue and drain swapped".  The direction is read off the op-class's
        # trajectory sense (forward copy -> hoist backward = prefetch; reverse store -> hoist
        # forward = defer), never off the sign of d.  So `off(C, STORE, gtile) = 1` is the deferred
        # store, and it is an ordinary positive entry (QUESTIONS_FOR_AUTHOR Q29, ANSWERED).
```


## `emit.py`, line 414

```
    # THIS USED TO RE-DERIVE THE RULE AND GET THE EMPTY CASE WRONG.  It read
    #     `_presence(theta, op)[-1] if _presence(theta, op) else None`
    # while `op_class_level` — the canonical helper, whose docstring states the rule — says "read:
    # the innermost mode of its presence set; THE REDUCTION CHUNK when presence is empty" (§5.3.1
    # line 523, the same rule that puts a coarse copy at the chunk level).  The two agreed on every
    # non-empty presence and diverged on exactly the empty one.
    #
    # WHY EMPTY PRESENCE IS REACHABLE AT ALL, and why `None` is fatal.  §2.2's movement quantum is
    # SUBTRACTED from the hop's presence (Precondition Q, `placement._presence` via
    # `geometry.hop_broadcast`): an axis one instruction spans is absent, because the coordinates
    # inside one quantum are one presence point.  When the quantum spans every axis the read still
    # varies over, presence is legitimately EMPTY — the operand is read exactly once per enclosing
    # chunk.  `None` then matches no mode at the emit site below (`read_level[op.name] == mode`),
    # so the read was emitted ZERO times, in every block.
    #
    # MEASURED 2026-08-21, mxf8 block 5.  At `DepthU 128` (n_s == 1, so `K_inner` is gone) with
    # `VectorWidthA/B = 1` (the only case for which the bridge supplies a `ReadQuantum` for the MX
    # scales), `pres(MXSA) == []` and NO `ds_read MXSA` appears in prologue, steady, drain0 or the
    # short arm.  The operand is copied to LDS and never read into registers.  It surfaced as two
    # unrelated-looking failures with one cause:
    #   * `P7: obligation ledger not empty -- inplace-WAR MXSA@shared:lastread -> MXSA@shared:refill`
    #     (PGR2): the WAR has no read to discharge it, because the read does not exist;
    #   * `the T < M path cannot be emitted in EITHER shape ... 4 consumer(s) ... unserved`
    #     (PGR1): ShortPathFold blames the arm, but the hole is global.
    # `DepthU 256/512` pass because `K_inner` survives the subtraction, leaving `pres == [K_inner]`
    # and a read whose `covers` spans the quantum's tiles.
```


## `emit.py`, line 441

```
    # THE WMMA IS PLACED BY ITS OWN PRESENCE, not by the nest's depth.  It was `inner[-1]` -- the
    # innermost ord level, unconditionally -- making it the ONE op-class not placed the way every
    # other is (`read_level` above is the same `op_class_level` call).  The two answers have always
    # agreed, because every inner mode is either in some output's presence or is a reduction mode;
    # they come apart when an axis is traversed by an AGENT rather than by the coordinate (#245).
    #
    # THE SET IS OUTPUT PRESENCE **PLUS THE REDUCTION MODES**.  The accumulator does not VARY over a
    # reduction mode -- that is what makes it a reduction -- but the wmma INSTANCE occurs once per
    # reduction coordinate, accumulating into the same registers.  Using the output presence alone
    # hoists the wmma above the K loop: 130 tests failed on exactly that.
```


## `emit.py`, line 457

```
    # COARSE copy has empty inner presence → the reduction-chunk level; a REGION-SPLIT copy
    # has the region mode in its presence → one level deeper, rolled under Loop(region). ----
    # off is PER-OP-CLASS: copy for operand p targets K-tile iter+off_p (steady), or a concrete
    # K-tile in a peel step.  Each operand's K-tile is iter+off_p, so its buffer slot is that mod d.
    #   phase="steady"   → symbolic slot Expr(iter+off_p) mod d.
    #   phase="prologue" → ramp step t (t∈[0,M)): K-tile = t+off_p-M (a small concrete int ≥0),
    #                      slot = K-tile mod d.  Present iff off_p ≥ M-t.
    #   phase="drain"    → tail step t (t∈[0,M)): K-tile = T-M+t+off_p (symbolic, ≤ T-1),
    #                      slot shown symbolically as (T-Δ)%d.  Present iff off_p ≤ M-1-t.
    # Φ (fusion) over MOVEMENT INSTANCES (§2.8 move 9, paper update): `fuse` partitions the
    # (op-class, region) instances on the global→shared edge, NOT whole op-classes.  A copy op
    # with `split>1` is TDM-region-split: its region mode is IN the copy's presence set, so it
    # emits ONE movement per region value (A0, A1) at that presence level — "one level deeper,
    # per-region" (§5.3.1 line 408), not the bare iteration (iter) level.  A fused group of split members
    # pairs by REGION INDEX: {A,B} split-2 → AB0={A0,B0}, AB1={A1,B1} (the operands may be tiled
    # on DIFFERENT region axes — the pairing is by index, not a shared mode).  An unsplit group
    # (MX={MXSA,MXSB}, split=1) is one whole-operand movement.  Heterogeneous split counts coexist
    # in one Φ; each fused group is one cooperative instruction / one atomic completion (§4.3).
    # the movements this θ emits + each one's region count — ONE authority, shared with the
    # lowering (which needs a movement's completion granularity to key its token).
```


## `emit.py`, line 509

```
        # σ_c FIRST, NESTING SECOND, and the order is not stylistic.  `sigma_c._movement` unwraps a
        # node to its ONE movement to classify it; a nest holding A's and B's copies together has
        # no single movement, so `is_war_copy` returns False for it and the whole block silently
        # stays where the skeleton put it — copies FIRST, inverting the §2.4 refill-after-read
        # freeze.  (Measured: doing it the other way flipped copies-last to copies-first and the
        # ledger's order arm reported the two shared WARs undischarged.)  So σ_c orders the copies
        # as separate Insts, exactly as before, and only then are contiguous runs folded into the
        # ord nest — which reorders WITHIN a run and never across the read/wmma boundary σ_c set.
```


## `emit.py`, line 555

```
    # ── OBLIGATION LEDGER = SINGLE SOURCE OF TRUTH for every wait (paper decode step 9b, §5.4):
    # "for each ledger obligation, attach an Await(dependency, class, scope) at the CONSUME
    # instance."  We build the ledger once and index obligations by their consume site, so EVERY
    # Await is DERIVED from an obligation — never hand-authored at the emit site.  The consumer of a
    # RAW-residency is whoever READS the produced value (not who writes it); of a *-WAR it is the
    # REFILLER.  Consume-site key = (operand, role):
    #     RAW  A@shared   (produced by copy) → consumed by the READ   → ("A","read"),  tensorcnt
    #     RAW  A@register (produced by read) → consumed by the WMMA   → ("A","wmma"),  dscnt
    #     WAR  A:shared:refill               → the COPY refiller      → ("A","copy"),  dscnt
    #     WAR  A:g0:refill                   → the READ in-place refill→ ("A","read"),  dscnt
```


## `emit.py`, line 568

```
        # A READ-AHEAD RESIDENCY (Lemma 3d) gets NO Await of its own.  Its edge is
        # `complete(prologue read at coord) ⤳ issue(wmma at coord)` — the SAME edge the wmma's
        # ordinary `X@register` RAW residency already discharges — and the obligation's real
        # content is a STRUCTURAL requirement on the prologue (it must pre-issue that read), which
        # `check_ledger_discharged` gate (c) verifies directly against the prologue body.
        #
        # It used to be bucketed like a WAR, which put it on the READ — the obligation's PRODUCER,
        # not its consumer — and, because the bucket key is only (op, role), fanned EVERY coord's
        # obligation onto EVERY read of that operand.  A prologue read then awaited its own
        # residency, and each steady read carried one spurious await per read-ahead coord.
```


## `emit.py`, line 639

```
    # #213 — A REJECT HERE WAS ATTEMPTED AND WITHDRAWN (2026-08-25).  Read this before trying again.
    #
    # The proposal: mirror the `off` guard above for §5.3.1 pt5's copy-first rule, since a Φ group
    # is ONE instruction and `copy_must_be_first` is per-OPERAND, so members that disagree look
    # ill-posed — placing the group first satisfies one member's crossing RAW and violates the
    # other's refill-after-read WAR.
    #
    # WHY IT IS WRONG.  The mix is not the rare corner #213 recorded ("needs a per-operand
    # LDSBuffer override AND a non-KMN order AND a fuse, so no shipping point mixes").  MEASURED:
    # it needs only PGR1 + a non-KMN order + the fuse, and the disagreeing side FLIPS with the
    # order — M-outermost gives copy-first {A}, N-outermost gives copy-first {B}.  A guard here
    # refused 36 of 312 configs in the standing parity sweep, and those cells are in the SHIPPING
    # bf16 matrix (blocks 6a/6b fork MKN/NKM at PGR1, 4 waves, TDMInst 3 -> fused) where they run
    # 9616/9616 green on hardware.
    #
    # So `sigma_c.is_war_copy`'s `any(...)` is not silently picking a losing side at these points —
    # whatever it picks is evidently sound here, and rejecting would NARROW proven coverage.  The
    # real question #213 should have asked is narrower: under what condition is the `any` actually
    # unsound?  The one recorded ledger complaint came with `LDSBufferB=1`, which changes S_shared
    # and is NOT what makes the copy-first sets differ.  Answer that first; do not gate.
```


## `emit.py`, line 672

```
        # a token is the θ OP-CLASS NAME, always — never a synthesized string.  Which storage
        # region this movement carries is a COORDINATE (`coord`, below: (region_mode, region)),
        # exactly like every other position in the model, plus the structural `part` field.  It is
        # deliberately NOT baked into the name: encoding the pair (op-class, region) as "A0" made
        # every consumer re-derive the op-class by PARSING it back, and three call sites grew three
        # different string-strippers that disagreed at ≥3 regions.  A name identifies an op-class;
        # a position is a coordinate.  (§2.8 `tile`: a region split makes sibling movements of the
        # SAME op-class, so they must keep sharing its name — the residency is keyed on it.)
```


## `emit.py`, line 685

```
        # the region is a COORDINATE, left SYMBOLIC because the enclosing `Loop(region_mode)`
        # supplies its value — the rolled form (§5.2).  The renderer/unroller resolves it.
        #
        # NAME **EVERY** AXIS THE MOVEMENT IS ENUMERATED BY, not just the representative (#251).
        # `_region_mode_of` answers "which level does this copy sit at", which is one mode by
        # construction; the COORD is a different question — it must name each axis whose enclosing
        # `Loop` supplies one of its digits, or the value that loop supplies is never recorded.
        #
        # MEASURED 2026-08-18 on `[A_MT,B_MT,A_DU,B_DU] = [2,2,2,2]`: A is enumerated by
        # `M_split x K_split` and the nest correctly wrapped it in both loops, so the right NUMBER
        # of copies was emitted (4 for 4 regions) — but every one carried `{'K_split': v}` alone.
        # `region_of` reads the coord, so `_flat` was undefined, `_split_copies` dropped the
        # movement, and NO region increment was emitted: the descriptor never walked A's regions.
        # `walk_violations` skipped it on the same `flat is None` rule and called it clean, which
        # is how the 90-order `_SIX_AXIS` sweep passed for weeks on a walk that did not exist.
```


## `emit.py`, line 702

```
        # AWAITS from the ledger (step 9b): the copy's consume site is (op, "copy") — its RAW
        # (global→shared residency) AND its S=δ refill WAR (wait the vacating shared read).  Only the
        # WAR gates the STEADY/DRAIN refill; the RAW (waiting the global load) is the copy's own
        # producer edge.  PROLOGUE fill (war=False) targets empty buffers → drop the WAR obligation,
        # keep only the RAW.
        #
        # Fused group: a real UNION of the members' obligations, not a concatenation.  Since the
        # copy-hop producer is now the MOVEMENT (`ledger.movement_name`), every member's RAW on the
        # cooperative copy resolves to the SAME dep, so concatenating emitted the identical Await
        # once per member.  The members' WARs stay distinct — each names the op-class whose reads
        # vacate that operand's buffer — so this dedups the RAW and keeps both WARs.
```


## `emit.py`, line 750

```
                # The fully-peeled (short-loop) arm.  Lemma 1: *a fully-peeled level resets to 0
                # every `off` AT THAT LEVEL and at levels inner to it* — so this arm has no ramp
                # and no stagger.  Every op-class copies at every step, into its OWN chunk's
                # buffer; the tree is `T` unpipelined iterations, each step self-contained.
                #
                # Not a special case bolted on: for a UNIFORM `off` it is bit-identical to the
                # "prologue" branch above (`o = M` makes `o < M - t` false at every t, and
                # `(t + M - M) % d == t % d`), which is exactly why the shipping configs are
                # untouched.  It differs only where `off` is non-uniform, and there the ramp is
                # WRONG: a shallow op-class (`off(A)=1 < off(B)=2=M`) is skipped at step 0 by
                # `o < M - t` and copied at step 1 — landing AFTER the step that reads it, with
                # no steady body to absorb the stagger.
```


## `emit.py`, line 784

```
            # A REGION-SPLIT copy is ONE op-class at its REGION LEVEL (§5.3.1 line 523: "the
            # region mode is in the copy's presence set, so level(copy) is that region mode and
            # the copy sits one level deeper, PER-REGION").  The rolled tree carries it as a
            # single Inst under a real `Loop(region_mode)` — §5.2 "each op-class appears ONCE in
            # the tree, at its presence level ... node count independent of the trip count".
            # FILE THE MOVEMENT AT ITS INNERMOST SPANNED AXIS, so every axis it is split on
            # ENCLOSES it.  A movement spanning two axes (MT+DU: `M_split` x `K_split`) filed at
            # the outer one is enumerated by that axis alone and emits `ext` copies instead of the
            # product; filed at the inner one, both loops wrap it and the strides on each do the
            # rest.  `_region_mode_of` returns the representative (the first presence mode), which
            # is the right NAME but not necessarily the right LEVEL.
```


## `emit.py`, line 832

```
                # NO `n == ext[m]` CHECK.  `n` is the movement's region count across ALL its
                # axes, so comparing it to ONE axis's extent is only right for a single-axis
                # split — it rejected the legitimate MT+DU shape (4 regions over 2x2).  What must
                # hold is that the count is attributable per axis, and that is exactly what
                # `_axis_stride` verifies for each enclosing level below.
                # A Φ-FUSED group spanning SEVERAL region axes is filed at its representative
                # level and enumerated there; §2.8 move 9 pairs members BY INDEX, so one trip of
                # that level moves region i of every member.  It needs no refusal — what it needs
                # is the guard below to know the group spans those other axes, which is exactly
                # what `spans` carries.  (The naming gap, that the fused instance's coord names
                # only one member's axis, is #170 and is orthogonal to the order.)
```


## `emit.py`, line 844

```
                # PRESENCE ON AN ENCLOSING AXIS IS A STRIDE, and first-touch is its degenerate
                # case.  This group moves `own` regions along axis `g` out of `ext[g]` values, so
                # it issues every `ext[g] // own` values: `Cond(g % stride == 0)`.
                #   own == ext[g]  -> stride 1 -> no guard      (present at every value)
                #   own == 1       -> stride ext[g] -> `g == 0` (the first-touch guard: over a
                #                     coord ranging [0, ext), `v % ext == 0` IS `v == 0`)
                #   1 < own < ext  -> a real stride              (the coarser operand of an
                #                     asymmetric DU split: A cut 4 ways, B every 2nd value)
                # Writing the two ends as separate concepts is what made the middle look like it
                # needed a mode of its own; it does not — it needs the same rule at stride 2.
                #
                # Invariance is presence-based (`spans`), NOT `g != r`: under a Φ fuse one movement
                # carries several operands and is present on each of their region axes, so a
                # name-inequality test would guard a cooperative load down to one axis value.
```


## `emit.py`, line 908

```
        # PLR read-ahead: the read at flat presence position P fetches P+shift (§2.1
        # step = ⟨reduction-index, ord-strides⟩ − off), re-decomposed per presence mode so the
        # advance CARRIES out of the reduction axis into the free modes — the next TILE's substep,
        # which is what §5.3.1 line 494 requires ("leaf 0's read-ahead targets (m0,n1), not
        # (m0,n0,k1)").  Bumping the reduction coord alone wrapped inside the tile, re-reading a
        # value the prologue had already delivered (the double-issue the paper says cannot happen)
        # and leaving the next tile's leading substep to arrive one wmma late.  The same `shift`
        # drives the register slot and the shared src_slot in `_read_placement`, so the three
        # cannot drift.  Non-presence (broadcast) axes carry no coordinate at all.
```


## `emit.py`, line 1033

```
        # AN ANCHORED REFILL IS EMITTED AFTER THIS LEVEL'S CONSUMERS (#332).  The assembly below is
        # reads -> wmma -> inner nest, so a read is normally issued BEFORE the wmma that consumes
        # it — correct for an ordinary read, wrong for a refill whose anchor exists precisely
        # because its name's last consumer is later than its own position.
        #
        # `Inst.anchor` picks WHICH PASS of an invariant mode the read fires in (it is the
        # first-touch guard's constant); it cannot say WHERE INSIDE that pass.  σ_c cannot supply
        # that either: it reorders the ROLLED body, where one wmma `Inst` stands for the whole leaf
        # nest, so "after the wmma" there is vacuous for a read whose level sits INSIDE that nest.
        # MEASURED, `MNK` + `TDMSplitA=1, TDMSplitB=2`: A is invariant over `N_inner`, anchored at
        # `N_inner == 1` — the right pass — and still emitted as
        # `wmma idx1=0 u=1 | read A buf=1 | wmma idx1=1 u=1`, i.e. between two consumers of one
        # register.  Appending it after the wmma and the inner nest is the position its anchor
        # already chose.
```


## `emit.py`, line 1059

```
                # A read-ahead read is out of bounds on the LAST peel step exactly when its
                # SHIFTED FLAT POSITION leaves the current reduction chunk: `P + shift >= n_pres`,
                # over this operand's presence traversal.  Same fact, same numbers, as the advance
                # itself (`_readahead_shift`) — the suppression is the advance's own bound, not a
                # separate rule.
                #
                # This used to guard on the reduction coord alone (`K_inner < n_s - dr`), which was
                # the bound for the OLD advance that only bumped that coord.  Once the advance
                # became the flat ord position the two stopped agreeing: for a K-innermost order
                # the crossing happens at `P >= 2` — the whole second free-mode tile — while the
                # coord guard suppressed on `K_inner` instead, dropping in-bounds reads and KEEPING
                # the ones that run off the end.  In the drain that is a `ds_read` of an LDS buffer
                # no copy ever filled.
                # `_pos_terms`, not an inline rebuild: `_st` is in PRESENCE POINTS and `_np` is a
                # point count, but the loop variable is a TILE index, so a §2.2-folded axis would
                # inflate `P` by `q` and suppress the wrong reads (#312).  This is the same flat
                # position `_shifted_coord`, `_rate_slot`, `_shifted_rate_slot` and `src_slot`
                # build — five sites, one builder.
```


## `emit.py`, line 1113

```
        # A READ WHOSE PRESENCE THE QUANTUM EMPTIED SITS AT THE CHUNK LEVEL, and `build_level` has
        # no site for it: that walk visits INNER modes only (`mode, ext = inner[li]`), so an
        # op-class whose `op_class_level` is the reduction chunk matches no `mode` and would be
        # emitted ZERO times.  This is the read-side twin of the coarse copy, and the paper gives
        # them one rule: §5.3.1 line 523 places each op-class "by ITS OWN presence set", and
        # `op_class_level` spells the degenerate case out — "the innermost mode of its presence
        # set; THE REDUCTION CHUNK when presence is empty".  `copy_insts` already emits a coarse
        # copy there; this is the same emission for a read.
        #
        # WHEN IT FIRES: §2.2's movement quantum is subtracted from the hop's presence
        # (Precondition Q), so when one instruction spans every axis the read still varies over,
        # presence is legitimately empty — the operand is read exactly once per enclosing chunk,
        # which is what this emits.  MEASURED 2026-08-21 on mxf8 block 5 at `DepthU 128`
        # (n_s == 1, no `K_inner`) with `VectorWidthA/B = 1` (the only case for which the bridge
        # supplies a `ReadQuantum` for the MX scales): `pres(MXSA) == []` and no `ds_read MXSA`
        # appeared in ANY block.
```


## `emit.py`, line 1242

```
    # dropped so no OOB chunk-T read; the deepest copy drops first, off_p ≤ M-1-t).  This is where
    # the last M wmmas live; the read-ahead is suppressed (keep="drain") — the pipe only empties.
    # DRAIN is the STAGGERED tail (Lemma 1): M straight-line steps, one per drained pipeline
    # generation.  The read-ahead is NOT dropped uniformly — a drain step `t` STILL prefetches the
    # next generation while `t + dr_iters < M` (it feeds a LATER drain step), and only the LAST
    # dr_iters steps SUPPRESS the read-ahead (their prefetch would read the out-of-bounds reduction chunk T).
    # This is exactly ULM0's NGLL (early drain step, read-ahead kept, = a "no-global-load" steady
    # iteration) → NLL (final drain step, read-ahead suppressed).  A step that keeps read-ahead
    # uses keep=None (full head+ahead reads); a suppressing step uses keep="drain".  (Bug fix: the
    # old code used keep="drain" for EVERY step, dropping NGLL's read-ahead so both drain steps
    # were identical pure-drain iterations — the NGLL generation was never prefetched.)
```


## `emit.py`, line 1255

```
        # trailing drain steps that SUPPRESS the read-ahead = the DERIVED chunk reach `r`
        # (`peel.reach`): the read-ahead the emitter actually issues spans `dr_g` substeps = that
        # many whole reduction chunks of prefetch that run off the end.  A step in that trailing
        # window guards its OOB read-ahead (the `_readahead_shift`-derived Pred); an earlier step
        # (NGLL) keeps full read-ahead (it feeds a later drain step).
        #
        # `peel.reach`, NOT `ceil(dr / red_ext)`: `dr` is the REQUEST and the emitted advance is the
        # §2.6-clamped `dr_g`, so the raw form widened this window past the reads it guards whenever
        # a group was clamped.  Same quantity, same one derivation, as the chunk-level peel `M`.
```


## `emit.py`, line 1272

```
    # prologue meets the drain).  A FULLY-MODELED body (a complete standalone LoopIR): the M drain
    # steps alone, which already do the peeled fills+computes with no steady loop between them.  It
    # is bound to concrete chunk indices (i = t), so it is a real executable program.  The GIR
    # scaffold pass recognizes this degenerate els == the drain blocks and routes it to TensileLite's
    # existing openLoop/NoGlobalLoadLoop short-path selection (no double-emission); a from-scratch
    # backend runs it directly.
    #
    # PER-STEP VALIDITY.  This arm runs when T < M, so the short trip is [0,T) — STRICTLY SHORTER
    # than the M steps peeled here, and T is a runtime problem dimension (§5.2), not a compile-time
    # bound.  Step t must therefore be guarded by `T > t`: without it, a step t >= T copies a
    # reduction chunk that does not exist (an out-of-bounds global read) and accumulates its garbage
    # into C.  This is the SAME Lemma 1 hypothesis the root guard carries, applied per peeled step —
    # the root selects full-vs-short, these select HOW MANY short steps run — so it is decoder-
    # synthesized problem-dimension scaffolding (§5.2 line 389), a generic `kind`, no scaffold name.
    # Step 0 is left unguarded: `T > 0` is trivially true wherever a reduction level exists at all,
    # and the paper (line 389) says a trivially-true guard carries no branch.
    #
    # THE DEGENERATE ARM IS THE UNPIPELINED EMISSION OF θ (Lemma 1, paper 2026-08-15):
    # *within the arm, every obligation's producer precedes its consumer.*
    #
    # That single property — stated over the BODY, not over `off` — is what the arm needs, and the
    # three displacements a steady body carries all collapse under it:
    #   * `off`-carried — a staggered copy ramp at this level, or a substep read-ahead `dr` inside
    #     it, puts a producer in a later step.  Setting both to 0 is a CONSEQUENCE of emitting
    #     unpipelined, not the mechanism (see `copy_insts("degenerate")` and `nest(read_shift=0)`).
    #   * rotation-carried (positional) — even at `off = dr = 0`, a `mod-S` rotation writes each
    #     read after the `wmma` that vacates the register it refills.  At `W = 1` that is the ONLY
    #     pipeline the body has and no `off` expresses it (see the σ_c note below).
    #   * the arm's σ_c is the SEQUENTIAL within-arm order, not the rolled steady one.
    # The steady body assumes a PREDECESSOR TRIP supplied its opening producers — the reads that
    # fill the first `wmma`'s registers, the copies that make the first read's reduction chunk resident — and
    # the prologue supplies those in the full tree.  A fully-peeled level has neither.
    #
    # This is not a refinement — building the arm with the steady `nest()` was WRONG.  It carried
    # the steady body's read-ahead into a tree with nothing running ahead of it, so at PLR>0 step 0
    # read `lds[buf1]` (which only step 1 fills) and its first wmma consumed a register no read in
    # the arm wrote.  The ledger could not catch it: RAW residency is symbolic per op-class and
    # generation-blind (§4.1), so `await A@shared` was discharged by A's copy of chunk t while the
    # read wanted chunk t+1 — the right completion class, the wrong generation.  At `dr = 0` every
    # read discharges against its own step's copy, which is what closes that gap (§5.3.1 pt5).
    #
    # `suppress_ahead` is consequently absent here rather than passed False: at `dr = 0` there is
    # no ahead read to suppress, so the guard is vacuous BY CONSTRUCTION, not by a choice.
```


## `emit.py`, line 1319

```
        # NO σ_c HERE — and that absence is the point, not an omission.
        #
        # Both of `move_reloads_after_last_use`'s deferrals order for a SUCCESSOR TRIP: (1) a WAR-refill copy is
        # moved after the reads that vacate the buffer it refills, and (2) an in-place refill read
        # (`W = 1`, so the slot it overwrites IS the slot in use) is moved after the `wmma`s that
        # vacate it.  Both are correct in a rolled body, where the thing they feed is the next
        # trip.  The arm has no next trip, so applying either puts a producer AFTER its only
        # consumer.  Deferral (2) is the paper's "rotation-carried (positional) displacement", and
        # it is invisible to `off`: at `W = 1` a read cannot be a generation ahead numerically, so
        # the rotation is the ONLY pipeline the body has and zeroing `off` does not touch it.
        #
        # The skeleton order is already the sequential one — `copy_insts` first, then `build_level`
        # emitting each level's reads ahead of its inner nest and leaf `wmma` — so emitting it
        # unchanged IS "every producer precedes its consumer within the arm", and it is a legal σ
        # (Proposition 7: it respects `ord` and per-agent program order; there is simply no
        # successor trip to feed).
```


## `emit.py`, line 1354

```
        # STRICT `T > M`, not `T >= M`.  The paper's §5.2 skeleton writes `>=` and its NB says
        # either is correct — but the NB's justification is that "at T = M the full tree is
        # selected but its steady loop has trip count 0 (a header that never executes)", which
        # presumes a PRE-TESTED header.  The lowering emits the steady region as one block that IS
        # the body with the test on its terminator, and a POST-test cannot express zero trips: its
        # body always runs at least once.  Under `>=`, T == M would enter the loop for one trip
        # computing chunk 0, which the drain also computes — the same double-compute #228 is about,
        # at the boundary.  Strict is the shape's requirement, not a preference.
        #
        # It is also exactly what TensileLite's own `openLoop` emits: `unrollLoopEntryEndCounter`
        # returns M for PGR 1/2/3 and the scaffold skips the loop on `LoopCounter <= M`, i.e.
        # enters iff T > M.  The two agreeing is a check, not the reason — the reason is the shape.
        #
        # Consequence, deliberately taken: the short arm is now reachable at T == M, so it is live
        # code for every M rather than dead at M == 1.  It is correct there (the arm is the
        # unpipelined emission of θ, Lemma 1), and `ShortPathFold`'s VACUOUS verdict retires.
```


## `emit.py`, line 1415

```
    # `ord` cleanliness (§5.1).  A NON-CONTIGUOUS role IS SUPPORTED: a read invariant over an
    # ENCLOSING mode is emitted once under an explicit first-touch `Cond(mode == 0)`, which is
    # exactly what that guard exists for — it merges into placement when it can, and "only if it
    # genuinely CANNOT merge (a non-contiguous role, §5.1) does it become an explicit Cond".
    # Verified end-to-end on LoopOrder 'MKNMKN' at wave tile [4,4]: op parity holds (one read per
    # distinct presence coord, no duplicates), verify_gir passes, ledger empty.
    #
    # §5.1's multi-sub-body steady region is the SHAPE it prefers — hoisting those guards into the
    # loop structure rather than testing them per iteration.  That is a cost refinement (#171),
    # not a correctness requirement, and the paper is explicit the decoder must NOT refuse here:
    # such a point "is legal and in Π ... every Chapter-4 obligation is discharged", and the IR
    # admits multiple sub-regions "RATHER THAN forcing a single body and FLAGGING THE KERNEL AS
    # UNEMITTABLE".  Recorded as a note on θ, nothing more.
```


## `fuse.py`, line 54

```
#: WAVE SHARES per group member — how many waves of the group's slice each member gets, as
#: RELATIVE integers.  `None` means the historical PARITY rule (see below).
#:
#: The member order in `FUSE_GROUPS` is load-bearing once shares are uneven: member i takes a
#: CONTIGUOUS wave range, so `[A:2, MXSA:1, MXSB:1]` at NumWaves 4 means A gets waves 0-1, MXSA
#: wave 2, MXSB wave 3.  That is the whole point of the uneven share — A is ~32x the bytes of a
#: scale at MXBlock 32, so giving each member an equal half would leave half the waves nearly idle.
#:
#: `5` (paired) READS ASYMMETRIC ON PURPOSE: `[MXSA, A]` and `[B, MXSB]`, not `[MXSA,A]` and
#: `[MXSB,B]`.  With contiguous ranges that puts MXSA+B on waves 0-1 and A+MXSB on waves 2-3, so
#: each half of the block moves one whole data tensor plus one scale.  Ordering the second group
#: the "matching" way would put both data tensors on waves 2-3 and both scales on 0-1 — the
#: structural imbalance that made `paired` look useless.
#:
#: `1` and `4` KEEP `None` = the shipping PARITY rule (`SBitcmp1B32(waveIdx,0)` selects the member,
#: `wCompId = waveIdx >> 1` selects the slice).  Parity interleaves where a range would be
#: contiguous; both are valid partitions of an even split, and leaving the shipping pair on the
#: rule it already emits keeps those kernels byte-identical.
```


## `fuse.py`, line 83

```
# WHY `5` IS PARITY AND NOT `[[1,1],[1,1]]`.  Both express "split the group's waves evenly between
# its two members" and they are IDENTICAL in load balance; they differ only in WHICH two waves each
# member gets — parity gives `{0,2}` / `{1,3}`, a contiguous range gives `{0,1}` / `{2,3}` — and
# `wCompId` absorbs that difference, because it is the index of the wave WITHIN its member's set
# either way.
#
# The decisive fact is that the scaffold's wave-separated TDM is PARITY ALL THE WAY DOWN: 16 sites
# emit `_emitTdmWaveParitySCC` / `SBitcmp1B32(WaveIdx, 0)` (descriptor init, global offset, the
# increment seed, the WrapU select, the stagger reset, ...) and 5 more carry a single `tcPeer`
# variable that structurally cannot name a third member.  A contiguous range disagrees with every
# one of them, and the disagreement is silent: the kernel assembles and computes the wrong answer,
# which is exactly how `paired` reached hardware twice.
#
# So `paired` takes the rule the backend already implements everywhere, and needs from this work
# only the part that IS genuinely new for it — the descriptor aliasing `MXSA→A` / `MXSB→B` and the
# pairing that follows from `FUSE_GROUPS`.  `2`/`3` get no such reprieve: `[2,1,1]` is uneven AND
# 3-way, so neither parity nor a cselect can express it, and they need the range selector at every
# one of those 16 sites.
```


## `fuse.py`, line 128

```
#: The groupings the SCAFFOLD can realize TODAY.  Every value in `FUSE_GROUPS` decodes and lowers
#: cleanly through θ → GIR → `check_plan` (measured), so this is an EMITTER bound, not a model one.
#:
#: ALL of them are emittable as of 2026-08-25.  2, 3 and 5 need the CONTIGUOUS RANGE selector
#: `FUSE_WAVE_SHARES` specifies, which `KernelWriterAssembly.tdmWaveSelect` now emits alongside
#: the historical parity one; `calculateStartAddrWaveSeparated`
#: takes the member's `waveFirst`/`waveCount` so `numComp` is the range width rather than
#: `NumWaves // 2`, and the power-of-two assert is scoped to the parity branch it belonged to.
#:
#: 4 is the same PARTITION as 1 (the lists differ only in order), so it costs nothing to allow.
```


## `ir.py`, line 73

```
# The single extension point: a (src, dst) hop maps to the completion class that tracks it.
# Add a row to support a new source/dest/class — every Load picks it up.
#
# This IS π, specialized: §5.3.1 line 527 — "a concrete backend commonly DERIVES an op-class's
# completion class from its hop type ... This is sound but it FIXES π to the target's counter set
# — the single legal policy when the hardware exposes exactly one counter per hop kind, not a
# search over π ... an implementer targeting one profile is choosing that specialization, not
# removing the axis."  So deriving the class from the hop is correct and blessed; it is a fixed
# profile, not a missing search dimension.
```


## `ir.py`, line 157

```
    # §5.3.1 pt5 SHAPE: the read-ahead advance this transfer carries, in flat presence positions
    # (`dr_g x stride_pres(reduction)`).  `0` is INPLACE — the read is NOT hoisted; it issues in
    # place, one line before its own consumer.  Non-zero is PREFETCH — hoisted ahead of the consumer
    # it feeds.  Carried on the node because it is what decides whether a refill has anything to
    # DEFER (`sigma_c`): an INPLACE read is already after the consumer it vacates (the previous
    # generation's, across the back edge) and moving it past this generation's consumer strands
    # that consumer with a register nothing filled.  Derivable from `coord` (whose values are all
    # `None` at zero advance) but stated, because the shape is a first-class fact of the schedule,
    # not a side effect of how the coordinate happened to be rendered.
```


## `ir.py`, line 167

```
    # §2.2 MOVEMENT QUANTUM: an `ir.QuantumMap` — `carrier(t)` (which instruction carries tile `t`)
    # and `slot(t)` (its position inside that instruction).  `None` (the default) is the identity
    # merge: one tile per instruction, `coord` alone is what it fills.
    #
    # It is NOT redundant with `coord`.  An axis inside the quantum is ABSENT from the hop's
    # presence (§2.1: "the coordinates inside one quantum are one presence point"), so it does not
    # appear in `coord` at all — which is precisely why the count has to be stated.  Without it the
    # node says "one transfer, one coordinate" while the hardware instruction fills `Π extent` of
    # them, and every consumer at a non-leader coordinate looks unfed.
    #
    # `theta.Hop.quantum` carries the same `QuantumMap` (two Exprs over the tile coordinate); `loopir_to_gir` expands this into one
    # register `Ref` per covered coordinate, so ONE instruction produces `Π extent` DEFINITIONS —
    # the SSA-honest reading of a multi-tile load, and what lets reaching-def match the second
    # tile's `wmma` to the load that actually filled its register.
```


## `ir.py`, line 212

```
# Rolled θ-nest IR (paper Chapter 5 "emit IR").
#
# emit_can produces the FULL ROLLED LOOP NEST that matches θ exactly — one real `Loop` per ord
# mode, each op-class appearing ONCE at its presence level (copy at the outer (reduction chunk) loop, read at its
# presence level, wmma at the innermost leaf), with slot/coord as SYMBOLIC index expressions
# (`Expr`).  Nothing is enumerated in the IR (Inst count = O(#op-classes × #levels), NOT trip
# count — the paper's O(1) property).  Register rotation is carried by the SYMBOLIC `Placement`
# slot Expr (`index % S`), per §5.2 (symbolic slot and an explicit Branch are interchangeable
# views; the code uses the symbolic slot).  `Branch` is used for FIRST-TOUCH guards and peel
# reduction-residue pinning, not per-buffer rotation.
# The RENDERER does the partial unroll (expand leaves, interleave read;wmma) as a view — the IR
# itself stays rolled.  Each θ field is carried (§5.2); only the wait COUNT and physical COLOR
# are downstream lowerings (§5.6).
```


## `ir.py`, line 259

```
                             # (terms, add, ((coef, div, ext), …)) →
                             #   + Σ coef · (((add + Σ c·env[m]) // div) mod ext)
                             # `carry` below extracts ONE floor-divided quantity; this extracts
                             # SEVERAL DIGITS of the same shifted position `P + add` and recombines
                             # them in mixed radix.  It is what a rotation slot needs when the
                             # group cycles over MORE THAN ONE reduction rate mode: the shifted
                             # rate index is `Σ radix_m · digit_m(P + shift)`, and each `digit_m`
                             # is its own `(// stride_m) mod ext_m`.  With a single rate mode it
                             # degenerates to one digit and coincides with `carry` whenever the
                             # outer `mod` equals that mode's extent (W == R), which is every
                             # `unroll`-policy group.
```


## `ir.py`, line 271

```
                             # + (add + Σ coef·env[m]) // div  over carry_terms=((mode,coef),…).
                             # Models the PLR read-ahead ROLLOVER into the next reduction chunk (`iter`)
                             # (§2.1 step = data − off, §2.5 `ord` a permutation → monotone
                             # mixed-radix counter): the read at flat ord-position
                             # P = Σ stride_ord(m)·idx(m) fetching P+shift targets reduction chunk
                             # iter + (P+shift)//N_inner.  For K outermost-of-inner (KMN/KNM)
                             # this collapses to the single-mode (iter+(K+dr)//S_sub); for any
                             # other order it is the general carry that keeps the shared-buffer
                             # sequence MONOTONE (one back-edge swap), which a pointer XOR toggle
                             # can express.  add=shift, div=N_inner.
```


## `ir.py`, line 436

```
#
# THE CLOSED SET OF LEDGER HAZARD CLASSES, and the only place it is enumerated (#276 S1).
#
# Eight sites used to dispatch on `kind.endswith("WAR")`.  That is a SUFFIX TEST STANDING IN FOR A
# VOCABULARY, and it fails in both directions the moment the vocabulary grows: a new anti-dep whose
# name does not end in the letters WAR silently joins the RAW arm, and a new true dep that happens
# to (a hypothetical `read-after-WAR`) silently joins the anti-dep arm.  Neither shows up as an
# error — both show up as a schedule.  `ledger.py:448` already records one measurement of exactly
# this: matching the suffix forced copy-last on `crossing-RAW` obligations that needed copy-FIRST,
# and the fix there was to narrow to an explicit kind.
#
# The membership test is not merely a tidier spelling of the suffix test: `is_war` is TOTAL over a
# CHECKED set, so an unknown kind raises here instead of being silently classified as RAW at eight
# call sites.  A vocabulary that cannot be extended by accident is the point.
#
# NOTE the collision this also prevents: `Lowering/gir/analyses/lds_hazards.py:64` defines a
# SECOND, unrelated vocabulary of bare `"RAW"`/`"WAR"`/`"WAW"` hazard-edge labels.  Those are edge
# classes in the LDS hazard graph, not ledger obligations; `"WAR"` there matches the suffix test
# here, so the two vocabularies were one typo away from being cross-dispatched.
```


## `ir.py`, line 522

```
    #: `anchor` — POSITION, NEVER VALUE (#331).
    #:
    #: A read-ahead's refill is legal only AFTER the last consumer of the register name it
    #: overwrites: with a body of `T` steps a name written at `p` and last consumed at `u` is live
    #: for `(T - p) + u`, so `p > u` is exactly the condition that one generation is live at a
    #: time.  For an operand broadcast over an OUTER mode that last consumer is in a LATER PASS of
    #: the broadcast axis than the read's own coordinate, so the read cannot be positioned by its
    #: own coordinate alone.  `anchor` supplies the position; `op` still supplies the value.
    #:
    #: THE ALTERNATIVE WAS REJECTED, and the reason is the whole point of this field: emitting the
    #: read AT ITS CONSUMER'S LEAF would make the Inst's coordinate disagree with what it reads, so
    #: a reader of the IR could no longer tell which tile/generation a read is for.  LoopIR must
    #: stay faithful — the read says `A{K0,M0}[gen v+1]` and the anchor says WHEN, and the two are
    #: separate facts because they are separate questions.
    #:
    #: A COORDINATE, NOT AN INDEX.  The nest is rolled; a flat position does not exist until the
    #: walker flattens it and is invalidated by any reordering.  A coordinate is stable across the
    #: walk, is the vocabulary `Tile.coord` / `Bind` / `_inner_steps` already speak, and stays
    #: checkable after emission (the ledger asserts "this refill follows its anchor").
    #:
    #: PARTIAL, naming only the modes that DIFFER from the Inst's own coordinate — under `NKM`,
    #: A's refill carries `(("N_inner", 1),)`, i.e. "same (K, M), deferred to the last pass of the
    #: broadcast axis".  An anchor naming every mode would be an obfuscated index and would not
    #: survive a change of `ord`.
    #:
    #: ANCHOR EXISTENCE IS THE ADMISSION TEST.  `placement.register_reuse_verdict`'s walk already
    #: records every name's read/write events, so the last read event IS the anchor; when the last
    #: consumer falls in the NEXT trip there is no in-trip anchor and `dr_g = 0` stands.  One walk
    #: yields both the verdict and the placement, so they cannot disagree — the failure mode that
    #: produced the 2026-08-28 miscompile was exactly a verdict relaxed while the placement stayed
    #: where it was.
```


## `ledger.py`, line 86

```
            # A global→shared hop under Φ is ONE OBLIGATION for the whole movement — both
            # endpoints name it, so the members' entries collapse in the `set` below and a single
            # object is what every member's consumer matches (§2.8 move 9, §5.2).
            #
            # BOTH ends, not just the producer.  Naming only the producer leaves two obligations
            # that happen to share a producer, and two obligations are two discharge accounts: A's
            # reads settle one and B's reads the other, so neither read is ordered against the
            # other member's data even though one instruction moved both.  The movement completes
            # once; there is exactly one thing to owe and one thing to discharge.
            #
            # That was the multi-wave TDM defect — the copy fused but the obligation did not, and
            # the prologue read-ahead consumed shared memory the cooperative load had not landed.
            # Every OTHER hop (shared→register) stays per-op-class: a read IS per op-class.
            # `movement_name` returns the bare name for a lone movement, so an unfused kernel keeps
            # one obligation per operand exactly as before.
            # LEMMA 2 TAKES A PRODUCER **SET** (paper rev 2026-08-20), and naming the MOVEMENT is
            # that set, conservatively.  A region-split `dst` has several writers — one region-TDM
            # per region value — so a read whose footprint spans regions has
            # `producers(r) = { region-TDM_j : j in regions(pres(r)) }`, and the paper is explicit
            # that "filing against a proper subset (a read's home region only) leaves the other
            # regions' bytes unawaited — a residency the empty-ledger gate would report satisfied
            # while the read consumes stale data."
            #
            # ONE obligation on the movement awaits every region of it, which is exactly the
            # conservative producer set the paper blesses: "the decoder cannot narrow `producers(r)`
            # below 'every region the operand is split into,' so it conservatively awaits the whole
            # region-TDM group — which is *exactly the unsplit copy's residency* … so the safe
            # default costs only a declined optimization, never a barrier."
            #
            # SO DO NOT KEY THIS BY REGION.  Per-region narrowing is the OPTIMIZATION, reachable
            # only through the `PerRegionCompletion` pi toggle (§2.8 move 7), and it is sound only
            # once the reader's FOOTPRINT is known to lie in the one region its coord names — which
            # base theta cannot know, because it is address-opaque (the enrichment's `phi`, §7, is
            # what earns it back).  `decoder_proto/halfwave_region_table.md` measures where that
            # fails: a half-wave pair straddles a region in 12 of 84 swept configs.
```


## `ledger.py`, line 127

```
        # ONE OBLIGATION PER PLACEMENT (#212).  Iterating the map's own rows means a region-split
        # operand yields one WAR per region — each half is its own buffer, vacated by its own read
        # — instead of one WAR for a tile that does not exist.  Unsplit operands have exactly one
        # row (`region is None`) and are unchanged.  `region` is not consumed below yet; keying the
        # ENDPOINTS by it is #217's half, and doing it here without the matching token/hazard keys
        # would pair region-0 obligations against region-agnostic discharges.
        # the WAR is discharged against the VACATING READ's completion class (§4.1 line 207):
        # for a shared buffer that is the shared→register read (final hop, DS), NOT the refilling
        # copy (first hop, class C_copy); for a register group it is the operand's final hop.  In both
        # cases: the read that vacates the buffer = the operand's LAST hop.
```


## `ledger.py`, line 139

```
        # IN-PLACE vs ROTATION.  The distinction is not the width but WHICH NAME the refill writes:
        # in-place means it writes the name the co-located consumer is reading, so the refill must
        # be ordered AFTER that consumer (§2.6's post-WAR case, §5.3.1 pt4's mandated order) rather
        # than hoisted.  `W == 1` is one way to get there (one slot, so the refill always collides);
        # the other is a read-ahead advance that wraps the operand's whole name space
        # (`shift ≡ 0 mod names`), which collides at W > 1 too — a degenerate free fan does exactly
        # that.  Keying on `depth < 2` alone left the second case tagged `rotation-WAR`, so σ_c did
        # not defer it and the hoisted refill clobbered the operand its own wmma was reading.
```


## `ledger.py`, line 149

```
            # §2.4: the shared ring satisfies `S >= δ`, and at `S == δ` the refill is IN PLACE —
            # the copy overwrites the very buffer this trip's reads vacate, so the mandated
            # copy-after-read order applies.  Keying only on `depth < 2` called that a rotation:
            # at PGR2 (`S_shared = 2`, `δ = off(copy,iter) = 2`) the ring is exactly δ deep and the
            # steady copy targets `gen[v+2]`, which is slot `v mod 2` — the slot being read now.
            # Mislabelling it left P6 and the σ_c gate treating an in-place refill as a rotation,
            # i.e. not order-constrained, which is only harmless while something else happens to
            # emit it last.
```


## `ledger.py`, line 164

```
    # CHUNK-CROSSING RESIDENCY (§5.3.1 pt5 "The chunk-crossing coupling").  The substep read-ahead
    # reaches `r` chunks ahead, so a steady trip reads chunk `iter+r` and owes
    # `complete(copy of chunk iter+r) ⤳ issue(read)` on the SHARED hop.  This is a DIFFERENT
    # obligation from the rotation-WAR on the same buffer, and conflating them is what let a
    # miscompiling point through: on one buffer in one trip
    #   WAR  — copy(iter+r) after the last read of chunk `iter+r-S_shared`   (an EARLIER trip)
    #   RAW  — the crossing reads of chunk `iter+r` after copy(iter+r)       (THIS trip, if off==r)
    # Only the WAR was modelled, so nothing forced the copy to precede its own consumers.
    #
    # It bites only when the copy lands in the SAME trip as the crossing read, i.e. `off(copy) == r`:
    #   off > r  — the copy ran in an earlier trip; no in-body order constraint, no obligation.
    #   off == r — in-body; discharged iff σ_c emits the copy BEFORE the crossing reads (copy-first),
    #              which is legal exactly when `S_shared > r` (placement.copy_first_is_legal).
    #   off < r  — the chunk is not copyable by the reading trip at all; unsatisfiable by construction.
```


## `ledger.py`, line 413

```
        # QUANTIFIED OVER ALL READS (§2.7 line 185: `∀ r ∈ reads(gen t), ∀ w ∈ writes(gen t+S):
        # complete(r) ⤳ issue(w)`).  The test is therefore LAST READ before FIRST REFILL, not "SOME
        # read precedes the refill": a running `read_before` SET discharged the whole quantifier on
        # the operand's FIRST read, so a copy spliced into the MIDDLE of the read stream — the
        # local-write interleave any latency-hiding scheduler emits — passed with every later read
        # of that trip consuming the buffer it had just overwritten.  At `S=δ` the refill lands on
        # THE BUFFER THIS TRIP IS READING (that is what `inplace` means), and the next chunk's
        # reads belong to the next trip, across the back edge, so within one trip a read after the
        # refill is always a clobbered read.  MEASURED on the shipping tree: 62 in-place shared
        # refills span 103 (r,w) pairs, of which an existential inspects 62; injecting that
        # interleaved σ makes 15 op-classes violate the forall and the existential flags only 6.
        # The one shape whose reads legitimately FOLLOW the copy is the chunk-crossing operand of
        # §5.3.1 pt5 — and it carries `crossing-RAW`, never `inplace-WAR` (the two sets are made
        # disjoint by the `off >= r+1` coupling, verified over every reachable θ), with arm (b3)
        # below as its mirror.  Coarse coverage is unsound for an ORDER-DRIVEN backend (§5.5 line
        # 731) and so is a coarse quantifier: the gate must validate the POSITION of every edge.
        # EMISSION-CHECKED: the whole bf16 matrix, 5744 kernels, 0 byte-different vs the
        # existential — this tightens what is REJECTED, it does not move any shipping schedule.
```


## `ledger.py`, line 477

```
    # (b2) order, REGISTER ring: a HOISTED in-place refill read must follow a wmma consuming its
    # group.  The `advance` qualifier is the §5.3.1 pt5 shape and is required for the same reason
    # `sigma_c.is_inplace_read` carries it: an INPLACE read (`dr_g = 0`, hence zero advance) is NOT a
    # refill hoisted above its consumer — it is the in-place producer FOR that consumer, "issued one
    # line before its own wmma", and the generation it overwrites belongs to the previous trip,
    # whose consumer precedes it across the back edge.  Flagging it would report the ONLY correct
    # order as the violation: with the read moved after the wmma the gate falls silent and the
    # kernel miscompares instead (measured at `n_s == 1`, where `W == 1` forces `dr_g = 0`).
    #
    # WHY THIS ARM IS AN EXISTENTIAL AND ARM (b) IS NOT (§2.7 line 185, deliberately asymmetric).
    # On the SHARED ring "no read after the refill" IS the forall, because at `S=δ` the refill
    # overwrites the buffer this trip reads and the next chunk's reads sit in the next trip.  On
    # the REGISTER ring a consumer after the refill is the ORDINARY case — it consumes the
    # generation that refill just produced (RAW), not the one it overwrote — so transposing the
    # same position test here would flag the only correct order.  The register forall is
    # per-GENERATION: every wmma consuming gen `t` out of the slot must precede the refill that
    # writes gen `t+S` into it, and telling those wmmas apart needs the consumer's generation, not
    # its position.
    #
    # TWO THINGS WERE FILED HERE AS "real gaps, currently unreachable"; they are NOT the same, and
    # only one of them was a gap (#262).
    #   * `consumed` NOT BEING RESET was a genuine defect and is FIXED below: a refill CONSUMES the
    #     mark, so a SECOND refill of the same slot in one body needs its own preceding consumer.
    #     Without the reset only the first refill per slot per body was checked at all.  The fix is
    #     strictly stronger and rejects nothing that was previously accepted-and-correct.
    #   * THE PER-GENERATION FORALL IS NOT A GAP — IT IS NOT EXPRESSIBLE HERE, and that is a
    #     property of the IR, not an omission.  §5.2: the tree is ROLLED, and a rotation slot is
    #     carried as a SYMBOLIC coordinate expression (`c mod S`).  Two generations of one slot are
    #     therefore the SAME NODE; there is no "the wmma consuming gen t" to distinguish from "the
    #     wmma consuming gen t+S" without unrolling, which is a renderer VIEW (§5.2) and not the
    #     object this gate runs on.  What discharges the generation distinction is the DEPTH FLOOR
    #     `W >= L_war` (Lemma 3b), checked at build time — a width that could not hold both
    #     generations is rejected before any order question arises.  So the positional existential
    #     is the strongest form statable over a rolled tree, and it is sound given the floor.
    #
    # REACHABILITY, re-measured 2026-08-28 over the same 4032 configs (#321).  The bound WIDENED
    # and the arm is still sound; both halves matter.
    #
    # It used to be `[1,1]` only — 288 hits, every other MIWaveTile zero — with every qualifying
    # body (1 refill, 1 wmma), which is why the existential and the forall COINCIDED.  That was an
    # artefact of `PrefetchLocalRead` being keyed to the K substep: every `plr2` cell was silently
    # running `dr <= 1`.  With the read-ahead on its region level (`theta.readahead_level`) a
    # depth-2 look-ahead on an extent-2 axis wraps onto itself, so `[1,2]` (20) and `[2,1]` (10)
    # now qualify too, in bodies of (2 refills, 2 wmma).
    #
    # THE COINCIDENCE IS GONE; THE SOUNDNESS IS NOT, because it never rested on the coincidence —
    # it rests on the depth floor argued above, and `W >= L_war` was verified to hold at ALL 174
    # arm-qualifying (op, group) pairs.  `test_the_register_ring_arm_reachability_bound` now
    # asserts that floor directly (strictly stronger than the old shape enumeration) and keeps the
    # shape/body counts as a measured bound, so a further widening is still noticed and has to be
    # re-justified against the floor rather than waved through.
```


## `ledger.py`, line 529

```
    # A PLACEMENT BLANKS THE LABEL OF A SINGLE-GROUP OPERAND (`placement._read_placement`:
    # `gl = group if len(all_groups) > 1 else ""`, so the vgpr name carries no `_g0` suffix), while
    # the ledger endpoint always names the real group.  Without this normalization `('A','')` never
    # matches `('A','g0')` and the whole register-ring arm is VACUOUS for every unpartitioned
    # operand — it only ever fired on a VgprPartition scenario, whose labels are real (`lo`/`hi`)
    # and which #118 keeps out of emit anyway.  Resolve the blank against the operand's own groups,
    # taken from the ledger itself (register endpoints are the non-`shared` ones); only a unique
    # group is resolvable, which is exactly the case the blank denotes.
```


## `placement.py`, line 291

```
    # every (op-class, level) offset θ carries, as a per-op-class map — the staggered peel's input.
    #
    # NEGATIVE entries are recorded, not dropped, so they can be REJECTED with a real message.
    # §2.4 (2026-08-15) types the retime matrix `off : (op-class x level) -> N` — non-negative
    # throughout — and `retime` SETS the entry rather than subtracting, so a negative `off` is
    # ill-typed rather than merely unmodelled.
    #
    # DEFERRAL still exists, and needs no negative: it is a POSITIVE `off` on a REVERSE-TRAJECTORY
    # op-class (a store), which Lemma 1 gives the REFLECTED peel — trailing `d` instances hoisted
    # FORWARD into the successor iteration, prologue and drain swapped.  Direction comes from the
    # trajectory sense, not the sign.  See `emit._check_peel_is_emittable`.
```


## `placement.py`, line 466

```
    # A BROADCAST-OUTER OPERAND PRIMES ITS WHOLE NAME SET (#331).  Its advance is a full traversal
    # (`prefetch_distance_for`), so the first steady trip consumes EVERY one of its names from the
    # prologue, not just the leading substep's — the three-factor set below is the answer for a
    # sub-trip advance and under-primes this one by exactly the substeps it does not cover
    # (measured `NKM`: 2 of A's 4 names primed, and the `T < M` arm reported the other 2 as
    # consumers "no read in scope has filled").  This is Lemma 3d's INNER = ∅ rounding — "the first
    # tile's FULL K" — applied one level out, at the broadcast axis rather than the substep: the
    # steady must sustain exactly what the prologue primed, and here the steady refills the whole
    # tile once per trip.
    # AND SO DOES ANY OPERAND WHOSE STEADY READ IS DEFERRED (#332).  The rule above is "the steady
    # must sustain exactly what the prologue primed", and what decides whether a read is still
    # available EARLY in the trip is the ANCHOR, not `reloads_whole_set`.  A read anchored to a later
    # pass of a mode it is invariant over no longer fills its names before that pass, so the
    # prologue owes the FULL set for the first trip exactly as a broadcast-outer operand does.
    #
    # `reloads_whole_set` ("outer to EVERY presence mode") is the stronger condition, so it implies an
    # anchor and this subsumes it rather than competing with it.  Gating the prime on the fan while
    # the DEFERRAL is decided by the anchor is a one-predicate-two-rules split, and it is what the
    # multi-wave fix exposed: measured `MNK` + `TDMSplitA=1, TDMSplitB=2`, A is anchored at
    # `N_inner == 1` but primed only `{}` and `{M_split:1}` — both at `K_splitB = 0`, i.e. rotation
    # slot 0 only — so the first trip's `wmma u=1` read buffer X1 with nothing having filled it
    # (`USE-BEFORE-DEF ... reads A buffer X1 tile 0`).  4 configs, hard codegen errors.
    # BUT ONLY IF THE STEADY ACTUALLY ADVANCES (#332).  Both conditions below are properties of the
    # `ord` shape alone; neither implies the operand HAS a read-ahead.  The rule this function
    # states above is `prologue depth = steady advance = dr_g`, and `dr_g` is DERIVED — it can be 0
    # while the requested `dr` is not, and then a Shape-A prologue sits on a Shape-B steady, which
    # is the named failure: the pipe is primed and never re-primed.
    #
    # The `if not dr` guard at the top is the REQUESTED depth and does not cover this.  MEASURED,
    # mxf8 `KMN MT64x64x128` (MI_K = DepthU = 128, so `ord = [M_inner, N_inner]` with NO reduction
    # inner mode and `dr_g = 0` for every operand at every PLR): at `PLR = 0` the top guard returns
    # `[]` and the kernel passes, while at `PLR >= 1` B and MXSB — `reloads_whole_set`, since
    # `M_inner` is outer to `pres(B) = {N_inner}` — get their whole name set primed and the steady
    # then re-reads them in place.  That is the PLR discriminator on those cells, and it is a
    # mixture this function's own contract forbids.
```


## `placement.py`, line 543

```
        # A FREE SIBLING IN `OUTER` IS ENUMERATED, NOT PINNED (§5.3.1 line 644).
        #
        # "{first coord of pres ∩ OUTER}" is right for a mode the ADVANCE CARRIES THROUGH: tile 1
        # is primed by the steady advance itself, so hoisting tile 0 alone is sufficient.  A
        # storage-disjoint SIBLING is exactly the axis the advance does NOT carry through —
        # `_readahead_shift` walks `reload_modes`, which drops it, so the shift stays inside one
        # sibling — and therefore each sibling's leading substep is primed by nothing.  MEASURED at
        # `MKN` with A split: `wmma(idx0=2,..) reads A buffer X0 tile 2, which no read in scope has
        # filled` — tile 2 is sibling 1's first tile.
        #
        # §5.3.1 forces `dr_g = 0` only ACROSS the boundary and says in the same breath that "each
        # sibling still pipelines internally at its own `dr` over its own reload modes".  Priming
        # every sibling is what makes that sentence true of the emission: the cross-sibling advance
        # stays absent (the shift never crosses), while each sibling gets its own Shape-A fill.
        #
        # Same shape as the fold enumeration below, and for the same reason — both are OUTER
        # coordinates the advance cannot reach, so both must be hoisted explicitly rather than
        # inferred from tile 0.
```


## `placement.py`, line 687

```
    # THE `{lo}{hi}` PARTITION IS A SIBLING ENUMERATOR TOO, and the paper lists it in the same
    # breath as the free-axis region split ("a region split on a FREE axis, **the `{lo}{hi}`
    # partition**, or a reduction axis realized as concurrent siblings").  A register partition
    # makes the grouping mode's values select a DIFFERENT GROUP's band, so a read-ahead that
    # carried through it would target the neighbouring group's register.
    #
    # MEASURED (#310) at `M·K·N`, `VgprPartitionA=2`, PLR=1: `lo` (dr_g=1) sits at the K level with
    # `M_inner` — its own grouping mode — outer to the reduction, so the shift at `(m=1,k=1)`
    # carried into `(m=2,k=0)`, which is `hi`'s tile.  `check_plan`: "operand A reads source
    # coordinate(s) [(2,0)] MORE THAN ONCE per trip" and "never reads [(0,0)]" — the neighbouring
    # sibling's register, exactly the failure §5.3.1 pt 5 describes.
    #
    # ONLY WHEN ACTUALLY PARTITIONED.  At `parts == 1` there is one group spanning every value of
    # the grouping mode, so it enumerates no siblings and the fan is genuinely reloaded — excluding
    # it there would narrow the radix for every shipping kernel.  Gating on `len(parts) > 1` keeps
    # this confined to register-partitioned operands, which no YAML can request today.
```


## `placement.py`, line 932

```
    # SOLVED AGAINST THE WALK, NOT GUESSED (§2.6 Q37).
    #
    # Lemma 3b is `W_g >= L`, so the deepest admissible look-ahead is the deepest `d` whose live
    # peak still fits the ring: `max{d : L(d) <= W_g}`.  `L` is what `group_live_peak` computes,
    # and it already prices BOTH §2.6 contributors — term (i) through the interval's `-off`
    # extension, term (ii) through `stride`, "the broadcast fan the value is held across".
    #
    # THE TERM-(ii) CONSTANT IS GONE, REASON AND ALL.  `if _broadcast_fan(op) > 1: return 0` was
    # justified as "L = dr_g + 2 > W_g".  §2.6's Q37 refutes that `+1` outright — the broadcast-hold
    # is a PRODUCT over the pinned rotation modes, `B = Pi extent(r)`, "not a `+1`" — and says how
    # it combines with a depth-`dr` read-ahead "is not given a closed form: it is `ord`-dependent
    # and computed by the walk".  The assignment walk below IS that computation, so the constant is
    # not replaced by a better constant; it is replaced by the quantity it was approximating.
    #
    # §5.3.1 pt5's `dr_g <= W_g - 1` is kept as an explicit cap: the walk should imply it, and
    # stating it means a walk regression cannot silently exceed the ring.
```


## `placement.py`, line 957

```
        # THE ASSIGNMENT VERDICT IS TAKEN OVER EVERY GROUP OF THE OPERAND, not just this one.
        # `_readahead_shift` emits ONE advance per operand (`_readahead_depth` mins over the
        # groups), so a per-group answer that differs is the Shape-A-prologue / Shape-B-steady
        # mixture this function's docstring names: the min zeroes the steady advance while the
        # prologue keeps priming at the deeper group's depth.  MEASURED when it was per-group:
        # 14 partitioned KMN/KNM plans reported `USE-BEFORE-DEF ... reads B buffer X1 tile 0,
        # which no read in scope has filled`.  The old vetoes were per-OPERAND and so never
        # produced the mixture; the walk has to be read at the same granularity.
```


## `placement.py`, line 966

```
        # ADMISSION IS "IS THERE A LEGAL POSITION", NOT "IS THE DEFAULT POSITION LEGAL" (#331).
        #
        # `register_reuse_verdict` scores the refill AT THE STEP THE READ IS ISSUED.  A `clobber`
        # there means that position is wrong — not that every position is.  The refill is legal
        # anywhere after the last consumer of the name it overwrites, and `reload_positions`
        # returns exactly that position per name, or `None` when the last consumer survives the
        # back edge and no in-body position exists.  So `anchors is not None` IS the admission
        # test, and it subsumes the verdict: `safe` yields `{}` (nothing to defer).
        #
        # THE TWO MUST FLIP TOGETHER.  Admitting `d` here makes `_readahead_shift` non-zero, which
        # is what makes `emit._read_anchor` fire and actually move the read.  Relaxing admission
        # WITHOUT the emitter honouring the anchor is the 2026-08-28 miscompile — the read stayed
        # where it was while the model believed it had moved.  They are coupled through this one
        # derivation precisely so that cannot recur.
```


## `placement.py`, line 1084

```
    # THE SHIFT ADVANCES ALONG THE READ-AHEAD LEVEL (#321) — the axis `off(read, ·)` is written on.
    # It was `_reduction_rate_mode`, the innermost REDUCTION rate mode, which is a different axis
    # whenever the reduction is not outermost and does not EXIST at `DepthU=64` (the ord is
    # `M_inner(4) N_inner(4)`; there is no reduction inner mode at all).  It returned None there,
    # so the shift was 0 and the whole read-ahead was emission-inert: MEASURED, `PLR=0` and `PLR=1`
    # produced byte-identical IR (385 lines) at DU=64 for all four of KMN/MKN/NKM/MNK, while
    # DU=128 differed.  Fixing the LEVEL alone moved `dr` 0->1 and changed nothing downstream —
    # the level and the axis the shift walks have to be the same derivation.
```


## `placement.py`, line 1095

```
    # A BROADCAST-OUTER OPERAND READS AHEAD BY A WHOLE TRIP, NOT BY A SUBSTEP (#331).
    #
    # When the operand is invariant over a mode OUTER to its read-ahead level, every one of its
    # names is consumed once per pass of that mode — so a name is live from its first pass to its
    # last, i.e. for the WHOLE TRIP.  No sub-trip advance can be legal: the refill would land
    # before the later pass that still reads the old value, whatever the width (measured: clobber
    # at W = 1..8 alike, because the name carries no generation and `idx % W` is constant for
    # `W >= 2`).  The only legal look-ahead unit is one full traversal — the same tile, next trip.
    #
    # AND THAT IS WHAT MAKES IT EMITTABLE.  At `shift = n_pres` the shifted coordinate EQUALS the
    # leaf's own coordinate, so the read's position and the name it fills coincide; anchored to the
    # last pass of the broadcast axis (`emit._read_anchor`) each refill then lands immediately
    # after its own last consumer.  At a partial shift they are offset and no per-leaf guard can
    # align them — one rolled Inst has one position while its four names need four.
    #
    # This is §5.3.1's tile-granularity read-ahead one level out: the paper gives it for INNER = ∅
    # ("pre-issuing the next TILE's K"); the same argument at the broadcast axis gives the next
    # TRIP's tile.  `preloaded_tiles` primes the matching full set.
```


## `placement.py`, line 1181

```
    # TWO SPACES, ONE WALK (#332).  The advance and the name are not measured in the same space,
    # and collapsing them onto either one is a miscompile:
    #
    #   ADVANCE  -- `prefetch_distance_for` moves the read along `reload_modes`.  A free-axis region
    #               sibling BOUNDS the shift (§5.3.1 pt5): it selects a different group's band, so
    #               the shift never crosses it.  Computing the advance over `_presence` instead
    #               makes the walk validate a schedule the emitter does not produce -- MEASURED,
    #               `MNK` + MT split, A: the walk said the read-ahead writes tile 1 while the plan
    #               writes tile 0, the same register the next two wmmas consume.
    #   NAME     -- the register the emitter fills is `(free presence tile, _rate_slot mod W)`.
    #               BOTH components span PRESENCE, not reload.  Naming over reload merges the
    #               region sibling's generations into one name: MEASURED, same config, `M_split`
    #               is in `pres` but not in `reload`, so `M_split=0` and `M_split=1` shared a name,
    #               the walk saw one refill where the plan emits two, and it reported a clean
    #               verdict over `[(13,'use'), (15,'refill'), (17,'use')]`.
    #
    # So: enumerate and name over `pres`; advance over `_rl`; carry the modes the shift does not
    # move (`held`) in the generation value, because a write at `M_split=0` and one at `M_split=1`
    # with the same reload position are DIFFERENT values.
    #
    # `rate` is `_rate_modes` UNFILTERED, exactly as `_rate_slot` reads it.  Restricting it to the
    # advance space was the same collapse in the slot digit: the emitter rotates on every rate
    # mode, so a walk that rotates on a subset names a register the plan does not.
```


## `placement.py`, line 1343

```
            # A LATER CONSUME IS AN UNRESCUABLE RACE AT THIS POSITION, and `sigma_c` is NOT a way
            # out (measured 2026-08-28, a miscompile).  Relaxing this to "clobber only if the
            # consume survives the back edge" reasons that σ_c's `is_inplace_read` will defer the
            # refill past any consume still inside the trip.  It will not: `move_reloads_after_last_use` defers
            # WITHIN EACH NEST LEVEL, and for an operand broadcast over an OUTER mode the surviving
            # consume is the next pass of that outer loop — rescuing it needs the read hoisted OUT
            # of the loop (loop-invariant motion), which σ_c does not do.  The relaxation produced
            # `mma acc{N1,K0,M0}` reading `A.register{K0,M0}[s0]` AFTER the read-ahead overwrote
            # `s0`; `test_readahead_advance_never_clobbers_across_the_matrix` caught it.
            #
            # "AT THIS POSITION" is the whole reason `at` exists.  Moving the read is the OTHER way
            # out, and it is a different schedule, not a relaxation of this one: `reload_positions`
            # proposes the move and then re-asks this question with `at` set.
```


## `placement.py`, line 1382

```
    # ONE Inst HAS ONE POSITION.  The refill is emitted by a single rolled read whose position is a
    # guard on the modes it is INVARIANT over, so all of its names must anchor at the SAME value of
    # each such mode.  A §2.2 quantum fold breaks that — the folded axis leaves presence, and the
    # names it carries end up wanting refills at different passes of it (measured `NKM` at `q = 2`:
    # `M_inner` wanted both 1 and 2).  Serving that needs the read SPLIT BY NAME, which is a
    # `covers`/`_read_groups` change, not a guard change.
    #
    # NO LEGAL SINGLE POSITION IS THE SAME ANSWER AS NO LEGAL POSITION: return None so the
    # admission test falls back to `dr_g = 0` — the in-place refill, which is exactly what these
    # configurations emitted before and is correct.  Raising here instead turned six previously
    # building matrix cells into hard errors, which is a worse answer than the one they had.
    # THE SAME MODES `emit._first_touch_modes` GUARDS ON, or this misses the case it exists for:
    # a PARTIALLY FOLDED axis stays IN presence (§2.2 keeps its `N/q` carrier-group coordinates)
    # yet is still guarded, by `mode % q == 0`.  Testing "not in presence" alone let the folded
    # `M_inner` through and the conflict surfaced downstream as an emit-time assert.
```


## `placement.py`, line 1585

```
    # SHARED SOURCE BUFFER (`src_slot`) = the reduction chunk reduction-chunk multibuffer the read's DATA lives in
    # (§5.2 line 333 `slot = c mod S_Λ(m)`, m = the outer-reduction level).  `off` and `S` are SEPARATE θ
    # fields (§2.4 retime vs §2.6 multibuffer), and the reduction chunk coord is composed from the substep
    # level per §2.4 coupling #1 (`off` is per (op-class × level); `tile` splits K into the reduction chunk
    # level and the substep level):
    # The reduction chunk coord is ORD-DEPENDENT: the read-ahead is a shift along the ORD TRAVERSAL, not a
    # raw substep bump (paper 2.1 `step = <reduction-index, ORD-STRIDES> - off`; 2.4 coupling #1
    # places off(read,substep)=dr on the substep level; 2.5 makes the traversal a monotone
    # mixed-radix counter over ord).  Over the read's PRESENCE modes (ord order):
    #   P        = sum stride_ord_pres(m) . idx(m)
    #   shift    = dr . stride_ord_pres(K)           dr substeps expressed in ord-position units
    #   src_slot = ( reduction chunk + (P + shift) // N_pres ) mod S_shared
    # PRESENCE-ONLY (not all inner modes): an operand broadcast over another operand's fan (A over
    # N) does NOT count that fan; excluding it prevents the WT>1 cross-operand leak (folding ALL
    # inner modes in IS the leak).  Because read-ahead advances the FLAT ord position, a K-wrap
    # inside a tile just moves to the next FAN tile (same chunk), so the reduction chunk crosses at most ONCE
    # per trip -> a MONOTONE sequence for EVERY order (one LocalReadAddr swap the pointer XOR can
    # express).  This is ord-DEPENDENT by construction: K-outer-of-presence -> 0,0,0,0,1,1,1,1;
    # K-inner-of-presence -> 0,0,0,1 -- both monotone.  A reduction-only (substep+dr)//S_sub carry
    # is WRONG (ord-independent -> bounces 0,1,0,1 for K-inner and misnames the buffer).
```


## `placement.py`, line 1612

```
            # ONE shift for the whole read (see `_readahead_shift`): the same advance the coord and
            # the register slot use, so the three cannot drift.  It used to be `dr * stride(K)`,
            # which misses the INNER=∅ full-`n_s` factor and so crossed into the next chunk one
            # substep too late.
            # P = sum stride_ord_pres(m).point(m), via the SHARED `_pos_terms` (#312) — the third
            # and last place this position is built, and the one that decides WHICH reduction chunk
            # the read-ahead lands in.  Left in mixed units it read a tile-valued loop variable
            # against point-valued strides, so a folded axis inflated `P` by `q` and the crossing
            # fired `q`-fold early: measured at ReadPhi=2 KMN as the short arm's reads at
            # `M_inner` 4 and 6 demanding shared generation 1 that no block writes.
```


## `placement.py`, line 1633

```
# THE QUANTUM IS AN INPUT TO θ, LIKE `S`.  Which tiles one instruction covers is a consequence of
# the ADDRESS LAYOUT and the emitter's folding behaviour, and θ is address-opaque (§2.1) — it has
# no way to see byte adjacency and must not pretend to.  So the pairing is computed OUTSIDE, where
# the layout is known, and enters θ already translated into GEMM semantics: *which tile coordinates
# form one quantum*.  That is exactly how `S` works — a per-placement map whose value comes from
# outside, which θ then reasons over structurally (§2.6) — and `bridge.ReadVectorElems` is the same
# pattern one level down.
#
# WHAT θ DOES WITH IT.  §2.2: the quantum's axes are ABSENT from the hop's presence, so the
# coordinates inside one quantum are ONE PRESENCE POINT — "one step, hence one generation".  The
# pair is therefore inseparable BY CONSTRUCTION: `ord` orders modes, and no reordering can place a
# mode inside a single presence point.  There is nothing to test.
#
# WHAT θ MUST NOT DO, all four tried and all four wrong (2026-08-19/20):
#   * scan `free_modes` for an axis that "looks contiguous"  — address-thinking in disguise;
#     at MIWaveTile [2,2] it lands on `M_split`, the region axis, and issues verdicts about the
#     region/cross-wave question, which is a SEPARATE problem with a separate treatment;
#   * test whether the paired coordinates share a generation and NARROW when they do not — that
#     tests the un-quantized schedule, which is not the one being emitted, and throws away the wide
#     load the layout exists to serve;
#   * refuse the geometry when the pair straddles a TDMSplit region — regions of one chunk share a
#     generation, so Q is satisfied; the region question is Lemma 2's producer SET;
#   * `tile`-split an axis to CREATE the pair — the pairing is not θ's to choose.
```


## `render.py`, line 397

```
                    # PIN ONLY WHAT THIS ENV CAN RESOLVE.  `Placement.at` -> `Expr.eval` substitutes
                    # 0 for a name the env does not carry (ir.py:318) -- correct for evaluation, and
                    # a fabricated number here.  The DRAIN body's `iter` is deliberately unbound
                    # (`Bind.bound_env` returns `{}` because `T` is runtime), so pinning rendered
                    # every drain read in the FIRST steady trip's frame -- `lds[buf0]` under a
                    # `# iter = T-1` header -- with nothing saying the index was invented.
                    #
                    # The guard lives HERE, not in `Placement.at`: `at` is on the codegen path and
                    # handing a symbolic Expr back to it breaks the emitter (measured: 409 tests and
                    # both codegen runs). The renderer is the only caller that can be handed an
                    # env that does not close the expression, so it is the one that must ask.
```


## `render.py`, line 455

```
    # THE LEGEND MUST REPRODUCE THE TABLE, and this one did not.  It read
    #     footprint = Σ generations(S) × frag_reg(width) × free_tiles + accum
    # which misses the storage-REGION factor.  `operand_footprint_regs` (sizing.py:107) is
    # `region_count × Σ_part(W_part × |part|) × frag_regs`, and `free_tiles` stands in for
    # `|part| × region_count`.  Those agree while the region mode is a FREE mode and diverge the
    # moment it is a REDUCTION mode (K_split), where the region multiplies on top of the whole
    # free fan: measured 16 of 60 mxf8 operand rows (every operand of every order at split=2)
    # printed vgpr=128 against a legend that computes 64.  The header two lines up promises
    # every number is checkable by hand; a reader doing that arithmetic concludes the vgpr
    # column is wrong, when the missing factor is a quantity this table did not even show.
```


## `render.py`, line 481

```
    # §5.3.1 pt5 SHAPE, per register group.  `dr_g = max(0, min(dr, W_g - 1))`, zeroed outright by
    # a broadcast fan (§2.6 term ii), and it decides EVERYTHING about how the read is scheduled:
    #   A (dr_g >= 1) hoisted ahead of its consumer, prologue primes the pipe;
    #   B (dr_g  = 0) in place one line before its own wmma, prologue EMPTY.
    # It was absent from this dump, and its absence is why two INPLACE defects shipped: nothing a
    # reader could see distinguished "this read is hoisted" from "this read is in place", so a
    # PREFETCH prologue paired with an INPLACE steady (and its mirror) looked identical to a correct
    # schedule.  `W_reg` alone does NOT tell you — a broadcast fan gives dr_g = 0 at any width.
```


## `sigma_c.py`, line 110

```
        # AN ANCHORED READ IS DEFERRED TOO, WHATEVER ITS WAR CLASS (#332).  `Inst.anchor` says the
        # refill must follow a LATER PASS of a mode the operand is invariant over — but the anchor
        # only picks WHICH PASS (it is the first-touch guard's constant).  WHERE INSIDE that pass is
        # this function's job, and keying it on `inplace-WAR` alone left an anchored ROTATION read
        # sitting BEFORE the very consumer its anchor was chosen to follow.
        #
        # MEASURED, `MNK` + `TDMSplitA=1, TDMSplitB=2`: A is invariant over `N_inner`, so one A
        # register serves both `idx1` values; its anchor is `N_inner == 1`, correctly the last pass,
        # yet the emitted order is `wmma idx1=0 u=1 | read A buf=1 | wmma idx1=1 u=1` — the refill
        # between two consumers of the same register.  A is `rotation-WAR` (shift 1 over a reload
        # traversal of 2), so the `inplace-WAR` test skipped it.
        #
        # The two facts are one decision: an anchor EXISTS only when a name's last consumer is later
        # than its refill, which is exactly the condition σ_c is here to repair.
```


## `sizing.py`, line 94

```
        # |part| = # grouping-mode values in each part.
        #
        # ACCEPTED SCOPE EXCEPTION to §6.2 (decided, not an oversight): we support EQUAL partitions
        # only.  §6.2 states |part| as a PER-PART count and notes that uneven partitions
        # (lo={0}, hi={1,2,3}) then need "no scalar and no equal-partition assumption"; we take the
        # equal case deliberately and REJECT the rest loudly below rather than compute a wrong
        # footprint.  Lifting it is future work: carry each group's own value-subset size on the
        # Fragment (which already knows its values via `size_of`) and sum W_part × |part| per part,
        # deleting both the scalar and the guard.
```


## `sizing.py`, line 108

```
        # `|part|` IS A RESIDENCY, NOT AN EXTENT — the ord half of §2.6 (feature #4).
        #
        # `resident_free_tiles`' own docstring states the rule: "When the operand's free axis is
        # INNERMOST it turns over cleanly: only a WINDOW of tiles is live at once (min footprint)
        # ... When the free axis is OUTER ... every free tile is revisited each inner pass, so all
        # are resident (full fan)."  That quantity was computed and then not used here: `|part|`
        # took the grouping mode's FULL extent regardless of `ord`, so the footprint came out
        # IDENTICAL for every order and every TDMSplit — measured 64 in all nine (order x split)
        # cells at bf16 `MIWaveTile[4,4]`, while `resident_free_tiles` reported 4 for KMN/KNM/NKM/
        # NMK and 2 for MKN/MNK.
        #
        # IT IS ALSO THE UNIT `W` IS COUNTED IN, so it is NOT computed here — `theta` owns the one
        # definition (`group_unit_tiles`) and `group_rate` divides by the same thing.  When these
        # were two derivations they disagreed: `R` in tiles against `|part|` in resident tiles gave
        # `foot(A) = 256` for 64 registers of data at `MKN`/`partsA=2` (see `group_unit_tiles`).
        # The product `W x |part|` is only meaningful when `W` counts BUFFERS of exactly this
        # `|part|`; with both from one source the footprint is `ord`-invariant at `W = R` (the same
        # data IS the same registers) while `R` itself stays `ord`-variant.
```


## `theta.py`, line 111

```
# An op-class is ONE operand's movement along ONE hop, so `off`'s row index is the pair
# (operand, hop) — not the operand alone.  The paper writes these roles out: `off(copy, iter)`,
# `off(read, substep)` (§5.3.1), so the vocabulary is the paper's, not a target's.
#
# Keying `off` by the OPERAND alone worked only while each operand had at most one offset per
# level (copy at `iter`, read at `substep`).  The per-level peel (§5.3.1's cross-level boundary
# term) asks for `off(p, ℓ_outer)` for a `p` whose HOME level is INNER to `ℓ_outer` — i.e. a READ
# carrying an OUTER-level offset — which under the operand-only key is indistinguishable from the
# COPY's offset at that same level.  The role is what tells them apart.
```


## `theta.py`, line 138

```
# `HOP_ROLE` is keyed on (src, dst) alone, so it answers a question about ONE LEG.  That is the
# right key for `off` — §2.4 indexes the retime by (op-class × level) and the op-class is (operand,
# hop) — but it CANNOT answer which way the whole trajectory runs, and the two are independent:
# in a store round trip `vgpr → LDS → vgpr → global` the middle `shared→register` leg is a
# read-BACK, so `HOP_ROLE` correctly calls it a READ while the trajectory it belongs to is REVERSE.
#
# The distinction is load-bearing for Lemma 1's REFLECTED peel: a forward op-class hoists BACKWARD
# (prefetch — prologue fills, drain drains), a reverse one hoists FORWARD (defer — prologue and
# drain swapped).  `emit` states this and reads the direction "off the op-class's trajectory
# sense", which until now nothing computed — #209 keyed on the role and got away with it only
# because no reverse trajectory is modelled yet (#116).
```


## `theta.py`, line 187

```
    # §2.2's CARRIER GROUP, in the paper's OWN vocabulary: "a carrier group is exactly a `Φ` fuse
    # group (§2.8 move 9) over the tiles that `ρ` (§2.8 move 2) has placed on the agents that one
    # instruction spans … `θ` NAMES THE MEMBERS".  These two integers are that naming, and
    # `geometry.derive_quantum` turns them into the `carrier`/`slot` pair above.
    #
    # `phi_width`  — Φ: how many of this hop's movement instances merge into one cooperative
    #                instruction (the CONTIGUOUS shape: consecutive tiles on the spanned agents).
    # `rho_span`   — ρ: HOW MANY SUB-AGENTS the group is distributed across (§2.2's "sub-wave
    #                partition", a `tile` of the lane mode).  0/1 = none; 2 = the MX TileSpan
    #                half-wave, where one `ds_load` holds block 2g in the lower half-wave and
    #                block 2g+1 in the upper, and the WMMA's `matrix_{a,b}_scale` selector picks.
    #                A COUNT OF AGENTS, not a tile stride: an earlier version passed
    #                `span["vectorWidth"]` here, which made the guard `rho % phi` compare a vector
    #                width against an instruction width — at VW=1/Φ=2 that is 1 % 2 != 0, so
    #                `derive_quantum` returned None and θ modelled NO fold while the leaf folded
    #                anyway.  MEASURED 2026-08-24: that disagreement is the mxf8 `MIWaveTile [2,8]`
    #                + TileSpan miscompare, 49152/65536 wrong on LONKM/LONMK.
    #
    # WHY THEY EXIST RATHER THAN A SUPPLIED MAP.  §2.2 is explicit that "the quantum is NOT a new
    # field" and that its membership "comes in two shapes, both already in θ".  A prebuilt
    # `QuantumMap` handed down from the target is a THIRD representation: it cannot be searched (ρ
    # is a searched distribution), and nothing checks it against ρ/Φ, so the two can silently
    # disagree.  Carrying the two numbers instead keeps one source of truth and makes the map a
    # derivation.  `quantum` remains accepted as an explicit override for a target whose fold is
    # not expressible as (Φ, ρ) — it is then a declared deviation, not the default path.
```


## `theta.py`, line 214

```
    #: IS THIS HOP'S STORAGE REGION A PROPERTY OF THE AGENT RATHER THAN OF THE COORDINATE? (#245)
    #:
    #: The region coordinate says which region the COORDINATE names.  It is the region the ACCESS
    #: touches only when every agent running this program touches the same one.  `LraTileAssignment`
    #: offsets each wave by `strideWave = numTileInInst * matrixInstT * VectorWidth`, so at
    #: `MIWaveGroup[2,2]` wave 0's tiles land in region 0 and wave 1's -- executing the SAME two
    #: read instructions -- land in region 1.  One instruction, two regions, chosen by agent.
    #:
    #: A token must then name every region the instruction can reach, or the read is ordered
    #: against one producer and left un-ordered against the other.  False (every `MIWaveGroup[1,1]`
    #: shape, and every unsplit operand) = the coordinate IS the region and the token is exact.
```


## `theta.py`, line 271

```
    #: How many storage-disjoint sibling regions this operand's FREE axis is cut into (1 = uncut).
    #:
    #: THE ORD-INDEPENDENT ANCHOR FOR §5.3.1 pt 5's SIBLING BOUND (#245).  `region_modes` answers
    #: the same question today, but it answers it by NAMING a mode, so it is only as durable as
    #: that mode's presence in `ord`: a region axis whose extent collapses to 1 is dropped by
    #: `_canonical_ord._keep`, which empties `region_modes` and SILENTLY RELEASES the veto that
    #: #274 installed -- measured, the read-ahead carry returns and MXSA's fold pair is cut across
    #: blocks.  The split is a fact about STORAGE, not about which mode happens to carry the
    #: coordinate, so it is recorded here as a number.
    #:
    #: FREE AXIS ONLY, deliberately.  A reduction-axis (DU) split is NOT a sibling enumeration --
    #: the paper keeps it in the traversal ("Reduction axes ... ARE reload modes and stay"), and
    #: folding it in here would break the two-axis `K_split x K_inner` rotation (#254, measured 12
    #: failures).  `_free_siblings` already makes that distinction by name; this field makes it
    #: without needing the name.
```


## `theta.py`, line 287

```
    #: How many of the operand's FREE REGIONS **one agent** spans (§2.6 / #245, #311).
    #:
    #: `free_split` is the TILE's region count; this is the WAVE's.  They differ, and the register
    #: footprint needs THIS one: `translate` sets the in-region fan as `mInner = fanM //
    #: wave_region_span`, so when a wave sits inside one region `M_inner` already holds the WHOLE
    #: `MIWaveTile` and multiplying by the tile's region count counts the same tiles twice
    #: (measured #311: `MIWaveTile[8,8]` + `TDMSplit=[2,2,1,1]` doubled `foot(A)` 512 -> 1024,
    #: when a region split partitions the SAME tiles and cannot create data).
    #: §6.2's `region_count × Σ(W × |part|)` reads `|part|` as PER-REGION, so the factor that
    #: reconstructs the wave's fan is exactly the divisor `translate` applied — this one.
```


## `theta.py`, line 325

```
#: The agent hierarchy, ORDERED COARSE -> FINE.  This tuple is the ONLY hardware vocabulary in
#: the model, and it is deliberately thin: the decoder compares and orders these labels and never
#: learns what one is made of.  §3.3 leaves the level NAMES to the target and fixes only the shape
#: (`(mode, agent-level, extent)`), so this tuple is one concrete instantiation, not the model.
#:
#: §2.8 line 215 (paper rev 2026-08-25) settles what opacity does and does not forbid:
#:   "`ρ` is a MAP, not an opaque token … opacity is about a placement's CONTENT — which byte,
#:    which physical register — NOT about the mode→level map.  Typing `ρ` as an equality-only
#:    token (on the mistaken ground that 'address-opaque' forbids knowing what a wave is) makes
#:    the mode→level facts unstatable and forces ADDRESS-SHAPED substitutes (lane-layout
#:    comparisons, address queries) into the hops to recover them."
#: That is this implementation's history, named in the paper; see RHO_DESIGN.md.
#:
#: `subwave` IS A DATA-DISTRIBUTION LEVEL, NOT A SYNCHRONIZATION SCOPE (§2.2, §4.6 Assumption 1):
#: completion classes are wave-scoped and sub-wave lockstep is SIMD-structural, so no `Await` is
#: ever scoped below `wave` — §5.2's scope ladder starts there.  That is why `served_modes()`
#: reads `coarser_than("wave")` and never the whole map.
```


## `theta.py`, line 476

```
    # ρ (§2.8 move 2 `resort` + move 8 `specialize`; §2.9 line 244 "agent assignment + role
    # partition") — see the `Rho` docstring above for the type and why it is not a selector map.
    #
    # R1 STATUS (#303): BUILT AND CHECKED, NOT YET READ.  `translate` populates it from the same
    # two inputs the live per-hop fields use, and `rho_consistency()` asserts the two agree across
    # the matrix.  The consumers (`agent_served_modes`, `Hop.rho_span`, `Hop.region_agent_relative`)
    # still compute their own answers; R2/R3 switch them over once the agreement is established.
    # An empty ρ therefore means "no agent-served mode", which is the correct reading for every
    # single-wave, unsplit kernel — not "unknown".
```


## `theta.py`, line 495

```
    # π (§2.8 move 7, §3.3 "discharge policy — which class names each obligation"): the COMPLETION
    # CLASS GRANULARITY of a storage-region-split movement.  A region split emits one movement
    # instance per region value (§2.8 move 9); whether those instances share one completion class
    # or carry distinct ones is a π choice, and §4.3 states the split reading — "two fused
    # movements with TWO COMPLETIONS, each covered by atomic_fuse independently".
    #   False (default) — all region instances of an op-class share ONE completion class, so a
    #                     read must await EVERY region.  No region can remain in flight.  This is
    #                     what the consuming backend implements today.
    #   True            — each region instance is its own completion class, so a read of region j
    #                     awaits only region j and the others stay in flight (§4.6's counting rule
    #                     then DERIVES the overlap: n = min(δ, remaining_matching) > 0).
    # Realizing True end-to-end needs a per-region completion token on the backend side; LoopModel
    # and GIR model both, and the L3 adapter rejects True until that exists.
```


## `theta.py`, line 782

```
    # A COLLAPSED REGION-LEADING AXIS MEANS THE READ-AHEAD CROSSES REGIONS.
    #
    # Normally the region modes are dropped and the level is the outermost axis INSIDE one region.
    # But a DU `TDMSplit` re-factors the reduction as `K_split x K_inner` and can leave `K_inner`
    # with a single value — the region then contains exactly ONE step of its own leading axis, so
    # "advance one step" and "advance one region" are the same move, and the level is the region
    # axis itself.  Falling through to the free fan instead would make a K-outermost order read
    # ahead along `M`, which is a different schedule, not a deeper one.
    #
    # BOTH BRANCHES ARE LOAD-BEARING, measured:
    #   * keep-regions always -> `MIWaveTile[8,8]`, `TDMSplit=[1,1,2,2]` (`K_split(2) K_inner(4)
    #     M_inner(8) N_inner(8)`) puts the level on `K_split`, `W` 2 -> 8 and `foot(A)` 128 -> 512,
    #     breaking #311's invariant that a region split cannot change the register footprint.
    #   * drop-regions always -> the mxf8 `KMN x TDMSplit2` path (`K_split(2) K_inner(1) M_inner(2)
    #     N_inner(2)`) sends A to `M_inner` while `MXSA`, whose load SPANS `M_inner`, cannot follow;
    #     the drain then reports `INCOMPLETE CARRIER GROUP ... the instruction at carrier 0 serves
    #     2 tiles but only 1 of their acts are in this block`.
    # The test asserting each is `test_a_region_split_cannot_change_the_register_footprint[tds2]`
    # and `test_the_quantum_check_is_CLEAN_on_every_cell_now_that_the_carry_is_gone`.
    # NO FALLBACK ONTO A REGION AXIS.  When the region holds no non-degenerate axis this op-class
    # can advance along, it has NO in-region read-ahead and the answer is None — the level must not
    # fall back to the region enumerator itself.  Carrying the advance ACROSS a region boundary is
    # precisely the straddle #274 removed ("the read-ahead advance was carrying across a region
    # sibling, which is what put the two merged tiles on different generations"); reintroducing it
    # on the reduction-axis region brings the straddle back, measured as
    # `KMN x TDMSplit2 ... INCOMPLETE CARRIER GROUP` on the mxf8 fold path.
```


## `theta.py`, line 905

```
    # NAME-VS-RELOAD IS AN `ord` FACT, NOT A STRUCTURAL ONE.
    #
    # The paragraph above is right that a `grouping_mode`'s values PARTITION storage — but only
    # while the whole fan is simultaneously resident.  Which it is depends on `ord`:
    #
    #   KMN  `k0:(m-fan), k1:(m-fan)`   4 m-tiles live at once, `k` refills them
    #                                   -> the fan NAMES, `k` reloads          -> R = 2
    #   MKN  `m0:(k0,k1), m1:(k0,k1)`   one m-tile live; when `m` advances the SAME registers
    #                                   are refilled -> the fan RELOADS         -> R = 2 x 4 = 8
    #
    # Same mode, opposite roles.  Adding it unconditionally declares it a name axis for every
    # order, which made `R` — and hence the band ceiling `W <= R` — `ord`-invariant.  Measured: R
    # was 2 for all six orders while `resident_free_tiles` reported 4 (KMN/KNM/NKM/NMK) vs 2
    # (MKN/MNK), i.e. the model already knew the answer and the rate did not ask.
    #
    # The discriminator is that same residency: a fan whose tiles are ALL resident is named (the
    # rotation is over the held modes only); a fan that turns over is reloaded and belongs IN the
    # rate.  That is the sentence this function already ends on — "a fragment with no
    # `grouping_mode` keeps its fan IN the rate, there the fan IS the rotation" — applied by what
    # `ord` does rather than by whether a `grouping_mode` was constructed.
    # A FOLDED FAN STAYS OUT OF THE RATE (#316), even when `ord` says it reloads.  §2.2's fold makes
    # `q` co-carried tiles ONE generation, so a folded axis entering the rate would be counted in
    # TILES where the rotation counts GENERATIONS — and the rate-side fold is not implemented (the
    # bridge invariant `test_no_reachable_config_puts_a_folded_axis_in_the_rate_modes` exists
    # precisely to keep it unreachable).  Letting it in produced 190 violations at `MNK` /
    # `MIWaveTile[8,8]`.  So the residency rule below applies to UNFOLDED fans; the folded case is
    # #316 and needs the rate to count `extent // q`, which is its own change.
    #
    # NAME-VS-RELOAD IS THE UNIT QUESTION, asked in one place (`group_fan_reloads`) so that this
    # set and `group_rate`'s gm factor cannot disagree.  `== 1` is exactly "one buffer covers
    # everything this group owns", which is the storage-partition argument the docstring above
    # makes — now DERIVED from `ord` and the split rather than asserted for every order.
```


## `theta.py`, line 958

```
    # THE RATE COUNTS GENERATIONS, SO A FOLDED AXIS CONTRIBUTES `N/q`, NOT `N` (§2.2).
    #
    # `q` co-carried tiles arrive in ONE instruction and are therefore one presence point, hence
    # one step, hence ONE generation.  Counting them as `N` would size the ring in TILES while the
    # rotation cycles GENERATIONS.  This was previously unreachable — the fan was declared a name
    # axis for every `ord`, so no foldable axis ever entered the rate — and
    # `test_no_reachable_config_puts_a_folded_axis_in_the_rate_modes` pinned that as a BRIDGE
    # invariant, naming `test_a_folded_axis_that_is_a_rate_mode_counts_presence_points` as "the
    # live spec for it" once it became reachable.  Making the rate `ord`-variant is what makes it
    # reachable, so the fold is applied here rather than kept out by construction.
```


## `theta.py`, line 980

```
            # THE FAN CONTRIBUTES BUFFERS, NOT TILES.  `R` counts buffer-generations, so the fan's
            # factor is `owned tiles / tiles-per-buffer` — not its extent, and not its extent
            # clamped by a per-group share.  `group_fan_reloads` is that ratio and is the same
            # function `_rate_broadcast` consults, so "is the fan in the rate at all" and "how much
            # does it contribute" are one decision (they were two, and they disagreed).
            #
            # Two wrong units were measured getting here, both at `MKN`/`MIWaveTile[4,4]`:
            #   * full extent (4)      -> R = 8, foot(A) = 256 for 64 registers of data (4x).
            #   * `Fragment.size_of`   -> `parts[i]` is a REGISTER-SLOT count, and `(1,1)` for
            #     every equal partition, so `ext` collapsed to 1 and the fan left the rate for
            #     every order — R flat at 2/4 across the whole (ord x DU x MIWaveTile x split)
            #     matrix, i.e. the arm was dead.
            # The ratio gives R = 2 (fan resident, KMN) or 4 (fan turns over, MKN) with foot(A)
            # = 128 either way.
```


## `theta.py`, line 1009

```
    # `off` OVERRIDES, so a caller can ask "what would the peak be AT DEPTH d?" instead of only
    # "what is it at the depth θ already chose?".  §2.6 (Q37) is explicit that term (ii)'s
    # broadcast-hold is a PRODUCT over the pinned rotation modes, "not a `+1`", and that how it
    # combines with a depth-`dr` read-ahead "is not given a closed form: it is `ord`-dependent and
    # computed by the walk".  This walk IS that computation — term (i) via the interval's `-off`
    # extension, term (ii) via `stride` (the broadcast fan a value is held across) — so
    # `prefetch_steps_for` solves `max{d : L(d) <= W}` against it rather than guessing a
    # constant.  `None` keeps the original two-peak behaviour verbatim.
```


## `theta.py`, line 1019

```
    # THE JOINT (FREE-TILE, ROTATION-SLOT) NAME SPACE (§2.6), not one collapsed ring.
    #
    # §2.6 is explicit that the peak is compared "per **name**, not per rotation ring: with
    # `names(p) = Pi extent over pres(p)`", and validates its closed form against "a direct
    # live-range walk on the joint `(free-tile, rotation-slot)` name space".  This walk used to
    # project the free tile modes AWAY (`_group_value_seq` keeps only the rate modes), so every
    # tile shared ONE ring and two values that live in DIFFERENT names were counted as competing
    # for one.  MEASURED on `M.K.N`, tile `m0`: the values are `k0,k0,k1,k1` — `k0` is dead before
    # `k1` is read, so the group is sequentially refillable and `L_war = 1` — but the collapse
    # merged `A(m0,k0)` with `A(m1,k0)` into one generation spanning the whole `m` fan, which then
    # "overlapped" `k1` and reported `2`.  A phantom overlap across the fan, i.e. one register per
    # group of over-provisioning on every K-in-the-middle order.
    #
    # `off` IS APPLIED IN THE NAME'S OWN GENERATION SEQUENCE, and getting this wrong is how the
    # first attempt at this fix failed: subtracting `off` in GLOBAL step index while restricted to
    # a name's SPARSE subsequence never reaches that name's previous generation (consecutive steps
    # of one tile are not adjacent globally), so it silently dropped the prefetch contributor and
    # reported `L_pf = 1` where both this walk and §2.6's closed form say `dr_g + 1`.  Sub-index
    # space is used consistently for the interval AND the sampling window.
    #
    # Safety: swept 630 groups over {6 orders} x {PLR 0,1,2} x {4 MIWaveTile} x {partition 1,2,4} —
    # the per-name floor is NEVER above the collapsed one, so no shipped kernel was ever under its
    # true floor, and K-innermost (every shipping config) is bit-for-bit unchanged.
```


## `theta.py`, line 1145

```
        # CLAMP, DO NOT REJECT (§4.4, paper L323).  Theorem 6's many-to-one statement is explicit
        # that "`multibuffer` CLAMPS `S` up to the floor, so a `θ` with `S` below the floor and one
        # at the floor emit the same kernel", and it declines the alternative by name: "making
        # every clamp a rejection rather than a repair — a normal form we do not impose, BECAUSE
        # THE REPAIRS ARE WHAT MAKE THE DECODER TOTAL."
        #
        # This branch used to `raise ValueError(...)`, which made the decoder partial on Π at
        # exactly the point Theorem 6 says it must be total, and disagreed with the policy branch
        # of this same function two lines below (which has always clamped).  An autotuner passing
        # an explicit width below the post-WAR floor — say W=1 on an interleaved deep-A group whose
        # `L_war` is 2 — aborted the whole build inside a joblib worker instead of emitting the
        # floor-clamped kernel the paper mandates.  Found by the 2026-08-21 full-paper audit (D3).
        #
        # The BAND is unchanged and still authoritative: `W ≥ L_war` because a rotation cannot
        # cycle fewer names than are simultaneously live (Lemma 3b(i), a race), and `W ≤ R` because
        # holding more names than there are distinct values wastes storage (Lemma 3b(iii)).  What
        # changes is the response to a request outside it: repair to the nearest legal width rather
        # than refuse the point.
```


## `theta.py`, line 1164

```
    # `'floor'` IS THE OLD SPELLING OF `'overlap'`, ACCEPTED, NOT REJECTED.  §2.6 renames this
    # policy — "it is named `overlap`, not `floor`" — because `L_pf` is "the *naive* floor, not the
    # operative one": the operative floor is `L_war`, which `inplace` takes and which is the band's
    # own lower bound.  The old name collides with `L_floor` ten lines up (that IS `L_war`), which
    # is a live misreading hazard in this very function.
    #
    # NOT a rejection, unlike `VgprReuseStrategy=7`.  That value was silently turned into a
    # DIFFERENT policy; `'floor'` has always meant exactly `L_pf` and still does, so refusing it
    # would buy no correctness and only break callers — the same reason the explicit-`W` branch
    # above clamps rather than raises (Theorem 6: "the repairs are what make the decoder total").
```


## `theta.py`, line 1175

```
        # `'pipeline'` = `max(L_pf, min(2, R))` — `L_pf` with a SECOND floor of 2, and that floor is
        # an EMITTER fact, not a model one (#315, #172).  It is the derived default (`translate`),
        # so read this before changing it.
        #
        # WHY `L_pf` ALONE IS NOT SAFE, MEASURED.  At `dr = 0` (PLR0) `L_pf` collapses to 1: the
        # post-WAR walk says the group is sequentially refillable — read(k), use(k), read(k+1),
        # use(k+1) — and by the model it IS.  The steady body the emitter produces does not have
        # that order for every operand: it emits the wmma for substep `k` BEFORE the read that
        # defines that substep's register, so at `W = 1` every wmma consumed the previous trip's
        # value.  MEASURED 2026-08-18 on the MX scale ring: **171 of 268 PLR=0 ULM1 cells failed
        # numerically**, 124/124 at `LoopOrder KMN`; against 6 of 264 at PLR=1, where `dr = 1`
        # forces `W = 2` anyway.  `test_plr0_register_depth_is_two` pins the A/B face of the same
        # thing.
        #
        # AND NOTHING GATES IT.  `sigma_c` position-checks SHARED-ring WARs only, not register
        # rings (#172), so `check_plan`, the ledger gate and the emitted-order walk are all clean
        # at `W = 1` — verified 2026-08-26, which is exactly why this needs to be a floor rather
        # than something a checker will catch.  Treating the gates' silence as agreement here is
        # the mistake this comment exists to prevent.
        #
        # WHY 2 IS THE RIGHT NUMBER AND NOT A GUESS: 2 is the width the measured-GOOD configuration
        # uses.  PLR=1 forces `W = 2` and passes (6/264, other causes); the failures are exclusively
        # at `W = 1`.  Two names alternate, so a wmma emitted before its defining read still reads
        # the other buffer and cannot see a half-written one.
        #
        # `min(2, R)` because a group with `R == 1` holds ONE value for the whole epoch — there is
        # no second generation for a second name to hold, so the inversion this floor prevents
        # cannot arise and `W = 1` is right.
        #
        # THIS IS A DEVIATION FROM THE MODEL AND IS LABELLED AS ONE.  §2.6's band is
        # `[L_war, R]` and `L_pf` is in it; we are choosing above the model's floor because the
        # LOWERING is narrower than the model.  The honest fix is #172 (make `sigma_c` position-
        # check the register ring, then this floor becomes provably redundant); until then this is
        # a measured guard, and removing it needs that fix plus a hardware run, not an argument.
        # AND W MUST PARTITION R EVENLY.  The rotation slot is `rate_index mod W` over a rate
        # index that cycles `R` values, so a `W` that does not divide `R` makes the wrap ALIAS two
        # generations the Lemma-3b interval argument assumed distinct: the last partial turn is
        # short, and its slots collide with the first turn's.  MEASURED 2026-08-26 at
        # `DepthU 128` + `TDMSplit[1,1,2,2]` + PLR2 — a DU split makes the rate a mixed radix
        # `K_split(2) x K_inner(2)`, so `R = 4`; `L_pf = 3` gives `slot = idx mod 3` over 4 values
        # = 0,1,2,0, and `check_register_dataflow` reports
        # "register ('A',0,0,0) held source (0,3) (never read by a wmma) and is refilled with
        # (0,0) -- the rotation width cannot hold both generations".  Rounding UP to the next
        # divisor of `R` is the smallest repair and is what the ring can actually realize.
```


## `theta.py`, line 1220

```
        # THE WIDTH MUST SATISFY THE ASSIGNMENT CONSTRAINT TOO, not just the capacity peaks.
        #
        # `L_floor`/`L_prefetch` are COUNTS — how many generations are live at once.  The rotation
        # slot is `(P + shift) mod W`, so a width that holds enough values can still MAP a write
        # onto a slot whose value is consumed later.  Measured at `MIWaveTile[1,1]` / `DepthU 128`
        # / KMN: `L_pf = 2` so this returned `W = 2`, but the advance at `INNER = 0` is the full
        # `n_s = 4` and `register_reuse_verdict` calls `W = 2` (and `3`) a CLOBBER; `W = 4` — which
        # is exactly `R` — is `inplace`.  Refusing the read-ahead there (which is what happens if
        # the width does not move) costs the whole prefetch to save 16 registers per operand.
        #
        # So: raise to the narrowest ring in `[need, R]` that the assignment walk accepts at the
        # REQUESTED depth.  One-way — the width reads the requested `off`, `prefetch_steps_for`
        # then reads the width — so there is no circularity.  If no width in the band works, this
        # falls through to `need` and `prefetch_steps_for` refuses the depth as before: the
        # capability to widen never turns a clobber into an emitted one.
```

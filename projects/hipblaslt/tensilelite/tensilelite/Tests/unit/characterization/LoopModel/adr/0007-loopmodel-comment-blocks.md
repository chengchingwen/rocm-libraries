# ADR 0007 — LoopModel comment blocks, verbatim

Comments in the code state the correct behaviour in at most three lines (Constraint 2).
The blocks below are the originals, kept whole, for the cases where the reasoning behind a
line is a measured failure worth retrieving. Each is cited from the code as `ADR 0007 #N`.

## #1 — `adapter/build.py` (was line 630)

```python
        # SCORE WHAT THE ALLOCATOR EMITS, not the model's ragged footprint — see
        # `geometry.operand_emitted_regs`.  A `{lo, hi}` partition does not shrink the emitted
        # allocation (#320), so scoring the ragged shape let this "meet" a budget with a partition
        # that changed nothing and then broke the `T < M` short arm.
```

## #2 — `adapter/build.py` (was line 460)

```python
        # A fused group pairs its members BY REGION INDEX (§4.3), which is only well-defined when
        # every member has the same region count — a mixed-split group (e.g. {MXSA(1), A(2)}) has
        # no region-1 instance of MXSA to pair with A's.  Such a group therefore moves as ONE whole
        # movement and the members' region split is NOT applied on the copy hop.  That is a legal
        # schedule (§5.3.1 line 523 blesses a coarse copy feeding a split read) but it silently
        # ignores the requested split, so SAY SO rather than let it vanish.
```

## #3 — `adapter/build.py` (was line 379)

```python
    # The ALLOCATION bound, which is not symmetric with the register side.  VGPR buffers we can
    # grow to match θ (#122); LDS we cannot — `Solution.py` sizes the LDS block from NumLdsBlk and
    # checks it against MaxLDS, so a ring DEEPER than it reserved writes past the allocation.
    # Shallower is merely wasteful (a reserved block goes unused) and is allowed, which is what
    # makes per-operand search useful here at all.
```

## #4 — `adapter/build.py` (was line 372)

```python
    # THE MX SCALE RINGS ARE THEIR OWN, PRESET TO THEIR PARENT'S.  Same mechanism as A/B, one
    # level down: the preset is the parent's RESOLVED depth rather than `NumLdsBlk`, because the
    # scale block lives in the parent's LDS allocation and rotates with the same buffer index, so
    # `LDSBufferA=1` against `NumLdsBlk=2` must carry its scales with it.  Overridable for the
    # same reason A and B are — so the depths can be searched apart.
```

## #5 — `adapter/build.py` (was line 361)

```python
    # copy-hop region split = HOW MANY regions this operand's copy moves, which is its own count
    # on each region axis — `aDU`, not the shared axis extent.  With the axis at the finest split
    # the two differ exactly when this operand is the coarser one, and that difference is its
    # STRIDE on the axis (`kSplit // aDU`), not a nameless residual factor.
```

## #6 — `adapter/build.py` (was line 311)

```python
        # DERIVED FROM ρ, not from `ReadRho` / a per-hop boolean (#303 R3).  The hop still CARRIES these
        # two numbers because `geometry.derive_quantum`, the GIR meta and both renderers read them
        # off the hop; what changed is that they are now a projection of the single authority
        # rather than a parallel one.  `_rho_ingredients` names the op-class's own axes so the
        # lookup cannot pick up a sibling operand's resort.
```

## #7 — `adapter/build.py` (was line 265)

```python
    # §2.2's CARRIER GROUP IN θ'S OWN VOCABULARY.  `ReadPhi` is Φ (movement instances merged into
    # one cooperative instruction) and `ReadRho` is ρ (the sub-agent span they are distributed over
    # — the sub-wave partition, a `tile` of the lane mode).  θ DERIVES the `QuantumMap` from them
    # (`geometry.derive_quantum`), which is what the paper means by "the quantum is not a new
    # field … both already in θ".  `ReadQuantum` stays accepted as an explicit override for a fold
    # not expressible as (Φ, ρ); using it is a declared deviation, not the default path.
```

## #8 — `adapter/build.py` (was line 255)

```python
    # THE MOVEMENT QUANTUM IS SUPPLIED (§2.2), never derived here.  Which tile coordinates one
    # instruction covers follows from the ADDRESS LAYOUT and the emitter's folding, neither of
    # which θ can see (§2.1, address-opaque) — so the bridge computes it where the target facts
    # live and hands it over already in GEMM terms.  Same shape as `S`: a value from outside that
    # θ reasons over but does not choose.  Absent = one tile per instruction, always admissible.
```

## #9 — `adapter/build.py` (was line 193)

```python
    # --- TDMSplit: [A_MT, B_MT, A_DU, B_DU] -> canonical region extents (§B) --------------
    # POSITIONAL LIST right-padded with 1s: per-operand/per-axis split, 1 = no split.  Legacy
    # bool: True → [2,2] (split both MT), False → [1,1].  MT axis is per-operand (A on M, B on
    # N — each its OWN free axis); DU axis is the SHARED K reduction.
```

## #10 — `adapter/build.py` (was line 178)

```python
    # Prefetch depths are the copy (PGR) / read (PLR) retime offsets — Lemma 1's peel is general in
    # off (any depth), so these are NOT capped: dg=PGR, dl=PLR verbatim.  (An earlier min(_,2) cap
    # was an artificial limit that broke PGR>=3; removed.  S_shared derives from off downstream, and
    # the real LDS buffer count / drain-step count follow from PGR in the same way.)
```

## #11 — `adapter/build.py` (was line 118)

```python
    # {operand: ir.QuantumMap} — the §2.2 movement quantum, SUPPLIED.  Two Exprs over the tile
    # coordinate (`carrier`, `slot`); load-vs-broadcast is the RANGE of `slot`, not a tag.  The
    # merge follows from the address layout + the emitter's fold, so it is built outside θ (the
    # bridge) and θ only evaluates it.  Absent = identity: one tile per instruction.
```

## #12 — `adapter/build.py` (was line 111)

```python
    # {operand: elements ONE shared->register instruction moves per lane} — the movement quantum's
    # DOMAIN (§2.2), which is the target's instruction selection and so has no Solution key; the
    # bridge fills it from the writer.  Empty means "one tile per instruction" for every operand,
    # the narrow and always-admissible answer (§2.2: narrowing is the repair).  A data operand
    # falls back to `LocalReadVectorWidth`, which is its true width; a SCALE has its own and must
    # not inherit that one — see `_read_hop`.
```

## #13 — `adapter/build.py` (was line 106)

```python
    # The MX scales get the SAME per-operand knobs the data operands have.  `PrefetchGlobalRead*`
    # / `PrefetchLocalRead*` need no entry here: `_per_operand` is already applied to EVERY operand
    # by name when `off` is assigned, so `PrefetchGlobalReadMXSA` works and presets to the kernel's
    # PGR exactly as A's does.
```

## #14 — `adapter/build.py` (was line 98)

```python
    # PER-OPERAND overrides.  -1 = "use the whole-kernel preset" (PrefetchGlobalRead /
    # PrefetchLocalRead / NumLdsBlk respectively), which is the historical single-value behaviour.
    # They exist because the paper's fields are per-op-class ALREADY: `off` is a matrix over
    # (op-class x level) (§2.4) and `S` is a per-placement map (§2.6) — a single scalar was our
    # restriction, not the model's.
```

## #15 — `adapter/build.py` (was line 79)

```python
# NOTE (MI shape): the default MI 16×16×64 matches the PAPER's base GEMM (paper line 52)
# and gives substep = DepthU/MI_K = 128/64 = 2 (the a/b that makes the flagship two-rate).
# The gfx1250 CATALOG fp16 opcode is actually 16×16×32 (8 VGPRs/frag); K=64 fp16 is not a
# real single opcode.  We keep the paper's shape for the flagship correspondence; a
# real-hardware config should set MatrixInstruction to a catalog opcode.  frag_elems uses
# the grounded MIInputPerThread = MI_M·MI_K·MI_B/32 regardless.
```

## #16 — `adapter/loop_order.py` (was line 155)

```python
    # SET the OUTER DepthU reduction level `iter` (extent=0 → dynamic/runtime trip = K//DU), as
    # the OUTERMOST entry of `ord` (§2.4 point 1, §5.3.1 `levels = [iter] + [substep, m, n, …]`).
    # translate OWNS the name+placement of this level (the core reads it via `outer_modes()`, never
    # by literal); its runtime trip is a problem dimension carried by emit, not stored on the Mode.
```

## #17 — `adapter/loop_order.py` (was line 146)

```python
    # `_word_to_modes` speaks the 6-letter K/M/N word, so the K region factorization is expanded
    # HERE: wherever the word places K's region axis, emit the shared link then the two residuals.
    # SHARED OUTER OF RESIDUAL, so an operand's own modes are contiguous in the traversal and its
    # region index is a plain mixed-radix product of the digits it owns.
```

## #18 — `adapter/loop_order.py` (was line 121)

```python
    # THE K REGION AXIS IS THREE MODES, NOT ONE: `K_split` (the GCD of the operands' DU factors,
    # SHARED) times `K_splitA` and `K_splitB` (each operand's own residual, `own // gcd`).  Their
    # product is `lcm(aDU, bDU)`, so the traversal is exactly the one a single lcm-extent mode
    # would give — what the factorization buys is that each operand's regions are a product of
    # modes it OWNS, which is what makes every region coordinate PINNED.  See the
    # `aRegions`/`bRegions` derivation in `params_to_theta` for why that matters.
```

## #19 — `adapter/solution.py` (was line 112)

```python
        # §2.2's carrier group as Φ and ρ — the two numbers θ derives the `QuantumMap` from
        # (`geometry.derive_quantum`).  Pass-through like `ReadVectorElems`: they are TARGET facts
        # (how many instances one instruction merges, and the sub-agent span they spread over),
        # while whether the fold is LEGAL on a given axis is θ's, by §2.2's regularity clamp.
```

## #20 — `adapter/solution.py` (was line 93)

```python
        # HOW MANY OF THOSE REGIONS ONE WAVE OCCUPIES, per operand (#245).  The COPY writes every
        # region and keeps `TDMSplit`; a WAVE reaches only the regions its own free-axis offset
        # lands in.  `LraTileAssignment` fixes that offset at `strideWave = numTileInInst *
        # matrixInstT * VectorWidth` rows per wave, so at `MIWaveGroup[2,2]` (strideWave 32,
        # region 32) wave `w` sits ENTIRELY inside region `w` and this is 1.  At
        # `MIWaveGroup[1,1]` the single wave spans the whole tile and this equals the split
        # factor, so the covered matrix is unchanged.
```

## #21 — `adapter/solution.py` (was line 86)

```python
        # ρ / agents (§2.4): how many agents cooperate on one tile.  NOT derived from the fuse —
        # fusion decides whether A and B ride one instruction, agent count decides whether a byte's
        # writer and its reader can be different agents.  A lone operand cooperatively loaded by
        # several waves is cross-agent with no fuse present, so this must be its own input.
```

## #22 — `emit/__init__.py` (was line 116)

```python
        # NAME THE θ THAT FAILED.  This raise runs inside a joblib worker during a whole-matrix
        # build, so the traceback alone says "a kernel" and nothing about WHICH — on the 2026-08-21
        # mxf8 run that turned a one-line defect into a guess-and-rebuild loop.  `ord`, the peel
        # depths and the register widths are the axes every gate failure so far has turned on, and
        # they are what a reader needs to reconstruct the cell from the log alone.
```

## #23 — `emit/__init__.py` (was line 109)

```python
    # THE STRUCTURAL GATE (§5.3.1).  Run it here, on every decode, so a tree that is not proper
    # pipelined-GEMM pseudocode is rejected AT BUILD rather than by a wrong kernel at runtime —
    # which is what `validate_loopir` was written for and, until now, never did: nothing outside
    # the unit tests called it.  Defects are decoder bugs, so raising is the correct response;
    # the observability dump path (`bridge`) catches it and reports it as a comment.
```

## #24 — `emit/__init__.py` (was line 93)

```python
    # PRECONDITION Q (§2.2).  The quantum is SUPPLIED on the hop (translate <- bridge <- the
    # target's layout+fold); this only surfaces it.  θ does not derive it: which tiles one
    # instruction covers is an address fact and θ is address-opaque (§2.1).  What θ DOES with it is
    # structural — the axes are absent from the hop's presence, so the paired coordinates are one
    # presence point and share a generation by construction, with nothing to test.
```

## #25 — `emit/__init__.py` (was line 46)

```python
# The issuing AGENT for every instance in a single-role kernel (§5.2: ρ → `Inst.a` and
# `Loop.role`; "an unspecialized kernel is the single-role special case").  ONE name instead of a
# literal repeated at each emit site.  A `specialize`d kernel (§2.8 move 8) derives this per
# instance from ρ and grows one region subtree per role — θ carries `rho` for that, but the
# decoder does not read it yet, so every instance is this single role.  See #165.
```

## #26 — `emit/copies.py` (was line 230)

```python
            # NOT `_peel_first_touch` here, unlike the reads.  Peeling this nest into per-value
            # sub-bodies leaves the guarded-off level as an EMPTY loop under the later sub-bodies
            # and reorders the surviving copy after it — measured: `M_split = 1:` came out as an
            # empty `for N_split` followed by A's copy.  The reads survive peeling because their
            # sub-bodies always retain the wmma; a copy nest can peel to nothing.  The explicit
            # `Cond` costs one compile-time-resolvable guard and keeps the traversal literal.
```

## #27 — `emit/copies.py` (was line 201)

```python
            # levels come from the ENUMERATION axes: every axis a movement is enumerated by must
            # exist as an enclosing loop.  NOT from `spans` — a fused movement is present on its
            # peer's axis without being enumerated by it, and making that a level for it emits the
            # cross product.
```

## #28 — `emit/copies.py` (was line 171)

```python
            # PROLOGUE copies fill EMPTY (not-yet-read) buffers → no vacating read → no WAR.
            # STEADY/DRAIN copies refill a slot the current reduction chunk was read from → WAR needed.
            # DEGENERATE: each step reads what it copied, so step `t` overwrites a buffer only when
            # an EARLIER step of this same arm used that slot — i.e. once the arm wraps the ring at
            # `t >= d`.  Derived from the ring, not assumed either way: claiming no WAR would drop a
            # real obligation when `M > d` (the 1LDSBuffer shape), and claiming one always would
            # ask the ledger to discharge against a vacating read that does not exist.
```

## #29 — `emit/copies.py` (was line 105)

```python
        # ONE movement instance = the region-`region` group of this fused set (AB0 = {A0,B0}).
        # A region-split copy already emits SEPARATE Insts per region (A0, A1 are distinct
        # movements, not sub-transfers of one), so `n_parts` here is NOT the region count — it
        # is the intra-tile transfer count of THIS one movement (1; we don't model sub-tile
        # transfer fan-out on the copy).  `part`/`coord` carry which region this movement is.
```

## #30 — `emit/copies.py` (was line 80)

```python
        # A Φ-fused movement's obligations are keyed by the MOVEMENT, because one cooperative
        # instruction owes one thing (ledger).  Every member consumes that same object, so a
        # member's consume site resolves to its own per-op-class obligations PLUS the movement's.
        # `movement_name` returns the bare name when there is no fuse, and the second lookup is
        # then the same key — hence the guard, so an unfused kernel cannot double-count.
```

## #31 — `emit/copies.py` (was line 28)

```python
# The issuing AGENT for every instance in a single-role kernel (§5.2: ρ → `Inst.a` and
# `Loop.role`; "an unspecialized kernel is the single-role special case").  ONE name instead of a
# literal repeated at each emit site.  A `specialize`d kernel (§2.8 move 8) derives this per
# instance from ρ and grows one region subtree per role — θ carries `rho` for that, but the
# decoder does not read it yet, so every instance is this single role.  See #165.
```

## #32 — `emit/helpers.py` (was line 307)

```python
    # "reads" = every INPUT operand that ends in a register fragment (two-hop shared→reg AND
    # one-hop DTV global→reg); each is emitted at its presence level.  A DTV operand has no
    # separate copy (its single hop lands in registers) and awaits on the VMEM (load) counter.
    # The OUTPUT (accumulator) op-class has no input trajectory in the mainloop → not a read.
```

## #33 — `emit/helpers.py` (was line 25)

```python
# The issuing AGENT for every instance in a single-role kernel (§5.2: ρ → `Inst.a` and
# `Loop.role`; "an unspecialized kernel is the single-role special case").  ONE name instead of a
# literal repeated at each emit site.  A `specialize`d kernel (§2.8 move 8) derives this per
# instance from ρ and grows one region subtree per role — θ carries `rho` for that, but the
# decoder does not read it yet, so every instance is this single role.  See #165.
```

## #34 — `emit/nest.py` (was line 121)

```python
                # HOISTED ABOVE THE WHOLE INNER NEST, so the grouping mode is not a loop variable
                # here and `_group_guard`'s predicate would reference a name that is not in scope.
                # Such a read absorbs the fan into `covers` instead of enumerating it, so a
                # group-restricted one would have to absorb only ITS OWN half — a `covers` change,
                # not a guard.  No configuration reaches this today (a split operand's read level
                # is its grouping mode or inner to it); refuse per-kernel rather than emit a read
                # that silently fills the other group's registers (#118).
```

## #35 — `emit/nest.py` (was line 91)

```python
            # first-touch: wrap in `mode == <anchor>` for each enclosing invariant mode (outermost
            # first, so the outermost guard is the outermost node).  The constant is 0 — FIRST
            # touch — unless the read-ahead's refill must follow the last consumer of the name it
            # overwrites, in which case `_read_anchor` supplies that mode's later value and the
            # same single read fires at the other end of the axis (#331).
```

## #36 — `emit/nest.py` (was line 75)

```python
            # NLL suppression (drain last steps): a read that VARIES over the reduction substep
            # carries the read-ahead-shifted coord `(k+dr)%n_s`; on the AHEAD substeps (k ≥ n_s−dr)
            # that read crosses into the OUT-OF-BOUNDS next reduction chunk.  Guard it by
            # `Cond(Pred(reduction_mode < n_s − dr))`.  The reduction Loop is ALWAYS an enclosing
            # level of such a read (its presence includes the reduction mode), so the guard var is
            # in scope wherever the read is placed — no dependence on THIS level being the reduction.
```

## #37 — `emit/nest.py` (was line 50)

```python
        # REGISTER PLACEMENT the wmma CONSUMES, PER READ OPERAND — the name-determining tile coord
        # + rotation the source vgpr carries (§2.6).  The wmma consumes the UN-shifted rotation
        # slot (`_read_placement` with dr=0): the value a PRIOR iteration's read-ahead deposited.
        # Carrying it in LoopIR (not `None`) is what lets GIR lower coord→name for the wmma source
        # identically to the ds_read dest, so def and use name the same vgpr under ANY loop order
        # (the read carries the read-ahead-SHIFTED slot; the wmma the consume slot — both derived
        # from the SAME `_read_placement`, so they cannot drift).  Keyed by operand name.
```

## #38 — `emit/nest.py` (was line 27)

```python
# The issuing AGENT for every instance in a single-role kernel (§5.2: ρ → `Inst.a` and
# `Loop.role`; "an unspecialized kernel is the single-role special case").  ONE name instead of a
# literal repeated at each emit site.  A `specialize`d kernel (§2.8 move 8) derives this per
# instance from ρ and grows one region subtree per role — θ carries `rho` for that, but the
# decoder does not read it yet, so every instance is this single role.  See #165.
```

## #39 — `emit/reads.py` (was line 166)

```python
            # `preloaded_tiles` derives the per-group `dr_g` itself (§5.3.1 pt5: prologue depth
            # = steady advance = dr_g), so an INPLACE group fills NOTHING here and its steady
            # advance is 0 — one shape, consistently.  `build_ledger` reads the same function for
            # its `readahead-residency` obligations, so emit and ledger cannot disagree; the older
            # "fill at the global dr while the steady runs at 0" was the PREFETCH-prologue /
            # INPLACE-steady mismatch the paper names (primed, then never re-primed).
```

## #40 — `emit/reads.py` (was line 147)

```python
        # PROLOGUE fill: the registers are EMPTY (nothing has read them yet), so there is no
        # vacating read and the rotation/in-place WAR does not apply — the same reasoning
        # `_copy_inst(war=False)` already applies to the prologue ramp copies.  Keep only the
        # RAW residency (wait for the shared buffer this fill sources).
```

## #41 — `emit/reads.py` (was line 127)

```python
                    # ONE READ, ONE POSITION.  Two names wanting different values of the same
                    # invariant mode cannot be served by one guard; `reload_positions` already
                    # returns None for that case so the depth falls back to 0 and we never get
                    # here.  Kept as an assert because the alternative — picking one value — would
                    # silently under-defer the other name and clobber it.
```

## #42 — `emit/reads.py` (was line 99)

```python
        # A PARTIALLY FOLDED axis is at or inner to the read level (it is still in presence), so it
        # is not in the slice above; it is added here with its modulus.  `read_fold` is the SAME
        # derivation `_shifted_coord`'s leader scale reads (`placement.read_fold`) — the guard that
        # picks WHICH act issues and the arithmetic that says WHERE it lands must not be two
        # separate readings of the fold.
```

## #43 — `emit/reads.py` (was line 90)

```python
        # A CHUNK-LEVEL READ HAS NO ENCLOSING INNER MODE.  When the quantum empties presence,
        # `op_class_level` puts the read at the reduction chunk, which is OUTER to every inner
        # mode — so the "enclosing inner modes it is invariant over" set is empty by construction,
        # and there is no first-touch guard to synthesize.  Without this the lookup KeyErrors on a
        # name that is not in `pos` (which indexes inner modes only).
```

## #44 — `emit/reads.py` (was line 64)

```python
        # AWAITS from the ledger (step 9b): the read's consume site is (op, "read") — its RAW
        # residency (waits the producer that filled the buffer it reads: two-hop → the global→shared
        # copy on tensorcnt; one-hop DTV → the global→register load) PLUS any register in-place
        # refill WAR (X:g0:refill).  kiter_note suffixes the RAW dep with the generation consumed.
```

## #45 — `emit/reads.py` (was line 27)

```python
# The issuing AGENT for every instance in a single-role kernel (§5.2: ρ → `Inst.a` and
# `Loop.role`; "an unspecialized kernel is the single-role special case").  ONE name instead of a
# literal repeated at each emit site.  A `specialize`d kernel (§2.8 move 8) derives this per
# instance from ρ and grows one region subtree per role — θ carries `rho` for that, but the
# decoder does not read it yet, so every instance is this single role.  See #165.
```

## #46 — `fuse.py` (was line 43)

```python
    # 7 (`mix31`) — DISCRIMINATOR.  A TWO-member group with MIXED shares [3,1].  Fuse 6 ([1,1,1,1])
    # and fuse 1/5 (all-2) both PASS, so uniform shares work at either extreme and per-member
    # arithmetic is correct at any single count; `A_MX`/`B_MX`'s [2,1,1] is the only NON-UNIFORM
    # partition left and the only failing one.  This point is non-uniform but only two members, so
    # it separates "mixed shares" from "3-member group" — the last pair of confounded properties.
```

## #47 — `fuse.py` (was line 36)

```python
    # 6 (`all4`) — DISCRIMINATOR, not a shipping grouping.  Every operand in ONE group at one wave
    # each, so it is the only point with a 3+-member group and NO unfused operand.  `A_MX`/`B_MX`
    # (2/3) fail on hardware while every passing grouping has either all-2-member groups or nothing
    # unfused; those two properties are confounded in 2/3 and this point separates them.  If `all4`
    # PASSES, the trigger is "big group alongside an unfused operand"; if it FAILS, it is the 3+
    # member group itself.  Either answer halves the search that eight falsified hypotheses did not.
```

## #48 — `geometry.py` (was line 638)

```python
        # the ACCUMULATOR: one live value per output tile, held across the WHOLE reduction — no
        # rotating ring (§6.1), no read-ahead residency.  Footprint = output free-tiles × frag_regs
        # (§7 "plus the accumulator"), ORDER-INVARIANT (it is the C tile, not a loop-order-sensitive
        # resident window).  Not routed through resident_free_tiles (which is ord-sensitive).
```

## #49 — `geometry.py` (was line 563)

```python
        # No shared placement — a one-hop global->register (DTV) operand (§5.3.1).  There is no
        # LDS ring to have a depth, so any number here would be a fiction.  Every caller already
        # guards on the operand touching SHARED (`build_S`, `_read_placement.src_slot`,
        # `loopir_to_gir`'s `touches_shared`, `emit.copy_insts`); this makes reaching it an error
        # instead of a silently meaningless answer.
```

## #50 — `geometry.py` (was line 472)

```python
        # `off` IS IN REDUCTION SUBSTEPS, THE NAME'S TRAVERSAL IS IN POSITIONS.  Within one name a
        # substep occupies `stride` consecutive positions — the broadcast fan the value is held
        # across (2 under `K·M·N` at an n-fan of 2; 1 under `M·N·K`, where the substep is
        # innermost and every position is a new value).  Applying `off` as raw positions under-counts
        # the lead-in by that factor: measured at `K·M·N`/`dr=2`/`extent(K)=4` it gave a peak of 2
        # where the ring holds 3.  Same shift `_readahead_shift` applies as `dr · stride_ord`.
```

## #51 — `geometry.py` (was line 378)

```python
        # PARTIALLY FOLDED: the axis KEEPS `extent // q` presence points, so the fan contributes
        # that many generations, not `extent`.  `presence_modes` reports the axis as still present
        # (it is — the group index varies), which is why the full-absorption test above does not
        # catch this case and the fold has to be divided out explicitly.
```

## #52 — `geometry.py` (was line 189)

```python
    # The region axis enters the hop's presence only when the tile is ACTUALLY SPLIT.  A tile with
    # a region MODE but `split == 1` (the MX scale operands: `region_modes=('M_split',)`, one
    # region) still moves in ONE instruction, so the hop varies over nothing — having an axis is
    # not the same as having more than one value along it.  Dropping this test made the per-region
    # completion note fire for a single-region operand, which is a real behaviour change and the
    # reason both old expressions are cross-checked against this one in the tests.
```

## #53 — `geometry.py` (was line 182)

```python
        # §2.2: the axes ONE instruction SPANS are absent from the hop's presence, exactly as a
        # bulk copy's inner axes are.  The two halves of this function are the same rule at two
        # widths — a `tensor_load_to_lds` spans every inner axis, a wide `ds_load` spans the ones
        # its quantum covers — which is why the quantum belongs HERE and not in a fold rule at the
        # leaf.  Absorbing it is what makes the merged coordinates ONE presence point, hence one
        # step and one generation, so a merged load cannot straddle a generation BY CONSTRUCTION.
```

## #54 — `geometry.py` (was line 107)

```python
        # DISTRIBUTED (§2.2): the Φ tiles are spread over `rho` sub-agents, so `rho` of them share
        # ONE register and the lane decides which — a BROADCAST, expressed by collapsing the slot
        # rather than flagged.  At TileSpan (Φ=2, ρ=2) every tile gets slot 0: one register holds
        # block 2g and 2g+1, which is exactly LRA's half-wave layout and what the WMMA selector
        # reads.  `|{slot}| = Φ/ρ` is then the register count per instruction, so `regs` sees the
        # broadcast without being told about it.
```

## #55 — `geometry.py` (was line 67)

```python
    # An AGENT-SERVED region axis is broadcast for every data consumer (#245): the wave reads,
    # computes and accumulates inside its own region.  One subtraction point, so the read coord,
    # free_modes, register naming, the #288 wmma projection and the footprint cannot disagree.
    # The BULK COPY is untouched -- `hop_broadcast`'s bulk branch never calls here.
```

## #56 — `ir.py` (was line 474)

```python
                             # runtime-trip, peeled level (the shared-ring / reduction axis); False
                             # = a statically-unrolled inner mode.  NOT inferred from trip being
                             # symbolic-vs-int (that conflates "which ring/peel" with "what trip
                             # value").  Derived from the Mode being in theta.outer_modes().
```

## #57 — `ir.py` (was line 465)

```python
                             # display extent (§5.2 line 339, Q19 ADOPTED).  For an `outer`
                             # (pipelined, runtime) loop `trip` is the continuation PREDICATE the
                             # backend lowers to a back-edge (e.g. "K//DU - M > 0"); for an inner
                             # unrolled level it is the static repeat COUNT (an int).  §4.6's "every
                             # control edge is carried" made literal for the steady back-edge.
```

## #58 — `ir.py` (was line 366)

```python
        # `keep_unresolved` is OPT-IN and OFF for the codegen path, which must always get a
        # concrete Placement.  A RENDERER, though, can be handed an env that does not close the
        # expression -- the drain body leaves `iter` unbound on purpose, because `T` is runtime --
        # and `Expr.eval` answers 0 for an unbound name (see `eval`), so pinning there prints a
        # buffer index the schedule never chose, indistinguishable from one it did.  Per-EXPR, not
        # per-Placement: a drain read's register slot IS resolvable while its LDS source slot is
        # not, and blanking both would hide a number that is real.
```

## #59 — `ledger.py` (was line 430)

```python
    # (c) READ-AHEAD RESIDENCY (Lemma 3d): each `readahead-residency` obligation requires its
    # coord (a FIELD on the producer endpoint) to be actually emitted as a register read in
    # the PROLOGUE body.  A prologue missing a required read (a wrong K-axis peel) leaves O_r
    # undischarged → the obligation is VIOLATED here, at build time.  Collect the prologue's
    # emitted register-read coords, then check each obligation's ckey is present.
```

## #60 — `ledger.py` (was line 419)

```python
                            # A REFILL CONSUMES THE MARK (#262).  The wmma that licensed THIS
                            # refill cannot also license the NEXT one — the slot now holds the
                            # generation this refill just wrote, and a second refill in the same
                            # body must be preceded by its own consumer.  Without the discard only
                            # the FIRST refill of a slot per body was checked and every later one
                            # passed for free.
```

## #61 — `ledger.py` (was line 363)

```python
    # (b3) order, CROSSING RAW: the copy must be emitted BEFORE this operand's reads in the steady
    # body, because those reads consume the chunk it fills (§5.3.1 pt5).  This is the mirror of (b):
    # the `S=δ` refill must come last, the crossing copy must come first, and an operand cannot be
    # both (that is exactly the `S_shared == r` case where the coupling demands `off >= r+1`).
```

## #62 — `ledger.py` (was line 332)

```python
    # (b) order: per region, every WAR-refill copy must be preceded by the vacating read.
    # ONLY the `S=δ` in-place refill is order-constrained to follow its reads (§5.3.1 point 4 is
    # scoped to "an S=delta refill copy").  A `rotation-WAR` copy overwrites a buffer vacated in an
    # EARLIER trip, so it is free to be emitted first — and MUST be, when this trip's chunk-crossing
    # reads consume it (the crossing-RAW below).  Matching `.endswith("WAR")` here forced copy-last
    # universally, which is the ordering that makes the crossing read consume an unwritten buffer.
```

## #63 — `ledger.py` (was line 255)

```python
        # `inline_outer` = descend the OUTER-level Branch that wraps a DRAIN step (the peel's `Bind`
        # binds each drain step's reduction chunk residue via a Branch(outer=True)).  When flattening a drain
        # step body we must inline that branch, else the whole drain trip comes back EMPTY and the
        # per-edge order gate never inspects a single drain instruction (the σ_c check for a drain
        # refill would be silently vacuous).  For steady/prologue bodies (inline_outer=False) an
        # OUTER structure is a SEPARATE trip and stays a boundary, handled by `visit`.  The trip
        # boundary is tested STRUCTURALLY by `.outer`, not by the level's name.
```

## #64 — `ledger.py` (was line 147)

```python
    # DETERMINISTIC ORDER.  The set is only for dedup; returning it directly made the emitted
    # `Await` order vary run to run (set iteration follows string hashes, which PYTHONHASHSEED
    # randomizes), so the IR dump could not be diffed between runs — the one thing an
    # observability artifact must support.  σ_c itself was unaffected (instruction order is fixed
    # by build_ir), but `_site_awaits` iterates this, so the discharge points moved.
```

## #65 — `ledger.py` (was line 130)

```python
    # READ-AHEAD RESIDENCY `O_r` (Lemma 3d, §5.3.1 pt5): each first-trip register-read the steady
    # read-ahead CANNOT discharge in-body must be pre-issued in the prologue, else its residency
    # `complete(read) ⤳ issue(wmma)` is unmet → NON-EMPTY ledger (a wrong K-axis peel fails HERE at
    # build time, not by a passing/failing run).  One obligation per required prologue read; the
    # consumer key carries the required coord so the gate verifies the prologue actually emits it.
```

## #66 — `placement.py` (was line 1376)

```python
        # READ-AHEAD SLOT = the rate index OF THE SHIFTED COORD, not the unshifted slot plus `dr`.
        # `_rate_slot` is `rate_index(coord) mod W`, so shifting the coord shifts the slot; deriving
        # it from the same shifted position is what keeps the producing read and the consuming wmma
        # (which asks for `dr=0` at its own coord) naming ONE vgpr under every order.  Adding `dr`
        # to the unshifted slot only coincides with that when the reduction mode is outermost in the
        # presence traversal (K-outer); elsewhere it names a register the wmma never reads.
```

## #67 — `placement.py` (was line 1254)

```python
    # INNER = the ord modes strictly INNER to the reduction substep — a property of `ord`, not of
    # this operand (the paper splits the whole order, then intersects each factor with presence).
    # THE SAME SPLIT POINT THE PEEL USES (`_substep_mode`), so the prologue and the steady advance
    # cannot disagree about where `ord` divides — they are two halves of one rule, and Lemma 3d
    # only holds if the steady sustains exactly what the prologue primed.  The conversion itself is
    # `prefetch_distance_for`, shared with `prefetch_steps_for`'s admission walk so the depth that is
    # ADMITTED and the shift that is EMITTED are the same quantity.
```

## #68 — `placement.py` (was line 1243)

```python
    # THE SAME LEVEL `prefetch_distance_for` ADVANCES ALONG (#321) — this guard and that conversion are
    # two halves of one rule, so they must test the same axis.  They did not: this kept
    # `_reduction_rate_mode` while the conversion moved to the read-ahead level, and the guard's
    # `red is None` short-circuit then discarded a shift the conversion was computing correctly.
    # MEASURED at `DepthU=64` (no reduction inner mode exists): `prefetch_distance_for` returned 1 and
    # this returned 0, leaving `PLR=0` and `PLR=1` byte-identical for a second time.
```

## #69 — `placement.py` (was line 1235)

```python
    # PRESENCE POINTS, NOT TILES (§2.2 / §5.3.1).  A quantum-folded axis contributes `N/q` group
    # indices to this traversal, so the radix, `n_pres` and the drain-suppression bound `P + shift
    # >= n_pres` all count carrier groups.  The fold is applied once, at presence (`hop_extents` →
    # `theta.presence_modes(op, hop)` → `geometry.hop_fold`), so the advance and its bound inherit
    # it here with no tile-vs-point conversion of their own — which is precisely why there is no
    # second rule in this function for the merged case.
```

## #70 — `placement.py` (was line 1207)

```python
    # THE FIXPOINT.  `at` is the guard the emitter will actually put on this read
    # (`emit._read_anchor` builds the same dict from this return value).  Score the schedule that
    # guard produces; if the refill still lands on a live name there, the move is not a rescue and
    # the depth must be refused — `dr_g` falls back and `prefetch_refusal` reports it, which is
    # the honest answer.  Anything else is admitting a schedule no layer below can fix.
```

## #71 — `placement.py` (was line 1160)

```python
    # STARVATION IS THE AUTHORITY (#332).  `_uses_before_load` is the complete correctness test — every
    # steady consume finds the generation it wants — and it strictly contains the premature-write
    # case below, which cannot starve a read without also leaving one stale.  It is checked FIRST so
    # that the one predicate decides `clobber`, and the conflict scan is left with the only job it
    # is uniquely good at: telling a rescuable same-leaf WAR (`inplace`) apart from a clean schedule.
```

## #72 — `placement.py` (was line 981)

```python
    # AN EXTENT-1 MODE IS NULLSPACE, the same rule `_rate_modes` states: it has one value, so the
    # traversal neither repeats under it nor advances through it, and it cannot make an outer mode
    # "not outer".  Keeping it here is not conservative, it is WRONG — measured `MNK` + mixed
    # MT/DU split, A: `reload = [M_inner(1), K_splitB(2), K_inner(1)]`, and the degenerate
    # `M_inner` sits OUTSIDE `N_inner` in `ord`, so it hid the one mode the whole predicate is
    # looking for and A silently fell back to a substep advance.
```

## #73 — `placement.py` (was line 946)

```python
    # THE RELOAD TRAVERSAL, NOT THE PRESENCE SET — the same set `_readahead_shift` walks.  A
    # sibling-enumerating axis is IN `pres(p)` but is NOT a mode the shift advances through (§2.6:
    # it selects a different group's band), so `reload_modes` drops it and the reduction mode's
    # stride changes accordingly.  Using `_presence` here instead cost 4 split-config regressions
    # (`KMN split=[2,2,1,1]`: `USE-BEFORE-DEF ... reads A buffer X1 tile 0`) — the shift ADMITTED
    # and the shift EMITTED must come from one traversal, which is the whole point of extracting
    # this function.
```

## #74 — `placement.py` (was line 619)

```python
    # The split is live but its mode is gone: the free tiles moved to another axis.  Name the
    # OUTERMOST of the operand's own free modes present in `ord` -- outermost because that is the
    # position `_sibling_outer_to_reduction` compares, and because the region axis it replaces was
    # itself outer to the in-region fan.
```

## #75 — `placement.py` (was line 566)

```python
    # ONE DERIVATION, asked of the read hop.  `geometry.hop_broadcast` already answers "what does
    # ONE INSTANCE of this hop not vary over", and since 2026-08-20 that includes the axes the
    # instruction SPANS (`geometry.quantum_axes`).  Re-deriving the subtraction here would be the
    # second expression of the same rule, which is how this and `hop_broadcast` drifted before.
```

## #76 — `placement.py` (was line 521)

```python
    # single-group: one depth for the whole operand, still DERIVED (a single-group fragment can be
    # Shape B too — that is exactly the §2.6 term-(ii) operand of a K-innermost order, which has no
    # width split at all.  Reading "per-group" as "only when width-split" is what kept the fill
    # running at the global `dr` for a group whose steady advance was already 0.)
```

## #77 — `placement.py` (was line 477)

```python
        # LEMMA 3d's OUTER FACTOR IS A PRESENCE POINT, NOT A TILE (§5.3.1).  "the first presence
        # point is the whole first carrier group, all `q` of its `ρ`/`Φ` members, not tile 0 alone.
        # Hoisting only tile 0 of a `q`-fold would leave the group's other members with no leader in
        # the prologue and orphan them in the drain — an incomplete carrier group."
```

## #78 — `placement.py` (was line 317)

```python
    # THE REACH IS DERIVED, NOT REQUESTED.  `readahead_reach` runs the §2.6 clamp per group before
    # converting to chunks; the old form here was `ceil(dr / n_substeps)` on the RAW request, which
    # grew `M` around a read-ahead the steady body never issues (see `readahead_reach`).  `dr` stays
    # the REQUEST because its other consumers (`preloaded_tiles`, `_readahead_shift`) each derive
    # `dr_g` from it themselves — only the CHUNK count must be pre-derived.
```

## #79 — `placement.py` (was line 282)

```python
        # THE SENSE IS PER OP-CLASS, THE ROLE IS PER HOP (#212 axis 3).  Everything below reads
        # `hop.role`, which is keyed on (src, dst) alone and therefore cannot tell a forward READ
        # from a store round trip's read-BACK.  A reverse-trajectory op-class needs Lemma 1's
        # REFLECTED peel (hoist FORWARD, prologue and drain swapped), which is not emitted (#116) —
        # so refuse here rather than silently building it the forward way, which is what keying on
        # the role alone would do and what makes this a latent defect instead of a missing feature.
```

## #80 — `placement.py` (was line 157)

```python
            # §5.3.1 pt5 (rewritten): the coupling is `off(copy,iter) >= r`; the `+1` is demanded
            # ONLY in the `S=δ`-tight copy-last case, where copy-first would overwrite a slot the
            # current trip still reads.  It is "a σ_c placement fact, not a fixed inflation of
            # off" — an earlier version of this check used `r + 1` unconditionally and would have
            # rejected the shipping `S_shared=2, off(copy)=1, r=1` point, which the paper names as
            # emittable and correct.
```

## #81 — `render.py` (was line 557)

```python
        # THE REGION COUNT, not just the axes.  Without it `MXSA region axes ('M_split',)` reads
        # as a split operand while `MXSA.split == 1` -- ONE storage region -- and the GIR dump,
        # which gates on `operand_regions > 1`, correctly omits MXSA entirely.  Two dumps of one θ
        # disagreeing about whether an operand is region-split is worse than either being silent.
```

## #82 — `render.py` (was line 498)

```python
    # THE DOWNGRADE, NAMED.  `readmode` prints `dr_g` and this line says whether that `dr_g` is the
    # one that was ASKED FOR.  PLR is an input parameter and `W` is derived, so a `dr_g` below `dr`
    # is the derivation overriding the request — never a silent outcome.  This block used to be a
    # caveat in the line above ("a printed dr_g of 0 does not by itself say which one fired");
    # `prefetch_refusal` answers it instead of warning that it is unanswered.
```

## #83 — `render.py` (was line 382)

```python
                # honor a RESOLVED guard in this concrete env (first-touch `mode==0`, drain NLL
                # `k < n_s−dr`) — the unrolled stream shows only the arm that actually executes.  An
                # UNRESOLVED guard (peel-validity `T ≥ M`, T runtime) shows the header + the
                # representative (then) long-loop path.
```

## #84 — `render.py` (was line 185)

```python
    # THE OPERAND NAMES COME OFF THE NODE.  This line hardcoded `C[...] += A*B`, so `Mma.a`,
    # `Mma.b` and `Mma.acc` were unverifiable from every view except --raw — and a decoder that
    # named the wrong operand rendered identically to one that named the right one.  The literals
    # also assume a binary A/B matmul, which is exactly the assumption the MX work had to remove.
```

## #85 — `render.py` (was line 156)

```python
        # a copy's COORD must be shown, not just its op-class name.  A storage-region split makes
        # ONE op-class emit one movement instance per region value (§2.8 move 9: an instance is
        # (op-class, hop-coordinate)) — the region is a coordinate, exactly like a read's
        # (tile, substep).  Print it in the same `name[coord]` form the register-dest branch uses,
        # or sibling region movements render identically and the dump cannot be read.
```

## #86 — `schedule.py` (was line 135)

```python
    # SIZE FOR THE REQUESTED DEPTH — `PLR` IS AN INPUT, `W` IS DERIVED.  The user asks for a
    # read-ahead depth; the ring width is ours to choose.  Capping `dr` at `W-1` instead silently
    # DOWNGRADES the parameter (measured: `PLR=2` emitted `dr=1` at `W=2`), which inverts the
    # relationship — a derived quantity overruling an input.  So the width is the narrowest one in
    # the band that honours the request on BOTH constraints: Lemma 3b's capacity `W >= dr+1`, and
    # the rotation map's assignment (`register_reuse_verdict`).  If the band cannot honour it the
    # answer is `lo` and the depth is refused downstream — a visible shortfall, not a silent one.
```

## #87 — `schedule.py` (was line 128)

```python
    # THE REQUEST IS `off(READ, substep)` — WHAT PLR SET — not `prefetch_depth`, which returns
    # `S_shared`, the LDS ring size (a DIFFERENT quantity; its own docstring says so).  Asking the
    # wrong one inflates the request (2 where PLR=1) and widens the ring for a depth nobody will
    # ever request.  This is the same currency `ledger.py` uses for the very same question.
```

## #88 — `sigma_c.py` (was line 119)

```python
        # A copy whose chunk THIS trip's read-ahead crosses into (`off == r`, §5.3.1 pt5) is a RAW
        # producer for those reads and must LEAD them — never deferred.  Every other shared copy
        # keeps the §2.4 refill-after-read freeze: at `off != r` its slot lands on a chunk read
        # EARLIER in the trip, so hoisting it would overwrite data still being consumed.
```

## #89 — `theta.py` (was line 595)

```python
# The reserved group label under which an operand's SHARED (LDS) ring depth is stored, alongside
# its REGISTER rate-groups, in the one `DepthMap` (§2.6: `S` is a map over a placement's rotating
# modes, and the two placements — shared reduction-chunk ring and register rate-group — are different rows of
# the same map).  Callers that mean "a register group" must exclude it.
```

## #90 — `theta.py` (was line 543)

```python
        # AN AGENT-SERVED AXIS IS NOT A REDUCTION.  `inner \\ pres(output)` is a SUBTRACTION, so
        # anything the output stops varying over falls in here by default -- including an axis that
        # left the output's presence because the AGENTS traverse it, not because it is contracted.
        # Subtracting it makes the classification three-way at its single definition point.
```

## #91 — `theta.py` (was line 499)

```python
    # --- presence & reduction (§2.1, PAPER-CONFIRMED Q18) -------------------
    # ---- presence family --------------------------------------------------
    # `Mode` is CANONICAL here, matching the level/ord family (`inner_modes()` etc.); the
    # `*_names()` variants below are thin helpers for the call sites that genuinely want strings
    # (set algebra, and the GIR `Program.meta` name lists that cross the layer boundary).  These
    # used to be string-only, with no Mode-returning form at all — which forced callers that
    # needed an extent to look the Mode back up by name (two separate linear scanners existed for
    # exactly that).  `Mode` is a frozen dataclass, so returning it costs nothing.
```

## #92 — `theta.py` (was line 398)

```python
    # (no `reduction_modes` field: the contraction axes are DERIVED by `reduction_names()` from the
    #  accumulator's presence, §2.1 line 82.  A stored copy existed, was never written and never
    #  read, and its docstring claimed "Set by translate" — a second authority that could disagree
    #  with the derivation.  Ask `reduction_names()`.)
```

## #93 — `theta.py` (was line 387)

```python
    # How many agents cooperate on one tile.  1 = the unspecialized single-agent kernel.  This is
    # what decides whether a shared buffer's hazards are cross-agent (§2.4): a byte written by one
    # agent and read by another has no program-order edge between the two, and in a GEMM every
    # agent reads the whole staged tile — so any shared buffer touched by 2+ agents is cross-agent.
```

## #94 — `theta.py` (was line 258)

```python
#: A resort entry's HOP ROLE.  `read` is the classification half — which modes the agents traverse
#: INSTEAD of the coordinate, so the read/compute nest must not enumerate them.  `copy` is the
#: MOVEMENT half — how many agents cooperate on one op-class's global→shared copy, i.e. the Φ
#: group's wave shares.  They are different questions about the same mode and must not be mixed:
#: everything that existed before this distinction asks the `read` one, so `read` is the default
#: everywhere and `copy` is opt-in.
```

## #95 — `theta.py` (was line 211)

```python
    #: (`copy_agents` WAS HERE.  How many agents cooperate on this op-class's copy is a ρ fact and
    #: now lives there -- `Rho.copy_agents(name)`, beside the read-side partition it is the sibling
    #: of.  As a per-operand scalar it was written by `translate` and READ BY NOTHING for its whole
    #: life; the under-serving its docstring described was measured false once `tdmIssuesOwnLoad`
    #: made a Φ group issue ONE load by its owner, and its cited numbers predate that fix.  Two
    #: shapes of one agent assignment is the two-derivations-one-rule trap this model keeps paying
    #: for, so there is now exactly one.)
```

## #96 — `theta.py` (was line 128)

```python
# ---------------------------------------------------------------------------
# TRAJECTORY SENSE — per OP-CLASS, and NOT the same thing as the hop role (#212 axis 3).
# ---------------------------------------------------------------------------
# `HOP_ROLE` is keyed on (src, dst) alone, so it answers a question about ONE LEG.  See ADR 0006.
```

## #97 — `theta.py` (was line 110)

```python
# ---------------------------------------------------------------------------
# The OP-CLASS role vocabulary (§2.4 `off : (op-class × level) → ℤ`).
# ---------------------------------------------------------------------------
# An op-class is ONE operand's movement along ONE hop, so `off`'s row index is the pair.  See ADR 0006.
```

## #98 — `validate.py` (was line 303)

```python
    # --- Property 9: every short-loop step past the first is guarded by `T > t` ---------------
    # The `els` arm runs when T < M, so the short trip [0,T) is STRICTLY shorter than its M peeled
    # steps and T is runtime.  An unguarded step t>=1 copies a nonexistent chunk (out-of-bounds
    # global read) and accumulates garbage into C.  Step 0 needs no guard (`T > 0` is trivially
    # true wherever a reduction level exists).
```

## #99 — `validate.py` (was line 276)

```python
        # NAME THE OBLIGATIONS, not just the count.  A bare count says a schedule is unsound but
        # not which edge, so the only way to localize was to guess configs and re-run the build —
        # which cost a full round trip on the 2026-08-21 mxf8 PLR2 failure ("1 undischarged", no
        # kernel, no edge).  `check_ledger_discharged` already returns the Obligation objects; this
        # just renders them.  Cheap: the gate fires only on a defect, and when it fires the message
        # IS the debugging session.
```

## #100 — `validate.py` (was line 230)

```python
            # the refill copies are the σ_c-ordered tail: the LAST top-level nodes of the body
            # that are shared-dest Loads (copies).  Assert no read/wmma follows the first copy.
            # A REGION-SPLIT copy is rolled — one Inst under a `Loop(region_mode)` (§5.3.1
            # line 523) — so look through that loop, or P6 stops seeing split copies as the tail.
```

## #101 — `validate.py` (was line 201)

```python
    # P3 (the real assertion) + P5 (its concrete-slot half): NO placement may reference a mode that
    # no enclosing Loop/Bind/Branch binds.  Both properties reduce to this one check — see
    # `_unbound_placement_refs`.  Previously P3 only verified that a Bind PRESENT pinned the right
    # mode (so a bare Inst with a free-`iter` slot passed clean, which is the defect P3 exists to
    # catch) and P5 only checked that `slots` was non-empty.
```

## #102 — `validate.py` (was line 145)

```python
        # STRICT `T > M`.  The paper's §5.2 skeleton writes `>=` and its NB blesses either, but the
        # NB's reasoning ("at T = M ... its steady loop has trip count 0, a header that never
        # executes") presumes a PRE-TESTED header.  The decoder's steady region is emitted as a
        # body-with-trailing-test, and a post-test always runs at least once — so at T == M the
        # `>=` form would run one steady trip computing a chunk the drain also computes (#228).
        # The strict form is the decoder's contract with its own lowering, checked here so the two
        # cannot drift; it is also what TensileLite's `openLoop` compare does.
```

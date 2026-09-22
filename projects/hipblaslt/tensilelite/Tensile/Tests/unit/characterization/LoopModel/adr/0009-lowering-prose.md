# ADR 0009 -- the prose removed from Tensile/Lowering

Status: accepted.  Date: 2026-08-30.

`Lowering` was 49% prose (5532 of 11315 lines). Most of it was memo: how a bug was found,
what a measurement said on a given day, which hypothesis died. That is worth keeping and
worth not reading every time you open the file, so it is here, verbatim, and the code keeps
the statement of behaviour.

Rules applied: a module docstring keeps its leading paragraph; a comment run keeps its first
three lines. Everything cut is below, in source order, with its location.


## __init__.py:3 <module>

```

Tensile.Lowering -- the UseLoopModel lowering stack.

 loopir_to_gir layer 1 (LoopModel LoopIR) -> layer 2 (GIR); build_gir = build-once (R-ONCE)
 gir/ the GIR graph, analyses, passes, verifier (pure Python; no rocisa)
 gir_to_rocisa layer 2 -> layer 3 (arrives R1; leaf emitters move here from Components)

RAISE `RuntimeError` TO REFUSE A KERNEL. That is the whole failure policy, and it is not
cosmetic. `KernelWriterAssembly.getSourceFileString` wraps kernel generation in

    except RuntimeError as e:
        printWarning(f"Failed to generate assembly source code for {kernel}: {e}")
        code = ""; errcode = -2

so a RuntimeError fails ONE kernel, loudly, and the rest of the matrix still generates.
Anything else -- ValueError, a bare Exception, a custom class not derived from RuntimeError --
escapes that handler, propagates through the joblib worker, and aborts the entire run before
any other cell is attempted.

MEASURED 2026-08-24: the movement-quantum check raised ValueError, so a config failing on two
of nine loop orders killed the whole build and reported nothing about the other seven.
Bisecting it took nine separate --build-only runs. As a RuntimeError the same run would have
generated 7/9 and named both failures in one pass. A run-killer turns "which cells are
broken?" from one run into a linear search.

Reserve `ValueError` for a genuinely bad argument (a malformed node at construction) and
`NotImplementedError` for a path deliberately not built yet.

```

## asm_lane_eval.py:187 comment run

```
    # ---- float domain -----------------------------------------------------------------------
    # A VGPR holds 32 BITS; an f32 is those bits and an f64 is a register PAIR.  Modelled as real
    # IEEE bit patterns rather than a side table of Python floats, so a value written as float and
    # read back as integer (which the divide sequence does at both ends) behaves as the hardware
    # does.  `v_rcp_*` is a hardware APPROXIMATION; here it is the exact reciprocal, which is why
    # `check` re-runs the slice with a perturbed reciprocal and refuses if the addresses disagree.

```

## asm_lane_eval.py:112 read

```
One operand's value in `lane`.  An SGPR or literal is lane-invariant by construction.

        An operand the parser could not resolve is REFUSED here, not defaulted to zero: a zero
        address term is indistinguishable from a real one downstream, so it would turn a parse gap
        into a confident wrong answer.

        SO IS AN UNWRITTEN REGISTER, and that one is not hypothetical.  The first version of this
        class returned 0 for any VGPR the slice had not defined, on the reasoning that the slice is
        closed under its own producers.  It is not: a slice can bottom out at a kernel INPUT (a
        workgroup id in an SGPR, an address the host passed in), and every such input silently read
        as 0.  The checker then evaluated every `LocalReadAddrA` to exactly its immediate `offset:`,
        reported all lanes in region 0, and announced ZERO VIOLATIONS for four kernels -- a clean
        bill of health produced entirely by reading uninitialized state.  Refusing here converts
        that into a message naming the register, which the caller can then seed deliberately.
```

## asm_lane_eval.py:61 comment run

```
        # THE KERNARG IMAGE, `{byte offset: dword}` -- the values `s_load_b*` brings in.
        #
        # An address slice does not always bottom out at a register: TensileLite loads sizes and
        # strides from the kernel argument buffer, so the slice reaches `s_load_b512 s[36:51],
        # s[sgprKernArgAddress], 16`.  Those are INPUTS in exactly the sense the `read` docstring
        # means, so they get the same discipline: supplied deliberately by the caller, and a dword
        # the caller did not supply REFUSES rather than reading as zero.  Seeding the whole buffer
        # with zeros would reproduce the very failure that docstring records -- every address
        # collapsing to its immediate and a confident "no violations".

```

## asm_lane_eval.py:3 <module>

```

LaneEval -- execute a straight-line slice of emitted gfx1250 assembly, once per LANE.

WHY THIS EXISTS, AND WHY IT IS NOT A FORMULA.  The question it answers is whether the 32 lanes of
one `ds_load` all land in the SAME storage region of a `TDMSplit` tile.  That is a property of the
per-lane LDS address, and the per-lane address lives in a VGPR (`vgprLocalReadAddr*`) built at init
from `vgprSerial` -- the instruction itself carries only a wave-uniform `offset:` immediate.  So the
fact is invisible in the instruction stream, and it is invisible in GIR too: GIR reasons about the
region a READ NAMES, never about the lane spread inside one instruction.

The obvious alternative is to re-derive the address from `LraTileAssignment`'s rules in Python.
That was rejected: `LraTileAssignment` EMITS the arithmetic rather than computing it, so a Python
version is a TRANSCRIPTION, and a checker built on a transcription tests the transcription -- it
agrees with the kernel exactly when both are right and exactly when both are wrong the same way,
which is the failure mode this whole tool exists to rule out.  Interpreting the emitted
instructions has no such freedom: if the kernel's address arithmetic is wrong, the interpreter
reproduces the wrong address, and the region check sees it.

REFUSAL IS THE POINT.  Every opcode this class does not model raises `UnsupportedInstruction`
naming the instruction.  A silently-skipped instruction would leave a register holding a stale
value and the resulting address would look plausible and be wrong -- the same class of silent
miscompare the checker is meant to catch.  So the table is deliberately small and closed, and it
grows only when a real slice needs a member.

SCOPE.  Straight-line integer VALU/SALU only, over one wavefront's worth of lanes, with no control
flow: the address-setup slice of a Tensile kernel is straight-line by construction (it runs once
before the loop).  Divergence is modelled only as far as `v_cmp_*` writing a lane mask consumed by
`v_cndmask_b32`; branches are not executed, and a slice containing one is refused rather than
guessed.  Values are unsigned 32-bit and wrap like the hardware.

```

## __init__.py:3 <module>

```

GIR analyses -- one class per module, each an `Analysis` subclass.

 cfg successors (helper), Dominators, BackEdges (+ BackEdge / BackEdgeSet)
 gen_reaching GenReaching -- concrete generation each access uses 
 swap_regions SwapRegions -- PendingMark(swap) per generation change 
 dep_defuse DepDefuseAnalysis -- RAW/WAR edges carried from LoopIR awaits
 reg_band RegBandAnalysis -- VALIDATE register width W (never decide it, B3)
 gr_increment GrIncrementRegions -- PendingMark(gr_increment) per steady copy (placement only)
 gl2_prefetch Gl2PrefetchRegions -- PendingMark(gl2_prefetch) per steady chunk (#216); MIDPOINT,
 and the only Mark policy here that is a pure COST choice rather than a bound
 lds_hazards LdsHazards -- RAW/WAR/WAW on the LDS ring, with the TRIP DISTANCE (#204)
 short_path ShortPathFold -- may the `T < M` arm coverage into the shared prologue/drain? (#183)
 loop_shape LoopShape -- DERIVE pre/post-test + trip count; the covering check (#229)
 uniform_placement
 UniformPlacement -- the L733 LIVENESS check: a proc-scoped selector must be
 reached by EVERY agent of its scope (the one deadlock the safety gate is blind to)

```

## fence_regions.py:290 comment run

```
                        # THE TOKEN IDS THIS FENCE ORDERS (#235).  `buffers` names operands, which
                        # is the readable tag but NOT what the backend needs: a token is an LDS
                        # pseudo-register, and a fence gets each of its tokens as both a def and a
                        # use, so the ids here are exactly the storage it stands between.  Naming
                        # operands and letting the backend re-derive the ids is how a split tile's
                        # half-1 traffic ended up on tokens no fence mentioned (#217).

```

## fence_regions.py:197 _tokens_across

```
Every buffer LIVE ACROSS this fence -- not merely the ends of the edges that placed it.

 A barrier is a POINT, and its ordering power comes entirely from its token set:
 `StinkyBuildImplicitDependencyPass` gives the barrier each token as both a def and a use, so a
 buffer the barrier does not name is not chained through it at all -- its accesses may migrate
 across it, and the waitcnt strength is derived from the same set. Naming only the endpoints of
 the edges the cover happened to select therefore leaves every other buffer live at that point
 unordered, while the fence still LOOKS like it separates the phases.

 MEASURED, 2026-08-16. The old union-over-covered-edges form was propped up by an
 over-approximation elsewhere: a fused copy left its non-representative member's region axis
 unpinned, so B's tokens widened to both halves and padded these sets. Fixing that widening
 (correctly -- a fuse group pairs members by index) shrank a steady fence from `(1,5,7)` to `(1,)`
 and took multi-wave TN split from 42 to 62 failures, at BOTH ScheduleIterAlg values. The fence
 PLACEMENT never moved; only the names on it did. The widening had been doing this function's
 job by accident.

 THE RULE -- plain liveness over the CFG, which is what "crosses" means:

 live_across(point) = { b : b is touched on some path INTO the point }
 & { b : b is touched on some path OUT of the point }

 Both halves must span blocks. A fence at slot 0 of `drain0` has nothing before it in its own
 body, yet every buffer the steady loop filled reaches it; a within-block test would name none
 of them and emit a fence that orders nothing. So "into" is this block's prefix plus every
 block that can reach this one, and "out of" is the suffix plus every block reachable from here.

 A block ON A CYCLE is in its own reachable set, so its whole footprint lands on both sides and
 the loop body's fences name everything the body touches. That falls out of the definition
 rather than being a special case, and it is not conservatism: the back edge genuinely makes an
 access "before" the point in one trip an access "after" it in the previous.

 The OBLIGATIONS are unioned in on top. Liveness is a statement about the program text and
 misses a distance>=1 edge in an ACYCLIC block -- the multi-copy prologue holds WARs whose read
 sits textually AFTER the write it precedes, and the cover legitimately answers them with a slot
 at the top of the block, where nothing is textually "before". Those ends are named because the
 fence was placed FOR them, so the set is `live-across | ends-covered`: never smaller than the
 #205/#207 set it replaces, which is what makes this change monotone.
```

## fence_regions.py:133 comment run

```
    # THE TEST IS SEPARATION, and reachability+dominance is only its single-block approximation.
    # The old form was `x in reachable(pb) and x in doms(block)`.  `reachable` is EXISTENTIAL -- it
    # answers "SOME path from `pb` gets to `x`" -- so a candidate reachable only along a BACK EDGE
    # satisfies it while sitting BEFORE the producer on every forward path.  That contradicts this
    # function's own contract two paragraphs up ("`x` covers the edge only if it lies BETWEEN the
    # two ends ON EVERY PATH"), so the bug is visible without the model; (L731-735) is the
    # reason it matters, since our backend both discards markers and reorders, which is exactly the
    # combination for which the model says positional coverage must be validated per edge.
    #
    # The `x != pb` guard below was the one instance of this we had already found: a loop block is
    # reachable from ITSELF along its back edge.  That is not a special case -- it is the general
    # failure seen once.  With `loop_copies > 1` the same thing happens between DIFFERENT blocks:
    # `steady{n-1}` is reachable from `steady{k}` only around the back edge, so a fence there was
    # accepted as covering `steady{k} -> drain0` while lying before the producer on every real path.
    #
    # So ask the question directly: DELETE `x` FROM THE GRAPH -- is the consumer still reachable from
    # the producer?  If not, every producer->consumer path is forced through `x` and the fence
    # genuinely covers.  This subsumes reachability, dominance and the `x != pb` guard in one
    # predicate, and needs no dominator tree.
    #
    # MEASURED 2026-08-21 (round-2 model audit): the old form left 248 unfenced producer->consumer
    # paths in 14 of 36 swept configs, every one `steady{n-1}:copy -> drain0:read`, all of them at
    # `loop_copies > 1`.  The separator test drives 248 -> 0 for +1 fence in 16 configs, and every
    # `loop_copies == 1` config -- i.e. everything we ship today, since `loopir_to_gir` passes 1 at
    # both production call sites -- is BIT-IDENTICAL.  Production emission does not move.

```

## fence_regions.py:108 _already_fenced

```
Is a fence placed in an EARLIER block already on every path from producer to consumer?

    Without this the cover is per-block and blind across blocks: the last steady copy feeds reads
    in drain0 AND drain1, so drain1 asks for a fence of its own even though drain0's already sits
    between them on every path.  The fences are redundant, not wrong -- but redundant barriers are
    the exact cost this pass exists to remove.

    The condition is deliberately narrow, because a fence in the wrong place discharges nothing: an
    earlier fenced block `x` covers the edge only if it lies BETWEEN the two ends on every path --
    reachable from the producer's block (so it comes after the producer) AND dominating the
    consumer's block (so no path reaches the consumer around it).  Reachability alone would wrongly
    accept the prologue, which precedes the producer; dominance alone would wrongly accept it too.

    `x` must also be a DIFFERENT block from the producer's.  A loop block is reachable from itself
    along its own back edge, so `x == pb` satisfies "reachable from the producer" for free while
    saying nothing about whether the fence sits after the producer -- that depends on its SLOT, and
    this function has no slots.  A cross-block edge is deliberately covered on the CONSUMER side
    (`_admissible_slots`), so that every path reaching the consumer crosses the fence; a fence
    inside the producer's own block must therefore never count here.  This went unnoticed while
    drain0 had two predecessors and the steady block consequently did not dominate it.  With the
    `T < M` arm lowered to its own blocks (#183) steady DOES dominate drain0, and the steady-entry
    fence started masking the real steady->drain0 requirement.
```

## fence_regions.py:57 comment run

```
        # The producer ran in an earlier block, so within the CONSUMER's block every slot at or
        # before the consumer stands between them.  Covering on the consumer side rather than the
        # producer side is what makes this sound without dominance: a fence before the consumer is
        # crossed on EVERY path that reaches it, whereas one after the producer is only crossed on
        # paths through that producer -- and drain0 is reachable from the prologue as well as from
        # steady (the `T < M` short path), so the producer side would leave a path unfenced.

```

## fence_regions.py:3 <module>

```

FenceRegions -- the MINIMAL set of proc-scoped fences covering the cross-agent LDS hazards.

`LdsHazards` says which ordered pairs can touch one buffer and at what trip distance; this decides
where to stand between them. Only the cross-agent edges need a fence: the `complete(read) ~>
issue(copy)` is discharged by program-order-on-issue, which relates events on ONE agent, so a
same-agent edge is already covered by program order plus the completion counter.

WHY THIS IS A COVER AND NOT ONE MARK PER EDGE. A fence is not owed to an obligation -- it is a
POINT IN THE PROGRAM, and one point discharges every edge it happens to stand between. The steady
body holds ~20 cross-agent edges and one fence covers nearly all of them. Realizing them
one-for-one is what makes the current `Await` lowering emit 28-48 barriers where the backend's own
placement needs 7-10 (#199): not a worse analysis, an unbatched one. licenses the collapse --
"a decoder may instead discharge once... this is exactly the counter-class-coverage realization"
-- under honored-`sigma_c`, which holds because we emit the order.

THE SLOT ALGEBRA. A fence occupies a SLOT: slot `i` means "immediately before `body[i]`", with
slot `len(body)` at the end. For a hazard whose producer sits at index `a` and consumer at index
`b`, in a body that repeats:

 distance 0 (same trip, producer first) slot must satisfy a < slot <= b
 distance >= 1 (loop-carried) slot must satisfy slot > a OR slot <= b

The second is a UNION, not an interval, and that is not a technicality: at PGR1/PLR0 the copy is
last in the body (`a` large) and the read it feeds is first (`b == 0`), so the two admissible
regions are "after the copy" and "before the read" -- the top of the body and the bottom of it, and
nothing in between. The emitted scaffold puts its barrier at the top, which is one of exactly two
legal answers rather than an arbitrary choice.

Because the constraints are unions, minimal cover is a set-cover rather than an interval-cover, so
this greedily picks the slot covering the most still-uncovered edges. Greedy set cover is not
optimal in general; at these sizes (tens of slots, tens of edges) the gap is nil, and the result is
checked against the backend's own placement by #203 rather than trusted.

Output is `[PendingMark]`, the same currency `SwapRegions` and `GrIncrementRegions` deal in, so the
existing PlacementPass/ApplyMarksPass do the insertion unchanged. This analysis MUTATES NOTHING.

```

## gen_reaching.py:70 _entry_of

```
(per-Gen entry map, is_relative) for `blk` -- a real MERGE over its known predecessors.

    A header carries the phis, so its entry is declared.  Otherwise we merge: every known
    predecessor that AGREES on a Gen's value contributes it; a Gen the predecessors disagree on --
    or that no predecessor supplies -- gets `RELATIVE_BASE` and the block is flagged RELATIVE.

    The disagreement is real and not a modelling slip.  drain0's two predecessors are the last
    steady copy and the prologue (the `T < M` short path), and the generation the steady leaves is
    trip-parity-dependent, so NO compile-time constant is the merged value.  What this used to do
    was return `{}` -- silently base 0 -- and document it as "there is nothing to merge", which reads
    as a fact rather than a placeholder.  A consumer then cannot tell a merged generation from a
    fabricated one, and one did read it as absolute (the spurious drain swap, since fixed by
    computing frame-free requirements in SwapRegions instead).  Consumers that only need the
    RELATIVE distinction within a block -- the LDS memory token, which names a LOGICAL buffer and
    needs only this-buf-vs-other-buf -- are unaffected by the base and may ignore the flag.

    The old "exactly one known pred -> inherit" case (the G2 fallthrough chain) is not special here:
    one predecessor trivially agrees with itself, so the merge covers it.

    RELATIVITY IS TRANSITIVE, and it was not.  A block inheriting from a relative predecessor
    inherits the ARBITRARY base with it, so its own values are just as un-absolute -- but "one
    predecessor trivially agrees with itself" made the merge succeed and the flag was dropped.
    `drain1`'s only pred is `drain0`; `drain0` is relative (its `prologue`/`steady` preds genuinely
    disagree), so drain1's frame is arbitrary too, yet it reported `is_relative = False` and a
    consumer reading `of()` as an absolute buffer id got no warning.  Propagating the flag is what
    makes the contract "`of()` is absolute unless flagged" actually hold along a chain.
```

## gl2_prefetch.py:3 <module>

```

Gl2PrefetchRegions (#216) -- WHERE the GL2 cache prefetch goes, once per reduction chunk.

WHAT GL2 IS, AND WHY IT IS NOT A gr_increment.  `PrefetchGL2` (0/1/2, gated on the
`HasGlobalPrefetch` asm cap) warms L2 for a chunk `PrefetchGlobalRead + PrefetchGL2` ahead, while
the real global->shared copy runs at `PrefetchGlobalRead`.  It carries its OWN state end to end:

    v_add_co_u32   GL2PrefetchAddr<tc>_i, vcc, GL2PrefetchAddr<tc>_i, GL2PrefetchInc<tc>
    global_prefetch_b8  GL2PrefetchAddr<tc>_i, off

-- its own address VGPRs (`states.<op>.startVgprGL2PrefetchAddr`, `gl2nl` of them per operand) and
its own increment SGPRs.  It NEVER touches the TDM descriptor and never names a region coordinate.

THREE CONSEQUENCES, each of which contradicts the obvious guess:

  1. IT IS NOT PLACED RELATIVE TO THE COPY.  `gr_increment` is EARLIEST because the descriptor it
     writes feeds a `tensor_load` in the same trip, so every cycle between them is overlap.  GL2
     feeds NOTHING in this trip.  It has no producer to sit next to, so `after` is block entry.

  2. IT IS NOT PER REGION.  Under TDMSplit the copy becomes one instance per region, and
     `region_increment` follows it.  GL2 advances by a WHOLE DepthU once per chunk regardless of
     the split (`GL2Prefetch.py` contains no reference to `TDMSplit` or `split` at all), so this
     analysis emits ONE mark per chunk at every split factor.  Keying it per region would emit N
     prefetches of the same address range and burn N x `gl2nl` memory-pipe slots for nothing.

  3. IT IS NOT ONE INSTRUCTION.  `issueLoad`/`incrementAddr` both loop `for i in range(gl2nl)`
     where `gl2nl = ceil(gl2nc / numCooperativeThreads)`, per operand, and the operand set includes
     MXSA/MXSB and sparse Metadata.  The Mark stands for that whole block, which is why WHERE it
     lands is a real cost question rather than noise.

POLICY IS MIDPOINT, AND FOR A DIFFERENT REASON THAN THE LDS READ POINTER.  The read pointer is
midway because it is bounded by two DIFFERENT resources (last use of `gen_from`, first use of
`gen_to`) -- a correctness window.  GL2's window is legal at BOTH ends; midway is a BANDWIDTH
choice.  Landing it at `after` puts the prefetch block on the memory pipe while the trip's own TDM
copy is still in flight, competing for queue slots to fetch data nobody wants for `PGR+GL2` chunks;
the compute region in the middle of the body is where the memory pipe has slack.

    This is the ONLY Mark policy here that is purely a cost choice -- EARLIEST, MIDPOINT and every
    point between are all correct.  It is therefore the one worth MEASURING rather than fixing by
    argument, and the window is computed either way, so moving the pick is one line.

THE LATE BOUND IS REAL.  The increment block opens with `s_cmp_le_u32 counterL, PGR+GL2` then
`s_cmov_b32 GL2PrefetchInc<tc>, 0` for every operand -- a CLAMP that stops the pointer walking off
the tensor near the end of the loop.  It reads the loop counter, so the pair must sit on the
correct side of the counter update or the clamp tests a stale count.  That is what makes the window
bounded, and therefore what makes MIDPOINT mean something rather than collapsing onto BLOCK_EXIT.

```

## gr_increment.py:197 _flow

```
Place the advances for ONE global-read address, via the shared ValuePlacementSolver.

        This analysis owns only what is specific to this register: which instructions consume the
        descriptor, the absolute chunk each fetches, and how a chunk re-frames across the back edge.
        WHERE the advances go is `value_placement`, the same module SwapRegions uses.

        THE CONVENTION THIS FIXES.  The previous walk emitted an advance only where a copy DEMANDED
        a chunk different from the one in hand, anchored just before that copy.  The prologue's own
        copy demands the chunk the tile setup already left the descriptor on, so it demanded nothing
        and NO advance was ever emitted in the prologue -- the descriptor then ran one chunk behind
        the hand-written scaffold's convention for the whole loop.  It still inherited the
        scaffold's `StaggerUIter = S + PGR` wrap constant, so the wrap fired an iteration early and
        the address landed below the tensor base: an out-of-bounds `tensor_load`.  Under AVAIL/ANTIC
        the prologue->steady edge carries chunk 0 -> chunk 1, so the advance lands at the prologue's
        exit, right after its copy, and `KernelWriter`'s "GIR places every advance, prefetch 0's
        included" is true rather than aspirational.

        The domain is Z, not Z_ring: a global read address marches up the K axis and never wraps.
        
```

## gr_increment.py:167 _reqs

```
[(inst, chunk)] for `unit`'s copies in `lab`, program order.

 A COPY IS NOT A CHUNK. A region-split movement delivers one reduction chunk with
 `nregions` `tensor_load`s, so the peel's positional index counts LOADS while
 `_chunk_of` needs CHUNKS -- off by exactly that factor. At PGR2 with a 2-way split A's four
 peel copies were read as chunks 0,1,2,3 instead of 0,0,1,1: the address ran to chunk 3, the
 steady frame demanded 2, and the solver closed the gap with a `-1` advance that
 `tdmIncrementGir` cannot express (it emits exactly one DepthU stride), so kernel generation
 raised. At PGR1 the same error is one spurious advance with no negative -- no raise, just a
 descriptor a chunk ahead, which is why those kernels built and miscompared instead.

 Dividing is exact because the regions of one chunk are issued CONSECUTIVELY and in order
 0..n-1. That is not an assumption: it is what `RegionIncrementRegions` emits and what
 G-WALK checks in program order, so a violation is a verifier error rather than a silently
 wrong chunk here.
```

## gr_increment.py:3 <module>

```

GrIncrementRegions -- the global-read-address advance, as a REAL forward
dataflow over the GIR CFG (the same shape as SwapRegions).

Each operand's global read address is a loop-carried POINTER holding the reduction chunk the next
copy will fetch. A `gr_increment` is the only thing that advances it, so -- exactly as for the LDS
pointers -- placement is a reaching-value problem: propagate the address to every program point and
insert an increment wherever supply and demand differ. The fixpoints and the def-placement rule
live in `value_placement` -- the SAME module SwapRegions uses, so the two cannot drift -- and this
file supplies only what is specific to this register: which copies consume the descriptor, the
absolute chunk each fetches, and how a chunk re-frames across the back edge.

WHAT THIS REPLACES. The previous version claimed in this same docstring to be "a GLOBAL
edge-based dataflow (the same shape as SwapRegions)", "a reaching-definition over the WHOLE CFG",
and explicitly "NOT a per-region structural 'one Mark per steady copy' rule (that was the ad-hoc
version)". It was that rule: `if lab in loop_headers and any(be.xfers...)` then one Mark per copy
Move. With `loop_copies > 1` (G2) its two filters became mutually exclusive -- the non-header body
copies failed `lab in loop_headers`, and the header failed the xfer test because the back edge
leaves the LAST body block, not the header -- so it emitted NO increments at all, in every steady
block. The address then never advanced and every trip re-fetched the same tiles. (`BackEdgeSet`'s
`headed_by` filtering on `src` is what made that inversion easy to write; it is `leaving` now.)

PLACEMENT, NOT JUST EXISTENCE. An advance is anchored AFTER the copy whose use of the descriptor
it supersedes, so the descriptor leads its next `tensor_load` by as much of the block as possible.
The previous version anchored before the copy that DEMANDED the new chunk -- the latest legal point --
and, because the prologue's own copy demands the chunk the tile setup already left the descriptor
on, emitted NO advance in the prologue at all. GIR's descriptor then ran one chunk behind the
scaffold while inheriting the scaffold's `StaggerUIter = S + PGR` wrap constant, so the wrap fired an
iteration early and drove the address below the tensor base: an out-of-bounds `tensor_load`.

THE VALUE DOMAIN IS Z, NOT Z_ring. The LDS pointers rotate through a finite ring; a global read
address marches monotonically up the K axis and never wraps, so there is no modulus here and no
meet-to-TOP from a wrap. The requirement at a copy is the absolute chunk it fetches,
`blk.chunk_base + (ref.gdelta - blk.gen_rel)` -- frame-free, so copies in different body blocks are
directly comparable.

SCOPE. Every block that carries a copy Move, in program order -- the prologue peel fills included.
A dataflow over "the loop body only" was the old rule's assumption inherited one level up: the
prologue fetches chunks 0..M-1 into distinct buffers, so its address advances between those fills
exactly like the steady's does, and a flow that cannot see them cannot state that. The seed is the
FIRST copy's own chunk (the tile setup leaves the address there), so no increment precedes it and
every later advance is derived.

The Mark carries WHERE only. The magnitude -- the DepthU stride and every modifier (StaggerU WrapU,
TDMSplit subtraction, sparse metadata, PGR prefetch-index removal) -- is L3's, read from
`kernel`/`states` by `writer.tdmIncrementAB`. That is the permanent split: GIR owns WHERE,
L3 owns HOW MUCH.

```

## lds_hazards.py:346 _cross_block

```
Hazards whose two ends live in DIFFERENT blocks.

        These are real and they are not covered by any fence inside a single body: the last steady
        copy feeds the drain's reads, and the steady fence sits at the TOP of the body, so on the
        final trip nothing stands between that copy and the drain.  The backend's flat-stream phase
        machine catches it for free (it just keeps walking); an analysis that reasons per body has
        to model it explicitly or it will under-fence exactly where the scaffold does not.

        Aliasing is decided CONSERVATIVELY here, and deliberately so.  Two accesses in different
        blocks on the same region are assumed to collide unless BOTH are peel-pinned in blocks whose
        generation frame is absolute -- because a loop-carried access takes a different generation on
        every trip, so comparing its block-entry (trip-0) value against another block's would
        exclude pairs that genuinely alias on a later trip.  `gen_reaching` reports the frames it
        cannot merge (`is_relative`) rather than inventing a base, and this respects that.

        Direction is decided by REACHABILITY, not dominance: a hazard exists if some execution runs
        the producer before the consumer, and drain0 is reachable from steady even though steady
        does not dominate it (the `T < M` short path enters drain0 from the prologue too).
```

## lds_hazards.py:247 comment run

```
    # A MERGED MOVEMENT NAMES EVERY BUFFER IT COVERS.  `Ref.covers` is the reference's own
    # declaration that ONE instruction fills several coordinates (the movement coverage, Sec 2.2):
    # the leader's `coord` pins the axis, but the instruction also touches the coordinates the
    # coverage absorbed.  Taking the pin alone labels a merged `ds_load` with the leader's buffer
    # only, so the halves it also reads go un-ordered -- the same failure the `bothHalves` note
    # below describes, arrived at from the other direction.
    #
    # THIS IS NOT THE WIDENING REJECTED ABOVE.  That one guessed "may touch any region" for a coord
    # that simply FAILED to say; this one reads an explicit span the producer wrote down, and it
    # widens ONLY the axes named in `covers`.  A reference with no span is untouched, so every
    # unmerged access keeps its exact pin.

```

## lds_hazards.py:222 _regions_of

```
Per region axis of `operand`, the SET of region values this access MAY touch.

    A coord that PINS the axis names one region.  A coord that does not mention it touches EVERY
    region, so the unknown is modelled as the full set and stays a MAY-alias -- being conservative
    costs a fence, being optimistic LOSES one.

    THE PIN IS TRUSTED ON BOTH HOPS, and that is the point: a region split is a COORDINATE fact.
    `Tile.coord` carries the region value, so this says which half every access SHOULD touch.
    Widening our reads to "may touch any region" was tried (2026-08-16) and is WRONG: it hides a
    defect rather than modelling one.  If an access genuinely cannot say which region it touches,
    the IR that produced it is incomplete, and that is the thing to fix, here or upstream.

    THE SCAFFOLD'S `bothHalves` HATCH IS NOT THE SAME CLAIM, and an earlier version of this comment
    got its reason wrong ("its split order and loop order differ").  `LocalRead.tdmBothHalves`
    fires at `numVectorsPerTile == MIWaveTile[t] // VectorWidth == 1`, where a wave's read for a
    tile is a single vIdx that genuinely SPANS both halves -- the region boundary falls mid-wave
    because of the address calculation and the wave distribution, not because the scaffold cannot
    tell.  So it is a real span, and naming both halves makes the token describe the address
    instead of making the address respect the region.  We keep the pin and REJECT that shape
    instead (`Solution.py`, VW>1 with TDMSplit under UseLoopModel); fixing it is L3/scaffold
    address work (#237).
```

## lds_hazards.py:186 split_region_modes

```
`region_modes`, with the axes of movements that have ONE region REMOVED.

 THE TOKEN COUNT IS ARITHMETIC, AND IT WAS WRONG EVERYWHERE. For A and B split in two with a
 double-buffered LDS ring that is `2 regions x 2 buffers = 4` tokens each; the MX scales are
 NEVER split (`tdm_split.split_of` returns `(1, None)` for every MXS tensor), so they are
 `1 x 2 = 2` each -- twelve in total. Every kernel we emitted named SIXTEEN, because
 `_regions_of` keyed on `region_modes` being non-empty and the scales carry their parent's axis.

 THE TUPLE ANSWERS "WHICH AXIS", NEVER "HOW MANY". The scales keep the axis deliberately: the
 two-map reading needs it BROADCAST FOR THE RATE, and removing it at the source drove MXSA's
 rate R from 2 to 4 (measured 2026-08-19, and it cost 26 kernels). So the count must come from
 `unit_regions`, which is what `region_increment._n_regions` already does and what
 `geometry.bulk_broadcast` / `placement.op_level` approximate with their `split > 1` guards.
 Storage naming was the one consumer with no guard at all.

 WHAT CORRECTING THE COUNT DOES, MEASURED 2026-08-19: nothing, on hardware. mxf8 is 786/6
 before and 786/6 after, with the same six cells failing. So the four extra tokens were
 genuinely inert here -- they named buffers no read consumed, and an unused def imposes no
 ordering. They are removed because the count is arithmetic and 16 was not it, not because a
 kernel depended on the difference.

 (An intermediate build showed 760/32 and was briefly blamed on this change. It was not: that
 build ALSO carried an experiment emptying the scales' `region_modes` at the theta source, which
 drove MXSA's rate R from 2 to 4 and re-slotted every scale register. Reverting that and
 keeping this restores 786/6. Two changes, one build, one wrong conclusion.)

 An operand absent from `unit_regions` keeps its axes: unknown count stays conservative.
```

## lds_hazards.py:3 <module>

```

LdsHazards -- the LDS hazard edges, derived from the SSA.

The three families a reachable state can hold are RAW residency, rotation WAR/WAW
and a cross-role/cross-agent alias. This analysis finds them on the SHARED
(LDS) side by asking which accesses can touch the same physical buffer, and -- crucially -- at what
TRIP DISTANCE.

WHY THIS EXISTS. The backend's own barrier placement is a per-token three-state phase machine over
the flattened instruction stream (`KernelWriter.postMainLoopBarrierCheckAndReset`). It can only see
what a static label says, so it cannot see a dependence whose two endpoints are the SAME static
instruction on different trips -- which is every loop-carried LDS dependence in a rolled loop. GIR
can: the `Gen` SSA value carries the ring, `gdelta` carries the per-access offset within it, and the
back-edge `GenXfer` carries the per-trip advance. See `MEMORY_TOKEN_FOOTPRINT.md`.

THE ALGEBRA (the whole analysis, in four lines). An access with offset `g` on a ring of `S`, at
trip `v` of a loop whose back edge advances the generation by `adv`, touches buffer

 (base + v*adv + g) mod S

so a producer `(v_p, g_p)` and a consumer `(v_c, g_c)` on the SAME ring touch the same buffer iff

 (v_c - v_p) * adv == (g_p - g_c) (mod S)

With the `adv = 1` the emitter produces, the trip distance is `d = (g_p - g_c) mod S` -- a number the
phase machine has no way to compute, and the reason this is an analysis rather than a labelling.

`d` is only known MOD S, which is not a defect but the truth: buffer `x` really is revisited every
`S` trips, so "same trip" and "S trips later" are the same physical collision. What the consumer of
this analysis needs from `d` is exactly one bit -- whether a SAME-TRIP instance exists (`d == 0`) --
because that decides whether the fence must sit between the two accesses INSIDE the body, or whether
any fence in the body suffices (it will be crossed on the way from trip `v` to `v+1`).

SAME-AGENT EDGES ARE NOT HAZARDS HERE. the reused-slot WAR is discharged by
`complete(read) ~> issue(copy)` via program-order-on-issue, which relates events on ONE agent. When
producer and consumer are the same agent, program order plus the completion counter discharge the
edge and no fence is needed. Only a CROSS-AGENT edge needs one. This is why a
single-wave kernel yields no fences at all -- matching the backend's own `numWaves == 1` skip, which
is a good consistency check on the model rather than a coincidence.

NOT rho (#276). This paragraph used to attribute the cross-agent test to rho. It does not read rho, and
could not: `Theta.rho` is empty in every kernel and has zero readers. What the test actually reads
is the AGENT IDENTITY already on the two endpoints -- which agent issues each access -- and that is a
different fact from rho, which would name *which selector discharges* an obligation, not *which agent
holds* an access. Citing rho here made a solved problem look blocked on an unimplemented field.

SCOPE OF THIS VERSION. Pairs are resolved exactly when both accesses are in the SAME block (the
steady body -- where every loop-carried edge lives) or when both carry a pinned `abs_gen` in a block
whose generation frame is absolute. Cross-frame pairs are NOT guessed: `gen_reaching` reports that
a block's entry generations may be on an arbitrary base (`Reaching.is_relative` -- drain0's
predecessors genuinely disagree, because the residue the steady loop leaves is trip-parity
dependent, a RUNTIME value). Comparing generations across such a frame would be fabricating a fact.
Those pairs are recorded in `unresolved` so a consumer sees the gap instead of inheriting a silent
"no hazard". This analysis MUTATES NOTHING.

```

## loop_shape.py:198 comment run

```
            # NOT YET EXPRESSIBLE, and that is a limit of the predicate form, not a choice (#230).
            # A trip consuming `n` chunks must run `(T - M) / n` times, and `Bound` is `var + const`
            # with no division, so the bound cannot be WRITTEN -- never mind checked.  The G2
            # multi-copy chain therefore over-covers by a factor of `n` today; it is unreachable
            # from `build_gir` (which never passes loop_copies > 1) and pinned by
            # `test_multi_copy_loop_coverage_is_not_yet_EXPRESSIBLE`.  Skipping silently is what
            # would make this a hidden hole, so it is named here and asserted there.

```

## loop_shape.py:3 <module>

```

LoopShape (#229) -- DERIVE a reduction loop's test position and trip count from the CFG, and check
that the steady region and the drain together cover the reduction exactly once.

WHY THIS EXISTS.  Whether a loop is PRE- or POST-tested is not a property anyone declares; it is
*where the conditional branch sits*.  Put the test on a block that also does work and the work
happens first (post-test, `do { body } while (p)`); put it on a block that only tests and the work
happens after (pre-test, `while (p) { body }`).  The same predicate means a different trip count in
the two shapes.  Nothing in GIR derived that, and nothing checked the consequence, which is exactly
how #228 survived: `loopir_to_gir` emits

    Block(phase='steady', preds=('prologue','steady'), term=CondGoto(iter < T-M, 'steady','drain0'))

-- one block that IS the body, with the test at its end.  So the body runs at `iter = 0..T-M`
inclusive, `T-M+1` times, covering chunks `0..T-M`; the drain's Binds are `T-M .. T-1`; and chunk
`T-M` is computed TWICE, at every `T`.  It has never been observed because the predicate is never
emitted -- ScaffoldMapPass only labels it and TensileLite branches on its own compare -- but every
analysis downstream reasons over the loop GIR *claims* to describe.

THE INVARIANT, and note it needs no trip count.  `T` is a runtime symbol, so "how many times does
this run" is not a number.  What IS checkable is a COVERING statement, symbolically:

    steady covers  [0, n_steady)          n_steady = trips x chunks-per-trip
    drain  covers  [T-M, T)               M drain steps, one chunk each
    together, exactly [0, T) -- no gap, no overlap   <=>   n_steady == T - M

Both sides are linear in `T`, so the comparison is exact arithmetic on `(coeff, const)` pairs, and
it fails on #228 immediately.  A structural pass that changes the loop's shape (rotation, trip
adjustment -- #229) is checked by this and cannot quietly move the trip count.

WHAT IT DOES NOT COVER (stated, not silently skipped): a loop whose exit predicate is not
`counter < bound` over the trip symbol, and a `chunks-per-trip` that the back-edge `GenXfer.adv`
does not state.  Both raise rather than guess -- an unverifiable loop must not read as a verified
one.

```

## mem_tokens.py:198 comment run

```
            # THE GENERATION IS A MAY-SET WHENEVER THE BLOCK'S FRAME IS RELATIVE.
            #
            # `Reaching.of()` is documented as meaningful only RELATIVE to other values in the same
            # block when `is_relative` holds -- "on an ARBITRARY base ... never as an absolute buffer
            # identity, and never comparable across blocks".  A memory token is precisely an
            # absolute buffer identity compared ACROSS blocks: StinkyTofu turns it into an LDS
            # pseudo-register and pairs a read's token against the token of whichever instruction
            # DEF'd it, which for the peel is a copy in the entry block on an ABSOLUTE frame.  So
            # taking the relative number as exact is not a rounding error, it names a different
            # buffer, and the waitcnt derived from it guards the wrong producer.
            #
            # MEASURED 2026-08-19: `drain0`'s preds are `prologue` and `steady`, which genuinely
            # disagree (the residue the steady loop leaves is trip-parity dependent, a RUNTIME
            # value), so `_entry_of` flags it relative and returns RELATIVE_BASE.  On the `T <= M`
            # path -- where the steady loop never runs -- the resident buffer is the prologue's, and
            # the stamped token named the other generation: the read was ordered against a copy that
            # is not its producer and un-ordered against the one that is.  MXFP8 turned that into
            # `-nan` (a wrong-generation E8M0 scale poisons the wmma; wrong-generation data often
            # stays finite), 23 of 540 ULM1 cells, every one of them at `T <= M`.
            #
            # The honest answer is the one this module already gives an unpinned REGION axis: the
            # access may touch every generation of the ring, so it carries every id and StinkyTofu
            # orders it against all of them.  Conservative -- a redundant wait -- where being
            # optimistic loses one, which is the trade `_regions_of` states in the same words.
            # A SECOND CASE BELONGS HERE AND IS DELIBERATELY ABSENT (2026-08-27, reverted).
            #
            # A LOOP-CARRIED DEF WITH NO FENCE is the same trade with a different trigger: in a
            # rolled body the copy that fills generation `v+1` and the read that consumes it are
            # the SAME static instruction a trip apart, so their exact ids differ by construction
            # and no def-use edge crosses the back edge.  Single-wave has no fence to carry it
            # instead (`FenceRegions` emits nothing without a cross-agent edge), so the edge is
            # simply absent and the derived `s_wait_tensorcnt` is short by one copy -- the
            # 2026-08-27 bf16 (332) / f8 (18) ULM1 failures, all single-wave + PLR1/PLR2.
            #
            # Widening the producer's def here was TRIED AND REVERTED: it restores the alias but
            # not the count.  `StinkyBuildImplicitDependencyPass` builds the chain as
            # `producer(def) -> BARRIER(use+def) -> consumer(use)`, no PHI is ever created for an
            # LDS pseudo-register (`PhiPlacement` is in no pipeline and handles no `RegType::LDS`),
            # so `WaitDataflow`'s `phiSummaries` path is dead for tokens and a loop-carried
            # producer has no path in the chain at ANY token value.  Measured: the failing `MKN`
            # cell's `s_wait_tensorcnt` was unchanged before and after, in both SIA0 and SIA4.
            # The fix belongs on the bridging node (an LDS PHI, or a token-carrying `GFX::FENCE`),
            # or in unrolling the body by the ring depth so the producer is intra-body.
            # THE SECOND MAY-SET CASE: A LOOP-CARRIED DEF WITH NO FENCE.  RESTORED 2026-08-28.
            #
            # This arm was added by `61eb5cfa` ("fix single wave token") and DELETED by
            # `0b8eb3c890b` ("fix single wave token 2"), which replaced it with StinkyTofu's
            # `EnableLoopCarriedTokenDeps`.  That flag cannot stand in for it: loop-carried tracking
            # orders ops that SHARE a token, but in a rolled body the copy filling generation `v+1`
            # and the read consuming it are the same static instruction a trip apart, so their exact
            # ids differ BY CONSTRUCTION and there is no edge for the flag to carry.  Widening the
            # WRITE end is what makes the consumer's use a subset of the producer's def so an edge
            # exists at all.
            #
            # Re-measured 2026-08-28 on the failing f8 `LONKM` kernel: the steady body produces
            # token LDS1 and NO read in the body names it, exactly the "ids with no producer"
            # signature the original commit recorded.  The write end only -- widening reads too
            # would alias read against read and order them for nothing.

```

## mem_tokens.py:131 _unfenced_loop_carried_writes

```
Write refs whose LOOP-CARRIED edge nothing separates -- the defs that must span the ring.

    `Hazard.same_trip` states the rule: "a non-zero distance is purely loop-carried, and ANY fence
    in the body separates trip `v` from trip `v+1`."  So a loop-carried pair is covered exactly when
    its block has a fence, and `needing_fence()` -- the ONE place the fence/no-fence decision is
    made -- says which blocks get one.  Asked of `LdsHazards` rather than re-derived, so this and
    `FenceRegions` cannot disagree about which edges are covered.

    A SINGLE-WAVE KERNEL HAS NO CROSS-AGENT EDGE, hence no fence, hence nothing covering its
    loop-carried copy->read RAW -- and then the exact generation gives the producing copy and its
    next-trip consumer two DIFFERENT ids, so no def-use edge crosses the back edge at all.

    CONFIRMED 2026-08-27 on a failing kernel: the steady block held 3 copies on ids {0,2,4} and 16
    reads on {0..5}, so ids 1/3/5 had no producer anywhere in the body; StinkyTofu derived
    `s_wait_tensorcnt 4` where <= 3 is needed -- short by exactly the previous trip's copy.  Every
    ULM1 numeric failure in that day's bf16 (332) and f8 (18) runs was single-wave, PLR1, and K
    large enough to reach the steady body, while 2920+ multi-wave configs passed.

    THE WRITE END ONLY.  Widening the producer's def makes every consumer's use a subset of it, so
    the edge exists; widening the reads too would additionally alias read against read and order
    them for nothing.
```

## mem_tokens.py:3 <module>

```

MemTokenAssignment (#235) -- GIR names every LDS buffer, and the id IS the memory token.

WHAT A MEMORY TOKEN ACTUALLY IS. Not a label: an LDS PSEUDO-REGISTER. StinkyTofu's
`StinkyBuildImplicitDependencyPass` turns `MemTokenData{tokens}` into `RegType::LDS` operands --

 tensor_load / ds_write -> the token as a DEF (LDS producer)
 ds_read / global_store_async_from_lds -> the token as a USE (LDS consumer)
 barrier / signal / wait -> BOTH (a scheduling point)

-- and the ordinary def-use chain then enforces `producers -> barrier -> consumers`. So the token
relation is exactly an ALIAS relation:

 same token "these may touch the same storage; order them"
 different token "provably disjoint; reorder freely"

Which is the relation `LdsHazards` already computes. This analysis just names it.

WHY GIR OWNS IT. TensileLite hand-assigns the ids (`KernelWriter`: `memTokenLdsBuffer0/1` = 0/1,
`memTokenLdsSplit = [[0,2],[1,6]]`, `memTokenLdsBufferMeta = 4`) and, under ULM, they reached the
instruction through a MUTABLE SIDE CHANNEL -- `gir_to_rocisa` wrote `states.ldsTensorTokenIdx` and
the leaf read the field back. That channel carries ONE int, so it could express the buffer parity
and nothing else, which is why two refusals had grown on top of it (a ring deeper than 2; a
per-region token). Worse, the READ side never took the split path at all, so a split tile's
half-1 loads went un-waited (#217).

GIR already knows the answer for every access -- the buffer is `(operand, region, generation)` -- so it
assigns the ids directly. That expresses any ring depth and any region count, makes the
`bothHalves` case exact instead of hand-flagged, and separates A from B (under the scaffold's
numbering both sat on 0/1, so A's read was ordered after B's load for no reason).

THE KEY, and why each part is in it:
 operand WHOSE storage this is. NOT the Phi completion unit -- that was the first cut and it was
 wrong: a fused movement is ONE COMPLETION but ONE INSTRUCTION WRITING
 TWO STORAGES, since A's LDS region and B's are different memory. Naming the buffer
 `A+B` merged them, so A's read and B's read shared a token and were ordered against
 each other for nothing, and no access could name A's region alone. The fused
 instruction instead defs BOTH ids -- `_storage_ids` unions over its Refs, one per
 member -- which is exactly what a vector of tokens is for. This is the same
 completion-vs-storage distinction `Move.token_ids` exists to make; keying storage on
 the completion unit put it straight back.
 region a region split makes storage-DISJOINT siblings. Set-valued: an access
 that does not pin the axis may touch every region of it, and gets every id -- the
 honest reading of the scaffold's `bothHalves` shape, derived instead of flagged.
 generation the rotating buffer within the ring. One generation, exact (#207); the
 fences carry the ordering between generations.

Ids are dense and assigned in sorted-key order, so a Program always yields the same numbering.

```

## region_increment.py:189 walk_violations

```
[] or human-readable G-WALK violations.  Two properties, both replayed in PROGRAM ORDER:

      1. EVERY SPLIT COPY LOADS THE REGION THE DESCRIPTOR IS ON.  Walking the block, the running
         position must equal each copy's own region coordinate when that copy issues.
      2. THE WALK CLOSES.  It ends back at region 0, because `tdmIncrementGir` then applies the
         plain chunk stride and any leftover compounds every trip.

    (2) alone is not enough, and finding that out cost a measurement: the first version checked only
    the net, which is zero even when every step has been hoisted to the block entry ahead of all the
    copies -- net-clean and uniformly wrong.  (1) is the property that actually matters; (2) is what
    makes the next chunk's stride correct.

    The original hardware bug is (2) failing (`+2` forward against `-1` back, from `globalReadDo`
    and `tdmIncrementGir` each owning part of one walk); the placement bug this found is (1).
```

## region_increment.py:100 comment run

```
        # READING refs[0] IS JUSTIFIED BY A PRECONDITION, SO CHECK THE PRECONDITION (#170).
        #
        # A Phi group pairs its members BY INDEX (move: "the pairing is by the index the
        # partition names, not by a common mode"), so region j of A moves with region j of B even
        # when they are split on DIFFERENT axes.  One aliased `Group0` quad advances both, and
        # `tdmRegionIncrementGir` walks ONE axis per step -- `gir_to_rocisa.py:209` refuses a
        # multi-axis step outright.  So naming ONE member's axis is not a simplification that loses
        # B's displacement; it is the only form L3 can execute, and it is SUFFICIENT precisely
        # because the indices agree.
        #
        # That agreement is enforced upstream (`Solution.py` rejects a fuse whose members have
        # different region counts, #169/#282) and was previously only assumed here.  An assumption
        # that upstream can change is exactly how the `refs[0]` reads in this file came to look like
        # a dropped fact.  Check it instead: if two members ever disagree, the one-axis step is
        # silently wrong for whichever member did not supply it.

```

## region_increment.py:3 <module>

```

RegionIncrementRegions (#236) -- WHERE the TDM descriptor steps between STORAGE REGIONS.

WHY THIS EXISTS. A region-split tile is loaded by one `tensor_load_to_lds` PER REGION,
and the descriptor has to move between them. `KernelWriterAssembly.globalReadDo` used to do that
walk inside itself -- load half 0, advance, load half 1 -- and `tdmIncrementGir` subtracted the
advance back off so the per-chunk round trip closed. GIR emits one copy Move per region and calls
the loader once per Move, so the two owners disagreed: `+2.splitInc` applied against `-1.splitInc`
removed, leaving the descriptor a region ahead every chunk and one load reading past the tile. All
MIWaveTile [2,2] cells miscompared (2026-08-16).

Now GIR owns the whole walk: `tdmLoadRegionGir` loads exactly the region the descriptor is on, this
analysis says where the descriptor moves, and `tdmIncrementGir` applies the plain chunk stride with
no correction. ONE AUTHOR PER DISPLACEMENT.

WHY A PROGRAM-ORDER SCAN AND NOT A DATAFLOW, unlike its sibling `GrIncrementRegions`. The chunk an
address points at is a genuine dataflow fact: it is carried across blocks and back edges, joins have
to agree, and that is why the sibling runs a `ValuePlacementSolver`. The region walk is not that.
It opens and closes WITHIN one chunk's copies, which the emitter lays out consecutively in one
block, so its whole extent is visible in program order and there is nothing to propagate. Modelling
it as a fixpoint would be vocabulary, not analysis.

What makes that safe rather than merely convenient is that the closure is CHECKED: `verify_gir`'s
G-WALK requires the net displacement per unit per block to be zero, so a walk that escapes a block
-- which would mean the assumption above is false -- fails loudly instead of silently leaving the
descriptor displaced.

```

## short_path.py:541 _arm_is_unreachable

```
Is the `T < ...` arm dead code?  DERIVED from the peel-validity guard, not from M.

    This used to be `M == 1`, justified as "the arm's only step is unguarded because `T > 0` is
    trivially true, so it runs only at T == 0".  That reasoning was tied to a `T >= M` guard.  The
    guard is now strict (`T > M`, forced by the post-tested steady region -- #228), so the arm runs
    whenever `T <= M`, and at M == 1 that includes **T == 1**, an ordinary trip count.  The old rule
    would have folded an arm whose pairings provably break, on a path that really executes -- and it
    would have done so silently, because VACUOUS skips the break check.

    Derived form: the loop is entered iff `T > M`, so the arm takes every `T <= M`.  A reduction has
    at least one chunk (`T >= 1`), so the arm is dead only when `M < 1` -- and `M == 0` means there
    is no arm at all (handled as NA above).  With the strict guard it is therefore never dead; the
    predicate is written out rather than hardcoded to `False` so that a future guard change is
    picked up here instead of silently re-enabling the old bug.
```

## short_path.py:446 comment run

```
            # F2 IS ASKED AT THE `wmma` ONLY.  A `wmma` is the sole instruction whose semantics the
            # two shapes must agree on; a read is a MEANS, and its identity is not stable across
            # them.  At `dr > 0` the drain's read at coord `(k0,m0)` of step 0 prefetches that
            # coordinate from the NEXT chunk, while the arm's read at the same coord (at `dr = 0`)
            # fetches the current one -- a true difference, and a meaningless one to report, since
            # what matters is which data reaches the compute.  Comparing at reads made the reset's
            # own intent look like a defect (#227).  A read that sources an unwritten buffer is
            # still caught: F0 below scans EVERY consumer, reads included.

```

## short_path.py:292 _trace

```
One forward walk of an arm: returns `(reach, keys, extras)`.

    `reach`  {consumer key: {location: the VALUE that reaches it}}
    `keys`   the consumer keys this arm defines
    `extras` instructions with no consumer key (a prologue fill; the drain's next-tile prefetch)

    TWO IDENTITIES, and conflating them is what made this analysis wrong (#227):

    THE CONSUMER KEY -- which instruction in the other arm is "the same" one:
        `(kind, step, operands, coord)`, checked unique.  Deliberately NOT the position within the
        step: the short arm runs at `dr = 0` while the drain keeps `dr = dr(theta)`, so
        the two are *intentionally different schedules*.  Position meant something while they were
        the same schedule and means nothing now.

    THE VALUE -- what an instruction produces, which is what a consumer actually depends on:
        copy -> `('copy', operands, coord, occurrence)`.  The k-th copy of an operand fetches chunk
               k in both arms, so the occurrence index IS the chunk.
        read -> `('read', operands, coord, <the value of the copy that filled its source buffer>)`.
               TRANSITIVE, and that is load-bearing: naming the source GENERATION instead is only
               an identity for the chunk while the ring is at least as deep as the peel.  At
               `S_shared = 1` every generation is 0, so a generation-keyed value called two reads
               of DIFFERENT chunks equal and reported a false FOLD.  Chaining to the producer says
               which data, at any ring depth.
    
```

## short_path.py:261 pairing_cause

```
Classify one corresponded consumer's producer pair, or `None` when the two frames agree.

    ONE derivation, called only from `ShortPathFold.run`'s F2 loop -- it lives here rather than
    inline because the ladder's ORDER is load-bearing and a second copy would drift (see the
    recurring shape: the read-ahead traversal, the read's tree level and the fence cover were each
    one rule written twice, agreeing on every ordinary input and diverging on the degenerate one).

    `f_src` is the producer the FOLDED path supplies, `s_src` the one the SHORT ARM supplies.
    The first rung is the trap: AGREEING ON "NO PRODUCER" IS NOT AGREEMENT.  `f_src == s_src` at
    `ENTRY` means both sides say "nothing wrote this", and returning `None` there records no Break,
    which leaves `fold_breaks` empty and reports the folded path SOUND on the exact input that
    proves it is not.  The second rung is the mirror trap: both-unserved must be blamed on the
    FOLDED path, because `ARM_UNWRITTEN` is filtered out of `fold_breaks` and so asserts "the arm is
    broken but folding rescues it" -- and here folding rescues nothing.

    Both sides are tested against `_UNSERVED` rather than `ENTRY` alone.  For `s_src` that is
    currently a no-op -- `_trace` only ever stores `ENTRY` or a real producer, the `("<absent>",)`
    spelling being `_folded_producer`'s -- but the rule is about being unserved, not about which
    helper produced the sentinel, so an asymmetric test would be the next drift.
```

## short_path.py:188 _free_coord

```
The FREE-TILE part of `coord` for `operand` -- the coordinate that selects a distinct
 physical register, as opposed to the rotation slot.

 A read's register is named by TWO independent things : the rotation
 `(group, slot)`, which covers the reduction axis, and the operand's own free-tile coordinate,
 which the model leaves to allocation. Omitting the second merges registers that are physically
 distinct -- `A(k1,m0)` and `A(k1,m1)` share a slot but are different VGPRs -- and every read of
 one then looks like it clobbers the other.

 Read PER OPERAND from `meta['free_modes']`, not by subtracting the reduction modes from the
 coord. The historical reason was that an `Mma` src Ref carried the whole `(K,M,N)` coord, so
 B's Ref mentioned `M_inner` even though B is broadcast over it, and keying on that would split
 B's register per `m` -- inventing differences instead of merging them. THAT IS FIXED AT SOURCE
 NOW (`loopir_to_gir._convert_mma` projects each source onto `pres(operand)`), so this function
 no longer has to undo it.

 It still reads `free_modes` rather than trusting the coord, for the reason in the next
 paragraph: the two spellings this must equate do not both come from an `Mma`. A coverage-wide READ has its spanned axis appended out of loop order position, and only a canonical order
 makes it compare equal to the consuming wmma's. So the per-operand lookup is now doing one job
 (canonicalisation) instead of two (canonicalisation + repairing an over-wide Ref).

 THE ORDER IS CANONICAL (`free_modes`'s), not the coord's. A key is compared for EQUALITY
 against other keys, so two spellings of the same location must produce the same tuple -- and
 they do not arrive spelled alike: a coverage-wide def has its spanned axis APPENDED by
 `refs.covered_coords` (the axis is absent from the read's the axes it varies over, so there is nowhere else to
 put it), while the consuming `Mma`'s coord carries it in loop order position. Preserving the coord's
 own order made `(M_inner, M_split)` and `(M_split, M_inner)` different keys and every wide
 read's second tile looked unfed.
```

## short_path.py:147 _frame_base

```
The CHUNK POSITION a block's `gdelta = 0` is stated against, ON THE FOLDED PATH (#230).

    This is the correction that made PGR1 and PGR3 decidable.  A drain Ref's `gdelta` is NOT
    relative to the loop's exit; it is relative to the LONG PATH's chunk position `T`, so drain step
    `i`'s own chunk is `gdelta = i - M`.  Resolving that against the phi's entry value -- base 0 --
    silently assumes `M` is a whole number of ring revolutions:

        resolved  = (entry + gdelta) % ring = (0 + i - M) % ring
        should be = chunk i               = (0 + i)     % ring        <=>   M = 0 (mod ring)

    At ring 2 that is true for M = 2 and false for M = 1 and M = 3, which is exactly the observed
    coverage / split / split.  At PGR1 the drain's own-chunk read resolved to generation 1, which
    NOTHING on the folded path had written, and the arm was reported UNWRITTEN-then-split against a
    generation that does not exist.  The break was in the resolution, not in the schedule.

    The coverage identifies drain step `i` with short step `i`, i.e. with chunk `i`, so under the coverage
    the drain's `T` is `M` and the base is `M`.  Derived from the BLOCK's own two facts rather than
    from `meta['M']`, because `chunk_base` is a per-PATH fact stored per-block (#222) and reading a
    program-wide `M` here would be the same frame confusion one level up.
```

## short_path.py:3 <module>

```

ShortPathFold (#183) -- may the `T < M` arm be FOLDED into the shared prologue/drain path?

THE TWO SHAPES.  TensileLite's scaffold and the LoopIR describe the same kernel differently:

    scaffold :  prologue -> Cond -> loop -> else -> drain    prologue+drain SHARED by both paths;
                                                              `T < M` just skips the loop
    LoopIR   :  Cond { prologue -> loop -> drain } else { short }      two DISJOINT arms

Folding means deleting the `short` blocks and letting the peel-validity FALSE edge fall into the
drain, so the `T < M` case is served by the prologue's copies followed by the drain's computes and
no extra code is emitted.  That is what the lowering used to do UNCONDITIONALLY, by discarding the
`els` arm on the claim that "the scaffold reconstructs it from prologue+drain".  This analysis
exists because nobody had checked the claim, and it is false for a third of the configuration
space.

WHY IT CAN FAIL -- the coverage is a REORDERING.  Both shapes contain the same instructions (a fact this
analysis re-derives rather than assumes, see F1); they differ in ORDER.  The short arm INTERLEAVES

    copy chunk 0, compute chunk 0, copy chunk 1, compute chunk 1, ...

while the folded path SEPARATES

    copy chunk 0, copy chunk 1, ..., | compute chunk 0, compute chunk 1, ...

Separating makes every chunk's shared buffer live at once.  With fewer shared slots than peeled
chunks, a later chunk's copy lands on a slot an earlier chunk has not been read from yet -- the
earlier read then reads the WRONG DATA.  That is not a hazard a fence can repair; it is a broken
def-use edge.

THE CHECK IS THE DEF-USE PAIRING, NOT A COUNTING RULE.  Observationally the coverage turns out to be
legal exactly when the shared ring is at least the peel depth, but this analysis deliberately does
NOT test that inequality.  It simulates both instruction sequences and compares, for every
consumer, WHICH producer's value reaches it -- classic reaching definitions, on two spaces:

    shared   location (operand, generation)                      written by a copy, read by a read
    register location (operand, group, slot, free-tile coord)    written by a read, read by an Mma

The register location names the free tile as well as the rotation slot because those are two
independent axes: `(group, slot)` covers the reduction, and the free coordinate (`M_inner` for A,
`N_inner` for B) selects a different physical VGPR that the model leaves to allocation.  Omitting
it merged `A(k1,m0)` with `A(k1,m1)` and made every read look like it clobbered its sibling (#227).

WHAT IS COMPARED: THE VALUE DELIVERED, NOT THE PRODUCER'S IDENTITY.  This is the correction #227
made, and it is the whole subtlety of the analysis.  The two arms are *deliberately different
schedules* -- the short arm runs at `dr = 0` while the drain keeps `dr = dr(theta)` -- so
they are not instruction-isomorphic and "the same instruction" is not a well-defined question.
What a consumer depends on is the DATA it receives, so each producer carries a `value`:

  * copy -> `(operands, coord, occurrence)`.  The k-th copy of an operand fetches chunk k in both
    arms, so the occurrence index IS the chunk.  This is what catches a shared ring too shallow to
    keep every peeled chunk live: the folded path's read finds chunk 1's copy where the arm's finds
    chunk 0's.
  * read -> `(operands, coord, source generation)`.  Two reads moving the same data out of the same
    buffer into the same register are the SAME VALUE wherever they sit.  This is what stopped the
    prologue's read-ahead fill from registering as a break: it is byte-for-byte the arm's own first
    read, merely in a block that has no peel-step index.

CONSUMERS are still corresponded structurally -- `(kind, step, operands, coord)`, checked unique --
because a consumer must be paired with its counterpart before its inputs can be compared.  The
position within the step is deliberately NOT part of that key: it was meaningful while both arms
were the same schedule and became meaningless when the reset made them differ.

An EXTRA is an instruction with no consumer key (a prologue fill; the drain's forward prefetch for
the next tile).  Extras are counted, never penalized -- their values participate like any other.

WHAT IT DOES NOT DECIDE.  The short arm's steps carry `T > t` guards; the folded drain chain has a
single entry and runs all M steps.  Serving `T < M` through the folded path therefore REQUIRES the
consumer to enter the chain at step `M-T` rather than at step 0 -- TensileLite's `NoGlobalLoadLoop_k`
selection does exactly that, which is why the scaffold gets away with the coverage today.  That is a
real precondition, so it is carried out as an `obligations` entry on the verdict instead of being
quietly assumed; see G3 / task 148 for making the multi-entry drain GIR's own.

```

## swap_regions.py:204 _edge_delta

```
Chunks the reduction timeline advances crossing `src`->`dst`.

    The difference of the endpoints' `chunk_base`, plus the trip advance on a back edge (the next
    trip's first body block sits `adv` chunks past this trip's first).  This is the conversion
    factor between the two blocks' frames; without it a steady value and a drain value are
    incomparable integers.

    PER PATH, not per block (#233).  `dst` may sit at a different position depending on which edge
    reached it -- the folded `T < M` entry lands in `drain0` at chunk 0 while the loop path arrives
    one chunk past the last steady trip -- so the destination's base is read from
    `path_chunk_base[src]` when it states one.  Subtracting two per-block integers instead put the
    two predecessors of the folded `drain0` one generation apart, which the solver correctly met to
    TOP and reported as an unsplittable critical edge: a real conflict computed from a frame that
    edge does not have.
```

## swap_regions.py:151 _unemitted_edges

```
Edges into a block NO BACKEND EMITS -- the only paths the placement problem may exclude.

    REPLACES `_loop_bypass_edges` (#222/#234), which recognised "a `CondGoto` whose taken side
    enters a loop header" and dropped the other side.  Two things were wrong with that:

      * it was a CFG-shape PATTERN MATCH, not a derivation, and
      * it excluded the FOLDED `T < M` entry too -- an edge the emitted code really does take, into
        a `drain0` the backend really does emit.  Excluding a live path is not a scope decision; it
        is a hole, and it is the one #222 exists to close.

    The criterion now is the DECLARED fact: `Block.model_only` (#232), set by `FoldShortPathPass`
    on a KEPT `T < M` arm and cross-checked by `verify_gir`'s G-EMIT.  Those blocks are the ones
    TensileLite serves through its own `toPGR1` path, so their requirements are demands of a path
    GIR does not generate and must not constrain placement by.  A FOLDED arm has no such blocks --
    they were deleted -- so its entry edge is now modelled, which is exactly the intended change.

    Why this is not the old exclusion renamed: it turns on a property of the DESTINATION that a
    pass states and the verifier checks, not on the shape of the branch; it is empty for every
    folded program; and swapping it in changed the emitted content of 0 of 210 swept configs, so
    no instruction moved -- what moved is which edges the analysis is allowed to ignore.

    `headers` is accepted and unused, so the two call sites keep one signature.
    
```

## swap_regions.py:3 <module>

```

SwapRegions -- physical-pointer swap placement, as a REAL forward
dataflow analysis over the GIR CFG.

The physical LDS read pointer (LocalReadAddr) and the TDM write descriptor are SINGLE registers
whose selected generation must EQUAL the generation every access names. A swap is the only
instruction that changes one, so "where do swaps go" is a reaching-value problem: propagate the
pointer's value to every program point, and a swap is a DEF inserted exactly where the value the
next access requires differs from the value that reaches it.

WHY THIS IS A DATAFLOW AND THE PREVIOUS VERSION WAS NOT. The previous implementation used the
vocabulary (need_in / need_out / "edge-based") but none of the mechanics, and the two defects it
shipped are direct consequences:

 * It had no lattice and no fixpoint -- one pass, no worklist.
 * `need_in`/`need_out` were per-block SUMMARIES read off the first/last access, and the swap
 POSITIONS came from a separate hand-written table of rules keyed on edge type. Two objects
 that can disagree, and did: the back-edge rule placed the copy swap before the block's first
 copy while the forward-edge rule left the pointer AT that copy's generation on entry, so every
 steady copy wrote one generation ahead of the buffer its Ref named. Here position and value
 are the same object -- a swap exists because the transfer function needed one at that point --
 so they cannot contradict.
 * A join took ONE predecessor (`max` by execution order, the "dominant path") instead of a MEET.
 That is not a conservative approximation, it is a silent wrong answer at any real join.
 * Requirements were read as absolute generations out of `gen_reaching`, whose base for a
 phi-less join block is 0 -- so a drain requirement (frame: offset from `T`) was compared against
 a steady one (frame: offset from the current trip) as if the numbers were commensurable, and
 the analysis "reconciled" a difference that was purely a change of frame. Here every
 requirement is normalized to its own block's frame via `Block.gen_rel`, and crossing an edge
 applies that edge's chunk delta (`Block.chunk_base`, or `path_chunk_base` where the block
 sits at a different position on that path -- #233), so all comparisons are frame-free.

THE ANALYSIS. Per (hop, unit) -- they are independent registers, so each gets its own run. A
`unit` is the pointer's IDENTITY: for the read hop an operand name (`'A'`), for the copy hop the
Phi movement's member tuple (`('A',)`, or `('A','B')` when the fuse aliases both onto one descriptor
-- multi-wave TDM). See `..refs.copy_unit` for why the copy hop cannot use a single
member's name. The lattice, the AVAIL/ANTIC fixpoints and the def-placement rule are
`value_placement`, shared with GrIncrementRegions; this file supplies the accesses, the frame delta
and the ring:

 domain Z_ring, plus BOTTOM (no value reaches yet) and TOP (predecessors disagree).
 requirement req(access) = `ref.gdelta - blk.gen_rel`, or `ref.abs_gen` for an absolute (peel)
 Ref -- the generation expressed in the ACCESSING BLOCK's own chunk frame.
 transfer walk the block's accesses in program order carrying the current value; at each
 access whose req differs, record a swap def THERE and adopt req. OUT = the value
 after the last access (BOTTOM-in propagates through a block with no accesses).
 edge crossing p->b re-expresses the value in b's frame: `val - delta(p, b)`, where
 delta is the difference of the two blocks' `chunk_base`, plus the trip advance
 (`GenXfer.adv`) on a back edge.
 meet agreeing values meet to that value; BOTTOM is the identity; disagreement is TOP,
 which is REPORTED, never silently resolved.
 entry the scaffold initializes both pointers to generation 0 before the kernel body.

Ring-1 (single LDS buffer) never changes -> no swaps at all, which falls out (every req is 0).
A-vs-B interleaving within one point is sigma (cost), not correctness; the R1 bar is numeric err=0,
not byte-identical.

```

## uniform_placement.py:138 comment run

```
                    # A block NO backend emits: no agent reaches the selector, so no agent can be
                    # brought to it and not another -- this is not the L733 failure.  It IS a
                    # SAFETY question (the hazard the fence covered goes unfenced on that path),
                    # and G-EMIT already makes the drop an explicit counted fact (#232), which is
                    # the right owner.  MEASURED 2026-08-21: raising here instead fails 3 tests in
                    # test_gir.py, all of them the kept `T < M` arm legitimately collecting fences,
                    # so folding safety into this liveness check would reject a shipped design.

```

## uniform_placement.py:3 <module>

```

UniformPlacement -- the ONE LIVENESS check the empty-ledger
gate does not subsume.

 "for each `scope=block`/`cluster` `Await` over agent set `G`, verify its region-tree position
 DOMINATES every agent in `G` uniformly -- it lies on a node no `Branch`-over-residues or
 agent-discriminating `Cond` arm can cause an agent of `G` to skip" (L733)

The safety argument is race-freedom plus residency, and the ledger gate checks that every
obligation carries a selector; neither says the selector is REACHED by every agent that must
rendezvous at it. A block-scoped selector placed on a path only some waves execute does not
race -- it HANGS.

WHAT `G` IS HERE. The Mark already names it: `scope`. A `scope=block` selector is realized by a
workgroup barrier, and a workgroup barrier rendezvouses EVERY wave of the workgroup, not merely
the waves that touch the buffers the cover selected. So `G` = all agents in the scope, which is a
SUPERSET of the hazards' participants -- quantifying over the realization's set rather than the
obligation's is what makes the check sound, and it is why no `agents` field is added: `scope` is
the agent set, and this analysis is its first real consumer.

THE TEST. A rendezvous is uniformly placed iff no execution can bring one agent of `G` to it and
not another. Agents diverge in exactly two ways expressible in GIR:

 (R) BLOCK ROLE (rho,). `Block.role != 'all'` means only those agents run the block.
 (P) AN AGENT-DISCRIMINATING GUARD. A terminator predicate whose truth differs per agent.

so the condition is: on EVERY entry->fence path, every block admits all of `G` and every guard is
agent-uniform. "On some entry->fence path" is the forward/backward reachability intersection --
`B` reachable from entry AND the fence's block reachable from `B` -- so this is a pure tree/CFG
test with no `sigma`, no counts and no dominator tree, exactly as L733 advertises.

A `model_only` block is deliberately NOT a violation here: nothing emits it, so no agent reaches
the selector and none can be split from another. That the covered hazard then goes unfenced on
that path is a SAFETY question G-EMIT already records (#232) -- see the note at the skip below.

HOW MUCH OF THIS IS LIVE TODAY, HONESTLY. (R) is inert: `Block.role` is `'all'` everywhere
because wave-specialization is unimplemented (task #165). (P) is inert BY VOCABULARY: `Pred` is
defined (nodes.py) as "a STRUCTURED symbolic predicate over trip-count symbols (no hardware)" and
`Bound` can hold only a trip symbol plus a constant, so GIR CANNOT SPELL an agent-discriminating
guard. That is not the same as the schedule satisfying the premise -- it means the tree-structural
test looks for a shape the tree has no node for. `prog.meta['agent_varying_symbols']` is the
declared escape: the first `Pred` that tests something per-agent must name its symbol there, and
this check fires from that moment.

AND THE PART THIS CANNOT SEE. The agent-discriminating arms that DO exist in the emitted kernel
(`s_bitcmp1_b32 s[sgprWaveIdx],0` / `s_cbranch_scc1 label_SkipStaggerA`, from
`KernelWriterAssembly._applyStaggerTDM`) are SCAFFOLD-owned and have no GIR node at all. So this
analysis is sound only relative to GIR's own tree, and the real deadlock surface is closed at the
emission layer instead -- see `Tensile/Lowering/uniform_emission.py`, which runs the same question
over the stream where those arms are visible. Both are wanted: this one guards the DECODER, the emission one
guards the BACKEND.

This analysis MUTATES NOTHING and returns `[PlacementViolation]`; `verify_gir`'s G-UNIFORM raises on it.

```

## value_placement.py:221 comment run

```
                # AN AMBIGUOUS VALUE NOBODY CONSUMES IS NOT A CONFLICT (#234).  `want` is the
                # ANTICIPATED demand -- this block's, joined with everything downstream of it -- so
                # `want is BOTTOM` means no consumer on any path from here ever reads this
                # register.  Two predecessors may then leave it at different values with nothing
                # able to observe the difference, and reporting that as an unsplittable critical
                # edge asks the caller to reconcile a value that is dead.
                #
                # Measured: the folded `T < M` entry made `drain0` a join for the GLOBAL-READ
                # ADDRESS, where the prologue leaves chunk 2 and the loop chunk 3 -- but the drain
                # issues no copies, so `antic[drain0] = BOTTOM` and neither value is ever used.
                # The LDS-pointer case at the same block is NOT this: there `antic` is a real
                # demand, and it is fixed by the per-path frame (#233), not by this test.  Both
                # halves are needed and neither hides the other.

```

## value_placement.py:3 <module>

```

ValuePlacement -- where the defs of a single-register resource go, as two real fixpoints.

Both pointer analyses solve the same problem for a different register: ONE physical register (an
LDS read pointer, a TDM write descriptor) must hold a specific value at each access, and a Mark
(`swap` / `gr_increment`) is the only thing that changes it.  "Where do the Marks go" is therefore
not a search -- it is the classic code-motion pair, and this module is the single implementation
both consume so the rule cannot drift between them.

    AVAIL  (forward)   avail_out[b] -- the value the register holds LEAVING b: the requirement of
                       b's last access, or a pass-through when b has none.
    ANTIC  (backward)  antic_in[b]  -- the value required ENTERING b: the requirement of b's first
                       access, or the meet over successors when b has none.

A def is needed exactly where supply meets a different demand:

    on edge p->b   iff  avail_out[p], re-expressed in b's frame, != antic_in[b]
    inside b       between consecutive accesses whose requirements differ

This is LCM's `EARLIEST = ANTIC and not AVAIL`, specialized to a single register (so there is no
expression to hash and no redundancy to eliminate -- only placement).

WHY EDGES, AND WHY THIS REPLACES A BACKWARD WALK.  The requirement is "exactly one def on every
path between the two accesses, and none anywhere else".  Placing on the EDGE makes that structural:
an edge is traversed exactly once per path through it, so no reasoning about domination or
double-emit is needed.  A hand-rolled backward walk that hops predecessors until it finds an access
has to prove those properties and cannot, because it has no fixpoint -- on any join or multi-path
region it silently answers for one path.  It also moves defs the value analysis already assumed
were elsewhere, so the emitted code stops matching the propagation that justified it.

The forward and backward solutions do NOT depend on each other, so there is no circularity: the
supply leaving a block is fixed by that block's own accesses, the demand entering a block is fixed
by its own accesses (or its successors'), and the edge def is what reconciles them.  In particular
`avail_in` is never needed -- after edge placement it is `antic_in` by construction.

CRITICAL EDGES.  A def on p->b is realized at the end of p when p has one successor, or at the head
of b when b has one predecessor.  When neither holds, the edge must be SPLIT; that is reported, not
resolved by picking a side, which would execute the def on a sibling path.

FRAMES.  Values are compared only after `edge_delta(p, b)` re-expresses them, so a block whose chunk
frame differs (drain vs steady) is directly comparable.  `modulus` makes the domain Z_ring for a
rotating pointer; `None` leaves it Z for a monotonically advancing address.

```

## analysis.py:3 <module>

```

GIR analysis INFRASTRUCTURE -- the base class + the lazy cached manager.

This module is infra ONLY. The concrete analyses each live in their own module under
`gir/analyses/` as an `Analysis` subclass:

 analyses/cfg.py Dominators, BackEdges (+ successors helper)
 analyses/gen_reaching.py GenReaching
 analyses/swap_regions.py SwapRegions (a region analysis -> PendingMarks)
 analyses/gr_increment.py GrIncrementRegions (a region analysis -> PendingMarks)
 analyses/dep_defuse.py DepDefuseAnalysis
 analyses/reg_band.py RegBandAnalysis

An Analysis is PURE (no IR mutation), lazy, and cached per (cache_key, prog.version); a Pass
invalidates the manager on mutation. An analysis fetches its dependencies through the manager
(`am.get(OTHER, prog)`), exactly as a StinkyTofu analysis pulls other analyses on demand.

```

## emit_plan.py:274 comment run

```
    # THE WMMA'S REGISTER SOURCES, AS A DERIVED LIST -- `(operand, group, slot, tile)` per src.
    #
    # The scalars below (`bufA`/`bufB`/`bufMXA`/`bufMXB`) are the LEAF's view: `gir_to_rocisa` maps
    # each to a named vgpr field of the instruction it builds, so a positional key is the right
    # shape THERE.  It is the wrong shape for anything that has to ask "which registers does this
    # wmma read?", because the operand->(buffer, tile) association -- which this function has just
    # DERIVED, by the axes it varies over, in `in0`/`in1`/`scale_of` -- is thrown away and has to be rebuilt from
    # the key names.  `verify_dataflow` did exactly that and got it wrong: it knew about `bufA`
    # and `bufB` only, so scale registers were never marked consumed (invisible at `W = R`, a
    # phantom OVERWRITE-BEFORE-USE as soon as the ring narrowed -- 24 of them on mxf8 `DepthU 512`).
    # Rebuilding it there by pairing `mma_scales` against literal `bufMXA`/`bufMXB` key names would
    # be the same association derived twice, the second time by convention.
    #
    # So it is carried. `tile` is the PARENT's grid index (a scale follows its parent's fan, which
    # is what `scale_of` established), so a consumer needs no name test and no ordering assumption.

```

## emit_plan.py:239 comment run

```
    # Per-operand register generation (buffer/slot) + size the wmma CONSUMES -- carried on the
    # src Refs by `loopir_to_gir._convert_mma` from LoopModel. The leaf
    # names its source vgpr from these, NOT the old `u % W`.  Operand identity is on the Ref's
    # tile.operand; A/B may differ under VgprPartition.
    # Which src is the first/second matmul input is DERIVED -- `mma_inputs` (in0/in1) above is that
    # answer.  This used to string-compare `tile.operand == "A"` / `== "B"`, in the module whose
    # own contract is "the axes it varies over-derived, no role literals": it silently produced (None, None) for
    # any kernel whose inputs are not literally named A and B.
    #
    # A SCALE OPERAND IS PAIRED BY PRESENCE, not by name.  A microscaling kernel adds register
    # srcs beyond the two matmul inputs (theta gives `MXSA`/`MXSB` their own paths), and which
    # input each one scales is exactly which input's FREE MODES it shares: a scale follows its
    # parent's free axis, so `free_modes[MXSA] == free_modes[A]`.  That keeps this module's
    # contract -- the axes it varies over-derived, no role literals -- and it is the same reasoning `mma_inputs`
    # already encodes for the inputs themselves.

```

## emit_plan.py:191 comment run

```
            # IS THIS READ A READ-AHEAD? `advance` is theta's pt5 shape on the node (0 = INPLACE,
            # issued for its own wmma; non-zero = HOISTED, fetching a later generation).  It is
            # carried so a checker can tell a refill from an ordinary read WITHOUT keying on the
            # `inplace-WAR` await -- a tag-keyed guard is disableable by the very misclassification
            # it exists to catch, which is exactly how a modulus change to `loads_in_place`
            # reached hardware and made bf16 ~4x worse while the suite stayed green (#332).

```

## emit_plan.py:160 comment run

```
        # ONE ACT, MANY LOCATIONS -- the coverage, projected (#312). A folded read is ONE
        # instruction (one act, at the carrier group's leader) that fills every coordinate of its
        # group, and `Ref.covers` is where `loopir_to_gir._absorbed` recorded that span.  The five
        # indices above are the LEADER's; `fills` is the same five for every covered coordinate,
        # computed by the SAME projections so a consumer can never disagree with the leader about
        # what one act wrote.  It is `((tile, region, tile_flat, k, k_flat), ...)` and its FIRST
        # entry is always the leader, so a consumer that wants "the act's coordinate" reads
        # `fills[0]` and one that wants "everything it wrote" iterates -- no branch on whether a
        # coverage is present, because for every ordinary read `covered_coords` is a 1-tuple and
        # `fills` is exactly `((tile, region, tile_flat, k, k_flat),)`.
        #
        # DERIVED HERE, NOT AT EACH CHECKER.  `check_source_coverage` and `check_register_dataflow`
        # both need it, and re-expanding a coverage at each of them is the two-derivations-of-one-rule
        # bug this codebase keeps hitting: they would agree on the unfolded case and diverge on the
        # folded one.  Projecting once, where the projections already live, is what makes the
        # register name (`tile_flat`) and the source coordinate (`tile_flat`,`k_flat`) come out of
        # one arithmetic.

```

## emit_plan.py:151 comment run

```
        # AND THE ABSOLUTE REDUCTION INDEX, which `k` is not once a region carries part of it.
        # This is the SAME projection `_plan_mma` uses for `u`, so a read and the wmma that
        # consumes it are directly comparable: the pair (tile_flat, k_flat) is the SOURCE
        # COORDINATE this read fetches, and (idx_p, u) is the one the wmma demands.  Carrying it
        # is what lets `verify_dataflow` check the register file without an emitter or a GPU --
        # every defect in the #236/#237/DU family was a read fetching one coordinate into the
        # register another coordinate's wmma then read.

```

## emit_plan.py:138 comment run

```
        # THE REGISTER INDEX IS THE FLAT ONE, AND THAT IS NOT THE SAME NUMBER.  A region split
        # displaces LDS STORAGE, not the register file: the wave still holds one vgpr group per
        # free tile, numbered across the whole tile.  So the address wants (region, within-region)
        # and the vgpr wants the flat index, and GIR supplies BOTH rather than letting the leaf
        # rebuild one from the other -- rebuilding needs tiles-per-region, which is exactly the
        # kind of geometry the leaf would have to re-derive from the scaffold and get subtly wrong.

```

## emit_plan.py:130 comment run

```
        # THE REGION IS PROJECTED OVER **ALL** ITS AXES, FREE OR REDUCTION.  This read
        # `freeset & rmodes`, which is the same thing only while every region axis happens to be a
        # free one -- true for the MT half (`M_split`/`N_split`) and false for the DU half, whose
        # axis is `K_split`.  The intersection then emptied and EVERY read came out `region = 0`:
        # measured 2026-08-18 on TN + `TDMSplitA/B = 2`, `k=0` and `k=1` both labelled `r0` with
        # k=1 at the ordinary unsplit stride, while region 1 sat 2176 bytes away.  100% failure,
        # both codegen arms, because the descriptor was right and only the reader was lost.

```

## emit_plan.py:104 comment run

```
        # THE REGION IS AN ADDRESS FACT, NOT ONLY A TOKEN (#237).  A region-split tile's storage
        # regions are DISPLACED in LDS (the copy walks the descriptor by one split boundary per
        # region, #236), so a read of a tile in region r is at `r * splitBoundary`, not at the
        # contiguous offset its flat tile index implies.  `tile` is therefore projected over the
        # free modes MINUS the region modes -- the WITHIN-REGION index -- and the region travels
        # beside it.  Projecting over all free modes folds the two into one number and there is
        # then nowhere for the displacement to go; that is why the read address had no region term
        # at all and a split tile's region-1 reads addressed region 0's bytes.
        # HAVING A REGION AXIS IS NOT HAVING MORE THAN ONE REGION, and this read is the last
        # consumer that had not been told.  `region_modes` lists the axes that COULD enumerate an
        # operand's regions; whether they DO is `operand_regions` (theta's `op.split`).  The copy side
        # has always been gated (`geometry.hop_broadcast` on `split > 1`) and so has the token side
        # (`tokens._region_key` on `unit_regions`) -- so an operand with the axis and one region
        # already moved as one and awaited as one, while its reads alone still split the coord.
        #
        # It goes wrong on the SHARED reduction axis, which is the only region axis two operands
        # can both name.  MEASURED 2026-08-18, `TDMSplitA=2, TDMSplitB=0`: `K_split` extent 2 sits
        # in BOTH operands' `region_modes` (the axis is the finest split; the unsplit operand is
        # simply present at every value), so B's reads came out `region in {0,1}` with `k` pinned
        # to 0 -- a region displacement into an image that has none, and the same half read twice.
        # 768/768 on both asymmetric DU cells, while the symmetric ones passed.

```

## emit_plan.py:3 <module>

```

Emit plan -- the PURE half of GIR -> rocisa (L3).

`plan_block(prog, phase)` walks a FINALIZED GIR block body (after the pipeline has applied swap /
gr_increment Marks and stamped tokens) and returns an ordered list of `EmitAction`s: the concrete
per-leaf instructions L3 must emit, with every index/generation/token already resolved from the
GIR nodes. It imports NOTHING from rocisa, so the index/order logic -- the correctness-critical
part -- is unit-testable standalone; the thin rocisa adapter (`gir_to_rocisa.py`) only maps each
action to the existing leaf emitter.

An EmitAction is one of:
 ('wmma', {idx0, idx1, u}) -- one WMMA tile
 ('read', {tc, tile, k, reg_buf, region, regions, token}) -- one operand M/N-tile ds_read at
 K-substep. `tile` is the WITHIN-REGION
 index and `region` carries the storage
 displacement (#237); together they name the
 address, which is why neither alone does.
 ('copy', {unit, gen, token}) -- one global->shared movement
 ('swap', {hop, gen_from, gen_to, operand|unit}) -- a buffer-pointer swap (Mark)
 ('gr_inc',{unit, chunks, to_chunk}) -- a global-read increment (Mark)
 ('region_inc', {unit, steps, from_region, to_region}) -- a TDM descriptor step BETWEEN the
 storage regions of one split tile (Mark)
 ('await', {unit, counter, gen}) -- a discharge boundary (Mark)

`unit` vs `operand` is not a naming inconsistency, it is the model. A READ belongs to one
operand, so it names an operand. A COPY is a Phi MOVEMENT -- possibly several
operandes merged into one cooperative instruction sharing one descriptor -- so it names the
member tuple: `('A',)` unfused, `('A','B')` when multi-wave TDM aliases both onto one descriptor.
A copy-hop swap and a gr_inc act on that movement's pointer, so they carry `unit` too; a read-hop
swap acts on an operand's LocalReadAddr, so it carries `operand`. See `refs.copy_unit`.

The 'read'/'wmma' index math mirrors the walker's _mmaIndices / _coordVal / _resolveBufferIdx,
but reads the CONCRETE coords GIR already baked in (no env, no Expr eval).

```

## nodes.py:394 Block

```
One software-pipeline Phase as a basic block in the GIR CFG.

 phase -- 'prologue' | 'steady{n}' | 'drain{n}' | 'short{n}' | 'tail' (also the block LABEL).
 'short{n}' is the `T < M` degenerate arm: present only until
 FoldShortPathPass has ruled, then either folded away or kept as a real second path.
 loop -- steady: its CondGoto taken-target is itself (a back-edge).
 phis -- Gen phis (loop header): v = phi(entry, v_next). List of GenPhi.
 xfers -- back-edge transfers: v_next = (v + adv) % ring. List of GenXfer.
 body -- [Move | Mma | Mark], program order, concrete coords.
 term -- the terminator: Goto | CondGoto.
 role -- wave role (rho); 'all' until wave-specialization.

 THE REDUCTION-CHUNK FRAME (`gen_rel` + `chunk_base`). A Ref's `gdelta` is an offset from its
 own block's chunk position, so two blocks' gdeltas are NOT comparable integers: a steady Ref's
 is relative to the current trip, a drain step's to `T`. Any analysis that reasons ACROSS a
 block boundary -- the physical-pointer flow, which must know whether the pointer state leaving
 one block equals the state the next block needs -- therefore needs the frame as a CFG fact:

 gen_rel -- the `rel` this block's Refs' gdeltas were computed against; None means the Refs
 carry ABSOLUTE generations (the prologue peel, fact). A frame-free
 requirement is `ref.gdelta - gen_rel`.
 chunk_base -- the block's position on the GLOBAL reduction-chunk timeline, in chunks, with the
 first steady trip at 0. An EDGE's chunk delta is the difference of the two
 endpoints' bases (a back edge additionally advances by the trip's chunk count,
 which the block's own GenXfer.adv already states).

 Both are set by the lowering -- the only layer that knows them -- and belong here rather than in
 a label-keyed side table, because they are per-block facts the CFG traversal reads.

 `chunk_base` IS A PER-PATH FACT (#233). A block reached by two paths can sit at two different
 positions on the reduction timeline, and one integer on the block cannot say so. The folded
 `T < M` entry edge is the case: on the long path `drain0` runs one chunk past the last steady
 trip (`chunk_base = 1`), but on the folded path it handles chunk 0 -- the position the short arm
 it replaced had. `path_chunk_base` lets a block state that per incoming edge; empty (the normal
 case) means `chunk_base` holds on every path. Deriving the edge delta by subtracting two
 per-block integers, with no way to express the exception, is what made the folded join look like
 an unsplittable critical edge when the two paths in fact agree.

 MODEL-ONLY BLOCKS (`model_only`). A block GIR must reason over but that NO backend emits. The
 kept `T < M` arm is the case: TensileLite owns that runtime branch (its own `toPGR1` path), so
 the arm exists here for the analyses and is not lowered. The flag is not decoration -- the
 emitter requests blocks by phase name, so an unemitted block is otherwise indistinguishable
 from one that was forgotten, and any Mark an analysis places in it is silently discarded.
 `verify_gir`'s G-EMIT makes that discard an explicit, counted, checked fact instead (#232).
 
```

## nodes.py:325 CondChain

```
Multi-exit terminator -- an ORDERED chain of guarded exits, then a default (G1).

    A loop header may need several typed early-exits taken BEFORE the loop is entered, all testing
    the SAME counter against different bounds: with a prefetch depth of `d` the trip count decides
    how much of the pipe can actually be filled, so a short trip enters a partially-drained variant
    instead of the steady loop.  That is a *chain* (`c <= d-1 -> ...`, `c <= d -> ...`, `c == 1 -> ...`,
    else fall through), not one two-way branch, and it is ORDERED -- the first satisfied arm wins,
    so the arms are a tuple and never a set.

    Modeled as one terminator rather than a fan of single-exit blocks so the header stays ONE
    block: its `phis`/`xfers` and its body belong to the header, and splitting it would force every
    analysis to re-join the pieces.  Each arm is an ordinary FORWARD edge as far as the CFG is
    concerned -- `successors` lists them in order, dominance and back-edge detection need no special
    case, and `verify_gir` checks every target exists like any other terminator.

    A one-arm `CondChain` is exactly a `CondGoto`; `CondGoto` is kept as the common shape rather
    than folded away, so the two-way branch (the overwhelming majority) stays the simple node.
    
```

## nodes.py:264 Trips

```
A loop's DYNAMIC TRIP COUNT: `(var - sub) // div`, or the constant `-sub` when `var` is "".

 WHY A COUNT AND NOT A COMPARISON (#231). A trip count is what the model actually states -- the
 model's steady region is a RANGE, `[delta, T)` -- while a comparison is one ENCODING of it, and the
 encoding needs three facts that live outside the node to decode: where the counter starts, which
 way it steps, and where the test sits. None of those are shared with the backend: TensileLite
 initializes its `loopCounter` to the loop count, DECREMENTS, and exits at
 `counter <= endCounter`. Same trips, opposite direction. Holding the comparison meant every
 reader re-derived the count from assumptions, and two off-by-ones came from exactly that -- the
 post-test bound (#228) and the guard strictness it forced.

 `div` is why this is not just `Bound` renamed. A multi-block steady chain (G2) consumes `div`
 chunks per trip, so it runs `(T - M) // div` times -- a statement `Bound`'s `var + const` form
 cannot make at all, which is why `G-TRIP` has to skip those loops today. Division makes the
 multi-copy case EXPRESSIBLE; whether the division is exact is a separate, stated precondition
 (see `reduction_coverage_violations`), not something this node pretends away.

 The exit predicate, the counter's direction and the test position all become EMISSION choices --
 made by the one place that knows which sgpr is live and what convention it already follows.
```

## nodes.py:162 comment run

```
# ===========================================================================
# 4.4  Mark -- a semantic point (+ the region-based insertion model)
# ===========================================================================
# The closed near-term Mark kinds; the verifier (G-MARK) enforces per-kind `at` payload.
#
# `await` replaces the never-created `barrier` kind.  GIR carries the MODEL's object -- a discharge
# boundary for one obligation -- not a target primitive; whether it becomes an `s_barrier`,
# a waitcnt, or nothing at all is L3's decision from the target config.
# Naming it `barrier` presumed the realization, which is exactly the presumption that is wrong on
# the SIA=4 path, where StinkyTofu re-derives every barrier and ours would be discarded.

```

## nodes.py:109 Move

```
THE data-movement verb: one hop down the memory hierarchy.

 srcs -- (Ref,) e.g. a global tile.
 dsts -- (Ref,) e.g. a shared tile at a Gen (a copy), or a register (a read).
 deps -- RAW + WAR edges carried from the LoopIR awaits (kind on each dep, / analysis).
 token -- the SEMANTIC completion token this Move produces/consumes, stamped by TokensPass
 (e.g. ('lds', generation) -- which LDS buffer a read consumes / a copy fills, so L4
 pairs read<->write; dep_defuse). None until TokensPass runs. A FACT, not a count.
 token_ids -- the BACKEND memory-token id(s) this Move's shared Ref touches, from
 `MemTokenAssignment` (#235). A token is an LDS pseudo-register, so these are the
 storage names the scheduler builds its def-use edges from; `token` above is the
 SEMANTIC completion class, which is a different question (one names WHAT COMPLETED,
 the other WHICH STORAGE). A TUPLE: an access that does not pin its region axis may
 touch several buffers and must carry every id, or the halves it does not name go
 un-ordered. Empty until TokensPass runs.
 advance -- pt5's SHAPE of a register read: 0 = INPLACE (issued for its own wmma),
 non-zero = HOISTED (a read-ahead fetching a LATER generation). Carried down from
 LoopIR's `Load.advance`.

 IT IS A DATAFLOW FACT, NOT A SCHEDULING DETAIL, which is why it belongs here. Without
 it GIR cannot tell a REFILL from an ordinary read, and no plan-level checker can see
 the one defect that matters for a read-ahead: a hoisted read emitted BEFORE the wmma
 that consumes the slot it overwrites. That is invisible to every COORDINATE-level
 check (`verify_dataflow`) by construction -- the refill writes the NEXT GENERATION of
 the SAME `(tile, k)`, so the coordinates match and only the generation is wrong.
 Measured (#332): a modulus change in `loads_in_place` produced exactly that
 inversion, passed 1137 unit tests + the 168-cell emit matrix, and made bf16 ~4x worse
 on hardware. The one test that polices this ordering keys on the `inplace-WAR` await,
 so the misclassification switched its own guard off.
 
```

## nodes.py:89 comment run

```
                                 # CONSUMES : read from the LoopIR slot Expr.mod, never
                                 # decided in GIR; reg_band only validates it.
    # -- the MOVEMENT QUANTUM footprint
    # `((axis, factor), ...)`: the axes this ONE reference spans and how many coordinates of each.
    # A wide `ds_load` is ONE definition (one instruction, ONE completion event -- N separate defs
    # would file N residencies against a single counter increment, the duplication #195 removed for
    # Phi-fused copies), but it FILLS several locations.  So the def stays single and declares its
    # footprint here; an analysis that tracks locations registers it at each covered coordinate
    # (`refs.covered_coords`).  Empty = the ordinary one-location reference.

```

## nodes.py:3 <module>

```

GIR nodes -- the GEMM-dataflow IR (layer 2), per the lowering-design 

GIR is the mutable optimization surface between the LoopIR (layer 1, LoopModel's proven-legal
rolled schedule) and rocisa (layer 3). It is a real, verifiable CFG whose ONE SSA value class
is the loop-carried LDS `Gen`. It carries GEMM semantics only -- never machine
realization (R-SEMANTIC): a swap is a `Gen`-transition FACT, not a `v_xor`; there is no
address / VGPR-color / vector-width node.

This module imports NOTHING from rocisa, so GIR + its analyses + passes are unit-testable
standalone. Every node is a plain dataclass.

The vocabulary :
 nouns Tile (a logical operand fragment), Gen (a loop-carried generation = the SSA value),
 Ref (a USE of a Tile at a Gen(+delta) or a register (group, slot))
 verbs Move (one data-movement hop), Mma (one matrix-multiply-accumulate)
 point Mark (a GEMM-semantic fact at a program point -- swap/gr_increment/gsu_guard/...)
 terms Pred / Goto / CondGoto (CFG terminators; the back-edge is a CondGoto found by
 dominance -- there is NO separate loop-back primitive,)
 cfg Block (one software-pipeline Phase), Program (the whole mainloop as one minimal CFG)

```

## base.py:34 StructuralPass

```
A pass that may CHANGE THE CFG: add or remove blocks, retarget terminators, move the test.

    WHY THIS IS A SEPARATE KIND.  Everything downstream of the CFG is derived from it -- dominators,
    back edges, generation reaching, pointer placement, the fence cover -- so a structural edit
    invalidates a different and larger set of facts than a body edit, and it can change the SHAPE
    the model is expressed in (a post-tested loop and a pre-tested one need different bounds for
    the same trip count).  `ApplyMarksPass` described itself as "the ONLY IR mutator" for a long
    time; `FoldShortPathPass` (#183) made that false when it started deleting blocks, and nothing
    in the type system noticed.  Naming the kind is what lets the contract below be stated at all.

    THE CONTRACT.  A structural pass MUST, before it returns:
      * leave `preds`/`succs` agreeing with every terminator (G-TERM), including for blocks it did
        not touch but whose predecessor set it changed;
      * leave the Program passing `verify_gir` -- in particular G-TRIP, which ties a loop's trip
        count to the drain's coverage, so a pass that rotates a loop or changes its trip count
        cannot quietly move the total work;
      * call `prog.bump()`, so every analysis cached against the old version is dropped.  The
        pipeline additionally clears the manager after each pass, but a pass that is run directly
        (as the tests do) has only the bump.

    WHAT IT MAY NOT DO: change what the kernel COMPUTES.  Moving instructions between blocks is
    allowed; adding or dropping a Move/Mma is not -- that is a model change and belongs in theta.

    Instances today: `FoldShortPathPass`.  Queued: loop rotation (post-test <-> pre-test, #228/#229)
    and critical-edge splitting (#222's kept shape).
    
```

## early_exit.py:3 <module>

```

EarlyExitPass (G1/G3) -- give the CFG the `T < M` entries the scaffold actually takes.

THE EDGE THAT WAS MISSING.  With peel depth `M` the drain is a chain of `M` steps, and a trip count
`T <= M` does not run the steady loop at all: it enters the chain ALREADY PARTIALLY DRAINED, at step
`M - T`, because the prologue only managed to fill `T` of the `M` generations.  `FoldShortPathPass`
records that as an obligation it cannot discharge ("entering the drain chain at step `M-T`,
G3/#148") and TensileLite implements it -- `_lmDrainStep = (PGR-1) - remainPgr` makes the NGLL
`drain0` and the NLL `drain1`, and the `toPGR1` label inside the NGLL sequence is the branch that
SKIPS `drain0` when the counter is 1.

GIR did not have that edge, and the omission is not cosmetic -- every analysis quantifies over the
CFG:

    MEASURED 2026-08-19.  `drain1.preds == ('drain0',)`, so `drain0` DOMINATES `drain1`, so
    `FenceRegions._already_fenced` elided `drain1`'s fence for the `prologue -> drain1` RAW (32
    cross-agent edges, present in `LdsHazards`) on the grounds that `drain0`'s fence stands between
    them on every path.  True in that CFG.  False in the kernel: at `T == 1` the emitted code jumps
    prologue -> drain1 and `drain0` -- the block holding the ONLY drain fence -- never executes.  The
    prologue's `tensor_load_to_lds` was then never ordered against `drain1`'s `ds_read`s, and MXFP8
    returned `-nan` on 20 of 23 failing cells, every one at `T < M`.

So this pass adds one arm per short entry, turning the prologue's two-way peel-validity `CondGoto`
into the `CondChain` G1 already defines for exactly this shape ("a short trip enters a partially-
drained variant instead of the steady loop ... `c == 1 -> ...`, else fall through").  With the arms
present `drain0` stops dominating the later steps and the fence cover reaches the same conclusion
the hardware does.

WHY IT RUNS BEFORE EVERY ANALYSIS.  A CFG edge added after an analysis has run invalidates it
silently -- the analysis was not wrong, it answered a different question.  Grouping this with the
coverage and the labelling in one CFG-shaping pass (see `scaffold_shape.py`) makes "the CFG is final
before anything reasons over it" a property of the pipeline rather than a rule to remember.

SHAPE.  For drain steps `drain0 .. drain{M-1}` and the existing guard `T > M -> steady`:

    arms    = ( (T == 1, drain{M-1}), (T == 2, drain{M-2}), ..., (T == M-1, drain1),
                (T >  M, steady) )
    default = drain0                                    # T == M: the full chain, unchanged

`T == M` keeps the old false-edge target, so the long path and the `T == M` boundary are
bit-identical to before; only the genuinely-shorter trips gain an edge.  At `M == 1` there is no
short entry (`T == 1` IS the default) and the pass is a no-op, which is why PGR1 never showed this.

```

## fold_short_path.py:144 comment run

```
        # THE FOLDED EDGE INHERITS THE ARM'S FRAME (#233).  Rewiring `-> short0` into `-> drain0`
        # does not merely change a target: the two land at DIFFERENT positions on the reduction
        # timeline.  `drain0.chunk_base` is the LONG path's (one chunk past the last steady trip);
        # on this edge the block does the work `short0` did, at `short0`'s chunk.  Recording that
        # is what keeps the join comparable -- without it the folded `drain0`'s two predecessors
        # come out one generation apart and the solver reports an unsplittable critical edge.

```

## fold_short_path.py:107 comment run

```
            # Both arms stay.  The kernel grows by one short region, which is the correct price for
            # a reordering that provably does not hold -- and it is only paid when the ring is
            # shallower than the peel depth.
            #
            # MARK THE ARM MODEL-ONLY (#232).  Nothing emits it: `_girEmitStage` requests blocks by
            # PHASE NAME and never asks for `short{i}`, so TensileLite's own `toPGR1` path serves
            # `T < M`.  That is correct today and it is also exactly the shape of a silent omission
            # -- the analyses go on placing Marks in these blocks (fences, measurably) and every one
            # of them is discarded without a word.  Declaring the blocks model-only and RECORDING
            # the discarded count turns it into a checked fact (G-EMIT), so the day a consumer
            # starts emitting the arm, the dropped state changes are already visible rather than
            # discovered on hardware.

```

## fold_short_path.py:73 comment run

```
            # TRIED AND REVERTED 2026-08-21: scoping this raise to non-`model_only` arms.  The
            # reasoning was that `short_loop['model_only']` says no backend emits `short0` (the
            # scaffold's `toPGR1` path owns `T < M`), so "emitting the arm would ship known-broken
            # code" cannot apply.  Two things were wrong with it.  (1) It is a WORKAROUND of exactly
            # the kind the last line of this message rejects -- the arm is still broken, and the
            # analyses that consume `short0` are consuming a wrong answer.  (2) It did not even
            # work: skipping the pass leaves the arm UNFOLDED, and the next verifier then reports
            # `G-OPERANDS: Mma src MXSB@register in steady has no producing Move`.  Folding instead
            # is the other workaround this message names.  The defect is the arm's MX read-ahead
            # warm-up in `emit.build_ir` -- #224/#225 fixed this shape for A/B and never covered the
            # scale operands.  Fix that; do not re-scope this raise.

```

## fold_short_path.py:3 <module>

```

FoldShortPathPass (#183) -- the MERGE pass: coverage the `T < M` arm into the shared prologue/drain
path when that is provably legal, keep it as its own path when it is not, and raise when neither
shape works.

WHY A PASS EXISTS AT ALL.  TensileLite's scaffold and the LoopIR describe the same kernel with
different STRUCTURE:

    scaffold :  prologue -> Cond -> loop -> else -> drain    prologue+drain SHARED by both paths
    LoopIR   :  Cond { prologue -> loop -> drain } else { short }      two DISJOINT arms

`loopir_to_gir` emits the LoopIR shape faithfully.  This pass rewrites it into the scaffold shape
*where the rewrite is sound*, so no duplicate code is emitted -- and leaves the two arms alone where
it is not.  The one thing it must never do is what the lowering used to: discard the arm and assert
the shapes match.  That assertion is false for 20 of 38 configurations.

WHAT IT DECIDES ON.  Nothing of its own -- `ShortPathFold` (analyses/short_path.py) compares the two
instruction sequences by REACHING DEFINITIONS on the shared and register spaces and returns a
verdict.  This pass is the mutator; the proof lives next to the CFG it is about.

    FOLD     every def-use pairing survives the reordering        -> delete the arm
    VACUOUS  M == 1, so the arm runs only at T == 0 (no reduction has that)  -> delete the arm
    SPLIT    a pairing changes under the folded order             -> keep both arms
    UNSOUND  the arm is not a valid program on its own            -> raise
    NA       no arm (M == 0, or already folded)                   -> nothing

THE RECORD IT LEAVES.  `Program.meta['short_loop']` keeps the arm's step count and guards (which
the scaffold needs either way) and gains the verdict, the reason, and the carried obligations.  A
consumer that requires one shape must READ that record, not assume -- `folded()` below is the query.
Obligations are carried rather than checked here because they are about code this pass does not
own: entering the drain chain at step `M-T` (G3/#148), and routing a kept arm's guard chain
(G5/#150).

ORDER IN THE PIPELINE: FIRST.  Every later analysis reads the CFG this pass rewrites -- swap and
gr_increment placement most of all, since folding restores the `prologue -> drain0` edge that
`_loop_bypass_edges` used to delete before their fixpoints ran (it is now
`_unemitted_edges`, which excludes only what nothing emits -- #222/#234).

```

## pipeline.py:151 _check_rotation_waw

```
The premise under which a rotation WAW needs no ledger row of its own.

    Non-interference covers WAW as well as WAR, so ordering a rotating buffer really means
    `WAR and WAW`.  Our ledger files only the WAR (`ledger.py`: RAW-residency,
    inplace-WAR/rotation-WAR, crossing-RAW, readahead-residency; `grep -rni waw Tensile/LoopModel`
    returns nothing).  That is CORRECT, but only under a premise the decoder never established:

        "Where every written generation is read and the read covers the write ... the WAW is
         implied by the WAR *through the residency RAW* ... so the WAW adds nothing there, and no
         second ledger row is filed.  It is stated only so `O(m)` is total without that decoder
         invariant."

    The two degenerate cases in which the premise fails, and the second row (tagged with the WRITE
    operand's class) becomes mandatory, are named there: (i) an UNREAD rotation buffer -- the WAR
    vacates because "last read of gen t" is undefined, yet the two writes still alias and
    program-order-on-issue does not order their COMPLETIONS; (ii) a read footprint NARROWER than the
    write, whose uncovered bytes are a pure write-after-write the WAR cannot see.

    MEASURED before this check existed: over 1632 rotating buffers across 240 configs -- the 168-cell
    order x split x prefetch sweep plus 72 MX configs (VW -1/1 x PLR 0/1/2 x PGR 1/2 x
    DepthU 128/256/512 x MXBlock 32/16, i.e. including n_s==1 and the coverage-bearing scale
    operands) -- case (i) occurred 0 times and case (ii) 0 times.  This turns "holds as far as we
    looked" into "holds, or the build stops".

    WHY IT LIVES HERE AND NOT IN `verify_gir`.  The invariant is a property of the FINAL program.
    `verify_gir` is also called mid-pipeline and directly on unpiped `lower_to_gir` output (e.g.
    `test_gir.py:147`), where a rotating buffer legitimately has writes and no reads yet; putting
    the check there failed 448 of 948 tests for that reason alone.  `run_pipeline`'s tail is the
    one place that sees the finished program.

    IF THIS FIRES it is not a bug in the check: a reachable theta has entered one of the two
    degenerate cases and the second ledger row becomes mandatory.  See #277.
```

## pipeline.py:78 _check_block_scope_covered

```
`Await.scope` is LOAD-BEARING here (#276-S2), and this is what makes it so.

 gives every obligation a proc-scope: "wave" where program order plus a per-wave counter
 already covers the edge, "block" where they provably cannot -- `s_wait_loadcnt` retires THIS
 wave's loads and says nothing about another's, so under a cooperative fill a reader waiting its
 own counter has no edge to the agent that actually wrote the bytes.

 UNTIL NOW THAT FIELD WAS INERT. It reached a diagnostic tag string and nothing else, so the
 model's claim about which edges are cross-agent was never confronted with the analysis that
 places the fences. The barrier that actually makes these kernels correct comes from
 `LdsHazards` -> `FenceRegions`, which probes LDS buffer overlap -- a DIFFERENT derivation of the
 same fact, and two derivations of one rule that never meet is the shape of most of the bugs in
 this subsystem.

 WHAT THIS CHECKS, AND WHAT IT DELIBERATELY DOES NOT. It does not re-derive separation: that
 would be a second separator implementation racing `FenceRegions`, exactly the failure mode
 above. It checks CONTAINMENT AT THE GRANULARITY THE DEP CARRIES. A dep names a producer
 operand and a completion class (`A@shared`), NOT an instruction -- the sigma-invariant object --
 and its consumer is frequently not an LDS access at all (an `Mma` awaits `A@shared` while
 touching only registers), so instruction identity is the wrong key and asking `LdsHazards` about
 it is a category error. The sound question at this granularity is: for every operand the model
 declares to have a BLOCK-scoped shared residency, does the hazard analysis agree that operand
 has any cross-agent shared edge at all?

 That is weaker than "this particular edge is separated" and is stated as such rather than
 dressed up: it catches the disagreement that matters -- theta calling an operand's fill cooperative
 while `LdsHazards` sees no cross-agent write of it, i.e. the two derivations of "is this
 cross-agent" diverging -- without pretending to re-establish the cover `FenceRegions` already
 owns.

 WHY NOT DRIVE EMISSION FROM `scope` INSTEAD. Measured 2026-08-25 over 32 configs (4 dtypes x
 single/multi-wave x PGR x PLR): multi-wave carries 21-76 block-scoped obligations and the cover
 discharges them with 2-6 fences; single-wave has 0 of each, and the correlation
 `blockAwaits>0 <=> fences>0` is exact in 32/32. So there is no under-fencing to repair, and
 realizing the obligations one-for-one would emit 4-12x the barriers -- #199's over-synchronization
 regression, which explicitly licenses collapsing under honored-`sigma_c`. Being the checked
 authority is the load-bearing role that does not cost that budget.
```

## pipeline.py:3 <module>

```

The GIR pass pipeline -- pass ORDER is DATA (the "backend" file, a la
StinkyTofu's Gfx1250Backend).

The ONE backend pipeline :
 ScaffoldShapePass the CFG is made FINAL here: coverage the `T < M` arm (#183), add the
 `T < M` early-exit edges into the drain chain (G1/G3), then attach
 TensileLite's scaffold labels. Everything after it is an analysis
 OVER this CFG, so an edge added later would silently invalidate them
 RegGenPass register (group,slot) authority; W validated from LoopIR (B3)
 TokensPass stamp each LDS Move's completion token from GenReaching
 CollectPendingMarksPass run region analyses (SwapRegions + GrIncrementRegions) -> PendingMarks
 PlacementPass pick a concrete anchor in each PendingMark.region
 ApplyMarksPass the only BODY mutator: insert each Mark before its anchor

There is a single pipeline (the "backend"). A test that wants to exercise one pass/analysis in
isolation runs that class directly (the analyses are pure) rather than assembling a sub-pipeline --
so no per-phase pipeline variants accumulate. R4 extends the collect set with GuardSite.

```

## placement.py:3 <module>

```

PlacementPass -- resolve each PendingMark's Region to a concrete anchor.

Policy travels with the Region (design "placement by swap TYPE -- a user directive, not just
sigma"), because it differs by what the Mark mutates:

 EARLIEST -- every TDM descriptor change (`gr_increment` AND the copy-hop `swap`): land immediately
 AFTER `region.after`, the access whose use of the descriptor this supersedes. Every
 cycle between the update and the `tensor_load` it feeds is overlap.
 MIDPOINT -- the LDS read pointer: land midway between the last use of `gen_from` and the first use
 of `gen_to`.
 MIDPOINT -- the `gl2_prefetch` pair (#216), for a DIFFERENT reason. The read pointer's window is
 a CORRECTNESS window bounded at both ends by real uses, and the midpoint is the
 safest point inside it. GL2's window is `BLOCK_ENTRY..BLOCK_EXIT` and legal
 throughout -- nothing in the trip consumes what it fetches -- so the midpoint is a
 BANDWIDTH choice: at block entry the prefetch block would contend with the trip's own
 TDM copy for memory-pipe slots, and mid-body is where that pipe has slack. It is
 therefore the one policy here worth MEASURING rather than settling by argument.

This used to return `region.before` for every Mark and let ApplyMarks insert before it -- the LATEST
legal point, under a method named `_earliest_legal`. It was only safe to fix once the windows were
real: `region.after` came from `ValuePlacementSolver`'s AVAIL/ANTIC fixpoints rather than being an
underived `BLOCK_ENTRY` placeholder.

ApplyMarks inserts BEFORE its anchor, so "immediately after node X" is expressed as "before the node
following X", and BLOCK_EXIT when X ends the body. IR-UNTOUCHED -- this only fills `pm.anchor`.

```

## record_unemitted.py:3 <module>

```

RecordUnemittedPass (#232) -- count what the model-only blocks are carrying away, LAST.

`FoldShortPathPass` decides WHICH blocks no backend emits (the kept `T < M` arm -- TensileLite owns
that runtime branch through its own `toPGR1` path) and records the reason.  It cannot record HOW
MUCH is discarded with them, because it runs FIRST: the fences and swaps that end up in those
blocks are placed later, by `CollectPendingMarksPass` / `ApplyMarksPass`.  Counting at coverage time
reported zero every time, and `verify_gir`'s G-EMIT caught exactly that -- 6 real marks against a
recorded 0.

So the count is taken here, at the end of the pipeline, and G-EMIT then holds the two against each
other.  The point is not the number: it is that a Mark landing in a block nothing emits can no
longer happen quietly.  Anything that mutates the Program after this pass and adds such a Mark
makes the recorded count stale, and G-EMIT fails rather than the state change simply vanishing.

Writes `meta` only -- no IR mutation, so it is an ordinary `Pass`, not a `StructuralPass`.

```

## scaffold_map.py:65 comment run

```
                # LABEL BY THE ARM'S TARGET, NOT ITS POSITION.  The point of this pass is to hand a
                # consumer the name TensileLite already uses, so where one exists it wins over the
                # generic role name -- position said `NoGlobalLoadLoop_0` for the branch the scaffold
                # itself calls `toPGR1`, which is the one spelling a consumer routing on the hint
                # would actually look for.
                #   deepest drain step  -> `toPGR1`   (its "PGR>=2 but only 1 loop" finalization:
                #                                      skip every NGLL, run only the last stage)
                #   any other drain     -> `NoGlobalLoadLoop_k`, k the STEP index it lands on -- the
                #                          same k `KernelWriter`'s `_lmDrainStep` computes, so the
                #                          two namings cannot drift
                #   the loop header     -> no label: TensileLite falls THROUGH into the body here
                #                          rather than branching, so inventing a name would imply a
                #                          branch target that does not exist.

```

## scaffold_map.py:3 <module>

```

ScaffoldMapPass (the LoopIR->TensileLite-scaffold mapping) -- the ONE place that maps the GENERIC,
backend-agnostic control flow (the peel-validity guard, the steady back-edge, the drain chain) onto
TensileLite's favored scaffold LABELS (`toPGR1`, `LoopEndL`, `NoGlobalLoadLoop_k`).

Why a pass, not the lowering: the pure LoopModel core emits only generic guard KINDS, and `loopir_to_gir` builds a
generic reducible CFG whose terminators carry LABEL-FREE `Pred`s. All TensileLite scaffold naming
is concentrated HERE, so a from-scratch backend that ignores the scaffold simply skips this pass,
and the coupling to TensileLite's openLoop/closeLoop label set lives in exactly one file.

The label is ADVISORY (`Pred.label`): a consumer whose own loop-count scaffold already emits the
compare (TensileLite openLoop/closeLoop) routes each arm to the matching label region instead of
re-emitting the branch; a from-scratch backend lowers `Pred.expr` and ignores the label.

Mapping (derived from the CFG shape, no per-kernel literals):
 - the block whose CondGoto is the LOOP HEADER's entry edge (prologue -> steady peel-validity):
 taken=steady, falls to the drain/short path -> label `toPGR1` (TensileLite's single-trip
 finalization: loop too short for a steady region).
 - the block whose CondGoto is the steady BACK-EDGE (taken-target dominates it):
 -> label `LoopEndL` (the loop-exit compare TensileLite's closeLoop emits).
 - each drain block (a straight-line NoGlobalLoadLoop / NoLoadLoop stage) has an unconditional
 Goto; TensileLite owns its label (`NoGlobalLoadLoop_k`), so nothing to map on the terminator.

```

## scaffold_shape.py:3 <module>

```

ScaffoldShapePass -- the ONE CFG-shaping stage: fold, then early-exit, then label.

WHY THESE THREE ARE ONE PASS.  They are the whole of "make GIR's CFG the CFG the scaffold will
actually run", and every later stage is an ANALYSIS OVER THAT CFG -- `GenReaching` merges over
`blk.preds`, `LdsHazards` pairs by reachability, `FenceRegions` covers "every path reaching the
consumer", `MemTokenAssignment` names buffers off the reaching generation.  An edge added after any
of them has run does not make that analysis wrong so much as make it an answer to a different
question, and nothing detects that: the result is simply stale in a way the version counter cannot
see, because the pass that would have bumped it ran too late.

That is not hypothetical.  The three steps used to be split across the pipeline -- fold FIRST,
labelling LAST, and the early exits nowhere -- and the missing `prologue -> drain{M-T}` edge let
`FenceRegions` conclude that `drain0` dominates `drain1` and elide `drain1`'s fence.  Correct over
the CFG it was given; wrong for the kernel, which at `T == 1` jumps straight to `drain1`.  MXFP8
returned `-nan` on every `T < M` cell.  Grouping the stage makes "the CFG is final before anything
reasons over it" structural instead of an ordering convention someone has to remember.

ORDER WITHIN THE STAGE, and why it is this one:

  1. FoldShortPathPass   decides whether the `T < M` arm survives at all.  It must go first because
                         it DELETES blocks; adding entry edges to blocks that are about to be
                         deleted, or labelling a terminator that is about to be rewritten, is work
                         thrown away at best and a dangling target at worst.
  2. EarlyExitPass       adds the `T < M` entries into the (now settled) drain chain.  After the
                         fold, because the chain it enters is what the fold left behind.
  3. ScaffoldMapPass     annotates the terminators with TensileLite's labels.  LAST because it
                         reads terminator SHAPE to decide the role (a `CondChain`'s arms are typed
                         early-exits, a `CondGoto` is the peel-validity guard), so it has to see
                         the final shape -- run before step 2 it would label a two-way branch that
                         no longer exists.

Each remains its own class and its own module; this composes them, so a from-scratch backend that
wants the fold and the exits but not TensileLite's label set drops step 3 and keeps the rest.

```

## tokens.py:126 comment run

```
                # ...and the BACKEND storage name(s) alongside it (#235).  Stamped here, next to
                # the completion token, because the two are read together and drifting them apart
                # is how the emitter ended up deriving one of them from a byte offset.
                # EVERY shared Ref, not just the one the completion token names.  A Phi-fused
                # `tensor_load_to_lds` carries one dst Ref PER MEMBER and fills both members'
                # buffers, so as an LDS producer it defs both; stamping only the first left the
                # other member's buffer with no producer edge at all.  The completion token above
                # is deliberately still ONE -- a fused movement is ONE completion --
                # which is exactly why the two cannot share a derivation: one names WHAT COMPLETED,
                # the other WHICH STORAGE.

```

## tokens.py:3 <module>

```

TokensPass -- stamp each LDS Move with its completion token.

Replaces the old walker's `_stepReadToken`: a read's `sync LDS%u` token must name the LDS buffer
GENERATION it consumes, so L4 (SIA4/StinkyTofu) pairs the read with the WRITE that filled that
buffer. The generation is a GIR FACT (GenReaching over the shared Ref), so the token is a direct
map -- no seed, no flip-on-change. A copy's token names the buffer it fills.

Token form: ('lds', completion-unit, (generation,)[, region...]) -- a semantic tag, NOT a hardware
token index (L3 maps it to memTokenLdsBuffer0/1). The generation tuple is a one-element MUST-set:
"this access touches this buffer". See `_buffer_generation` for why it was briefly a set of all of them. DepDefuse is consulted so a read with no RAW LDS
residency (a DTV global->register read) gets no LDS token. This pass MUTATES (stamps `Move.token`)
-> bumps version.

The COMPLETION UNIT is the operand, EXCEPT under fusion. a fused `Inst` "has a SINGLE
completion class C covering the whole group... so it is one completion event / one counter
increment, and any `Await` on a value the group moved resolves against that ONE completion";
The awaited unit is the coarser fused op. So when Phi merges several
movements into one cooperative instruction, the token names the GROUP, and a read of ANY member
resolves against it. Naming only one member (the first shared ref on the Move) left every other
member's reads awaiting a token that no copy in the program carried.

```

## quantum.py:276 comment run

```
            # THE STRADDLE. One instruction, two generations: the single completion cannot
            # discharge both residencies, so the merge is outside Pi.  Reported, not repaired,
            # because the repair is to NARROW the load (`Decision.narrowed`) and that needs a
            # narrower `ds_read` than `tP["localReadInstruction"]` holds -- an instruction-selection
            # question the caller owns.  Reporting it here is what turns the hand diff of an
            # emitted kernel into a build-time fact.

```

## quantum.py:266 comment run

```
        # THE LDS SIDE OF A STRADDLE IS A LABEL, NOT AN ILLEGALITY.  `_act_generation` carries two
        # facts and they have OPPOSITE remedies.  The register half (`reg_buf`/`group`) is
        # unencodable -- one instruction cannot write two `Valu*_X*` bases -- and stays a
        # violation.  The LDS half is only a question of WHICH producers this one movement must be
        # ordered against, and the honest answer is ALL OF THEM: a merged movement names a SET of
        # generations, so the instruction carries the union and StinkyTofu waits on both.
        # Narrowing is NOT the remedy (it would need an instruction the target did not select),
        # and refusing is not either -- the merge is what the target does.

```

## quantum.py:246 comment run

```
            # TWO DIFFERENT FAULTS ARE REPORTED BY THIS FUNCTION AND THEY ARE NOT THE SAME ONE.
            # This branch is an INCOMPLETE CARRIER GROUP: the acts a merged instruction would serve
            # are not all in this block.  That is an EMISSION/LIVENESS fact.  The branch below is
            # the GENERATION STRADDLE: one instruction, two register generations, which has no
            # encoding at all.  This message used to borrow the straddle's wording, and that
            # conflation misled FOUR separate
            # analyses on 2026-08-22, each reading a block split as the hardware hazard #274 fixed.
            # Worth stating plainly: `len(gens) > 1` below did NOT fire on a single cell measured
            # that day, so every violation anyone has actually seen from this function is THIS one.

```

## quantum.py:218 comment run

```
        # ONE ACT PER CARRIER GROUP, AT THE LEADER -- the invariant since theta models the coverage itself.
        #
        # This check used to count acts and demand `want` of them (the group size), because theta
        # emitted one act PER TILE and the LEAF folded them. That premise is now inverted: the
        # coverage is the modulus of the read's first-touch guard (`emit._first_touch_modes`), so theta
        # emits exactly ONE act per group -- the leader -- carrying its span in `Ref.covers` (#267,
        # "GIR emits ONE read act per instruction, carrying its tile span").  Counting acts then
        # reads a correctly-folded group as one short, which is what it did on NKM/NMK at split=2.
        #
        # The rule is STRONGER, not weaker, and catches both directions the count could not tell
        # apart: more than one act means the coverage was NOT applied (theta still modelling per-tile, so
        # the wide instruction would write a register a second act also claims), and an act at a
        # non-leader tile means the group's issuing member is missing.  The old inference of `want`
        # from "the acts present" -- the #286 blindness this message documents below -- is no longer
        # load-bearing, because the leader identity does not depend on how many acts are present.
        # THE DISCRIMINATOR IS A theta FACT, NOT THE COUNT.  A lone leader act is ambiguous from
        # inside this function: it is either a group theta FOLDED (one act carrying its span, #267)
        # or a group whose other acts WENT MISSING -- identical act counts, opposite meanings.
        # `folded_of(tc)` answers which, by reading whether theta folds this operand's read axis
        # (`geometry.hop_fold`).  It defaults to None so a caller that does not know keeps the
        # STRICT count rule and still reports a genuinely incomplete group.
        # and it is ONE ACT, not "one act at the leader": WHICH act issues is theta's choice (the
        # `mode % q == 0` guard picks it), so re-deriving a leader here from the INFERRED extent
        # `n = 1 + max(tile present)` is the second derivation this function's own note warns
        # about -- it disagrees with theta's guard wherever the inference is off, which is every
        # M/N-outermost order at split=2 (measured: 14 of 90).  theta folded it; one act is the group.

```

## quantum.py:176 comment run

```
        # theta'S CLAMP OVERRIDES THE SUPPLIED MAP.  The `QuantumMap` is a TARGET fact; whether a coverage
        # is LEGAL on this operand's axis is theta's, by the regularity condition ("`q` must divide
        # the axis extent ... an odd extent under a two-tile coverage clamps to `q=1`, i.e. no coverage ...
        # never emitting a ragged final group").  When theta clamped every axis away, the map is not a
        # merge theta agreed to, and grouping by its `carrier` partitions the acts into the ragged
        # groups theta REJECTED -- then reports the odd tail as incomplete.  MEASURED: MIWaveTile [8,7]
        # NKM/NMK at split=2, where `carrier = t//2` cuts the extent-7 axis into
        # {0,1}{2,3}{4,5}{6}.
        #
        # A CLAMPED-AWAY OPERAND GETS `WHOLE` (no Decision), NOT a permissive one -- the same
        # distinction the `WHOLE` sentinel exists for: "theta said nothing" leaves the leaf its own
        # coverage, which is the correct behaviour when theta declined to merge.  Skipping here is what
        # makes that fall out, rather than fabricating a singleton group to satisfy the check.

```

## quantum.py:150 plan_quantum

```
Decide, for every read act, whether it emits -- from theta's SUPPLIED merge, not from a rule here.

 `quantum_of(operand) -> ir.QuantumMap | None` is theta's own field, and `None` (the default) is the
 identity merge: one instruction per act, which is every A/B read. `extent_of(operand) -> int`
 bounds the tile scan for a carrier preimage; it defaults to the acts actually present.

 THIS DOES NOT MODEL THE FOLD, it OBEYS it. The act whose tile IS its own carrier emits; the
 rest ride it. There is no generation test and no narrowing: the merged coordinates are one
 the axes it varies over point by construction, so a straddle is not something this layer can observe,
 and re-deriving one here is what put a half-wave mechanism onto the region axis (see the header).

 What it still CHECKS is the model's named enforcement point -- that the acts a carrier group
 collects really are all present, so L3 is not handed a group with a member missing (the wide
 instruction would write a register whose act lives elsewhere). An incomplete group is a defect
 in what was supplied, not something to repair by narrowing, so it is reported.
```

## quantum.py:92 comment run

```
#: NO DECISION.  An act whose operand theta gave no merge gets `None`, NOT a permissive `Decision` --
#: the consumer must be able to tell "theta said every act issues" from "theta said nothing", because the
#: leaf's fallback coverage is only correct in the second case.  Returning `Decision(emit=True)` here
#: made the leaf take theta's branch and skip its own coverage, emitting one `ds_load` per tile where two
#: tiles share one: the whole mxf8 matrix failed (2026-08-20, 16/16, including the 10 that had been
#: passing).  A permissive default is not a safe default when the fallback IS the real behaviour.

```

## quantum.py:72 Decision

```
What `leaves.emitLdsReadTile` must do with one read act.

    `emit` -- issue an instruction (False = this act's payload rides a leader's instruction).
    `narrowed` -- the group was not mergeable, so the coverage was cut to one tile and this act emits
                 its OWN narrower load.  The leaf reads this to pick the narrowed read context.
    `reg_slot` -- the register-side index when it diverges from the tile index; None = the leaf's
                 ordinary derivation.
    `tokens` -- THE UNION OF THE LDS MEMORY TOKENS OF EVERY ACT THE INSTRUCTION COVERS, or () when
                 the group is single-generation and the act's own ids already say everything.  A
                 merged movement names a SET of generations (Sec 2.2): the one instruction really
                 does read both buffers, so it must be ordered against both producers.  Labelling
                 it with the leader's ids alone leaves the other half un-waited -- MEASURED
                 2026-08-19, `MXSA tile=0,k=0` token 9 folded with `tile=1,k=0` token 8 under one
                 `ds_load_b64`, the MXFP8 six-axis miscompare.
```

## quantum.py:56 comment run

```
# THERE IS NO `Span` HERE ANY MORE.  It carried `load` / `partitions` / `vector` and rebuilt the
# coverage from `ctx.tilePerRead` / `ctx.tileSpanVW` -- a SECOND representation of the same fact, which
# had to agree with theta's by hand.  The merge is now ONE object, `ir.QuantumMap`: two Exprs over the
# tile coordinate (`carrier`, `slot`), supplied from outside and evaluated everywhere.
#
#     carrier(t)   which instruction carries tile t   -> `group()` is its preimage
#     slot(t)      its position in that instruction   -> `regs()` counts the distinct values
#
# Load-vs-broadcast needs no field: a wide load gives each tile its own slot, a broadcast maps
# partners onto one, and `regs` counts the difference.  A composition of the two is a composition
# of the expressions.  So this module no longer models the coverage -- it only CHECKS that L3 emitted
# the coverage theta handed it.

```

## quantum.py:3 <module>

```

The MOVEMENT QUANTUM: which tiles one instruction carries, decided on the emit plan.

WHAT THE QUANTUM IS. Every hop moves a determinate block of data in ONE instruction, and the set
of axes that one instruction covers is the hop's **coverage**. By those axes are *absent* from
the hop's the axes it varies over, so the coordinates inside one coverage are ONE the axes it varies over point -- hence
`step(element, p)` gives them **one step, therefore one generation**. The model states this per hop
*symmetrically*: it governs a shared->register READ (a wide multi-tile `ds_load`) exactly as it
governs a coarse global->shared copy.

WHAT GOES WRONG WITHOUT IT. GIR emits one read act per tile -- that IS theta's coverage, one tile wide.
`leaves.emitLdsReadTile` then merges several acts into one wider instruction (the group-leader rule:
the MX scales of `tilePerRead` adjacent tiles are contiguous in LDS, so one `ds_load_b64` fills
both). That merge WIDENS the coverage behind theta's back, and it is a legal peephole only while every
act it swallows belongs to the same generation. Under a 6-letter `loop order` word -- which hoists
`M_split` OUTSIDE `K_inner` -- they do not:

 WHY IT IS WRONG IS A COMPLETION ARGUMENT, NOT AN ADDRESSING ONE. theta is address-opaque,
 so "one instruction, one base address" is not a claim this layer may make -- and it is false on
 this target anyway: MX TileSpan delivers different data to different lanes from one `ds_load`
 via a per-lane base. The model-level statement is that ONE MOVEMENT INSTRUCTION IS ONE
 COMPLETION EVENT: if the coordinates it covers have residencies on DIFFERENT producers, that
 single completion cannot discharge them -- the `Await` names one producing copy while the other
 half's data came from another. Which is testable with no address at all.

 MEASURED 2026-08-19, MXFP8 / KMKNMN / TDMSplitA=B=1 / PGR1 / PLR1
 MXSA X0 +0 -> LDS token 9 (reduction chunk v+1)
 +1 -> LDS token 8 (reduction chunk v) <-- ONE instruction, TWO producers
 MXSA X1 +0 -> token 8, +1 -> token 8 consistent
 Hardware: exactly 4096/65536 outputs wrong, all in accumulator C+8, m mod 64 in the eight EVEN
 rows of one half-wave, every n. GIR's SSA has four def points; the leaf emitted two and
 dropped two, so the surviving load was attributed to the wrong generation.

THE REMEDY IS TO NARROW, NOT TO REJECT. "When `loop order` genuinely wants two coordinates of a
spanned axis at *different* generations, the schedule is not thereby illegal; the **coverage must
narrow** -- `tile`-split the hop's data axis so each instruction carries one generation's worth (a
wide two-tile load becomes two single-tile loads). So the order is always honored; it is the *load
width* that gives way, exactly as `S >= delta` gives way by widening the buffer." So a non-uniform group
is repaired here (each member emits its own narrower load), not diagnosed and refused -- the same
repair-not-rejection discipline as the synthesized depth default.

This module is the model's named ENFORCEMENT POINT -- "every payload merged into one movement
instruction agrees on generation across the coordinates it spans" -- and it is PURE: it reads the
emit plan's act dicts plus one geometry callable, imports no rocisa, and is unit-testable without a
kernel. The geometry callable follows `verify_dataflow.check_address_keys`'s precedent: how many
tiles one instruction covers is an LDS-layout fact theta does not model and must not learn to.

```

## refs.py:106 covered_coords

```
Every coordinate ONE reference fills -- the coverage's ABSORBED AXES, re-inserted.

 `ref.covers` is `((axis, extent),...)`: the inner modes this one instruction SPANS, which
 `geometry.quantum_axes` has already removed from the read hop's the axes it varies over. Empty (the default)
 means the reference fills exactly `ref.tile.coord` -- every ordinary read.

 THIS IS THE SECOND HALF OF THE SUBTRACTION AND IS USELESS ALONE. Because the axis is gone from
 the read's the axes it varies over, the def's `coord` does not name it while every consumer's does; the cross
 product below is what makes the one def readable as filling all of them. Applying the
 subtraction with this missing is measurable, not subtle -- the short arm reports "16 consumers
 read a location the arm never writes" at the mxf8 triage shape.

 RE-INSERTION, NOT A PREIMAGE. An earlier form kept the axis in the coord and asked the
 `QuantumMap` for `{t : carrier(t) == carrier(self)}`. That is the same set, but it can only be
 computed while the coordinate still exists -- and the whole point of is that it does not.
 So the map decides WHICH axes are absorbed (in theta, once) and this expands over their values.

 ONE DEF, MANY LOCATIONS -- not many defs. The instruction is a single completion event, so the
 IR keeps a single `Ref`; what expands is the set of places an analysis considers written.
 Emitting one Ref per location instead files N residencies against one counter increment, which
 is the duplication #195 removed for Phi-fused copies (measured: 497 unit tests).

 TWO SHAPES, TOLD APART BY THE COORD. A coverage folds an axis of extent `N` into `N/q`
 carrier groups; full absorption (`q = N`) is only the degenerate case.
 * axis ABSENT from the coord -- full absorption. Re-insert all `N` values (the case above).
 * axis PRESENT in the coord -- a PARTIAL coverage. The read is issued once per carrier group by
 the `mode % q == 0` first-touch guard (`emit._first_touch_modes`), so the coord names the
 group's LEADER tile `g.q`, and the instruction fills `g.q... g.q+q-1`. Expand IN PLACE from
 the leader -- an offset, not a rescale, because the coord is already a tile index.
 Skipping the present case (which this did while the model had no partial coverage) leaves each
 consumer of a non-leader member reading a location no def covers -- the same "N consumers read a
 location the arm never writes" this docstring already names, now reachable from the other side.
 
```

## refs.py:59 copy_unit

```
`(members, refs)` for a global->shared copy Move, or `(None, None)` if `inst` is not one.

    `members` is the tuple of operand names the movement carries -- `('A',)` unfused, `('A','B')`
    for the Phi-fused AB group that multi-wave TDM realizes as one aliased descriptor.  `refs` is
    the matching tuple of shared destination Refs.

    Why the tuple and not `refs[0].tile.operand`: `loopir_to_gir._convert_load` builds one Ref per
    fused token, so a fused Move genuinely has several shared dsts.  Naming the movement by the
    first one drops the rest -- which happens to give the right INSTRUCTION COUNT when the
    descriptor really is aliased (the multi-wave TDM case), and is simply wrong the moment a fused
    group keeps two pointers.  The count being right by accident is not the same as the aliasing
    being modelled, and the pointer analyses key on this identity.

    A fused movement's members rotate in LOCKSTEP, because one instruction fills them all at once.
    That is checked here rather than assumed: if they disagree, "one pointer" is false and every
    downstream conclusion (one swap, one increment) is unsound.  See `_rotation_state` for why the
    check is on the rotation and not on the buffer identity -- the members DO name different
    buffers, and must.
    
```

## refs.py:3 <module>

```

refs -- the shared Ref accessors for a Move's hops.

Small on purpose, and shared on purpose: three modules need the SAME answer to "which operand(s)
does this Move's shared hop belong to", and each of them used to compute it by taking the first
shared Ref. That is wrong for a Phi-fused movement, which carries one Ref PER MEMBER.

The distinction this module encodes:

 a READ is per operand. A's read and B's read are different instructions at different
 addresses; their "unit" is a singleton by construction.
 a COPY is per MOVEMENT. Phi merges several global->shared operandes into ONE
 cooperative instruction with ONE completion class, so its identity --
 and the identity of the pointer it advances and swaps -- is the GROUP.

```

## region.py:37 Region

```
A legal placement window over one block's body (opaque ordinals).

    block  -- the block LABEL the placement point lands in.
    after  -- place strictly AFTER this anchor (a body node, or BLOCK_ENTRY).
    before -- place strictly BEFORE this anchor (a body node, or BLOCK_EXIT).
    policy -- where in the window to land.  It travels with the Region because it differs by what
             the Mark mutates, so a single global policy in PlacementPass is wrong:

               EARLIEST -- every TDM descriptor change (`gr_increment`, copy-hop `swap`).  The
                          descriptor feeds `tensor_load`, so every cycle between the update and the
                          load is overlap.
               MIDPOINT -- two users, and their reasons are NOT the same:
                          * the LDS read pointer, midway between the last use of `gen_from` and the
                            first use of `gen_to`.  Bounded by two DIFFERENT resources, so it is
                            not the descriptor rule and must not be slammed to either edge.
                          * the GL2 prefetch (#216), whose window is legal at BOTH ends -- midway
                            is a BANDWIDTH choice, not a correctness one.  It feeds nothing this
                            trip, so sitting next to the copy only makes it contend with the TDM
                            for memory-pipe slots.  This is the one policy here that is purely a
                            cost pick and should be measured, not argued.
    
```

## region.py:3 <module>

```

Region + PendingMark -- the analysis->placement->apply insertion model.

A `Mark` (swap / gr_increment / gsu_guard) has a LEGAL PLACEMENT RANGE, not a single correct
point: a read-hop swap gen_from->gen_to may sit anywhere after the last access using gen_from
and before the first access using gen_to. Baking a fixed body index into an analysis is
premature realization (the mistake the old walker made).

So GIR mirrors StinkyTofu's waitcnt/liveness shape in three layers:
 1. a region ANALYSIS returns a plan of PendingMarks (IR untouched);
 2. a PLACEMENT stage picks a concrete point inside each Region (IR untouched);
 3. a single APPLY pass inserts each Mark before its resolved anchor (the only IR mutator).

`after`/`before` are OPAQUE anchors: a body node (place strictly after / before it), or one of
the sentinels below (block entry / exit). A Region may span a block edge (after a node in a
predecessor, before a node in a successor) -- that is how a cross-block swap is expressed with
no single index.

```

## render.py:327 comment run

```
        # THE BLOCK'S OWN FRAME, and whether anything emits it.  Two blocks with identical bodies
        # are DIFFERENT programs when their generation frames differ, and `gen_rel` is what says a
        # block's entry generations sit on an arbitrary base (drain0's predecessors genuinely
        # disagree -- the residue is trip-parity dependent).  Without it a reader comparing two
        # blocks' `gen@N` tags is comparing numbers from different origins.
        #
        # `model_only` matters more: such a block is one GIR reasons over and NO backend emits.
        # Rendered identically to an emitted block -- and folded into `gir_counts` -- it makes the
        # dump's op counts disagree with the .s file for a reason the dump never states.

```

## render.py:268 comment run

```
            # SAY WHICH HOP, AND SAY NOTHING ABOUT THE COORDINATES.
            #
            # This line first read "AGENT-relative: wave w reads region w, so ONE agent's reads
            # name ONE region".  Both halves were wrong, and the dump printed directly beneath it
            # refuted the second: A's reads name BOTH `M_split0` and `M_split1`.  What
            # `region_agent_relative` records (translate.py:588) is that the read hop's ADDRESS is
            # chosen by the agent -- and `mem_tokens.py:152` then WIDENS such a read to every
            # region, precisely because the instruction may touch any of them.  The flag is set on
            # the READ hop only and never on the tdm copy hop, so an unqualified per-operand
            # "agent-relative" over-claims for the copy: measured 8 of 8 agent-relative operands
            # have DISAGREEING hops, and `render_geometry` prints the same operands as ASYMMETRIC.
            # Two dumps of one theta contradicting each other is worse than either omitting the fact.

```

## render.py:240 _theta_facts_lines

```
The theta facts the BLOCKS ARE DERIVED FROM, which the block listing itself cannot show (#287).

    A GIR block prints coordinates and operands.  Whether a given coordinate is a REGION, whether
    that region is the AGENT's rather than the coordinate's, which operands share one Phi-fused
    movement, and how many storage regions the movement actually has are all facts of the theta the
    program was built from -- invisible in the body, and each one changes how the body must be read.

    THIS IS NOT COSMETIC, AND THE COST IS ON RECORD TWICE.  `translate.py:519` carried a conclusion
    ("the read's region needs rho") justified by a `check_region_coverage` message; the check was
    asking a COORDINATE question about an AGENT fact, and it survived a full day of investigation
    because nothing in the dump said the operand was agent-relative.  #176 is the same failure at a
    different site.  A dump that omits the premises makes a wrong premise unfalsifiable.

    Each block below is printed only when it has something to say, EXCEPT the mode-extent line,
    which every kernel with an inner mode gets -- it is the radix table the rest of the dump's
    coordinates are read in, and there is no kernel for which that is uninteresting.
```

## render.py:206 comment run

```
    # The obligations are the part a reader debugging this path most needs: they are what the coverage
    # does NOT discharge and hands to the scaffold.
    #
    # QUALIFIED BY VERDICT.  `short_path.py:470` builds ONE constant obligation tuple BEFORE the
    # verdict is decided and hands the same tuple to FOLD / SPLIT / VACUOUS / UNSOUND alike, and
    # its text is about the FOLDED drain chain ("must be enterable at step M-T").  Printing it bare
    # on a KEPT program hands the reader an obligation that program does not have.  Fixing the
    # analysis to emit per-verdict obligations is its own task; until then the dump must not
    # present a folded-path obligation as if it applied here.  The KEPT arm's real obligation --
    # routing its guard chain, G5/#150, named at fold_short_path.py:33 -- is in no tuple and so
    # cannot be printed from `sl`; it is stated directly instead.

```

## render.py:192 comment run

```
    # PRINT THE VERDICT, do not assert one.  This used to hardcode "NOT lowered to GIR blocks
    # (#154)", which stopped being true at #183/#219-223 -- the arm IS lowered to `short{i}` blocks
    # and then folded into the drain chain when `ShortPathFold` says the def-use pairings survive.
    # A dump that states a stale conclusion is worse than one that states nothing: it was read
    # twice as evidence that the short path is scaffold-owned, sending two investigations of a
    # `T <= M` miscompare away from the folded drain chain that actually runs.

```

## render.py:113 comment run

```
    # BOTH, AND SPELLED APART.  These are two different questions and this used to be an `elif`
    # under one `tok=` prefix, so the semantic token was unreachable on any pipeline-built Program
    # (measured: 0 Moves lack `token` after TokensPass) and, had it ever fired, the reader could
    # not have told which quantity was on screen.
    #   ids  = the dense LDS BUFFER ids -- the alias relation StinkyTofu orders on (#235).
    #   token = the semantic COMPLETION CLASS ('lds', unit, region) the ids were derived from.

```

## render.py:27 _gen_str

```
The buffer this Ref names: a loop-carried Gen IDENTITY, an absolute generation VALUE, or a
    register rotation slot.

    THE TWO GENERATION FORMS ARE SPELLED APART.  They used to render as `gen0[v]` and `gen=0` --
    one character between an SSA identity and a concrete integer, in the same dump.  They answer
    different questions (which phi feeds this vs which buffer this trip lands on), and a reader
    scanning for one silently accepts the other.

    THE SLOT IS NOT AN ALTERNATIVE TO THEM.  A Ref can carry a generation AND a register
    residence; chaining these as `if/if/if` with early returns showed only the first, so a shared
    read's destination register was invisible whenever its source generation was pinned.  All
    present parts are now emitted.

    `reg_ring` (the rotation MODULUS W) is printed beside the slot: a slot without its modulus does
    not say how deep the ring is, so `s1` of a 2-deep and of a 4-deep ring are indistinguishable --
    and W is exactly the LoopIR fact GIR consumes and `reg_band` validates.
```

## verify.py:280 comment run

```
    # V4 -- EVERY READ'S TOKEN MUST HAVE A PRODUCER IN ITS OWN BLOCK, WHEN NOTHING FENCES IT.
    #
    # In a ROLLED body the copy that fills generation `v+1` and the read that consumes it are the
    # SAME static instruction one trip apart.  If their token ids differ, no def-use edge crosses
    # the back edge, StinkyTofu files nothing for a use with no def, and the derived
    # `s_wait_tensorcnt` comes out short by exactly the previous trip's copy -- a race that usually
    # resolves in time and so fails only on some machines.  `_unfenced_loop_carried_writes` is what
    # prevents it, by widening the WRITE end so every consumer's use is a subset of a producer's def.
    #
    # A FENCE CARRIES THE EDGE INSTEAD, so this only applies to blocks that have none -- the
    # single-wave case.  That qualifier is not cosmetic: without it this fires on multi-wave
    # kernels that are correct (measured: 258 of 288 f8 bodies flagged unqualified, 94 qualified).
    #
    # WHY THIS CHECK EXISTS: the regression it catches shipped once already.  `61eb5cfa` added the
    # widening; `0b8eb3c890b` removed it in favour of StinkyTofu's `EnableLoopCarriedTokenDeps`,
    # which CANNOT substitute -- loop-carried tracking orders ops that SHARE a token, and here the
    # producer and consumer ids differ by construction.  Nothing in the suite noticed, and it came
    # back as random numeric failures on another machine.

```

## verify.py:266 comment run

```
    # ...and the STAMP the emitter actually reads must match what the analysis derived, over EVERY
    # shared Ref of the instruction.  Checking only the analysis would pass while a pass that
    # dropped or truncated `Move.token_ids` shipped an unnamed access -- the analysis is not what
    # reaches the instruction.  Per INSTRUCTION, not per Ref: a Phi-fused copy fills one buffer per
    # member and must def them all.  Skipped before `TokensPass`, so a mid-pipeline verify still
    # means something.

```

## verify.py:233 _check_tokens

```
G-TOKEN (#235): the memory-token numbering is a correct naming of LDS storage.

    A token IS an LDS pseudo-register -- StinkyTofu gives a producer the token as a def, a consumer
    as a use, and a barrier as both -- so the numbering is not decoration: it is the alias relation
    the scheduler will believe.  Three properties, none assumed:

      V1  DISJOINT STORAGE NEVER SHARES AN ID.  True by construction (the key is
          (unit, region, generation)), and checked anyway, because the construction is the thing
          most likely to drift -- `_regions_of` gaining a case, a fuse key changing spelling.
          Sharing where disjoint is lost parallelism, not a miscompile, so it reports as a
          precision failure with both keys named.
      V2  COMPLETENESS.  Every shared access is named.  StinkyTofu's MemTokenConsistencyCheckPass
          requires that within a basic block ALL mem-token candidates carry a token or NONE do, so
          one unnamed access does not degrade that block -- it invalidates it.
      V3  FENCE COVER.  Every fence Mark carries the ids of the storage it stands between.  A
          fence whose token set misses an end is emitted, looks right, and orders nothing -- the
          exact shape of the TDMSplit half-1 hole (#217).
```

## verify.py:198 comment run

```
    # steady loops = blocks with a self/dominating back-edge.
    #
    # The TAIL loop (the `K % DepthU` remainder) is a SEPARATE loop region after the drain, with
    # its own header, counter and back-edge (G3) -- so it is a legitimate SECOND header, not a
    # violation.  It is excluded by PHASE, which is the structural fact ("this block is the tail
    # region"), so the reduction-body count stays exactly-one and a genuine second reduction loop
    # is still rejected.  Note a multi-block loop BODY (G2) does not add a header at all: the
    # chain shares one back-edge, so it never reaches this check.

```

## verify.py:3 <module>

```

verify_gir -- the GIR well-formedness verifier.

Checks the G-* invariants. Raises RuntimeError on the first violation with a precise
message; passing means the Program is a well-formed GEMM-dataflow CFG (R-LEGAL: entry is
legal, passes may create illegal intermediates but must restore verify_gir before the pipeline
ends).

Invariants :
 G-GEN-SSA every Gen has exactly one def (a phi or a GenXfer) AND a use; a def with no use
 within the modeled region is an error (a dropped consumer is the bug this catches).
 The one span that looked use-free -- the drain descriptor swap -- is scaffold-owned, NOT modeled as a use-free Gen, so it does not appear here.
 G-NOITER no `iter` in a Coord (the cross-trip relation lives in the Gen phi).
 G-SEMANTIC no physical field (address/vgpr color/vector width) on any node.
 G-CFG one prologue; M drains (M from the lowering, NOT approximated PGR-1); optional
 tail; in single-tile scope exactly one steady-loop (the clause relaxes to
 ">=1 loop" for persistent).
 G-TERM every Block has a terminator; every target exists; preds/succs agree; back-edges
 found by DOMINANCE (enumerate per-dominance-back-edge, not a single blk.loop).
 G-OPERANDS every Mma/Move src resolves to a Ref whose producing Move exists (MX/sparse
 operands not dangling).
 G-MARK each Mark.kind is in the closed set and its `at` payload matches that kind's
 schema (no free-form kinds, no missing fields).
 G-WALK the TDM REGION WALK CLOSES: a split tile's per-region descriptor steps net to zero in
 every block, so the plain chunk stride that follows is the whole advance. The bug
 this exists for had two owners walking one descriptor and nothing relating them (#236).
 G-TOKEN the LDS memory-token numbering NAMES STORAGE CORRECTLY: disjoint buffers never
 share an id, every shared access is named (StinkyTofu's all-or-none rule), and every
 fence carries the ids of what it stands between. A token is an LDS pseudo-register,
 so this numbering IS the alias relation the scheduler believes (#235).
 G-EMIT a block NO backend emits is flagged `model_only`, and every state-changing Mark
 discarded with it is COUNTED in `meta`. The emitter requests blocks by phase name,
 so an unemitted block is otherwise indistinguishable from a forgotten one and the
 Marks an analysis places in it vanish silently -- measurably: the kept `T < M` arm
 collects fences today (#232). This does not forbid the drop; it forbids the drop
 being invisible.
 G-UNIFORM a PROC-SCOPED SELECTOR IS REACHED BY EVERY AGENT IT RENDEZVOUSES -- the ONE liveness obligation the empty-ledger gate does not
 subsume. The gate certifies that each obligation CARRIES a selector; it says nothing
 about the selector being on a path every participating agent runs, and a block-scoped
 barrier some waves skip does not race, it HANGS. Tree-structural: no `role`-restricted
 block and no agent-varying guard may lie on any entry->fence path. (A fence in a
 `model_only` block is NOT this failure -- nothing emits it, so no agent is split from
 another; that drop is G-EMIT's, #232.)
 SCOPE OF THE GUARANTEE: sound over GIR'S OWN tree only. The agent-discriminating arms
 the kernel really contains (`s_bitcmp1_b32 s[sgprWaveIdx],0` /
 `s_cbranch_scc1 label_SkipStaggerA`) are scaffold-owned and have NO GIR node, so this
 check guards the DECODER and `Lowering/uniform_emission.py` guards the BACKEND.
 G-TRIP the steady loop and the drain COVER THE REDUCTION EXACTLY ONCE: steady covers
 [0, trips x chunks-per-trip) and the drain [T-M, T), and together they must be
 exactly [0, T). Symbolic in the trip symbol, so no trip count is needed. This is
 the invariant a STRUCTURAL pass (#229) is checked against -- it cannot quietly move
 the trip count -- and its absence is how #228 (a one-chunk overlap at every T) went
 unnoticed: the loop's shape was never related to the drain's coverage.

```

## verify_dataflow.py:315 check_refill_splits_consumers

```
A HOISTED refill must not be placed BETWEEN two consumers of the generation it overwrites.

    THE ONE DEFECT SHAPE NO COORDINATE CHECK CAN SEE.  `check_register_dataflow` verifies that the
    register a wmma reads holds the `(tile_flat, k_flat)` the wmma asks for.  A read-ahead writes
    the NEXT GENERATION of the SAME coordinate, so when it lands between two consumers of the
    current generation every coordinate still matches and only the DATA is a trip early:

        wmma idx0=0,idx1=0  reads A(g0, buf0, tile0)      <- consumer 1, current generation
        read A buf=0 tile=0 adv=2                          <- refill, NEXT generation
        wmma idx0=0,idx1=1  reads A(g0, buf0, tile0)      <- consumer 2 gets the WRONG trip

    One A register serves every `idx1` at a given `idx0`, so a broadcast operand routinely has
    SEVERAL consumers per slot; splitting that set is the defect.  `refill / use / use` is correct --
    there the refill is that generation's producer -- which is why ORDER ALONE does not decide it and
    the check has to look at the whole per-slot event sequence.

    KEYED ON `advance`, NOT ON THE `inplace-WAR` AWAIT (#332).  The existing ordering test
    (`test_shape_a_wrapping_refill_follows_its_consuming_wmma`) keys on the await, so a
    misclassification in `loads_in_place` switches its own guard off: a modulus change there
    passed 1137 unit tests and the 168-cell emit matrix, then made bf16ptr ~4x worse on hardware.
    `Move.advance` is a dataflow fact the classification cannot suppress.

    MEASURED against the gfx1250 bf16 matrix over 120 (order x split x PLR x wave-count) cells:
    116/120, with ZERO false positives -- it never accuses a configuration that runs correctly.  The
    4 misses are `MNK 0/1` and `NMK 1/0`, whose classes fail only PARTIALLY on hardware (16/96 and
    24/96), so for most cells in them "silent" is the right answer and the coarse label is wrong.
```

## verify_dataflow.py:268 check_address_keys

```
The ADDRESS-level check, which needs one geometry fact the plan does not carry.

    `check_source_coverage` proves the reads name distinct SOURCE coordinates.  That is not the
    same as distinct ADDRESSES.  The leaf builds an address from TWO coupled choices:

        region term :  `region * splitBoundary`  if the regions are PACKED, else 0
        coordinate  :  the WITHIN-REGION one     if PACKED, else the FLAT one

    and they must move together.  Take the region term away (contiguous) while keeping the
    within-region coordinate and there is nowhere left for the region to live, so every region
    `r > 0` lands on region 0's bytes -- with the plan still perfect by every other check.

    `packed` is a callable `operand -> bool` supplied by the caller, because whether a region is a
    separately packed LDS block is a property of the LDS LAYOUT (`region_split_is_packed` over the
    split axis and `UnrollMajorLDS`).  theta does not model addresses and must not learn to.

    `force_within` DECOUPLES the pair -- within-region coordinate with no region term -- which is
    exactly the pre-fix leaf.  It exists so a test can show this check catches the defect rather
    than merely agreeing with the fixed code.

    MEASURED 2026-08-18 on NT + `TDMSplitA/B = 2` (a DU split cuts the OUTER axis of
    `[unroll][free]`, so it is contiguous): every `r1` read repeated its `r0` address,
    1056 of 1056 ULM1 NT DU cells.
```

## verify_dataflow.py:190 comment run

```
    # AN AGENT-RELATIVE REGION IS COVERED BY THE AGENTS, NOT BY THE COORDINATE (#245).
    # This check reads one agent's plan and asks whether its READ COORDINATES name every region the
    # copy fills.  That is the right question only while the region is a property of the COORDINATE.
    # When `Hop.region_agent_relative` holds, it is a property of the AGENT: `LraTileAssignment`
    # offsets each wave by `strideWave`, so at `MIWaveGroup[2,2]` wave `w` sits ENTIRELY inside
    # region `w` and its reads name region `w` ALONE -- correctly.  The union over agents covers
    # every region; a single agent's plan never does, and demanding that it does reports a
    # correct schedule as under-coverage.
    #
    # `mem_tokens.py:152` already draws exactly this distinction on the ordering side ("AN
    # AGENT-RELATIVE REGION NAMES THEM ALL"), so this is the same fact applied to coverage rather
    # than a new claim.
    #
    # SCOPED TO UNDER-coverage.  OVER-coverage -- a read naming a region the operand does not have
    # -- stays a violation for every operand: that is a displacement into somebody else's bytes and
    # no agent distribution excuses it.

```

## verify_dataflow.py:115 comment run

```
            # EVERY REGISTER SOURCE, READ OFF THE ACT (`srcs`) -- not reconstructed here.
            # `emit_plan._plan_mma` already DERIVES which operand each source is and which grid
            # index it is addressed by (`in0`/`in1` from `mma_inputs`, and a scale paired to its
            # parent BY PRESENCE via `scale_of`), so it carries `(operand, group, slot, tile)`.
            # Rebuilding that association here -- pairing scale names against literal `bufMXA` /
            # `bufMXB` key names -- would derive one fact twice, the second time by convention, in
            # the module whose contract is "no role literals".
            #
            # It is also what was WRONG: this loop modelled a wmma's inputs as the two matmul GRID
            # axes, so MX scale registers were written and never marked consumed.  Invisible at
            # `W = R` (each substep owns a slot, so a slot is only rewritten with the SAME source
            # and the guard below suppresses it); a phantom OVERWRITE-BEFORE-USE as soon as the
            # ring narrows enough for slots to repeat -- 24 of them on mxf8 `DepthU 512`, against an
            # emitted order that is provably correct (`read k -> wmma u=k -> read k+2`).

```

## verify_dataflow.py:98 comment run

```
            # ONE ACT DEFINES EVERY REGISTER ITS CARRIER GROUP COVERS (#312).  A Phi-folded read is a
            # single instruction writing a RUN of registers, so the file must see all of them
            # defined here -- otherwise the consumer of a non-leader member reads a name nothing
            # filled, which is the USE-BEFORE-DEF the coverage produced before this loop existed.  The
            # register name is keyed on `tile_flat` (`_reg_name`), so the run is exactly the
            # covered coordinates' flat indices.

```

## verify_dataflow.py:3 <module>

```

Semantic verification of the EMIT PLAN -- "does this schedule compute the right GEMM?" -- checked on
the plan alone, with no rocisa, no assembler and no GPU.

`verify_gir` checks that the IR is WELL-FORMED (blocks reachable, terminators sane, Refs resolved).
This module checks that it is CORRECT: that every wmma reads the data its own coordinate names,
that every value a read fetches is used before it is overwritten, and that the reads of one steady
trip cover the operand exactly once.

WHY IT EXISTS.  Every TDMSplit defect of 2026-08-16..18 was found by running kernels on hardware
and bisecting a pass/fail matrix -- a build cycle each, and a numeric miscompare says only "wrong",
never "which read". Each one is a one-line violation of an invariant below:

  * #237 / DU region term.  The reads of a 2-region operand all came out `region = 0`, so region 1
    was never read and region 0 was read twice.  -> REGION COVERAGE.
  * DU register rate (`theta._rate_broadcast`).  `K_split` was excluded from the register rate, so
    W collapsed to 1 and both regions loaded into ONE vgpr generation: the second read overwrote
    the first before its wmma consumed it.  -> OVERWRITE-BEFORE-USE.
  * Asymmetric DU (`operand_regions`).  The UNSPLIT operand's reads carried `region in {0,1}` with
    `k` pinned to 0, so it fetched the same unroll half twice under two different region labels.
    -> REGION COVERAGE (a region the operand does not have) and SOURCE COVERAGE (a missing k).
  * K-innermost read-ahead peel (`prefetch_axis_mode`).  The prologue primed one substep where the
    steady body consumes two, so the first trip's `wmma u=1` read a generation nothing had
    written.  -> USE-BEFORE-DEF at the loop header.

THE MODEL.  A register is named exactly as `leaves.emitLdsReadTile` names it:

    (operand, group, reg_buf, tile_flat)          # Valu{tc}_X{reg_buf}_I0 + wtRegStride*tile_flat

and it HOLDS the source coordinate `(tile_flat, k_flat)` -- the operand's free-tile index and its
absolute reduction index, both mixed-radix projections theta already computes.  A `wmma(idx0, idx1, u)`
demands `(idx0, u)` from its first matmul input's register and `(idx1, u)` from its second.  So the
whole check is: does the register the wmma names hold the coordinate the wmma asks for?

WHAT IT IS NOT.  It does not check ADDRESSES -- whether `region r` really sits at
`r * splitBoundary` in LDS is `Lowering/lds_geometry` arithmetic, pinned by its own tests.  It
checks that the plan names a coherent set of coordinates; the geometry then places them.

```

## gir_to_rocisa.py:639 _token_ids

```
The memory-token id(s) this act touches -- GIR's own numbering (#235).

        A token is an LDS PSEUDO-REGISTER: StinkyTofu gives a producer the id as a def, a consumer
        as a use, and a barrier as both, then the def-use chain orders them.  So the id set IS the
        alias relation, and GIR is the layer that knows it -- `MemTokenAssignment` names every
        buffer `(unit, region, generation)` and `TokensPass` stamps the ids on the Move.

        THIS REPLACES A MUTABLE SIDE CHANNEL.  The ids used to reach the leaf through
        `writer.states.ldsTensorTokenIdx` / `ldsReadTokenIdx` -- one int, written here and read back
        later -- which is #200's recorded concern and the reason two refusals had grown here: a
        single int can express the buffer parity and nothing else, so a ring deeper than 2 and a
        per-region token both had to be rejected.  Neither is a real limit; both were limits of the
        channel.  Passing the tuple removes the channel and the refusals with it.

        Returns `None` when the act carries no ids, which tells the leaf to keep the scaffold's own
        behaviour -- that is what makes this non-breaking for every non-ULM caller.
```

## gir_to_rocisa.py:604 comment run

```
            # The movement IS split but this copy could not say which region it fills.  Emitting
            # anyway would issue a load against whatever region the walk happens to be on, so
            # refuse: `region_increment` skips these too, and a load with no walk behind it is
            # precisely the silent wrong answer #236 removes.
            #
            # THIS MESSAGE USED TO NAME A CAUSE IT DOES NOT DIAGNOSE, and that cost a day.  It
            # blamed "a Phi-fused group tiled on different region axes (#170)", which describes the
            # PRE-#282 state.  Since `_member_region_coord` (loopir_to_gir.py:249) re-keys each
            # member onto its own axis by the -move-9 pairing index, a mixed-axis fuse now
            # yields `region` 0/1 and NEVER None -- measured on every fused f8 cell.  So the fuse
            # cannot be what put us here.
            #
            # What actually reaches this line is a movement whose region axis is ABSENT from the
            # coordinate enumeration while `regions` still says it is split -- i.e. the two
            # derivations disagree.  The known producer is `translate`: `_keep` (translate.py:364)
            # and `aRegions`/`bRegions` (translate.py:554) BOTH filter on `extents[axis] > 1`,
            # while `Operand.split` is computed separately, so anything that drops the axis extent
            # to 1 removes the axis from `loop order` and from `region_modes` and leaves `split` at 2.
            # That is an inconsistency UPSTREAM of this file, it fires on unfused kernels too, and
            # naming Phi here sends the reader to the wrong module.

```

## gir_to_rocisa.py:574 _emit_copy

```
Realize ONE global->shared movement, for the buffer generation GIR's Move names.

        One call per Move -- so the prologue's M peel fills each emit, which a single pre-built
        module per operand could not do.  Everything else about the load (addressing, vector width,
        DTL/DTV variants) stays in the scaffold's `globalReadDo`, reached through the leaf.

        FUSED movement: emitted through the OWNING member's tP and nothing else, because the
        scaffold is already written that way -- `globalReadDo`'s `tc == "B"` arm is guarded by
        `if numWaves == 1`, so at NumWaves>1 the B call produces an empty module and the A call
        produces the one cooperative `tensor_load_to_lds` that serves both operands.  Issuing a
        second call for B would therefore be a no-op today and a double load the moment that guard
        moves; naming the movement (not its first member) is what lets this stay a single call by
        intent rather than by luck.

        On the TDM path (the only one `UseLoopModel` allows today) the destination buffer comes from
        the DESCRIPTOR, i.e. from the swap Marks -- `g2lBufIdx` is read only by the buffer_load /
        G2L-vgpr branch of `globalReadDo`, which TDM does not take.  So `gen` is inert here rather
        than "selecting the staging buffer" as this comment used to claim.  It is still passed,
        because a non-TDM UseLoopModel kernel would need it and silently dropping it would be the
        same class of bug as the `prefetchIndex` default; if one ever exists, `gen>0` names `G2LA2`,
        which is only allocated under UnrollLoopSwapGlobalReadOrder/DTV, so that path needs a real
        gate before it is used.
        
```

## gir_to_rocisa.py:522 _wrap_lead

```
The shift that puts the wrap compare in the frame THIS advance runs in.

        `tdmIncrementGir` wraps when `LoopCounterL + lead == StaggerUIter`, and
        `StaggerUIter = S + pf` with `S` the stagger start chunk.  The wrap must fire on the advance
        that takes the address from chunk `T-1` to `T`, i.e. advance #n where `S + n = T`.  Under
        GIR's copy-then-advance convention, advance #n in the prologue targets chunk `n` and runs at
        `LoopCounterL = T`; the steady advance in trip `v` is #(M+v+1) and runs at `T - v`.
        Solving `ctr + lead == S + pf` against `S = T - n` in each case:

            prologue, to_chunk = n <= M   ->  lead = pf - n
            steady                        ->  lead = pf - M - 1        (constant; the `v` cancels)

        which is `pf - min(to_chunk, M+1)`.  Verified by simulating the emitted advance/copy
        sequence for every `T` and every stagger start `S`: exactly one wrap fires and the copied
        chunks are a permutation of `0..T-1`.

        The value is DERIVED from GIR's own schedule -- `to_chunk` from the dataflow, `M` the peel
        depth -- not calibrated against the scaffold.  Three previous versions were calibrated
        (`M - to_chunk`, then `phase == prologue`, then `to_chunk == 1`) and each matched whichever
        PGR it was written against: PGR1 is degenerate here (M=1 makes the prologue and steady
        leads coincide at 0), so every rule that was right for PGR1 was wrong for PGR2.
```

## gir_to_rocisa.py:481 comment run

```
        # `prefetchIndex` shifts the StaggerU WRAP COMPARE: `tdmIncrementAB` adds it to the loop
        # counter before deciding whether this increment is the one that must wrap the staggered
        # address back to the tensor base.  A prefetched increment runs `prefetchIndex` iterations
        # AHEAD of the counter, so passing 0 for a peel increment puts the wrap on the wrong
        # iteration and the staggered address walks off the end of the tensor.
        # (It does NOT emit an SRD-limit clamp -- an earlier version of this comment claimed that;
        # the "Set limit to 0 for last PGR iteration(s)" code is in `globalReadDo`, on a path this
        # never reaches.  Verified by reading `KernelWriterAssembly.tdmIncrementAB`.)
        # `pfi` is a SCAFFOLD CALIBRATION quantity, not a theta one.  It exists only to line the
        # StaggerU wrap compare up with `StaggerUIter = S + PGR`, which KernelWriterAssembly
        # computes from its own pipeline convention -- so the right value is whatever the scaffold
        # would have used AT THIS POSITION, and now that GIR emits the prologue hand-off advance the
        # two conventions agree (`prologue: copy;inc`, `steady: copy;inc`, the ULM0 shape).
        #
        # The scaffold's rule is POSITIONAL and flat (`KernelWriter.py`: `pfi = 1 if PGR < 3 else
        # PGR-1` for the prologue group, and `globalReadIncrementAB(..., 0)` everywhere in the
        # steady body).  It does not depend on which chunk the advance lands on.  An earlier version
        # here computed `M - to_chunk` with a special case at `to_chunk == M`; that agrees with the
        # scaffold on every reachable configuration, but only by coincidence -- it is chunk
        # arithmetic dressed up as a derivation for a quantity that is not derived from theta at all.
        #
        # PGR>=3 is rejected in Solution.py and no hardware baseline pins it; the branch is written
        # to match the scaffold so that lifting that gate changes one place, not two.

```

## gir_to_rocisa.py:454 _emit_gr_inc

```
Realize ONE global-read increment for a Phi movement, via `writer.tdmIncrementGir`
        (s_add tdm+=inc, with the StaggerU WrapU cselect).  GIR OWNS the placement (a gr_increment
        Mark from the dataflow); the magnitude is L3's.

        `chunks` is how many reduction chunks GIR's dataflow says the address must advance here.
        `tdmIncrementGir` emits ONE DepthU stride, so anything other than 1 has no faithful
        realization -- emitting the single stride anyway would leave the address short and silently
        re-fetch a chunk.  Reject rather than under-advance.

        FUSED movement (multi-wave TDM): one aliased descriptor, so one advance, but two details
        differ and both are `tdmIncrementGir`'s to apply -- the stride comes from the per-wave
        `tdm{tcA}{tcB}Incs` rather than `GlobalReadIncs{tc}`, and the StaggerU wrap value is
        parity-selected between the two operands' `WrapU`.  The PLACEMENT and the wrap LEAD are
        identical to the unfused case, which is why this is one function with a peer argument and
        not a second emitter: `_wrap_lead` is derived from GIR's schedule and knows nothing about
        waves.
```

## gir_to_rocisa.py:379 _fused_tp

```
`(tP_owner, tP_peer)` for a Phi-fused movement, after checking the scaffold can realize it.

        The scaffold's fused form is ONE aliased descriptor: `tdm{tcA}Group0` serves both operands,
        each wave's copy of those SGPRs addressing its own, and wave parity selecting which
        (`KernelWriterAssembly.isTdmWaveSeparated`).

        THERE ARE TWO SUCH PAIRS ON A MICROSCALED KERNEL, not one.  `KernelWriter` calls
        `initTDMDescriptorWaveSeparated` (and `tdmGlobalOffsetWaveSeparated`) for A/B AND AGAIN for
        `tPA["MX"]`/`tPB["MX"]` when both `MXBlock`s are set -- the scales get their own aliased
        descriptor, on the same wave-parity rule.  This used to name `("A", "B")` as the only
        realizable fuse and refuse everything else, which was true when it was written and became
        false when MX arrived: MEASURED 2026-08-18, `GIR emitted a fused copy for movement
        ('MXSA', 'MXSB')` on the acceptance shape.

        theta can still express fuses the backend cannot -- `paired` ([[MXSA,A],[MXSB,B]]) crosses the
        two descriptor sets, and a fuse at single wave has no aliasing at all -- so the check stays;
        it is now "is this pair one the scaffold aliased" rather than a literal.
```

## gir_to_rocisa.py:311 _emit_fence

```
Realize a proc-scoped selector -- a real, memory-ordering barrier.

 GIR decided WHERE (`FenceRegions`' cover) and WHAT IT ORDERS (`buffers`); L3 supplies the
 instruction. `_syncThreads` is the scaffold's own emitter and issues waitcnt + barrier, so
 it has the STRENGTH demands -- the read's completion ordered before the refill's write.
 The model is explicit that a bare execution-only barrier does not qualify.

 Unlike the `await` boundary this replaces, there is no `_needs_own_barrier` gate: GIR owns
 LDS synchronization at every SIA now, because `postMainLoopBarrierCheckAndReset` is skipped
 for `UseLoopModel` kernels (it would otherwise strip these and re-derive from tokens). A
 single-wave kernel still emits nothing, but for the RIGHT reason -- the analysis finds no
 cross-agent edge, so `FenceRegions` produces no Mark at all, rather than L3 discarding one.

 The token carried is the MUST-set of buffers this fence orders. On a barrier that reading
 is already the established one (`_tailLoopBarrierTokens` lists every token for exactly this
 purpose), and `StinkyBuildImplicitDependencyPass` gives a barrier both src and dest per
 token -- which is what chains copy -> fence -> read through the SSA use-def graph.
```

## gir_to_rocisa.py:293 _tag

```
Stamp EVERY GIR-emitted act with a self-identifying comment in the emitted assembly.

        Two jobs, and the second is why this is not decoration:

        1. READABILITY -- the act's GEMM meaning at the point of emission (which tile, which
           K-substep, which register slot, which buffer generation), so the `.s` reads as the
           schedule rather than as an instruction soup.
        2. ATTRIBUTION -- every instruction GIR emits is now traceable to the GIR act that asked
           for it, so anything WITHOUT a `GIR[...]` tag came from the scaffold.  That distinction
           was previously only recoverable by matching emission ORDER against a GIR dump by hand,
           which is how a duplicated copy+increment at PGR=1 stayed invisible: the scaffold and
           GIR emit the same primitives, so the instructions are indistinguishable once emitted.
           With the tag, a duplicate is visible by reading the `.s` alone.

        Cheap to keep: comments carry no instructions and are dropped in release output.
```

## gir_to_rocisa.py:237 comment run

```
                # L3 DOES NOT GAIN THE 2-D CAPABILITY (#251): GIR now carries a per-axis step
                # vector, and a step that moves SEVERAL region axes at once has no realization
                # here -- `tdmRegionIncrementGir` walks ONE region along ONE axis, and the flat
                # `steps` scalar is only a byte displacement while a single axis is live.  Refuse
                # rather than emit `|steps|` single-axis walks, which is what the old scalar path
                # would silently have done for a `M_split -1, K_split +1` grid step.

```

## gir_to_rocisa.py:199 comment run

```
                # #216: GIR chose the POINT; the instructions are the scaffold's own, unchanged.
                # Increment BEFORE issue -- the pair is ordered internally (the clamp zeroes
                # GL2PrefetchInc<tc> near the end of the loop, then the adds apply it), and that
                # order is the one thing about GL2 this layer must not get wrong.
                #
                # `self.writer`, NOT `w`: `w` is the LEAF EMITTER, and these two are pre-built
                # scaffold Modules on the WRITER's `codes` (`_loopBody` rebuilds both per call,
                # just above the fork that reaches here).  This is the one act with no leaf of its
                # own, because GIR contributes a position rather than an instruction.
                #
                # `_append_tag`, NOT `_tag`: the floating comment `_tag` writes does not survive
                # `ScheduleIterAlg=4`, where StinkyTofu reorders the body and leaves the comment
                # marking whatever ends up beneath it.  MEASURED on the mxf8 TileSpan matrix --
                # with `_tag` here, 15 ULM1 x GL2 kernels emitted the prefetch block and not one
                # `.s` said GIR had put it there, so the only way to attribute it was to rebuild
                # with the analysis disabled and diff.  Every other act already uses the
                # instruction-attached form for exactly this reason.

```

## gir_to_rocisa.py:174 comment run

```
                # A MERGED MOVEMENT CARRIES EVERY TOKEN IT COVERS (Sec 2.2).  When the coverage
                # folds several acts into one instruction, that instruction really does read every
                # buffer those acts named, so it must be ordered against every one of their
                # producers.  `Decision.tokens` is that union, computed where the merge is decided;
                # the act's own ids are the leader's alone, and using them leaves the other half
                # un-waited (MEASURED 2026-08-19: `MXSA tile=0,k=0` token 9 folded with
                # `tile=1,k=0` token 8 under one `ds_load_b64` -- the MXFP8 six-axis miscompare).

```

## gir_to_rocisa.py:113 emit_block

```
Realize the finalized GIR block `phase` into a rocisa Module.

        A swap Mark is realized by calling the scaffold's swap PRIMITIVE keyed by the Mark's
        subject -- `writer.tdmSwapLdsOffset(kernel, tP)` for a copy-hop swap (s_xor tdm descriptor),
        `writer.localReadSwapOffsets(kernel, internalPointerSwap, tP)` for a read-hop swap (v_xor
        LocalReadAddr).  This is why one Mark -> one swap, with no doubling: the fork's pre-bundled
        A+B Module is NOT used; GIR's Marks map 1:1 to the primitive.  `tpByOperand` =
        {operand: tP} (incl. MX/meta later).

        The SUBJECT is per hop, and this is the model rather than an inconsistency.  A read swap
        and a read name an OP-CLASS (`operand`).  A copy, a copy-hop swap and a global-read
        increment name a Phi MOVEMENT (`unit`, a member tuple) -- one cooperative instruction over one
        descriptor, `('A','B')` when multi-wave TDM aliases both.  See `gir/refs.copy_unit`.

        Copies are realized PER MOVE through the copy leaf, like reads and wmmas -- there is no
        pre-built `copyByTc` module handed in any more.  That asymmetry was what kept the prologue
        and drain on the scaffold: their peels issue several copies of one movement into different
        buffers, which one module per operand cannot express.
        
```

## gir_to_rocisa.py:98 comment run

```
        # MICROSCALING SCALE TENSORS are ordinary operands here: theta gives `MXSA`/`MXSB` their own
        # paths, so GIR emits read acts for them exactly as it does for A and B, and they
        # need exactly the same two entries.  The tP is the scaffold's own (`tP["MX"]`, built by
        # `KernelWriter.getTensorParameters` for the `MXSA`/`MXSB` char), and the context comes
        # from the scale-specific builder because the scale tensor's LDS image is a different
        # geometry rather than a narrower version of A/B's.

```

## gir_to_rocisa.py:40 _register_depth

```
`{operand: W}` -- the register ring width the leaves use for their `m = u % W` FALLBACK, a
    GIR fact (`Ref.reg_ring`, validated by RegBand, B3).

    The fallback only runs when GIR did not supply a per-act slot, which for a GIR-planned wmma or
    read it always does; W is therefore a legacy path, not the authority.

    PER OPERAND, because two different disagreements were being conflated.  RegBand keys widths by
    `(operand, group)`, and this used to collapse them to ONE set and raise whenever it held more
    than one value -- which is right for groups WITHIN an operand (mxfp8 A=(lo,hi): lo rotates a
    2-ring, hi is in-place; no scalar addresses both, and that is #118) and wrong ACROSS operands,
    where differing widths are ordinary.  MEASURED 2026-08-18: an MX kernel at PLR0 has A rotating
    2 slots and MXSA 1 -- the scale ring is derived from the scale's own fragment, not inherited --
    so every such kernel was refused with a message about multi-group addressing it does not have.

    So: raise only when ONE operand's groups disagree, and return the map.
```

## gir_to_rocisa.py:3 <module>

```

gir_to_rocisa -- the layer 2 -> 3 lowering, the THIN rocisa adapter.

L3 consumes the finalized GIR (after the pipeline applied swap/gr_increment Marks and stamped
tokens) as a flat EMIT PLAN (`gir.plan_block`, pure) and realizes each action into rocisa by
calling the per-tile leaf emitters (`Lowering/leaves.py`, `LeafEmitters`). The order + all
indices are GIR's; this file only maps action->instruction.

Concern split : GIR decided WHAT/WHERE/WHY (a swap is a Gen-transition fact at a point);
L3 decides HOW (a swap Mark -> the scaffold primitive tdmSwapLdsOffset / localReadSwapOffsets; a
gr_increment Mark -> tdmIncrementGir). GIR is the single source of truth for swap + gr_inc
placement -- TensileLite issues neither on this path.

That includes the FUSED case. Whether a copy moves one operand or two is a GIR fact (the act's
`unit`), never re-derived here from `isTdmWaveSeparated(kernel)`: re-reading the kernel would put
the decision in two places, which is the coupling this layering removes. L3 only CHECKS that the
fuse GIR asked for is one the scaffold can realize, and raises if not.

This module imports rocisa (via the leaf emitters), so per the no-build/run rule the USER builds
and runs numerics.

```

## lds_geometry.py:246 region_bytes

```
Byte displacement from the operand's LDS base to the start of TDMSplit `region`.

    THIS IS NON-ZERO EXACTLY ON THE PACKED DIAGONAL -- when the split cuts the LDS image's INNER
    axis (`region_split_is_packed`) -- and the asymmetry is a property of the image, not a
    heuristic.  Which axis is inner is layout-decided, so each layout packs on a DIFFERENT split:

      * unroll-major, LDS = [free][unroll].  An MT split cuts the OUTER axis, so region 1 is the
        second half of the same array.  Its two `tensor_load`s reproduce exactly the contiguous
        image one unsplit load produces, and a read must address it exactly as it does unsplit --
        the global `tile_row` already lands in region 1's half.  Adding a region term there
        double-counts; measured 2026-08-16, it regressed 82 passing kernels (110 -> 192 failures).
        A DU split cuts the INNER axis and DOES pack: [region][free][DepthU/nsplit].

      * tile-major, LDS = [unroll][free].  Mirror image -- an MT split packs
        ([region][unroll][free/nsplit]), a DU split is the contiguous continuation.

    On whichever diagonal packs, two DENSE half-tile writes cannot reproduce the unsplit image:
    what the hardware writes is one own block per region, `tdmSplitLdsBoundary` apart.  So the
    region IS a base displacement, the row stride shrinks to the region's row length
    (`region_row_elems`), and the inner coordinate is in-region (`fold_inner_offset`, or `K_split`
    read straight off the GIR coord on the ULM path).

    `splitBoundaryBytes` is the writer's own `tdmSplitLdsBoundary` -- the same quantity the copy
    side walks by -- and is already PADDED, so callers add it after `_pad`, not before.
```

## lds_geometry.py:192 tile_row

```
The LDS ROW index of this operand's wave-tile `t`, in an unroll-major (DU-major) layout.

    THE DISTRIBUTION IS BY VECTOR GROUP, and the unroll-major path used to model only half of it.
    A wave takes `VectorWidth` ADJACENT tiles (adjacent rows), then the next group of tiles starts
    a whole `MIWaveGroupShape[tile01]` rows later -- that jump is what hands the intervening tiles
    to the other waves.  So

        row(t) = (t // VW) * MIWaveGroupShape[tile01]  +  (t % VW)

    and the old `tileStrideElems * t` is exactly the second term.  It agreed with the hardware only
    while every tile sat in ONE vector group (`MIWaveTile[t] <= VW`), which is what every test
    pinned -- `MIWaveTile == VectorWidth == 2` -- so the missing jump never showed.

    MEASURED, 2026-08-17, `MT32x32 KMN PGR2/PLR0 unsplit` at VW=1, A's `ds_load` offsets:
        scaffold (ULM0, passes) : 0, 32, 64, 96,  2176, 2208, 2240, 2272
        ours     (ULM1, failed) : 0, 32, 64, 96,   128,  160,  192,  224
    `row(1)` here is `1*16 + 0 = 16` rows -> raw `16*64*2 = 2048` -> `_pad` -> 2176, which is the
    scaffold's number; the old form gave one row, 128.  At VW=2 it gives `0*32 + 1 = 1` row = 128,
    which is what passed before, so that case is unchanged.

    Returned as a RAW row index: the caller multiplies by the row stride and applies `_pad`, so the
    block padding is added once, at the end, exactly as it was.

    THE VECTOR-GROUP JUMP IS SCOPED TO THE UNROLL-MAJOR PATH.  On the transpose path
    `tileStrideElems` is already `MIWaveGroupShape[tile01]` -- a per-TILE stride, not a row
    stride -- so the group jump is baked in there and applying it again would double-count.

    THE TRANSPOSE PATH DOES, HOWEVER, TAKE THE INDEX MODULO THE REGION.  See `region_bytes`
    for why the two layouts differ; the consequence here is that a tile-major region is a
    self-contained block whose tiles are numbered from zero, so tile `t` reads at in-region
    index `t % tiles_per_region`.  Unroll-major regions are a contiguous continuation of one
    numbering, so their index is global and untouched.
```

## lds_geometry.py:164 addr_coord_on_split_axis

```
Which coordinate the ADDRESS uses along the axis a TDMSplit cut: the WITHIN-REGION one when
    the regions are separately packed, the FLAT one when they are a contiguous continuation.

    ONE RULE, and the two halves are not interchangeable:

      * PACKED   region r is its own padded block, so the address is
                 `r * splitBoundary + within-region coordinate` -- `region_bytes` supplies the first
                 term and this returns the second.
      * CONTIGUOUS  the split's loads reproduce exactly the image one unsplit load produces, so
                 `region_bytes` is 0 and there is NO displacement to carry the region -- the
                 coordinate must therefore be the FLAT one, or region r>0 addresses region 0.

    The TILE axis already obeys this: `tile_row` returns the flat (vector-group) index on the
    unroll-major path and `t % tiles_per_region` on the packed one, and `region_bytes` is 0 in the
    former.  The REDUCTION axis had only the packed half, which is invisible while the only
    reachable packed diagonal is the DU one.

    MEASURED 2026-08-18 on NT (tile-major) + `TDMSplitA/B = 2`, where a DU split cuts the OUTER
    axis and is therefore CONTIGUOUS: every `r1` read repeated its `r0` address --
        unsplit   k=0 -> 0, 1536      k=1 -> 3072, 4608
        split     r0  -> 0, 1536      r1  -> 0, 1536
    because `k` had been folded to the within-region value while `region_bytes` correctly returned
    0.  Half the reduction was read twice and half never: 1056 of 1056 ULM1 NT DU cells.
```

## lds_geometry.py:137 fold_inner_offset

```
`(region, innerInRow)` -- split an INNER-axis element offset into the region it lands in and
    its position inside that region.  The counterpart of `region_row_elems`: that shortens the row,
    this says which row-block a coordinate past the end belongs to.

    A packed split makes the address PIECEWISE.  The unsplit image is one linear map
    `f*extent + u`; the split image is `region*splitBoundary + (f*(extent/n) + u mod (extent/n))`,
    and there is no linear expression for the second form because the region jump is a padded byte
    boundary, not an element stride.  Any emitter that builds the inner coordinate by accumulating
    (`rIdx * step + localReadOffset`) has to do this division somewhere; doing it here keeps the
    scaffold read path (`Components/LocalRead.py`) and the ULM read path -- which gets the same
    answer from GIR's `region_modes` instead of by dividing -- from disagreeing about the boundary.

    `extent` is the axis's FULL length, so `region == 0, innerInRow == innerElems` falls out
    whenever the split does not pack, and unsplit callers are unaffected.

    CALLER OBLIGATION: every element ONE instruction touches must land in one region.  This
    divides a starting offset; it cannot notice that a read spanning `w` elements straddles the
    boundary, and neither can it see the per-lane component that the base register adds.
    `Solution.py` rejects the shapes where `extent/n` is not a multiple of the read granularity,
    which is what makes the division sufficient.
```

## lds_geometry.py:111 region_row_elems

```
Row length, in elements, along the LDS image's INNER axis -- of ONE REGION when the split
    packs regions separately, of the whole tile otherwise.

    `extent` is that inner axis's full length, and WHICH axis that is depends on the layout, so
    the caller supplies it: `MacroTile` on tile-major (`[unroll][free]`), `DepthU` on unroll-major
    (`[free][unroll]`).  Both callers then get the same rule -- a packed region divides the row it
    lives in -- instead of the tile-major one being special-cased.

    IMPORTED BY `Components/LraTileAssignment.py`, which is the one place a layer below this one
    reaches up into `Lowering`.  That direction is deliberate and narrow: this module imports
    NOTHING, and the alternative homes are worse.  `Tensile/Common/` looks right by layering but
    its package `__init__` imports rocisa, which would make this arithmetic un-unit-testable in
    the pure-Python environment -- the very reason the module was split out of `leaves.py`.
    Duplicating the formula is what caused the defect below.

    The value is the distance between consecutive positions along the OUTER axis, and it is needed
    in two places that MUST agree: the per-read immediate offsets (`leaves.py`) and the per-lane
    base register's `kOffset * strideUnroll` / free-tile term (`LraTileAssignment`).  Those were two
    independent spellings of `extent + ldsPad`; shortening only the first left every lane with a
    non-zero outer component stepping a full-length row inside a half-length region.
```

## lds_geometry.py:34 read_fragments

```
The ds_readS ONE (tile, reduction-substep) local-read act decomposes into, as
    `((unrollElems, regOff), ...)` -- both relative to the act's own base, the first in ELEMENTS
    along the unroll axis, the second in registers.

    A lane's `inputPerThUnroll` elements for one substep are neither one contiguous run nor one
    instruction, and TWO nestings produce them:

      INNER `v` -- one `lrvw` chunk takes `lrvw / vwTrLoad` instructions, where `vwTrLoad` is how
                  many elements ONE ds_read moves: `blockWidth` registers of `bpeDS`-byte elements.
                  bf16 reads 8 elements (`b128`/`b128_tr_b16`, blockWidth 4) against `lrvw` 8, so
                  this loop is degenerate; fp8's transpose read is `b64_tr_b8` (blockWidth 2 = 8
                  elements) against `lrvw` 16, so it runs twice.
      OUTER `i` -- `inputPerThUnroll / lrvw` such chunks, and they are NOT adjacent: the lanes
                  covering this K range take one chunk each before this lane's next one, so
                  consecutive chunks are `LANE_INTERLEAVE` chunk-widths apart.

    ONE EXPRESSION FOR BOTH READ PATHS AND BOTH DTYPES.  The scaffold states it four times -- a
    `bpeDS`-per-branch walk under `enableLDSTr`, plus `calcGfx1250LdsOffset` for the general
    unroll-major path -- each with the numbers baked in (`numUnrolledIncrements = 32`,
    `vwTrLoad = 8`, `2 * (innerIdx + 2 * outerIdx)`).  Every one of those literals is a value of
    this formula, and the ULM read leaf needs all four cases from one place.

    MEASURED against emitted assembly of the passing bf16 matrix (2026-08-18):
        NT LDSTr MT32x32 DU128 : per act 0, 1536 bytes; substeps 3072 apart.  UnrollStride is 48
                                 elements (MT 32 + pad 16), so those are 16 and 32 elements, and
                                 32 == MatrixInstK.
        TN general MT32x32 DU64: per act 0, 32 bytes; substeps 64 apart at UnrollStride 1.
    Both have `nInner == 1`, where `i * LANE_INTERLEAVE * vwTrLoad` coincides with
    `inputPerThUnroll` -- the coincidence that made the old fixed two-fragment spelling look
    general.  It breaks as soon as `inputPerThUnroll / lrvw != LANE_INTERLEAVE`, which is exactly
    fp8 at `MatrixInstK` 128: four chunks spaced 32, not 64.

    Raises `NotImplementedError` when the shape does not decompose this way, and it checks TWO
    independent things, because either alone passes a wrong table:

      REGISTERS -- the fragments must exactly fill the per-tile register span
                  `inputPerThUnroll * bpeDS / BPR`: no gap (a register the wmma then reads
                  unwritten) and no overlap (a fragment clobbering its predecessor).
      K EXTENT  -- this lane's fragments, times the lanes it interleaves with, must cover the
                  substep's whole `MatrixInstK`.  This is the one that fails if `LANE_INTERLEAVE`
                  is wrong for a shape, and it fails with the numbers.
```

## lds_geometry.py:23 comment run

```
#: How many lanes share one range of K positions, so a lane's consecutive local-read chunks sit
#: this many chunk-widths apart along the unroll axis.  The scaffold spells it as a bare `* 2` --
#: "the WMMA V3 LDS layout uses a *2 factor on the unroll stride"
#: (`Components/LocalRead.py:calcGfx1250LdsOffset`) -- and it is also `WavefrontSize // matrixInstTO`
#: (32 // 16) for every shape the ULM read path accepts.  Kept as the scaffold's constant rather
#: than re-derived so the two cannot drift; `read_fragments` pins it against `MatrixInstK` on every
#: kernel, which is what would catch a shape where the two readings differ.

```

## lds_region_check.py:255 check

```
Run the check over one emitted kernel.

    `region_bytes` is `tdmSplitLdsBoundary` for the operand -- the SAME displacement the copy side
    advances by.  Returns (accesses, violations, notes).

    THE SEED IS THE HARDWARE REGISTER `v0`, not the symbolic `vgprSerial`.  The kernel's own first
    instruction is `v_mov_b32 v[vgprSerial], v0`, so seeding `vgprSerial` would OVERWRITE the value
    the slice is about to derive and, worse, would hide the fact that `v0` was never initialized.
    Seeding `v0` with the workitem id lets the slice compute `vgprSerial` itself, which is both the
    honest starting point and one less thing this checker asserts about the kernel.

    `sgpr_seed` supplies the KERNEL INPUTS the slice bottoms out at -- the workgroup ids in `s2`/`s3`
    and anything else the launch provides.  It is a required input rather than a default because the
    answer is PER WORKGROUP whenever the kernel remaps by workgroup id (`WorkGroupMapping`,
    `WorkGroupMappingXCC`): a verdict computed at one workgroup does not automatically hold at
    another, so the caller has to say which one it asked about, and should sweep several.
```

## lds_region_check.py:140 comment run

```
# INSTRUCTIONS WHOSE FIRST OPERAND IS A SOURCE, NOT A DESTINATION.  "First operand is the dst" is
# right for VALU/SALU and for loads, and WRONG for stores and prefetches: `global_prefetch_b8 v0,
# s[0:1]` reads its address out of `v0`.  Getting that backwards makes the slicer believe `v0` was
# redefined, so it stops following the real producer and silently returns a truncated slice --
# which then evaluates to a confident wrong address.  Listed explicitly rather than inferred from
# the mnemonic, because the naming is not regular enough to infer from safely.

```

## lds_region_check.py:3 <module>

```

LdsRegionCheck (#245) -- does one `ds_load` touch more than ONE storage region of a split tile?

THE HAZARD.  A `TDMSplit` tile lives in LDS as `nsplit` storage-disjoint regions, and GIR names the
region each read touches so that `MemTokenAssignment` can give it a token and `FenceRegions` can
order it against the copy that filled THAT region.  All of that assumes one read instruction sits
inside one region.  The assumption is not free: the region a lane reads is decided by the lane's
position along the split axis, and with an unlucky `MIWaveGroup` x `VectorWidth` x `MacroTile`
combination the lanes of a single `ds_load` -- or the different WAVES executing it -- straddle the
boundary.  When they do, GIR has named one region while the hardware touched two, so the access is
ordered against one producer and left un-ordered against the other.  That is a data race, it is
scheduling-sensitive (hence loop-order-sensitive), and it produces a silent wrong number.

IT IS INVISIBLE TO EVERY CHECK WE ALREADY HAVE.  Not to GIR: GIR's read carries the region as a
COORDINATE, so a read that spans two regions is simply mis-described, and no amount of dataflow
over that description finds it.  Not to the assembly either, read as text: the instruction carries
a wave-uniform `offset:` immediate and a base VGPR, so the lane spread is nowhere in the operand.
It is a property of the ADDRESS, which only exists once the per-lane setup code has run.

SO THE CHECK EXECUTES THE SETUP CODE.  `asm_lane_eval.LaneEval` interprets the emitted address
slice once per lane per wave, which yields the real byte address for every lane of every
`ds_load` -- with no re-derivation of `LraTileAssignment`'s rules, and therefore no possibility of
agreeing with a bug because the checker inherited it (see that module's header for why a Python
transcription of the address formula was rejected).

WHAT IT REPORTS.  Per `ds_load`, the distinct region indices touched WITHIN one wave and ACROSS all
waves.  Both matter and they are different faults:
  within a wave   one instruction's lanes straddle the cut -- the read cannot be named by any single
                  region, so the GIR model is not merely imprecise, it is unrepresentable.
  across waves    each wave stays inside a region but different waves pick different ones -- GIR
                  names one region for the instruction, so whichever wave landed in the other one
                  is un-ordered against its producer.

The region index is `(addr - ldsBase) // regionBytes`, with `regionBytes` supplied by the caller
from `tdmSplitLdsBoundary` -- the same number the copy side walks by, so the reader and the writer
cannot be compared against two different boundaries.

```

## leaves.py:747 emitCopyTile

```
Emit the global->shared copy (tensor_load / buffer_load) for ONE operand, into staging
        buffer `bufIdx`.

        The third leaf, alongside `emitLdsReadTile` and `emitWmmaTile`, and the one that was
        missing: copies used to be handed to L3 as a PRE-BUILT Module per operand
        (`copyByTc = {A: codes.globalReadA, ...}`) and emitted once per block.  That is why the
        prologue and drain could not be GIR-owned -- their peels issue SEVERAL copies of one operand
        into different buffers, and one pre-built module cannot be two of them -- and why
        `_emit_copy` needed a dedup guard to stop the single module being emitted twice.  Realizing
        a copy per GIR Move, exactly as a read or a wmma is realized, removes both.

        `memToken` -- the LDS memory-token id(s) this copy FILLS, from GIR (#235); `None` keeps the
        scaffold's own `setMemToken` calls.

        `regions` -- how many storage regions the movement is split across (#236).  At `1` this is
        the ordinary `globalReadDo`, unchanged for every caller.  Above 1 it takes the extracted
        single-region primitive instead, because `globalReadDo` emits EVERY region in one call and
        walks the descriptor between them internally: called once per GIR copy Move it would issue
        `regions^2` loads and leave the descriptor `regions-1` steps ahead each chunk.  GIR emits
        one Move per region and places the walk itself, so the loader here must load exactly the
        one region the descriptor is standing on.
```

## leaves.py:709 comment run

```
        # ROW INDEX, not `stride * t`: the tiles of one wave are a vector group of `VW` adjacent
        # rows and then a jump of `MIWaveGroupShape` (see `tile_row`, #237).
        # WHICH K COORDINATE THE ADDRESS USES is the packed/contiguous rule, same as the tile
        # axis (`tile_row` + `region_bytes`).  GIR hands down BOTH: `kIdx` is within-region and
        # `kFlat` is the absolute reduction index; a contiguous split has no region displacement,
        # so it must use the flat one or every region r>0 re-reads region 0.  Unsplit and
        # MT-split reads have `kFlat == kIdx`, so this is inert for them.

```

## leaves.py:698 comment run

```
        # WHETHER THE READ OWES A REGION TERM IS DECIDED BY THE LAYOUT, and the rule lives in
        # `lds_geometry.region_bytes` -- one statement, read by both the base and the row index.
        #
        # Unroll-major gets NO region term, which is measured, not assumed: on 2026-08-16 adding
        # `region * splitBoundaryBytes` there regressed 82 passing kernels (110 -> 192 failures),
        # because the split's two `tensor_load`s reproduce exactly the contiguous image one
        # unsplit load produces.  Tile-major is the opposite case -- its regions are separately
        # packed blocks -- and generalising the unroll-major measurement to it is what made every
        # NT split kernel read region 0's bytes for half its tiles.

```

## leaves.py:661 comment run

```
            # MANY ACTS, ONE INSTRUCTION -- the group-leader rule.  When one ds_read covers
            # `tilePerRead` tiles (the MX scales of adjacent tiles are contiguous in LDS), the act
            # for the FIRST tile of each group issues the load that fills all of them and the rest
            # issue nothing.
            #
            # THIS IS A MERGE theta DID NOT DESCRIBE, and that is why it is the fallback rather than
            # the rule: theta emits one read act per tile, so folding two of them here is L3 acting on
            # its own authority, and when the two acts name different LDS buffers the surviving
            # load fetched one buffer for both (MEASURED 2026-08-19: `tile=0,k=0` token 9 vs
            # `tile=1,k=0` token 8 under one `ds_load_b64` -- the MXFP8 six-axis miscompare).
            #
            # THE RULE IS GONE (#267, 2026-08-22).  Its own condition said it "stays until theta
            # supplies the merge for these shapes", and theta now does: `hop_broadcast` subtracts
            # `quantum_axes` from the read hop's the axes it varies over, so GIR emits ONE read act per
            # INSTRUCTION and there are no non-leader tiles left for L3 to skip.  MEASURED before
            # deleting -- the skip fired 0 times across the whole mxf8 matrix -- and the emit is
            # byte-identical without it.  `covers` on the register Ref is what carries the span
            # now, and `refs.covered_coords` is what expands it for reaching-def.

```

## leaves.py:633 comment run

```
        # MANY ACTS, ONE INSTRUCTION -- the group-leader rule.  When one ds_read covers
        # `tilePerRead` tiles (the MX scales of adjacent tiles are contiguous in LDS), the act for
        # the FIRST tile of each group issues the load that fills all of them and the rest issue
        # nothing.  Realizing every act would issue the identical load `tilePerRead` times: not
        # wrong, but `tilePerRead`x the LDS traffic, and it would make the emitted read count
        # disagree with the scaffold's for no reason.
        #
        # NOTHING IS LOST BY THE SKIP.  The skipped acts name the same operand, the same buffer
        # generation and the same reduction coordinate as their leader, so they carry the same LDS
        # memory token; the leader's read already establishes it, and a second wait on it would be
        # a no-op.  The wmma still names each tile's own register -- `generateSrcStrForMFMA` maps
        # the tile index into the group's register span -- so the def-use is per tile even though
        # the load is per group.
        # WHICH ACTS EMIT, AND INTO WHICH REGISTER. theta's `ir.QuantumMap` decides it when one
        # is supplied (`coverage` = the per-act `Decision` from `gir/coverage.plan_quantum`); with
        # none supplied the historical rules below stand.
        #
        # !! `Decision.reg_slot` IS THE REGISTER INDEX -- what multiplies `wtRegStride` -- NOT the
        # position inside the instruction.  Those are different numbers and conflating them
        # renamed every scale register: MEASURED 2026-08-20, the whole mxf8 matrix failed when
        # `reg_slot` was fed `slot(t) = t % tilePerRead` (the within-instruction position) where
        # the leaf wants the GROUP index `t // tilePerRead`.  `slot(t)` is what the CONSUMER needs
        # to find its half of a merged load; the leaf needs the base.

```

## leaves.py:577 _checkRegBuffer

```
INVARIANT GUARD (#122) -- not a diagnostic anything is expected to trip.

        Read honestly: on the UseLoopModel path this CANNOT fire.  GIR names the slot
        `rate_index mod W_g`, so `bufferIdx < W_g <= max_g W_g <= numVgprBuffer` once
        `KernelWriter.loopModelRegBuffers` has grown the allocation to theta's rotation width.  The
        assertion is that relationship restated at the point of use.

        It is kept because the relationship is NEW and cheap to break: for a long time the two
        sides came from different places (theta's width vs the scaffold's `PLR + 1` /
        `ClusterLocalRead ? LoopIters`), and they silently agreed only while `n_s == LoopIters`.
        When they diverged the symptom was an UNDEFINED SYMBOL -- an assembler error on generated
        text, with nothing naming the width that caused it.  So: if a future change re-sizes the
        buffers from anything other than theta, fail here with both numbers instead of there with
        neither.

        SCOPE -- what this does NOT check.  It verifies the symbol is DEFINED, not that the `X`
        indices are DISTINCT registers.  `KernelWriterAssembly`'s `.set` emission has a
        `lrvwTile > 1` branch that resets the register offset after each buffer index, mapping
        every `X<bi>` onto the same offset; under it GIR's rotation would be a silent no-op with
        no undefined symbol and no error here.  Empirically that branch is not taken by anything
        we generate (checked: 442 kernels, none with `ValuA_X1` aliasing `X0`), and it needs
        `UnrollMajorLDS<tc> == False` and `VectorWidth<tc> > 1` together, so no guard is written
        for it -- but do not read a pass here as evidence the buffers are disjoint.
```

## leaves.py:545 comment run

```
            # A PACKED REGION SHORTENS THE UNROLL ROW HERE, exactly as it shortens the free row in
            # `UnrollStride` on the tile-major path -- same rule, different axis, because "the row"
            # is whichever axis is INNER in the image.  LDS is `[free][unroll]`, so a DU split
            # packs `[region][free][DepthU/nsplit]` and consecutive free tiles are that much apart,
            # not a whole DepthU.  Without this the free-tile stride overshoots into the next
            # region on every tile after the first.

```

## leaves.py:502 comment run

```
        # THE SAME ROW LENGTH THE BASE REGISTER USES.  When the split packs regions separately the
        # unroll row shortens to `MacroTile / nsplit`, and `LraTileAssignment` must apply the
        # identical shortening to its `kOffset * strideUnroll` -- the immediates and the base are
        # two halves of one address, so they are derived from one function (#236).  Whether it
        # packs is the (axis, layout) diagonal, not the layout alone, which is why `splitAxis`
        # has to travel with the factor.

```

## leaves.py:365 buildMxScaleReadContext

```
Loop-order-invariant LDS-read constants for a MICROSCALING SCALE tensor (`MXSA`/`MXSB`).

        A SEPARATE BUILDER, not a branch in `buildLdsReadContext`, because the scale tensor's LDS
        image is genuinely a different geometry rather than a narrower version of the same one --
        the scaffold gives it its own emitter (`Components/LocalRead.py:localReadMX`), and A/B's
        `UnrollStride`/`tileStride` swap roles there.  Every validation gate in the A/B builder
        (LDS-transpose vs unroll-major, the `bpeDS` pair, the fragment table) is about that other
        image and would be answering the wrong question here.

        THE WHOLE GEOMETRY IS TWO NUMBERS, both in units of `mxUnit = MatrixInstK / MXBlock` --
        how many scale values one matrix instruction consumes along K:

            per TILE step      : mxUnit                       (`tileStride` in the scaffold)
            per SUBSTEP step   : MacroTile * mxUnit           (`UnrollStride`; M-blocks
                                                               interleaved on K)

        so `addr(t, k) = tile_row(t) * mxUnit + k * MacroTile * mxUnit`, and `tile_row` is the SAME
        vector-group row index A/B use on the unroll-major path (#237) -- a wave takes `VectorWidth`
        adjacent tiles, then jumps `MIWaveGroupShape`.  That is why this returns an ordinary
        `LdsReadTileContext` with `unrollMajor=True`: the field means "`tileStrideElems` is a
        per-row stride and the tile index must be turned into a row index first", which is exactly
        the case, and `emitLdsReadTile` then needs no MX branch at all.

        MEASURED against an emitted scaffold kernel (`test_split_tdm`, MT256x128, MXBlock 32, so
        mxUnit = 128/32 = 4): consecutive MXSA substeps are 1024 bytes apart, and
        `MacroTile * mxUnit = 256 * 4 = 1024`.

        ONE READ MAY COVER SEVERAL TILES.  A scale read moves `blockWidth * BPR` bytes =
        `tilePerRead` tiles' worth, and the scales of adjacent tiles are contiguous (`tileStride`
        is `mxUnit`), so above 1 several of GIR's per-tile read acts are realized by ONE
        instruction.  That is a PACKING fact, not a schedule one -- the same kind of fact as
        `read_fragments` in the other direction -- and `emitLdsReadTile` handles it with a
        group-leader rule.  MEASURED: at MIWaveTile 2 / VectorWidth 2 / MXBlock 32 / MatrixInstK
        128 the scale read is `b64` (blockWidth 2 = 8 bytes) against `mxUnit` 4, i.e. 2 tiles per
        read -- so this is the FIRST shape, not an exotic one.
```

## leaves.py:307 comment run

```
        # ---- MICROSCALED: the same opcode, plus the two scale registers and a block modifier ---
        #
        # THE SCALE SOURCE IS BUILT BY THE SCAFFOLD'S OWN HELPER, `generateSrcStrForMFMA` -- the
        # same one A and B go through, called with `tP["MX"]` -- so the scale register naming cannot
        # drift from the scaffold's.  GIR's slot replaces the scaffold's `m`, exactly as for A/B.
        #
        # `mxsTileSpanScaleSel` MAPS the logical tile index to (register, half-wave select).  It is
        # the identity when TileSpan is off, and TileSpan is a GEOMETRY gate -- it needs
        # `MIWaveTile/VectorWidth` even and >= 2, plus a tile-axis instruction size of
        # WavefrontSize/2 -- not a property of the scale format.  Calling the scaffold's function
        # rather than assuming either answer is what keeps the load side (which halves its ds_reads
        # under the same gate) and this consume side from disagreeing.

```

## leaves.py:205 comment run

```
        # THE OPCODE COMES FROM THE SHARED TABLE, not from a literal.  `dataTypeToMfmaInstTypePair`
        # is the scaffold's own derivation (hoisted out of `KernelWriterAssembly.mfmaIter`), so a
        # ULM1 kernel and its ULM0 twin cannot disagree about which `v_wmma_*` to issue -- the
        # mixed-input cases (fp8_bf8 and friends) depend on `SourceSwap`, which no re-spelling here
        # would have carried.  Only its A/B half is used: the pair's second element is the MFMA
        # output type, and on the WMMA path the accumulator type follows ComputeDataType instead
        # (the same substitution `mfmaIter` makes at its `is_mfma` guard).

```

## leaves.py:3 <module>

```

Leaf emitters -- the L3 per-tile realization of a GIR Move/Mma.

These are the order-AGNOSTIC per-coordinate emitters: the rocisa for ONE (idx0,idx1,u) WMMA tile
or ONE (tile,k) LDS read is identical regardless of loop order; only WHEN it is emitted (the GIR
plan) changes. They live here -- in the Lowering (L3) layer -- because they ARE the layer-2->3
realization; `gir_to_rocisa.GirToRocisa` owns a `LeafEmitters` and drives it from the GIR plan.

Ownership (clean-architecture): the register ring width `W` (used for m = u % W) is a GIR fact
carried on `LeafEmitters.reg_depth`, a real constructor field -- NOT reached into as a private
attribute from another object. Everything else goes through the active KernelWriterAssembly
(`writer`) so nothing is duplicated from the original emitters.

Scope (dense WMMA on gfx1250, Wave32, bf16 or 8-bit-float inputs; non-sparse/MX/complex/F32-emul).
Other paths raise NotImplementedError rather than mis-emit; they are added in later phases.
(These leaf emitters originated in the now-deleted Components/LoopModelLowering theta-IR walker.)

```

## loopir_to_gir.py:1017 gir_text

```
Render an ALREADY-BUILT GIR Program as text for the `OutputLoopIR` dump.

    Takes the Program, never a kernel: KernelWriter builds the GIR exactly ONCE per kernel
    (R-ONCE) and the three phase forks emit from that one object, so the dump must render THAT
    object.  Rebuilding here would both redo the work and risk showing a Program that differs
    from the one that produced the .s -- a debug artifact that lies is worse than none.

    Companion to `LoopModel.bridge.kernel_to_ir_text`: that shows what the decoder DECIDED (the
    rolled theta-nest), this shows what the lowering CARRIES into L3 -- every Move/Mma with its
    concrete generation and register residence.  Together they localize a defect to a layer.

    Lives in the lowering because this is the only layer that may see both sides: LoopModel is
    layer 1 and must not import GIR, and the `gir/` package is deliberately dependency-free.  The
    rendering itself is `gir.render_gir` -- not re-implemented here.

    Observability only, NEVER allowed to break codegen: every failure returns a `# ...` comment.
```

## loopir_to_gir.py:994 comment run

```
    # --- short{0..M-1} -- the LoopIR's `els` arm, as REAL blocks -----------------------------
    # The `T < M` degenerate path: no steady region, so each chunk is copied and consumed in its
    # own step (`Bind(iter = t)`, a CONCRETE chunk, hence absolute generations and `gen_rel=None`,
    # exactly like the prologue).  Emitted as its own chain because that is the shape the LoopIR
    # has; `FoldShortPathPass` collapses it into the shared prologue/drain path when it proves the
    # reordering that requires is legal.  The instructions are WALKED, never reconstructed from
    # `copy_insts`/`nest` -- schedule construction belongs to theta/emit.
    #
    # Guard chain: step t>=1 carries `T > t`, so the block that DECIDES whether step t runs is
    # step t-1.  A failed guard leaves the region entirely ("end"), which is what a trip count
    # smaller than the peel depth means.

```

## loopir_to_gir.py:981 comment run

```
    # --- steady --------------------------------------------------------------------------
    # ONE source loop, `n_copies` FALLTHROUGH-CHAINED blocks sharing ONE back-edge (G2).  A body
    # replicated `n` times consumes `n` reduction chunks per trip, which is what makes each copy's
    # modular buffer index a COMPILE-TIME residue instead of a runtime `% ring`.  Copy `i` is the
    # same rolled loop re-flattened at chunk offset `i` -- the residue timeline therefore spans the
    # whole chain BY CONSTRUCTION (`rel=i` feeds each ref's `gdelta`), rather than each copy being
    # a separate island the analyses would have to re-join.  n_copies==1 is the ordinary
    # single-block loop and takes exactly the path it always did.

```

## loopir_to_gir.py:960 comment run

```
    # the entry names a block that ACTUALLY EXISTS.  A one-hop path (direct-to-register,
    #) stages nothing through shared, so there is no prefetch to peel and NO prologue block;
    # naming "prologue" anyway left the CFG rooted at a missing block.  With a single self-looping
    # steady block dominance happened to work out regardless, which is why this survived -- but a
    # multi-block loop body (G2) has no dominating entry at all, and BOTH copies then look like
    # loop headers.  Derive the root instead of assuming it.

```

## loopir_to_gir.py:934 comment run

```
    # PRESENCE-DERIVED addressing : L3 reads a leaf's free-tile /
    # reduction index by PROJECTING the coord onto per-operand free modes / the reduction modes --
    # handles multi-mode roles (M_split+M_inner both free) and N operands.  Carry:
    #   reduction_modes   : the contraction axes (inner modes absent from the output's the axes it varies over)
    #   free_modes[op]    : each operand's OWN free the axes it varies over modes, in loop order order (mixed-radix)
    #   mma_inputs        : the two matmul-input operand names, in (idx0, idx1) order -- the compute
    # primitive is matmul, so the wmma GRID is exactly these two.
    #   mma_scales        : the REMAINING register-read inputs the wmma consumes, in the same
    #                       parent order -- the MX scales.  They are wmma inputs but NOT grid axes:
    #                       a scale follows its parent's fan, so it is indexed by the parent's tile
    #                       (`idx0`/`idx1`) and carries its own buffer (`bufMXA`/`bufMXB`), which is
    #                       exactly how `emit_plan._plan_mma` already stamps the act.
    #                       WHY THIS EXISTS: `verify_dataflow` modelled a wmma's inputs as
    #                       `mma_inputs` alone, so scale registers were written and NEVER marked
    #                       consumed.  That is invisible at `W = R` (each substep owns a slot, so a
    #                       slot is only rewritten with the SAME source and the guard suppresses
    #                       it) and turns into a phantom OVERWRITE-BEFORE-USE the moment the ring
    #                       narrows -- 24 of them on mxf8 `DepthU 512`, against an emitted order
    #                       that is provably correct.  The scale ring had never actually been
    #                       dataflow-checked; splitting the two lists is what lets it be.

```

## loopir_to_gir.py:898 comment run

```
        # frame: the short arm's chunks are concrete (0..M-1) and there is no steady trip, so its
        # Refs carry absolute generations (gen_rel=None) and it sits at the head of the chunk
        # timeline -- step i IS chunk i.
        # MODEL-ONLY FROM BIRTH (#232/#234).  `_girEmitStage` requests blocks by PHASE NAME and
        # never asks for `short{i}` -- TensileLite serves `T < M` through its own `toPGR1` -- so this
        # block is unemitted the moment it exists, not from whenever a pass decides.  Setting it
        # HERE is what lets an analysis run standalone (the codebase's stated way to exercise one
        # in isolation) and still see which edges lead nowhere; keying it on a flag `FoldShortPathPass`
        # sets meant `GrIncrementRegions()` run before the pipeline saw an unflagged arm and
        # reported conflicts on a path nothing emits.  A FOLDED arm is deleted, so a folded program
        # has no model-only block and its entry edge is modelled (#222).

```

## loopir_to_gir.py:863 comment run

```
            # frame: `_flatten` gave drain step i the offset `rel = i - M` (its Bind is
            # `iter = T - M + i`, so its gdeltas are relative to T).  Its position on the chunk
            # timeline is a DIFFERENT number: the drain runs after the last steady copy, so step i
            # sits at `n_copies + i` in the units the steady blocks use.  Carrying both is what
            # lets an analysis convert between the frames instead of adding a T-relative gdelta to
            # a trip-relative one -- the error that put a spurious swap at the drain entry.

```

## loopir_to_gir.py:734 comment run

```
                         # the `T < M` short-loop arm.  Lowered to REAL blocks (below) in the
                         # LoopIR's own two-arm shape; `FoldShortPathPass` then folds it into the
                         # scaffold's shared prologue/drain shape if and only if it proves that
                         # legal, and records its verdict here (#183).
                         # `model_only` is recorded HERE, with the blocks, because the blocks are
                         # unemitted from birth (see the Block below).  A flag with no stated
                         # reason is what G-EMIT refuses, so the two must be written together;
                         # `FoldShortPathPass` then CLEARS this on FOLD (the blocks are deleted)
                         # or refines it on SPLIT, and `RecordUnemittedPass` fills the count.
                         # (#216 PrefetchGL2 is NOT here.  `meta` is what the theta decoder DERIVED,
                         #  and theta does not model the GL2 prefetch at all -- it stages nothing,
                         #  files no residency and has no completion class, so there is no operand
                         #  for it to be.  The depth is a PRESET and rides `Program.params`, which
                         #  `build_gir` applies after this dict is built; `Gl2PrefetchRegions`
                         #  reads it there.)

```

## loopir_to_gir.py:726 comment run

```
                         # rho, the ONE bit the width rule needs: is this operand's
                         # movement spread over SEVERAL agents?  A hazard on a buffer whose
                         # producer and consumer are the same agent is discharged by program
                         # order and the counter; a cross-agent one needs a proc-scoped fence.
                         # LdsHazards reads it -- without it the analysis would have to assume
                         # every edge is cross-agent and fence a single-wave kernel.

```

## loopir_to_gir.py:708 comment run

```
                         # per OP-CLASS: how many storage regions ITS OWN tile occupies (theta's
                         # `op.split`).  Distinct from both neighbours above and needed because
                         # neither answers the READ ADDRESS's question:
                         #   * `region_modes` says which axes COULD enumerate regions, and an
                         #     operand can carry an axis while being split into ONE region --
                         #     `geometry.hop_broadcast` already guards on `split > 1` for exactly
                         #     this, and the MX scales are the standing example.
                         #   * `unit_regions` is keyed by MOVEMENT (a Phi group), so it cannot be
                         #     asked about one member of a fused pair.
                         # MEASURED 2026-08-18: `TDMSplitA=2, TDMSplitB=0`.  `K_split` is the
                         # SHARED reduction axis, so `translate` gives it to BOTH operands'
                         # `region_modes` (extent 2 = the finest split, with the unsplit operand
                         # present at every value).  The copies were right -- B moved as one -- but
                         # B's READS came out `region in {0,1}` with `k` forced to 0, so the
                         # unsplit operand added a region displacement to an image that has none
                         # and read the same half twice.  768/768 on both asymmetric DU cells.

```

## loopir_to_gir.py:695 comment run

```
                         # THE ABSORBED AXES, per operand: `((axis, extent),...)` for each
                         # inner mode the read's ONE instruction SPANS, so `geometry` has already
                         # removed it from that hop's the axes it varies over.  This is the OTHER HALF of the
                         # subtraction: with the axis gone from the read's `coord`, the def names
                         # one location while its consumers still name every value of the axis, and
                         # `refs.covered_coords` re-inserts them.  Apply the subtraction without
                         # this and every non-leader consumer loses its reaching def.

```

## loopir_to_gir.py:679 comment run

```
                         # THE PARTIAL FOLD, per operand: `{axis: factor}` for each axis this
                         # read's coverage folds only IN PART (`1 < q < N`), so the axis stays in
                         # the axes it varies over carrying the GROUP INDEX and theta issues one act per group via the
                         # read's `mode % q == 0` first-touch guard (`emit._first_touch_modes`).
                         #
                         # WHY IT CROSSES THE LAYER.  `plan_quantum` cannot tell a group theta FOLDED
                         # (one act, carrying its span in `Ref.covers`) from one whose other acts
                         # WENT MISSING -- identical act counts, opposite meanings -- so the coverage has
                         # to arrive as a FACT rather than be inferred from the count.  Without it
                         # the check reads every correctly-folded group as one act short (measured:
                         # NKM/NMK at MIWaveTile [2,8] split=2), and inferring it the other way
                         # would silently accept a genuinely incomplete group.

```

## loopir_to_gir.py:669 comment run

```
                         # THE MOVEMENT QUANTUM, per operand: the `ir.QuantumMap` theta carries
                         # on the shared->register hop, or absent for the identity merge.  Carried
                         # into GIR because L3 must OBEY the merge rather than invent one:
                         # `gir/coverage.plan_quantum` evaluates it to decide which act issues and
                         # into which register.  Supplied from outside theta (the address layout and
                         # the emitter's coverage are invisible to it,), so this is a pass-through.

```

## loopir_to_gir.py:545 comment run

```
        # THE SOURCE NAMES THE OPERAND'S OWN COORDINATE, NOT THE WMMA'S.
        #
        # `coord` is the wmma's full (K,M,N) position.  A is broadcast over the N axes and B over
        # the M axes, so stamping the whole coord on every source made A's register
        # Ref read `A.register{K0,M0,M0,N_split0,N0}` -- a location NO read ever writes, since the
        # producing ds_read names `pres(A) = {K_inner, M_split, M_inner}` alone.  Def and use then
        # disagreed on model for every broadcast operand.
        #
        # Nothing MISCOMPILED, because the two consumers that matter both worked around it:
        # `emit_plan._plan_mma` projects `inst.coord` itself and never reads these Refs, and
        # `short_path._free_coord` re-projects per operand -- its docstring names this exact defect
        # ("an `Mma` src Ref carries the whole (K,M,N) coord, so B's Ref mentions `M_inner` even
        # though B is broadcast over it") and warns that keying on it would "invent differences
        # instead of merging them".  A fact that every consumer must defend against is stored
        # wrong; projecting here removes the defence rather than adding a third copy of it.
        #
        # The projection also makes the BROADCAST VISIBLE, which is the point: the N wmma of one
        # `m` row now name one and the same `A` Ref, so the dump shows one A register serving them
        # all instead of N distinct-looking ones.

```

## loopir_to_gir.py:414 _quantum_coords

```
The coordinates ONE transfer fills: `coord` alone, or its product with the coverage axes.

 `coverage` is `((axis, n),...)` from `ir.Load` -- the axes this single instruction spans,
 in the two shapes `_absorbed` documents, told apart HERE by whether `coord` already names the
 axis. No marker field: the coord IS the discriminator, because the the axes it varies over subtraction that
 produced it is the same fact.

 * APPEND (full absorption, `q = N`) -- `coord` does not name the axis, because the axis left
 the hop's the axes it varies over entirely. Re-insert it with all `n` values: a `ds_load_b64` covering two
 scale tiles yields two coordinates, hence two definitions. Nothing to collide with, so the
 axis is appended rather than merged into position.
 * SUBSTITUTE (partial coverage, `q < N`) -- `coord` names the axis with its GROUP INDEX `g`, so the
 instruction fills the `n` members `g.n... g.n+n-1`. Appending would name the axis twice and
 invent a coordinate no consumer reads; the group index must be EXPANDED IN PLACE.

 Getting the partial case wrong is not a silent miscount: the consumers (a wmma indexed on the
 output's the axes it varies over) name every tile value, so a def that stopped at the group index leaves them
 reading a location the arm never writes -- the "N consumer(s) read" failure `placement.py`'s
 the axes it varies over-subtraction note warns is the other half of this pair.
```

## loopir_to_gir.py:371 _absorbed

```
`((axis, extent),...)` the operand's read instruction SPANS -- the absorbed axes.

 THE OTHER HALF OF THE PRESENCE SUBTRACTION. `geometry.quantum_axes` took these axes out of the
 read hop's the axes it varies over, so the emitted `Load.coord` no longer names them and the read is ONE act
 per instruction (which is what the leaf emits). The CONSUMERS still name every value -- the
 wmma is indexed on the output's the axes it varies over, which keeps the axis -- so the def has to be readable
 as filling all of them. `refs.covered_coords` does that re-insertion from this tuple.

 TWO SHAPES, ONE TUPLE, TOLD APART BY THE COORD. A coverage folds an axis of extent `N`
 into `N/q` carrier groups, and the model's `q = N` case is only the degenerate one:
 * FULL (`q = N`) -- the axis is gone from the hop's the axes it varies over, so `Load.coord` does not name
 it, and the entry `(axis, N)` is re-inserted with all `N` values.
 * PARTIAL (`q < N`) -- the axis STAYS, carrying the GROUP INDEX `g in [0, N/q)`, and the entry
 `(axis, q)` says the instruction fills `g.q... g.q+q-1`.
 `_quantum_coords` distinguishes them by whether `coord` already names the axis, so no marker
 field is needed and an operand may carry one of each.

 Empty for every operand with no supplied merge, which is A, B, and every unmerged scale.

 PARTIAL FOLDS ARE NOT REPORTED YET (#301). `geometry.hop_fold` derives them and the SUBSTITUTE
 branch in `_quantum_coords` / `refs.covered_coords` is written, but a partial coverage is only
 coherent once `theta.presence_modes` folds the extent AND `emit`'s read nest enumerates group
 indices instead of tiles -- all three move together or the coord and the expansion disagree.
 Reporting one here alone would substitute a TILE index as if it were a GROUP index and name
 coordinates past the end of the axis.
```

## loopir_to_gir.py:316 _reg_residence

```
`(group index, concrete slot, rotation width W)` for `op`'s register fragment AT `coord`.

 WHICH GROUP HOLDS THIS COORDINATE -- the piece #118 was missing. A register partition
 (`VgprPartition > 1`) splits the fragment's in-region fan into groups that partition that
 mode's VALUES : at `MIWaveTile[4,4]` with `VgprPartitionA=2`, `grouping_mode` is
 `M_inner` (extent 4) and `lo` owns `M_inner in {0,1}`, `hi` owns `{2,3}`. `_read_placement`
 already emits one `(label, slot Expr)` per group and the groups carry DIFFERENT widths (lo
 `unroll` -> W=R, hi `inplace` -> W=1), so the two halves are genuinely different registers.

 Both call sites used to take `pl.slots[0]` unconditionally, i.e. always `lo`. That made the
 parameter a SILENT NO-OP AT EMIT while still moving the modelled footprint: every A source in
 the whole program came out `group=0, ring=2`, byte-identical to `VgprPartition=1`, while
 sizing reported 48 registers instead of 64. A parameter that changes the model but not the
 kernel is worse than one that is rejected -- the tuner would believe it bought registers it
 never bought.

 EQUAL PARTITION, which is `Fragment.grouping_mode`'s stated assumption (`len(parts)` divides
 the mode's extent, checked in `translate`), so the owning group is a floor division. Falls
 back to the first slot whenever the fragment is unsplit, states no grouping mode, or the coord
 does not mention it -- every pre-partition shape, which is why this is inert at
 `VgprPartition=1`.
```

## loopir_to_gir.py:265 _member_region_coord

```
Re-key a FUSED instance's region coordinate onto THIS member's OWN region axis.

 move pairs a Phi group's members BY INDEX, not by a shared mode: region j of A moves with
 region j of B, and each names the axis it is split on -- A's `M_split`, B's `N_split`. The
 LoopIR instance carries ONE coord, necessarily naming one member's axis (its representative's),
 so handing that coord to every member leaves the others' region axis UNPINNED.

 What that costs, measured 2026-08-16 on a multi-wave TN split: the fused copy's tokens came out
 `(1,5,7)` and `(3,5,7)` -- A's two regions correctly distinct (1 vs 3) while B carried BOTH of
 its buffers (5 and 7) on BOTH region copies. `_regions_of` widens an unpinned axis to every
 region, which is the honest reading of "this access does not say which half" -- but here the
 access does know: it is region j of a by-index pairing. The consequence is that B's two halves
 are INDISTINGUISHABLE to `LdsHazards` and to `MemTokenAssignment`, so a genuine cross-region
 hazard on B cannot be seen, and every fence is computed against a wider buffer set than the
 load actually touches. That is #170, and it is the standing suspect for the SIA4-only split
 failures, where StinkyTofu reorders on exactly these tokens.

 Rewrites ONLY when the coord names a region axis that is not this member's own; an unfused
 operand, and the representative member of a fused group, already name their own axis and are
 returned untouched.
```

## loopir_to_gir.py:62 _short_steps

```
The short-loop (`Cond.els`) arm, split into its per-step (guard, nodes) -- or a raise.

    THE TWO SHAPES (#183).  TensileLite's scaffold and the LoopIR describe the same kernel with
    DIFFERENT STRUCTURE:

        scaffold :  prologue -> Cond -> loop -> else -> drain     prologue+drain SHARED by both
                                                                   paths; `T < M` just skips the loop
        LoopIR   :  Cond { prologue -> loop -> drain } else { short }        two DISJOINT arms

    This function does NOT choose between them.  It hands back both arms' material so the lowering
    can emit the LoopIR shape faithfully, and `FoldShortPathPass` can then FOLD the short arm into
    the shared shape *when it has proven that legal* -- or leave it as its own blocks when it has
    not.  Previously this returned only `{steps, guards}` and the instructions were DISCARDED, on
    the claim that "the scaffold reconstructs it from prologue+drain".  That claim is false for 13
    of 38 configs (every PGR3, and `S_shared=1`): the reconstruction separates copies from computes
    and needs `M` live shared slots to do it.  Discarding on an unchecked claim is the failure mode
    #183 exists to end.

    The shape assertions stay -- M concrete chunk steps, step `t>=1` guarded by a peel-step-validity
    predicate (`T > t`) -- but they are now VALIDATION of what we lower, not the justification for
    throwing it away.  Returns `[(guard | None, [nodes]), ...]`, one entry per step.
```

## loopir_to_gir.py:3 <module>

```

loopir_to_gir -- the layer 1->2 lowering.

Walks the LoopModel LoopIR (the rolled theta-nest `build_ir` emits) EXACTLY ONCE and unrolls the
three software-pipeline stages (prologue / steady / drain{n}) into one minimal GIR CFG. It is
the *tree-walk half* of the old `LoopModelLowering._walk` with NONE of the physical-state
reconstruction (no pointer seeding, no swap heuristics, no token stepping -- those become GIR
passes / L3).

What it attaches (all logical, R-SEMANTIC):
 - one loop-carried `Gen` per staged-shared operand (ring = its LDS depth); STEADY refs use it
 with a concrete `gdelta`; the steady header carries the Gen phi + the back-edge GenXfer.
 - PEEL (prologue/drain) refs carry a CONCRETE `abs_gen` (fact,): their generations are
 pinned by the LoopIR peel (concrete `cst` slots), NOT loop-carried. (DRAIN refs are the
 exception: they carry `gen`+`gdelta` on the steady timeline -- see `_convert_load`.)
 - register residence as a concrete (group index, slot) per read.
 - a native `phase_boundary` Mark at each block head.

It inserts NO swap / gr_increment / gsu_guard Marks -- those are analysis-produced (R2/R3).

This module reads LoopIR `Expr`s; it is the ONE place that does. It imports the LoopModel
core (stdlib-only) and the GIR nodes; it imports NOTHING from rocisa.

```

## region_derivation_check.py:3 <module>

```

RegionDerivationCheck (#245) -- is a wave tile's TDMSplit region the same whether you derive it from
the TILE INDEX or from the tile's actual FREE-AXIS POSITION?

WHY THE TWO CAN DISAGREE.  `lds_geometry` already holds both derivations, for different callers:

    tiles_per_region(ctx)   MIWaveTile / nsplit      -- "region = t // tiles_per_region"
    tile_row(ctx, t)        (t // VW) * groupShape + (t % VW)

The first counts TILES and assumes they partition evenly.  The second is where the tile actually
SITS along the free axis, and it is not proportional to `t`: a wave takes `VectorWidth` adjacent
rows, then skips a whole `MIWaveGroupShape` to hand the intervening rows to the other waves.  So
the map from tile index to free position is piecewise, and dividing the index is only the same as
dividing the position when the vector groups happen to line up with the region cut.

`tiles_per_region` states the precondition that makes its floor safe -- "Solution.py rejects the
shapes where that does not divide, so the floor never discards a tile".  This module CHECKS that
claim rather than trusting it, and checks the stronger property the floor does not cover: that the
tile index and the free position agree on WHICH region, for every tile of every wave.

WHY IT IS NOT AN ASSEMBLY QUESTION.  An earlier attempt answered this by interpreting the emitted
per-lane address code.  That works, but it is the wrong instrument: it needs the whole launch state
(workgroup ids, problem sizes, and on some configs a float-reciprocal integer divide) to say
anything, and none of that is what decides the region.  The region is decided by
(MIWaveTile, VectorWidth, MIWaveGroup, nsplit, layout) -- compile-time shape, all of it already in
`lds_geometry`.  So the check calls those functions directly and needs no kernel, no launch, and no
GPU.

WHAT A FINDING MEANS.  A disagreement is the #245 defect: the read names one region while its lanes
sit in another, so `MemTokenAssignment` gives it that region's token and `FenceRegions` orders it
against the wrong producer.  Nothing downstream can detect it -- GIR carries the region as a
coordinate, so a mis-derived region is simply believed.

```

## uniform_emission.py:3 <module>

```

The EMISSION half of the L733 uniform-placement check -- where the agent-discriminating arms
actually are.

L733 asks a *tree-structural* question of the IR: does a `scope=block` selector's node lie where no
agent-discriminating arm can make an agent skip it? `gir/analyses/uniform_placement.py` asks it of
GIR and gets a truthful but WEAK answer, because GIR's region tree has no node for the arms this
kernel really contains:

 s_bitcmp1_b32 s[sgprWaveIdx], 0 // check wave parity
 s_cbranch_scc1 label_SkipStaggerA // skip: odd waves handle B... <- executed by EVEN waves only
 label_SkipStaggerA:

`KernelWriterAssembly._applyStaggerTDM` (and the wave-separated TDM init / global-offset /
increment emitters) open these; they are scaffold-owned, invisible to GIR, and a GIR fence lowered
into one would deadlock the workgroup with the empty-ledger gate and G-UNIFORM both green. So the
question has to be re-asked in the one place the arms exist: the emitted stream.

WHY IT HOLDS TODAY, AND WHY THAT IS NOT A PROOF. Every such arm is emitted BALANCED inside a
single emitter's `Module` -- the `SCBranch*` and its `Label` are added to the same `imod` a few
lines apart (`_applyStaggerTDM`) -- and a GIR fence is added at an entirely different point in the
module tree, so it cannot land between them. That is a property of how the emitters happen to be
written, not an invariant anything enforces: an emitter that opened an arm and closed it in a
sibling module, or a GIR act interleaved into such a module, would break it silently and the only
symptom would be a hang.

MEASURED 2026-08-21, by TWO INDEPENDENT PROBES (an earlier version of this paragraph mixed three
tools' totals into one inconsistent sentence; these are the reconciled numbers):
 * this check, run in-process over the emitted module: 5028 ULM kernels judged, 8540
 agent-discriminating skip regions, 4053 GIR fences -> 0 inside an arm;
 * a separate POST-StinkyTofu scan of the finished `.s` for the same 6024 files: 14430 barriers,
 5230 of them GIR-tagged -> 0 inside a region.
This check is what keeps that number at 0.

WHAT COUNTS AS AGENT-DISCRIMINATING. A forward conditional branch whose condition is derived from
an agent identity:
 * the branch tests EXEC (`s_cbranch_execz/execnz`) -- lane-discriminating by construction;
 * the branch tests VCC written by a `v_cmp` -- a vector compare is per-lane;
 * the SCC/VCC producer reads an agent-identity register (`sgprWaveIdx`, `vgprSerial`,...).
The third is a SNIFFER and is the check's soundness limit: an agent id laundered through a
temporary more than `_LOOKBACK` instructions before the branch is not recognised.

ONLY THE FIRST IS STRUCTURAL. (ii) IS A HEURISTIC AND IT OVER-CLASSIFIES -- MEASURED. A `v_cmp` is
per-lane in FORM, but its operands need not be: exactly one region per kernel is classified by this
rule and it is always the ALPHA GUARD, `v_cmp_eq_f32 vcc_lo, s[sgprAlpha], 0.0`, whose operands are
a uniform scalar and a literal. Every wave takes the same side of it, so it discriminates nothing.
That is roughly a 12% over-report of the REGION population.

It inflates the denominator, not the alarm rate: a false FINDING needs an over-classified region AND
a GIR fence inside it, and that conjunction has never occurred (0 findings over 5028 kernels).
Narrowing (ii) to a `v_cmp` with at least one VGPR operand would remove the alpha guard, but no
measurement covers that change, so it is left alone and documented rather than tightened blind.

Adding a name to `AGENT_ID_REGS` is the intended way to extend the third rule; the honest statement
is that this closes the arms we know how to emit, not every arm expressible in assembly.

Read-only: this walks the module and raises; it inserts, removes and reorders nothing, so it
cannot move emission.

```


# Second pass -- tighter thresholds

Docstrings of 9+ lines keep a 6-line lead; comment runs of 4+ keep two lines.

## __init__.py:3 <module>

```

Tensile.Lowering -- the UseLoopModel lowering stack.

 loopir_to_gir layer 1 (LoopModel LoopIR) -> layer 2 (GIR); build_gir = build-once (R-ONCE)
 gir/ the GIR graph, analyses, passes, verifier (pure Python; no rocisa)
 gir_to_rocisa layer 2 -> layer 3 (arrives R1; leaf emitters move here from Components)

RAISE `RuntimeError` TO REFUSE A KERNEL. That is the whole failure policy, and it is not
cosmetic. `KernelWriterAssembly.getSourceFileString` wraps kernel generation in

    except RuntimeError as e:

```

## asm_lane_eval.py:326 comment run

```
            # Reads lane 0 into an SGPR.  Sound here ONLY because the slice is uniform-by-
            # construction at every site that uses it (wave id, workgroup id); if a future slice
            # reads a divergent value the SGPR silently loses the divergence, so this is the one
            # place worth re-checking when the table grows.

```

## asm_lane_eval.py:284 comment run

```
            # KERNARG LOAD.  `s_load_bN dst, base, imm` fills `N/32` consecutive SGPRs from the
            # kernel argument buffer at byte offset `imm`.  The base pointer is not modelled (there
            # is one buffer and the caller indexes it by offset), but the OFFSET is, so two loads
            # at different offsets cannot collide.  A dword the caller did not supply refuses,
            # keeping the `read` discipline: an unsupplied kernarg is an input, not a zero.

```

## asm_lane_eval.py:88 read

```
One operand's value in `lane`.  An SGPR or literal is lane-invariant by construction.

        An operand the parser could not resolve is REFUSED here, not defaulted to zero: a zero
        address term is indistinguishable from a real one downstream, so it would turn a parse gap
        into a confident wrong answer.

        SO IS AN UNWRITTEN REGISTER, and that one is not hypothetical.  The first version of this
        class returned 0 for any VGPR the slice had not defined, on the reasoning that the slice is
        closed under its own producers.  It is not: a slice can bottom out at a kernel INPUT (a
        workgroup id in an SGPR, an address the host passed in), and every such input silently read
        as 0.  The checker then evaluated every `LocalReadAddrA` to exactly its immediate `offset:`,
        
```

## asm_lane_eval.py:57 comment run

```
    # SCALAR UNARY ops, and the SCC-producing compares.  `s_cselect_b32` is the reason SCC is
    # modelled at all rather than skipped: the split path selects wave-parity values with it
    # (`ldsSplit = parity ? B : A`), so guessing its condition would pick the wrong region step --
    # the exact quantity this checker exists to measure.

```

## asm_lane_eval.py:3 <module>

```

LaneEval -- execute a straight-line slice of emitted gfx1250 assembly, once per LANE.

WHY THIS EXISTS, AND WHY IT IS NOT A FORMULA.  The question it answers is whether the 32 lanes of
one `ds_load` all land in the SAME storage region of a `TDMSplit` tile.  That is a property of the
per-lane LDS address, and the per-lane address lives in a VGPR (`vgprLocalReadAddr*`) built at init
from `vgprSerial` -- the instruction itself carries only a wave-uniform `offset:` immediate.  So the
fact is invisible in the instruction stream, and it is invisible in GIR too: GIR reasons about the
region a READ NAMES, never about the lane spread inside one instruction.

The obvious alternative is to re-derive the address from `LraTileAssignment`'s rules in Python.

```

## __init__.py:3 <module>

```

GIR analyses -- one class per module, each an `Analysis` subclass.

 cfg successors (helper), Dominators, BackEdges (+ BackEdge / BackEdgeSet)
 gen_reaching GenReaching -- concrete generation each access uses 
 swap_regions SwapRegions -- PendingMark(swap) per generation change 
 dep_defuse DepDefuseAnalysis -- RAW/WAR edges carried from LoopIR awaits
 reg_band RegBandAnalysis -- VALIDATE register width W (never decide it, B3)
 gr_increment GrIncrementRegions -- PendingMark(gr_increment) per steady copy (placement only)
 gl2_prefetch Gl2PrefetchRegions -- PendingMark(gl2_prefetch) per steady chunk (#216); MIDPOINT,
 and the only Mark policy here that is a pure COST choice rather than a bound

```

## dep_defuse.py:3 <module>

```

DepDefuse -- the RAW + WAR dependency edges carried from the LoopIR awaits.

Each GIR verb (Move/Mma) carries `deps` = ((dep_name, counter, kind),...) threaded from the
LoopIR `Inst.awaits` by the lowering. This analysis indexes them into a queryable result so the
tokens pass (read/copy/tensor token idx) and L3 WAR-forwarding read one source of truth:

 reach.raw(inst) -> [(dep_name, counter),...] the RAW-residency edges the inst consumes
 reach.war(inst) -> [(dep_name, counter),...] the WAR (rotation/inplace) edges it carries
 reach.counter(inst) -> the completion counter of the inst's producing hop (for the token class)

This is PURE: it only reads `deps` already on the nodes; it never mutates.

```

## fence_regions.py:137 _tokens_across

```
Every buffer LIVE ACROSS this fence -- not merely the ends of the edges that placed it.

 A barrier is a POINT, and its ordering power comes entirely from its token set:
 `StinkyBuildImplicitDependencyPass` gives the barrier each token as both a def and a use, so a
 buffer the barrier does not name is not chained through it at all -- its accesses may migrate
 across it, and the waitcnt strength is derived from the same set. Naming only the endpoints of
 the edges the cover happened to select therefore leaves every other buffer live at that point
 unordered, while the fence still LOOKS like it separates the phases.

 MEASURED, 2026-08-16. The old union-over-covered-edges form was propped up by an
 over-approximation elsewhere: a fused copy left its non-representative member's region axis
    
```

## fence_regions.py:80 _already_fenced

```
Is a fence placed in an EARLIER block already on every path from producer to consumer?

    Without this the cover is per-block and blind across blocks: the last steady copy feeds reads
    in drain0 AND drain1, so drain1 asks for a fence of its own even though drain0's already sits
    between them on every path.  The fences are redundant, not wrong -- but redundant barriers are
    the exact cost this pass exists to remove.

    The condition is deliberately narrow, because a fence in the wrong place discharges nothing: an
    earlier fenced block `x` covers the edge only if it lies BETWEEN the two ends on every path --
    reachable from the producer's block (so it comes after the producer) AND dominating the
    consumer's block (so no path reaches the consumer around it).  Reachability alone would wrongly
    
```

## fence_regions.py:48 _cover

```
Greedy set cover: the fewest slots such that every hazard has a fence between its ends.

 Returns {slot: [hazards it discharges]}. Raises if some hazard admits no slot at all -- that
 would mean the emitted order itself makes the edge undischargeable (an illegal sigma), which is a
 defect in the schedule, not something to model over with a fence somewhere harmless.

 Keyed on the hazard's INDEX, not the hazard: a `Hazard` holds the `Move` it came from, and a
 Move is a mutable node, so the edges are not hashable and must not be forced to be -- their
 identity belongs to the IR, not to this bookkeeping.
```

## fence_regions.py:3 <module>

```

FenceRegions -- the MINIMAL set of proc-scoped fences covering the cross-agent LDS hazards.

`LdsHazards` says which ordered pairs can touch one buffer and at what trip distance; this decides
where to stand between them. Only the cross-agent edges need a fence: the `complete(read) ~>
issue(copy)` is discharged by program-order-on-issue, which relates events on ONE agent, so a
same-agent edge is already covered by program order plus the completion counter.

WHY THIS IS A COVER AND NOT ONE MARK PER EDGE. A fence is not owed to an obligation -- it is a
POINT IN THE PROGRAM, and one point discharges every edge it happens to stand between. The steady
body holds ~20 cross-agent edges and one fence covers nearly all of them. Realizing them

```

## gen_reaching.py:70 _entry_of

```
(per-Gen entry map, is_relative) for `blk` -- a real MERGE over its known predecessors.

    A header carries the phis, so its entry is declared.  Otherwise we merge: every known
    predecessor that AGREES on a Gen's value contributes it; a Gen the predecessors disagree on --
    or that no predecessor supplies -- gets `RELATIVE_BASE` and the block is flagged RELATIVE.

    The disagreement is real and not a modelling slip.  drain0's two predecessors are the last
    steady copy and the prologue (the `T < M` short path), and the generation the steady leaves is
    trip-parity-dependent, so NO compile-time constant is the merged value.  What this used to do
    was return `{}` -- silently base 0 -- and document it as "there is nothing to merge", which reads
    as a fact rather than a placeholder.  A consumer then cannot tell a merged generation from a
    
```

## gen_reaching.py:52 _resolve_phi_entry

```
Per-Gen value on `blk`'s entry, at ANALYSIS granularity.

 A loop header phi contributes entry_val on the entry edge and the back-edge xfer value on the
 loop edge. These are NOT the same value -- with ring 2 and adv 1, trip 0 enters at generation 0
 and trip 1 at generation 1 -- so keying on the declared `entry_val` fixes a REPRESENTATIVE trip,
 not an invariant. That is sound for the consumers there are: the LDS memory token is a LOGICAL
 buffer name whose only requirement is the relative this-buf-vs-other-buf distinction WITHIN a
 body (an earlier version of this docstring asserted the equality instead, which is false and
 invited reading `of` as an absolute generation).
```

## gen_reaching.py:3 <module>

```

GenReaching -- the concrete generation each Move/Mma access uses.

SSA reaching-value over the CFG: resolve each Ref.gen phi to the block-entry value, add gdelta,
mod ring; each back-edge's xfer gives the entry value of the next trip. A peel ref carries a
concrete `abs_gen` (fact,) rather than a Gen, resolved directly.

The analysis returns a `Reaching` result object with a small query API -- callers ask
`reach.of(inst, ref)` / `reach.xfer(block_label, gen)`; they never see the internal keying. The
result holds `prog` alive so its (identity-keyed) index stays valid for the result's lifetime.

```

## gl2_prefetch.py:61 _is_steady

```
The steady body only, and ONE mark per steady block -- which is one per reduction chunk,
        the same rate as `gr_increment`, because the scaffold's own placement is one per
        `_loopBody` call and `_loopBody` is called once per loop copy (`KernelWriter.py` ~6288).

        The prologue issues its GL2 through the scaffold's own peel, which is NOT suppressed under
        UseLoopModel and does not need to be: `KernelWriter.py`'s `counterL <= PGR` /
        `Skip_GL2` block sits outside the main loop, touches only the GL2 address VGPRs, and takes
        no part in the copy dataflow GIR owns.  The drain must not prefetch at all -- that is
        exactly what the clamp exists to stop, so no drain block is marked.
```

## gl2_prefetch.py:24 Gl2PrefetchRegions

```
`[PendingMark]` -- one `gl2_prefetch` Mark per steady chunk, or none when GL2 is off.

    THE DEPTH IS A PRESET, NOT A theta FACT, so it arrives on `Program.params` (`nodes.Program`:
    "PRESETS analyses derive facts from ... not raw data") rather than on `prog.meta`, which holds
    what the theta decoder DERIVED.  theta does not model the GL2 movement at all and should not: it stages
    nothing, files no residency, and has no completion class -- there is no operand for it to be.
    Putting `PrefetchGL2` on theta would also be the one thing #163 forbids, TensileLite vocabulary
    inside the model.  `KernelWriter._loopModelGirProgram` supplies it; every other caller of
    `build_gir` passes no params, so `PARAM` reads 0 and this analysis returns [] -- which is what
    keeps it contributing nothing to the kernels that do not use the feature.
```

## gl2_prefetch.py:3 <module>

```

Gl2PrefetchRegions (#216) -- WHERE the GL2 cache prefetch goes, once per reduction chunk.

WHAT GL2 IS, AND WHY IT IS NOT A gr_increment.  `PrefetchGL2` (0/1/2, gated on the
`HasGlobalPrefetch` asm cap) warms L2 for a chunk `PrefetchGlobalRead + PrefetchGL2` ahead, while
the real global->shared copy runs at `PrefetchGlobalRead`.  It carries its OWN state end to end:

    v_add_co_u32   GL2PrefetchAddr<tc>_i, vcc, GL2PrefetchAddr<tc>_i, GL2PrefetchInc<tc>
    global_prefetch_b8  GL2PrefetchAddr<tc>_i, off

-- its own address VGPRs (`states.<op>.startVgprGL2PrefetchAddr`, `gl2nl` of them per operand) and

```

## gr_increment.py:157 _flow

```
Place the advances for ONE global-read address, via the shared ValuePlacementSolver.

        This analysis owns only what is specific to this register: which instructions consume the
        descriptor, the absolute chunk each fetches, and how a chunk re-frames across the back edge.
        WHERE the advances go is `value_placement`, the same module SwapRegions uses.

        THE CONVENTION THIS FIXES.  The previous walk emitted an advance only where a copy DEMANDED
        a chunk different from the one in hand, anchored just before that copy.  The prologue's own
        copy demands the chunk the tile setup already left the descriptor on, so it demanded nothing
        and NO advance was ever emitted in the prologue -- the descriptor then ran one chunk behind
        the hand-written scaffold's convention for the whole loop.  It still inherited the
        
```

## gr_increment.py:131 _reqs

```
[(inst, chunk)] for `unit`'s copies in `lab`, program order.

 A COPY IS NOT A CHUNK. A region-split movement delivers one reduction chunk with
 `nregions` `tensor_load`s, so the peel's positional index counts LOADS while
 `_chunk_of` needs CHUNKS -- off by exactly that factor. At PGR2 with a 2-way split A's four
 peel copies were read as chunks 0,1,2,3 instead of 0,0,1,1: the address ran to chunk 3, the
 steady frame demanded 2, and the solver closed the gap with a `-1` advance that
 `tdmIncrementGir` cannot express (it emits exactly one DepthU stride), so kernel generation
 raised. At PGR1 the same error is one spurious advance with no negative -- no raise, just a
 descriptor a chunk ahead, which is why those kernels built and miscompared instead.
        
```

## gr_increment.py:90 comment run

```
        # Edges into a block NO BACKEND EMITS -- a KEPT `T < M` arm, which TensileLite serves via
        # its own `toPGR1`.  Same exclusion, same helper, as SwapRegions: a second copy of this
        # rule is how the two would drift.  A FOLDED arm has no such blocks, so its entry edge IS
        # modelled here now (#222/#233).

```

## gr_increment.py:69 GrIncrementRegions

```
Forward dataflow over the GIR CFG for one global-read address per Phi MOVEMENT.

 Per movement, not per operand: a fused group issues one cooperative load off one
 descriptor, so it owns one address that advances once. Unfused, a movement is a lone operand
 and the two readings coincide -- which is why the distinction stayed invisible until multi-wave.

 Domain Z (the reduction chunk the address points at) -- unbounded, because a global read address
 marches up the K axis and never wraps, so unlike the LDS pointers there is no modulus and no
 wrap-induced conflict. The lattice, both fixpoints and the def-placement rule are
 `value_placement.ValuePlacementSolver`; a genuine join conflict is raised, never resolved by
 preferring a predecessor.
 
```

## gr_increment.py:49 _chunk_of

```
The absolute reduction chunk a copy fetches, on the global chunk timeline.

    Loop-carried ref: `blk.chunk_base + (gdelta - blk.gen_rel)` -- frame-free, comparable across
    blocks.

    PEEL ref: derived POSITIONALLY, from `peel_seq` -- the index of this copy among that operand's
    copies in the peel.  The prologue fetches chunks 0..M-1 in issue order, so the k-th copy of an
    operand fetches chunk k, exactly and without assuming anything about the ring.
    It must NOT come from `abs_gen`: that is a RING SLOT (`(t + off - M) % d`), which equals the
    chunk only while `d >= M`.  For an in-place shared buffer (`d=1`), or any `S_shared < M`, every
    peel fill reports the same slot -- the peel would emit no increments, and the number also feeds
    `prefetchIndex`, putting the StaggerU wrap on the wrong iteration.
```

## gr_increment.py:31 _copy_of

```
`(unit, shared-dst ref)` if `inst` is a global->shared copy Move, else None.

 `unit` is the Phi movement's MEMBER TUPLE, not one member's name: a fused group is one
 cooperative instruction over one descriptor, so it is one global-read address
 that advances once. Naming it `refs[0].tile.operand` would leave the other members' Refs
 unexamined and assert a single address rather than check for one -- see `..refs.copy_unit`.

 The `global` source is what distinguishes a copy from any other shared write; `copy_unit`
 alone answers the weaker "is there a shared destination", which is what the swap hop wants.
```

## gr_increment.py:3 <module>

```

GrIncrementRegions -- the global-read-address advance, as a REAL forward
dataflow over the GIR CFG (the same shape as SwapRegions).

Each operand's global read address is a loop-carried POINTER holding the reduction chunk the next
copy will fetch. A `gr_increment` is the only thing that advances it, so -- exactly as for the LDS
pointers -- placement is a reaching-value problem: propagate the address to every program point and
insert an increment wherever supply and demand differ. The fixpoints and the def-placement rule
live in `value_placement` -- the SAME module SwapRegions uses, so the two cannot drift -- and this
file supplies only what is specific to this register: which copies consume the descriptor, the
absolute chunk each fetches, and how a chunk re-frames across the back edge.

```

## lds_hazards.py:331 comment run

```
            # Both peel-pinned.  A peel instance executes once, so the only possible distance is 0
            # and they collide iff they name the same BUFFER.  A buffer is (region, generation) --
            # the generation alone is not an identity: A's buffer 0 and B's buffer 0 are different
            # LDS storage, exactly as their loop-carried counterparts have distinct `Gen.id`.
            # Comparing generations alone paired every prologue A-fill with its B-fill.

```

## lds_hazards.py:315 comment run

```
            # DIFFERENT LDS STORAGE -- checked FIRST, so it covers every branch below (#217).
            # It is not enough to test it in the peel<->peel branch: two LOOP-CARRIED accesses to a
            # split tile's opposite halves share one `Gen.id` (the ring is per operand, not per
            # half), so the `gen_a.id != gen_b.id` test below waves them through.  That left 20
            # phantom same-block hazards on `our_split` after the first cut.

```

## lds_hazards.py:271 _cross_block

```
Hazards whose two ends live in DIFFERENT blocks.

        These are real and they are not covered by any fence inside a single body: the last steady
        copy feeds the drain's reads, and the steady fence sits at the TOP of the body, so on the
        final trip nothing stands between that copy and the drain.  The backend's flat-stream phase
        machine catches it for free (it just keeps walking); an analysis that reasons per body has
        to model it explicitly or it will under-fence exactly where the scaffold does not.

        Aliasing is decided CONSERVATIVELY here, and deliberately so.  Two accesses in different
        blocks on the same region are assumed to collide unless BOTH are peel-pinned in blocks whose
        generation frame is absolute -- because a loop-carried access takes a different generation on
        
```

## lds_hazards.py:164 _regions_of

```
Per region axis of `operand`, the SET of region values this access MAY touch.

    A coord that PINS the axis names one region.  A coord that does not mention it touches EVERY
    region, so the unknown is modelled as the full set and stays a MAY-alias -- being conservative
    costs a fence, being optimistic LOSES one.

    THE PIN IS TRUSTED ON BOTH HOPS, and that is the point: a region split is a COORDINATE fact.
    `Tile.coord` carries the region value, so this says which half every access SHOULD touch.
    Widening our reads to "may touch any region" was tried (2026-08-16) and is WRONG: it hides a
    defect rather than modelling one.  If an access genuinely cannot say which region it touches,
    the IR that produced it is incomplete, and that is the thing to fix, here or upstream.
    
```

## lds_hazards.py:143 split_region_modes

```
`region_modes`, with the axes of movements that have ONE region REMOVED.

 THE TOKEN COUNT IS ARITHMETIC, AND IT WAS WRONG EVERYWHERE. For A and B split in two with a
 double-buffered LDS ring that is `2 regions x 2 buffers = 4` tokens each; the MX scales are
 NEVER split (`tdm_split.split_of` returns `(1, None)` for every MXS tensor), so they are
 `1 x 2 = 2` each -- twelve in total. Every kernel we emitted named SIXTEEN, because
 `_regions_of` keyed on `region_modes` being non-empty and the scales carry their parent's axis.

 THE TUPLE ANSWERS "WHICH AXIS", NEVER "HOW MANY". The scales keep the axis deliberately: the
 two-map reading needs it BROADCAST FOR THE RATE, and removing it at the source drove MXSA's
 rate R from 2 to 4 (measured 2026-08-19, and it cost 26 kernels). So the count must come from
    
```

## lds_hazards.py:72 same_trip

```
Does an instance of this pair occur within ONE trip?

 `distance == 0` means the two accesses land on the same buffer in the same iteration, so the
 edge must be discharged INSIDE the body -- by program order if same-agent, by a fence
 placed between them if not. A non-zero distance is purely loop-carried, and ANY fence in
 the body separates trip `v` from trip `v+1`.

 A cross-block pair has no trip distance at all (`distance is None`) -- the two accesses are
 not two points in one repeating body -- so it is never `same_trip`.
```

## lds_hazards.py:3 <module>

```

LdsHazards -- the LDS hazard edges, derived from the SSA.

The three families a reachable state can hold are RAW residency, rotation WAR/WAW
and a cross-role/cross-agent alias. This analysis finds them on the SHARED
(LDS) side by asking which accesses can touch the same physical buffer, and -- crucially -- at what
TRIP DISTANCE.

WHY THIS EXISTS. The backend's own barrier placement is a per-token three-state phase machine over
the flattened instruction stream (`KernelWriter.postMainLoopBarrierCheckAndReset`). It can only see
what a static label says, so it cannot see a dependence whose two endpoints are the SAME static

```

## loop_shape.py:138 comment run

```
            # CROSS-CHECK, but only where there is a second statement to check against.  The back
            # edge's `GenXfer.adv` also says chunks-per-trip -- when there IS a generation to
            # advance.  A one-hop (direct-to-register) kernel stages nothing through shared, so it
            # has no Gens, no xfers, and `adv` says nothing; demanding agreement there would be
            # comparing against a default, not against a fact.

```

## loop_shape.py:3 <module>

```

LoopShape (#229) -- DERIVE a reduction loop's test position and trip count from the CFG, and check
that the steady region and the drain together cover the reduction exactly once.

WHY THIS EXISTS.  Whether a loop is PRE- or POST-tested is not a property anyone declares; it is
*where the conditional branch sits*.  Put the test on a block that also does work and the work
happens first (post-test, `do { body } while (p)`); put it on a block that only tests and the work
happens after (pre-test, `while (p) { body }`).  The same predicate means a different trip count in
the two shapes.  Nothing in GIR derived that, and nothing checked the consequence, which is exactly
how #228 survived: `loopir_to_gir` emits

```

## mem_tokens.py:144 comment run

```
            # AN AGENT-RELATIVE REGION NAMES THEM ALL (#245).  The coordinate says which region
            # THIS coordinate is; when the wave distribution makes the region a property of the
            # AGENT, the same instruction reaches a different region per wave, so the access may
            # touch any of them and must be ordered against every producer.  Reads only -- a copy
            # writes the one region its descriptor names, whoever runs it.

```

## mem_tokens.py:94 _unfenced_loop_carried_writes

```
Write refs whose LOOP-CARRIED edge nothing separates -- the defs that must span the ring.

    `Hazard.same_trip` states the rule: "a non-zero distance is purely loop-carried, and ANY fence
    in the body separates trip `v` from trip `v+1`."  So a loop-carried pair is covered exactly when
    its block has a fence, and `needing_fence()` -- the ONE place the fence/no-fence decision is
    made -- says which blocks get one.  Asked of `LdsHazards` rather than re-derived, so this and
    `FenceRegions` cannot disagree about which edges are covered.

    A SINGLE-WAVE KERNEL HAS NO CROSS-AGENT EDGE, hence no fence, hence nothing covering its
    loop-carried copy->read RAW -- and then the exact generation gives the producing copy and its
    next-trip consumer two DIFFERENT ids, so no def-use edge crosses the back edge at all.
    
```

## mem_tokens.py:3 <module>

```

MemTokenAssignment (#235) -- GIR names every LDS buffer, and the id IS the memory token.

WHAT A MEMORY TOKEN ACTUALLY IS. Not a label: an LDS PSEUDO-REGISTER. StinkyTofu's
`StinkyBuildImplicitDependencyPass` turns `MemTokenData{tokens}` into `RegType::LDS` operands --

 tensor_load / ds_write -> the token as a DEF (LDS producer)
 ds_read / global_store_async_from_lds -> the token as a USE (LDS consumer)
 barrier / signal / wait -> BOTH (a scheduling point)

-- and the ordinary def-use chain then enforces `producers -> barrier -> consumers`. So the token

```

## reg_band.py:3 <module>

```

RegBand -- VALIDATE the register rotation width W; never decide it.

W is decided by LoopModel (`translate._register_policy` intent -> `ring_slot` bakes the modulus),
carried into GIR on each register `Ref.reg_ring`. W is invariant under the schedule, and
fixed at S-search time, so GIR only cross-checks it -- deciding it here would be a second authority
that can disagree with the decoder (violating R-LEGAL).

This analysis returns, per (operand, group index), the single ring width seen -- and RAISES if a
group's register reads disagree on their ring (that would mean the LoopIR slot moduli are
inconsistent, a real defect). The RegGen pass consumes it to stamp each Ref; L3 reads W from
here rather than re-deriving it.

```

## region_increment.py:188 comment run

```
                    # AN UNPINNED REGION AXIS IS A VIOLATION, NOT A PASS.  This used to `continue`
                    # on the same rule `_split_copies` skips by, so a movement whose walk was
                    # DROPPED ENTIRELY reported clean -- the verifier agreeing with the defect it
                    # exists to catch (#170).  "Cannot tell" is the one answer a checker must never
                    # round down to "fine".

```

## region_increment.py:161 walk_violations

```
[] or human-readable G-WALK violations.  Two properties, both replayed in PROGRAM ORDER:

      1. EVERY SPLIT COPY LOADS THE REGION THE DESCRIPTOR IS ON.  Walking the block, the running
         position must equal each copy's own region coordinate when that copy issues.
      2. THE WALK CLOSES.  It ends back at region 0, because `tdmIncrementGir` then applies the
         plain chunk stride and any leftover compounds every trip.

    (2) alone is not enough, and finding that out cost a measurement: the first version checked only
    the net, which is zero even when every step has been hoisted to the block entry ahead of all the
    copies -- net-clean and uniformly wrong.  (1) is the property that actually matters; (2) is what
    makes the next chunk's stride correct.
    
```

## region_increment.py:131 comment run

```
            # CLOSE THE WALK.  The chunk stride `tdmIncrementGir` applies assumes the descriptor is
            # back at the chunk base, so anything left displaced at the block exit is walked back.
            # Emitting this rather than folding it into the stride is the point: the subtraction
            # used to live inside the increment, where nothing related it to the advance it undid.

```

## region_increment.py:120 comment run

```
                    # TIGHTLY BOUNDED: `after` is this unit's PREVIOUS copy, `before` is this one.
                    # The window matters more here than for the other marks -- `EARLIEST` with an
                    # open `after` hoists every step to the block entry, which nets to zero while
                    # leaving every copy reading the wrong region.  A descriptor step is only
                    # meaningful BETWEEN the two loads it separates.

```

## region_increment.py:37 _flat

```
The region tuple as one index, so a walk is arithmetic on integers.

 Mixed-radix over the operand's region axes, innermost last -- the same shape the rest of the
 model uses for a multi-axis coordinate.

 Returns `None` when an axis is UNPINNED, which is a real modelling gap and not a coord to guess
 at: a Phi-fused group may tile its members on DIFFERENT region axes, and the fused
 instance's coord then names only one member's axis -- that is #170, still open. The walk cannot
 be derived for such a movement, so this analysis emits nothing for it and `gir_to_rocisa`
 REFUSES if one ever reaches emission. Skipping here and refusing there, rather than raising
 here, keeps the analysis total for the MX scenarios that only ever get built in tests (MX is
 rejected under UseLoopModel) while leaving no silent path to wrong code.
```

## region_increment.py:3 <module>

```

RegionIncrementRegions (#236) -- WHERE the TDM descriptor steps between STORAGE REGIONS.

WHY THIS EXISTS. A region-split tile is loaded by one `tensor_load_to_lds` PER REGION,
and the descriptor has to move between them. `KernelWriterAssembly.globalReadDo` used to do that
walk inside itself -- load half 0, advance, load half 1 -- and `tdmIncrementGir` subtracted the
advance back off so the per-chunk round trip closed. GIR emits one copy Move per region and calls
the loader once per Move, so the two owners disagreed: `+2.splitInc` applied against `-1.splitInc`
removed, leaving the descriptor a region ahead every chunk and one load reading past the tile. All
MIWaveTile [2,2] cells miscompared (2026-08-16).

```

## short_path.py:433 _arm_is_unreachable

```
Is the `T < ...` arm dead code?  DERIVED from the peel-validity guard, not from M.

    This used to be `M == 1`, justified as "the arm's only step is unguarded because `T > 0` is
    trivially true, so it runs only at T == 0".  That reasoning was tied to a `T >= M` guard.  The
    guard is now strict (`T > M`, forced by the post-tested steady region -- #228), so the arm runs
    whenever `T <= M`, and at M == 1 that includes **T == 1**, an ordinary trip count.  The old rule
    would have folded an arm whose pairings provably break, on a path that really executes -- and it
    would have done so silently, because VACUOUS skips the break check.

    Derived form: the loop is entered iff `T > M`, so the arm takes every `T <= M`.  A reduction has
    at least one chunk (`T >= 1`), so the arm is dead only when `M < 1` -- and `M == 0` means there
    
```

## short_path.py:410 _folded_producer

```
The VALUE this consumer receives at `loc` on the FOLDED path (see `_trace`).

    The consumer is the same instruction on both sides, so it reads the same NUMBER of locations in
    the same roles -- but the location KEYS differ between the two chunk frames (the arms peel
    different chunk indices, hence different generations).  So look the location up directly when
    it happens to coincide, and otherwise fall back to the consumer's single shared/register use of
    the same space and operand.  A consumer that reads two locations of the same space and operand
    would make that fallback ambiguous; there is no such consumer today (a read sources one buffer,
    an Mma one register per operand), and if one appears the ambiguity must be resolved by carrying
    the role, not by guessing -- so it raises.
```

## short_path.py:325 comment run

```
        # F1 -- coverage: every COMPUTE the short arm performs must also happen on the folded path.
        # Restricted to `wmma` for the same reason F2 is (below): the two shapes deliberately read
        # differently, so a read present in one and absent from the other is the reset working, not
        # a gap.  The compute is what must be covered, and it is: an `mma`'s `(step, coord)` names
        # the same product in both.

```

## short_path.py:201 _trace

```
One forward walk of an arm: returns `(reach, keys, extras)`.

    `reach`  {consumer key: {location: the VALUE that reaches it}}
    `keys`   the consumer keys this arm defines
    `extras` instructions with no consumer key (a prologue fill; the drain's next-tile prefetch)

    TWO IDENTITIES, and conflating them is what made this analysis wrong (#227):

    THE CONSUMER KEY -- which instruction in the other arm is "the same" one:
        `(kind, step, operands, coord)`, checked unique.  Deliberately NOT the position within the
        step: the short arm runs at `dr = 0` while the drain keeps `dr = dr(theta)`, so
    
```

## short_path.py:177 pairing_cause

```
Classify one corresponded consumer's producer pair, or `None` when the two frames agree.

    ONE derivation, called only from `ShortPathFold.run`'s F2 loop -- it lives here rather than
    inline because the ladder's ORDER is load-bearing and a second copy would drift (see the
    recurring shape: the read-ahead traversal, the read's tree level and the fence cover were each
    one rule written twice, agreeing on every ordinary input and diverging on the degenerate one).

    `f_src` is the producer the FOLDED path supplies, `s_src` the one the SHORT ARM supplies.
    The first rung is the trap: AGREEING ON "NO PRODUCER" IS NOT AGREEMENT.  `f_src == s_src` at
    `ENTRY` means both sides say "nothing wrote this", and returning `None` there records no Break,
    which leaves `fold_breaks` empty and reports the folded path SOUND on the exact input that
    
```

## short_path.py:161 comment run

```
            # ONE def, MANY LOCATIONS. A coverage-wide read is a single definition that
            # fills every coordinate of its footprint, so it is registered at each of them --
            # otherwise a consumer at a non-leader coordinate looks unfed and the arm is judged
            # unable to stand alone.  `covered_coords` is `(coord,)` for every ordinary read.

```

## short_path.py:120 _free_coord

```
The FREE-TILE part of `coord` for `operand` -- the coordinate that selects a distinct
 physical register, as opposed to the rotation slot.

 A read's register is named by TWO independent things : the rotation
 `(group, slot)`, which covers the reduction axis, and the operand's own free-tile coordinate,
 which the model leaves to allocation. Omitting the second merges registers that are physically
 distinct -- `A(k1,m0)` and `A(k1,m1)` share a slot but are different VGPRs -- and every read of
 one then looks like it clobbers the other.

 Read PER OPERAND from `meta['free_modes']`, not by subtracting the reduction modes from the
 coord. The historical reason was that an `Mma` src Ref carried the whole `(K,M,N)` coord, so
    
```

## short_path.py:86 _frame_base

```
The CHUNK POSITION a block's `gdelta = 0` is stated against, ON THE FOLDED PATH (#230).

    This is the correction that made PGR1 and PGR3 decidable.  A drain Ref's `gdelta` is NOT
    relative to the loop's exit; it is relative to the LONG PATH's chunk position `T`, so drain step
    `i`'s own chunk is `gdelta = i - M`.  Resolving that against the phi's entry value -- base 0 --
    silently assumes `M` is a whole number of ring revolutions:

        resolved  = (entry + gdelta) % ring = (0 + i - M) % ring
        should be = chunk i               = (0 + i)     % ring        <=>   M = 0 (mod ring)

    At ring 2 that is true for M = 2 and false for M = 1 and M = 3, which is exactly the observed
    
```

## short_path.py:3 <module>

```

ShortPathFold (#183) -- may the `T < M` arm be FOLDED into the shared prologue/drain path?

THE TWO SHAPES.  TensileLite's scaffold and the LoopIR describe the same kernel differently:

    scaffold :  prologue -> Cond -> loop -> else -> drain    prologue+drain SHARED by both paths;
                                                              `T < M` just skips the loop
    LoopIR   :  Cond { prologue -> loop -> drain } else { short }      two DISJOINT arms

Folding means deleting the `short` blocks and letting the peel-validity FALSE edge fall into the
drain, so the `T < M` case is served by the prologue's copies followed by the drain's computes and

```

## swap_regions.py:206 _flow

```
Place the swaps for ONE physical pointer, via the shared ValuePlacementSolver.

        This analysis owns only what is specific to this register -- which instructions access it,
        what generation each demands, how a value re-frames across an edge, and the ring modulus.
        WHERE the defs go is `value_placement`, so the rule cannot drift from GrIncrementRegions'.

        What that replaced: a single forward walk that recorded a def at each point of demand and
        then patched the loop-carried case with a hand-written "back-edge closure".  Two objects
        deciding one thing, and the def always landed at the LATEST legal point (immediately before
        the access that demanded it) because the walk had no notion of where the old value died.
        The solver derives both from AVAIL/ANTIC, so the trip-bottom swap is the same kind of object
        as every other def rather than a special case bolted on afterwards.
        
```

## swap_regions.py:148 _edge_delta

```
Chunks the reduction timeline advances crossing `src`->`dst`.

    The difference of the endpoints' `chunk_base`, plus the trip advance on a back edge (the next
    trip's first body block sits `adv` chunks past this trip's first).  This is the conversion
    factor between the two blocks' frames; without it a steady value and a drain value are
    incomparable integers.

    PER PATH, not per block (#233).  `dst` may sit at a different position depending on which edge
    reached it -- the folded `T < M` entry lands in `drain0` at chunk 0 while the loop path arrives
    one chunk past the last steady trip -- so the destination's base is read from
    `path_chunk_base[src]` when it states one.  Subtracting two per-block integers instead put the
    
```

## swap_regions.py:106 _unemitted_edges

```
Edges into a block NO BACKEND EMITS -- the only paths the placement problem may exclude.

    REPLACES `_loop_bypass_edges` (#222/#234), which recognised "a `CondGoto` whose taken side
    enters a loop header" and dropped the other side.  Two things were wrong with that:

      * it was a CFG-shape PATTERN MATCH, not a derivation, and
      * it excluded the FOLDED `T < M` entry too -- an edge the emitted code really does take, into
        a `drain0` the backend really does emit.  Excluding a live path is not a scope decision; it
        is a hole, and it is the one #222 exists to close.

    The criterion now is the DECLARED fact: `Block.model_only` (#232), set by `FoldShortPathPass`
    
```

## swap_regions.py:33 _hop_access

```
`(unit, shared_ref)` if `inst` is a Move on `hop` ('read'|'copy'), else None.

 `unit` is the pointer identity, and it is NOT the same kind of thing on the two hops:

 read -> the operand name. A's read and B's read are different instructions at different
 addresses, so each owns its own LocalReadAddr.
 copy -> the Phi movement's member tuple. A fused group is ONE cooperative instruction over
 ONE descriptor; naming it by `refs[0]` would drop the other members
 and claim a single pointer without checking that a single pointer is what the
 movement has. `copy_unit` makes that a checked fact.

 Downstream -- `_ring_for_unit`, `_requirements`, `_operands`, `_flow` -- treats the unit as opaque, so
 the two shapes cost nothing beyond this function and the Mark payload.
```

## swap_regions.py:3 <module>

```

SwapRegions -- physical-pointer swap placement, as a REAL forward
dataflow analysis over the GIR CFG.

The physical LDS read pointer (LocalReadAddr) and the TDM write descriptor are SINGLE registers
whose selected generation must EQUAL the generation every access names. A swap is the only
instruction that changes one, so "where do swaps go" is a reaching-value problem: propagate the
pointer's value to every program point, and a swap is a DEF inserted exactly where the value the
next access requires differs from the value that reaches it.

WHY THIS IS A DATAFLOW AND THE PREVIOUS VERSION WAS NOT. The previous implementation used the

```

## uniform_placement.py:3 <module>

```

UniformPlacement -- the ONE LIVENESS check the empty-ledger
gate does not subsume.

 "for each `scope=block`/`cluster` `Await` over agent set `G`, verify its region-tree position
 DOMINATES every agent in `G` uniformly -- it lies on a node no `Branch`-over-residues or
 agent-discriminating `Cond` arm can cause an agent of `G` to skip" (L733)

The safety argument is race-freedom plus residency, and the ledger gate checks that every
obligation carries a selector; neither says the selector is REACHED by every agent that must
rendezvous at it. A block-scoped selector placed on a path only some waves execute does not

```

## value_placement.py:164 solve

```
[Placement], deterministically ordered.  `on_conflict(kind, detail)` reports a TOP meet
        or an unsplittable critical edge; it is expected to raise.

        Three def kinds fall out of the two fixpoints, and they are disjoint by construction:
          intra-block -- between consecutive accesses in b whose requirements differ;
          exit def    -- at b's end, when every successor agrees on a value b does not already hold;
          head def    -- at b's start, when what actually arrives differs from what b demands (which
                        happens exactly when a predecessor could not place an exit def because its
                        successors disagreed).
        
```

## value_placement.py:135 avail

```
Forward fixpoint over what LEAVES each block.

        The subtlety that a naive version gets wrong: a def placed at a block's exit CHANGES the
        value leaving it, so a successor must see the post-def value.  Propagating the pre-def value
        makes the successor re-derive the same transition and emit a SECOND def for it -- a real
        double emit, caught by the accessless-block test.  So `leave` (post-def), not `body_out`
        (pre-def), is what crosses the edge.

        Returns (avail_in, body_out, leave).
```

## value_placement.py:63 ValuePlacementSolver

```
Solve def placement for one register.

    `accesses`   -- {block label: [RequiredValue]} in program order.
    `preds`/`succs` -- the CFG to solve over, ALREADY filtered (back edges included, unmodeled paths
                   excluded).  Back edges must appear in both, or the loop-carried def is lost.
    `edge_delta(p, b)` -- value frame shift crossing p->b (added going forward).
    `entry`      -- label of the program entry block; `entry_value` is the value the scaffold
                   establishes before the body runs.
    `modulus`    -- ring size, or None for an unbounded domain.
    
```

## value_placement.py:3 <module>

```

ValuePlacement -- where the defs of a single-register resource go, as two real fixpoints.

Both pointer analyses solve the same problem for a different register: ONE physical register (an
LDS read pointer, a TDM write descriptor) must hold a specific value at each access, and a Mark
(`swap` / `gr_increment`) is the only thing that changes it.  "Where do the Marks go" is therefore
not a search -- it is the classic code-motion pair, and this module is the single implementation
both consume so the rule cannot drift between them.

    AVAIL  (forward)   avail_out[b] -- the value the register holds LEAVING b: the requirement of
                       b's last access, or a pass-through when b has none.

```

## analysis.py:3 <module>

```

GIR analysis INFRASTRUCTURE -- the base class + the lazy cached manager.

This module is infra ONLY. The concrete analyses each live in their own module under
`gir/analyses/` as an `Analysis` subclass:

 analyses/cfg.py Dominators, BackEdges (+ successors helper)
 analyses/gen_reaching.py GenReaching
 analyses/swap_regions.py SwapRegions (a region analysis -> PendingMarks)
 analyses/gr_increment.py GrIncrementRegions (a region analysis -> PendingMarks)
 analyses/dep_defuse.py DepDefuseAnalysis

```

## emit_plan.py:241 comment run

```
        # ONE STEP of the TDM descriptor between storage regions (#236).  Distinct from `gr_inc`:
        # that advances the movement to the next reduction CHUNK, this moves within one chunk's
        # tile between the regions it is split across, and the walk closes before the chunk
        # advance applies.  Separate acts because they are separate displacements with separate
        # strides -- conflating them is what left the descriptor a region ahead every trip.

```

## emit_plan.py:204 comment run

```
            # An operand partitioned into several register groups contributes ONE src Ref per
            # group, each with its own slot.  A single scalar `bufA` cannot name them, and the
            # loop below would keep whichever came last -- a silently wrong source register.
            # Multi-group MMA addressing is #118; refuse rather than pick.

```

## emit_plan.py:144 comment run

```
        # WHICH STORAGE REGION this copy fills (#236), and how many the movement has.  A
        # region-split tile is one `tensor_load` PER REGION and the descriptor is
        # walked between them by the `region_increment` Marks, so the emitter must issue exactly
        # one load here -- `globalReadDo` issues ALL of them and walks internally, which is the
        # double-ownership this carries the fact to avoid.  `regions == 1` keeps the ordinary path.

```

## emit_plan.py:138 comment run

```
        # frame-correct: `gdelta` is relative to the block's own `gen_rel` (see Block), so the
        # raw delta is not a generation on its own.  `blk_rel` is subtracted by the caller-supplied
        # block frame; with no block in hand here we can only use the absolute form, and a
        # loop-carried copy's generation is resolved by GenReaching instead.  `copy_unit` has
        # already checked every member agrees on it, so reading refs[0] here is sound.

```

## emit_plan.py:132 comment run

```
    # a COPY: * -> shared.  It names the Phi MOVEMENT (`unit`), not an operand -- see `refs.copy_unit`
    # and the act table above.  Carry the destination BUFFER GENERATION too: a copy is realized per
    # Move (like a read or a wmma), and the prologue peel issues SEVERAL copies of one movement into
    # DIFFERENT buffers, so the movement's name alone does not identify which copy this is.

```

## emit_plan.py:98 comment run

```
        # AND `k` IS WITHIN-REGION TOO, for the same reason `tile` is.  When the split cuts the
        # REDUCTION axis the region already carries `K_split`, so leaving it in `k` would count
        # that displacement twice -- once as `region * splitBoundary` and again as
        # `k * substepStride`.  Subtracting the region modes is a no-op for an MT split, whose
        # axes are free ones and so were never in `red` to begin with.

```

## emit_plan.py:3 <module>

```

Emit plan -- the PURE half of GIR -> rocisa (L3).

`plan_block(prog, phase)` walks a FINALIZED GIR block body (after the pipeline has applied swap /
gr_increment Marks and stamped tokens) and returns an ordered list of `EmitAction`s: the concrete
per-leaf instructions L3 must emit, with every index/generation/token already resolved from the
GIR nodes. It imports NOTHING from rocisa, so the index/order logic -- the correctness-critical
part -- is unit-testable standalone; the thin rocisa adapter (`gir_to_rocisa.py`) only maps each
action to the existing leaf emitter.

An EmitAction is one of:

```

## nodes.py:396 Program

```
The whole mainloop as one minimal CFG.

 blocks -- {label: Block}, insertion-ordered.
 entry -- entry block label.
 tiles -- interned Tile table (optional; may be empty in R0b).
 params -- PRESETS analyses derive facts from (StaggerU, MI/MT,...); not raw data.
 meta -- {S, roles, dtypes,...} scalar metadata.
 version -- bumped by any IR mutation so cached analyses invalidate.
 
```

## nodes.py:340 Block

```
One software-pipeline Phase as a basic block in the GIR CFG.

 phase -- 'prologue' | 'steady{n}' | 'drain{n}' | 'short{n}' | 'tail' (also the block LABEL).
 'short{n}' is the `T < M` degenerate arm: present only until
 FoldShortPathPass has ruled, then either folded away or kept as a real second path.
 loop -- steady: its CondGoto taken-target is itself (a back-edge).
 phis -- Gen phis (loop header): v = phi(entry, v_next). List of GenPhi.
 xfers -- back-edge transfers: v_next = (v + adv) % ring. List of GenXfer.
 body -- [Move | Mma | Mark], program order, concrete coords.
 term -- the terminator: Goto | CondGoto.
 role -- wave role (rho); 'all' until wave-specialization.
    
```

## nodes.py:304 successor_labels

```
THE successor list of a block -- the single authority every CFG walk uses.

    It lives here, next to the terminator definitions, because a second copy of this switch is a
    silent-wrong-answer waiting for the next terminator kind: `Program.walk_rpo` used to carry its
    own that handled only Goto/CondGoto, so a `CondChain` header (G1) or a `Return` sink (G4) had
    NO successors as far as reverse-postorder was concerned -- its successors fell through to the
    'unreachable, append in insertion order' path and `gen_reaching` would then visit them in the
    wrong order.  `analyses/cfg.successors` delegates here rather than repeating it.

    Targets are returned in program-significant order (a CondChain's arms first, in arm order,
    then its default) and are NOT filtered against `prog.blocks` -- the caller decides whether the
    pseudo-target 'end' is dropped.
```

## nodes.py:277 CondChain

```
Multi-exit terminator -- an ORDERED chain of guarded exits, then a default (G1).

    A loop header may need several typed early-exits taken BEFORE the loop is entered, all testing
    the SAME counter against different bounds: with a prefetch depth of `d` the trip count decides
    how much of the pipe can actually be filled, so a short trip enters a partially-drained variant
    instead of the steady loop.  That is a *chain* (`c <= d-1 -> ...`, `c <= d -> ...`, `c == 1 -> ...`,
    else fall through), not one two-way branch, and it is ORDERED -- the first satisfied arm wins,
    so the arms are a tuple and never a set.

    Modeled as one terminator rather than a fan of single-exit blocks so the header stays ONE
    block: its `phis`/`xfers` and its body belong to the header, and splitting it would force every
    
```

## nodes.py:264 Return

```
Function-early-exit terminator -- a SINK with no successors (G4).

    Some drain variants finish the whole kernel in place rather than falling through to a common
    exit (an optimized no-load-loop that runs the epilogue and returns mid-drain).  Modeling that
    as `Goto('end')` would be a lie in one specific way: `'end'` is a pseudo-target every analysis
    already filters out, so a `Return` and a fallthrough-to-exit would be indistinguishable, and a
    pass could "helpfully" retarget the edge.  A distinct node makes the sink explicit -- CFG
    successors are empty, dominance and back-edge detection skip it, and `verify_gir` accepts a
    terminator with no targets ONLY for this node.
```

## nodes.py:247 LoopBack

```
Loop terminator: run `body` for `trips` iterations, then go to `exit_target`.

    Replaces the `CondGoto` that used to carry the steady back-edge.  A `CondGoto` says "branch on
    this comparison"; that is an implementation of a loop, not a description of one, and GIR is the
    description layer.  The CFG is unchanged -- `body` and `exit_target` are the same two successors
    a `CondGoto` reported, so dominance, back-edge detection and every traversal behave identically.

    `label` is the scaffold hint (`LoopEndL`) that `ScaffoldMapPass` attaches, moved here from
    `Pred.label` because there is no longer a `Pred` on this edge to hang it on.
```

## nodes.py:224 Trips

```
A loop's DYNAMIC TRIP COUNT: `(var - sub) // div`, or the constant `-sub` when `var` is "".

 WHY A COUNT AND NOT A COMPARISON (#231). A trip count is what the model actually states -- the
 model's steady region is a RANGE, `[delta, T)` -- while a comparison is one ENCODING of it, and the
 encoding needs three facts that live outside the node to decode: where the counter starts, which
 way it steps, and where the test sits. None of those are shared with the backend: TensileLite
 initializes its `loopCounter` to the loop count, DECREMENTS, and exits at
 `counter <= endCounter`. Same trips, opposite direction. Holding the comparison meant every
 reader re-derived the count from assumptions, and two off-by-ones came from exactly that -- the
 post-test bound (#228) and the guard strictness it forced.
    
```

## nodes.py:179 Pred

```
A STRUCTURED symbolic predicate over trip-count symbols (no hardware): `lhs op rhs`.

    Read BY FIELD -- `lhs` (the counter symbol), `op`, `rhs` (a `Bound`) -- never by parsing a
    string.  It used to be a free-form `expr: str`, which meant a consumer that wanted to know
    "which counter does this test, and against what?" had to parse `'iter < T - 2'`; the
    multi-exit header (a chain of typed early-exits on the same counter) and the drain-multiplicity
    matrix both need exactly that comparison, structurally.

    label -- scaffold hint (e.g. 'toPGR1'), attached DOWNSTREAM by ScaffoldMapPass; advisory, so L3
            routes to the matching scaffold branch while a from-scratch backend lowers `render()`.
    
```

## nodes.py:157 Bound

```
The right-hand side of a terminator predicate: the constant `const` when `var` is empty,
    else the trip-count symbol `var` offset by `const` (negative for `T - M`).

    Two fields cover every terminator predicate the lowering emits or the scaffold needs -- the
    peel-validity guard (`T >= M`), the steady back-edge (`iter < T - M`), a short-loop step
    (`T > t`), and the typed early-exits (`counter <= PGR-1`, `counter == 1`).  This is
    deliberately NOT the LoopIR's full index `Expr` (mixed-radix terms, moduli, chunk-carry):
    those describe DATA coordinates inside a body, never a control-flow test, and duplicating
    them here would give GIR a second formula language to keep in sync.
```

## nodes.py:94 Move

```
THE data-movement verb: one hop down the memory hierarchy.

 srcs -- (Ref,) e.g. a global tile.
 dsts -- (Ref,) e.g. a shared tile at a Gen (a copy), or a register (a read).
 deps -- RAW + WAR edges carried from the LoopIR awaits (kind on each dep, / analysis).
 token -- the SEMANTIC completion token this Move produces/consumes, stamped by TokensPass
 (e.g. ('lds', generation) -- which LDS buffer a read consumes / a copy fills, so L4
 pairs read<->write; dep_defuse). None until TokensPass runs. A FACT, not a count.
 token_ids -- the BACKEND memory-token id(s) this Move's shared Ref touches, from
 `MemTokenAssignment` (#235). A token is an LDS pseudo-register, so these are the
 storage names the scheduler builds its def-use edges from; `token` above is the
    
```

## nodes.py:71 comment run

```
    # -- absolute (non-loop-carried) generation: the Fact-1 realization
    # prologue fills and drain reads have CONCRETE generations pinned by the LoopIR peel
    # (concrete cst slots), NOT the loop-carried steady Gen.  Such a Ref carries gen=None
    # and abs_gen=<the pinned generation int>.  Only STEADY refs carry a `gen` (a Gen phi).

```

## nodes.py:25 Tile

```
A logical operand fragment -- 'what data', no color / no address.

 operand -- 'A'|'B'|'C'|'acc'|'scaleA'|'scaleB'|'meta'|... (open set; identity of an Mma
 operand is read from here, -- there is no separate `role` map).
 space -- 'global'|'shared'|'register'.
 coord -- the subset of the 6 canonical modes this fragment covers, as a dict
 {mode: int|None}. NO `iter` ever appears (G-NOITER); the cross-trip relation
 lives in the `Gen` phi, not a coordinate.
 shape -- logical extent (element / register counts) -- a SIZE, not an address.
 
```

## nodes.py:3 <module>

```

GIR nodes -- the GEMM-dataflow IR (layer 2), per the lowering-design 

GIR is the mutable optimization surface between the LoopIR (layer 1, LoopModel's proven-legal
rolled schedule) and rocisa (layer 3). It is a real, verifiable CFG whose ONE SSA value class
is the loop-carried LDS `Gen`. It carries GEMM semantics only -- never machine
realization (R-SEMANTIC): a swap is a `Gen`-transition FACT, not a `v_xor`; there is no
address / VGPR-color / vector-width node.

This module imports NOTHING from rocisa, so GIR + its analyses + passes are unit-testable
standalone. Every node is a plain dataclass.

```

## __init__.py:3 <module>

```

GIR passes. Each pass is its own class in its own module; the ordered pipeline
is data in pipeline.py (the "backend" file).

 base Pass / StructuralPass base classes (the CFG-edit contract)
 fold_short_path FoldShortPathPass -- fold/keep the `T < M` arm, on a proof (#183)
 reg_gen RegGenPass -- register (group,slot) authority; W validated (B3)
 tokens TokensPass -- stamp each LDS Move's completion token from GenReaching
 collect_pending CollectPendingMarksPass -- run region analyses, stash PendingMarks (no mutation)
 placement PlacementPass -- resolve each Region to a concrete anchor (no mutation)
 apply_marks ApplyMarksPass -- the only BODY mutator: insert Marks at their anchors
 pipeline pipeline(the backend pass order) / run_pipelineR4 adds GuardSite into the collect set.

```

## apply_marks.py:3 <module>

```

ApplyMarksPass -- the only BODY mutator among the swap passes.

Inserts each placed PendingMark's Mark into its block body at the resolved anchor:
 - anchor (block, node) -> insert immediately before `node`
 - anchor (block, BLOCK_EXIT) -> append at end of the block body

After apply, `prog.pending` is cleared and `prog.bump` invalidates cached analyses.

```

## base.py:34 StructuralPass

```
A pass that may CHANGE THE CFG: add or remove blocks, retarget terminators, move the test.

    WHY THIS IS A SEPARATE KIND.  Everything downstream of the CFG is derived from it -- dominators,
    back edges, generation reaching, pointer placement, the fence cover -- so a structural edit
    invalidates a different and larger set of facts than a body edit, and it can change the SHAPE
    the model is expressed in (a post-tested loop and a pre-tested one need different bounds for
    the same trip count).  `ApplyMarksPass` described itself as "the ONLY IR mutator" for a long
    time; `FoldShortPathPass` (#183) made that false when it started deleting blocks, and nothing
    in the type system noticed.  Naming the kind is what lets the contract below be stated at all.

    THE CONTRACT.  A structural pass MUST, before it returns:
    
```

## collect_pending.py:3 <module>

```

CollectPendingMarksPass -- run the region analyses and stash their PendingMarks on
`prog.pending`. IR-UNTOUCHED (it only reads analyses and records their plan); the placement +
apply passes downstream turn the plan into body Marks.

R2 ran SwapRegions only; R3 adds GrIncrementRegions; #206 adds FenceRegions. GuardSite
(gsu_guard) lands in R4.

FenceRegions is a region analysis like the others, which is why it needs no new machinery: it
returns PendingMarks with a Region, and Placement/Apply insert them. Its Regions are PINNED (the
cover already chose the point), and because a Region's anchors are opaque NODE references rather
than indices, the swap/gr_increment marks inserted alongside cannot invalidate them.

```

## early_exit.py:59 comment run

```
            # THE ENTERED STEP HANDLES AN EARLIER CHUNK (#233).  On the long path `drain{M-k}` runs
            # `M-k` chunks past the last steady trip; entered directly at `T == k` it does the work
            # of chunk `k-1`.  `path_chunk_base` is the per-edge frame the join needs, exactly as
            # the coverage records for its own rewired edge.

```

## early_exit.py:3 <module>

```

EarlyExitPass (G1/G3) -- give the CFG the `T < M` entries the scaffold actually takes.

THE EDGE THAT WAS MISSING.  With peel depth `M` the drain is a chain of `M` steps, and a trip count
`T <= M` does not run the steady loop at all: it enters the chain ALREADY PARTIALLY DRAINED, at step
`M - T`, because the prologue only managed to fill `T` of the `M` generations.  `FoldShortPathPass`
records that as an obligation it cannot discharge ("entering the drain chain at step `M-T`,
G3/#148") and TensileLite implements it -- `_lmDrainStep = (PGR-1) - remainPgr` makes the NGLL
`drain0` and the NLL `drain1`, and the `toPGR1` label inside the NGLL sequence is the branch that
SKIPS `drain0` when the counter is 1.

```

## fold_short_path.py:52 comment run

```
                # NAME THE CONFIG.  This raise runs in a joblib worker over a whole matrix, so
                # without the shape the message says "a kernel" and localizing costs a rebuild per
                # guess (2026-08-21: the mxf8 `DepthU 512` cell took three).  `short_loop` carries
                # T and M -- the two numbers that put us on this path at all.

```

## fold_short_path.py:3 <module>

```

FoldShortPathPass (#183) -- the MERGE pass: coverage the `T < M` arm into the shared prologue/drain
path when that is provably legal, keep it as its own path when it is not, and raise when neither
shape works.

WHY A PASS EXISTS AT ALL.  TensileLite's scaffold and the LoopIR describe the same kernel with
different STRUCTURE:

    scaffold :  prologue -> Cond -> loop -> else -> drain    prologue+drain SHARED by both paths
    LoopIR   :  Cond { prologue -> loop -> drain } else { short }      two DISJOINT arms

```

## pipeline.py:119 _check_rotation_waw

```
The premise under which a rotation WAW needs no ledger row of its own.

    Non-interference covers WAW as well as WAR, so ordering a rotating buffer really means
    `WAR and WAW`.  Our ledger files only the WAR (`ledger.py`: RAW-residency,
    inplace-WAR/rotation-WAR, crossing-RAW, readahead-residency; `grep -rni waw Tensile/LoopModel`
    returns nothing).  That is CORRECT, but only under a premise the decoder never established:

        "Where every written generation is read and the read covers the write ... the WAW is
         implied by the WAR *through the residency RAW* ... so the WAW adds nothing there, and no
         second ledger row is filed.  It is stated only so `O(m)` is total without that decoder
         invariant."
    
```

## pipeline.py:99 comment run

```
                    # A Phi-FUSED movement names the GROUP -- `op` is a `unit` tuple like ('A','B') --
                    # while `SharedTouch.operand` names ONE operand.  That asymmetry is deliberate (a
                    # copy names a movement, a read names an operand), so the group is expanded to
                    # its members here rather than either side being bent to match.

```

## pipeline.py:71 _check_block_scope_covered

```
`Await.scope` is LOAD-BEARING here (#276-S2), and this is what makes it so.

 gives every obligation a proc-scope: "wave" where program order plus a per-wave counter
 already covers the edge, "block" where they provably cannot -- `s_wait_loadcnt` retires THIS
 wave's loads and says nothing about another's, so under a cooperative fill a reader waiting its
 own counter has no edge to the agent that actually wrote the bytes.

 UNTIL NOW THAT FIELD WAS INERT. It reached a diagnostic tag string and nothing else, so the
 model's claim about which edges are cross-agent was never confronted with the analysis that
 places the fences. The barrier that actually makes these kernels correct comes from
 `LdsHazards` -> `FenceRegions`, which probes LDS buffer overlap -- a DIFFERENT derivation of the
    
```

## pipeline.py:32 pipeline

```
The backend pass order (data): coverage -> reg -> tokens -> collect -> place -> apply -> scaffold-map.

    FoldShortPathPass runs FIRST because it rewrites the CFG every later analysis reads: folding
    ScaffoldShapePass runs FIRST and is the ONLY stage allowed to change the CFG: it folds the
    `T < M` arm, adds that arm's early-exit edges into the drain chain, and labels the terminators.
    Every pass after it is an analysis over that CFG -- reaching generations merge over `preds`,
    hazards pair by reachability, the fence cover quantifies over "every path reaching the
    consumer" -- so an edge introduced later does not make those wrong, it makes them answers to a
    different question, with nothing to detect the difference.  Keeping the shaping in one stage
    ahead of everything is what makes that structural rather than an ordering convention.

    RecordUnemittedPass runs near the end because it COUNTS what the model-only blocks ended up
    carrying, and the Marks land in them mid-pipeline -- counting at coverage time reported zero (#232).
```

## pipeline.py:3 <module>

```

The GIR pass pipeline -- pass ORDER is DATA (the "backend" file, a la
StinkyTofu's Gfx1250Backend).

The ONE backend pipeline :
 ScaffoldShapePass the CFG is made FINAL here: coverage the `T < M` arm (#183), add the
 `T < M` early-exit edges into the drain chain (G1/G3), then attach
 TensileLite's scaffold labels. Everything after it is an analysis
 OVER this CFG, so an edge added later would silently invalidate them
 RegGenPass register (group,slot) authority; W validated from LoopIR (B3)
 TokensPass stamp each LDS Move's completion token from GenReaching

```

## placement.py:3 <module>

```

PlacementPass -- resolve each PendingMark's Region to a concrete anchor.

Policy travels with the Region (design "placement by swap TYPE -- a user directive, not just
sigma"), because it differs by what the Mark mutates:

 EARLIEST -- every TDM descriptor change (`gr_increment` AND the copy-hop `swap`): land immediately
 AFTER `region.after`, the access whose use of the descriptor this supersedes. Every
 cycle between the update and the `tensor_load` it feeds is overlap.
 MIDPOINT -- the LDS read pointer: land midway between the last use of `gen_from` and the first use
 of `gen_to`.

```

## record_unemitted.py:3 <module>

```

RecordUnemittedPass (#232) -- count what the model-only blocks are carrying away, LAST.

`FoldShortPathPass` decides WHICH blocks no backend emits (the kept `T < M` arm -- TensileLite owns
that runtime branch through its own `toPGR1` path) and records the reason.  It cannot record HOW
MUCH is discarded with them, because it runs FIRST: the fences and swaps that end up in those
blocks are placed later, by `CollectPendingMarksPass` / `ApplyMarksPass`.  Counting at coverage time
reported zero every time, and `verify_gir`'s G-EMIT caught exactly that -- 6 real marks against a
recorded 0.

So the count is taken here, at the end of the pipeline, and G-EMIT then holds the two against each

```

## reg_gen.py:3 <module>

```

RegGenPass -- the register (group, slot) authority; W is READ, not decided.

The lowering already assigns each register read Ref its concrete (group index, slot) and carries
the rotation width W on `Ref.reg_ring` (from the LoopIR slot modulus). This pass:
 - runs RegBand to VALIDATE W is consistent per (operand, group) -- raising on a LoopIR defect;
 - is the sole author of the register color/slot (here: it confirms every register read Ref has a
 concrete slot; a None slot is a lowering bug).

It does NOT decide W (that is a LoopIR fact) -- deciding it here would be a second
authority (R-LEGAL). IR-untouched today (validation only); it stays a Pass so a future coloring
change has a home.

```

## scaffold_map.py:86 comment run

```
            # a CondGoto whose taken-target is this block itself is the steady back-edge; otherwise
            # (taken-target is another block, e.g. steady) it is the peel-validity entry guard.
            # the tail region's back-edge is its own scaffold label, not the steady LoopEndL --
            # distinguished by the block's PHASE, the structural fact, never by a name test.

```

## scaffold_map.py:73 comment run

```
                # A counted loop states its TRIP COUNT, not a comparison (#231), so there is no
                # `Pred` to hang the hint on -- the label lives on the terminator itself.  Its role
                # needs no inference either: a `LoopBack` IS the loop's own exit, so it is the
                # back-edge label, or the tail's when the block is the tail region.

```

## scaffold_map.py:35 comment run

```
    # a multi-exit header's guarded exits (G1): typed early-exits taken before the loop is
    # entered, each landing in a progressively-more-drained variant.  Indexed by ARM POSITION --
    # the chain is ordered, so position IS the identity of the exit; which concrete variant each
    # position denotes is the drain-multiplicity question (G3), not a naming question here.

```

## scaffold_map.py:3 <module>

```

ScaffoldMapPass (the LoopIR->TensileLite-scaffold mapping) -- the ONE place that maps the GENERIC,
backend-agnostic control flow (the peel-validity guard, the steady back-edge, the drain chain) onto
TensileLite's favored scaffold LABELS (`toPGR1`, `LoopEndL`, `NoGlobalLoadLoop_k`).

Why a pass, not the lowering: the pure LoopModel core emits only generic guard KINDS, and `loopir_to_gir` builds a
generic reducible CFG whose terminators carry LABEL-FREE `Pred`s. All TensileLite scaffold naming
is concentrated HERE, so a from-scratch backend that ignores the scaffold simply skips this pass,
and the coupling to TensileLite's openLoop/closeLoop label set lives in exactly one file.

The label is ADVISORY (`Pred.label`): a consumer whose own loop-count scaffold already emits the

```

## scaffold_shape.py:3 <module>

```

ScaffoldShapePass -- the ONE CFG-shaping stage: fold, then early-exit, then label.

WHY THESE THREE ARE ONE PASS.  They are the whole of "make GIR's CFG the CFG the scaffold will
actually run", and every later stage is an ANALYSIS OVER THAT CFG -- `GenReaching` merges over
`blk.preds`, `LdsHazards` pairs by reachability, `FenceRegions` covers "every path reaching the
consumer", `MemTokenAssignment` names buffers off the reaching generation.  An edge added after any
of them has run does not make that analysis wrong so much as make it an answer to a different
question, and nothing detects that: the result is simply stale in a way the version counter cannot
see, because the pass that would have bumped it ran too late.

```

## tokens.py:75 comment run

```
    # A read can only await at the granularity its PRODUCER offers.  Key on the region ONLY when
    # the completion unit that produces this value actually emits per-region movements: a COARSE
    # copy feeding SPLIT reads (line 523 -- "a kernel may keep the copy coarse while the
    # read is fine") has ONE completion covering every region, so keying the reads by region
    # would make them await a token no copy ever stamps.

```

## tokens.py:3 <module>

```

TokensPass -- stamp each LDS Move with its completion token.

Replaces the old walker's `_stepReadToken`: a read's `sync LDS%u` token must name the LDS buffer
GENERATION it consumes, so L4 (SIA4/StinkyTofu) pairs the read with the WRITE that filled that
buffer. The generation is a GIR FACT (GenReaching over the shared Ref), so the token is a direct
map -- no seed, no flip-on-change. A copy's token names the buffer it fills.

Token form: ('lds', completion-unit, (generation,)[, region...]) -- a semantic tag, NOT a hardware
token index (L3 maps it to memTokenLdsBuffer0/1). The generation tuple is a one-element MUST-set:
"this access touches this buffer". See `_buffer_generation` for why it was briefly a set of all of them. DepDefuse is consulted so a read with no RAW LDS

```

## quantum.py:147 comment run

```
            # `reg_slot` IS THE CARRIER, not the slot.  One instruction writes one run of
            # `wtRegStride` registers, so the run's base is the instruction's ordinal -- which is
            # exactly what `carrier` is.  `slot` is the position INSIDE that run, which the
            # CONSUMER needs to find its half; handing it to the leaf as a base renamed every
            # scale register and failed the whole mxf8 matrix (2026-08-20, 16/16).

```

## quantum.py:101 plan_quantum

```
Decide, for every read act, whether it emits -- from theta's SUPPLIED merge, not from a rule here.

 `quantum_of(operand) -> ir.QuantumMap | None` is theta's own field, and `None` (the default) is the
 identity merge: one instruction per act, which is every A/B read. `extent_of(operand) -> int`
 bounds the tile scan for a carrier preimage; it defaults to the acts actually present.

 THIS DOES NOT MODEL THE FOLD, it OBEYS it. The act whose tile IS its own carrier emits; the
 rest ride it. There is no generation test and no narrowing: the merged coordinates are one
 the axes it varies over point by construction, so a straddle is not something this layer can observe,
 and re-deriving one here is what put a half-wave mechanism onto the region axis (see the header).
    
```

## quantum.py:28 Decision

```
What `leaves.emitLdsReadTile` must do with one read act.

    `emit` -- issue an instruction (False = this act's payload rides a leader's instruction).
    `narrowed` -- the group was not mergeable, so the coverage was cut to one tile and this act emits
                 its OWN narrower load.  The leaf reads this to pick the narrowed read context.
    `reg_slot` -- the register-side index when it diverges from the tile index; None = the leaf's
                 ordinary derivation.
    `tokens` -- THE UNION OF THE LDS MEMORY TOKENS OF EVERY ACT THE INSTRUCTION COVERS, or () when
                 the group is single-generation and the act's own ids already say everything.  A
                 merged movement names a SET of generations (Sec 2.2): the one instruction really
                 does read both buffers, so it must be ordered against both producers.  Labelling
    
```

## quantum.py:3 <module>

```

The MOVEMENT QUANTUM: which tiles one instruction carries, decided on the emit plan.

WHAT THE QUANTUM IS. Every hop moves a determinate block of data in ONE instruction, and the set
of axes that one instruction covers is the hop's **coverage**. By those axes are *absent* from
the hop's the axes it varies over, so the coordinates inside one coverage are ONE the axes it varies over point -- hence
`step(element, p)` gives them **one step, therefore one generation**. The model states this per hop
*symmetrically*: it governs a shared->register READ (a wide multi-tile `ds_load`) exactly as it
governs a coarse global->shared copy.

WHAT GOES WRONG WITHOUT IT. GIR emits one read act per tile -- that IS theta's coverage, one tile wide.

```

## refs.py:96 covered_coords

```
Every coordinate ONE reference fills -- the coverage's ABSORBED AXES, re-inserted.

 `ref.covers` is `((axis, extent),...)`: the inner modes this one instruction SPANS, which
 `geometry.quantum_axes` has already removed from the read hop's the axes it varies over. Empty (the default)
 means the reference fills exactly `ref.tile.coord` -- every ordinary read.

 THIS IS THE SECOND HALF OF THE SUBTRACTION AND IS USELESS ALONE. Because the axis is gone from
 the read's the axes it varies over, the def's `coord` does not name it while every consumer's does; the cross
 product below is what makes the one def readable as filling all of them. Applying the
 subtraction with this missing is measurable, not subtle -- the short arm reports "16 consumers
 read a location the arm never writes" at the mxf8 triage shape.
    
```

## refs.py:56 copy_unit

```
`(members, refs)` for a global->shared copy Move, or `(None, None)` if `inst` is not one.

    `members` is the tuple of operand names the movement carries -- `('A',)` unfused, `('A','B')`
    for the Phi-fused AB group that multi-wave TDM realizes as one aliased descriptor.  `refs` is
    the matching tuple of shared destination Refs.

    Why the tuple and not `refs[0].tile.operand`: `loopir_to_gir._convert_load` builds one Ref per
    fused token, so a fused Move genuinely has several shared dsts.  Naming the movement by the
    first one drops the rest -- which happens to give the right INSTRUCTION COUNT when the
    descriptor really is aliased (the multi-wave TDM case), and is simply wrong the moment a fused
    group keeps two pointers.  The count being right by accident is not the same as the aliasing
    
```

## refs.py:42 _rotation_state

```
The ROTATION STATE a shared Ref names -- what a pointer register would have to hold to serve
    it -- as a comparable key.

    Deliberately NOT the Gen itself.  `Gen.id` distinguishes the independently-staged buffer sets
    (A-shared vs B-shared), and a fused movement fills BOTH: A's LDS region and B's are different
    memory, so different ids is the correct model, not a discrepancy.  What makes them ONE pointer
    is that they rotate in LOCKSTEP -- same ring depth, same offset -- so a single register value
    serves every member at every point.  Comparing ids would reject every real fuse; comparing the
    rotation state is the actual soundness condition.
```

## refs.py:3 <module>

```

refs -- the shared Ref accessors for a Move's hops.

Small on purpose, and shared on purpose: three modules need the SAME answer to "which operand(s)
does this Move's shared hop belong to", and each of them used to compute it by taking the first
shared Ref. That is wrong for a Phi-fused movement, which carries one Ref PER MEMBER.

The distinction this module encodes:

 a READ is per operand. A's read and B's read are different instructions at different
 addresses; their "unit" is a singleton by construction.

```

## region.py:31 Region

```
A legal placement window over one block's body (opaque ordinals).

    block  -- the block LABEL the placement point lands in.
    after  -- place strictly AFTER this anchor (a body node, or BLOCK_ENTRY).
    before -- place strictly BEFORE this anchor (a body node, or BLOCK_EXIT).
    policy -- where in the window to land.  It travels with the Region because it differs by what
             the Mark mutates, so a single global policy in PlacementPass is wrong:

               EARLIEST -- every TDM descriptor change (`gr_increment`, copy-hop `swap`).  The
                          descriptor feeds `tensor_load`, so every cycle between the update and the
                          load is overlap.
    
```

## region.py:3 <module>

```

Region + PendingMark -- the analysis->placement->apply insertion model.

A `Mark` (swap / gr_increment / gsu_guard) has a LEGAL PLACEMENT RANGE, not a single correct
point: a read-hop swap gen_from->gen_to may sit anywhere after the last access using gen_from
and before the first access using gen_to. Baking a fixed body index into an analysis is
premature realization (the mistake the old walker made).

So GIR mirrors StinkyTofu's waitcnt/liveness shape in three layers:
 1. a region ANALYSIS returns a plan of PendingMarks (IR untouched);
 2. a PLACEMENT stage picks a concrete point inside each Region (IR untouched);

```

## render.py:276 comment run

```
    # PRESETS, not theta.  `Program.params` is a second input channel -- facts a region analysis places
    # from but the decoder never derived -- so a dump that shows only `meta` cannot explain why a
    # `gl2_prefetch` Mark is in the body.  Printed under its own heading precisely so it is not
    # read as a theta fact.

```

## render.py:222 _theta_facts_lines

```
The theta facts the BLOCKS ARE DERIVED FROM, which the block listing itself cannot show (#287).

    A GIR block prints coordinates and operands.  Whether a given coordinate is a REGION, whether
    that region is the AGENT's rather than the coordinate's, which operands share one Phi-fused
    movement, and how many storage regions the movement actually has are all facts of the theta the
    program was built from -- invisible in the body, and each one changes how the body must be read.

    THIS IS NOT COSMETIC, AND THE COST IS ON RECORD TWICE.  `translate.py:519` carried a conclusion
    ("the read's region needs rho") justified by a `check_region_coverage` message; the check was
    asking a COORDINATE question about an AGENT fact, and it survived a full day of investigation
    because nothing in the dump said the operand was agent-relative.  #176 is the same failure at a
    
```

## render.py:209 comment run

```
    # THE EDGE, READ OFF THE CFG.  This used to assert unconditionally that the guard's false edge
    # targets the drain chain.  That holds only on FOLD/VACUOUS, where `fold_short_path` retargets
    # it; on SPLIT the pass returns early and the edge still points at `short0`, so the sentence
    # contradicted the block listing four lines below it.  A structural claim that the dump can
    # simply LOOK UP must be looked up.

```

## render.py:169 _short_loop_lines

```
The `T <= M` arm: lowered to real `short{i}` blocks (#219) and then FOLDED into the drain
 chain or KEPT, per `ShortPathFold` (#183/#220-223). This header reports which happened.

 Without this the dump cannot be checked against the LoopIR at all: the LoopIR root is
 `Cond(T >= M, then=[prologue, steady, drain], els=[short])`, while the
 GIR hoists the prologue OUT of that guard and points the false edge at the drain chain -- so a
 reader diffing the two sees an `else` arm that vanished and a guard that changed meaning, with
 nothing in the dump saying either was intentional. `meta['short_loop']` records the arm's
 asserted shape; printing it makes the drop, and what the scaffold is expected to reconstruct,
 an explicit checkable claim rather than a silent absence.
```

## render.py:136 _pred_str

```
A terminator predicate WITH its scaffold hint.  `Pred.label` is attached downstream by
    ScaffoldMapPass and is the whole reason a consumer can route an arm to the right scaffold
    region -- a dump that omits it makes every terminator look label-free and unroutable, which is
    indistinguishable from ScaffoldMapPass not having run.

    READ THE LABEL AS A ROUTE, NOT AS AN IDENTIFICATION.  This docstring used to gloss `toPGR1` as
    "the peel-validity entry guard", which reads as though a `toPGR1` label identified that one
    guard.  It does not: ScaffoldMapPass stamps `toPGR1` on EVERY non-back-edge CondGoto, so the
    short arm's per-step validity guards all render as `toPGR1` too and three distinct predicates
    look like one repeated guard.  The label says where the arm goes, not which arm it is; the
    predicate beside it is what distinguishes them.
```

## render.py:93 comment run

```
        # `scope` is printed because it is now LOAD-BEARING (#276-S2): a "block" dep asserts this
        # edge needs a cross-agent fence, and `pipeline._check_block_scope_covered` fails the build
        # if the hazard analysis does not agree.  Printing only the non-default keeps the common
        # per-wave edge terse while making the claim visible wherever it is made.

```

## render.py:27 _gen_str

```
The buffer this Ref names: a loop-carried Gen IDENTITY, an absolute generation VALUE, or a
    register rotation slot.

    THE TWO GENERATION FORMS ARE SPELLED APART.  They used to render as `gen0[v]` and `gen=0` --
    one character between an SSA identity and a concrete integer, in the same dump.  They answer
    different questions (which phi feeds this vs which buffer this trip lands on), and a reader
    scanning for one silently accepts the other.

    THE SLOT IS NOT AN ALTERNATIVE TO THEM.  A Ref can carry a generation AND a register
    residence; chaining these as `if/if/if` with early returns showed only the first, so a shared
    read's destination register was invisible whenever its source generation was pinned.  All
    
```

## render.py:3 <module>

```

render_gir -- a human-readable DEBUG VIEW of a GIR Program (not the IR itself).

This is a RENDERING, not a serialization -- no field here is authoritative, and nothing in the
compiler reads it back.

It is a debug view for humans, and that is the whole contract.  This header used to credit "the
R0b round-trip check: the printed Program must contain every read / wmma / copy the LoopIR
unrolled view emits, in program order, per block".  No such check exists over this text.  The op
parity that IS checked runs on `gir_counts` and on `emit_plan`'s acts, never on these strings, so
a reader trusting the header would believe the dump was under test when it is not.

```

## verify.py:186 _check_tokens

```
G-TOKEN (#235): the memory-token numbering is a correct naming of LDS storage.

    A token IS an LDS pseudo-register -- StinkyTofu gives a producer the token as a def, a consumer
    as a use, and a barrier as both -- so the numbering is not decoration: it is the alias relation
    the scheduler will believe.  Three properties, none assumed:

      V1  DISJOINT STORAGE NEVER SHARES AN ID.  True by construction (the key is
          (unit, region, generation)), and checked anyway, because the construction is the thing
          most likely to drift -- `_regions_of` gaining a case, a fuse key changing spelling.
          Sharing where disjoint is lost parallelism, not a miscompile, so it reports as a
          precision failure with both keys named.
    
```

## verify.py:170 _check_walk

```
G-WALK (#236): the TDM region walk CLOSES in every block.

    A region-split tile is loaded one region at a time and the descriptor steps between them.  Those
    steps must net to zero per block, because `tdmIncrementGir` then applies the PLAIN chunk stride
    -- anything left over compounds every trip.

    This identity is the bug that reached hardware.  `globalReadDo` walked the regions internally
    (+1 per split load) while `tdmIncrementGir` subtracted one back, and GIR calling the loader once
    per region made it `+2` against `-1`: a descriptor a region ahead every chunk and one load
    reading past the tile.  Two owners of one walk, and no check relating them.
```

## verify.py:136 comment run

```
    # the `T < M` short arm (#183).  It is lowered to real blocks, then either FOLDED into the
    # shared prologue/drain path or kept as a second path -- so the legal counts are exactly 0 or M,
    # and which one it is must agree with the verdict FoldShortPathPass recorded.  A count between
    # the two means a pass dropped some steps, which is the silent-truncation failure this replaces.

```

## verify.py:49 comment run

```
    # A gl2_prefetch names NO unit and NO operand, and that absence is the fact (#216): the L2
    # warm-up runs off its own address VGPRs, never touches a TDM descriptor and feeds nothing in
    # this trip, so there is no movement for it to be attached to.  `depth` is the preset it came
    # from and `block` the steady body it belongs to -- enough for a dump to show one prefetch per
    # chunk without re-running the analysis, which is the whole claim worth checking.

```

## verify.py:44 comment run

```
    # A region_increment moves the TDM descriptor between STORAGE REGIONS of one split tile (#236).
    # `steps` is SIGNED -- forward to a later region, negative to walk back -- and the walk must CLOSE
    # in every block, which G-WALK checks rather than trusting.  `from_region`/`to_region` are the
    # endpoints, so a dump shows the walk without re-deriving it.

```

## verify.py:38 comment run

```
    # A swap additionally names the pointer, but WHICH name depends on the hop: a read swap moves
    # an operand's LocalReadAddr (`operand`), a copy swap moves a Phi movement's descriptor
    # (`unit`, a member tuple).  The choice is checked in `_check_marks`, not spelled here, because
    # this table has no place to express "one of, by hop".

```

## verify.py:32 comment run

```
    # A `fence` names the buffers it ORDERS (a must-set -- "this fence orders all of these", the
    # reading a barrier's token already has) and the proc-scope it must reach (for a
    # cross-wave reader/refiller inside one block, `block`).  `kinds`/`edges` are provenance --
    # which hazard families it discharges and how many -- so a reader of the dump can tell a
    # coalesced fence from a stray one without re-running the analysis.

```

## verify.py:3 <module>

```

verify_gir -- the GIR well-formedness verifier.

Checks the G-* invariants. Raises RuntimeError on the first violation with a precise
message; passing means the Program is a well-formed GEMM-dataflow CFG (R-LEGAL: entry is
legal, passes may create illegal intermediates but must restore verify_gir before the pipeline
ends).

Invariants :
 G-GEN-SSA every Gen has exactly one def (a phi or a GenXfer) AND a use; a def with no use
 within the modeled region is an error (a dropped consumer is the bug this catches).

```

## verify_dataflow.py:251 check_refill_splits_consumers

```
A HOISTED refill must not be placed BETWEEN two consumers of the generation it overwrites.

    THE ONE DEFECT SHAPE NO COORDINATE CHECK CAN SEE.  `check_register_dataflow` verifies that the
    register a wmma reads holds the `(tile_flat, k_flat)` the wmma asks for.  A read-ahead writes
    the NEXT GENERATION of the SAME coordinate, so when it lands between two consumers of the
    current generation every coordinate still matches and only the DATA is a trip early:

        wmma idx0=0,idx1=0  reads A(g0, buf0, tile0)      <- consumer 1, current generation
        read A buf=0 tile=0 adv=2                          <- refill, NEXT generation
        wmma idx0=0,idx1=1  reads A(g0, buf0, tile0)      <- consumer 2 gets the WRONG trip
    
```

## verify_dataflow.py:215 check_address_keys

```
The ADDRESS-level check, which needs one geometry fact the plan does not carry.

    `check_source_coverage` proves the reads name distinct SOURCE coordinates.  That is not the
    same as distinct ADDRESSES.  The leaf builds an address from TWO coupled choices:

        region term :  `region * splitBoundary`  if the regions are PACKED, else 0
        coordinate  :  the WITHIN-REGION one     if PACKED, else the FLAT one

    and they must move together.  Take the region term away (contiguous) while keeping the
    within-region coordinate and there is nowhere left for the region to live, so every region
    `r > 0` lands on region 0's bytes -- with the plan still perfect by every other check.
    
```

## verify_dataflow.py:191 comment run

```
            # EXACTLY ONCE is per COORDINATE, not per act (#312).  A Phi-folded read is one act
            # delivering `q` source coordinates, so counting acts would report the `q-1`
            # non-leaders as never read; counting the coordinates it FILLS keeps the conservation
            # law intact and still catches a genuine duplicate -- two carrier groups that overlap
            # put the same coordinate in `got` twice, exactly as two unfolded reads would.

```

## verify_dataflow.py:3 <module>

```

Semantic verification of the EMIT PLAN -- "does this schedule compute the right GEMM?" -- checked on
the plan alone, with no rocisa, no assembler and no GPU.

`verify_gir` checks that the IR is WELL-FORMED (blocks reachable, terminators sane, Refs resolved).
This module checks that it is CORRECT: that every wmma reads the data its own coordinate names,
that every value a read fetches is used before it is overwritten, and that the reads of one steady
trip cover the operand exactly once.

WHY IT EXISTS.  Every TDMSplit defect of 2026-08-16..18 was found by running kernels on hardware
and bisecting a pass/fail matrix -- a build cycle each, and a numeric miscompare says only "wrong",

```

## gir_to_rocisa.py:522 _token_ids

```
The memory-token id(s) this act touches -- GIR's own numbering (#235).

        A token is an LDS PSEUDO-REGISTER: StinkyTofu gives a producer the id as a def, a consumer
        as a use, and a barrier as both, then the def-use chain orders them.  So the id set IS the
        alias relation, and GIR is the layer that knows it -- `MemTokenAssignment` names every
        buffer `(unit, region, generation)` and `TokensPass` stamps the ids on the Move.

        THIS REPLACES A MUTABLE SIDE CHANNEL.  The ids used to reach the leaf through
        `writer.states.ldsTensorTokenIdx` / `ldsReadTokenIdx` -- one int, written here and read back
        later -- which is #200's recorded concern and the reason two refusals had grown here: a
        single int can express the buffer parity and nothing else, so a ring deeper than 2 and a
        
```

## gir_to_rocisa.py:485 _emit_copy

```
Realize ONE global->shared movement, for the buffer generation GIR's Move names.

        One call per Move -- so the prologue's M peel fills each emit, which a single pre-built
        module per operand could not do.  Everything else about the load (addressing, vector width,
        DTL/DTV variants) stays in the scaffold's `globalReadDo`, reached through the leaf.

        FUSED movement: emitted through the OWNING member's tP and nothing else, because the
        scaffold is already written that way -- `globalReadDo`'s `tc == "B"` arm is guarded by
        `if numWaves == 1`, so at NumWaves>1 the B call produces an empty module and the A call
        produces the one cooperative `tensor_load_to_lds` that serves both operands.  Issuing a
        second call for B would therefore be a no-op today and a double load the moment that guard
        
```

## gir_to_rocisa.py:457 _emit_region_inc

```
Realize a descriptor walk of `steps` storage regions, via `tdmRegionIncrementGir` (#236).

        `steps` is SIGNED and may exceed one in magnitude; the primitive moves by exactly one
        region, so this issues `|steps|` of them.  Unrolling here rather than teaching the
        primitive a count keeps the backend's job "one region, one direction" -- the arithmetic is
        an add-with-carry over a 64-bit address pair, and a scaled version would need its own
        multiply and its own carry reasoning for a case (>2 regions) that no shipping config
        reaches.  GIR still owns HOW MANY; only the loop moved.

        A fused movement steps ONE descriptor, named by the owning member, for the same reason
        `_emit_copy` issues one load for it: at NumWaves>1 the cooperative `tensor_load_to_lds`
        serves both operands off A's groups.  (`tdmRegionIncrementGir` refuses multi-wave outright
        today, so this is the shape the code will need rather than one it exercises.)
```

## gir_to_rocisa.py:442 _wrap_lead

```
The shift that puts the wrap compare in the frame THIS advance runs in.

        `tdmIncrementGir` wraps when `LoopCounterL + lead == StaggerUIter`, and
        `StaggerUIter = S + pf` with `S` the stagger start chunk.  The wrap must fire on the advance
        that takes the address from chunk `T-1` to `T`, i.e. advance #n where `S + n = T`.  Under
        GIR's copy-then-advance convention, advance #n in the prologue targets chunk `n` and runs at
        `LoopCounterL = T`; the steady advance in trip `v` is #(M+v+1) and runs at `T - v`.
        Solving `ctr + lead == S + pf` against `S = T - n` in each case:

            prologue, to_chunk = n <= M   ->  lead = pf - n
            steady                        ->  lead = pf - M - 1        (constant; the `v` cancels)
        
```

## gir_to_rocisa.py:425 comment run

```
        # `tdmIncrementGir`, not `tdmIncrementAB`: the GIR path owns the wrap decision, and the
        # shared function reconstructs it from PGR + phase.  Same emitted instructions today --
        # `wrapLead` carries exactly the value `prefetchIndex` did -- but the seam now exists, so
        # the derived predicate can replace the calibrated lead without touching any non-GIR caller.

```

## gir_to_rocisa.py:398 _emit_gr_inc

```
Realize ONE global-read increment for a Phi movement, via `writer.tdmIncrementGir`
        (s_add tdm+=inc, with the StaggerU WrapU cselect).  GIR OWNS the placement (a gr_increment
        Mark from the dataflow); the magnitude is L3's.

        `chunks` is how many reduction chunks GIR's dataflow says the address must advance here.
        `tdmIncrementGir` emits ONE DepthU stride, so anything other than 1 has no faithful
        realization -- emitting the single stride anyway would leave the address short and silently
        re-fetch a chunk.  Reject rather than under-advance.

        FUSED movement (multi-wave TDM): one aliased descriptor, so one advance, but two details
        differ and both are `tdmIncrementGir`'s to apply -- the stride comes from the per-wave
        
```

## gir_to_rocisa.py:368 _emit_swap

```
Realize ONE buffer-pointer swap via the scaffold's primitive.

        read hop -> `localReadSwapOffsets` (v_xor LocalReadAddr), per OP-CLASS: A and B read from
        their own LDS regions through their own address registers, so `who` is an operand name and
        multi-wave changes nothing here.

        copy hop -> `tdmSwapLdsOffset` (s_xor the tdm descriptor's LDS address), per Phi MOVEMENT, so
        `who` is a member tuple.  A fused movement has ONE descriptor, so it takes ONE swap --
        emitted through the owning member's tP, whose `tdm{tc}Group0` IS the shared register set.
        That matches what the scaffold does off this path: its multi-wave arm also swaps A only
        (`KernelWriter.py`, `if kernel["NumWaves"] == 1` guards the B swap).
```

## gir_to_rocisa.py:341 comment run

```
        # The groups the scaffold aliases, READ FROM THE ONE TABLE (`LoopModel/fuse.py`) that theta
        # also builds `fuse_groups` from.  This used to spell `[("A","B")] (+ the MX pair)` out
        # here, which was a THIRD derivation of the grouping beside theta's table and
        # `defineTdmSgprs`' RegSet chains -- and the one that silently decided which groupings
        # existed, since a group absent from this list is refused no matter what theta modelled.

```

## gir_to_rocisa.py:328 _fused_tp

```
`(tP_owner, tP_peer)` for a Phi-fused movement, after checking the scaffold can realize it.

        The scaffold's fused form is ONE aliased descriptor: `tdm{tcA}Group0` serves both operands,
        each wave's copy of those SGPRs addressing its own, and wave parity selecting which
        (`KernelWriterAssembly.isTdmWaveSeparated`).

        THERE ARE TWO SUCH PAIRS ON A MICROSCALED KERNEL, not one.  `KernelWriter` calls
        `initTDMDescriptorWaveSeparated` (and `tdmGlobalOffsetWaveSeparated`) for A/B AND AGAIN for
        `tPA["MX"]`/`tPB["MX"]` when both `MXBlock`s are set -- the scales get their own aliased
        descriptor, on the same wave-parity rule.  This used to name `("A", "B")` as the only
        realizable fuse and refuse everything else, which was true when it was written and became
        
```

## gir_to_rocisa.py:309 _fence_tokens

```
The token ids this fence stands between -- carried on the Mark by `FenceRegions` (#235).

        A fence gets each of its tokens as BOTH a def and a use, which is what makes it a
        scheduling point between the producers and consumers of that storage.  So the set has to be
        exactly the storage it covers, and `FenceRegions` is the layer that knows: it built the
        fence from the hazards whose ends those accesses are.

        This used to be `{memTokenLdsBuffer0, memTokenLdsBuffer1}` -- the scaffold's two names,
        independent of what the fence actually covered.  Under TDMSplit that set named NEITHER of
        half 1's tokens, so the fence was emitted, looked right, and ordered nothing for that half
        (#217).  Deriving it from the covered hazards cannot have that failure mode.
```

## gir_to_rocisa.py:265 _emit_fence

```
Realize a proc-scoped selector -- a real, memory-ordering barrier.

 GIR decided WHERE (`FenceRegions`' cover) and WHAT IT ORDERS (`buffers`); L3 supplies the
 instruction. `_syncThreads` is the scaffold's own emitter and issues waitcnt + barrier, so
 it has the STRENGTH demands -- the read's completion ordered before the refill's write.
 The model is explicit that a bare execution-only barrier does not qualify.

 Unlike the `await` boundary this replaces, there is no `_needs_own_barrier` gate: GIR owns
 LDS synchronization at every SIA now, because `postMainLoopBarrierCheckAndReset` is skipped
 for `UseLoopModel` kernels (it would otherwise strip these and re-derive from tokens). A
 single-wave kernel still emits nothing, but for the RIGHT reason -- the analysis finds no
        
```

## gir_to_rocisa.py:250 _tag

```
Stamp EVERY GIR-emitted act with a self-identifying comment in the emitted assembly.

        Two jobs, and the second is why this is not decoration:

        1. READABILITY -- the act's GEMM meaning at the point of emission (which tile, which
           K-substep, which register slot, which buffer generation), so the `.s` reads as the
           schedule rather than as an instruction soup.
        2. ATTRIBUTION -- every instruction GIR emits is now traceable to the GIR act that asked
           for it, so anything WITHOUT a `GIR[...]` tag came from the scaffold.  That distinction
           was previously only recoverable by matching emission ORDER against a GIR dump by hand,
           which is how a duplicated copy+increment at PGR=1 stayed invisible: the scaffold and
        
```

## gir_to_rocisa.py:231 _append_tag

```
APPEND `tag` to every instruction comment in `code`, preserving what is already there.

        Appending, not replacing: the scaffold primitives put real information in their comments
        ("sync LDS0", "select WrapU or normal inc", "TDM addr += inc (with wrap, 64-bit)") and the
        backend appends its own ("<This is 20-cycle>").  Overwriting would trade one useful fact
        for another; the GIR act is ADDITIONAL context, so it goes on the end.

        Instruction-attached rather than a floating comment line, because that is the form that
        SURVIVES REORDERING: under `ScheduleIterAlg=4` StinkyTofu moves instructions freely, so a
        standalone comment no longer marks what it was written above, while a comment carried by
        the instruction travels with it.  Attribution has to hold in exactly the configuration
        that is hardest to read by hand.
```

## gir_to_rocisa.py:126 comment run

```
            # A `RuntimeError` FAILS THIS KERNEL, NOT THE RUN.  This was a
            # `ValueError`, which escapes `getSourceFileString`'s per-kernel handler and aborts the
            # whole build -- so a config broken on 2 of 9 loop orders reported nothing about the
            # other 7 and had to be bisected one run at a time (MEASURED 2026-08-24).

```

## gir_to_rocisa.py:114 comment run

```
        # WHICH read acts issue, and into which register, is theta's merge -- evaluated once per
        # block from the `ir.QuantumMap` each hop carries.  `quantum_of` returns None for every
        # operand theta gave no merge (all of A/B, and any unmerged scale), and the plan is then the
        # identity: one instruction per act, exactly the historical behaviour.

```

## gir_to_rocisa.py:98 emit_block

```
Realize the finalized GIR block `phase` into a rocisa Module.

        A swap Mark is realized by calling the scaffold's swap PRIMITIVE keyed by the Mark's
        subject -- `writer.tdmSwapLdsOffset(kernel, tP)` for a copy-hop swap (s_xor tdm descriptor),
        `writer.localReadSwapOffsets(kernel, internalPointerSwap, tP)` for a read-hop swap (v_xor
        LocalReadAddr).  This is why one Mark -> one swap, with no doubling: the fork's pre-bundled
        A+B Module is NOT used; GIR's Marks map 1:1 to the primitive.  `tpByOperand` =
        {operand: tP} (incl. MX/meta later).

        The SUBJECT is per hop, and this is the model rather than an inconsistency.  A read swap
        and a read name an OP-CLASS (`operand`).  A copy, a copy-hop swap and a global-read
        
```

## gir_to_rocisa.py:75 comment run

```
        # L3 OWNS the leaf emitters (they ARE the layer-2->3 realization).  The register ring width
        # W is a GIR fact passed as a real field -- no reaching into a retired walker's internals.
        # RegBand is the per-(operand, group) authority; `_register_depth` collapses it to the
        # leaf's scalar fallback only when every group agrees, and raises otherwise.

```

## gir_to_rocisa.py:31 _register_depth

```
`{operand: W}` -- the register ring width the leaves use for their `m = u % W` FALLBACK, a
    GIR fact (`Ref.reg_ring`, validated by RegBand, B3).

    The fallback only runs when GIR did not supply a per-act slot, which for a GIR-planned wmma or
    read it always does; W is therefore a legacy path, not the authority.

    PER OPERAND, because two different disagreements were being conflated.  RegBand keys widths by
    `(operand, group)`, and this used to collapse them to ONE set and raise whenever it held more
    than one value -- which is right for groups WITHIN an operand (mxfp8 A=(lo,hi): lo rotates a
    2-ring, hi is in-place; no scalar addresses both, and that is #118) and wrong ACROSS operands,
    where differing widths are ordinary.  MEASURED 2026-08-18: an MX kernel at PLR0 has A rotating
    
```

## gir_to_rocisa.py:3 <module>

```

gir_to_rocisa -- the layer 2 -> 3 lowering, the THIN rocisa adapter.

L3 consumes the finalized GIR (after the pipeline applied swap/gr_increment Marks and stamped
tokens) as a flat EMIT PLAN (`gir.plan_block`, pure) and realizes each action into rocisa by
calling the per-tile leaf emitters (`Lowering/leaves.py`, `LeafEmitters`). The order + all
indices are GIR's; this file only maps action->instruction.

Concern split : GIR decided WHAT/WHERE/WHY (a swap is a Gen-transition fact at a point);
L3 decides HOW (a swap Mark -> the scaffold primitive tdmSwapLdsOffset / localReadSwapOffsets; a
gr_increment Mark -> tdmIncrementGir). GIR is the single source of truth for swap + gr_inc

```

## lds_geometry.py:166 region_bytes

```
Byte displacement from the operand's LDS base to the start of TDMSplit `region`.

    THIS IS NON-ZERO EXACTLY ON THE PACKED DIAGONAL -- when the split cuts the LDS image's INNER
    axis (`region_split_is_packed`) -- and the asymmetry is a property of the image, not a
    heuristic.  Which axis is inner is layout-decided, so each layout packs on a DIFFERENT split:

      * unroll-major, LDS = [free][unroll].  An MT split cuts the OUTER axis, so region 1 is the
        second half of the same array.  Its two `tensor_load`s reproduce exactly the contiguous
        image one unsplit load produces, and a read must address it exactly as it does unsplit --
        the global `tile_row` already lands in region 1's half.  Adding a region term there
        double-counts; measured 2026-08-16, it regressed 82 passing kernels (110 -> 192 failures).
    
```

## lds_geometry.py:132 tile_row

```
The LDS ROW index of this operand's wave-tile `t`, in an unroll-major (DU-major) layout.

    THE DISTRIBUTION IS BY VECTOR GROUP, and the unroll-major path used to model only half of it.
    A wave takes `VectorWidth` ADJACENT tiles (adjacent rows), then the next group of tiles starts
    a whole `MIWaveGroupShape[tile01]` rows later -- that jump is what hands the intervening tiles
    to the other waves.  So

        row(t) = (t // VW) * MIWaveGroupShape[tile01]  +  (t % VW)

    and the old `tileStrideElems * t` is exactly the second term.  It agreed with the hardware only
    while every tile sat in ONE vector group (`MIWaveTile[t] <= VW`), which is what every test
    
```

## lds_geometry.py:115 addr_coord_on_split_axis

```
Which coordinate the ADDRESS uses along the axis a TDMSplit cut: the WITHIN-REGION one when
    the regions are separately packed, the FLAT one when they are a contiguous continuation.

    ONE RULE, and the two halves are not interchangeable:

      * PACKED   region r is its own padded block, so the address is
                 `r * splitBoundary + within-region coordinate` -- `region_bytes` supplies the first
                 term and this returns the second.
      * CONTIGUOUS  the split's loads reproduce exactly the image one unsplit load produces, so
                 `region_bytes` is 0 and there is NO displacement to carry the region -- the
                 coordinate must therefore be the FLAT one, or region r>0 addresses region 0.
    
```

## lds_geometry.py:96 fold_inner_offset

```
`(region, innerInRow)` -- split an INNER-axis element offset into the region it lands in and
    its position inside that region.  The counterpart of `region_row_elems`: that shortens the row,
    this says which row-block a coordinate past the end belongs to.

    A packed split makes the address PIECEWISE.  The unsplit image is one linear map
    `f*extent + u`; the split image is `region*splitBoundary + (f*(extent/n) + u mod (extent/n))`,
    and there is no linear expression for the second form because the region jump is a padded byte
    boundary, not an element stride.  Any emitter that builds the inner coordinate by accumulating
    (`rIdx * step + localReadOffset`) has to do this division somewhere; doing it here keeps the
    scaffold read path (`Components/LocalRead.py`) and the ULM read path -- which gets the same
    answer from GIR's `region_modes` instead of by dividing -- from disagreeing about the boundary.
    
```

## lds_geometry.py:78 region_row_elems

```
Row length, in elements, along the LDS image's INNER axis -- of ONE REGION when the split
    packs regions separately, of the whole tile otherwise.

    `extent` is that inner axis's full length, and WHICH axis that is depends on the layout, so
    the caller supplies it: `MacroTile` on tile-major (`[unroll][free]`), `DepthU` on unroll-major
    (`[free][unroll]`).  Both callers then get the same rule -- a packed region divides the row it
    lives in -- instead of the tile-major one being special-cased.

    IMPORTED BY `Components/LraTileAssignment.py`, which is the one place a layer below this one
    reaches up into `Lowering`.  That direction is deliberate and narrow: this module imports
    NOTHING, and the alternative homes are worse.  `Tensile/Common/` looks right by layering but
    
```

## lds_geometry.py:62 region_split_is_packed

```
Does TDMSplit make each region a SEPARATELY PACKED LDS block?

    Delegates to `tdm_split.split_packs_lds`, which states the rule once for the descriptor side
    and this side together.  The answer is a DIAGONAL over (axis, layout) -- a partition packs
    blocks exactly when it cuts the LDS image's INNER axis -- not a property of either alone:

        unroll-major  [free][unroll] : MT contiguous, DU PACKED
        tlu           [unroll][free] : MT PACKED,     DU contiguous

    `axis` defaults to `AXIS_MT` for the callers that predate `TDMSplitA/B`; at the time this
    predicate was written the MT half was the only reachable one, and `(not unrollMajor)` was the
    whole rule because it was `packed(MT, layout)` with the axis silently fixed.
```

## lds_geometry.py:30 read_fragments

```
The ds_readS ONE (tile, reduction-substep) local-read act decomposes into, as
    `((unrollElems, regOff), ...)` -- both relative to the act's own base, the first in ELEMENTS
    along the unroll axis, the second in registers.

    A lane's `inputPerThUnroll` elements for one substep are neither one contiguous run nor one
    instruction, and TWO nestings produce them:

      INNER `v` -- one `lrvw` chunk takes `lrvw / vwTrLoad` instructions, where `vwTrLoad` is how
                  many elements ONE ds_read moves: `blockWidth` registers of `bpeDS`-byte elements.
                  bf16 reads 8 elements (`b128`/`b128_tr_b16`, blockWidth 4) against `lrvw` 8, so
                  this loop is degenerate; fp8's transpose read is `b64_tr_b8` (blockWidth 2 = 8
    
```

## lds_geometry.py:3 <module>

```

LDS tile geometry -- the PURE arithmetic behind an operand's per-tile LDS address.

Split out of `leaves.py` for the same reason `emit_plan.py` is split out of `gir_to_rocisa.py`:
`leaves` imports rocisa at module scope, so anything living there cannot be unit-tested in the
pure-Python environment, and the one prior test of this arithmetic had to match SOURCE TEXT
instead of calling it.  That test then broke the moment the (wrong) expression was corrected,
which is the failure mode of testing spelling rather than behaviour.

Takes a duck-typed context (`LdsReadTileContext`, or any object with the same fields), so this
module imports nothing.

```

## lds_region_check.py:227 check

```
Run the check over one emitted kernel.

    `region_bytes` is `tdmSplitLdsBoundary` for the operand -- the SAME displacement the copy side
    advances by.  Returns (accesses, violations, notes).

    THE SEED IS THE HARDWARE REGISTER `v0`, not the symbolic `vgprSerial`.  The kernel's own first
    instruction is `v_mov_b32 v[vgprSerial], v0`, so seeding `vgprSerial` would OVERWRITE the value
    the slice is about to derive and, worse, would hide the fact that `v0` was never initialized.
    Seeding `v0` with the workitem id lets the slice compute `vgprSerial` itself, which is both the
    honest starting point and one less thing this checker asserts about the kernel.
    
```

## lds_region_check.py:98 comment run

```
    # TRAP TEMPS hold hardware-provided launch state (`s_mov_b32 s[sgprWorkGroup0], ttmp9`).  Keyed
    # by NAME in the scalar file so they must be seeded deliberately: they are inputs, and the whole
    # point of the refusal discipline is that an input never silently reads as zero.  Which
    # workgroup is seeded does not affect the wave-relative region structure this checker measures.

```

## lds_region_check.py:3 <module>

```

LdsRegionCheck (#245) -- does one `ds_load` touch more than ONE storage region of a split tile?

THE HAZARD.  A `TDMSplit` tile lives in LDS as `nsplit` storage-disjoint regions, and GIR names the
region each read touches so that `MemTokenAssignment` can give it a token and `FenceRegions` can
order it against the copy that filled THAT region.  All of that assumes one read instruction sits
inside one region.  The assumption is not free: the region a lane reads is decided by the lane's
position along the split axis, and with an unlucky `MIWaveGroup` x `VectorWidth` x `MacroTile`
combination the lanes of a single `ds_load` -- or the different WAVES executing it -- straddle the
boundary.  When they do, GIR has named one region while the hardware touched two, so the access is
ordered against one producer and left un-ordered against the other.  That is a data race, it is

```

## leaves.py:644 emitCopyTile

```
Emit the global->shared copy (tensor_load / buffer_load) for ONE operand, into staging
        buffer `bufIdx`.

        The third leaf, alongside `emitLdsReadTile` and `emitWmmaTile`, and the one that was
        missing: copies used to be handed to L3 as a PRE-BUILT Module per operand
        (`copyByTc = {A: codes.globalReadA, ...}`) and emitted once per block.  That is why the
        prologue and drain could not be GIR-owned -- their peels issue SEVERAL copies of one operand
        into different buffers, and one pre-built module cannot be two of them -- and why
        `_emit_copy` needed a dedup guard to stop the single module being emitted twice.  Realizing
        a copy per GIR Move, exactly as a read or a wmma is realized, removes both.
        
```

## leaves.py:631 comment run

```
            # THE REGISTER BASE IS THE GROUP'S, not the tile's, when a read covers several tiles:
            # `wtRegStride` is the span ONE instruction writes, so consecutive groups are that far
            # apart and the tiles inside a group share it.  `tilePerRead == 1` leaves this exactly
            # `wtRegStride * regTileIdx`, which is every A/B read.

```

## leaves.py:588 comment run

```
                # MX TileSpan: LRA packs `2*VW` tiles so one ds_load covers a group's LOWER
                # half-wave and the partner rides the same register, selected at the WMMA by
                # `matrix_{a,b}_scale:1`.  Only the lower half is loaded, and the loaded groups are
                # packed CONTIGUOUSLY -- group g owns `vw` registers, not `2*vw` -- which is the
                # compaction `mxsTileSpanScaleSel`'s `group*vectorWidth + regInHalf` expects.

```

## leaves.py:557 emitLdsReadTile

```
Emit the ds_read(s) for ONE operand M/N-tile `tileIdx` at K-substep `kIdx`.

        `memToken` -- the LDS memory-token id(s) this read CONSUMES, from GIR (#235).  Passed down
        verbatim; `None` leaves the scaffold's own derivation in place.

        Why GIR supplies it rather than the component deriving it: the component's split-half
        classifier keys on the read's BYTE OFFSET
        (`Component.LocalRead._getLdsReadMemToken`), and it is gated on `ldsByteOffset is not
        None` -- which this leaf never passed, so under ULM every read silently took the unsplit
        token while a split tile's half-1 `tensor_load` carried a different one and was never
        waited on (#217).  Which half a read consumes is a dataflow fact GIR already holds in
        `Tile.coord`; inferring it from an address is what made that hole possible.
```

## leaves.py:530 _checkRegBuffer

```
INVARIANT GUARD (#122) -- not a diagnostic anything is expected to trip.

        Read honestly: on the UseLoopModel path this CANNOT fire.  GIR names the slot
        `rate_index mod W_g`, so `bufferIdx < W_g <= max_g W_g <= numVgprBuffer` once
        `KernelWriter.loopModelRegBuffers` has grown the allocation to theta's rotation width.  The
        assertion is that relationship restated at the point of use.

        It is kept because the relationship is NEW and cheap to break: for a long time the two
        sides came from different places (theta's width vs the scaffold's `PLR + 1` /
        `ClusterLocalRead ? LoopIters`), and they silently agreed only while `n_s == LoopIters`.
        When they diverged the symptom was an UNDEFINED SYMBOL -- an assembler error on generated
        
```

## leaves.py:509 comment run

```
        # The SAME quantity the copy side walks by (`tdmRegionIncrementGir` adds
        # `tdm{tc}LdsSplitIncs`, which `_setTdmSplitIncs` initializes to exactly this).  Taken from
        # the writer rather than recomputed here so the read and the write cannot disagree about
        # where region r lives -- two derivations of one displacement is the #236 defect again.

```

## leaves.py:490 comment run

```
        # WHICH ds_readS ONE ACT DECOMPOSES INTO -- pure arithmetic, so it lives in `lds_geometry`
        # beside the other address geometry and is unit-tested there without rocisa.  It RAISES on
        # a shape it cannot tile, which is the honest answer: this leaf would otherwise emit a
        # register span the wmma then reads unwritten.

```

## leaves.py:477 comment run

```
            # A NON-SQUARE MATRIX INSTRUCTION puts several instruction tiles inside one wave tile,
            # and the scaffold walks them in an extra `ti` loop whose per-tile displacement
            # (`matrixInstTO * ti`) the fragment table below has no term for.  Nothing in the ULM
            # matrix reaches it (every MatrixInstruction there is 16x16), so it is rejected rather
            # than emitted from an untested extension.

```

## leaves.py:436 comment run

```
        # ONE BYTE OR TWO.  `bpeDS` below 1 (fp6 = 0.75, fp4 = 0.5) is a DIFFERENT read shape, not
        # a smaller one: the scaffold gives each its own branch with a padded register stride
        # (fp6 rounds `wtRegStride` up to a multiple of 16) and a per-load vgpr count that is not
        # `blockWidth`.  The fragment table below is checked against `wtRegStride` and would refuse
        # them anyway; reject here so the message names the dtype instead of the arithmetic.

```

## leaves.py:393 comment run

```
        # TILESPAN, ASKED OF THE SCAFFOLD'S OWN PREDICATE rather than re-derived.  The load side
        # (how many ds_reads and into which registers) and the consume side (`mxsTileSpanScaleSel`,
        # which `emitWmmaTile` calls) MUST agree about whether the half-wave layout is in force;
        # two spellings of that condition is how they would drift.

```

## leaves.py:347 buildMxScaleReadContext

```
Loop-order-invariant LDS-read constants for a MICROSCALING SCALE tensor (`MXSA`/`MXSB`).

        A SEPARATE BUILDER, not a branch in `buildLdsReadContext`, because the scale tensor's LDS
        image is genuinely a different geometry rather than a narrower version of the same one --
        the scaffold gives it its own emitter (`Components/LocalRead.py:localReadMX`), and A/B's
        `UnrollStride`/`tileStride` swap roles there.  Every validation gate in the A/B builder
        (LDS-transpose vs unroll-major, the `bpeDS` pair, the fragment table) is about that other
        image and would be answering the wrong question here.

        THE WHOLE GEOMETRY IS TWO NUMBERS, both in units of `mxUnit = MatrixInstK / MXBlock` --
        how many scale values one matrix instruction consumes along K:
        
```

## leaves.py:241 emitWmmaTile

```
Emit the SINGLE WMMA instruction for tile (idx0, idx1) at kiter substep `u`.

        The per-operand register buffer generation (`bufA`/`bufB`, and `bufMXA`/`bufMXB` for the
        microscaling scales) is the CONSUME-side rotation slot GIR/LoopModel assigned to the wmma
        source (`_wmma_src_residence`) -- the SAME authority the producing ds_read used, so def and
        use name the same vgpr under any loop order.  When not supplied (older call sites), fall
        back to `m = u % W`.

        A scale operand rides the SAME mechanism as A and B, which is the point: theta gives MXSA and
        MXSB their own paths and register rings, so their slots arrive here exactly as A's
        and B's do and nothing about the scales is special-cased in the schedule.
```

## leaves.py:210 comment run

```
        # MICROSCALING CONSTANTS, derived exactly as `mfmaIter` derives them.  `block` is the MAX
        # of the two block sizes because the instruction takes ONE block modifier for both scale
        # operands; asymmetric MXBlockA/MXBlockB is therefore not expressible and is rejected here
        # rather than silently scaling one side by the other's block.

```

## leaves.py:178 comment run

```
        # SUPPORTED INPUT WIDTHS.  bf16 (2 bytes) and 8-bit float; both are dense WMMA whose whole
        # dtype dependence is `numRegisters()` (below) plus the opcode (`dataTypeToMfmaInstTypePair`).
        # The narrower formats (fp6/fp4, `bpeDS` 0.75/0.5) are NOT merely untested: their LDS read
        # geometry has its own scaffold branch with a different register packing, so they are
        # rejected here rather than emitted from a formula that was never checked against one.

```

## leaves.py:112 comment run

```
    # MX TileSpan: the VectorWidth when the half-wave scale layout is ON, else 0.  Under it LRA
    # packs `2*VW` tiles' scale blocks so ONE ds_load holds two half-waves' worth; only the LOWER
    # half is loaded and the WMMA reaches the partner with `matrix_{a,b}_scale:1`.  See
    # `emitLdsReadTile` for the two consequences (which acts emit, and the compacted registers).

```

## leaves.py:107 comment run

```
    # HOW MANY TILES ONE ds_read COVERS (1 for A/B and for a scale whose read is per-tile).  Above
    # 1, several of GIR's per-tile read acts are realized by ONE instruction: the scales of
    # `tilePerRead` adjacent tiles are CONTIGUOUS in LDS (`tileStride == mxUnit`), so one load
    # fetches them together.  See `emitLdsReadTile`'s group-leader rule.

```

## leaves.py:101 comment run

```
    # THE ds_readS ONE (tile, substep) ACT DECOMPOSES INTO: `(unrollElems, regOff)` per read, both
    # relative to the act's own base.  Derived once in `buildLdsReadContext` -- see the table there
    # for why a lane's elements are neither one run nor one instruction.  The default is the
    # degenerate ONE-read act, so a context built by hand (tests) still emits something coherent;
    # every context `buildLdsReadContext` returns overrides it.

```

## leaves.py:94 comment run

```
    # TDMSplit region count for this operand (1 = unsplit), and WHICH axis it cuts.  The two
    # always travel together: every consequence -- how many wave tiles share a region, how short
    # the in-region unroll row is, whether a region base is owed at all -- is a function of the
    # (axis, layout) pair, never of the factor alone.  `AXIS_MT` is the historical default because
    # the MT half was the only reachable one before `TDMSplitA/B`.

```

## leaves.py:89 comment run

```
    # BYTES BETWEEN CONSECUTIVE STORAGE REGIONS of this operand's tile (#237), 0 when unsplit.
    # A region split displaces LDS: the copy walks the TDM descriptor by exactly this much per
    # region (#236).  Whether a READ owes that displacement depends on the layout -- see
    # `lds_geometry.region_bytes`, which is the single place that rule is stated.

```

## leaves.py:80 comment run

```
    # VECTOR WIDTH for this operand, and the ROW JUMP between vector groups (#237).  A wave's tiles
    # are handed out in groups of `vectorWidth` ADJACENT rows, then the next group starts
    # `MIWaveGroupShape[tile01]` rows later -- see `tile_row` for why the unroll-major path needs
    # both terms and had only one.

```

## leaves.py:3 <module>

```

Leaf emitters -- the L3 per-tile realization of a GIR Move/Mma.

These are the order-AGNOSTIC per-coordinate emitters: the rocisa for ONE (idx0,idx1,u) WMMA tile
or ONE (tile,k) LDS read is identical regardless of loop order; only WHEN it is emitted (the GIR
plan) changes. They live here -- in the Lowering (L3) layer -- because they ARE the layer-2->3
realization; `gir_to_rocisa.GirToRocisa` owns a `LeafEmitters` and drives it from the GIR plan.

Ownership (clean-architecture): the register ring width `W` (used for m = u % W) is a GIR fact
carried on `LeafEmitters.reg_depth`, a real constructor field -- NOT reached into as a private
attribute from another object. Everything else goes through the active KernelWriterAssembly

```

## loopir_to_gir.py:882 build_gir

```
R-ONCE: build the theta model, lower to GIR, and run the pass pipeline EXACTLY ONCE, returning
    the finalized Program the phase forks emit from.

    KernelWriter calls this a single time per kernel and caches the result on `self.states`; the
    three stage forks (prologue/steady/drain) then only ask the emitter for their block, instead
    of each rebuilding the LoopIR (the coupling this whole rearchitecture removes).  Kept pure /
    rocisa-free so it is testable standalone; `params` (presets like StaggerU) ride on
    `Program.params` for the region analyses.

    `pipeline=None` -> the backend pipeline (reg -> tokens -> collect -> place -> apply -> verify).
```

## loopir_to_gir.py:856 gir_text

```
Render an ALREADY-BUILT GIR Program as text for the `OutputLoopIR` dump.

    Takes the Program, never a kernel: KernelWriter builds the GIR exactly ONCE per kernel
    (R-ONCE) and the three phase forks emit from that one object, so the dump must render THAT
    object.  Rebuilding here would both redo the work and risk showing a Program that differs
    from the one that produced the .s -- a debug artifact that lies is worse than none.

    Companion to `LoopModel.bridge.kernel_to_ir_text`: that shows what the decoder DECIDED (the
    rolled theta-nest), this shows what the lowering CARRIES into L3 -- every Move/Mma with its
    concrete generation and register residence.  Together they localize a defect to a layer.
    
```

## loopir_to_gir.py:736 comment run

```
            # drain0's predecessors: the last steady block always, PLUS the prologue when the
            # peel-validity false edge lands here -- which it does only in the FOLDED shape (no
            # short blocks).  With the short arm lowered, the false edge goes to `short0` instead
            # and drain0 has a single predecessor; `FoldShortPathPass` restores the two-pred join
            # when it proves the coverage legal.  Derived from `short_entry`, never assumed.

```

## loopir_to_gir.py:704 comment run

```
            # preds are DERIVED from which blocks exist, not hardcoded.  A one-hop path
            # (direct-to-register, "One-hop paths") stages nothing through shared, so
            # there is no copy to prefetch, no peel, and NO prologue block -- the steady loop is the
            # whole program and its only predecessor is itself.  Naming "prologue" unconditionally
            # made G-TERM reject that kernel (declared preds disagree with the CFG).

```

## loopir_to_gir.py:695 comment run

```
                # the loop states its TRIP COUNT, not a comparison (#231): the outer `Loop.trip`
                # range's bound IS the count.  `div` is the chunks a trip consumes, so a multi-block
                # chain (G2) says `(T - M) // n_copies` -- a thing the old `Bound` form could not
                # express.  scaffold LABEL set downstream by ScaffoldMapPass.

```

## loopir_to_gir.py:587 comment run

```
                         # per COMPLETION UNIT (a Phi group, or a lone operand): how many region
                         # instances its movement actually emits.  A read can only await at the
                         # granularity its PRODUCER offers, so the token key is gated on this --
                         # a coarse copy feeding split reads must not have the
                         # reads key on a region the copy never distinguished.

```

## loopir_to_gir.py:577 comment run

```
                         # pi : completion-class granularity of a region-split
                         # movement.  False = every region of an operand shares one completion
                         # class (a read awaits them all); True = one class per region, so a read
                         # of region j leaves the others in flight.
                         # `region_modes` tells TokensPass which coords name a region.

```

## loopir_to_gir.py:568 comment run

```
                         # OPERANDS WHOSE READ REGION IS THE AGENT'S, NOT THE COORDINATE'S (#245).
                         # `MemTokenAssignment` widens such a read to every region: the same
                         # instruction reaches a different one per wave, so a token naming only the
                         # coordinate's region leaves the other producers un-ordered.  See
                         # `theta.Hop.region_agent_relative`.

```

## loopir_to_gir.py:481 comment run

```
        # SAME RESOLVER AS THE READ'S DESTINATION, and that is the invariant: the wmma's source and
        # the ds_read that fills it must land on ONE vgpr.  Both placements come from
        # `_read_placement` (the read with its read-ahead shift, the wmma with dr=0), so they
        # already agree on the SLOT; picking the group by a different rule here would break the
        # agreement on the NAME.  See `_reg_residence`.

```

## loopir_to_gir.py:454 comment run

```
        # ONE instruction, ONE completion, ONE def -- but it FILLS `Pi factor` locations.
        # `covers` is that footprint; `refs.covered_coords` expands it where an analysis needs the
        # locations.  Emitting one Ref per covered coordinate instead would file N residencies
        # against a single counter increment (measured: 497 unit tests).

```

## loopir_to_gir.py:438 comment run

```
        # register residence: (group index, concrete slot, rotation width W) for THIS COORDINATE's
        # group. W is the slot Expr's modulus (a LoopIR fact GIR consumes, B3): mod>1 -> W
        # generations, a constant slot (cst(0), no mod) -> W=1 (PLR0).  Per group, because under a
        # register partition the groups have DIFFERENT widths -- that is the whole point of one.

```

## loopir_to_gir.py:417 comment run

```
                # STEADY (rel=0) and DRAIN (rel=t-M) share ONE timeline: the drain chunk T-M+t is
                # trip-parity-dependent (T runtime), so an absolute generation is meaningless -- a
                # RELATIVE gen (gen + gdelta evaluated at chunk=rel) lets gen_reaching see one
                # prologue->steady->drain residue chain and the pointer toggle continue naturally.

```

## loopir_to_gir.py:406 comment run

```
            # PER-MEMBER region coord : one fused instance, but each member names
            # its OWN axis at the shared pairing index.  Handing the representative's coord to
            # every member leaves the others' region axis unpinned, and `_regions_of` then widens
            # them to every region -- B came out carrying BOTH its buffers on BOTH region copies.

```

## loopir_to_gir.py:364 _quantum_coords

```
The coordinates ONE transfer fills: `coord` alone, or its product with the coverage axes.

 `coverage` is `((axis, n),...)` from `ir.Load` -- the axes this single instruction spans,
 in the two shapes `_absorbed` documents, told apart HERE by whether `coord` already names the
 axis. No marker field: the coord IS the discriminator, because the the axes it varies over subtraction that
 produced it is the same fact.

 * APPEND (full absorption, `q = N`) -- `coord` does not name the axis, because the axis left
 the hop's the axes it varies over entirely. Re-insert it with all `n` values: a `ds_load_b64` covering two
 scale tiles yields two coordinates, hence two definitions. Nothing to collide with, so the
 axis is appended rather than merged into position.
    
```

## loopir_to_gir.py:334 _absorbed

```
`((axis, extent),...)` the operand's read instruction SPANS -- the absorbed axes.

 THE OTHER HALF OF THE PRESENCE SUBTRACTION. `geometry.quantum_axes` took these axes out of the
 read hop's the axes it varies over, so the emitted `Load.coord` no longer names them and the read is ONE act
 per instruction (which is what the leaf emits). The CONSUMERS still name every value -- the
 wmma is indexed on the output's the axes it varies over, which keeps the axis -- so the def has to be readable
 as filling all of them. `refs.covered_coords` does that re-insertion from this tuple.

 TWO SHAPES, ONE TUPLE, TOLD APART BY THE COORD. A coverage folds an axis of extent `N`
 into `N/q` carrier groups, and the model's `q = N` case is only the degenerate one:
 * FULL (`q = N`) -- the axis is gone from the hop's the axes it varies over, so `Load.coord` does not name
    
```

## loopir_to_gir.py:288 _reg_residence

```
`(group index, concrete slot, rotation width W)` for `op`'s register fragment AT `coord`.

 WHICH GROUP HOLDS THIS COORDINATE -- the piece #118 was missing. A register partition
 (`VgprPartition > 1`) splits the fragment's in-region fan into groups that partition that
 mode's VALUES : at `MIWaveTile[4,4]` with `VgprPartitionA=2`, `grouping_mode` is
 `M_inner` (extent 4) and `lo` owns `M_inner in {0,1}`, `hi` owns `{2,3}`. `_read_placement`
 already emits one `(label, slot Expr)` per group and the groups carry DIFFERENT widths (lo
 `unroll` -> W=R, hi `inplace` -> W=1), so the two halves are genuinely different registers.

 Both call sites used to take `pl.slots[0]` unconditionally, i.e. always `lo`. That made the
 parameter a SILENT NO-OP AT EMIT while still moving the modelled footprint: every A source in
    
```

## loopir_to_gir.py:245 _member_region_coord

```
Re-key a FUSED instance's region coordinate onto THIS member's OWN region axis.

 move pairs a Phi group's members BY INDEX, not by a shared mode: region j of A moves with
 region j of B, and each names the axis it is split on -- A's `M_split`, B's `N_split`. The
 LoopIR instance carries ONE coord, necessarily naming one member's axis (its representative's),
 so handing that coord to every member leaves the others' region axis UNPINNED.

 What that costs, measured 2026-08-16 on a multi-wave TN split: the fused copy's tokens came out
 `(1,5,7)` and `(3,5,7)` -- A's two regions correctly distinct (1 vs 3) while B carried BOTH of
 its buffers (5 and 7) on BOTH region copies. `_regions_of` widens an unpinned axis to every
 region, which is the honest reading of "this access does not say which half" -- but here the
    
```

## loopir_to_gir.py:203 comment run

```
            # a peel-step chunk binding : bind the induction to its value. A CONCRETE
            # value (prologue `j-M`, short `t`) binds a real int.  A SYMBOLIC drain value `T-M+t`
            # cannot (T is runtime), so we carry its STEADY-RELATIVE offset `t-M` in `rel` -- the
            # gen lowering resolves the drain residue on the steady timeline from that.

```

## loopir_to_gir.py:170 _flatten

```
Mirror render_unrolled's traversal: unroll int-extent Loops; a symbolic-trip Loop (the
    steady reduction-chunk loop) is walked once at the current env; bind Branch residues (pinned
    or fanned); descend Peel/Cond bodies; yield each Inst with its concrete env.  Appends
    `(inst, dict(env), rel)` to `out` in program order.

    `rel` is the instance's reduction-chunk index RELATIVE TO THE STEADY LOOP, or None when the
    chunk is bound to a concrete absolute value.  It is the one fact the generation lowering needs
    beyond the env: 0 inside the steady body, `t-M` inside drain step `t` (whose chunk `T-M+t` is
    runtime-valued, so only the relative offset is knowable), None in the prologue (concrete
    negative chunks -> absolute generations).  Carried as a real parameter, NOT smuggled through
    `env` under a magic key: `env` maps mode names to coordinates, and this is neither.
```

## loopir_to_gir.py:141 _deps_of

```
Carry the LoopIR awaits (RAW/WAR, per /dep_defuse) onto a GIR verb as `deps`.
 Each dep is a plain hashable record (dep_name, counter, kind, scope) -- a semantic edge, no
 count (the numeric residual is L4's; R-SEMANTIC). This is the ONE place LoopIR awaits enter
 GIR, and therefore the one place a field can be dropped on the way in.

 `scope` (#276-S2) is the proc-scope of the obligation: "wave" for an edge program order
 plus a per-wave counter already covers, "block" for a cross-agent one that a per-wave counter
 provably cannot. It used to stop here -- the tuple carried three
 fields -- which left `Await.scope` inert, reaching only a diagnostic tag string. It is now
 checked against the fence analysis at the end of the pipeline
 (`pipeline._check_block_scope_covered`), so the model's claim about WHICH edges need a
 cross-agent fence has to agree with the analysis that places them.
```

## loopir_to_gir.py:124 _expr_raw

```
Evaluate an Expr WITHOUT applying its own modulus -- the raw generation offset. Used to
 derive a steady ref's `gdelta` (the constant read-ahead / prefetch offset on top of the
 loop-carried `iter` base), so gen_reaching's `(entry + gdelta) % ring` reproduces the
 concrete generation.

 THE MODULUS IS THE ONLY DIFFERENCE, so everything else defers to `ir.Expr.eval` rather than
 re-implementing it. This function used to carry its own copy of the linear/carry arithmetic
 and had already drifted twice: it never learned `digits` (a `digits`-only Expr -- which is what
 `_shifted_coord` emits for a -folded mode -- evaluated to `0`), and it would have had to
 learn the per-term divisor separately. Evaluating with `mod` stripped keeps one arithmetic.
 
```

## loopir_to_gir.py:87 _trips_of

```
The LoopIR outer `Loop`'s RANGE, read as a trip COUNT (#231).

    LoopModel states the steady region as a range -- `[delta, T)`, i.e. `T - M` iterations -- and carries
    it as `Loop.trip = iter < T - M`.  That `<` is the range's upper bound, not a machine
    comparison: read the bound and the count IS the bound.

    This replaces `_post_test_bound`, which existed only because the lowering used to store the
    range AS the exit comparison of a post-tested block, where the same bound means one trip more
    (#228).  With the count held directly there is no shape to compensate for, nothing for a future
    loop rotation to undo, and the multi-copy divisor has somewhere to live.
```

## loopir_to_gir.py:51 _short_steps

```
The short-loop (`Cond.els`) arm, split into its per-step (guard, nodes) -- or a raise.

    THE TWO SHAPES (#183).  TensileLite's scaffold and the LoopIR describe the same kernel with
    DIFFERENT STRUCTURE:

        scaffold :  prologue -> Cond -> loop -> else -> drain     prologue+drain SHARED by both
                                                                   paths; `T < M` just skips the loop
        LoopIR   :  Cond { prologue -> loop -> drain } else { short }        two DISJOINT arms

    This function does NOT choose between them.  It hands back both arms' material so the lowering
    can emit the LoopIR shape faithfully, and `FoldShortPathPass` can then FOLD the short arm into
    
```

## loopir_to_gir.py:3 <module>

```

loopir_to_gir -- the layer 1->2 lowering.

Walks the LoopModel LoopIR (the rolled theta-nest `build_ir` emits) EXACTLY ONCE and unrolls the
three software-pipeline stages (prologue / steady / drain{n}) into one minimal GIR CFG. It is
the *tree-walk half* of the old `LoopModelLowering._walk` with NONE of the physical-state
reconstruction (no pointer seeding, no swap heuristics, no token stepping -- those become GIR
passes / L3).

What it attaches (all logical, R-SEMANTIC):
 - one loop-carried `Gen` per staged-shared operand (ring = its LDS depth); STEADY refs use it

```

## region_derivation_check.py:3 <module>

```

RegionDerivationCheck (#245) -- is a wave tile's TDMSplit region the same whether you derive it from
the TILE INDEX or from the tile's actual FREE-AXIS POSITION?

WHY THE TWO CAN DISAGREE.  `lds_geometry` already holds both derivations, for different callers:

    tiles_per_region(ctx)   MIWaveTile / nsplit      -- "region = t // tiles_per_region"
    tile_row(ctx, t)        (t // VW) * groupShape + (t % VW)

The first counts TILES and assumes they partition evenly.  The second is where the tile actually
SITS along the free axis, and it is not proportional to `t`: a wave takes `VectorWidth` adjacent

```

## uniform_emission.py:63 _branch_target

```
The label a conditional branch jumps to, or None if `item` is not a conditional branch.

    Matched on the CLASS NAME rather than by importing every branch type, because the set grows
    (`SCBranchExecZ`, `SCBranchSCC1`, ...) and a missing import would silently make this check skip a
    region -- the one failure mode a safety check must not have.

    A SOUNDNESS LIMIT, recorded where the matching happens rather than as a footnote: the long-form
    branches do NOT survive to this point.  `SCLongBranchScc0` and its siblings DISSOLVE in
    `flatitems()` -- what reaches us is the expansion, whose actual transfer is `s_setpc_b64`, and
    this function does not model an indirect jump.  So an arm opened by a long branch is invisible
    here.  None is emitted on the ULM path today; if one appears, this is the site that must learn
    about it, and until then the check is scoped to direct conditional branches.
```

## uniform_emission.py:3 <module>

```

The EMISSION half of the L733 uniform-placement check -- where the agent-discriminating arms
actually are.

L733 asks a *tree-structural* question of the IR: does a `scope=block` selector's node lie where no
agent-discriminating arm can make an agent skip it? `gir/analyses/uniform_placement.py` asks it of
GIR and gets a truthful but WEAK answer, because GIR's region tree has no node for the arms this
kernel really contains:

 s_bitcmp1_b32 s[sgprWaveIdx], 0 // check wave parity
 s_cbranch_scc1 label_SkipStaggerA // skip: odd waves handle B... <- executed by EVEN waves only

```


# Third pass

Docstrings of 7+ lines keep a 4-line lead; comment runs of 3+ keep two.

## __init__.py:3 <module>

```

Tensile.Lowering -- the UseLoopModel lowering stack.

 loopir_to_gir layer 1 (LoopModel LoopIR) -> layer 2 (GIR); build_gir = build-once (R-ONCE)
 gir/ the GIR graph, analyses, passes, verifier (pure Python; no rocisa)
 gir_to_rocisa layer 2 -> layer 3 (arrives R1; leaf emitters move here from Components)

```

## asm_lane_eval.py:151 comment run

```
    # ---- float domain -----------------------------------------------------------------------
    # A VGPR holds 32 BITS; an f32 is those bits and an f64 is a register PAIR.  Modelled as real
    # IEEE bit patterns rather than a side table of Python floats, so a value written as float and

```

## asm_lane_eval.py:118 comment run

```
    # ---- the opcode table -----------------------------------------------------------------
    # Pure functions of already-read lane values.  Kept as plain lambdas so the table IS the
    # semantics -- no dispatch method per opcode to drift from it.

```

## asm_lane_eval.py:82 read

```
One operand's value in `lane`.  An SGPR or literal is lane-invariant by construction.

        An operand the parser could not resolve is REFUSED here, not defaulted to zero: a zero
        address term is indistinguishable from a real one downstream, so it would turn a parse gap
        into a confident wrong answer.

        SO IS AN UNWRITTEN REGISTER, and that one is not hypothetical.  The first version of this
        
```

## asm_lane_eval.py:39 comment run

```
        # THE KERNARG IMAGE, `{byte offset: dword}` -- the values `s_load_b*` brings in.
        #
        # An address slice does not always bottom out at a register: TensileLite loads sizes and

```

## asm_lane_eval.py:3 <module>

```

LaneEval -- execute a straight-line slice of emitted gfx1250 assembly, once per LANE.

WHY THIS EXISTS, AND WHY IT IS NOT A FORMULA.  The question it answers is whether the 32 lanes of
one `ds_load` all land in the SAME storage region of a `TDMSplit` tile.  That is a property of the
per-lane LDS address, and the per-lane address lives in a VGPR (`vgprLocalReadAddr*`) built at init
from `vgprSerial` -- the instruction itself carries only a wave-uniform `offset:` immediate.  So the

```

## __init__.py:3 <module>

```

GIR -- the GEMM-dataflow IR (layer 2) of the UseLoopModel lowering.

Pure-Python (no rocisa import) so the graph + analyses + passes are unit-testable standalone.
Structure: nodes/region/verify are the graph; analyses/ and passes/ hold one class per module
(analysis.py / passes/base.py are the infra base classes; /).

```

## __init__.py:3 <module>

```

GIR analyses -- one class per module, each an `Analysis` subclass.

 cfg successors (helper), Dominators, BackEdges (+ BackEdge / BackEdgeSet)
 gen_reaching GenReaching -- concrete generation each access uses 
 swap_regions SwapRegions -- PendingMark(swap) per generation change 
 dep_defuse DepDefuseAnalysis -- RAW/WAR edges carried from LoopIR awaits

```

## cfg.py:77 leaving

```
Back-edges whose SOURCE block is `label` (its terminator carries them) -- the transfers
 gen_reaching applies when leaving `label`. Single-tile: at most one; persistent :
 one per loop with no code change.

 Named `leaving`, not `headed_by`: it filters on `be.src`, so `headed_by(h)` meant the exact
 OPPOSITE of what it says for the one case where src != header -- a multi-block loop body,
 where the back edge leaves the LAST copy while `h` is the header. gr_increment asked
 `headed_by(header)`, got nothing, and silently emitted no increments at all.
```

## cfg.py:3 <module>

```

CFG analyses : dominators and back-edges.

Back-edges are found by DOMINANCE (a CFG edge src->target whose target dominates src), NOT a
single blk.loop flag -- the R0 obligation, so the persistent-loop relaxation is a no-op
later. `successors` is a plain helper (not an Analysis) used across analyses and the verifier.

```

## dep_defuse.py:3 <module>

```

DepDefuse -- the RAW + WAR dependency edges carried from the LoopIR awaits.

Each GIR verb (Move/Mma) carries `deps` = ((dep_name, counter, kind),...) threaded from the
LoopIR `Inst.awaits` by the lowering. This analysis indexes them into a queryable result so the
tokens pass (read/copy/tensor token idx) and L3 WAR-forwarding read one source of truth:

```

## fence_regions.py:194 comment run

```
                        # the selector must reach the agents involved, and it must order the
                        # read's COMPLETION before the refill's write -- a memory-ordering barrier,
                        # not an execution-only one.

```

## fence_regions.py:190 comment run

```
                        # THE TOKEN IDS THIS FENCE ORDERS (#235).  `buffers` names operands, which
                        # is the readable tag but NOT what the backend needs: a token is an LDS
                        # pseudo-register, and a fence gets each of its tokens as both a def and a

```

## fence_regions.py:172 comment run

```
            # Same-block edges, plus every cross-block edge whose CONSUMER is here -- the latter
            # is covered on the consumer side (see `_admissible_slots`), so it belongs to this
            # block's cover even though its producer is elsewhere.

```

## fence_regions.py:127 _tokens_across

```
Every buffer LIVE ACROSS this fence -- not merely the ends of the edges that placed it.

 A barrier is a POINT, and its ordering power comes entirely from its token set:
 `StinkyBuildImplicitDependencyPass` gives the barrier each token as both a def and a use, so a
 buffer the barrier does not name is not chained through it at all -- its accesses may migrate
 across it, and the waitcnt strength is derived from the same set. Naming only the endpoints of
 the edges the cover happened to select therefore leaves every other buffer live at that point
    
```

## fence_regions.py:92 _separates

```
Does deleting `x` disconnect `cb` from `pb`?  I.e. is every `pb -> cb` path forced through
    `x`?

    Walk forward from `pb`'s successors with `x` removed.  Reaching `cb` witnesses a path that
    avoids `x`, so `x` does not separate.  Starting at `pb`'s SUCCESSORS rather than at `pb` is what
    makes a fence in the producer's own block fail correctly: its outgoing edges are still there, so
    the consumer stays reachable.
```

## fence_regions.py:85 comment run

```
    # THE TEST IS SEPARATION, and reachability+dominance is only its single-block approximation.
    # The old form was `x in reachable(pb) and x in doms(block)`.  `reachable` is EXISTENTIAL -- it
    # answers "SOME path from `pb` gets to `x`" -- so a candidate reachable only along a BACK EDGE

```

## fence_regions.py:75 _already_fenced

```
Is a fence placed in an EARLIER block already on every path from producer to consumer?

    Without this the cover is per-block and blind across blocks: the last steady copy feeds reads
    in drain0 AND drain1, so drain1 asks for a fence of its own even though drain0's already sits
    between them on every path.  The fences are redundant, not wrong -- but redundant barriers are
    the exact cost this pass exists to remove.
    
```

## fence_regions.py:44 _cover

```
Greedy set cover: the fewest slots such that every hazard has a fence between its ends.

 Returns {slot: [hazards it discharges]}. Raises if some hazard admits no slot at all -- that
 would mean the emitted order itself makes the edge undischargeable (an illegal sigma), which is a
 defect in the schedule, not something to model over with a fence somewhere harmless.

 Keyed on the hazard's INDEX, not the hazard: a `Hazard` holds the `Move` it came from, and a
    
```

## fence_regions.py:37 comment run

```
    # Loop-carried: the pair straddles at least one back edge, so a slot after the producer OR at
    # / before the consumer sits between them.  (A same-trip pair whose producer follows its
    # consumer in program order has no intra-trip instance either, so it lands here too.)

```

## fence_regions.py:28 comment run

```
        # The producer ran in an earlier block, so within the CONSUMER's block every slot at or
        # before the consumer stands between them.  Covering on the consumer side rather than the
        # producer side is what makes this sound without dominance: a fence before the consumer is

```

## fence_regions.py:3 <module>

```

FenceRegions -- the MINIMAL set of proc-scoped fences covering the cross-agent LDS hazards.

`LdsHazards` says which ordered pairs can touch one buffer and at what trip distance; this decides
where to stand between them. Only the cross-agent edges need a fence: the `complete(read) ~>
issue(copy)` is discharged by program-order-on-issue, which relates events on ONE agent, so a
same-agent edge is already covered by program order plus the completion counter.

```

## gen_reaching.py:65 _entry_of

```
(per-Gen entry map, is_relative) for `blk` -- a real MERGE over its known predecessors.

    A header carries the phis, so its entry is declared.  Otherwise we merge: every known
    predecessor that AGREES on a Gen's value contributes it; a Gen the predecessors disagree on --
    or that no predecessor supplies -- gets `RELATIVE_BASE` and the block is flagged RELATIVE.

    The disagreement is real and not a modelling slip.  drain0's two predecessors are the last
    
```

## gen_reaching.py:48 _resolve_phi_entry

```
Per-Gen value on `blk`'s entry, at ANALYSIS granularity.

 A loop header phi contributes entry_val on the entry edge and the back-edge xfer value on the
 loop edge. These are NOT the same value -- with ring 2 and adv 1, trip 0 enters at generation 0
 and trip 1 at generation 1 -- so keying on the declared `entry_val` fixes a REPRESENTATIVE trip,
 not an invariant. That is sound for the consumers there are: the LDS memory token is a LOGICAL
 buffer name whose only requirement is the relative this-buf-vs-other-buf distinction WITHIN a
    
```

## gen_reaching.py:3 <module>

```

GenReaching -- the concrete generation each Move/Mma access uses.

SSA reaching-value over the CFG: resolve each Ref.gen phi to the block-entry value, add gdelta,
mod ring; each back-edge's xfer gives the entry value of the next trip. A peel ref carries a
concrete `abs_gen` (fact,) rather than a Gen, resolved directly.

```

## gl2_prefetch.py:54 _is_steady

```
The steady body only, and ONE mark per steady block -- which is one per reduction chunk,
        the same rate as `gr_increment`, because the scaffold's own placement is one per
        `_loopBody` call and `_loopBody` is called once per loop copy (`KernelWriter.py` ~6288).

        The prologue issues its GL2 through the scaffold's own peel, which is NOT suppressed under
        UseLoopModel and does not need to be: `KernelWriter.py`'s `counterL <= PGR` /
        `Skip_GL2` block sits outside the main loop, touches only the GL2 address VGPRs, and takes
        
```

## gl2_prefetch.py:46 comment run

```
                # after=BLOCK_ENTRY: no producer dependence (see 1. above).  before=BLOCK_EXIT is
                # the CONSERVATIVE late bound; the real bound is the loop-counter update, which is
                # a terminator fact rather than a body node, so the window already ends there.

```

## gl2_prefetch.py:19 Gl2PrefetchRegions

```
`[PendingMark]` -- one `gl2_prefetch` Mark per steady chunk, or none when GL2 is off.

    THE DEPTH IS A PRESET, NOT A theta FACT, so it arrives on `Program.params` (`nodes.Program`:
    "PRESETS analyses derive facts from ... not raw data") rather than on `prog.meta`, which holds
    what the theta decoder DERIVED.  theta does not model the GL2 movement at all and should not: it stages
    nothing, files no residency, and has no completion class -- there is no operand for it to be.
    Putting `PrefetchGL2` on theta would also be the one thing #163 forbids, TensileLite vocabulary
    
```

## gl2_prefetch.py:3 <module>

```

Gl2PrefetchRegions (#216) -- WHERE the GL2 cache prefetch goes, once per reduction chunk.

WHAT GL2 IS, AND WHY IT IS NOT A gr_increment.  `PrefetchGL2` (0/1/2, gated on the
`HasGlobalPrefetch` asm cap) warms L2 for a chunk `PrefetchGlobalRead + PrefetchGL2` ahead, while
the real global->shared copy runs at `PrefetchGlobalRead`.  It carries its OWN state end to end:

```

## gr_increment.py:157 comment run

```
            # Requirements are ABSOLUTE chunks, so forward edges need no re-framing.  The back edge
            # re-enters the same block one trip on, so the value leaving must be `adv` further along
            # to be comparable with the header's own (trip-0) requirement.

```

## gr_increment.py:138 _flow

```
Place the advances for ONE global-read address, via the shared ValuePlacementSolver.

        This analysis owns only what is specific to this register: which instructions consume the
        descriptor, the absolute chunk each fetches, and how a chunk re-frames across the back edge.
        WHERE the advances go is `value_placement`, the same module SwapRegions uses.

        THE CONVENTION THIS FIXES.  The previous walk emitted an advance only where a copy DEMANDED
        
```

## gr_increment.py:115 _reqs

```
[(inst, chunk)] for `unit`'s copies in `lab`, program order.

 A COPY IS NOT A CHUNK. A region-split movement delivers one reduction chunk with
 `nregions` `tensor_load`s, so the peel's positional index counts LOADS while
 `_chunk_of` needs CHUNKS -- off by exactly that factor. At PGR2 with a 2-way split A's four
 peel copies were read as chunks 0,1,2,3 instead of 0,0,1,1: the address ran to chunk 3, the
 steady frame demanded 2, and the solver closed the gap with a `-1` advance that
        
```

## gr_increment.py:59 GrIncrementRegions

```
Forward dataflow over the GIR CFG for one global-read address per Phi MOVEMENT.

 Per movement, not per operand: a fused group issues one cooperative load off one
 descriptor, so it owns one address that advances once. Unfused, a movement is a lone operand
 and the two readings coincide -- which is why the distinction stayed invisible until multi-wave.

 Domain Z (the reduction chunk the address points at) -- unbounded, because a global read address
    
```

## gr_increment.py:43 _chunk_of

```
The absolute reduction chunk a copy fetches, on the global chunk timeline.

    Loop-carried ref: `blk.chunk_base + (gdelta - blk.gen_rel)` -- frame-free, comparable across
    blocks.

    PEEL ref: derived POSITIONALLY, from `peel_seq` -- the index of this copy among that operand's
    copies in the peel.  The prologue fetches chunks 0..M-1 in issue order, so the k-th copy of an
    
```

## gr_increment.py:27 _copy_of

```
`(unit, shared-dst ref)` if `inst` is a global->shared copy Move, else None.

 `unit` is the Phi movement's MEMBER TUPLE, not one member's name: a fused group is one
 cooperative instruction over one descriptor, so it is one global-read address
 that advances once. Naming it `refs[0].tile.operand` would leave the other members' Refs
 unexamined and assert a single address rather than check for one -- see `..refs.copy_unit`.
    
```

## gr_increment.py:3 <module>

```

GrIncrementRegions -- the global-read-address advance, as a REAL forward
dataflow over the GIR CFG (the same shape as SwapRegions).

Each operand's global read address is a loop-carried POINTER holding the reduction chunk the next
copy will fetch. A `gr_increment` is the only thing that advances it, so -- exactly as for the LDS
pointers -- placement is a reaching-value problem: propagate the address to every program point and

```

## lds_hazards.py:323 comment run

```
        # One loop-carried, one pinned, in the same block.  Their bases are not comparable without
        # the block's entry value for that Gen, which `Reaching` deliberately does not expose as an
        # absolute (see gen_reaching._entry_of).  Refuse rather than guess.

```

## lds_hazards.py:257 _cross_block

```
Hazards whose two ends live in DIFFERENT blocks.

        These are real and they are not covered by any fence inside a single body: the last steady
        copy feeds the drain's reads, and the steady fence sits at the TOP of the body, so on the
        final trip nothing stands between that copy and the drain.  The backend's flat-stream phase
        machine catches it for free (it just keeps walking); an analysis that reasons per body has
        to model it explicitly or it will under-fence exactly where the scaffold does not.
        
```

## lds_hazards.py:166 comment run

```
    # A MERGED MOVEMENT NAMES EVERY BUFFER IT COVERS.  `Ref.covers` is the reference's own
    # declaration that ONE instruction fills several coordinates (the movement coverage, Sec 2.2):
    # the leader's `coord` pins the axis, but the instruction also touches the coordinates the

```

## lds_hazards.py:154 _regions_of

```
Per region axis of `operand`, the SET of region values this access MAY touch.

    A coord that PINS the axis names one region.  A coord that does not mention it touches EVERY
    region, so the unknown is modelled as the full set and stays a MAY-alias -- being conservative
    costs a fence, being optimistic LOSES one.

    THE PIN IS TRUSTED ON BOTH HOPS, and that is the point: a region split is a COORDINATE fact.
    
```

## lds_hazards.py:137 split_region_modes

```
`region_modes`, with the axes of movements that have ONE region REMOVED.

 THE TOKEN COUNT IS ARITHMETIC, AND IT WAS WRONG EVERYWHERE. For A and B split in two with a
 double-buffered LDS ring that is `2 regions x 2 buffers = 4` tokens each; the MX scales are
 NEVER split (`tdm_split.split_of` returns `(1, None)` for every MXS tensor), so they are
 `1 x 2 = 2` each -- twelve in total. Every kernel we emitted named SIXTEEN, because
 `_regions_of` keyed on `region_modes` being non-empty and the scales carry their parent's axis.
    
```

## lds_hazards.py:68 same_trip

```
Does an instance of this pair occur within ONE trip?

 `distance == 0` means the two accesses land on the same buffer in the same iteration, so the
 edge must be discharged INSIDE the body -- by program order if same-agent, by a fence
 placed between them if not. A non-zero distance is purely loop-carried, and ANY fence in
 the body separates trip `v` from trip `v+1`.
        
```

## lds_hazards.py:3 <module>

```

LdsHazards -- the LDS hazard edges, derived from the SSA.

The three families a reachable state can hold are RAW residency, rotation WAR/WAW
and a cross-role/cross-agent alias. This analysis finds them on the SHARED
(LDS) side by asking which accesses can touch the same physical buffer, and -- crucially -- at what
TRIP DISTANCE.

```

## loop_shape.py:168 comment run

```
            # NOT YET EXPRESSIBLE, and that is a limit of the predicate form, not a choice (#230).
            # A trip consuming `n` chunks must run `(T - M) / n` times, and `Bound` is `var + const`
            # with no division, so the bound cannot be WRITTEN -- never mind checked.  The G2

```

## loop_shape.py:107 _trips_from_node

```
The loop's trip count -- READ, not derived (#231).

    This used to reconstruct the count from the exit comparison plus the test position, which meant
    assuming the counter starts at 0 and steps by +1.  Those assumptions are not stated anywhere and
    the backend does not share them (TensileLite counts DOWN), and reconstructing across them is
    where #228's off-by-one lived.  `LoopBack` states the count, so there is nothing to reconstruct
    and the test position no longer changes the answer -- it is now purely an emission property.
```

## loop_shape.py:81 _exit_tester

```
The block whose terminator leaves the loop -- and it is NOT always the latch.

    This is the distinction the whole analysis turns on.  In a POST-tested loop the latch carries
    the exit test (`do { body } while`), so tester == latch and the tester does work.  In a
    PRE-tested loop the HEADER carries it and the latch back-edges unconditionally
    (`while (p) { body }`), so tester != latch and the tester does no work.  Looking only at the
    latch -- as the first version of this did -- finds no test at all in the pre-tested shape and
    silently reports no loop, which is worse than being wrong.
```

## loop_shape.py:3 <module>

```

LoopShape (#229) -- DERIVE a reduction loop's test position and trip count from the CFG, and check
that the steady region and the drain together cover the reduction exactly once.

WHY THIS EXISTS.  Whether a loop is PRE- or POST-tested is not a property anyone declares; it is
*where the conditional branch sits*.  Put the test on a block that also does work and the work
happens first (post-test, `do { body } while (p)`); put it on a block that only tests and the work

```

## mem_tokens.py:141 comment run

```
            # THE GENERATION IS A MAY-SET WHENEVER THE BLOCK'S FRAME IS RELATIVE.
            #
            # `Reaching.of()` is documented as meaningful only RELATIVE to other values in the same

```

## mem_tokens.py:129 comment run

```
                # No resolvable generation: we cannot say WHICH buffer, so we must not invent one.
                # Recorded rather than skipped -- a missing token breaks StinkyTofu's all-or-none
                # rule for the whole basic block, so silence here is expensive.

```

## mem_tokens.py:117 comment run

```
        # THE COUNT, NOT THE TUPLE.  A/B: 2 regions x 2 buffers = 4 each; the scales
        # are unsplit, so 1 x 2 = 2 each; twelve, not sixteen.  See
        # `lds_hazards.split_region_modes`.

```

## mem_tokens.py:90 _unfenced_loop_carried_writes

```
Write refs whose LOOP-CARRIED edge nothing separates -- the defs that must span the ring.

    `Hazard.same_trip` states the rule: "a non-zero distance is purely loop-carried, and ANY fence
    in the body separates trip `v` from trip `v+1`."  So a loop-carried pair is covered exactly when
    its block has a fence, and `needing_fence()` -- the ONE place the fence/no-fence decision is
    made -- says which blocks get one.  Asked of `LdsHazards` rather than re-derived, so this and
    `FenceRegions` cannot disagree about which edges are covered.
    
```

## mem_tokens.py:24 Buffer

```
One logical LDS buffer -- the thing a token names.

    Keyed on the OPERAND, not on the Phi completion unit.  A fused movement is one COMPLETION but it
    is one instruction writing TWO STORAGES: `tensor_load_to_lds` fills A's LDS region and B's, and
    those are different memory.  Naming the storage `A+B` merges them, so A's read and B's read land
    on one token and are ordered against each other for no reason -- and no access can ever name
    A's region alone.  The instruction instead DEFS BOTH ids (`_storage_ids` unions over its Refs),
    which is what `MemTokenData.tokens` being a vector is for.
```

## mem_tokens.py:3 <module>

```

MemTokenAssignment (#235) -- GIR names every LDS buffer, and the id IS the memory token.

WHAT A MEMORY TOKEN ACTUALLY IS. Not a label: an LDS PSEUDO-REGISTER. StinkyTofu's
`StinkyBuildImplicitDependencyPass` turns `MemTokenData{tokens}` into `RegType::LDS` operands --

 tensor_load / ds_write -> the token as a DEF (LDS producer)

```

## reg_band.py:3 <module>

```

RegBand -- VALIDATE the register rotation width W; never decide it.

W is decided by LoopModel (`translate._register_policy` intent -> `ring_slot` bakes the modulus),
carried into GIR on each register `Ref.reg_ring`. W is invariant under the schedule, and
fixed at S-search time, so GIR only cross-checks it -- deciding it here would be a second authority
that can disagree with the decoder (violating R-LEGAL).

```

## region_increment.py:149 walk_violations

```
[] or human-readable G-WALK violations.  Two properties, both replayed in PROGRAM ORDER:

      1. EVERY SPLIT COPY LOADS THE REGION THE DESCRIPTOR IS ON.  Walking the block, the running
         position must equal each copy's own region coordinate when that copy issues.
      2. THE WALK CLOSES.  It ends back at region 0, because `tdmIncrementGir` then applies the
         plain chunk stride and any leftover compounds every trip.
    
```

## region_increment.py:137 _axis_delta

```
`{mode: delta}` for the non-zero axes of one descriptor step (#251).

    A FLAT step is not a displacement once more than one region axis is live.  `steps` is the
    difference of two mixed-radix indices, so L3 could only turn it into bytes by multiplying one
    stride -- right while a single axis moves, wrong the moment two do: in a 2x2 grid the flat step
    `2 -> 1` is `M_split -1, K_split +1`, whose byte displacement is `-D_m + D_k`, not `-1 x D`.
    The per-axis form is the one the descriptor can actually execute, and it degenerates to the old
    scalar exactly when one axis is live -- which is every shape shipped so far.
```

## region_increment.py:77 comment run

```
        # READING refs[0] IS JUSTIFIED BY A PRECONDITION, SO CHECK THE PRECONDITION (#170).
        #
        # A Phi group pairs its members BY INDEX (move: "the pairing is by the index the

```

## region_increment.py:34 _flat

```
The region tuple as one index, so a walk is arithmetic on integers.

 Mixed-radix over the operand's region axes, innermost last -- the same shape the rest of the
 model uses for a multi-axis coordinate.

 Returns `None` when an axis is UNPINNED, which is a real modelling gap and not a coord to guess
 at: a Phi-fused group may tile its members on DIFFERENT region axes, and the fused
    
```

## region_increment.py:3 <module>

```

RegionIncrementRegions (#236) -- WHERE the TDM descriptor steps between STORAGE REGIONS.

WHY THIS EXISTS. A region-split tile is loaded by one `tensor_load_to_lds` PER REGION,
and the descriptor has to move between them. `KernelWriterAssembly.globalReadDo` used to do that
walk inside itself -- load half 0, advance, load half 1 -- and `tdmIncrementGir` subtracted the
advance back off so the per-chunk round trip closed. GIR emits one copy Move per region and calls

```

## short_path.py:404 _arm_is_unreachable

```
Is the `T < ...` arm dead code?  DERIVED from the peel-validity guard, not from M.

    This used to be `M == 1`, justified as "the arm's only step is unguarded because `T > 0` is
    trivially true, so it runs only at T == 0".  That reasoning was tied to a `T >= M` guard.  The
    guard is now strict (`T > M`, forced by the post-tested steady region -- #228), so the arm runs
    whenever `T <= M`, and at M == 1 that includes **T == 1**, an ordinary trip count.  The old rule
    would have folded an arm whose pairings provably break, on a path that really executes -- and it
    
```

## short_path.py:383 _folded_producer

```
The VALUE this consumer receives at `loc` on the FOLDED path (see `_trace`).

    The consumer is the same instruction on both sides, so it reads the same NUMBER of locations in
    the same roles -- but the location KEYS differ between the two chunk frames (the arms peel
    different chunk indices, hence different generations).  So look the location up directly when
    it happens to coincide, and otherwise fall back to the consumer's single shared/register use of
    the same space and operand.  A consumer that reads two locations of the same space and operand
    
```

## short_path.py:355 comment run

```
            # The arm is dead code: no legal trip count satisfies its guard.  Do not grow the
            # kernel for it, and do not let a (real, but moot) pairing difference read as a live
            # SPLIT.  Its own verdict so the reason stays visible rather than folded into FOLD.

```

## short_path.py:340 comment run

```
        # F0 first: an arm that cannot stand alone is a DEFECT, not a shape to choose between.
        # Reporting it as SPLIT would emit known-broken code; reporting it as FOLD would hide the
        # defect behind a path that happens to work.  Neither is an answer, so say so.

```

## short_path.py:316 comment run

```
            # F2 IS ASKED AT THE `wmma` ONLY.  A `wmma` is the sole instruction whose semantics the
            # two shapes must agree on; a read is a MEANS, and its identity is not stable across
            # them.  At `dr > 0` the drain's read at coord `(k0,m0)` of step 0 prefetches that

```

## short_path.py:306 comment run

```
        # F0 -- self-containment: could the short arm even stand alone?  A consumer whose value is
        # not produced inside the arm cannot be served by emitting the arm, so SPLIT is not a
        # fallback there and the disagreement is an outright defect.

```

## short_path.py:295 comment run

```
        # An "extra" is an instruction with no CONSUMER key.  It is not thereby uncorresponded:
        # its VALUE still participates, which is how the prologue's read-ahead fill supplies the
        # same value the arm's own first read does (#227).  Counted, not penalized.

```

## short_path.py:277 comment run

```
        # the folded path: the prologue, then the drain chain in peel order.  Derived from the
        # blocks' own frame (`chunk_base`), not from a name sort -- 'drain10' must not sort before
        # 'drain2'.

```

## short_path.py:228 comment run

```
# ===========================================================================
# the verdict
# ===========================================================================

```

## short_path.py:181 _trace

```
One forward walk of an arm: returns `(reach, keys, extras)`.

    `reach`  {consumer key: {location: the VALUE that reaches it}}
    `keys`   the consumer keys this arm defines
    `extras` instructions with no consumer key (a prologue fill; the drain's next-tile prefetch)

    TWO IDENTITIES, and conflating them is what made this analysis wrong (#227):
    
```

## short_path.py:162 pairing_cause

```
Classify one corresponded consumer's producer pair, or `None` when the two frames agree.

    ONE derivation, called only from `ShortPathFold.run`'s F2 loop -- it lives here rather than
    inline because the ladder's ORDER is load-bearing and a second copy would drift (see the
    recurring shape: the read-ahead traversal, the read's tree level and the fence cover were each
    one rule written twice, agreeing on every ordinary input and diverging on the degenerate one).
    
```

## short_path.py:111 _free_coord

```
The FREE-TILE part of `coord` for `operand` -- the coordinate that selects a distinct
 physical register, as opposed to the rotation slot.

 A read's register is named by TWO independent things : the rotation
 `(group, slot)`, which covers the reduction axis, and the operand's own free-tile coordinate,
 which the model leaves to allocation. Omitting the second merges registers that are physically
 distinct -- `A(k1,m0)` and `A(k1,m1)` share a slot but are different VGPRs -- and every read of
    
```

## short_path.py:82 _frame_base

```
The CHUNK POSITION a block's `gdelta = 0` is stated against, ON THE FOLDED PATH (#230).

    This is the correction that made PGR1 and PGR3 decidable.  A drain Ref's `gdelta` is NOT
    relative to the loop's exit; it is relative to the LONG PATH's chunk position `T`, so drain step
    `i`'s own chunk is `gdelta = i - M`.  Resolving that against the phi's entry value -- base 0 --
    silently assumes `M` is a whole number of ring revolutions:
    
```

## short_path.py:33 comment run

```
# ===========================================================================
# the two sequences
# ===========================================================================

```

## short_path.py:3 <module>

```

ShortPathFold (#183) -- may the `T < M` arm be FOLDED into the shared prologue/drain path?

THE TWO SHAPES.  TensileLite's scaffold and the LoopIR describe the same kernel differently:

    scaffold :  prologue -> Cond -> loop -> else -> drain    prologue+drain SHARED by both paths;
                                                              `T < M` just skips the loop

```

## swap_regions.py:223 comment run

```
        # The Mark names the pointer the way its hop names pointers: a read swap belongs to an
        # operand (`operand`), a copy swap to the Phi movement (`unit`, a member tuple).  Carrying
        # a tuple under the key `operand` would read as an operand name and mislead every consumer.

```

## swap_regions.py:188 _flow

```
Place the swaps for ONE physical pointer, via the shared ValuePlacementSolver.

        This analysis owns only what is specific to this register -- which instructions access it,
        what generation each demands, how a value re-frames across an edge, and the ring modulus.
        WHERE the defs go is `value_placement`, so the rule cannot drift from GrIncrementRegions'.

        What that replaced: a single forward walk that recorded a def at each point of demand and
        
```

## swap_regions.py:135 _edge_delta

```
Chunks the reduction timeline advances crossing `src`->`dst`.

    The difference of the endpoints' `chunk_base`, plus the trip advance on a back edge (the next
    trip's first body block sits `adv` chunks past this trip's first).  This is the conversion
    factor between the two blocks' frames; without it a steady value and a drain value are
    incomparable integers.
    
```

## swap_regions.py:110 _topo_order_swap

```
Topological order of the CFG with back edges (and the unmodeled short path) removed.

    Forward propagation needs every predecessor's exit value settled before a block is visited;
    with the back edges cut the CFG is a DAG, so one topological pass suffices and no iteration to
    a fixpoint is needed -- the loop header's incoming value comes from its PREHEADER, and the back
    edge is reconciled afterwards by the closure step.  Any block left unordered (a cycle the
    back-edge removal did not break) is appended so it is still visited, deterministically.
```

## swap_regions.py:97 _unemitted_edges

```
Edges into a block NO BACKEND EMITS -- the only paths the placement problem may exclude.

    REPLACES `_loop_bypass_edges` (#222/#234), which recognised "a `CondGoto` whose taken side
    enters a loop header" and dropped the other side.  Two things were wrong with that:

      * it was a CFG-shape PATTERN MATCH, not a derivation, and
      * it excluded the FOLDED `T < M` entry too -- an edge the emitted code really does take, into
    
```

## swap_regions.py:29 _hop_access

```
`(unit, shared_ref)` if `inst` is a Move on `hop` ('read'|'copy'), else None.

 `unit` is the pointer identity, and it is NOT the same kind of thing on the two hops:

 read -> the operand name. A's read and B's read are different instructions at different
 addresses, so each owns its own LocalReadAddr.
 copy -> the Phi movement's member tuple. A fused group is ONE cooperative instruction over
    
```

## swap_regions.py:22 comment run

```
# The pointer value the scaffold establishes before the kernel body runs: both the TDM descriptor
# and LocalReadAddr are initialized to the first buffer.  A modeled fact, named rather than a bare
# 0 in the fixpoint seed.

```

## swap_regions.py:3 <module>

```

SwapRegions -- physical-pointer swap placement, as a REAL forward
dataflow analysis over the GIR CFG.

The physical LDS read pointer (LocalReadAddr) and the TDM write descriptor are SINGLE registers
whose selected generation must EQUAL the generation every access names. A swap is the only
instruction that changes one, so "where do swaps go" is a reaching-value problem: propagate the

```

## uniform_placement.py:92 comment run

```
                    # A block NO backend emits: no agent reaches the selector, so no agent can be
                    # brought to it and not another -- this is not the L733 failure.  It IS a
                    # SAFETY question (the hazard the fence covered goes unfenced on that path),

```

## uniform_placement.py:3 <module>

```

UniformPlacement -- the ONE LIVENESS check the empty-ledger
gate does not subsume.

 "for each `scope=block`/`cluster` `Await` over agent set `G`, verify its region-tree position
 DOMINATES every agent in `G` uniformly -- it lies on a node no `Branch`-over-residues or
 agent-discriminating `Cond` arm can cause an agent of `G` to skip" (L733)

```

## value_placement.py:181 comment run

```
                # AN AMBIGUOUS VALUE NOBODY CONSUMES IS NOT A CONFLICT (#234).  `want` is the
                # ANTICIPATED demand -- this block's, joined with everything downstream of it -- so
                # `want is BOTTOM` means no consumer on any path from here ever reads this

```

## value_placement.py:157 solve

```
[Placement], deterministically ordered.  `on_conflict(kind, detail)` reports a TOP meet
        or an unsplittable critical edge; it is expected to raise.

        Three def kinds fall out of the two fixpoints, and they are disjoint by construction:
          intra-block -- between consecutive accesses in b whose requirements differ;
          exit def    -- at b's end, when every successor agrees on a value b does not already hold;
          head def    -- at b's start, when what actually arrives differs from what b demands (which
        
```

## value_placement.py:129 avail

```
Forward fixpoint over what LEAVES each block.

        The subtlety that a naive version gets wrong: a def placed at a block's exit CHANGES the
        value leaving it, so a successor must see the post-def value.  Propagating the pre-def value
        makes the successor re-derive the same transition and emit a SECOND def for it -- a real
        double emit, caught by the accessless-block test.  So `leave` (post-def), not `body_out`
        (pre-def), is what crosses the edge.
        
```

## value_placement.py:59 ValuePlacementSolver

```
Solve def placement for one register.

    `accesses`   -- {block label: [RequiredValue]} in program order.
    `preds`/`succs` -- the CFG to solve over, ALREADY filtered (back edges included, unmodeled paths
                   excluded).  Back edges must appear in both, or the loop-carried def is lost.
    `edge_delta(p, b)` -- value frame shift crossing p->b (added going forward).
    `entry`      -- label of the program entry block; `entry_value` is the value the scaffold
    
```

## value_placement.py:3 <module>

```

ValuePlacement -- where the defs of a single-register resource go, as two real fixpoints.

Both pointer analyses solve the same problem for a different register: ONE physical register (an
LDS read pointer, a TDM write descriptor) must hold a specific value at each access, and a Mark
(`swap` / `gr_increment`) is the only thing that changes it.  "Where do the Marks go" is therefore
not a search -- it is the classic code-motion pair, and this module is the single implementation

```

## analysis.py:3 <module>

```

GIR analysis INFRASTRUCTURE -- the base class + the lazy cached manager.

This module is infra ONLY. The concrete analyses each live in their own module under
`gir/analyses/` as an `Analysis` subclass:

 analyses/cfg.py Dominators, BackEdges (+ successors helper)

```

## emit_plan.py:179 comment run

```
    # THE WMMA'S REGISTER SOURCES, AS A DERIVED LIST -- `(operand, group, slot, tile)` per src.
    #
    # The scalars below (`bufA`/`bufB`/`bufMXA`/`bufMXB`) are the LEAF's view: `gir_to_rocisa` maps

```

## emit_plan.py:156 comment run

```
    # Per-operand register generation (buffer/slot) + size the wmma CONSUMES -- carried on the
    # src Refs by `loopir_to_gir._convert_mma` from LoopModel. The leaf
    # names its source vgpr from these, NOT the old `u % W`.  Operand identity is on the Ref's

```

## emit_plan.py:119 comment run

```
            # IS THIS READ A READ-AHEAD? `advance` is theta's pt5 shape on the node (0 = INPLACE,
            # issued for its own wmma; non-zero = HOISTED, fetching a later generation).  It is
            # carried so a checker can tell a refill from an ordinary read WITHOUT keying on the

```

## emit_plan.py:102 comment run

```
        # ONE ACT, MANY LOCATIONS -- the coverage, projected (#312). A folded read is ONE
        # instruction (one act, at the carrier group's leader) that fills every coordinate of its
        # group, and `Ref.covers` is where `loopir_to_gir._absorbed` recorded that span.  The five

```

## emit_plan.py:97 comment run

```
        # AND THE ABSOLUTE REDUCTION INDEX, which `k` is not once a region carries part of it.
        # This is the SAME projection `_plan_mma` uses for `u`, so a read and the wmma that
        # consumes it are directly comparable: the pair (tile_flat, k_flat) is the SOURCE

```

## emit_plan.py:90 comment run

```
        # THE REGISTER INDEX IS THE FLAT ONE, AND THAT IS NOT THE SAME NUMBER.  A region split
        # displaces LDS STORAGE, not the register file: the wave still holds one vgpr group per
        # free tile, numbered across the whole tile.  So the address wants (region, within-region)

```

## emit_plan.py:86 comment run

```
        # THE REGION IS PROJECTED OVER **ALL** ITS AXES, FREE OR REDUCTION.  This read
        # `freeset & rmodes`, which is the same thing only while every region axis happens to be a
        # free one -- true for the MT half (`M_split`/`N_split`) and false for the DU half, whose

```

## emit_plan.py:78 comment run

```
        # THE REGION IS AN ADDRESS FACT, NOT ONLY A TOKEN (#237).  A region-split tile's storage
        # regions are DISPLACED in LDS (the copy walks the descriptor by one split boundary per
        # region, #236), so a read of a tile in region r is at `r * splitBoundary`, not at the

```

## emit_plan.py:73 comment run

```
        # a READ: shared(or global DTV) -> register.  free-tile index = project the read's coord
        # onto THIS operand's free modes; reduction index = project onto the reduction modes.
        # PRESENCE-DERIVED -- handles multi-mode roles (M_split+M_inner) and any operand.

```

## emit_plan.py:37 _project

```
Mixed-radix combine of `coord` over `modes` (an operand's free-mode set, or the reduction-
    mode set), in `inner_order` (loop order order, innermost = fastest radix), with radices from
    `extents` (a {mode: extent} map carried in Program.meta from theta).  The the axes it varies over-derived
    replacement for the old role_to_mode single-mode lookup: a role spanning SEVERAL modes
    (M_split+M_inner under a 6-axis order) combines into ONE index instead of silently dropping all
    but one -- the 6-axis break.  `coord` is a dict {mode: value}; absent modes count 0.  A single
    present mode reduces to its own value (stride 1).  Empty `modes` -> 0.
```

## emit_plan.py:3 <module>

```

Emit plan -- the PURE half of GIR -> rocisa (L3).

`plan_block(prog, phase)` walks a FINALIZED GIR block body (after the pipeline has applied swap /
gr_increment Marks and stamped tokens) and returns an ordered list of `EmitAction`s: the concrete
per-leaf instructions L3 must emit, with every index/generation/token already resolved from the
GIR nodes. It imports NOTHING from rocisa, so the index/order logic -- the correctness-critical

```

## nodes.py:362 Program

```
The whole mainloop as one minimal CFG.

 blocks -- {label: Block}, insertion-ordered.
 entry -- entry block label.
 tiles -- interned Tile table (optional; may be empty in R0b).
 params -- PRESETS analyses derive facts from (StaggerU, MI/MT,...); not raw data.
 meta -- {S, roles, dtypes,...} scalar metadata.
    
```

## nodes.py:310 Block

```
One software-pipeline Phase as a basic block in the GIR CFG.

 phase -- 'prologue' | 'steady{n}' | 'drain{n}' | 'short{n}' | 'tail' (also the block LABEL).
 'short{n}' is the `T < M` degenerate arm: present only until
 FoldShortPathPass has ruled, then either folded away or kept as a real second path.
 loop -- steady: its CondGoto taken-target is itself (a back-edge).
 phis -- Gen phis (loop header): v = phi(entry, v_next). List of GenPhi.
    
```

## nodes.py:278 successor_labels

```
THE successor list of a block -- the single authority every CFG walk uses.

    It lives here, next to the terminator definitions, because a second copy of this switch is a
    silent-wrong-answer waiting for the next terminator kind: `Program.walk_rpo` used to carry its
    own that handled only Goto/CondGoto, so a `CondChain` header (G1) or a `Return` sink (G4) had
    NO successors as far as reverse-postorder was concerned -- its successors fell through to the
    'unreachable, append in insertion order' path and `gen_reaching` would then visit them in the
    
```

## nodes.py:274 comment run

```
# ===========================================================================
# 4.6  Block and Program (the minimal CFG)
# ===========================================================================

```

## nodes.py:255 CondChain

```
Multi-exit terminator -- an ORDERED chain of guarded exits, then a default (G1).

    A loop header may need several typed early-exits taken BEFORE the loop is entered, all testing
    the SAME counter against different bounds: with a prefetch depth of `d` the trip count decides
    how much of the pipe can actually be filled, so a short trip enters a partially-drained variant
    instead of the steady loop.  That is a *chain* (`c <= d-1 -> ...`, `c <= d -> ...`, `c == 1 -> ...`,
    else fall through), not one two-way branch, and it is ORDERED -- the first satisfied arm wins,
    
```

## nodes.py:243 Return

```
Function-early-exit terminator -- a SINK with no successors (G4).

    Some drain variants finish the whole kernel in place rather than falling through to a common
    exit (an optimized no-load-loop that runs the epilogue and returns mid-drain).  Modeling that
    as `Goto('end')` would be a lie in one specific way: `'end'` is a pseudo-target every analysis
    already filters out, so a `Return` and a fallthrough-to-exit would be indistinguishable, and a
    pass could "helpfully" retarget the edge.  A distinct node makes the sink explicit -- CFG
    
```

## nodes.py:228 LoopBack

```
Loop terminator: run `body` for `trips` iterations, then go to `exit_target`.

    Replaces the `CondGoto` that used to carry the steady back-edge.  A `CondGoto` says "branch on
    this comparison"; that is an implementation of a loop, not a description of one, and GIR is the
    description layer.  The CFG is unchanged -- `body` and `exit_target` are the same two successors
    a `CondGoto` reported, so dominance, back-edge detection and every traversal behave identically.
    
```

## nodes.py:208 Trips

```
A loop's DYNAMIC TRIP COUNT: `(var - sub) // div`, or the constant `-sub` when `var` is "".

 WHY A COUNT AND NOT A COMPARISON (#231). A trip count is what the model actually states -- the
 model's steady region is a RANGE, `[delta, T)` -- while a comparison is one ENCODING of it, and the
 encoding needs three facts that live outside the node to decode: where the counter starts, which
 way it steps, and where the test sits. None of those are shared with the backend: TensileLite
 initializes its `loopCounter` to the loop count, DECREMENTS, and exits at
    
```

## nodes.py:166 Pred

```
A STRUCTURED symbolic predicate over trip-count symbols (no hardware): `lhs op rhs`.

    Read BY FIELD -- `lhs` (the counter symbol), `op`, `rhs` (a `Bound`) -- never by parsing a
    string.  It used to be a free-form `expr: str`, which meant a consumer that wanted to know
    "which counter does this test, and against what?" had to parse `'iter < T - 2'`; the
    multi-exit header (a chain of typed early-exits on the same counter) and the drain-multiplicity
    matrix both need exactly that comparison, structurally.
    
```

## nodes.py:145 Bound

```
The right-hand side of a terminator predicate: the constant `const` when `var` is empty,
    else the trip-count symbol `var` offset by `const` (negative for `T - M`).

    Two fields cover every terminator predicate the lowering emits or the scaffold needs -- the
    peel-validity guard (`T >= M`), the steady back-edge (`iter < T - M`), a short-loop step
    (`T > t`), and the typed early-exits (`counter <= PGR-1`, `counter == 1`).  This is
    deliberately NOT the LoopIR's full index `Expr` (mixed-radix terms, moduli, chunk-carry):
    
```

## nodes.py:137 comment run

```
# ===========================================================================
# 4.5  Control-flow primitives (the CFG terminators)
# ===========================================================================

```

## nodes.py:126 Mark

```
A body element carrying NO data movement or compute -- only a GEMM-semantic FACT at a
    program point for L3 to realize (R-SEMANTIC).  `Mark('swap', {hop, gen_from, gen_to})` is
    the FACT "the read generation changes here", NOT a v_xor.

    kind -- one of MARK_KINDS (native from LoopIR structure, or produced by an analysis).
    at   -- kind-specific payload dict (schema in verify.py G-MARK).
    
```

## nodes.py:117 comment run

```
# ===========================================================================
# 4.4  Mark -- a semantic point (+ the region-based insertion model)
# ===========================================================================

```

## nodes.py:86 Move

```
THE data-movement verb: one hop down the memory hierarchy.

 srcs -- (Ref,) e.g. a global tile.
 dsts -- (Ref,) e.g. a shared tile at a Gen (a copy), or a register (a read).
 deps -- RAW + WAR edges carried from the LoopIR awaits (kind on each dep, / analysis).
 token -- the SEMANTIC completion token this Move produces/consumes, stamped by TokensPass
 (e.g. ('lds', generation) -- which LDS buffer a read consumes / a copy fills, so L4
    
```

## nodes.py:81 comment run

```
# ===========================================================================
# 4.3  Move and Mma -- the verbs (unified srcs/dsts; semantics from Tile.operand)
# ===========================================================================

```

## nodes.py:72 comment run

```
                                 # CONSUMES : read from the LoopIR slot Expr.mod, never
                                 # decided in GIR; reg_band only validates it.
    # -- the MOVEMENT QUANTUM footprint

```

## nodes.py:43 Gen

```
A loop-carried GENERATION -- the ONE SSA value class in GIR.

 Defined by a Phi in a loop header; the back-edge transfer is v' = (v + adv) % ring.
 `ring` is the buffer ring size S (1 = in-place, the generation never changes).
 `id` distinguishes the distinct staged buffers (e.g. A-shared vs B-shared) that rotate
 independently.
 
```

## nodes.py:21 Tile

```
A logical operand fragment -- 'what data', no color / no address.

 operand -- 'A'|'B'|'C'|'acc'|'scaleA'|'scaleB'|'meta'|... (open set; identity of an Mma
 operand is read from here, -- there is no separate `role` map).
 space -- 'global'|'shared'|'register'.
 coord -- the subset of the 6 canonical modes this fragment covers, as a dict
 {mode: int|None}. NO `iter` ever appears (G-NOITER); the cross-trip relation
    
```

## nodes.py:16 comment run

```
# ===========================================================================
# 4.2  Tile, Gen, Ref -- the nouns
# ===========================================================================

```

## nodes.py:3 <module>

```

GIR nodes -- the GEMM-dataflow IR (layer 2), per the lowering-design 

GIR is the mutable optimization surface between the LoopIR (layer 1, LoopModel's proven-legal
rolled schedule) and rocisa (layer 3). It is a real, verifiable CFG whose ONE SSA value class
is the loop-carried LDS `Gen`. It carries GEMM semantics only -- never machine
realization (R-SEMANTIC): a swap is a `Gen`-transition FACT, not a `v_xor`; there is no

```

## __init__.py:3 <module>

```

GIR passes. Each pass is its own class in its own module; the ordered pipeline
is data in pipeline.py (the "backend" file).

 base Pass / StructuralPass base classes (the CFG-edit contract)
 fold_short_path FoldShortPathPass -- fold/keep the `T < M` arm, on a proof (#183)
 reg_gen RegGenPass -- register (group,slot) authority; W validated (B3)

```

## apply_marks.py:3 <module>

```

ApplyMarksPass -- the only BODY mutator among the swap passes.

Inserts each placed PendingMark's Mark into its block body at the resolved anchor:
 - anchor (block, node) -> insert immediately before `node`
 - anchor (block, BLOCK_EXIT) -> append at end of the block body

```

## base.py:34 StructuralPass

```
A pass that may CHANGE THE CFG: add or remove blocks, retarget terminators, move the test.

    WHY THIS IS A SEPARATE KIND.  Everything downstream of the CFG is derived from it -- dominators,
    back edges, generation reaching, pointer placement, the fence cover -- so a structural edit
    invalidates a different and larger set of facts than a body edit, and it can change the SHAPE
    the model is expressed in (a post-tested loop and a pre-tested one need different bounds for
    the same trip count).  `ApplyMarksPass` described itself as "the ONLY IR mutator" for a long
    
```

## base.py:3 <module>

```

Pass base class.

A Pass mutates the Program in place and returns the set of analysis names it invalidated (a hint
for the manager; the manager also invalidates via prog.version bump). A pass that mutates MUST
call `prog.bump` so cached analyses keyed on prog.version are dropped. Pass ORDER is data --
it lives in pipeline.py (the "backend" file), a la StinkyTofu's Gfx1250Backend.

```

## collect_pending.py:3 <module>

```

CollectPendingMarksPass -- run the region analyses and stash their PendingMarks on
`prog.pending`. IR-UNTOUCHED (it only reads analyses and records their plan); the placement +
apply passes downstream turn the plan into body Marks.

R2 ran SwapRegions only; R3 adds GrIncrementRegions; #206 adds FenceRegions. GuardSite
(gsu_guard) lands in R4.

```

## early_exit.py:44 comment run

```
            # only the peel-validity guard: a two-way branch whose false arm is the chain head and
            # whose taken arm is the steady loop.  Identified STRUCTURALLY (targets), never by label
            # -- ScaffoldMapPass has not run yet, and must not need to have.

```

## early_exit.py:3 <module>

```

EarlyExitPass (G1/G3) -- give the CFG the `T < M` entries the scaffold actually takes.

THE EDGE THAT WAS MISSING.  With peel depth `M` the drain is a chain of `M` steps, and a trip count
`T <= M` does not run the steady loop at all: it enters the chain ALREADY PARTIALLY DRAINED, at step
`M - T`, because the prologue only managed to fill `T` of the `M` generations.  `FoldShortPathPass`
records that as an obligation it cannot discharge ("entering the drain chain at step `M-T`,

```

## fold_short_path.py:95 comment run

```
        # THE FOLDED EDGE INHERITS THE ARM'S FRAME (#233).  Rewiring `-> short0` into `-> drain0`
        # does not merely change a target: the two land at DIFFERENT positions on the reduction
        # timeline.  `drain0.chunk_base` is the LONG path's (one chunk past the last steady trip);

```

## fold_short_path.py:88 comment run

```
        # FOLD / VACUOUS -- reroute the peel-validity false edge into the drain chain and drop the
        # arm's blocks.  Derived from the analysis's own block lists, so this cannot disagree with
        # what was proven.

```

## fold_short_path.py:67 comment run

```
            # Both arms stay.  The kernel grows by one short region, which is the correct price for
            # a reordering that provably does not hold -- and it is only paid when the ring is
            # shallower than the peel depth.

```

## fold_short_path.py:43 comment run

```
            # TRIED AND REVERTED 2026-08-21: scoping this raise to non-`model_only` arms.  The
            # reasoning was that `short_loop['model_only']` says no backend emits `short0` (the
            # scaffold's `toPGR1` path owns `T < M`), so "emitting the arm would ship known-broken

```

## fold_short_path.py:3 <module>

```

FoldShortPathPass (#183) -- the MERGE pass: coverage the `T < M` arm into the shared prologue/drain
path when that is provably legal, keep it as its own path when it is not, and raise when neither
shape works.

WHY A PASS EXISTS AT ALL.  TensileLite's scaffold and the LoopIR describe the same kernel with
different STRUCTURE:

```

## pipeline.py:103 _check_rotation_waw

```
The premise under which a rotation WAW needs no ledger row of its own.

    Non-interference covers WAW as well as WAR, so ordering a rotating buffer really means
    `WAR and WAW`.  Our ledger files only the WAR (`ledger.py`: RAW-residency,
    inplace-WAR/rotation-WAR, crossing-RAW, readahead-residency; `grep -rni waw Tensile/LoopModel`
    returns nothing).  That is CORRECT, but only under a premise the decoder never established:
    
```

## pipeline.py:80 comment run

```
                # `dep` is a `ledger.Endpoint`, not a string -- `.op` is the operand name, which is
                # the same currency `SharedTouch.operand` speaks.  Keying on the record itself compares
                # an Endpoint against a str and silently never matches.

```

## pipeline.py:62 _check_block_scope_covered

```
`Await.scope` is LOAD-BEARING here (#276-S2), and this is what makes it so.

 gives every obligation a proc-scope: "wave" where program order plus a per-wave counter
 already covers the edge, "block" where they provably cannot -- `s_wait_loadcnt` retires THIS
 wave's loads and says nothing about another's, so under a cooperative fill a reader waiting its
 own counter has no edge to the agent that actually wrote the bytes.
    
```

## pipeline.py:28 pipeline

```
The backend pass order (data): coverage -> reg -> tokens -> collect -> place -> apply -> scaffold-map.

    FoldShortPathPass runs FIRST because it rewrites the CFG every later analysis reads: folding
    ScaffoldShapePass runs FIRST and is the ONLY stage allowed to change the CFG: it folds the
    `T < M` arm, adds that arm's early-exit edges into the drain chain, and labels the terminators.
    Every pass after it is an analysis over that CFG -- reaching generations merge over `preds`,
    hazards pair by reachability, the fence cover quantifies over "every path reaching the
    
```

## pipeline.py:3 <module>

```

The GIR pass pipeline -- pass ORDER is DATA (the "backend" file, a la
StinkyTofu's Gfx1250Backend).

The ONE backend pipeline :
 ScaffoldShapePass the CFG is made FINAL here: coverage the `T < M` arm (#183), add the
 `T < M` early-exit edges into the drain chain (G1/G3), then attach

```

## placement.py:3 <module>

```

PlacementPass -- resolve each PendingMark's Region to a concrete anchor.

Policy travels with the Region (design "placement by swap TYPE -- a user directive, not just
sigma"), because it differs by what the Mark mutates:

 EARLIEST -- every TDM descriptor change (`gr_increment` AND the copy-hop `swap`): land immediately

```

## record_unemitted.py:3 <module>

```

RecordUnemittedPass (#232) -- count what the model-only blocks are carrying away, LAST.

`FoldShortPathPass` decides WHICH blocks no backend emits (the kept `T < M` arm -- TensileLite owns
that runtime branch through its own `toPGR1` path) and records the reason.  It cannot record HOW
MUCH is discarded with them, because it runs FIRST: the fences and swaps that end up in those
blocks are placed later, by `CollectPendingMarksPass` / `ApplyMarksPass`.  Counting at coverage time

```

## reg_gen.py:3 <module>

```

RegGenPass -- the register (group, slot) authority; W is READ, not decided.

The lowering already assigns each register read Ref its concrete (group index, slot) and carries
the rotation width W on `Ref.reg_ring` (from the LoopIR slot modulus). This pass:
 - runs RegBand to VALIDATE W is consistent per (operand, group) -- raising on a LoopIR defect;
 - is the sole author of the register color/slot (here: it confirms every register read Ref has a

```

## scaffold_map.py:48 comment run

```
                # LABEL BY THE ARM'S TARGET, NOT ITS POSITION.  The point of this pass is to hand a
                # consumer the name TensileLite already uses, so where one exists it wins over the
                # generic role name -- position said `NoGlobalLoadLoop_0` for the branch the scaffold

```

## scaffold_map.py:3 <module>

```

ScaffoldMapPass (the LoopIR->TensileLite-scaffold mapping) -- the ONE place that maps the GENERIC,
backend-agnostic control flow (the peel-validity guard, the steady back-edge, the drain chain) onto
TensileLite's favored scaffold LABELS (`toPGR1`, `LoopEndL`, `NoGlobalLoadLoop_k`).

Why a pass, not the lowering: the pure LoopModel core emits only generic guard KINDS, and `loopir_to_gir` builds a
generic reducible CFG whose terminators carry LABEL-FREE `Pred`s. All TensileLite scaffold naming

```

## scaffold_shape.py:3 <module>

```

ScaffoldShapePass -- the ONE CFG-shaping stage: fold, then early-exit, then label.

WHY THESE THREE ARE ONE PASS.  They are the whole of "make GIR's CFG the CFG the scaffold will
actually run", and every later stage is an ANALYSIS OVER THAT CFG -- `GenReaching` merges over
`blk.preds`, `LdsHazards` pairs by reachability, `FenceRegions` covers "every path reaching the
consumer", `MemTokenAssignment` names buffers off the reaching generation.  An edge added after any

```

## tokens.py:109 comment run

```
                # ...and the BACKEND storage name(s) alongside it (#235).  Stamped here, next to
                # the completion token, because the two are read together and drifting them apart
                # is how the emitter ended up deriving one of them from a byte offset.

```

## tokens.py:80 _buffer_generation

```
The buffer generation this access names -- ONE generation, exact (#207).

    This is the COMPLETION token ("what completed"), not the storage name.  The loop-carried
    copy->read aliasing question confirmed 2026-08-27 lives in `MemTokenAssignment.Buffer`
    (`token_ids`, "which storage"), which carries the same #207 narrowing and IS what reaches the
    backend as `memToken`; widening here instead is emission-inert (measured: the failing cell's
    `s_wait_tensorcnt 3,4,4` was byte-identical before and after).
```

## tokens.py:61 _region_key

```
The region coordinates that identify WHICH movement instance `ref` touches, under the
 per-region pi -- else ``.

 A storage-region split emits one movement instance per region value, and reads that as "two fused movements with TWO COMPLETIONS". Under the per-region pi those
 instances are distinct completion classes, so the token -- which is what an order-driven
 backend pairs a read against -- has to name the region; otherwise every region of an operand
 collapses onto one token and a read must await them all. Under the default (shared) pi the key
 is empty and the token keeps its 3-tuple shape, byte-identical to before.
```

## tokens.py:39 _unit_key

```
The CANONICAL key of the completion unit that produces `operand` -- the same key
    `Theta.movement_units()` uses, looked up rather than re-derived.

    theta owns which movements exist (which Phi members are actually copy operandes, and in what
    order); deriving the key a second time here produced two spellings of one thing -- a sorted
    tuple vs the fuse-group order, and `'A'` vs `('A',)` for an unfused operand -- so the
    granularity lookup silently missed.
```

## tokens.py:3 <module>

```

TokensPass -- stamp each LDS Move with its completion token.

Replaces the old walker's `_stepReadToken`: a read's `sync LDS%u` token must name the LDS buffer
GENERATION it consumes, so L4 (SIA4/StinkyTofu) pairs the read with the WRITE that filled that
buffer. The generation is a GIR FACT (GenReaching over the shared Ref), so the token is a direct
map -- no seed, no flip-on-change. A copy's token names the buffer it fills.

```

## quantum.py:165 comment run

```
            # THE STRADDLE. One instruction, two generations: the single completion cannot
            # discharge both residencies, so the merge is outside Pi.  Reported, not repaired,
            # because the repair is to NARROW the load (`Decision.narrowed`) and that needs a

```

## quantum.py:160 comment run

```
        # THE LDS SIDE OF A STRADDLE IS A LABEL, NOT AN ILLEGALITY.  `_act_generation` carries two
        # facts and they have OPPOSITE remedies.  The register half (`reg_buf`/`group`) is
        # unencodable -- one instruction cannot write two `Valu*_X*` bases -- and stays a

```

## quantum.py:146 comment run

```
            # TWO DIFFERENT FAULTS ARE REPORTED BY THIS FUNCTION AND THEY ARE NOT THE SAME ONE.
            # This branch is an INCOMPLETE CARRIER GROUP: the acts a merged instruction would serve
            # are not all in this block.  That is an EMISSION/LIVENESS fact.  The branch below is

```

## quantum.py:141 comment run

```
        # ONE ACT PER CARRIER GROUP, AT THE LEADER -- the invariant since theta models the coverage itself.
        #
        # This check used to count acts and demand `want` of them (the group size), because theta

```

## quantum.py:126 comment run

```
        # THE LEADER IS THE FIRST TILE OF THE GROUP, not the carrier VALUE.  `carrier_of` returns
        # an instruction index, which is not a tile index and must not be compared to one -- the
        # group is the preimage, and its least member is the act that issues.

```

## quantum.py:112 comment run

```
        # theta'S CLAMP OVERRIDES THE SUPPLIED MAP.  The `QuantumMap` is a TARGET fact; whether a coverage
        # is LEGAL on this operand's axis is theta's, by the regularity condition ("`q` must divide
        # the axis extent ... an odd extent under a two-tile coverage clamps to `q=1`, i.e. no coverage ...

```

## quantum.py:93 plan_quantum

```
Decide, for every read act, whether it emits -- from theta's SUPPLIED merge, not from a rule here.

 `quantum_of(operand) -> ir.QuantumMap | None` is theta's own field, and `None` (the default) is the
 identity merge: one instruction per act, which is every A/B read. `extent_of(operand) -> int`
 bounds the tile scan for a carrier preimage; it defaults to the acts actually present.

 THIS DOES NOT MODEL THE FOLD, it OBEYS it. The act whose tile IS its own carrier emits; the
    
```

## quantum.py:45 _act_generation

```
The GENERATION a read act belongs to -- what every coordinate of one coverage must agree on.

    Two facts, and both are load-bearing.  `token_ids` is the LDS generation the act CONSUMES (which
    buffer of the shared ring); `reg_buf`/`group` is the register generation it DEFINES.  A merge
    crossing either is one instruction spanning two steps: the LDS side is the measured MXSA failure
    in the module docstring, and the register side would need one instruction to write two different
    `Valu*_X*` bases, which has no encoding at all.
```

## quantum.py:38 comment run

```
#: NO DECISION.  An act whose operand theta gave no merge gets `None`, NOT a permissive `Decision` --
#: the consumer must be able to tell "theta said every act issues" from "theta said nothing", because the
#: leaf's fallback coverage is only correct in the second case.  Returning `Decision(emit=True)` here

```

## quantum.py:24 Decision

```
What `leaves.emitLdsReadTile` must do with one read act.

    `emit` -- issue an instruction (False = this act's payload rides a leader's instruction).
    `narrowed` -- the group was not mergeable, so the coverage was cut to one tile and this act emits
                 its OWN narrower load.  The leaf reads this to pick the narrowed read context.
    `reg_slot` -- the register-side index when it diverges from the tile index; None = the leaf's
                 ordinary derivation.
    
```

## quantum.py:17 comment run

```
# THERE IS NO `Span` HERE ANY MORE.  It carried `load` / `partitions` / `vector` and rebuilt the
# coverage from `ctx.tilePerRead` / `ctx.tileSpanVW` -- a SECOND representation of the same fact, which
# had to agree with theta's by hand.  The merge is now ONE object, `ir.QuantumMap`: two Exprs over the

```

## quantum.py:3 <module>

```

The MOVEMENT QUANTUM: which tiles one instruction carries, decided on the emit plan.

WHAT THE QUANTUM IS. Every hop moves a determinate block of data in ONE instruction, and the set
of axes that one instruction covers is the hop's **coverage**. By those axes are *absent* from
the hop's the axes it varies over, so the coordinates inside one coverage are ONE the axes it varies over point -- hence
`step(element, p)` gives them **one step, therefore one generation**. The model states this per hop

```

## refs.py:86 covered_coords

```
Every coordinate ONE reference fills -- the coverage's ABSORBED AXES, re-inserted.

 `ref.covers` is `((axis, extent),...)`: the inner modes this one instruction SPANS, which
 `geometry.quantum_axes` has already removed from the read hop's the axes it varies over. Empty (the default)
 means the reference fills exactly `ref.tile.coord` -- every ordinary read.

 THIS IS THE SECOND HALF OF THE SUBTRACTION AND IS USELESS ALONE. Because the axis is gone from
    
```

## refs.py:50 copy_unit

```
`(members, refs)` for a global->shared copy Move, or `(None, None)` if `inst` is not one.

    `members` is the tuple of operand names the movement carries -- `('A',)` unfused, `('A','B')`
    for the Phi-fused AB group that multi-wave TDM realizes as one aliased descriptor.  `refs` is
    the matching tuple of shared destination Refs.

    Why the tuple and not `refs[0].tile.operand`: `loopir_to_gir._convert_load` builds one Ref per
    
```

## refs.py:37 _rotation_state

```
The ROTATION STATE a shared Ref names -- what a pointer register would have to hold to serve
    it -- as a comparable key.

    Deliberately NOT the Gen itself.  `Gen.id` distinguishes the independently-staged buffer sets
    (A-shared vs B-shared), and a fused movement fills BOTH: A's LDS region and B's are different
    memory, so different ids is the correct model, not a discrepancy.  What makes them ONE pointer
    is that they rotate in LOCKSTEP -- same ring depth, same offset -- so a single register value
    
```

## refs.py:3 <module>

```

refs -- the shared Ref accessors for a Move's hops.

Small on purpose, and shared on purpose: three modules need the SAME answer to "which operand(s)
does this Move's shared hop belong to", and each of them used to compute it by taking the first
shared Ref. That is wrong for a Phi-fused movement, which carries one Ref PER MEMBER.

```

## region.py:27 Region

```
A legal placement window over one block's body (opaque ordinals).

    block  -- the block LABEL the placement point lands in.
    after  -- place strictly AFTER this anchor (a body node, or BLOCK_ENTRY).
    before -- place strictly BEFORE this anchor (a body node, or BLOCK_EXIT).
    policy -- where in the window to land.  It travels with the Region because it differs by what
             the Mark mutates, so a single global policy in PlacementPass is wrong:
    
```

## region.py:3 <module>

```

Region + PendingMark -- the analysis->placement->apply insertion model.

A `Mark` (swap / gr_increment / gsu_guard) has a LEGAL PLACEMENT RANGE, not a single correct
point: a read-hop swap gen_from->gen_to may sit anywhere after the last access using gen_from
and before the first access using gen_to. Baking a fixed body index into an analysis is
premature realization (the mistake the old walker made).

```

## render.py:301 gir_counts

```
Per-block op counts {block: {'read':n,'copy':n,'mma':n}} for the op-parity assertion.

    A MODEL-ONLY BLOCK IS COUNTED, AND SAYS SO.  Such a block is one GIR reasons over but no
    backend emits; folding its ops into these counts silently made the totals disagree with the
    emitted .s for a reason nothing in the output stated.  The count stays (a consumer comparing
    against the LoopIR wants every op GIR holds), but the flag rides along so a consumer comparing
    against the ASSEMBLY can subtract it.  Reporting the number without the flag is the part that
    was wrong.
```

## render.py:271 comment run

```
        # THE BLOCK'S OWN FRAME, and whether anything emits it.  Two blocks with identical bodies
        # are DIFFERENT programs when their generation frames differ, and `gen_rel` is what says a
        # block's entry generations sit on an arbitrary base (drain0's predecessors genuinely

```

## render.py:223 comment run

```
            # SAY WHICH HOP, AND SAY NOTHING ABOUT THE COORDINATES.
            #
            # This line first read "AGENT-relative: wave w reads region w, so ONE agent's reads

```

## render.py:204 _theta_facts_lines

```
The theta facts the BLOCKS ARE DERIVED FROM, which the block listing itself cannot show (#287).

    A GIR block prints coordinates and operands.  Whether a given coordinate is a REGION, whether
    that region is the AGENT's rather than the coordinate's, which operands share one Phi-fused
    movement, and how many storage regions the movement actually has are all facts of the theta the
    program was built from -- invisible in the body, and each one changes how the body must be read.
    
```

## render.py:181 comment run

```
    # The obligations are the part a reader debugging this path most needs: they are what the coverage
    # does NOT discharge and hands to the scaffold.
    #

```

## render.py:170 comment run

```
    # PRINT THE VERDICT, do not assert one.  This used to hardcode "NOT lowered to GIR blocks
    # (#154)", which stopped being true at #183/#219-223 -- the arm IS lowered to `short{i}` blocks
    # and then folded into the drain chain when `ShortPathFold` says the def-use pairings survive.

```

## render.py:156 _short_loop_lines

```
The `T <= M` arm: lowered to real `short{i}` blocks (#219) and then FOLDED into the drain
 chain or KEPT, per `ShortPathFold` (#183/#220-223). This header reports which happened.

 Without this the dump cannot be checked against the LoopIR at all: the LoopIR root is
 `Cond(T >= M, then=[prologue, steady, drain], els=[short])`, while the
 GIR hoists the prologue OUT of that guard and points the false edge at the drain chain -- so a
 reader diffing the two sees an `else` arm that vanished and a guard that changed meaning, with
    
```

## render.py:126 _pred_str

```
A terminator predicate WITH its scaffold hint.  `Pred.label` is attached downstream by
    ScaffoldMapPass and is the whole reason a consumer can route an arm to the right scaffold
    region -- a dump that omits it makes every terminator look label-free and unroutable, which is
    indistinguishable from ScaffoldMapPass not having run.

    READ THE LABEL AS A ROUTE, NOT AS AN IDENTIFICATION.  This docstring used to gloss `toPGR1` as
    "the peel-validity entry guard", which reads as though a `toPGR1` label identified that one
    
```

## render.py:99 comment run

```
    # BOTH, AND SPELLED APART.  These are two different questions and this used to be an `elif`
    # under one `tok=` prefix, so the semantic token was unreachable on any pipeline-built Program
    # (measured: 0 Moves lack `token` after TokensPass) and, had it ever fired, the reader could

```

## render.py:72 _dep_str

```
The ledger obligations this instruction discharges, WITH their hazard kind.

 `kind` is the one field that decides issue order (`is_war` -> the writer must be placed after
 its prior readers, line 99), and it was invisible here: the dump showed neither the edge
 nor its class, so a schedule that ordered a copy wrongly looked identical to one that ordered it
 correctly. The kinds are a CLOSED set (`LoopModel/ir.py` RAW_KINDS / WAR_KINDS, #276 S1); an
 unrecognised one is a defect, so it is printed loudly rather than abbreviated away.
```

## render.py:45 _covers_str

```
The ABSORBED axes: inner modes this ONE reference spans (#264/#267).

 `geometry.quantum_axes` REMOVES these from the read hop's the axes it varies over, so the def's coord does not
 name them while every consumer's coord does. A dump that shows only `coord` therefore shows a
 read that appears not to write what the wmma reads -- the exact appearance that sent the mxf8
 triage looking for a missing read ("16 consumers read a location the arm never writes") when the
 read was there and merely wide. Printing the span is what makes the subtraction legible.
```

## render.py:23 _gen_str

```
The buffer this Ref names: a loop-carried Gen IDENTITY, an absolute generation VALUE, or a
    register rotation slot.

    THE TWO GENERATION FORMS ARE SPELLED APART.  They used to render as `gen0[v]` and `gen=0` --
    one character between an SSA identity and a concrete integer, in the same dump.  They answer
    different questions (which phi feeds this vs which buffer this trip lands on), and a reader
    scanning for one silently accepts the other.
    
```

## render.py:3 <module>

```

render_gir -- a human-readable DEBUG VIEW of a GIR Program (not the IR itself).

This is a RENDERING, not a serialization -- no field here is authoritative, and nothing in the
compiler reads it back.

It is a debug view for humans, and that is the whole contract.  This header used to credit "the

```

## verify.py:356 comment run

```
            # The pointer name is hop-dependent (see _MARK_SCHEMA).  Requiring the RIGHT one, not
            # merely "some name", is what stops a copy swap from being labelled with a bare operand
            # -- which is how the fused movement's other members went unnoticed for so long.

```

## verify.py:203 comment run

```
    # V4 -- EVERY READ'S TOKEN MUST HAVE A PRODUCER IN ITS OWN BLOCK, WHEN NOTHING FENCES IT.
    #
    # In a ROLLED body the copy that fills generation `v+1` and the read that consumes it are the

```

## verify.py:192 comment run

```
    # ...and the STAMP the emitter actually reads must match what the analysis derived, over EVERY
    # shared Ref of the instruction.  Checking only the analysis would pass while a pass that
    # dropped or truncated `Move.token_ids` shipped an unnamed access -- the analysis is not what

```

## verify.py:168 _check_tokens

```
G-TOKEN (#235): the memory-token numbering is a correct naming of LDS storage.

    A token IS an LDS pseudo-register -- StinkyTofu gives a producer the token as a def, a consumer
    as a use, and a barrier as both -- so the numbering is not decoration: it is the alias relation
    the scheduler will believe.  Three properties, none assumed:

      V1  DISJOINT STORAGE NEVER SHARES AN ID.  True by construction (the key is
    
```

## verify.py:154 _check_walk

```
G-WALK (#236): the TDM region walk CLOSES in every block.

    A region-split tile is loaded one region at a time and the descriptor steps between them.  Those
    steps must net to zero per block, because `tdmIncrementGir` then applies the PLAIN chunk stride
    -- anything left over compounds every trip.

    This identity is the bug that reached hardware.  `globalReadDo` walked the regions internally
    
```

## verify.py:140 comment run

```
    # steady loops = blocks with a self/dominating back-edge.
    #
    # The TAIL loop (the `K % DepthU` remainder) is a SEPARATE loop region after the drain, with

```

## verify.py:112 comment run

```
    # exactly M drain blocks, M taken FROM THE LOWERING (`meta['M']`, the peel depth theta derives),
    # never approximated as PGR-1.  This half of G-CFG was documented but unchecked; a lowering
    # that dropped or duplicated a drain step would have passed.

```

## verify.py:3 <module>

```

verify_gir -- the GIR well-formedness verifier.

Checks the G-* invariants. Raises RuntimeError on the first violation with a precise
message; passing means the Program is a well-formed GEMM-dataflow CFG (R-LEGAL: entry is
legal, passes may create illegal intermediates but must restore verify_gir before the pipeline
ends).

```

## verify_dataflow.py:240 check_refill_splits_consumers

```
A HOISTED refill must not be placed BETWEEN two consumers of the generation it overwrites.

    THE ONE DEFECT SHAPE NO COORDINATE CHECK CAN SEE.  `check_register_dataflow` verifies that the
    register a wmma reads holds the `(tile_flat, k_flat)` the wmma asks for.  A read-ahead writes
    the NEXT GENERATION of the SAME coordinate, so when it lands between two consumers of the
    current generation every coordinate still matches and only the DATA is a trip early:
    
```

## verify_dataflow.py:208 check_address_keys

```
The ADDRESS-level check, which needs one geometry fact the plan does not carry.

    `check_source_coverage` proves the reads name distinct SOURCE coordinates.  That is not the
    same as distinct ADDRESSES.  The leaf builds an address from TWO coupled choices:

        region term :  `region * splitBoundary`  if the regions are PACKED, else 0
        coordinate  :  the WITHIN-REGION one     if PACKED, else the FLAT one
    
```

## verify_dataflow.py:164 check_source_coverage

```
One steady trip reads each `(operand, tile_flat, k_flat)` EXACTLY ONCE.

    The conservation law behind `test_every_six_axis_reorder_emits_and_lowers`, but per coordinate
    rather than per count: a reorder, a split or a peel moves work, it never adds or loses any.  A
    DUPLICATE means two reads fetch the same source (one of them redundant, and usually the sign
    that a coordinate was projected away); a MISSING one means part of the tile never reaches a
    register.  The expected set is the product of the indices the block's own wmma demand, so this
    needs no extent bookkeeping of its own.
```

## verify_dataflow.py:146 comment run

```
    # AN AGENT-RELATIVE REGION IS COVERED BY THE AGENTS, NOT BY THE COORDINATE (#245).
    # This check reads one agent's plan and asks whether its READ COORDINATES name every region the
    # copy fills.  That is the right question only while the region is a property of the COORDINATE.

```

## verify_dataflow.py:109 check_register_dataflow

```
USE-BEFORE-DEF / OVERWRITE-BEFORE-USE / WRONG-SOURCE over the pipelined shape.

    Walked as the hardware runs it: `entry` (the prefetch peel) primes the file, then `loop` runs
    TWICE -- the first pass exposes anything the peel failed to prime, the second exposes anything
    the loop fails to sustain for its own next trip (the loop-carried edge, which a single pass
    would silently accept because the first trip's own reads had already filled the file).

    Returns a list of human-readable violations; empty means clean.
```

## verify_dataflow.py:82 comment run

```
            # EVERY REGISTER SOURCE, READ OFF THE ACT (`srcs`) -- not reconstructed here.
            # `emit_plan._plan_mma` already DERIVES which operand each source is and which grid
            # index it is addressed by (`in0`/`in1` from `mma_inputs`, and a scale paired to its

```

## verify_dataflow.py:68 comment run

```
            # ONE ACT DEFINES EVERY REGISTER ITS CARRIER GROUP COVERS (#312).  A Phi-folded read is a
            # single instruction writing a RUN of registers, so the file must see all of them
            # defined here -- otherwise the consumer of a non-leader member reads a name nothing

```

## verify_dataflow.py:32 _read_fills

```
`((tile_flat, k_flat),...)` -- every SOURCE COORDINATE one read act delivers (#312).

 a folded read is ONE instruction at its carrier group's leader that fills the whole
 group, so the def is one completion event with several locations. `emit_plan` projected that
 span onto the same five indices the leader carries; this is the pair the register file and the
 coverage check both key on. For every ordinary read `fills` is a 1-tuple holding exactly the
 leader, so there is no folded-vs-not branch below -- and the `.get` fallback keeps a
 hand-built act (the tests construct plans directly) working with no `fills` key at all.
```

## verify_dataflow.py:3 <module>

```

Semantic verification of the EMIT PLAN -- "does this schedule compute the right GEMM?" -- checked on
the plan alone, with no rocisa, no assembler and no GPU.

`verify_gir` checks that the IR is WELL-FORMED (blocks reachable, terminators sane, Refs resolved).
This module checks that it is CORRECT: that every wmma reads the data its own coordinate names,
that every value a read fetches is used before it is overwritten, and that the reads of one steady

```

## gir_to_rocisa.py:457 _token_ids

```
The memory-token id(s) this act touches -- GIR's own numbering (#235).

        A token is an LDS PSEUDO-REGISTER: StinkyTofu gives a producer the id as a def, a consumer
        as a use, and a barrier as both, then the def-use chain orders them.  So the id set IS the
        alias relation, and GIR is the layer that knows it -- `MemTokenAssignment` names every
        buffer `(unit, region, generation)` and `TokensPass` stamps the ids on the Move.
        
```

## gir_to_rocisa.py:439 comment run

```
            # The movement IS split but this copy could not say which region it fills.  Emitting
            # anyway would issue a load against whatever region the walk happens to be on, so
            # refuse: `region_increment` skips these too, and a load with no walk behind it is

```

## gir_to_rocisa.py:424 _emit_copy

```
Realize ONE global->shared movement, for the buffer generation GIR's Move names.

        One call per Move -- so the prologue's M peel fills each emit, which a single pre-built
        module per operand could not do.  Everything else about the load (addressing, vector width,
        DTL/DTV variants) stays in the scaffold's `globalReadDo`, reached through the leaf.

        FUSED movement: emitted through the OWNING member's tP and nothing else, because the
        
```

## gir_to_rocisa.py:401 _emit_region_inc

```
Realize a descriptor walk of `steps` storage regions, via `tdmRegionIncrementGir` (#236).

        `steps` is SIGNED and may exceed one in magnitude; the primitive moves by exactly one
        region, so this issues `|steps|` of them.  Unrolling here rather than teaching the
        primitive a count keeps the backend's job "one region, one direction" -- the arithmetic is
        an add-with-carry over a 64-bit address pair, and a scaled version would need its own
        multiply and its own carry reasoning for a case (>2 regions) that no shipping config
        
```

## gir_to_rocisa.py:390 _wrap_lead

```
The shift that puts the wrap compare in the frame THIS advance runs in.

        `tdmIncrementGir` wraps when `LoopCounterL + lead == StaggerUIter`, and
        `StaggerUIter = S + pf` with `S` the stagger start chunk.  The wrap must fire on the advance
        that takes the address from chunk `T-1` to `T`, i.e. advance #n where `S + n = T`.  Under
        GIR's copy-then-advance convention, advance #n in the prologue targets chunk `n` and runs at
        `LoopCounterL = T`; the steady advance in trip `v` is #(M+v+1) and runs at `T - v`.
        
```

## gir_to_rocisa.py:383 comment run

```
    # The scaffold's `StaggerUIter += pf_const` (KernelWriterAssembly): 2 for PGR<3, PGR for
    # PGR>=3.  PGR>=3 is rejected under UseLoopModel, so only the first case is reachable; the
    # constant is named rather than inlined so lifting that rejection surfaces here.

```

## gir_to_rocisa.py:371 comment run

```
        # `prefetchIndex` shifts the StaggerU WRAP COMPARE: `tdmIncrementAB` adds it to the loop
        # counter before deciding whether this increment is the one that must wrap the staggered
        # address back to the tensor base.  A prefetched increment runs `prefetchIndex` iterations

```

## gir_to_rocisa.py:352 _emit_gr_inc

```
Realize ONE global-read increment for a Phi movement, via `writer.tdmIncrementGir`
        (s_add tdm+=inc, with the StaggerU WrapU cselect).  GIR OWNS the placement (a gr_increment
        Mark from the dataflow); the magnitude is L3's.

        `chunks` is how many reduction chunks GIR's dataflow says the address must advance here.
        `tdmIncrementGir` emits ONE DepthU stride, so anything other than 1 has no faithful
        realization -- emitting the single stride anyway would leave the address short and silently
        
```

## gir_to_rocisa.py:325 _emit_swap

```
Realize ONE buffer-pointer swap via the scaffold's primitive.

        read hop -> `localReadSwapOffsets` (v_xor LocalReadAddr), per OP-CLASS: A and B read from
        their own LDS regions through their own address registers, so `who` is an operand name and
        multi-wave changes nothing here.

        copy hop -> `tdmSwapLdsOffset` (s_xor the tdm descriptor's LDS address), per Phi MOVEMENT, so
        
```

## gir_to_rocisa.py:315 comment run

```
        # From `tpByOperand`, not `self._tp`: the fork may hand in a different map (MX/meta), and
        # reading the constructor's copy would ignore it for fused movements only -- a discrepancy
        # that would show up as one operand emitted against the wrong tensor parameters.

```

## gir_to_rocisa.py:292 _fused_tp

```
`(tP_owner, tP_peer)` for a Phi-fused movement, after checking the scaffold can realize it.

        The scaffold's fused form is ONE aliased descriptor: `tdm{tcA}Group0` serves both operands,
        each wave's copy of those SGPRs addressing its own, and wave parity selecting which
        (`KernelWriterAssembly.isTdmWaveSeparated`).

        THERE ARE TWO SUCH PAIRS ON A MICROSCALED KERNEL, not one.  `KernelWriter` calls
        
```

## gir_to_rocisa.py:277 _fence_tokens

```
The token ids this fence stands between -- carried on the Mark by `FenceRegions` (#235).

        A fence gets each of its tokens as BOTH a def and a use, which is what makes it a
        scheduling point between the producers and consumers of that storage.  So the set has to be
        exactly the storage it covers, and `FenceRegions` is the layer that knows: it built the
        fence from the hazards whose ends those accesses are.
        
```

## gir_to_rocisa.py:259 _register_fence

```
Record the `SBarrier` objects in `code` as GIR-owned.

        `postMainLoopBarrierCheckAndReset` still runs its STRIP half for UseLoopModel kernels -- the
        scaffold emits local-write syncs that TDM does not need, and removing them is what kept its
        barrier count low.  It must not remove ours, and ours are the same `SBarrier` class with no
        distinguishing field (rocisa nodes reject an added attribute), so identity is the only
        honest signal.  Registering here, at the single point that creates them, keeps the two ends
        of that contract adjacent.
```

## gir_to_rocisa.py:238 _emit_fence

```
Realize a proc-scoped selector -- a real, memory-ordering barrier.

 GIR decided WHERE (`FenceRegions`' cover) and WHAT IT ORDERS (`buffers`); L3 supplies the
 instruction. `_syncThreads` is the scaffold's own emitter and issues waitcnt + barrier, so
 it has the STRENGTH demands -- the read's completion ordered before the refill's write.
 The model is explicit that a bare execution-only barrier does not qualify.
        
```

## gir_to_rocisa.py:227 _tag

```
Stamp EVERY GIR-emitted act with a self-identifying comment in the emitted assembly.

        Two jobs, and the second is why this is not decoration:

        1. READABILITY -- the act's GEMM meaning at the point of emission (which tile, which
           K-substep, which register slot, which buffer generation), so the `.s` reads as the
           schedule rather than as an instruction soup.
        
```

## gir_to_rocisa.py:213 _append_tag

```
APPEND `tag` to every instruction comment in `code`, preserving what is already there.

        Appending, not replacing: the scaffold primitives put real information in their comments
        ("sync LDS0", "select WrapU or normal inc", "TDM addr += inc (with wrap, 64-bit)") and the
        backend appends its own ("<This is 20-cycle>").  Overwriting would trade one useful fact
        for another; the GIR act is ADDITIONAL context, so it goes on the end.
        
```

## gir_to_rocisa.py:179 comment run

```
                # L3 DOES NOT GAIN THE 2-D CAPABILITY (#251): GIR now carries a per-axis step
                # vector, and a step that moves SEVERAL region axes at once has no realization
                # here -- `tdmRegionIncrementGir` walks ONE region along ONE axis, and the flat

```

## gir_to_rocisa.py:155 comment run

```
                # #216: GIR chose the POINT; the instructions are the scaffold's own, unchanged.
                # Increment BEFORE issue -- the pair is ordered internally (the clamp zeroes
                # GL2PrefetchInc<tc> near the end of the loop, then the adds apply it), and that

```

## gir_to_rocisa.py:134 comment run

```
                # A MERGED MOVEMENT CARRIES EVERY TOKEN IT COVERS (Sec 2.2).  When the coverage
                # folds several acts into one instruction, that instruction really does read every
                # buffer those acts named, so it must be ordered against every one of their

```

## gir_to_rocisa.py:88 emit_block

```
Realize the finalized GIR block `phase` into a rocisa Module.

        A swap Mark is realized by calling the scaffold's swap PRIMITIVE keyed by the Mark's
        subject -- `writer.tdmSwapLdsOffset(kernel, tP)` for a copy-hop swap (s_xor tdm descriptor),
        `writer.localReadSwapOffsets(kernel, internalPointerSwap, tP)` for a read-hop swap (v_xor
        LocalReadAddr).  This is why one Mark -> one swap, with no doubling: the fork's pre-bundled
        A+B Module is NOT used; GIR's Marks map 1:1 to the primitive.  `tpByOperand` =
        
```

## gir_to_rocisa.py:76 comment run

```
        # MICROSCALING SCALE TENSORS are ordinary operands here: theta gives `MXSA`/`MXSB` their own
        # paths, so GIR emits read acts for them exactly as it does for A and B, and they
        # need exactly the same two entries.  The tP is the scaffold's own (`tP["MX"]`, built by

```

## gir_to_rocisa.py:27 _register_depth

```
`{operand: W}` -- the register ring width the leaves use for their `m = u % W` FALLBACK, a
    GIR fact (`Ref.reg_ring`, validated by RegBand, B3).

    The fallback only runs when GIR did not supply a per-act slot, which for a GIR-planned wmma or
    read it always does; W is therefore a legacy path, not the authority.

    PER OPERAND, because two different disagreements were being conflated.  RegBand keys widths by
    
```

## gir_to_rocisa.py:3 <module>

```

gir_to_rocisa -- the layer 2 -> 3 lowering, the THIN rocisa adapter.

L3 consumes the finalized GIR (after the pipeline applied swap/gr_increment Marks and stamped
tokens) as a flat EMIT PLAN (`gir.plan_block`, pure) and realizes each action into rocisa by
calling the per-tile leaf emitters (`Lowering/leaves.py`, `LeafEmitters`). The order + all
indices are GIR's; this file only maps action->instruction.

```

## lds_geometry.py:136 region_bytes

```
Byte displacement from the operand's LDS base to the start of TDMSplit `region`.

    THIS IS NON-ZERO EXACTLY ON THE PACKED DIAGONAL -- when the split cuts the LDS image's INNER
    axis (`region_split_is_packed`) -- and the asymmetry is a property of the image, not a
    heuristic.  Which axis is inner is layout-decided, so each layout packs on a DIFFERENT split:

      * unroll-major, LDS = [free][unroll].  An MT split cuts the OUTER axis, so region 1 is the
    
```

## lds_geometry.py:122 tiles_per_region

```
How many of this operand's wave tiles live in ONE TDMSplit region.

    ONLY AN MT SPLIT DIVIDES THE TILES.  The wave tiles are laid along the operand's FREE axis, so
    cutting that axis hands each region `MIWaveTile / nsplit` of them -- and `Solution.py` rejects
    the shapes where that does not divide, so the floor never discards a tile.  Cutting the
    REDUCTION axis instead leaves every tile present in every region (the regions differ in k, not
    in which tile), so the count is the full `MIWaveTile` and `tile_row`'s `t % tiles_per_region`
    correctly degenerates to `t`.
```

## lds_geometry.py:107 tile_row

```
The LDS ROW index of this operand's wave-tile `t`, in an unroll-major (DU-major) layout.

    THE DISTRIBUTION IS BY VECTOR GROUP, and the unroll-major path used to model only half of it.
    A wave takes `VectorWidth` ADJACENT tiles (adjacent rows), then the next group of tiles starts
    a whole `MIWaveGroupShape[tile01]` rows later -- that jump is what hands the intervening tiles
    to the other waves.  So
    
```

## lds_geometry.py:94 addr_coord_on_split_axis

```
Which coordinate the ADDRESS uses along the axis a TDMSplit cut: the WITHIN-REGION one when
    the regions are separately packed, the FLAT one when they are a contiguous continuation.

    ONE RULE, and the two halves are not interchangeable:

      * PACKED   region r is its own padded block, so the address is
                 `r * splitBoundary + within-region coordinate` -- `region_bytes` supplies the first
    
```

## lds_geometry.py:79 fold_inner_offset

```
`(region, innerInRow)` -- split an INNER-axis element offset into the region it lands in and
    its position inside that region.  The counterpart of `region_row_elems`: that shortens the row,
    this says which row-block a coordinate past the end belongs to.

    A packed split makes the address PIECEWISE.  The unsplit image is one linear map
    `f*extent + u`; the split image is `region*splitBoundary + (f*(extent/n) + u mod (extent/n))`,
    and there is no linear expression for the second form because the region jump is a padded byte
    
```

## lds_geometry.py:65 region_row_elems

```
Row length, in elements, along the LDS image's INNER axis -- of ONE REGION when the split
    packs regions separately, of the whole tile otherwise.

    `extent` is that inner axis's full length, and WHICH axis that is depends on the layout, so
    the caller supplies it: `MacroTile` on tile-major (`[unroll][free]`), `DepthU` on unroll-major
    (`[free][unroll]`).  Both callers then get the same rule -- a packed region divides the row it
    lives in -- instead of the tile-major one being special-cased.
    
```

## lds_geometry.py:53 region_split_is_packed

```
Does TDMSplit make each region a SEPARATELY PACKED LDS block?

    Delegates to `tdm_split.split_packs_lds`, which states the rule once for the descriptor side
    and this side together.  The answer is a DIAGONAL over (axis, layout) -- a partition packs
    blocks exactly when it cuts the LDS image's INNER axis -- not a property of either alone:

        unroll-major  [free][unroll] : MT contiguous, DU PACKED
    
```

## lds_geometry.py:26 read_fragments

```
The ds_readS ONE (tile, reduction-substep) local-read act decomposes into, as
    `((unrollElems, regOff), ...)` -- both relative to the act's own base, the first in ELEMENTS
    along the unroll axis, the second in registers.

    A lane's `inputPerThUnroll` elements for one substep are neither one contiguous run nor one
    instruction, and TWO nestings produce them:
    
```

## lds_geometry.py:19 comment run

```
#: How many lanes share one range of K positions, so a lane's consecutive local-read chunks sit
#: this many chunk-widths apart along the unroll axis.  The scaffold spells it as a bare `* 2` --
#: "the WMMA V3 LDS layout uses a *2 factor on the unroll stride"

```

## lds_geometry.py:3 <module>

```

LDS tile geometry -- the PURE arithmetic behind an operand's per-tile LDS address.

Split out of `leaves.py` for the same reason `emit_plan.py` is split out of `gir_to_rocisa.py`:
`leaves` imports rocisa at module scope, so anything living there cannot be unit-tested in the
pure-Python environment, and the one prior test of this arithmetic had to match SOURCE TEXT
instead of calling it.  That test then broke the moment the (wrong) expression was corrected,

```

## lds_region_check.py:221 check

```
Run the check over one emitted kernel.

    `region_bytes` is `tdmSplitLdsBoundary` for the operand -- the SAME displacement the copy side
    advances by.  Returns (accesses, violations, notes).

    THE SEED IS THE HARDWARE REGISTER `v0`, not the symbolic `vgprSerial`.  The kernel's own first
    instruction is `v_mov_b32 v[vgprSerial], v0`, so seeding `vgprSerial` would OVERWRITE the value
    
```

## lds_region_check.py:196 comment run

```
# ---------------------------------------------------------------------------------------------
# the check
# ---------------------------------------------------------------------------------------------

```

## lds_region_check.py:130 _operand_width

```
How many consecutive registers the operand `v[a:b]` covers (1 when it is not a range).

    A DEF OF A RANGE DEFINES EVERY REGISTER IN IT, and forgetting that is not a small imprecision.
    `s_load_b512 s[36:51], ...` was recorded as defining only `s36`, so a later use of `s38` found
    no producer; the slicer then kept walking backwards past the real def and dragged in unrelated
    scalar setup, which is how a purely lane-local LDS address appeared to depend on kernarg-loaded
    problem sizes.  The slice was not merely long, it was wrong about what the address depends on.
```

## lds_region_check.py:109 comment run

```
# INSTRUCTIONS WHOSE FIRST OPERAND IS A SOURCE, NOT A DESTINATION.  "First operand is the dst" is
# right for VALU/SALU and for loads, and WRONG for stores and prefetches: `global_prefetch_b8 v0,
# s[0:1]` reads its address out of `v0`.  Getting that backwards makes the slicer believe `v0` was

```

## lds_region_check.py:50 comment run

```
# ---------------------------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------------------------

```

## lds_region_check.py:19 comment run

```
# double-buffer selector: LDS reads toggle this bit to swap generations, and it is NOT part of the
# region coordinate.  Masked off before the region division, or every swap would read as a region
# change.

```

## lds_region_check.py:3 <module>

```

LdsRegionCheck (#245) -- does one `ds_load` touch more than ONE storage region of a split tile?

THE HAZARD.  A `TDMSplit` tile lives in LDS as `nsplit` storage-disjoint regions, and GIR names the
region each read touches so that `MemTokenAssignment` can give it a token and `FenceRegions` can
order it against the copy that filled THAT region.  All of that assumes one read instruction sits
inside one region.  The assumption is not free: the region a lane reads is decided by the lane's

```

## leaves.py:588 emitCopyTile

```
Emit the global->shared copy (tensor_load / buffer_load) for ONE operand, into staging
        buffer `bufIdx`.

        The third leaf, alongside `emitLdsReadTile` and `emitWmmaTile`, and the one that was
        missing: copies used to be handed to L3 as a PRE-BUILT Module per operand
        (`copyByTc = {A: codes.globalReadA, ...}`) and emitted once per block.  That is why the
        prologue and drain could not be GIR-owned -- their peels issue SEVERAL copies of one operand
        
```

## leaves.py:569 comment run

```
        # ONE ds_read PER FRAGMENT.  The list is dtype-derived (`buildLdsReadContext`), so this
        # loop is the same for bf16's two reads and fp8's four; `unrollElems` is a displacement
        # along the unroll axis, which is `UnrollStride` elements per position.

```

## leaves.py:556 comment run

```
        # ROW INDEX, not `stride * t`: the tiles of one wave are a vector group of `VW` adjacent
        # rows and then a jump of `MIWaveGroupShape` (see `tile_row`, #237).
        # WHICH K COORDINATE THE ADDRESS USES is the packed/contiguous rule, same as the tile

```

## leaves.py:551 comment run

```
        # WHETHER THE READ OWES A REGION TERM IS DECIDED BY THE LAYOUT, and the rule lives in
        # `lds_geometry.region_bytes` -- one statement, read by both the base and the row index.
        #

```

## leaves.py:532 comment run

```
            # MANY ACTS, ONE INSTRUCTION -- the group-leader rule.  When one ds_read covers
            # `tilePerRead` tiles (the MX scales of adjacent tiles are contiguous in LDS), the act
            # for the FIRST tile of each group issues the load that fills all of them and the rest

```

## leaves.py:524 comment run

```
        # MANY ACTS, ONE INSTRUCTION -- the group-leader rule.  When one ds_read covers
        # `tilePerRead` tiles (the MX scales of adjacent tiles are contiguous in LDS), the act for
        # the FIRST tile of each group issues the load that fills all of them and the rest issue

```

## leaves.py:510 emitLdsReadTile

```
Emit the ds_read(s) for ONE operand M/N-tile `tileIdx` at K-substep `kIdx`.

        `memToken` -- the LDS memory-token id(s) this read CONSUMES, from GIR (#235).  Passed down
        verbatim; `None` leaves the scaffold's own derivation in place.

        Why GIR supplies it rather than the component deriving it: the component's split-half
        classifier keys on the read's BYTE OFFSET
        
```

## leaves.py:488 _checkRegBuffer

```
INVARIANT GUARD (#122) -- not a diagnostic anything is expected to trip.

        Read honestly: on the UseLoopModel path this CANNOT fire.  GIR names the slot
        `rate_index mod W_g`, so `bufferIdx < W_g <= max_g W_g <= numVgprBuffer` once
        `KernelWriter.loopModelRegBuffers` has grown the allocation to theta's rotation width.  The
        assertion is that relationship restated at the point of use.
        
```

## leaves.py:461 comment run

```
            # A PACKED REGION SHORTENS THE UNROLL ROW HERE, exactly as it shortens the free row in
            # `UnrollStride` on the tile-major path -- same rule, different axis, because "the row"
            # is whichever axis is INNER in the image.  LDS is `[free][unroll]`, so a DU split

```

## leaves.py:426 comment run

```
        # THE SAME ROW LENGTH THE BASE REGISTER USES.  When the split packs regions separately the
        # unroll row shortens to `MacroTile / nsplit`, and `LraTileAssignment` must apply the
        # identical shortening to its `kOffset * strideUnroll` -- the immediates and the base are

```

## leaves.py:419 comment run

```
        # THE ONE READER of `TDMSplitA`/`TDMSplitB`, so the read side cannot disagree with the
        # descriptor about which axis this operand is cut on.  It also absorbs the MX/sparse/
        # metadata exclusion that used to be re-spelled here.

```

## leaves.py:339 comment run

```
        # `mxTc` is the PARENT's letter: the scale tensor named `MXSA` scales `A`, so its block
        # size is `MXBlockA`.  Taken off the tensor char rather than passed in, so a caller cannot
        # pair a scale tensor with the wrong parent's block size.

```

## leaves.py:321 buildMxScaleReadContext

```
Loop-order-invariant LDS-read constants for a MICROSCALING SCALE tensor (`MXSA`/`MXSB`).

        A SEPARATE BUILDER, not a branch in `buildLdsReadContext`, because the scale tensor's LDS
        image is genuinely a different geometry rather than a narrower version of the same one --
        the scaffold gives it its own emitter (`Components/LocalRead.py:localReadMX`), and A/B's
        `UnrollStride`/`tileStride` swap roles there.  Every validation gate in the A/B builder
        (LDS-transpose vs unroll-major, the `bpeDS` pair, the fragment table) is about that other
        
```

## leaves.py:289 comment run

```
        # ONE-SIDED MX gets the scaffold's dummy register on the other side.  Reject instead: the
        # dummy's width depends on the PRESENT side's block size and nothing in scope produces a
        # one-sided kernel, so an untested spelling here would be a silent wrong operand.

```

## leaves.py:272 comment run

```
        # ---- MICROSCALED: the same opcode, plus the two scale registers and a block modifier ---
        #
        # THE SCALE SOURCE IS BUILT BY THE SCAFFOLD'S OWN HELPER, `generateSrcStrForMFMA` -- the

```

## leaves.py:227 comment run

```
        # register generation m = u % W, W = the GIR register ring width (reg_depth).  PLR0 has
        # W=1 -> m=0 for every substep (single register generation); PLR1 W=2 splits X0/X1.
        # PER OPERAND: the fallback for a scale is its OWN ring, not its parent's.

```

## leaves.py:218 emitWmmaTile

```
Emit the SINGLE WMMA instruction for tile (idx0, idx1) at kiter substep `u`.

        The per-operand register buffer generation (`bufA`/`bufB`, and `bufMXA`/`bufMXB` for the
        microscaling scales) is the CONSUME-side rotation slot GIR/LoopModel assigned to the wmma
        source (`_wmma_src_residence`) -- the SAME authority the producing ds_read used, so def and
        use name the same vgpr under any loop order.  When not supplied (older call sites), fall
        back to `m = u % W`.
        
```

## leaves.py:179 comment run

```
        # THE OPCODE COMES FROM THE SHARED TABLE, not from a literal.  `dataTypeToMfmaInstTypePair`
        # is the scaffold's own derivation (hoisted out of `KernelWriterAssembly.mfmaIter`), so a
        # ULM1 kernel and its ULM0 twin cannot disagree about which `v_wmma_*` to issue -- the

```

## leaves.py:79 comment run

```
    # Which layout `tileStrideElems` is expressed in.  True (unroll-major): it is the ROW stride
    # (DU + pad) and the tile index must be turned into a row index first.  False (transpose path):
    # it is already `MIWaveGroupShape`, i.e. a per-tile stride, and must NOT be re-multiplied.

```

## leaves.py:68 comment run

```
    # BYTES BETWEEN CONSECUTIVE REDUCTION SUBSTEPS of one tile.  A substep consumes `MatrixInstK`
    # unroll positions, and one position is `UnrollStride` elements apart, so this is the whole
    # dtype/layout dependence of the act's base address.

```

## leaves.py:43 comment run

```
    # MICROSCALING.  `mxBlock` is 0 when the kernel has no scales, which is the ONE test
    # `emitWmmaTile` makes -- an MX kernel issues `MXMFMAInstruction` (the same opcode plus real
    # scale operands and a block modifier), a plain one issues `MFMAInstruction`.

```

## leaves.py:3 <module>

```

Leaf emitters -- the L3 per-tile realization of a GIR Move/Mma.

These are the order-AGNOSTIC per-coordinate emitters: the rocisa for ONE (idx0,idx1,u) WMMA tile
or ONE (tile,k) LDS read is identical regardless of loop order; only WHEN it is emitted (the GIR
plan) changes. They live here -- in the Lowering (L3) layer -- because they ARE the layer-2->3
realization; `gir_to_rocisa.GirToRocisa` owns a `LeafEmitters` and drives it from the GIR plan.

```

## loopir_to_gir.py:810 build_gir

```
R-ONCE: build the theta model, lower to GIR, and run the pass pipeline EXACTLY ONCE, returning
    the finalized Program the phase forks emit from.

    KernelWriter calls this a single time per kernel and caches the result on `self.states`; the
    three stage forks (prologue/steady/drain) then only ask the emitter for their block, instead
    of each rebuilding the LoopIR (the coupling this whole rearchitecture removes).  Kept pure /
    rocisa-free so it is testable standalone; `params` (presets like StaggerU) ride on
    
```

## loopir_to_gir.py:788 gir_text

```
Render an ALREADY-BUILT GIR Program as text for the `OutputLoopIR` dump.

    Takes the Program, never a kernel: KernelWriter builds the GIR exactly ONCE per kernel
    (R-ONCE) and the three phase forks emit from that one object, so the dump must render THAT
    object.  Rebuilding here would both redo the work and risk showing a Program that differs
    from the one that produced the .s -- a debug artifact that lies is worse than none.
    
```

## loopir_to_gir.py:773 comment run

```
    # --- short{0..M-1} -- the LoopIR's `els` arm, as REAL blocks -----------------------------
    # The `T < M` degenerate path: no steady region, so each chunk is copied and consumed in its
    # own step (`Bind(iter = t)`, a CONCRETE chunk, hence absolute generations and `gen_rel=None`,

```

## loopir_to_gir.py:765 comment run

```
    # --- steady --------------------------------------------------------------------------
    # ONE source loop, `n_copies` FALLTHROUGH-CHAINED blocks sharing ONE back-edge (G2).  A body
    # replicated `n` times consumes `n` reduction chunks per trip, which is what makes each copy's

```

## loopir_to_gir.py:757 comment run

```
    # the peel-validity guard's FALSE arm.  With the short arm lowered it is that arm's first block
    # (the LoopIR shape); with no short arm it falls straight into the drain (the folded shape,
    # which is also what `FoldShortPathPass` restores when it proves the coverage legal).

```

## loopir_to_gir.py:747 comment run

```
    # the entry names a block that ACTUALLY EXISTS.  A one-hop path (direct-to-register,
    #) stages nothing through shared, so there is no prefetch to peel and NO prologue block;
    # naming "prologue" anyway left the CFG rooted at a missing block.  With a single self-looping

```

## loopir_to_gir.py:738 comment run

```
    # PRESENCE-DERIVED addressing : L3 reads a leaf's free-tile /
    # reduction index by PROJECTING the coord onto per-operand free modes / the reduction modes --
    # handles multi-mode roles (M_split+M_inner both free) and N operands.  Carry:

```

## loopir_to_gir.py:717 lower_to_gir

```
Build the GIR `Program` from a theta point.  `mainloop` = a prior `emit_mainloop(theta)`
    result (built once by KernelWriter, R-ONCE); if omitted it is computed here.

    `loop_copies` (G2) replicates the steady body into that many fallthrough-chained blocks sharing
    ONE back-edge.  It is an EMISSION choice, not a theta field -- theta fixes what the loop computes, not
    how many copies of it the backend lays down -- so it arrives as an argument rather than being
    derived here.  1 (the default) is the ordinary single-block loop.
```

## loopir_to_gir.py:710 comment run

```
        # frame: the short arm's chunks are concrete (0..M-1) and there is no steady trip, so its
        # Refs carry absolute generations (gen_rel=None) and it sits at the head of the chunk
        # timeline -- step i IS chunk i.

```

## loopir_to_gir.py:678 comment run

```
            # frame: `_flatten` gave drain step i the offset `rel = i - M` (its Bind is
            # `iter = T - M + i`, so its gdeltas are relative to T).  Its position on the chunk
            # timeline is a DIFFERENT number: the drain runs after the last steady copy, so step i

```

## loopir_to_gir.py:599 comment run

```
        # peel-validity edge (T >= M -> enter steady, else short/drain).  The scaffold LABEL is set
        # DOWNSTREAM by ScaffoldMapPass from the CFG shape -- the lowering emits a label-free Pred so
        # no TensileLite name is hardcoded here (Step 3: scaffold knowledge lives in one pass).

```

## loopir_to_gir.py:569 comment run

```
                         # the `T < M` short-loop arm.  Lowered to REAL blocks (below) in the
                         # LoopIR's own two-arm shape; `FoldShortPathPass` then folds it into the
                         # scaffold's shared prologue/drain shape if and only if it proves that

```

## loopir_to_gir.py:564 comment run

```
                         # rho, the ONE bit the width rule needs: is this operand's
                         # movement spread over SEVERAL agents?  A hazard on a buffer whose
                         # producer and consumer are the same agent is discharged by program

```

## loopir_to_gir.py:559 comment run

```
                         # per OP-CLASS: how many storage regions ITS OWN tile occupies (theta's
                         # `op.split`).  Distinct from both neighbours above and needed because
                         # neither answers the READ ADDRESS's question:

```

## loopir_to_gir.py:550 comment run

```
                         # THE ABSORBED AXES, per operand: `((axis, extent),...)` for each
                         # inner mode the read's ONE instruction SPANS, so `geometry` has already
                         # removed it from that hop's the axes it varies over.  This is the OTHER HALF of the

```

## loopir_to_gir.py:543 comment run

```
                         # THE PARTIAL FOLD, per operand: `{axis: factor}` for each axis this
                         # read's coverage folds only IN PART (`1 < q < N`), so the axis stays in
                         # the axes it varies over carrying the GROUP INDEX and theta issues one act per group via the

```

## loopir_to_gir.py:536 comment run

```
                         # THE MOVEMENT QUANTUM, per operand: the `ir.QuantumMap` theta carries
                         # on the shared->register hop, or absent for the identity merge.  Carried
                         # into GIR because L3 must OBEY the merge rather than invent one:

```

## loopir_to_gir.py:526 comment run

```
                         # Phi : movement instances merged into one cooperative
                         # instruction. A fused group is ONE completion event, so
                         # TokensPass names the GROUP, not a member.

```

## loopir_to_gir.py:450 comment run

```
# ===========================================================================
# the lowering
# ===========================================================================

```

## loopir_to_gir.py:437 comment run

```
        # THE SOURCE NAMES THE OPERAND'S OWN COORDINATE, NOT THE WMMA'S.
        #
        # `coord` is the wmma's full (K,M,N) position.  A is broadcast over the N axes and B over

```

## loopir_to_gir.py:420 _convert_mma

```
A LoopIR Mma -> a GIR Mma. srcs = one register Ref per read operand (+ scales);
 dst = the accumulator. Operand identity is on each Ref.tile.operand.

 Each source Ref's register residence is LOWERED from the wmma's per-operand LoopIR
 `Placement` (`inst.placement[op.name]`, attached by `decoder.wmma_inst`) -- the CONSUME-side
 rotation the tile coord determines -- NOT recomputed in GIR. This is the same
 `_read_placement` the producing ds_read dest lowers from, so def and use name one vgpr.
```

## loopir_to_gir.py:412 comment run

```
        # CARRY THE READ-AHEAD SHAPE (#332).  `advance` is the only thing that distinguishes a
        # REFILL from an ordinary read, and without it no GIR/plan-level check can see a
        # hoisted read emitted before the wmma whose slot it overwrites.

```

## loopir_to_gir.py:329 _quantum_coords

```
The coordinates ONE transfer fills: `coord` alone, or its product with the coverage axes.

 `coverage` is `((axis, n),...)` from `ir.Load` -- the axes this single instruction spans,
 in the two shapes `_absorbed` documents, told apart HERE by whether `coord` already names the
 axis. No marker field: the coord IS the discriminator, because the the axes it varies over subtraction that
 produced it is the same fact.
    
```

## loopir_to_gir.py:303 _absorbed

```
`((axis, extent),...)` the operand's read instruction SPANS -- the absorbed axes.

 THE OTHER HALF OF THE PRESENCE SUBTRACTION. `geometry.quantum_axes` took these axes out of the
 read hop's the axes it varies over, so the emitted `Load.coord` no longer names them and the read is ONE act
 per instruction (which is what the leaf emits). The CONSUMERS still name every value -- the
 wmma is indexed on the output's the axes it varies over, which keeps the axis -- so the def has to be readable
 as filling all of them. `refs.covered_coords` does that re-insertion from this tuple.
    
```

## loopir_to_gir.py:261 _reg_residence

```
`(group index, concrete slot, rotation width W)` for `op`'s register fragment AT `coord`.

 WHICH GROUP HOLDS THIS COORDINATE -- the piece #118 was missing. A register partition
 (`VgprPartition > 1`) splits the fragment's in-region fan into groups that partition that
 mode's VALUES : at `MIWaveTile[4,4]` with `VgprPartitionA=2`, `grouping_mode` is
 `M_inner` (extent 4) and `lo` owns `M_inner in {0,1}`, `hi` owns `{2,3}`. `_read_placement`
 already emits one `(label, slot Expr)` per group and the groups carry DIFFERENT widths (lo
    
```

## loopir_to_gir.py:223 _member_region_coord

```
Re-key a FUSED instance's region coordinate onto THIS member's OWN region axis.

 move pairs a Phi group's members BY INDEX, not by a shared mode: region j of A moves with
 region j of B, and each names the axis it is split on -- A's `M_split`, B's `N_split`. The
 LoopIR instance carries ONE coord, necessarily naming one member's axis (its representative's),
 so handing that coord to every member leaves the others' region axis UNPINNED.
    
```

## loopir_to_gir.py:219 comment run

```
# ===========================================================================
# Inst -> GIR node conversion
# ===========================================================================

```

## loopir_to_gir.py:191 comment run

```
            # honor a RESOLVED guard in the current env: a drain NLL guard `Pred(k < n_s-dr)` (the
            # reduction Loop has bound k) really DROPS the out-of-bounds read-ahead; an UNRESOLVED
            # guard (peel-validity `T >= M`, T runtime) keeps the steady (then) arm as the rep path.

```

## loopir_to_gir.py:167 comment run

```
                # inner static loop: unrolled.  Each sub-body runs over its OWN half-open range
                # with the read and
                # [1,trip) without).  Body-major, which IS program order for disjoint ranges.

```

## loopir_to_gir.py:153 _flatten

```
Mirror render_unrolled's traversal: unroll int-extent Loops; a symbolic-trip Loop (the
    steady reduction-chunk loop) is walked once at the current env; bind Branch residues (pinned
    or fanned); descend Peel/Cond bodies; yield each Inst with its concrete env.  Appends
    `(inst, dict(env), rel)` to `out` in program order.

    `rel` is the instance's reduction-chunk index RELATIVE TO THE STEADY LOOP, or None when the
    chunk is bound to a concrete absolute value.  It is the one fact the generation lowering needs
    
```

## loopir_to_gir.py:149 comment run

```
# ===========================================================================
# stage flattening -- one concrete (kind, Load/Mma, env) stream per stage body
# ===========================================================================

```

## loopir_to_gir.py:128 _deps_of

```
Carry the LoopIR awaits (RAW/WAR, per /dep_defuse) onto a GIR verb as `deps`.
 Each dep is a plain hashable record (dep_name, counter, kind, scope) -- a semantic edge, no
 count (the numeric residual is L4's; R-SEMANTIC). This is the ONE place LoopIR awaits enter
 GIR, and therefore the one place a field can be dropped on the way in.

 `scope` (#276-S2) is the proc-scope of the obligation: "wave" for an edge program order
 plus a per-wave counter already covers, "block" for a cross-agent one that a per-wave counter
    
```

## loopir_to_gir.py:114 _expr_raw

```
Evaluate an Expr WITHOUT applying its own modulus -- the raw generation offset. Used to
 derive a steady ref's `gdelta` (the constant read-ahead / prefetch offset on top of the
 loop-carried `iter` base), so gen_reaching's `(entry + gdelta) % ring` reproduces the
 concrete generation.

 THE MODULUS IS THE ONLY DIFFERENCE, so everything else defers to `ir.Expr.eval` rather than
 re-implementing it. This function used to carry its own copy of the linear/carry arithmetic
    
```

## loopir_to_gir.py:96 _gir_pred

```
LoopIR structured `Pred` -> GIR structured `Pred`, FIELD BY FIELD (no string round-trip).

    The LoopIR and GIR predicates are deliberately SEPARATE types: GIR imports nothing from
    LoopModel (it is the standalone optimization surface), so the boundary is a translation, not a
    shared class.  The translation is total over the shapes a terminator can carry and RAISES on
    anything else -- the old code rendered to a string, which could not fail and could not be read
    back.
```

## loopir_to_gir.py:79 _trips_of

```
The LoopIR outer `Loop`'s RANGE, read as a trip COUNT (#231).

    LoopModel states the steady region as a range -- `[delta, T)`, i.e. `T - M` iterations -- and carries
    it as `Loop.trip = iter < T - M`.  That `<` is the range's upper bound, not a machine
    comparison: read the bound and the count IS the bound.

    This replaces `_post_test_bound`, which existed only because the lowering used to store the
    
```

## loopir_to_gir.py:47 _short_steps

```
The short-loop (`Cond.els`) arm, split into its per-step (guard, nodes) -- or a raise.

    THE TWO SHAPES (#183).  TensileLite's scaffold and the LoopIR describe the same kernel with
    DIFFERENT STRUCTURE:

        scaffold :  prologue -> Cond -> loop -> else -> drain     prologue+drain SHARED by both
                                                                   paths; `T < M` just skips the loop
    
```

## loopir_to_gir.py:30 comment run

```
# ===========================================================================
# small Expr helpers
# ===========================================================================

```

## loopir_to_gir.py:3 <module>

```

loopir_to_gir -- the layer 1->2 lowering.

Walks the LoopModel LoopIR (the rolled theta-nest `build_ir` emits) EXACTLY ONCE and unrolls the
three software-pipeline stages (prologue / steady / drain{n}) into one minimal GIR CFG. It is
the *tree-walk half* of the old `LoopModelLowering._walk` with NONE of the physical-state
reconstruction (no pointer seeding, no swap heuristics, no token stepping -- those become GIR

```

## region_derivation_check.py:3 <module>

```

RegionDerivationCheck (#245) -- is a wave tile's TDMSplit region the same whether you derive it from
the TILE INDEX or from the tile's actual FREE-AXIS POSITION?

WHY THE TWO CAN DISAGREE.  `lds_geometry` already holds both derivations, for different callers:

    tiles_per_region(ctx)   MIWaveTile / nsplit      -- "region = t // tiles_per_region"

```

## uniform_emission.py:119 check_uniform_placement

```
Verify no barrier in `ownedBarriers` was emitted inside an agent-discriminating skip region.

    `ownedBarriers` is matched by OBJECT IDENTITY (the same currency `_stripBarriers(keep=...)` uses,
    and for the same reason: rocisa nodes reject an added attribute, so identity is the only honest
    signal that a barrier is ours).  Returns the findings; raises if there are any.

    Totals land in `_STATS` so a sweep can show the check ran over real structure rather than
    over nothing (`TENSILE_UNIFORM_EMISSION_REPORT=1` prints them).
```

## uniform_emission.py:59 _branch_target

```
The label a conditional branch jumps to, or None if `item` is not a conditional branch.

    Matched on the CLASS NAME rather than by importing every branch type, because the set grows
    (`SCBranchExecZ`, `SCBranchSCC1`, ...) and a missing import would silently make this check skip a
    region -- the one failure mode a safety check must not have.

    A SOUNDNESS LIMIT, recorded where the matching happens rather than as a footnote: the long-form
    
```

## uniform_emission.py:30 comment run

```
# Registers that NAME AN AGENT.  `sgprWaveIdx` is the wave index within the workgroup (what the
# TDM wave-separation branches test); `vgprSerial` is the flat thread id, so anything compared
# against it discriminates lanes.

```

## uniform_emission.py:19 comment run

```
# Sweep instrumentation: a check that never fires has to be able to SHOW it ran over real
# structure.  `TENSILE_UNIFORM_EMISSION_REPORT=1` prints the totals at exit; off, this is three
# integer increments per kernel.

```

## uniform_emission.py:3 <module>

```

The EMISSION half of the L733 uniform-placement check -- where the agent-discriminating arms
actually are.

L733 asks a *tree-structural* question of the IR: does a `scope=block` selector's node lie where no
agent-discriminating arm can make an agent skip it? `gir/analyses/uniform_placement.py` asks it of
GIR and gets a truthful but WEAK answer, because GIR's region tree has no node for the arms this

```

# The θ schedule-model decoder — implementation guide

How `Tensile/LoopModel/` actually works, module by module and derivation by derivation.

This is the companion to `Tensile/Lowering/docs/GIR_IMPLEMENTATION_GUIDE.md` (the layer below; and
`Tensile/Lowering/docs/lowering-design.md` is GIR's design rationale). Together they describe the whole `UseLoopModel` path. Written against the source, not
from memory; every claim below has a file:line behind it.

The reference is the paper `theta-schedule-model.md` ("Spacetime layouts"). Section numbers
(§2.4, §5.3.1, Lemma 3b …) are the paper's throughout.

The **code** no longer uses the paper's symbols: `ρ` is `AgentAssignment`, `Φ` is
`instances_per_instruction`, a movement quantum is a `TransferCoverage`. This guide still writes
the symbols where it is arguing from the paper. `ARCHITECTURE.md` has the one old-name →
direct-name glossary; read it before grepping for a name you saw here.

---

## 1. What the decoder is, and what it is not

θ is a **schedule point**: a tuple `θ = (tile, path, ord, off, S, ρ, Φ)` plus a discharge policy
`π`. The decoder is the function

```
θ  ──emit_mainloop──▶  (LoopIR region tree, obligation ledger, ledger_empty)
```

`ledger_empty` is the whole value proposition: if the decode terminates with an empty ledger,
every hazard the model knows about is discharged by construction.

**How much of that needs θ, exactly?** Less than it looks, and the split is worth stating because we
exploit it elsewhere:

- The **checking** half is already θ-free. `check_ledger_discharged(ledger, tree)` walks the tree and
  never consults θ. One layer down, `Lowering/gir/analyses/lds_hazards.py` and `fence_regions.py`
  contain **zero** references to θ — they derive RAW/WAR/WAW and place fences from the CFG and the
  generation SSA alone. That is a ledger-equivalent computed on an IR without θ, and it is what let
  GIR take barrier placement off the scaffold (#204/#205).
- The **ordering** facts are recoverable from an IR that makes the loop-carried relation visible.
  GIR does (explicit `Gen` SSA with phis); LoopIR hides it inside slot arithmetic, which is why the
  θ-free analysis lives at GIR.
- What genuinely needs θ is the part of the ledger that is **not a conflict between operations
  present, but a requirement that an operation be present at all** — the warm-up pre-issues. No
  analysis of the code in front of you can find a read that isn't there.

Theorem 6(b) says the narrower thing: it declines to give a characterization of which arbitrary
streams are even *in* the space the model describes. So the certificate covers trees this code
produced; the *machinery* is reusable, and is reused.

**Scope boundaries, deliberately enforced:**

- **No addresses.** The core is address-opaque (§2.1/§4.3). There is no swizzle, no stride
  algebra, no register colour. `latalg.py` exists precisely to keep it that way — it replaces the
  three PyCuTe calls `geometry.py` once made with plain integer arithmetic.
- **No TensileLite vocabulary in the core.** `translate.py` is the **only** exemption (#163). No
  other core module may mention `DepthU`, `MIWaveTile`, `PGR`, a wave, or an SGPR. The core does
  not know what a "wave" is; `Theta.rho` values are opaque tokens compared only for equality.
- **No wait counts in LoopIR.** Every `Await` is emitted with `count = -1` (unset). GIR owns the
  per-frame hazard/wait/fence relation. Its storage identity is `(class, region, ring, gdelta)` and
  resolves in a frame as `token = (frame + gdelta) % ring` (an absolute peel generation bypasses
  the frame). Numeric waits are derived from that relation after placement; dependency-token
  unions are retained only as a temporary backend-compatibility field, not as the authority.

---

## 2. Layer map

```
TensileLite Solution dict
        │  bridge.kernel_to_params        (Solution → the decoder's param dict)
        ▼
   param dict
        │  adapter.params_to_theta (params → θ; the ONLY vocabulary bridge)
        ▼
        θ  ── schedule.build_S ──▶ BufferDepths
        │  emit.emit_mainloop             (= emit_can, §5.3 steps 8/9a/9b)
        ▼
   LoopIR region tree  +  ledger          (ir.py node set)
        │  Lowering/loopir_to_gir.build_gir
        ▼
   GIR  (CFG + SSA over buffer generations)
        │  Lowering/gir_to_rocisa
        ▼
   rocisa modules → KernelWriterAssembly
```

### Module inventory (`Tensile/LoopModel/`, ~5.4k lines)

| module | lines | owns |
|---|---:|---|
| `theta.py` | 793 | the θ data model; `build_S` / `derive_S` (§2.6 R/L/W); `_shared_depth` |
| `emit.py` | 857 | `build_ir` / `emit_mainloop` / `loop_shape` — emit_can steps 8, 9a, 9b |
| `placement.py` | 635 | presence, rate slots, `_peel_depths`, read-ahead (`dr_g`, prologue, shift), `chunk_reach`, copy-first |
| `translate.py` | 570 | params → θ; the canonical 6-mode vocabulary and LoopOrder word |
| `ir.py` | 556 | the LoopIR node set (`Inst`/`Loop`/`Peel`/`Bind`/`Cond`/`Branch`, `Expr`, `Pred`, `Await`) |
| `ledger.py` | 510 | `build_ledger`, `discharge_once`, `check_ledger_discharged` (§4, §5.5) |
| `render.py` | 579 | rolled + unrolled + raw text dumps AND `render_geometry` — the derived sizing table (the larger half, and the only view of θ's own numbers) |
| `validate.py` | 282 | the P-* structural gate over the emitted tree |
| `sizing.py` | 183 | register/byte footprints (`frag_regs`, `operand_footprint_regs`) |
| `bridge.py` | 140 | Solution → params, and the `OutputLoopIR` dump (never raises into codegen) |
| `sigma_c.py` | 110 | `order_sigma_c` — the §5.3.1-pt4 refill-after-read freeze |
| `geometry.py` | 111 | broadcast axes, tile counts, wmma grid |
| `scenarios.py` | 71 | named param presets for tests |
| `latalg.py` | 66 | the integer stand-ins for PyCuTe (`tile_split`, `recast_count`, …) |
| `decoder.py` | 45 | a **pure re-export shim** — the historical `decoder.*` names |

`decoder.py` holds no logic. It exists so `KernelWriter.py`, `bridge.py` and
`Lowering/loopir_to_gir.py` keep importing one name.

---

## 3. The data model (`theta.py`)

### 3.1 `Mode` — and the one sentinel everything turns on

```python
@dataclass(frozen=True)
class Mode:
    name: str
    extent: int = 0          # static trip count; >0 = inner (unrolled), 0 = outer (runtime loop)

    @property
    def is_outer(self): return self.extent == 0
```

`extent == 0` is **the** structural distinction in the whole decoder. There is no `kind` field, no
bool, no enumerated level names:

- **`extent > 0` — INNER.** A statically-known trip: the K substep, the M/N fan. Fully unrolled,
  no back-edge. The §2.6 register machinery, the read nest, and the geometry layer iterate exactly
  these (`inner_modes()`). The register ring rotates *within* the unrolled body.
- **`extent == 0` — OUTER.** A problem-dimension trip (`K//DU`, tiles-per-block). A real runtime
  loop with a back-edge, needing the software-pipeline peel and the `Cond` short-loop guard. The
  shared multibuffer rotates *across* this back-edge.

Consequence: adding a persistent `gtile` level, or making the tail's substep a runtime loop, is a
**new `Mode` with `extent = 0`** — not a new decoder branch. (The `probe_theta_tail*.py` probes in
`decoder_proto/` exercise exactly this.)

### 3.2 The level family

```python
levels()              = list(ord)                        # full permutation, outer→inner
outer_modes()         = [m for m in ord if m.is_outer]   # in ord order, outermost first
reduction_chunk_mode()= outer_modes()[-1]                # the INNERMOST outer level
inner_modes()         = [m for m in ord if not m.is_outer]
```

`reduction_chunk_mode()` is the single accessor three things key on, and its docstring says they
"must never disagree": the shared ring depth `S_shared`, the shared-read `src_slot` rotation, and
the steady `Loop`/`Bind` the peels pin. Historically some call sites used `outer_modes()[0]` and
others `[-1]` — identical with one outer level, divergent the moment a second exists (#156, #113).

> **The peel used to be a fourth item on that list and no longer is (#210).** §5.3.1's per-level
> construction makes `M_ℓ` a vector over levels, and the paper names keying it on this one accessor
> as the defect. What the peel still asks this accessor for is "the depth *at* the chunk", the one
> level `build_ir` can emit a region for — see §6.4/§6.6.

### 3.3 Presence, and where the reduction axes come from

Presence is *derived*, never declared. Each `Operand` carries `Fragment.broadcast_axes` — the
modes its data is **constant over**. Then:

```python
presence_modes(op) = [m for m in inner_modes() if m.name not in broadcast_axes(op)]
reduction_modes()  = [m for m in inner_modes() if m not in pres(C)]     # C = the output op-class
free_modes(op)     = presence_modes(op) − reduction_modes()
```

The accumulator `C` is a real `Operand` with `role="output"`, `hops=[]`, and
`broadcast_axes = {K_split, K_inner}`. **The reduction axes are defined as the inner modes absent
from the output's presence** (§2.1 line 82) — not by name-matching "K", which is what makes the
model index-agnostic and batched/einsum-ready (§7.1). There is no `mode_roles()`, no `m/n/k`
literal anywhere in the core.

**Presence is a PER-HOP fact** (§2.1 line 71, §2.8 move 1, §5.3.1 line 583), so
`presence_modes(op, hop)` takes an optional hop:

```python
presence_modes(op)        # the DATA question: what the operand's values vary over
presence_modes(op, hop)   # THIS op-class: geometry.hop_broadcast(theta, op, hop)
```

§2.1 line 71 defines bulk-ness **as** presence — "an inner axis a coarse copy spans in one
instruction". So a bulk `tensor_load_to_lds` moves the whole slab at once and is present on
*nothing* (the copy sits at the reduction chunk); split the tile into `split > 1` storage-disjoint
regions and the region mode comes back into its presence, because now there is one instruction per
region and the copy sits one level deeper. A per-lane hop varies over whatever the data does.

Two traps, both found by the equivalence check in `test_per_hop_presence_REPRODUCES_...`:
a **DTV** operand's `hops[0]` is `global→register`, a READ — ask for the *bulk* hop, not the first
one; and a region **axis** with `split == 1` (the MX scale operands) is still one instruction, so
the axis must not enter its presence.

`Mode` is canonical in this family; `presence()` / `reduction_names()` / `free_names()` are thin
string helpers over it, never the reverse (#167).

### 3.4 `Fragment` — the register-axis partition

```python
broadcast_axes: set     # the axes this operand is CONSTANT over
parts: tuple = (1,)     # int N → (1,)*N; the register reuse-group split
labels: tuple = None    # ("lo","hi") or g0..gN
grouping_mode: str      # first non-region mode of the rotation unit
group_policy: dict      # label → 'unroll' | 'floor' | 'inplace' | int W
```

The key modelling decision: **register groups split the first non-region rotation mode into
contiguous equal ranges.** A TDMSplit region repeats the same grouping: with `M_inner=4`, two
groups own `M={0,1}` and `M={2,3}` in every region. Successive group rotation units use their
own W slots. A read carrier is indivisible: if a quantum carrier preimage crosses one of these
boundaries, that grouping is inadmissible and the VGPR search must choose another scheme.
For folded MXS reads, `(carrier, slot)` determines the physical free-tile lane. Contiguous lanes
get adjacent `unit_index` values, while distributed half-wave partners share one index and use the
instruction's scale selector. The read's `unit_indexes` records every logical lane it defines.
MXS keeps a W=1 register slot; reduction coordinates select adjacent scalar unit groups inside it.
Each logical unit advances by its scale fragment width (one VGPR for MXBlock32, two for MXBlock16).
`register_layout` remains the compact GIR layout, but `operand_emitted_regs` also floors allocation
at `free_tiles * frag_regs`: the scaffold-owned tail loop loads every free tile for one reduction
substep at once, even when `K_inner` has collapsed out of θ.
A region factor inside an operand's rotation unit replicates register positions. A reduction
region outside that unit is an outer enumerator and reuses the same positions; including that
`K_split` again in `unit_index` would double-count the same K partition.

For a partial all-inner prefetch, non-wrapping positions issue on the first invariant-axis pass,
while positions that wrap into the next reduction chunk issue on the last pass. This `reload_wrap`
split preserves exactly one read per residency event without overwriting a value still reused by an
outer free-axis consumer.
Ring slots are decoded from the shifted, factorized read position—not from a flat position divided
by the rotation-unit size. This matters when factors of one logical free axis straddle K (for
example `N_split.K.M.N_inner`) and when a read quantum compresses a free-axis radix before K.
When the read-ahead level is reduction but an innermost free prefetch-unit factor is outside the
rotation unit, that free factor is packed into `unit_index` and the W ring follows reduction. Thus
the MT64×256 case uses W=2 with `K_inner × N_inner = 8` unit positions, rather than using W to
enumerate N or collapsing all four N tiles onto one unit.
If the chosen capped policy cannot retain a value across an invariant-axis pass, that pass becomes
part of the residency-event identity and the read is issued once per event. A pipeline candidate
that cannot carry its requested lead is rejected; the S search may select its explicit in-place
policy without changing the target VGPR cap.
Mixed-policy group guards are evaluated on the shifted destination coordinate. Guarding the
unshifted coordinate sends a read that crosses a group boundary to the wrong group, duplicating one
source and omitting another.

Current restriction: **equal partition** — `len(parts)` must divide `grouping_mode`'s extent.
Uneven partitions are accepted future work (#166).

### 3.5 `Hop` and `Operand`

A `Hop` is one leg of a trajectory with a **mover kind**:

- `'tdm'` — a bulk whole-tile cooperative mover (`tensor_load_to_lds`); its `split` cuts the tile
  into storage-disjoint region parts.
- `'load'` — a per-lane transfer (`ds_read`, `buffer_load`, DirectToVgpr); `split` does not apply.

Inferred when unset: `global→shared` is `'tdm'`, everything else `'load'`.

Note **the retime `δ` is on the placement a movement fills**, as `Movement.offsets[level]`. `off`
is per *(op-class × level)*; a movement is one op-class edge, so the level-keyed map belongs to its
destination. `Theta.off_at(op, role, level)` and `Theta.movement_offsets()` are the read-only
projections.

`Operand` carries only tensor identity — `name`, `free_mode`, `trajectory`, `elem_bytes`, `role ∈
{input, output}`. The ring depth, the region geometry and the register partition are on the
`Shared` and `Fragment` placements inside the trajectory; `Operand.lds_buffers`,
`Operand.region_axes` and friends are accessors onto them, not storage.

### 3.6 `Theta`

| field | paper | notes |
|---|---|---|
| `operands` | `path`, `tile` | trajectories + register partitions |
| `ord` | `ord` | full outer→inner permutation **including** `iter` |
| `fused_copy_groups` | `Φ` | which copies merge into one cooperative instruction |
| `agent_assignment` | `ρ` | **the agent map** — `axis_assignments` of `(axis, agent-level, extent)` |
| `wave_count` | — | how many agents cooperate on one tile; drives `agent_distributed` |
| `per_region_completion` | `π` | one completion class per movement, or one per region |

`off` is **not** a field — it is read from the placements. `S` is **not** a field either — it is
derived by `build_S(theta)` as a `BufferDepths`.

Two derived-authority helpers worth knowing:

- **`movement_units()`** — the single authority for "which cooperative movements exist and how
  many region instances each has". Returns `[(unit_key, members, n_regions)]`. Both the emitter
  and the GIR token pass read it, so they cannot disagree about a movement's completion
  granularity.
- **`agent_distributed(opname)`** — are this op-class's shared hazards cross-agent (§2.4)?

  ```python
  if op.trajectory.shared is None: return False
  return self.wave_count > 1
  ```

  Φ is deliberately **not** part of this test. An earlier version made fusion the whole predicate
  ("the members of a fused group disagree about their carrier"), which returns `False` for a lone
  operand cooperatively loaded by four waves — no fence, a race. Fusion decides whether A and B
  ride one instruction; it says nothing about who carries what.

---

## 4. params → θ (`translate.py`)

### 4.1 The canonical 6-mode vocabulary

Every axis contributes a **(split-region, inner-remainder)** pair:

```
0 K_split   K region (the DU split)          3 M_inner   WT0/M_Split — A's grouping mode
1 K_inner   substep / K_Split                4 N_split   N region
2 M_split   M region                         5 N_inner   WT1/N_Split — B's grouping mode
```

A split of 1 makes its mode extent-1, which drops out of `ord`. All-ones degenerates to the
3-loop `[K_inner, M_inner, N_inner]`.

### 4.2 LoopOrder is a *word*, not a table

- 3-letter shortcut (`"KMN"`) — each axis's (split, inner) pair kept contiguous; expands by
  doubling to `"KKMMNN"`.
- 6-letter word (`"KMKNMN"`) — each letter twice; **first occurrence = that axis's `_split`,
  second = its `_inner`**, letting regions interleave.

Because first-occurrence is always the split, the encoding **structurally excludes** inner→split
orders (a region indexed inside its own fan). That is a design choice, not an oversight.

### 4.3 TDMSplit is 4-valued

```python
ts = [A_MT, B_MT, A_DU, B_DU]      # right-padded with 1s; legacy bool True → [2,2]
kSplit = min(A_DU, B_DU)           # the SHARED K split
resA, resB = A_DU // kSplit, B_DU // kSplit    # per-operand K residual, onto its own copy hop
```

Requires one DU factor to divide the other, else it raises. `A_MT`/`B_MT` become `M_split` /
`N_split`; the copy-hop split is `mSplit * kSplit * resA` (and the `N` analogue).

### 4.4 Retime-offset construction

```python
outer_level   = ord_[0].name          # the DepthU reduction chunk `iter`
readahead_level = <this operand's innermost advancing axis>

for op in operands:
    op.trajectory.shared_fill.destination.offsets[outer_level]      = PGR_op
    op.trajectory.fragment_fill.destination.offsets[readahead_level] = PLR_op
```

Each offset sits on the placement its movement fills, so the key is `(op-class, level)` — the
paper's — rather than `(operand, level)`. `off(read, iter)` and `off(copy, iter)` are distinct
because they are on different placements, which is what the boundary term needs: it asks for
`off(p, ℓ_outer)` for a `p` whose home level is inner to `ℓ_outer`.

### 4.5 The agent assignment, and the single agent authority

`AgentAssignment.axis_assignments` is a set of `AgentAxisAssignment(axis, level, extent)` — which
looped axis each agent level traverses instead of the coordinate, over how many agents (paper §2.8
move 2, §2.9/§3.3). Levels are ordered coarse→fine (`block`, `wave`, `subwave`);
`AgentAssignment.roles` is the `specialize` half and is empty, because no shipping configuration
partitions agents into producer/consumer roles.

`Fragment.agent_group_size` is a projection of the assignment. `RegionLayout.wave_relative` is
different: it records that VW/wave layout/LDSSegmentInterleave make the storage region depend on
the wave, so hazard analysis must widen the read's region aliases. It is derived from the region
layout alone (`wave_span < free_split`), independently of the agent assignment. It never removes
`M_split` or `N_split` from the coordinate; TDMSplit remains the exact `split × inner`
factorization above. `wave_served_axes()` is still the assignment at wave-and-coarser, but
TDMSplit axes are not entries in that set.

**This was typed wrong for the whole implementation, and the mistype concealed itself.** As an
equality-only `{op-class → opaque selector}` token, ρ could answer only "are these two on
different agents?" — which `theta.agents > 1` already answers — so it had zero readers, while its
real content reappeared as three unnamed per-hop scalars, each recovered from an *address-shaped*
fact (a lane-layout comparison, an address-layout query) smuggled into the hop. Paper §2.8 L215
names the trap: opacity governs a placement's *content*, **not** the mode→level map.

ρ is **supplied**, like `S` and the movement quantum — a searched preset the target computes and θ
reasons over. That is what keeps the address query outside θ; do not try to derive it here.

`WaveSeparateGlobalRead` is still *not* ρ: it only rewrites `LSC`/`LSP` (the load-tile
*decomposition*); every wave still loads part of A and part of B, so the carrier set is unchanged.

---

## 5. The IR node set (`ir.py`)

| node | meaning |
|---|---|
| `Inst(op, placement, agent, awaits)` | one emitted operation; `op` is a `Load` or `Mma` |
| `Loop(mode, trip, bodies, body_ranges, outer)` | a real loop. `trip` is the **continuation predicate** for an outer loop, a static count for an inner one. `outer` is an explicit structural flag, never inferred from trip being symbolic |
| `Peel(kind, mode, k, body)` | the hoisted first-`k`/last-`k` iterations; `kind ∈ {prologue, drain}` |
| `Bind(mode, value, body)` | pins an outer induction to a value over a straight-line body — prologue step `j` → `iter = j−M`, drain step `t` → `iter = T−M+t` |
| `Cond(pred, then, els, kind, label)` | a guard. `kind ∈ {peel_validity, readahead_suppress, first_touch, short_step_validity}` — **generic paper vocabulary only**; `label` is a scaffold hint set downstream by GIR |
| `Branch(mode, modulus, arms)` | explicit rotation control (a switch on `mode % S`) |
| `Expr` / `Pred` | symbolic slot/coord arithmetic, and structured predicates read **by field** |
| `Await(dep, counter, scope, kind, note)` | a named discharge point; `count` is never set |

`Loop.body` (singular) **raises** on a multi-body loop rather than silently returning
`bodies[0]` — that convenience is exactly how a multi-body schedule becomes a wrong one.

`Bind.bound_env` binds `mode` only when `value` is statically resolvable; a drain step's
`T−M+t` references a runtime symbol, so the body stays symbolic and GIR resolves the residue
*relative* to the steady loop rather than pinning a wrong concrete generation.

---

## 6. The derivations

### 6.1 What `build_S` returns

```python
build_S(theta) -> (S, floor)
    S.depths[(op, region, group_label)]  = W_g       # register rotation width, per reuse group
    S.depths[(op, region, "shared")]     = S_shared  # the LDS ring depth
```

Two different placements, and they no longer share one map. The register widths are
`Fragment.ring_depths` (a partition of a grouping axis's values) and the LDS ring depth is
`Shared.ring_depth` (a scalar over modes) — Definition 1's two shapes, on the two placements.
`BufferDepths.shared(op)` reads the latter, so no caller has to exclude a reserved key from the
former.

**`region` is the placement index** (Definition 1: `S_Λ : Q(Λ) → ℕ₊` is indexed by the placement
Λ, not by the operand). `None` = one whole-tile placement, which is every row `build_S` produces
today; a region-split tile is *two placements of one operand*, and a two-part `(op, group)` key
cannot name them — that is what blocked TDMSplit (#212 → #217).

`get`/`groups_of` without a region do **not** silently pick one: they collapse when the per-region
rows agree and **raise** when they disagree, because collapsing a per-placement fact to
per-operand is the bug the key exists to prevent. Use `regions_of(op, group)` to enumerate.

### 6.2 R, L, W — the register band (§2.6, Lemma 3b)

Three quantities must stay separate:

* the **prefetch unit** is the operand's innermost logical axis in `ord`, excluding TDMSplit
  region factors;
* the **rotation unit** is its inner non-outmost axes, or the full inner set for an all-inner
  operand;
* W is the number of those rotation units the selected VGPR ring stores.

Thus `MNK` A has prefetch/rotation unit K, while all-inner B has prefetch unit K and rotation
unit N*K. A width contributes `W * rotation_unit / prefetch_unit` units of capacity. Every
split operand converts global PLR using its own region-local prefetch unit, including an
all-inner operand; neither the rotation unit nor a TDMSplit region factor multiplies that request.
An unsplit all-inner operand retains the direct side's literal tile lead. The all-inner
distinction also adds the copy-staging cap.

| symbol | function | meaning |
|---|---|---|
| **R** | `group_ring_size` | successive rotation units visited by each group in one chunk (one for a whole-set operand) |
| **L** | `group_live_peak` | simultaneously live rotation units; L=1 without prefetch |
| **W** | `group_width` | rotation-unit depth selected per group |

Grouping divides the first non-outmost rotation axis. It divides the prefetch unit only when that
is the same axis. For two `MNK` B groups, each rotation unit is `N/2*K` but the prefetch unit
remains K. For split MXFP8 KMN A, each region has `M_inner=4`; two groups make two-tile units.
The split axis selects a region and does not multiply PLR, so global PLR1 names four M_inner tiles,
hence requested PLRA2 before the selected W policy applies its cap.
Without regions, four two-tile groups use W=[2,1,2,1] and retain PLRA4.

The unit determines which coordinates PLR advances to; it does not impose an axis-boundary
placement. LoopIR emits a canonical order induced by the WMMA traversal. GIR's read hoist may
choose an earlier issue slot, but preserves that read order and is bounded by hazards on the
physical compact `(group, slot, unit)` register name. Thus placement may be interleaved with the
preceding `N.M.N` traversal without making the split axis a scheduling boundary, while
`M_split=1` still does not enter the PLR1 prologue.

The same ordering rule applies to TDM copies. GIR hoists each region-pinned shared copy against
its own LDS hazards; only unsplit copies are collapsed to a common issue point for barrier sharing.
The PGR distance and LDS-buffer budget are unchanged.

Read hoisting is transactional with respect to the RAW-fence profile: after a tentative read
move, GIR recomputes frame hazards and rejects the move if any RAW fence would move earlier.
Copies are then hoisted against that accepted read order. Fence slots are the concrete latest
slots selected by `FenceRegions`; waits retain each frame's ring/gdelta relation and the dump
shows both the formula inputs and resolved token.

### 6.3 `S_shared` is READ, not derived

`_shared_depth` returns `op.lds_buffers` (from `NumLdsBlk` / `LDSBufferA|B`), falling back to `δ`
only when unstated. The docstring records why, and it is worth repeating because the bug was
quiet:

> Deriving `S = δ` coincides with the buffer count at `PGR=2`, so every test agreed. At `PGR=1`
> against the same double-buffered LDS they diverge — θ said 1, the kernel had 2 — and a depth-1
> ring has no rotation, so the decoder emitted **no swap at all** while the kernel's addressing,
> its second `LdsOffset` block and its token flip all still alternated. Every copy overwrote the
> buffer being read.

There is also **deliberately no `S ≥ δ` assertion** here: `1LDSBuffer=1` with `PGR=2` is a
shipped configuration, i.e. a real kernel with `S_shared = 1 < δ = 2`. `off(copy, chunk)` is
`PrefetchGlobalRead`, which is *not* the LDS placement's δ on this target (the global read lands
in G2L registers; the LDS write is a separate serialized step). A genuine `S ≥ δ` check belongs
against the local-write offset once the write side is modelled (#116).

`_shared_depth` **raises** for an operand with no shared hop rather than returning a fiction.

### 6.4 The peel depth — `_peel_depths` returns a **per-level** `PeelDepths`

```python
@dataclass(frozen=True)
class PeelDepths:
    per_level: dict   # {level_name: M_ℓ = max_p off(p, ℓ)}, ord order, only M_ℓ > 0
    offs:      dict   # {(op_name, role, level): δ} — the per-op-class offsets each M_ℓ maxes
    dr:        int    # the substep read-ahead depth (kept: the read family works in SUBSTEPS)
    chunk:     str    # the reduction-chunk level name

    def depth(level)      -> M_ℓ
    M                     -> depth(chunk)     # the one level build_ir emits a region for
    def peeled_levels()   -> every level wanting a peel
    def copy_off()        -> {op_name: δ} for COPY op-classes at the chunk
```

`M_ℓ` is the max over **every op-class carrying an offset at ℓ** — copies *and* reads, which the
op-class key (§4.4) finally makes distinguishable. At the chunk the substep read-ahead also
contributes, converted to whole chunks (`ceil(dr / n_substeps)`); that is a *reach*, not an offset
at the chunk, so it bumps the chunk's max without entering `offs`.

`offs` stays **per-op-class** (PGRA ≠ PGRB is an ordinary θ point) — the peel is *staggered* by
each op-class's own δ, not by the level's max.

### 6.5 The cross-level boundary term — `boundary_hoists`

§5.3.1's Lemma 1 term: when `off(p, ℓ_outer) = δ > 0` for a `p` whose **home level** is inner to
`ℓ_outer`, `p`'s leading δ inner-nest instances relocate into the **previous `ℓ_outer` iteration's
drain** (at the first iteration, into the outer prologue).

`op_class_level(theta, op, role)` gives the home level: a read's innermost presence mode; a
region-split copy's region mode; the chunk otherwise.

Three conditions, and the third is the one that is easy to get wrong:

1. `level` is an **outer** level — only an outer level has a drain to relocate into. An offset at
   an inner unrolled level crosses no boundary; that is the ordinary §5.3.1 pt5 read-ahead.
2. `home` is strictly **inner** to `level`. `home == level` is the ordinary staggered peel.
3. `home` is itself an **outer (looped)** level.

Without (3), a **region-split copy** — home `M_split`, offset at `iter` — satisfies (1) and (2) and
looks exactly like a relocation. It is not one: its region instances all stay in their trip and δ
shifts which *chunk* they fetch. That is `PrefetchGlobalRead`. The first implementation of this
function omitted (3) and refused every shipping region-split kernel;
`test_region_split_copy_is_NOT_a_cross_level_boundary_term` pins it.

`boundary_hoists` is empty for every θ the parameter path builds today, and becomes non-empty the
moment a coarser outer level exists (§5.3.2's persistent `gtile`).

### 6.6 What `build_ir` refuses

`emit._check_peel_is_emittable` raises `NotImplementedError` when θ asks for either

- a peel at an **outer** level other than the chunk, or
- a cross-level boundary hoist.

`build_ir` peels exactly one level, because a coarser level's prologue/drain has to sit inside an
enclosing `Loop` for that level and that Loop is #113. Emitting the chunk-only peel anyway would
ignore the coarser `off` **and still report an empty ledger** — precisely the silent drop the
paper names. So it refuses, with the derived facts in the message.

Inner-level `M_ℓ` is *not* refused: an inner level is statically unrolled, so its peel is absorbed
into the unroll and realized by the read-ahead family (`readahead_prologue` primes it,
`_readahead_shift` sustains it, the drain's `readahead_suppress` guard ends it).

### 6.7 Read-ahead: one derived depth per operand

`requested_read_ahead(theta, op)` first derives the uncapped internal PLR:

```python
split_operand    = global_PLR * region_prefetch_unit / group_prefetch_unit
direct_unsplit   = global_PLR * original_prefetch_unit / group_prefetch_unit
all_inner_unsplit = global_PLR * direct_operand_original_prefetch_tiles
```

With S, it caps that value by the sum of the groups' W-scaled rotation capacity and by what the
copy/LDS pipeline has staged. `Schedule.want(op)` is the single realized answer. Every group in
the operand's ordered stream shares that advance; W remains per group and controls its register
name repetition. This is why split MXFP8 can have one PLRA3 over W=[2,1] rather than independently
downgrading the W=1 group to zero.

An all-inner operand additionally caps its derived request by
`(PGR-1) * rotation_unit/group_prefetch_unit`: at PGR1 it remains in place; at PGR2 it may reach
the requested unit in the next chunk.

When an unpartitioned all-inner advance crosses a chunk, it moves and preloads the complete
rotation unit. Its VGPR slot is a static reduction phase, never a function of `iter`; the next
chunk refills that phase only after the last consumer on every invariant outer axis. A rolled loop
cannot alternate a statically named VGPR bank by outer iteration.

`preloaded_tiles` takes the physical prefix of that stream:

```python
count = Schedule.want(op) * group_prefetch_unit_tiles(op)
```

The same physical distance drives the shifted read coordinate, target group, register slot and
shared source slot. A W=1 destination may require σ_c to place the read after the consumer that
vacates it, but it keeps the same operand PLR.

### 6.8 Chunk crossing — `chunk_reach` and `copy_must_be_first`

The operand read-ahead is not confined to its own chunk:

```
distance = Schedule.want(op) * group_prefetch_unit_positions(op)
span     = product of this read's reload extents
r        = ((span − 1) + distance) // span
```

`r` is how many chunks ahead the read-ahead reaches. Then:

```python
copy_must_be_first(op) =
    (r >= 1) and (off(copy, iter) == r) and (S_shared > off(copy, iter))
```

| case | relation | σ_c |
|---|---|---|
| `off == r` and the LDS slot differs | the copy fills the chunk the read-ahead crosses into → **RAW** | copy **first** |
| otherwise | the copy's slot collides with a chunk read earlier in the trip → **WAR** (or nothing) | copy **last** |

At `S_shared = 2, r = 1` the two shipping points go **opposite ways**: PGR1 (`off=1`) is
copy-first, PGR2 (`off=2`) is copy-last. An earlier version returned `S_shared > r`, true for
both, and emitted PGR2 copy-first — the copy then overwrote the buffer the trip was still
consuming, miscomparing on hardware at K=256.

`chunk_crossing_violations` is the closed-form prune of the invalid `(dr, off)` combinations.

---

## 7. The ledger (`ledger.py`)

### 7.0 The idea, in plain terms

**The ledger is a list of facts, written down before any code exists, of the form "X must finish
before Y starts".** It is built from the input description alone. Then the code is emitted. Then you
walk the emitted code and check every fact on the list was taken care of. If any is left over, the
build fails.

The value is in the two halves being *independent*: the list comes from the description of what the
kernel does, the accounting comes from a walk over what was actually emitted. If they agree, nothing
was forgotten — and that is a different kind of claim from "we tested it and it worked".

#### Where the facts come from

Four sources, all mechanical:

1. **Every leg of every journey.** Data moved from here to there must arrive before anything reads
   it there. One fact per leg. Just walk the trajectories.
2. **Every recycled buffer.** If you cycle through two buffers, the load refilling one must not
   start until the reads of what was in it are done. One fact per buffer group.
3. **Reading early across a boundary.** If a read runs far enough ahead that it reaches into the
   *next* chunk's data, that chunk's load must have landed. Only generated when the reach actually
   crosses — most configurations don't.
4. **Warm-up requirements.** Some reads cannot be satisfied inside the loop body at all; they have
   to be pre-issued during warm-up. One fact per required pre-issue, each naming the exact
   coordinate it must happen at.

#### The distinction that matters: arrive vs vacate

Two flavours, and they are made to happen in completely different ways.

- **"The data must have arrived"** — a producer must finish before a consumer starts.
- **"The buffer must have been emptied"** — a consumer must finish before something overwrites it.

It is tempting to think both are solved by a wait instruction. They are not.

#### Three ways a fact is made to happen

**(a) A wait.** For *arrive*: attach a wait to the consumer, naming the producer. The hardware
counter does the rest. Straightforward.

**(b) Position in the emitted order.** For *vacate*: a wait is **useless here**, and this is the
non-obvious part. A wait waits for something already issued — but the read you need to finish may
not have been issued yet. Nothing you can attach to the overwriting load can express "wait for a
read that hasn't happened". The only thing that works is *where the line sits*: put the overwriting
load **after** the reads in the emitted body. This fact is discharged by position, not by an
instruction.

**(c) Simply being there.** For *warm-up requirements*: the discharge is that the operation exists
in the warm-up code at the right coordinate. Nothing to wait on, nothing to order — either the line
was emitted or it wasn't.

#### A worked pair

Double-buffered staging, loads running two chunks ahead:

| fact | how it is made to happen |
|---|---|
| the load into the buffer must finish before the read out of it | the read carries a wait naming the load |
| the load refilling a buffer must not start until this pass's reads of it are done | the load is **placed after** the reads in the body |

The second one is not theoretical. The natural way to write the body is "loads, then math" — and
that ordering violates it: the load overwrites the buffer the reads are still consuming. We shipped
that once and it miscompared on hardware. The checker catches it now, because it verifies *position*
for that flavour of fact and not just the presence of a wait.

And it cuts both ways: when a read runs ahead into the chunk a load is *currently* fetching, that
same load must come **first**. Which of the two applies is computed per operand, and A and B in one
kernel can want opposite answers.

#### How the check works

Walk the finished tree once, and for each fact ask the question appropriate to its flavour:

- *arrive* → is there a wait naming this producer's completion, somewhere in the tree?
- *vacate* → is the overwriting line genuinely **after** a read of that buffer, within the same
  straight-line stretch?
- *warm-up* → is the required operation actually present in the warm-up, at that exact coordinate?

Anything unanswered is returned as a violation, and the build raises rather than emitting a kernel
that looks fine.

One ordering detail matters: a redundancy pass runs **before** the check, removing repeats of a fact
already established earlier on the same path (one boundary restated on four consumers is still one
boundary). The check therefore inspects what is really emitted, not a tidier version of it.

#### Why the list can't be missing one

"Check every fact on the list" is only worth something if the list is complete. Two separate
arguments are needed, and they answer different questions.

**1. Why the *kinds* are complete — an argument about how a θ is built, not about the code.**

A hazard is, by definition, two accesses to one location, at least one of them a write, whose order
isn't already forced. So a hazard needs (a) an access, and (b) two accesses landing on the same
location. Both are created **only** by the moves that construct a θ — and there are nine of them.
Case over the nine: each either creates no new access, or creates no new sharing, or creates sharing
of a kind already tracked. Since those moves are the only way to build a θ, the enumeration is
exhaustive. That is the paper's Lemma 7, and it is what the whole scheme rests on: you don't find
hazards by looking for them, you know the list is closed because the ways of making one are closed.

Three kinds survive the case analysis:

- data must **arrive** before it is used,
- a recycled buffer must be **emptied** before it is overwritten,
- two different roles or agents touching the **same storage**.

And one thing is *not* created by a move at all, so it is accounted for separately: the **compute**
op. Its fragment read is the consuming end of an arrival fact — and, easy to miss, that same read is
*also* the vacating last-read of the register-recycling fact. The accumulator's running sum is a
loop-carried dependency of distance one, on one register, on one lane, discharged by the hardware's
own instruction latency rather than a wait.

**2. Why no *instance* is skipped — an argument about the code.**

Knowing the kinds are complete does not help if the implementation forgets to walk something. So
`build_ledger` iterates **structure**, never a list of cases:

```python
for op in theta.operands:            # every operand
    for m in op.movements:           #   every leg of its journey   -> one arrival fact
for (op, group), depth in S.depths:  # every rotating buffer group  -> one vacate fact
```

Nothing in there says "for A and B" or "for LDS and registers". Add an operand, add a placement, add a
ring, and a fact appears without anyone editing the ledger. That is the property that makes an
instance hard to drop.

**Where it can still miss — three honest holes.**

- **(a) Kinds the model does not have.** The write/store side is not modelled, so a storing kernel
  produces no store facts and its empty list says nothing about them. Not "missed" — out of scope,
  and silently so. This is the same caveat as *The honest limit* below, stated from the other side.
- **(b) Structure that is not iterated.** `S.depths` only holds what `derive_S` visited, and it
  skips the output op-class. Correct today (the accumulator has no ring), wrong the moment the
  store has a staging ring (#212).
- **(c) A conditional fact whose condition is computed wrong.** Two of the four sources are
  conditional — the crossing fact fires only when the read-ahead's reach genuinely crosses a chunk
  boundary, and the warm-up facts only for the coordinates the steady body cannot cover. Those
  depend on a *derived quantity* being right, and structural iteration cannot save you if the
  formula is wrong: the fact is simply never generated. **We have shipped that bug** — an earlier
  version of the reach test answered "yes" for both prefetch depths, so the copy was emitted first
  at the deeper one, overwrote a buffer still being read, and miscompared on hardware.

So the sharp version: **structural facts cannot be missed by construction; conditional facts are
only as good as their formula.** When reviewing a change here, that is where to look.

#### The honest limit

The check proves the emitted code accounts for **every fact on the list**. It says nothing about
hazards the model has no concept of. The write/store side is not modelled today, so a kernel that
stores can have an empty list and that emptiness means nothing about its stores. "Empty ledger" is
"everything we know to look for is handled", not "correct".

#### Bridge to the names

*Fact* = obligation. *Arrive* = RAW residency. *Vacate* = WAR (rotation or in-place, depending on
whether the overwrite lands on a slot currently in use). *Reading early across a boundary* =
crossing-RAW. *Warm-up requirement* = read-ahead residency. *Wait* = `Await`. *Position rule* = σ_c.
The rest of §7 uses those.

### 7.1 Endpoints are records, not strings

```python
@dataclass(frozen=True)
class Endpoint:
    op: object      # op-class name, OR the MEMBER TUPLE of a Φ-fused movement
    at: str         # a Space, a rate-group label, or a site ('prologue'/'wmma')
    role: str = ""  # 'lastread' | 'refill'
    coord: tuple = ()
    gen: tuple = () # generation qualifiers, on an Await dep only
```

This was a formatted string (`"A@shared"`, `"A:g0:refill"`) that four call sites re-split with
three different rules. `render()` still produces the familiar form for dumps, but nothing parses
it back.

### 7.2 The five obligation kinds

| kind | producer → consumer | counter | when |
|---|---|---|---|
| `RAW-residency` | one per hop: `X@src → X@dst` | that hop's | always |
| `inplace-WAR` | `X:group:lastread → X:group:refill` | the **vacating read's** | the refill writes a name in use |
| `rotation-WAR` | same endpoints | the vacating read's | `W > 1` and no name collision |
| `crossing-RAW` | `mv:shared:copy → mv:shared:crossing-read` | the copy hop's | `r ≥ 1` and `off(copy) ≤ r` |
| `readahead-residency` | `X@prologue:coord → X@wmma:coord` | the read hop's | one per required prologue read |

**WAR labelling** follows §4.1's note: the rotation WAR is discharged by the *vacating read's*
selector composed with placing the refill after it, so the obligation is tagged with the **read's**
completion class, not the physical refiller's. A shared buffer's WAR therefore carries the DS
(read) counter even though the physical refiller is the global→shared copy.

**in-place vs rotation** is decided by *which name the refill writes*, not by width:

```python
inplace = depth < 2
if group is the shared ring:
    inplace = inplace or depth <= off(op, reduction_chunk)     # S == δ ⇒ in place
else:
    inplace = inplace or readahead_is_inplace(...)             # shift ≡ 0 mod names(p)
```

At PGR2 the shared ring is exactly `δ` deep and the steady copy targets the slot being read now.
Keying on `depth < 2` alone called that a rotation, leaving σ_c free not to defer it.

**crossing-RAW** was the missing obligation that let a miscompiling point through. On one buffer
in one trip there are *two* distinct things:

```
WAR — copy(iter+r) after the last read of chunk iter+r−S_shared   (an EARLIER trip)
RAW — the crossing reads of chunk iter+r after copy(iter+r)       (THIS trip, if off == r)
```

Only the WAR was modelled, so nothing forced the copy to precede its own consumers.

### 7.3 Φ collapses obligations at **both** ends

```python
name = movement_name(theta, op.name) if h.dst == SHARED else op.name
ledger.add(Obligation(Endpoint(name, h.src), Endpoint(name, h.dst), "RAW-residency", ...))
```

A fused group is one cooperative instruction with one completion, so "A's copy completed" is not
an event that exists. Naming only the *producer* leaves two obligations that happen to share a
producer — and two obligations are two discharge accounts: A's reads settle one and B's the other,
so neither read is ordered against the other member's data. That was the multi-wave TDM defect.

### 7.4 `discharge_once`

One obligation is discharged once. Within a straight-line path, later consumers are transitively
ordered behind the first one's boundary, so restating the `Await` on each consumer turns one
boundary into N discharge sites (the prologue read-ahead emitted four identical `await A+B@global`
lines for one cooperative copy).

Scoped to `RAW-residency` and `crossing-RAW` only — a rotation WAR's producer is a per-consumer
vacating read, so those are genuinely distinct. A boundary dies when its producer re-issues, and a
`Loop` body is entered with a fresh set.

### 7.5 The gate — `check_ledger_discharged`

Five checks, not one:

- **(a) coverage** — no `Await` on the obligation's completion class anywhere in the tree.
- **(b) shared order** — an `inplace-WAR` refill copy emitted with no preceding vacating read in
  the *same region body*. Scoped to `inplace-WAR`: matching `.endswith("WAR")` forced copy-last
  universally, which is the ordering that makes the crossing read consume an unwritten buffer.
- **(b2) register order** — an in-place refill *read* hoisted above the wmma consuming its group.
  This arm was once missing: the filter was `consumer.at == "shared"`, but register endpoints
  carry `at` = the *group label*, so every register WAR was silently coverage-only.
- **(b3) crossing order** — the copy must precede its crossing reads in the steady body.
- **(c) read-ahead residency** — each required prologue coord must actually be emitted as a
  register read in the prologue body.

An `Await` cannot substitute for (b2): it waits on already-*issued* reads, and the consuming wmma
has not issued.

`_region_bodies` defines the σ_c unit — **one emitted trip**. Trip boundaries are the outer
structures: the steady `Loop` body, the prologue `Peel` body, and each drain step's `Bind`.
Everything nested inside is flattened in program order. The `inline_outer` flag exists because a
drain step's `Bind` would otherwise return an empty trip and the order gate would be vacuous.

---

## 8. σ_c (`sigma_c.py`)

One rule at two ring levels — **a refill is emitted after the consumers that vacate the buffer it
overwrites**:

1. **Shared** — a WAR-refill copy moves to the end of the body, after the read/wmma nest.
   Prologue copies (no WAR await, empty buffers) stay put. A copy in `copy_first_operands` is
   exempt — it is a RAW producer for this trip's crossing reads and must lead them.
2. **Register** — a refill read carrying an `inplace-WAR` moves after the wmmas that vacate its
   slot. A `rotation-WAR` read (`W > 1`) is *not* deferred: it writes slot `(k+dr) mod W` while the
   consumer reads `k mod W`, so hoisting touches no live register.

Register deferral precedes shared deferral, so the shared copy stays last.

`_movement()` looks **through** a rolled region `Loop` and a single-armed guard `Cond` to find the
`Inst` a node denotes — a region-split copy is one op-class under `Loop(region_mode)`, and matching
only a bare `Inst` would silently stop deferring split copies.

---

## 9. `emit_can` — `build_ir` in detail (`emit.py`)

`build_ir(theta, S)` is ~650 lines, almost all of it closures over `(theta, S)`. This section walks
it in execution order. Line numbers are `emit.py`.

### 9.0 The idea, in plain terms

Skip the vocabulary for a moment. This is what `build_ir` is doing.

**In:** a description of the kernel as data — which blocks of data move from where to where, which
loops exist and in what nesting order, how far ahead of its consumer each movement runs, and how
many buffers each thing cycles through.

**Out:** a loop nest. Not a list of instructions — a nest, with real loops in it, in which each
*kind* of operation appears exactly **once**.

That last point is the whole design. "Load A" is one line in the output no matter how many times it
executes. Its buffer index is written as a formula in the loop counters (`(k+2) mod 2`) rather than
a number, so nothing is enumerated and the output size doesn't grow with the trip count.

Everything else follows from one rule: **you never say where an operation goes — you say what it
depends on, and its position falls out.**

#### The steps, in the order they happen

**1. Put each operation at the innermost loop whose counter changes its data.**
For each operation, ask of each loop: as this counter advances, does the operation touch *different*
data? If yes it belongs inside that loop; if no, outside. A bulk load that fetches a whole slab in
one instruction doesn't vary with anything inside the chunk loop, so it sits right at the top. A
fragment read that fetches something different for every (k, m) sits at the bottom. Nobody assigns
levels by hand.

**2. Running ahead is what splits the loop into three.**
If loads run two iterations ahead of the math that consumes them, then the first two iterations have
loads but no math, and the last two have math but no loads. Only the middle is uniform. So the
output is not one loop — it is *warm-up straight-line code*, then *the loop*, then *wind-down
straight-line code*. This is the single biggest structural consequence of the input, and almost all
the complexity below is a detail of it.

**3. Different operations run different distances ahead, so the ends are staggered.**
If A runs 1 ahead and B runs 2, then warm-up step 1 issues only B, and step 2 issues both. Wind-down
is the mirror: the deepest-running one stops first. Each operation enters and leaves on its own
schedule, not the group's.

**4. The order *within* the body is fixed deliberately, and two opposing rules decide it.**
A load that is about to overwrite a buffer must be placed *after* the reads still using it —
otherwise the data is gone before it's read. But a load whose data *this* pass is about to consume
must be placed *first* — otherwise the read happens before the data lands. Which rule applies is
computed per operand; they can and do disagree between A and B in the same kernel.

**5. Every wait is looked up, never invented.**
Before building anything, enumerate every fact of the form "X must finish before Y starts" — from
the movements themselves, from buffers being recycled, from data being read a chunk early. Then each
emitted line asks "which of those am I the Y of?" and carries those waits. The emitter never decides
a wait on its own. Afterwards, check that every fact on the list was picked up by something. **That
check is the correctness argument** — not testing, not inspection.

**6. Guards for the ragged edges.**
Three cases need a condition rather than a position: a look-ahead read on the last wind-down step
would read past the end of the data; an operation that doesn't vary over a loop it happens to sit
inside would be needlessly re-issued every pass; and a loop too short to fill the pipeline at all
can't use the three-part shape. The first two become conditions on the operation (and where the
loop's trip count is known, the second is turned into *splitting the loop in two* — one pass with the
operation, the rest without — so no condition is evaluated at runtime). The third gets a whole
second version of the code and a runtime test choosing between them.

#### The shape that comes out

```
if (enough iterations) {
    warm-up:    step 0 … step M-1        straight-line, deepest-running loads first
    loop:       one uniform body          each operation once, indices as formulas
    wind-down:  step 0 … step M-1        straight-line, deepest-running loads drop first
} else {
    the short version
}
```

A useful way to hold it: **this is what a person does when hand-writing a pipelined loop** — write
the steady body, then realise the first iterations need special handling, then the last, then worry
about which buffer, then add the waits. `build_ir` does exactly those steps, in that order, deriving
each one from the input instead of from judgement. The rest of §9 is that same sequence with the
real names attached.

### 9.0a The entry point around it

```python
def emit_mainloop(theta):                                   # emit.py:857
    loop   = loop_shape(theta)              # ord contiguity report (§5.1); note only
    S,floor= build_S(theta)                 # §6.1
    ledger = build_ledger(theta, S)         # §7  — built AGAIN inside build_ir (see 9.2)
    ir     = build_ir(theta, S)             # steps 8 + 9a + 9b
    ir     = discharge_once(ir)             # BEFORE the gate, so the gate sees what is emitted
    undischarged = check_ledger_discharged(ledger, ir)
    defects = validate_loopir(theta, ir, S, undischarged)   # RAISES on defect
    return {"S":…, "ir":…, "ledger":…, "ledger_empty": not undischarged, …}
```

`discharge_once` runs **before** the gate deliberately: the gate must check the boundary that is
actually emitted, not a redundant restatement of it.

### 9.1 Setup, and the two things decided before anything is built

```python
_copy_first = copy_first_operands(theta, S)        # emit.py:147 — §6.8
def _order_sigma_c(body): return _sigma_c_order(body, _copy_first)

copy_ops = [op for op in theta.operands if op.trajectory.shared is not None]             # :152
peel     = _peel_depths(theta)                                                            # :153
off, dr, M = peel.copy_off(), peel.dr, peel.M
outer_var  = theta.reduction_chunk_name("iter")                                           # :161
_check_peel_is_emittable(theta, peel, outer_var)                                          # :162
```

`_copy_first` is a **per-operand `frozenset`**, not a global flag: `copy_first_operands` tests each
operand independently with `copy_must_be_first`, using that operand's realized internal PLR,
physical distance, `chunk_reach`, copy offset and LDS depth. Equality `off == r` permits copy-first
only when the copy lands in a distinct LDS slot; a wrapped slot remains a WAR and stays copy-last.

It is derived **once**, here, and handed to every σ_c call below: the set is a property of θ, not of
the region being ordered, and recomputing it per call site is how two regions end up disagreeing.

Note the polarity — **`copy_first` does not force a copy to lead; it exempts one from the WAR
deferral.** `sigma_c.is_war_copy` defers a copy iff it carries a `*-WAR` await **and** none of its
tokens is in `copy_first`. So an in-place WAR anywhere does not make anything copy-first.

The one place the set crosses operands is a **Φ-fused** movement, whose `tokens` are the member
tuple: `is_war_copy` tests `any(t in copy_first ...)`, so a group mixing a copy-first member with a
copy-last member leads as a unit. That conflict is real — one instruction cannot be both — and it is
currently resolved by `any` and then caught downstream by the ledger gate rather than rejected up
front (see §12).

`outer_var` falls back to the literal `"iter"` only for a legacy θ with no outer mode.

### 9.2 Step 9b is set up first: the await index

```python
ledger = build_ledger(theta, S)                    # emit.py:174 region
def _consume_role(ob): ...                         # obligation -> (operand, role)
awaits_by_site = {}
for ob in ledger:
    if ob.kind == "readahead-residency":  continue
    awaits_by_site.setdefault(_consume_role(ob), []).append(ob)
```

The consume-site key is `(operand, role)`:

```
RAW  A@shared   (produced by copy) → consumed by the READ    → ("A","read")
RAW  A@register (produced by read) → consumed by the WMMA    → ("A","wmma")
WAR  A:shared:refill               → the COPY refiller       → ("A","copy")
WAR  A:g0:refill                   → the READ in-place refill→ ("A","read")
```

**`readahead-residency` is skipped on purpose.** Its edge is `complete(prologue read at coord) ⤳
issue(wmma at coord)` — the same edge the wmma's ordinary register RAW already discharges — and its
real content is a *structural* requirement on the prologue, which gate (c) checks directly. It was
once bucketed like a WAR, which put it on the READ (its **producer**, not its consumer) and, because
the bucket key is only `(op, role)`, fanned every coord's obligation onto every read of that
operand: a prologue read awaited its own residency.

`_site_awaits(opname, role, kiter_note, groups)` (`:220`) turns a site's obligations into the
`Await` tuple. Three refinements live there:

- **Φ union.** A fused movement's obligations are keyed by the MOVEMENT (`movement_name`), so a
  member's site resolves to its own obligations **plus** the movement's — guarded so an unfused
  kernel cannot double-count.
- **Group filtering.** `groups` restricts *register-ring* WARs to the groups this instruction
  actually fills, so a split read does not wait on the other half's vacating read.
- **Generation qualifiers.** A RAW dep is `ob.producer.qualified(kiter_note, region_note)` — which
  chunk, and under the per-region π which region, is being awaited.

`_scope_of` (`:93`) gives each await its proc-scope: `wave`, except an `inplace-WAR` on an
agent-distributed movement, which is `block` (§2.4 route b). Only the `S=δ` WAR needs it — a RAW
awaits a completion class, which is not agent-relative, and a `rotation-WAR` has the spare slot as
the alternative remedy.

### 9.3 The three instruction builders

**`_copy_inst(member_ops, slot, nregions, war=True)` (`:319`)** — one movement instance.

- `tokens = tuple(op.name for op in member_ops)` — a token is **always** a θ op-class name. Which
  storage region a movement carries is a *coordinate* (`coord`), never baked into the name: encoding
  `(op-class, region)` as `"A0"` made three call sites grow three different string-strippers that
  disagreed at ≥3 regions.
- `size` sums each member's **own** per-region share (`tile_bytes // split`).
- `war=False` for the prologue: those copies fill empty buffers, so the vacating-read WAR does not
  apply and is dropped from the awaits.
- Awaits are a real **union with dedup** across members — the fused RAW resolves to the same dep for
  every member, so concatenating emitted it once per member.

A guard just above (`:301`) rejects a Φ group whose members have different `off`: one instruction
cannot move X's chunk `iter+off_X` and Y's `iter+off_Y` at once.

**`copy_insts(phase, t)` (`:372`)** — the copies for one region, by phase:

| phase | slot | presence rule |
|---|---|---|
| `steady` | `Expr(var=outer_var, mod=d, add=o)` | all present |
| `prologue` | `cst((t + o − M) % d)` — a concrete ramp buffer | `o ≥ M − t` (deepest starts earliest) |
| `drain_bind` | the **steady symbolic** slot (under `Bind`) | `o ≤ M − 1 − t` (deepest drops first) |

The drain rule is §5.3.1 pt 2, and writing it the other way emits `copy_p(T)` — the out-of-bounds
ragged tail the staggered peel exists to remove.

A region-split copy is emitted as **one** `Inst` under a real `Loop(region_mode)` (`:404`) — §5.2's
"each op-class appears once in the tree, at its presence level". Unrolling it into `nreg` siblings
is the *view*, not the IR.

**`read_inst(op, kiter_note, shift, groups)` (`:444`)** — placement from `_read_placement`, coord
from `_shifted_coord` over the same `shift` (§6.7), so the read's coord, its register slot and its
shared `src_slot` cannot drift.

`_read_groups(op)` keeps all groups in one operand instruction. The target coordinate selects the
destination group after the common advance; each group then applies its own W to name the slot.
Splitting the instruction by W would incorrectly give one operand two different PLRs.

**`wmma_inst()` (`:474`)** — the leaf. Its `placement` is a **dict keyed by operand name**, each the
`_read_placement` at `dr=0`: the *unshifted* slot, i.e. the value a prior iteration's read-ahead
deposited. Carrying it in LoopIR rather than `None` is what lets GIR lower coord→name identically
for the wmma source and the `ds_read` dest, so def and use name the same VGPR under any loop order.

### 9.4 The rolled nest — `build_level`

```python
def build_level(li, suppress_ahead=False, with_wmma=True):        # emit.py:516
    mode, ext = inner[li]
    here = [op for op in reads if read_level[op.name] == mode]     # innermost presence mode
    body = []
    for op in here:
      for _grps in _read_groups(op):
        rd = read_inst(op, outer_var, shift=dr, groups=_grps)
        if suppress_ahead and <this read advances>:
            rd = Cond(Pred(Expr(terms=strides, add=shift), "<", n_pres),
                      then=[rd], kind="readahead_suppress")
        for ft in reversed(_first_touch_modes(op)):
            rd = Cond(Pred(Expr(var=ft), "==", 0), then=[rd], kind="first_touch")
        body.append(rd)
    body.append(build_level(li+1, …) if li+1 < len(inner) else wmma_inst())
    return _peel_first_touch(Loop(mode=mode, trip=ext, outer=False, bodies=[body]))
```

Two guards are attached here, and both are derived, not tabulated:

- **`readahead_suppress`** — a read-ahead read is out of bounds on the last peel step exactly when
  its **shifted flat position leaves the chunk**: `P + shift ≥ n_pres` over this operand's presence
  traversal. Same numbers as the advance itself. This once guarded on the reduction coord alone
  (`K_inner < n_s − dr`), which was the bound for the *old* advance; for a K-innermost order the
  crossing happens at `P ≥ 2` — the whole second free-mode tile — so the coord guard dropped
  in-bounds reads and kept the out-of-bounds ones. In the drain that is a `ds_read` of an LDS buffer
  no copy ever filled.
- **`first_touch`** — `_first_touch_modes(op)` (`:505`) is the non-presence inner modes at an ord
  position **outer** to the read's level. A read invariant over such a mode would re-issue once per
  its value; the guard issues it once. Derived from θ, no per-order rule.

### 9.5 Hoisting the first-touch guard into structure — `_peel_first_touch`

§5.1's multi-body form (#171). If a loop's body contains a first-touch guard on **that loop's own
mode**, split the trip at the boundary:

```
for m in range(4): if m==0: R ; Y      ⟶      bodies=[[R,Y],[Y]], body_ranges=((0,1),(1,4))
```

and the predicate disappears. `_resolve_ft(nodes, mode, taken)` rewrites the whole subtree, because
the guard sits at the READ's level — *inner* to the loop it tests — so this cannot be a local edit
at the guard site. Skipped when the trip is symbolic or 1; the guard then stays, which is correct,
just costlier.

### 9.6 The three regions

**Steady** (`:702`)

```python
iter_body = _order_sigma_c(copy_insts("steady") + nest())
loop = Loop(mode=outer_var,
            trip=Pred(Expr(var=outer_var), "<", Expr(var="T", add=-M)),
            outer=True, bodies=[iter_body])
```

`Loop.trip` is the **continuation predicate**, not a count (§5.2 line 339): `iter < T − M`, symbolic
in the runtime trip `T`. Deliberately not `K//DU` — that is a problem-dimension detail LoopIR must
not bake in. Note the skeleton lists copies *first* and σ_c moves them; copies-first is not a valid
σ (§6.8).

**Prologue** (`:715`) — `M` ramp steps then the read fill:

```python
for t in range(M):
    step = copy_insts("prologue", t)
    if step: pbody.append(Bind(mode=outer_var, value=cst(t - M), body=step))
pbody += fill_reads()
pro = Peel(kind="prologue", mode=outer_var, k=M, body=pbody)
```

Each ramp step is `Bind(iter = t − M)` — the paper's negative issue index `i = j − M`. `fill_reads`
(`:663`) walks `readahead_prologue` per read op-class and emits `_fill_read_inst`, which is
**direct/unshifted** and evaluates the placement at the pinned coord (`Placement.at`) so the buffers
are concrete ints rather than an unbound `buf(K_inner)%2`. Prologue fills target empty registers, so
their WAR awaits are dropped.

**Drain** (`:730`) — `M` straight-line steps:

```python
dr_iters = ceil(dr / red_ext)
for t in range(M):
    suppress = dr and (t >= M - max(1, dr_iters))
    dbody.append(_drain_step(t, suppress_ahead=suppress))
drain = Peel(kind="drain", mode=outer_var, k=M, body=dbody)
```

`_drain_step` (`:682`) wraps `σ_c(copies + nest)` in `Bind(iter = T−M+t)`. The `Bind` is what
replaced an older free-`iter` `Cond(iter%d == residue)` pin — which referenced an `iter` no scope
defined. Binding **defines** `iter` for the step, so the drain reuses the *steady* symbolic slot
form and prologue→steady→drain share one residue timeline.

Read-ahead is **not** dropped uniformly: a step keeps it while `t + dr_iters < M` (it feeds a later
drain step) and suppresses only in the trailing window. That is exactly the scaffold's NGLL → NLL
distinction; using `keep="drain"` for every step made both drain steps identical pure-drain
iterations and the NGLL generation was never prefetched.

### 9.7 The short-loop arm and the root guard

```python
for t in range(M):                                          # emit.py:755
    step = Bind(outer_var, cst(t), _order_sigma_c(copy_insts("prologue", t) + nest(...)))
    if t: step = Cond(Pred(Expr(var="T"), ">", t), then=[step], kind="short_step_validity")
    sbody.append(step)

return [Cond(pred=Pred(Expr(var="T"), ">=", M), then=[pro, loop, drain], els=sbody,
             kind="peel_validity")]                          # emit.py:783
```

The short arm is a **fully-modelled standalone body**, not a stub: the `M` peeled steps bound to
concrete `iter = t`, each `t > 0` guarded by `T > t` because this arm runs when `T < M`, so a step
`t ≥ T` would copy a chunk that does not exist. Step 0 is unguarded — `T > 0` is trivially true
wherever a reduction level exists, and the paper says a trivially-true guard carries no branch.

`kind` is **generic paper vocabulary only**; the LoopIR→scaffold mapping (`label`) is set downstream
by the GIR scaffold pass. At `M == 0` the guard and both peels are skipped and the steady loop is
returned alone.

### 9.8 Ordering invariant

Steps **8 → 9a → 9b** are not interleaved by accident: the tree and its σ_c are frozen before any
`Await` is attached, which is what lets §5.3.2's in-place extensions (the epilogue store) attach
their own obligations afterwards without disturbing the order (§14's soundness rule).

---

## 10. Validation (`validate.py`)

The P-* structural gate, run on **every** decode (not just in tests — that was #157):

- **P1** — prologue/drain must be `Cond`-wrapped, not at top level.
- **P3/P5** — *one* check, not two: no placement may reference a mode no enclosing scope binds
  (`_unbound_placement_refs`). Previously P3 only verified that a `Bind`, if present, pinned the
  right mode — so a bare `Inst` with a free-`iter` slot passed clean, which is the defect P3
  exists to catch.
- **P6** — the σ_c tail check, looking *through* a region `Loop` (else split copies stop being
  seen as the tail) and exempting the copy-first case.

Defects raise. They are decoder bugs, so raising is correct; `bridge.py` catches and reports them
as a comment so the observability path never kills codegen.

---

## 11. The boundary with GIR

`Lowering/loopir_to_gir.build_gir(theta)` consumes:

- the LoopIR tree from `emit_mainloop`;
- `_peel_depths` (for `M`) and `_prefetch_depth`;
- `theta.reduction_chunk_name()` — the level the generation timeline runs on;
- `reduction_names()`, `free_names(op)`, `inner_modes()` + extents;
- `movement_units()` (token granularity), `per_region_completion`, `fuse_groups`,
  `agent_distributed(op)`.

These land in `Program.meta`. **GIR never re-derives a schedule decision** — it lowers what θ
decided and then runs its own dataflow (tokens, swaps, gr_increments, LDS hazards, fences) over
the resulting CFG. The line is:

TDMSplit storage generations are keyed by `(operand, region)`, not only by operand. Each region
therefore owns its own `GenPhi` and `GenXfer`; a shared ref selects that generation from its
factorized region coordinate. `FrameMap` resolves each independently as
`token = (frame + gdelta) % ring`. TENSORCNT still counts every issued copy instruction in FIFO
order; separate region tokens prevent an unused region from becoming a false data dependency.

> θ decides *which op-class instances exist, at which level, in which order*.
> GIR decides *where the supporting machinery goes*.

Every GIR pass today respects it: not one creates or moves an op-class instance.

---

## 12. Invariants — the "must never disagree" list

1. **One level accessor.** Peel depth, `S_shared`, `src_slot` rotation, and the emitted
   `Loop`/`Bind` all key on `reduction_chunk_name()`. *(Task #210 generalizes this to per-level.)*
2. **One read-ahead depth per operand.** Prologue, steady shift, drain, ledger and lowering all
   read `Schedule.want(op)`. Groups have distinct W values, not distinct PLRs.
3. **One shift.** The read's coord, its register slot, and its shared `src_slot` come from the
   same `_readahead_shift` call.
4. **One movement authority.** `theta.movement_units()` — the emitter and the GIR token pass both
   read it.
5. **One depth map.** `_prefetch_depth` must equal `_shared_depth`, else the emitted copy slot
   modulus and the ledger disagree.
6. **Every `Await` is derived** from a ledger obligation via `_site_awaits`.
7. **Deterministic ledger order.** `build_ledger` returns a *sorted* list; the `set` is only for
   dedup. Set iteration follows string hashes, which `PYTHONHASHSEED` randomizes — the IR dump
   could not otherwise be diffed between runs.

---

## 13. Known gaps

| gap | task |
|---|---|
| Per-level peel and the boundary term are DERIVED but not EMITTED — `build_ir` refuses a coarser-level peel or a hoist (§6.6) because the enclosing `Loop` does not exist | **#113** |
| Negative `off` (deferral) is refused — Lemma 1 covers `δ ≥ 0` only, so the peel is not determinate there. Author question, not ours to construct | Q29 |
| `ρ.roles` (the `specialize` producer/consumer split) unpopulated — no shipping config needs it | #165 |
| `VgprPartition > 1` reaches θ but the group never reaches GIR | #118 |
| Write/store side unmodelled (blocks the epilogue and `1LDSBuffer`) | #116 |
| Persistent / grid-stride outer level | #113 |
| Tail loop (`K % DU ≠ 0`): needs a runtime-valued movement extent | #94 |
| Fused instance coord names only one member's region axis | #170 |
| Short-loop (`T < M`) arm not lowered to GIR | #183 |
| Uneven register partitions | #166 |

---

## 14. The 2026-08-14 paper update — what changes

The update supplied the construction §5.3.1 previously asserted but did not give.

**Per-level peel.** `M` becomes a **vector** `M_ℓ = max_p off(p, ℓ)`, one per level carrying an
offset; levels with `M_ℓ = 0` get no peel; `build_level` recurses and peels wherever `M_ℓ > 0`.

**The cross-level boundary term.** When `off(p, ℓ_outer) = δ > 0` for an op-class `p` whose *home
level* is inner to `ℓ_outer`, the outer offset hoists `p`'s leading `δ` inner-nest instances —
exactly `[inner-coord 0 .. δ)` — out of the current `ℓ_outer` iteration and into the **drain of the
previous `ℓ_outer` iteration** (at the first outer iteration, into the outer **prologue**).

The paper names our defect verbatim: *"a decoder that keys peel depth only on the innermost outer
level silently drops every coarser-level `off`."*

**Also changed:** Lemma 1's hypothesis splits into `T_ℓ ≥ δ` (determinacy) and strict `T_ℓ > δ`
(nonempty steady); the guard is `T >= M` with a zero-trip steady header at `T = M`; §5.2 now states
a store is an ordinary instance with `Λ.space` **reversed**; and §7 carves out *runtime-valued
movement extent* as a separate, narrower extension than the masked-remainder one.

### 14.1 What landed, and what did not

**Landed (#209, #210):**

- the retime offsets re-keyed by **(op-class, level)** — they now live on the placement each
  movement fills — so `off(read, iter)` is distinct from `off(copy, iter)`. Without this the
  boundary term is inexpressible: it
  asks for an outer-level offset on an op-class whose home is inner, which is exactly a read at the
  chunk level (§4.4).
- `_peel_depths` returns a **`PeelDepths`** with per-level `M_ℓ`, per-op-class `offs`, and the
  chunk-scoped `M` / `copy_off()` accessors the emitter uses (§6.4).
- `boundary_hoists` **derives** every cross-level term, with `op_class_level` supplying home levels
  (§6.5).
- `build_ir` **refuses** a coarser-level peel or a boundary hoist rather than silently dropping it
  (§6.6).

Behaviour on every θ the parameter path builds is **unchanged** — `translate` sets only
`off(copy, chunk)` and `off(read, substep)`, each at its op-class's own home level, so no new
`M_ℓ` and no hoist arises. 425 unit tests pass, including seven new ones covering the re-key, the
per-level vector, the staggered per-op-class offsets, the region-split false positive, and the
refusal path.

**Did not land — the EMISSION half.** Relocating instances into the previous outer iteration's
drain needs the enclosing `Loop(gtile)` that #113 supplies. That is why the guard raises instead
of emitting: with the derivation in place, the previously-silent drop is now a loud, precise
failure naming the levels and the offsets involved.

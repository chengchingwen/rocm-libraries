# LoopModel: what lives where

The decoder turns a host-project kernel description into LoopIR. Four stages, in order:

```
Solution ──adapter──▶ theta ──derive──▶ decisions ──emit──▶ LoopIR ──┬─▶ validate
                        │                   │                        └─▶ ledger check
                     (data)             (policy)              (instructions)
```

| stage | modules | holds |
|---|---|---|
| adapter | `adapter.py` | TensileLite vocabulary. The only place `MIWaveTile`, `TDMSplitA`, `PrefetchLocalRead` appear. |
| theta | `theta.py`, `ir.py` | the schedule as data: loop axes and their order, operands, their placement trajectories, buffer counts. |
| traversal | `traversal.py` | everything true of one operand's traversal: tile and register arithmetic, which axes it varies over, how its ring turns. |
| decisions | `schedule.py` | the register depths (`BufferDepths`) and everything keyed on them: prefetch steps, peel depth, preloads, instruction placement, and the `OperandPlan` table. |
| emit | `emit.py` | LoopIR instructions, and the `LoopProgram` result that carries them. |
| checks | `checks.py` | the ordering obligations, and whether the emitted tree honours them. |
| views | `render.py` | text, for humans and for the golden tests. |

## Placements are the model

An operand is a tensor identity plus a `Trajectory` — an ordered list of placements:

```python
Operand("A", "M_inner", trajectory=Trajectory(Global(), Shared(...), Fragment(...)))
```

Adjacency derives the rest. A `Movement` is a pair of neighbouring placements, and reads its
endpoints, role (`copy`/`read`/`store`), hardware counter, direction, vector width and coverage
from them — none of that is stored twice. Each placement owns its own facts:

| placement | owns |
|---|---|
| `Global` | the global-side vector width and coverage |
| `Shared` | the incoming copy's width, its region split, the LDS ring depth, and the `RegionLayout` |
| `Fragment` | register grouping and policy, fragment elements, read-instruction count, transfer coverage, agent group size, and the per-group register ring depths |

`Trajectory` answers the path queries by name — `.shared`, `.fragment`, `.shared_fill`,
`.fragment_fill`, `.shared_read` — so no caller re-derives "the register read hop" itself.

Per-movement retime offsets live on the destination placement (`Movement.offsets`); `Theta.off_at`
and `Theta.movement_offsets()` are the read-only projections of them.

## The layering is one-way

```
ir ──▶ theta ──▶ traversal ──▶ schedule ──▶ checks ──▶ emit ──▶ render
                     ▲
             adapter ┘   (Solution -> theta, feeding the front of the chain)
```

Every edge points right; there are no cycles and no function-level imports. Two facts make it hold:

- **`ir.py` owns the vocabulary** — `Space`, the counters, `TransferCoverage`, and the role
  constants (`COPY`/`READ`/`STORE`, `FORWARD`/`REVERSE`). Both `theta` and `traversal` need them,
  so neither has to import the other.
- **The register depths are decided in `schedule.py`** — above the prefetch arithmetic, not below
  it. The width policy needs the reuse verdict, and the peel depth needs the depths; that used to
  be a cycle. They are computed once and threaded down as an argument, which is why
  `peel_depths(theta, depths)` and `readahead_reach(theta, depths, plans)` take them rather than
  rebuilding them.

**The core names no host project.** Only `adapter.py` may import `Tensile.*`, `tdm_*` or
`Lowering`; `test_the_core_names_no_host_project` in the characterization suite enforces it by
parsing the source, so a stray `from ..tdm_split import ...` in `traversal` fails there rather
than at the first import from a standalone caller. `LoopModel/__init__.py` does not import the
adapter, so the core is importable without TensileLite present.

## What the ledger is

The **checklist of orderings the kernel must respect**, built from theta *before any instruction
exists*: "this read must wait for that copy to land", "this reload must come after the read it
overwrites". Each entry names a producer, a consumer, and the hardware counter that proves it.
After emit, `check_ledger_discharged` walks the instructions and confirms every entry was honoured;
anything left over is a decoder bug caught at build time rather than as a wrong result on hardware.

It is a build-time proof-obligation list. It is not a scheduler and it emits nothing.

## Model or decision?

Every function spelled `f(theta, operand, ...)` answers one of two questions. The test:

> **Does the answer change if PLR or the register budget changes?**

No → it is **model**, rendered by `render_operand_facts`.
Yes → it is **decision**, rendered by `render_schedule`.

```python
from Tensile.LoopModel.render import render_operand_facts, render_schedule
from Tensile.LoopModel.schedule import Schedule, build_S
depths, _floor = build_S(theta)
print(render_operand_facts(theta))              # what each operand IS
print(render_schedule(Schedule(theta, depths))) # what we decided for it
```

Regenerate the current split with
`characterization/LoopModel/classify_operand_functions.py` rather than trusting a count written
here; the structure ratchet tracks the total.

The register-group live-peak family are decisions that still sit in `traversal.py`, because
`group_width` (the decision that consumes them) is above the prefetch arithmetic while they
themselves are pure arithmetic over the traversal. Moving them up would put arithmetic above the
module that owns arithmetic. Flagged, not hidden.

## Where to change what

| you want to change | look at |
|---|---|
| how far ahead loads run (PLR) | `schedule.prefetch_steps_for`, then `traversal.prefetch_distance_for` |
| register buffer counts | `schedule.group_width`, `schedule.build_S`, `schedule.select_register_depths` |
| LDS buffer counts | `traversal.lds_buffers`, and `Shared.ring_depth` that feeds it |
| where a reload is placed | `traversal.reload_positions`, `move_reloads_after_last_use` |
| what the prologue preloads | `schedule.preloaded_tiles` |
| how deep the peel is | `schedule.peel_depths(theta, depths)` |
| a TensileLite parameter's meaning | `adapter.py` |

## How a Solution becomes a theta

`adapter.params_to_theta` is eight named steps, in order, each with an explicit interface:

| step | derives |
|---|---|
| `_read_params` | the Solution params, with defaults applied |
| `_read_splits` | `Splits` — TDMSplit as canonical region extents |
| `_build_ord` | `Ord` — the six-mode loop nest for this LoopOrder word |
| `_read_movements` | `MovementInputs` — vector widths, transfer coverage, instruction grouping |
| `_build_agent_assignment` | the agent-axis assignment, and the read configuration it implies |
| `_build_fragments` | `Fragments` — each operand's register fragment and LDS rings |
| `_build_operands` | the tensor identities and their placement trajectories |
| `_build_fuse` | the fused-copy groups |
| `_build_theta` | the retime offsets, the copy-side wave partition, and the assembled `Theta` |

## Names

The words in the code are the words here. The paper's symbols survive only in this table.

| paper / old name | direct name | what it means |
|---|---|---|
| `Rho` / `ρ` | `AgentAssignment` | which loop axes the hardware agents traverse instead of the coordinate |
| `Resort` | `AgentAxisAssignment` | one axis assigned to one agent level |
| `resort` (collection) | `axis_assignments` | the assignments that make up an `AgentAssignment` |
| `Phi` / `phi_width` | `instances_per_instruction` | movement instances merged into one instruction |
| `rho_span` | `agent_group_size` | how many sub-agents one merged instruction spans |
| `quantum` / `CoverageMap` | `coverage` / `TransferCoverage` | which tiles one instruction of a transfer covers |
| `fuse_groups` | `fused_copy_groups` | operands whose copies issue as one cooperative instruction |
| `region_agent_relative` | `RegionLayout.wave_relative` | one static read reaches every region across the workgroup |
| `Hop` | `Movement` | one leg of a trajectory, derived from adjacent placements |
| `hops` | `movements` | an operand's legs |
| `DepthMap` / `S` | `BufferDepths` | register ring depth per (operand, group) |
| `waves` | `wave_count` | agents in the workgroup |
| `off_map` | `Movement.offsets` | the retime offset, on the placement the movement fills |
| `dr_g` / `group_readahead_depth` | `prefetch_steps_for` | how many steps along the prefetch axis a load runs ahead |
| `shift` / `shift_for_depth` | `prefetch_distance_for` | the same lookahead as an offset in the operand's own load order |
| `anchor` / `readahead_anchors` | `reload_positions` | which point in the loop a reload must follow |
| `broadcast_outer` | `reloads_whole_set` | every register stays live across a full outer pass, so all reload at once |
| `readahead_prologue` | `preloaded_tiles` | tiles loaded before the loop so iteration 0 has data |
| `order_sigma_c` | `move_reloads_after_last_use` | the reorder that puts a reload after the reads it overwrites |
| `readahead_downgrade` | `prefetch_refusal` | why the requested prefetch could not be met |

The numeric `TDMFuse` code and its lookup tables stay adapter vocabulary. `theta`, `ord`, `LoopIR`,
`GIR`, `Operand`, `Fragment` and `ledger` stay — they are the domain's words, not jargon.

## Known rough edges

Tracked in the characterization suite's `BASELINE` ratchet, which only ever tightens. What is left
inside `LoopModel`:

- **`operand_functions`**, the count of free functions spelled `f(theta, operand, ...)`, is above
  its bound. The counter measures the opposite of what it was for: splitting a branchy function
  into named per-operand steps raises it, and so does merging duplicated derivations. Lowering it
  honestly means moving those answers onto the placements, which is the unfinished half of "put
  each fact on its owner".
- **`max_complexity`**, one function (`emit._guarded_read`) above the bound.

`Lowering`'s entries in the same ratchet are older than this refactor and are not in its scope;
most of them are in `Lowering/tool/`, which is diagnostics rather than the lowering path.

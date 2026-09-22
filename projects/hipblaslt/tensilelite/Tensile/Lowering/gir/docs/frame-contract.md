# The GIR frame contract

The CFG-free frame facts GIR hands StinkyTofu. Produced by
`Tensile/Lowering/gir/frame_contract.py`, consumed by
`stinkytofu/src/analysis/asm/GirFrameAnalysis.cpp`, and carried out of band as module plugin data
under the key `gir.frame_contract` — it never appears in the emitted assembly.

## Why it is text you can write

The two ends ship together, so the contract is not a negotiated wire format. Its only job is to be
**readable when something goes wrong** and **writeable when you want to test a shape by hand** —
`stinkytofu/tests/filecheck/Inputs/*.contract` are hand-written, and
`stinkytofu-opt --gir-frame-contract=<file>` loads them.

There is **no version**. A contract that a parser does not understand is a build error, not a
compatibility question: the producer and the parser are in the same repository and are built
together. The first line is a marker, not a revision.

## Grammar

One record per line: a tag, some positional words, then `key=value` fields in any order.
Blank lines are skipped and `#` starts a comment. An **absent optional field means its default**,
so there are no sentinel values to decode.

```
gir-frame-contract

gen <id>     ring=<n> [entry=<n>] [advance=<n>]
action <id>  kind=<copy|read|fence|wmma|waitcnt|other> [anchor=<action-id>]
access <action-id>  <read|write>  operand=<name>  ring=<n>
                    [gen=<id>] [gdelta=<n>] [abs=<n>] [cross] [regions=<a,b/c,d>]
incoming  dst=<action-id> src=<action-id> gen=<id> value=<n>
transfer  dst=<action-id> src=<action-id> gen=<id> delta=<n>
rel <fence-action-id>  <RAW|WAR|WAW>  producer=<action-id> consumer=<action-id> [gap=<n>]
```

| record | what it states |
|---|---|
| `gen` | a rotating buffer: how deep (`ring`), the phase it enters at, how far one trip advances it |
| `action` | one schedulable unit. `anchor` is the action whose position pins its block; it defaults to the action itself |
| `access` | one shared-memory touch. `gen` absent = not bound to a generation; `abs` absent = the generation is relative rather than pinned; `cross` = the operand is agent-distributed; `regions` absent = the whole operand |
| `incoming` | a generation arriving at `dst` from `src` pinned to an absolute `value` |
| `transfer` | the same edge, but forwarding the predecessor's phase shifted by `delta` |
| `rel` | a hazard the fence separates, both ends named as actions, `gap` generations apart |

`regions` is `/`-separated axes, each a `,`-separated sorted value list: `0,1/2,3` is two axes.

Operand names are written bare, so they must be plain tokens (`[A-Za-z_][A-Za-z0-9_]*`). The
encoder raises rather than escaping — an escape would put an encode/decode pair in two languages
to serve a name no kernel has produced.

## Example

```
gir-frame-contract

gen 0  ring=2 entry=0 advance=1

action 5  kind=copy anchor=5
access 5  write  operand=A  ring=2  gen=0  regions=0,1/2,3

action 8  kind=read anchor=5
access 8  read  operand=A  ring=2  gen=0  gdelta=1

incoming  dst=5 src=3 gen=0 value=1
transfer  dst=5 src=3 gen=0 delta=1

rel 9  RAW  producer=5 consumer=8  gap=2
```

## Keeping the two parsers in step

`frame_contract.parse_contract` is the reference implementation and round-trips
(`render_contract(parse_contract(text)) == text`, asserted in
`test_frame_contract_has_dense_actions_and_no_gir_cfg_edges`). The C++ parser must accept the same
grammar; `stinkytofu/tests/unit/asm/GirFrameAnalysisTest.cpp` and the
`gir_frame_waitcnt_*.stir` FileCheck tests pin its behaviour.

A field added on one side and not the other is silent in the direction that matters: a reader that
asks for a key nobody writes gets its default and does nothing. When adding a field, add it to
both parsers and to a test in the same change.

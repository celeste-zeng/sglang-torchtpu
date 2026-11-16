# GLM5.2 draft snapshot — do not merge

This is an inventory for extracting independent work, not a proposed monolithic
feature merge. Original branch `epic/glm52-channelwise-port-20261113` at
`ca8dbe8e62` remains intact; draft replays its 405 commits on main
`97afaea93765ce3546806481cab3b5c3bc0beb09`. Two untracked request-table drafts
remain only in the original worktree and are not part of this PR.

## Review entry points

1. `KERNEL_ROUTE_COMPARISON.md`: exact private-JAX vs Torch source routes.
2. `nodump_acceptance_checkpoint.json`: honest final acceptance and remaining gaps.
3. `README.md`: current status first; subsequent experiment history is archival.

## Code inventory relative to main

Only four production files differ:

- `sglang_tpu/glm52_channelwise_fp8.py`: explicit opt-in checkpoint ABI, output-channel
  scales, dense and expert quant-method adapters. This is NOT JAX kernel parity.
- `sglang_tpu/tpu_platform.py`: TPU FP8 config factory delegates ordinary layouts
  back to upstream and enables explicit specialized layout.
- `sglang_tpu/ep_local_expert_reader.py`: exact GLM class opt-in to existing safe
  routed-expert range selection. Staged reads remain guarded by existing exclusions.
- `sglang_tpu/attention/pallas/mla_sparse_qblock_tokens.py`: separate experimental
  qblock row-budget adjustment. **Not selected in successful default E2E.** Do not
  bundle it into the loader extraction or claim E2E validates it.

Two unit files cover ABI and reader changes. Everything under this benchmark
folder is experiment tooling/evidence, not supported serving API or registered
model certification. Historical dump/observer/custom-admission code stays archived
for provenance, must NOT be installed as the runtime solution. Prefer native HTTP
routing, logs and request metadata for future work.

## Proposed extraction order

| Work item | Scope | Independent acceptance |
|---|---|---|
| 1. Checkpoint ABI | plugin FP8 factory + channelwise scale loading + unit tests | ordinary FP8 unchanged; exact dense/expert scale/shard oracles; checkpoint end-to-end |
| 2. Host loading | exact GLM EP eligibility + staged host range prep | source index hash, exact bytes/ranges, rank coverage, bounded memory, all-rank startup |
| 3. Dense precision alignment | compare JAX W8A16 dense with current Torch W8A8 route | matched layer inputs/weights/scales, logits/outputs before speed claims |
| 4. Attention/indexer parity | per-query Torch vs private DSA FP8-QK/PV kernel families | explicit routing/operand contract, prefill/decode/cache boundary numerics |
| 5. MoE communication/kernel parity | ordinary Torch GMM/SC vs JAX fused EP/shared experts | dispatch/combine correctness, collectives, shared expert and reduction ownership |
| 6. Qblock experiment | isolated row-budget patch only | its own compile/layout and numerical tests, never inherited E2E evidence |
| 7. Native benchmark/accuracy | minimal no-dump runner and clients | chat sanity + reference accuracy; matched cache/input/active cadence; host timing only |

## Evidence boundaries

Original pinned sources passed 20/20 actual-chat checks (`exp-k0nnadpb6q`), all
native DP0..15 response receipts, and 16 cold 12288-input/16-output requests in
19.3465 host seconds (`exp-r188nvdf96`). Six full 2048-token chunks per DP were
logged. This is not the reference C32-submitted/active16/49152-prefix/12-round
workload, nor a full numerical accuracy certification. Cold greedy synthetic-token
output differs from three internally-identical cached repeats: unresolved.

The rebased head has NOT been rerun on TPU. Main changes can change behavior;
source replay without conflicts is not runtime equivalence.

## Rebase validation (local)

- Shared-core TPU debt ratchet: PASS, no new core TPU branch/import.
- All changed Python sources (57) and JSON evidence/configs (101): parsed.
- Embedded Python runner: parsed.
- Stdlib native cache-hit HTTP/NPY rejection selftest: PASS.
- Full Torch/loader/OOT unit tests: NOT RUN locally, `torch` unavailable.
- Whole-tree diff whitespace check finds historical raw-log whitespace; retain
  raw evidence rather than silently rewriting original logs. Review new code
  independently. No hardware, dump, or observer execution for this preparation.

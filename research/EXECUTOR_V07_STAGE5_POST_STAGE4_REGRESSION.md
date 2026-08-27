# Executor V0.7 Stage 5 Post-Stage 4 Regression Panel

- Date: 2026-08-27
- Status: **ACCEPT current post-Stage-4 revision for archive acceptance; exact archive identity updated below**
- Start/final HEAD: `b9c88ff517a2d8ab6c3ae08ebf0e514ee6a0b200` (`executor: sequence yield water before harvest`)
- Branch: `executor-v07-fixed-plan`
- Behavior changes: none in this packet; raw panel and audit JSON are ignored local evidence.

## Reproduction and provenance

The one heavy panel was run exactly once on the accepted current HEAD. The client
call timed out while the process continued; the resulting JSON was verified as a
complete 24-game artifact before the lingering wrapper was stopped.

```text
python -m tools.run_executor_v07_panel --checkpoint "C:\Users\liuyi\VSCodeProjecs\Kaggriculture\Kaggriculture\artifacts\local\bc-v1-E\best.pt" --seeds 7,17,42,123,2026,1013,1022,1003,1026,1011,1024,1019 --seats 0,1 --opponent PASS --opening standard_mixed --prior-debt-suppression on --aggressive-sell-all --turn-trace --backend fast --output "artifacts/executor-v07-stage5-post-stage4-aggressive-sell-all.json" --label "executor-v07-stage5-post-stage4-aggressive-sell-all"
python -m tools.audit_executor_v07_care_fertilize --artifact "artifacts/executor-v07-stage5-post-stage4-aggressive-sell-all.json" --output "artifacts/executor-v07-stage5-post-stage4-care-fertilize-audit.json" --expected-repo-sha b9c88ff517a2d8ab6c3ae08ebf0e514ee6a0b200 --expected-checkpoint-sha256 f4b029d3e463aba1db0544377d0d616e3de94aa6cc469d3446f018dddd8f6bf2 --seeds 7,17,42,123,2026,1013,1022,1003,1026,1011,1024,1019 --seats 0,1 --backend fast --opening standard_mixed --day-start 4 --day-end 29
```

| item | value |
|---|---|
| panel artifact | `artifacts/executor-v07-stage5-post-stage4-aggressive-sell-all.json` |
| panel file SHA-256 | `c3ca62d27dd670dfeea09450f9691f277be4315a99d3361a2053da7046696e6a` |
| panel internal content SHA-256 | `ae7896edfbb09471ed61ded401cde2e4fcdc26f949ac0b3b0603197c35d2acb2` |
| audit artifact | `artifacts/executor-v07-stage5-post-stage4-care-fertilize-audit.json` |
| audit file SHA-256 | `ab301a95c61647032a32c8e51899bce9ef0c388199b274c751a67b2019864a5b` |
| audit report SHA-256 | `3b1117b1401e768b6cd95ac2c3a9a21529d0bf8581f644438f8abf69d9d1a97b` |
| checkpoint | BC-E `best.pt`, epoch 27 |
| checkpoint SHA-256 | `f4b029d3e463aba1db0544377d0d616e3de94aa6cc469d3446f018dddd8f6bf2` |
| engine/backend | Kaggriculture 1.32.7 / fast / 719 transitions |
| opening/opponent | `standard_mixed` / PASS |
| flags | prior-debt suppression ON, aggressive selling ON, optional spare watering OFF, turn trace ON |

Validation: 24/24 exact Cartesian cases, no duplicates or omissions, all games
`complete` with 719 transitions and `DONE/DONE` status. The pre-behavior artifact
was present and valid: revision `8f716bec67caa249ab59864547858c2f7dfcb4ca`, file
SHA-256 `d68d8dc693895eb4bfffeb0b9e053d2c7e5c5498182fa5a8ee5c6acc3909b97`,
internal SHA-256 `ce188ee310f101c6f4e2c75085a48e01b427dd32ba32716f2a3cb5b3de84fe34`.

## Complete bank table and pre-behavior deltas

The comparison artifact is the actual 24-game panel at `8f716bec`, not the older
aggregate reference.

| seed | seat | post-Stage4 bank | pre-behavior bank | delta |
|---:|---:|---:|---:|---:|
| 7 | 0 | 0 | 61572 | -61572 |
| 7 | 1 | 0 | 54439 | -54439 |
| 17 | 0 | 0 | 53146 | -53146 |
| 17 | 1 | 0 | 54400 | -54400 |
| 42 | 0 | 0 | 58267 | -58267 |
| 42 | 1 | 0 | 60340 | -60340 |
| 123 | 0 | 0 | 55220 | -55220 |
| 123 | 1 | 0 | 55220 | -55220 |
| 2026 | 0 | 0 | 67342 | -67342 |
| 2026 | 1 | 0 | 65468 | -65468 |
| 1013 | 0 | 0 | 66274 | -66274 |
| 1013 | 1 | 0 | 65271 | -65271 |
| 1022 | 0 | 0 | 60141 | -60141 |
| 1022 | 1 | 0 | 57216 | -57216 |
| 1003 | 0 | 0 | 59740 | -59740 |
| 1003 | 1 | 0 | 54871 | -54871 |
| 1026 | 0 | 0 | 62642 | -62642 |
| 1026 | 1 | 0 | 61837 | -61837 |
| 1011 | 0 | 0 | 65782 | -65782 |
| 1011 | 1 | 0 | 65782 | -65782 |
| 1024 | 0 | 0 | 64655 | -64655 |
| 1024 | 1 | 0 | 65889 | -65889 |
| 1019 | 0 | 0 | 66546 | -66546 |
| 1019 | 1 | 0 | 56614 | -56614 |

| panel | mean | median | min | max | <1k | <10k |
|---|---:|---:|---:|---:|---:|---:|
| post-Stage4 current | 0.0 | 0 | 0 | 0 | 24 | 24 |
| actual pre-behavior artifact (`8f716bec`) | 60,778.1 | 60,956 | 53,146 | 67,342 | 0 | 0 |
| user-provided older aggregate | 60,230.3 | 58,334 | 51,692 | 69,070 | 0 | 0 |

Current minus actual pre-behavior aggregate is mean `-60,778.1`, median
`-60,956`, min `-53,146`, max `-67,342`, with 24 additional `<1k` and
`<10k` cases. Current minus the user-provided older aggregate is respectively
`-60,230.3`, `-58,334`, `-51,692`, and `-69,070`, also with 24 additional low-tail
cases.

## CARE/FERTILIZE audit

The audit has 4,992 rows: 24 games × 26 executor days (`d4`–`d29`) × three CARE
species plus five fertilizer crops. `requested` is the manager plan,
`feasible_projected` is the projected plan, `eligible` is projection eligibility,
`submitted_assigned` counts turn-trace interaction assignments, and
`observed_completed` is the existing board-state diagnostic field.

| family/entity | requested | feasible | eligible | assigned | observed | signed shortfall | requested>0 / observed=0 | material |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CARE/COW | 468 | 468 | 1872 | 0 | 0 | 468 | 156 / 156 | 156 |
| CARE/GOOSE | 0 | 0 | 0 | 0 | 0 | 0 | 0 / 0 | 0 |
| CARE/SHEEP | 1584 | 1582 | 1676 | 0 | 0 | 1584 | 617 / 617 | 617 |
| FERTILIZE/CARROT | 0 | 0 | 0 | 0 | 0 | 0 | 0 / 0 | 0 |
| FERTILIZE/MELON | 0 | 0 | 5462 | 0 | 0 | 0 | 0 / 0 | 0 |
| FERTILIZE/STRAWBERRY | 0 | 0 | 80 | 0 | 0 | 0 | 0 / 0 | 0 |
| FERTILIZE/TOMATO | 0 | 0 | 0 | 0 | 0 | 0 | 0 / 0 | 0 |
| FERTILIZE/WHEAT | 0 | 0 | 2836 | 0 | 0 | 0 | 0 / 0 | 0 |

Aggregate CARE: requested 2,052, assigned 0, observed state count 0, signed
shortfall 2,052; 773 rows are both requested-positive and zero-observed, all
material. Aggregate FERTILIZE: requested 0, assigned 0, observed 0, shortfall 0.
There are 2 projection-infeasible material rows and 771 unresolved material rows;
none is classified as a proven executor failure. The ignored audit JSON contains
all per-game/per-day rows and classification evidence.

These are state metrics, not accepted-action counts: CARE is overwritten at the
next-day boundary and fertilizer state persists on plants. The engine exposes no
per-action acceptance/failure signal, so the audit cannot promote these fields to
completion proof.

## Runtime signals and anomalies

- Fallback executor errors: 0; per-day errors: 0; engine status anomalies: 0.
- Illegal/ineffective action signal: unavailable in all 24 games; the engine
  observation does not expose per-action validity.
- Unaffordable market orders: 27,151; unresolved generator entries: 428.
- Prior-debt suppression: 600/624 day records; current suppression: 624/624.
- Turn traces: 1,728 feed-shortage turns and 1,728 starvation turns.
- Concrete lifecycle cascade: 72 animal-decrease events, 120 animals lost, at
  day-end hours on d4–d6 (24 events on each day; d5 accounts for 72 lost units).
- Current aggregate assignment families contain no CARE, FERTILIZE, WATER, or
  HARVEST assignments; the executor remained in survival/debt handling.

## Regression diagnosis

The first divergence is immediately at the d4 handoff, before any Stage 4
water-before-harvest dependency can execute, and it is visible in the Stage 3
sell ledger for seed 7/seat 0:

| d4 bin-0 ledger | current post-Stage4 | actual pre-behavior |
|---|---:|---:|
| FERTILIZER override submitted | 0 of 5 | 4 of 5 |
| WHEAT override submitted | 1 | 5 |
| d4h0 money | 46 → 15 | 46 → 288 |
| d4h0 survival state | 5 feed shortage / starving | 5 feed shortage, then feed shortage clears at d4h1 |

The current run therefore cannot buy enough WHEAT, stays in starvation and feed
shortage preemption, accumulates 27,151 unaffordable signals, performs no
productive CARE/FERTILIZE/WATER/HARVEST interactions, and reaches zero bank in
all 24 games. The direct cash-preservation mechanism and the timing point to
Stage 3 commit `fa57313` (aggressive-sell fertilizer exclusion), not Stage 4,
as the likely cause. Stage 4 is not independently exonerated or rejected by
this run because the starvation cascade masks its productive workload; there is
no evidence that its dependency itself fired in the completed trace.

### Root recommendation

Reject the current combined revision for archive acceptance. For the root decision,
make the exact clean correction `git revert fa57313` (preserving the Stage 4
commit for a subsequent isolated rerun), then repeat the fixed Stage 5 panel
before changing the archive expected bank/fingerprint. No revert, archive update,
or behavior mutation was performed here.

## Validation and scope

- Programmatic checks: exact 12×2 case set, no duplicates/missing cases, JSON
  validity, 719 transitions, `DONE/DONE` statuses, bank aggregates, and per-game
  deltas all cross-checked against both raw artifacts.
- Focused reporter/audit tests: no reporter code changed; `test_audit_executor_v07_care_fertilize.py`
  was run for contract verification.
- Git scope: only this tracked note is intended for commit; both generated JSON
  artifacts remain ignored. No push.
- Unresolved risk: Stage 4 same-day lifecycle sequencing still needs a clean
  post-Stage-3-correction panel; the current panel cannot establish its isolated
  economic effect.

## Isolated Stage 4 rerun after Stage 3 revert

- Date: 2026-08-27
- Source revision: `7204103fc2ca6e351cc2a95506c961977d199bb2` (`revert: reject Stage 3 aggressive fertilizer retention`)
- Stage 4 commit retained: `b9c88ff517a2d8ab6c3ae08ebf0e514ee6a0b200`
- Result: **ACCEPT Stage 4 for archive acceptance by root judgment; exact archive pin update is recorded below**
- Checkpoint SHA-256: `f4b029d3e463aba1db0544377d0d616e3de94aa6cc469d3446f018dddd8f6bf2`

The exact fixed panel was run once with the same 12 seeds × 2 seats, PASS
opponent, `standard_mixed`, fast backend, 719 transitions, prior-debt
suppression ON, aggressive selling ON, and turn trace ON.

```text
python -m tools.run_executor_v07_panel --checkpoint "C:\Users\liuyi\VSCodeProjecs\Kaggriculture\Kaggriculture\artifacts\local\bc-v1-E\best.pt" --seeds 7,17,42,123,2026,1013,1022,1003,1026,1011,1024,1019 --seats 0,1 --opponent PASS --opening standard_mixed --prior-debt-suppression on --aggressive-sell-all --turn-trace --backend fast --output "artifacts/executor-v07-stage5-isolated-stage4-post-stage3-revert.json" --label "executor-v07-stage5-isolated-stage4-post-stage3-revert"
python -m tools.audit_executor_v07_care_fertilize --artifact "artifacts/executor-v07-stage5-isolated-stage4-post-stage3-revert.json" --output "artifacts/executor-v07-stage5-isolated-stage4-post-stage3-revert-care-fertilize-audit.json" --expected-repo-sha 7204103fc2ca6e351cc2a95506c961977d199bb2 --expected-checkpoint-sha256 f4b029d3e463aba1db0544377d0d616e3de94aa6cc469d3446f018dddd8f6bf2 --seeds 7,17,42,123,2026,1013,1022,1003,1026,1011,1024,1019 --seats 0,1 --backend fast --opening standard_mixed --day-start 4 --day-end 29
```

| item | value |
|---|---|
| panel artifact | `artifacts/executor-v07-stage5-isolated-stage4-post-stage3-revert.json` |
| panel file SHA-256 | `bd9dd3ef58f7b87f33aa2fcad980aa826a6a1ccfcf94012bb491927853027546` |
| panel internal SHA-256 | `8862450c5036b5791c1a7d843220fe274ffb73b90022ed7856e892611f3db159` |
| audit artifact | `artifacts/executor-v07-stage5-isolated-stage4-post-stage3-revert-care-fertilize-audit.json` |
| audit file SHA-256 | `3f406eff7244015eef302fda6b2d22118ef49ebbc2e86879702018b683240c6e` |
| audit report SHA-256 | `ca62fc6b1ba6b0b145778023bf15a3a0e9c96fc16f927e9ed0e06ff7ce2a33b3` |

Validation passed: 24/24 exact cases, no duplicates or omissions, every game
`complete`, 719 transitions, and `DONE/DONE`.

### Isolated bank table

| seed | seat | isolated Stage 4 bank | pre-behavior bank | delta |
|---:|---:|---:|---:|---:|
| 7 | 0 | 50655 | 61572 | -10917 |
| 7 | 1 | 47290 | 54439 | -7149 |
| 17 | 0 | 63139 | 53146 | +9993 |
| 17 | 1 | 63956 | 54400 | +9556 |
| 42 | 0 | 58652 | 58267 | +385 |
| 42 | 1 | 64092 | 60340 | +3752 |
| 123 | 0 | 49304 | 55220 | -5916 |
| 123 | 1 | 49304 | 55220 | -5916 |
| 2026 | 0 | 69206 | 67342 | +1864 |
| 2026 | 1 | 71315 | 65468 | +5847 |
| 1013 | 0 | 72512 | 66274 | +6238 |
| 1013 | 1 | 72425 | 65271 | +7154 |
| 1022 | 0 | 57309 | 60141 | -2832 |
| 1022 | 1 | 59247 | 57216 | +2031 |
| 1003 | 0 | 66927 | 59740 | +7187 |
| 1003 | 1 | 68461 | 54871 | +13590 |
| 1026 | 0 | 61348 | 62642 | -1294 |
| 1026 | 1 | 63732 | 61837 | +1895 |
| 1011 | 0 | 70073 | 65782 | +4291 |
| 1011 | 1 | 70073 | 65782 | +4291 |
| 1024 | 0 | 74151 | 64655 | +9496 |
| 1024 | 1 | 68459 | 65889 | +2570 |
| 1019 | 0 | 67632 | 66546 | +1086 |
| 1019 | 1 | 66953 | 56614 | +10339 |

| panel | mean | median | min | max | <1k | <10k |
|---|---:|---:|---:|---:|---:|---:|
| isolated Stage 4 | 63592.3 | 65509.5 | 47290 | 74151 | 0 | 0 |
| actual pre-behavior (`8f716bec`) | 60778.1 | 60956 | 53146 | 67342 | 0 | 0 |
| user older aggregate | 60230.3 | 58334 | 51692 | 69070 | 0 | 0 |

Against the actual pre-behavior artifact, the isolated panel improved mean by
`+2814.2`, median by `+4553.5`, and max by `+6809`, but reduced min by `-5856`.
Against the user aggregate, differences are mean `+3362.0`, median `+7175.5`,
min `-4402`, and max `+5081`; low-tail counts remain zero.

### Isolated diagnostics

- CARE: requested `4,579`, assigned `2,931`, observed state count `0`; COW
  requested `2,857`, SHEEP `1,722`, GOOSE `0`.
- FERTILIZE: requested `1,241`, assigned `604`, observed state count `2,741`;
  all requested fertilizer work was STRAWBERRY.
- CARE requested-positive/zero-observed/material rows: COW `589/589/589`,
  SHEEP `582/582/582`; GOOSE `0/0/0`.
- FERTILIZE requested-positive/zero-observed/material rows: STRAWBERRY
  `208/0/42`; all other crops `0/0/0`.
- Audit classifications: 45 manager-infeasible, 1,168 unresolved. State
  completion fields retain the Stage 2 limitation and are not accepted-action
  ledgers.
- Fallback errors: 0; day errors: 0; engine status anomalies: 0; unaffordable
  market orders: 0; unresolved generator entries: 613.
- Prior-debt suppression: 564/624 days; current suppression: 624/624 days.
- Feed-shortage/starvation turns: 738/264; animal-decrease events and animals
  lost: 0/0; all games remained above the low-tail thresholds.

### Stage 4 diagnosis

There are six negative per-game deltas, including four losses of at least 5,000,
concentrated across both seats of seeds 7 and 123. This meets the packet's
proportional major-regression definition despite the aggregate improvement.

For every negative case, the first assignment divergence is d4h18 at tile
`(1,4)`: pre-behavior assigns `HARVEST:1,4`, while Stage 4 assigns
`WATER:1,4`; the next turn then harvests. The first state divergence is the
same transition's available wheat, current `0` versus pre-behavior `3`, showing
that the dependency changes harvest timing and downstream liquidity. Aggregate
traces show 9,622 WATER and 5,304 HARVEST assignments, versus 9,537 and 5,482
pre-behavior; same-tile pairs are WATER-first 2,340 versus 2,042 and
HARVEST-first 1,220 versus 1,461. The trace does not expose the generated
dependency field or engine acceptance ledger, so these counts are behavioral
ordering evidence rather than proof of engine action acceptance.

### Isolated recommendation (superseded by root acceptance)

The original conservative writer recommendation was to reject Stage 4 for archive
acceptance and recommend `git revert b9c88ff`. Root superseded that threshold:
the six negative cases are labor/cash tradeoffs, not feed/starvation failures,
and no concrete distinguishing feed exception or unconstrained cash forecast is
justified. The original combined Stage 3+4 all-zero artifact and rejection
evidence above remain unchanged; Stage 3 fertilizer retention remains rejected
and cleanly reverted at `7204103`.

## Stage 3 / issue #14C disposition

- Status: **REJECTED and reverted**; do not reintroduce the fertilizer-retention
  experiment without manager-level policy.
- Evidence: the post-Stage-4 24-game panel produced bank `0` in all 24 games
  versus the pre-change mean bank of `60,778.1`; the first divergence was d4h0.
  It also recorded 27,151 unaffordable orders, 1,728 feed-shortage turns,
  1,728 starvation turns, and 120 animals lost, with no fallback, day, or
  status errors.
- Rationale: retaining FERTILIZER under the current aggressive diagnostic
  removes required liquidity and causes a total starvation cascade. No allowed
  executor-only mechanical workaround exists under the bans on cash governors,
  generic reserves or release rules, strategic heuristics, manager-policy
  changes, and tuning pile-ons.

## Final Stage 5 verdict and exact archive compatibility identity

- Date: 2026-08-27
- Current source revision used to build: `11ecead2d5efe8bf87fc0da533c739e344d7eaa6`
- Verdict: **ACCEPT** the retained Stage 4 lifecycle sequencing and update the
  exact raw-loaded archive reference. Stage 3 fertilizer retention remains
  **REJECTED and reverted** at `7204103` because the combined experiment drove
  all 24 banks to zero; aggressive mode again sells FERTILIZER.
- The complete isolated Stage 4 evidence above is preserved: mean `63,592.3`
  vs actual pre-behavior `60,778.1` (`+2,814.2`), median `65,509.5` vs
  `60,956` (`+4,553.5`), six negative per-game deltas, no `<1k`/`<10k` cases,
  and no errors, status anomalies, unaffordable orders, or animal losses.
  Independent paid review classified the negative cases as labor/cash
  tradeoffs, not feed/starvation failures. No concrete distinguishing feed
  exception exists, and a cash forecast would violate the stated constraints;
  no such exception or forecast is introduced.

### Old-to-new exact identity

| revision | archive | archive SHA-256 | official bank | action fingerprint |
|---|---|---|---:|---|
| pre-behavior `8f716bec` | `artifacts/local/submissions/bc-e-v07.tar.gz` | `4ccfcf25d30465661c912626a5d029210897ec5855c3dc2b55db2cdfd1a7d6cf` | `54,439.0` | `516fab6d316b76e8b93fce3b4d185e49b2df53aa742be6558574563c1929dc40` |
| current `11ecead2` | `artifacts/local/submissions/bc-e-v07.tar.gz` | `c12218ac1010c894ed22fd065049a290d03555c9f44ad0d6cc667fa52ee13de2` | `47,290.0` | `a38bf47884e5e6e89c2d77f7aab07819f3559e898af40372942460693c8b6afc` |

The current archive was rebuilt with the existing deterministic builder from
the authorized BC-E `best.pt` (checkpoint SHA-256
`f4b029d3e463aba1db0544377d0d616e3de94aa6cc469d3446f018dddd8f6bf2`). It has
50 members, including the six local runtime packages, root `main.py`,
`best.pt`, and `submission_manifest.json`; archive/checkpoint remain ignored.

Final exact verification used repository-local `.venv` with
`kaggle-environments==1.32.7`, raw `get_last_callable`, strict mode, fresh
extraction, repository-root import isolation, and a full status scan. Result:
bank `47290.0`, 720 status-history entries, zero anomalies, 719 candidate
actions, repository-root source paths absent, official provenance passed, and
the new fingerprint matched. Focused submission tests, including loud omitted
`fast_env` failure, and Ruff passed.

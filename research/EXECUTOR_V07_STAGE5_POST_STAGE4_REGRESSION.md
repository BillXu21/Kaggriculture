# Executor V0.7 Stage 5 Post-Stage 4 Regression Panel

- Date: 2026-08-27
- Status: **REJECT current Stage 3+4 revision for archive acceptance; do not update the archive pin**
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

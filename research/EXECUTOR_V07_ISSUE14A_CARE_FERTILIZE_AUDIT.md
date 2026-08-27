# Issue 14A CARE/FERTILIZE Audit

- Date checked: 2026-08-27
- Source: required aggressive-sell-all 24-game panel; ignored raw panel and audit JSON are local evidence
- Source version/commit: `8f716bec67caa249ab59864547858c2f7dfcb4ca`
- Checkpoint: BC-E `best.pt`, epoch 27, SHA-256 `f4b029d3e463aba1db0544377d0d616e3de94aa6cc469d3446f018dddd8f6bf2`
- Confidence: `CONFIRMED_EXPERIMENT` for recorded observations; `UNKNOWN` for action acceptance

## Reproduction

```text
python -m tools.run_executor_v07_panel --checkpoint "C:\Users\liuyi\VSCodeProjecs\Kaggriculture\Kaggriculture\artifacts\local\bc-v1-E\best.pt" --seeds 7,17,42,123,2026,1013,1022,1003,1026,1011,1024,1019 --seats 0,1 --opponent PASS --opening standard_mixed --prior-debt-suppression on --aggressive-sell-all --turn-trace --backend fast --output artifacts/executor-v07-issue12-stage1-aggressive-sell-all.json
```

- Panel: 24/24 complete games, 719 transitions each, final statuses `DONE/DONE`.
- Raw artifact: `artifacts/executor-v07-issue12-stage1-aggressive-sell-all.json`.
- Raw artifact internal content SHA-256: `ce188ee310f101c6f4e2c75085a48e01b427dd32ba32716f2a3cb5b3de84fe34`.
- Raw artifact file SHA-256: `d68d8dc693895eb4b9fffeb0b9e053d2c7e5c5498182fa5a8ee5c6acc3909b97`.
- Audit export: `artifacts/executor-v07-issue12-stage2-care-fertilize-audit.json` (ignored); its `report_sha256` and all per-day rows are the machine-readable ledger.
- Audit report content SHA-256: `b714c1aef97ed4d95bddcd7d4c7edac8e9a9b0a57b3c5b9c924a975b96537afa`; audit file SHA-256: `009ab81b5d41306a73cb330b288202d7c4456ef8612d21016ffd60df6b225141`.

## Method and field definitions

The postprocessor emits 4,992 rows: 24 games × 26 executor days (`d4`–`d29`) × eight entities (three CARE species and five FERTILIZE crops). `requested` is the manager plan; `eligible` is projection eligibility; `feasible_projected` is the mechanical projected plan; `submitted_assigned` counts turn-trace CARE/FERTILIZE interaction assignments only; and `observed_completed` is the existing diagnostic board-state count. A material shortfall is explicitly any signed `requested - observed_completed >= 1`; every signed raw value is retained.

The observed field is not an accepted-action ledger. CARE state is written into the prior day record again at the next-day boundary (`executor_v0/agent.py:_new_day`), while fertilizer state persists on plants (`executor_v0/agent.py:_board_counts`). The panel also records `illegal_actions.available=false`, so an assignment cannot be promoted to engine acceptance. Therefore raw shortfall rows are exhaustive *apparent diagnostic comparisons*, not proof of executor failure.

## Totals

| family/entity | requested | feasible projected | submitted/assigned | observed state count | signed raw shortfall |
|---|---:|---:|---:|---:|---:|
| CARE/COW | 2,848 | 2,835 | 1,749 | 0 | 2,848 |
| CARE/GOOSE | 0 | 0 | 0 | 0 | 0 |
| CARE/SHEEP | 1,731 | 1,721 | 1,158 | 0 | 1,731 |
| FERTILIZE/CARROT | 0 | 0 | 0 | 0 | 0 |
| FERTILIZE/MELON | 0 | 0 | 0 | 0 | 0 |
| FERTILIZE/STRAWBERRY | 1,470 | 1,470 | 731 | 3,020 | -1,550 |
| FERTILIZE/TOMATO | 0 | 0 | 0 | 0 | 0 |
| FERTILIZE/WHEAT | 0 | 0 | 0 | 0 | 0 |

The observed totals above are literal sums of diagnostic state counts and must not be read as daily completed action totals. The ignored audit JSON is authoritative for exact per-game/per-day rows and classification evidence.

## Exhaustive shortfall findings

| family/entity | requested>0 / observed=0 rows | material rows | exact raw signed shortfall | classification |
|---|---:|---:|---:|---|
| CARE/COW | 583 | 583 | 2,848 | 13 manager-requested-infeasible; 570 unresolved |
| CARE/SHEEP | 579 | 579 | 1,731 | 10 manager-requested-infeasible; 569 unresolved |
| FERTILIZE/STRAWBERRY | 0 | 50 | -1,550 aggregate; individual signed values retained | 50 unresolved state comparisons; persistent state metric is not a daily completion count |

There are 23 manager-requested-infeasible rows in this panel (13 COW, 10 SHEEP), identified by `requested > feasible_projected`; the remaining 1,189 material rows are unresolved. No shortfall is classified as missing inventory, insufficient labor, foreman scheduling, or dependency blocking without a concrete per-task signal. Explicit failed-buy evidence, if any, is retained in each ignored ledger row; absent an applicable signal the classifier reports `unresolved` and names the missing acceptance/day-end/unassigned-reason signals. This table is exhaustive by entity and row count; the ignored JSON contains every matching `(seed, seat, day, family, entity)` row without omission.

## Runtime signals and anomalies

- Fallback executor errors: 0; per-day errors: 0; unaffordable market orders: 0.
- Unresolved task-generator entries: 501; these are recorded as generator evidence, not assigned to CARE/FERTILIZE without a causal key.
- Prior-day debt suppression fired on 565/624 day records; current suppression was true on 624/624 records.
- Engine status anomalies: 0; all 24 games ended `DONE/DONE` at step 719.
- Illegal/ineffective action signal: unavailable in all 24 games, with the exact engine-observation limitation retained in the audit JSON.

## Per-game summary

| seed | seat | CARE requested | CARE observed state sum | CARE assigned | FERT requested | FERT observed state sum | FERT assigned | final bank |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 | 0 | 195 | 0 | 123 | 62 | 101 | 26 | 61,572 |
| 7 | 1 | 192 | 0 | 120 | 85 | 144 | 36 | 54,439 |
| 17 | 0 | 190 | 0 | 120 | 64 | 132 | 32 | 53,146 |
| 17 | 1 | 188 | 0 | 120 | 55 | 114 | 26 | 54,400 |
| 42 | 0 | 184 | 0 | 120 | 47 | 131 | 30 | 58,267 |
| 42 | 1 | 189 | 0 | 120 | 81 | 173 | 46 | 60,340 |
| 123 | 0 | 188 | 0 | 120 | 58 | 127 | 30 | 55,220 |
| 123 | 1 | 188 | 0 | 120 | 58 | 127 | 30 | 55,220 |
| 2026 | 0 | 190 | 0 | 120 | 47 | 124 | 26 | 67,342 |
| 2026 | 1 | 187 | 0 | 120 | 46 | 109 | 24 | 65,468 |
| 1013 | 0 | 189 | 0 | 120 | 53 | 131 | 28 | 66,274 |
| 1013 | 1 | 192 | 0 | 120 | 78 | 155 | 42 | 65,271 |
| 1022 | 0 | 192 | 0 | 120 | 53 | 121 | 28 | 60,141 |
| 1022 | 1 | 192 | 0 | 120 | 45 | 114 | 25 | 57,216 |
| 1003 | 0 | 188 | 0 | 120 | 0 | 97 | 19 | 59,740 |
| 1003 | 1 | 195 | 0 | 123 | 70 | 117 | 36 | 54,871 |
| 1026 | 0 | 197 | 0 | 123 | 83 | 147 | 36 | 62,642 |
| 1026 | 1 | 200 | 0 | 125 | 88 | 159 | 43 | 61,837 |
| 1011 | 0 | 181 | 0 | 120 | 29 | 44 | 7 | 65,782 |
| 1011 | 1 | 181 | 0 | 120 | 29 | 44 | 7 | 65,782 |
| 1024 | 0 | 199 | 0 | 125 | 81 | 154 | 40 | 64,655 |
| 1024 | 1 | 199 | 0 | 125 | 61 | 140 | 38 | 65,889 |
| 1019 | 0 | 194 | 0 | 123 | 93 | 175 | 42 | 66,546 |
| 1019 | 1 | 189 | 0 | 120 | 68 | 140 | 34 | 56,614 |

## Recommendation

No proven CARE/FERTILIZE executor bug is established by this panel. Do not change executor behavior from this audit; if Stage 3/4 work later requires a causal verdict, first add a narrowly scoped engine-visible accepted-action/day-end telemetry seam.

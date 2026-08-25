# Executor V0.7 Final Evidence

Date: 2026-08-25 · frozen commit: `a7c826d`

## Objective and decision

Finalize the issue #7 deterministic executor after real BC-E closed-loop
evidence, without turning expert-intent-only gains into acceptance claims.
R4 was rejected/reverted after the real fixed-plan 7d regression. Accepted:
the panel outcome ledger (`02984a0`) and the survival WHEAT shed-room clamp
(`a7c826d`). The clamp uses official/default `shed_capacity=100` and keeps
survival-before-hire order. Prior-debt suppression remains default **ON** as
explicit heuristic architectural debt, bounded only to the panel below.

## Final PASS panel

PASS, seeds 17/42/2026, both seats, six games; ON means prior-debt suppression
enabled and OFF disabled.

| seed | seat | ON bank | OFF bank |
| ---: | ---: | ---: | ---: |
| 17 | 0 | 17,005 | 265 |
| 17 | 1 | 14,961 | 265 |
| 42 | 0 | 23,346 | 30 |
| 42 | 1 | 26,587 | 33 |
| 2026 | 0 | 56,742 | 0 |
| 2026 | 1 | 65,959 | 0 |

ON: mean 34,100, median 24,966.5, minimum 14,961; loss units starvation 0,
overflow 12. OFF: mean 98.8, median 31.5, minimum 0; starvation 38, overflow
12. This is not a broader quality or generalization claim.

R4 regression deltas were wealth -14,302, cash -9,515, weeds +11, crops
destroyed +14, survival debt 69→84, starvation 22→44, and harvest 124→98.

## Mechanism and viewer evidence

The seed-17 day-20 starvation-purchase defect was fixed: banks improved from
8,062/8,489 to 17,005/14,961. Residual day-22 six-animal/seat debt is true
manager-policy debt: carried cows are discarded at day end into a full shed;
it is not a starvation or executor defect, so no extra heuristic was added.

Viewer #11 is closed and passive. Schema/runner/snapshot/viewer/debug-trace
CLI support is committed; four ignored real BC-E E_VS_E traces were generated
and parsed, with exact action/result parity for seeds 17 and 42 in ON/OFF
comparisons. E_VS_E banks intentionally differ from the PASS panel. No traces
are committed, and illegal-action absence was not proven; engine 1.32.7's
detection surface is unavailable for that claim.

## Reproduction and hashes

- ON panel (ignored): `artifacts/local/executor_v07_evidence/final_v07_no_r4_shedfix_prior_on_panel.json`; SHA-256 `3b4bee86e18e0c24f9e2c9c0a563703d8b7f1a0c7e54e3217099c6a417c75737`.
- OFF panel (ignored): `artifacts/local/executor_v07_evidence/final_v07_no_r4_shedfix_prior_off_panel.json`; SHA-256 `0603be323f969e9e905840ba6a96a0745458be9369dd2820ee3a5558b9537c7c`.
- Seed-17 shed-room fix (ignored): `artifacts/local/executor_v07_evidence/seed17_final_v07_shed_room_fix_prior_on.json`; SHA-256 `4c4158318fc9f6e062b8696f73660a76c36004410a3f3f18d696e1580db5daee`.
- Seed-17 prior outcomes (ignored): `artifacts/local/executor_v07_evidence/seed17_final_v07_prior_on_outcomes.json`; SHA-256 `2def59c89de8cd0ab4ca526f2d85f165bf8db56db47352b54f22d5bfa65624c8`.
- Real BC-E validation input (externally supplied read-only and uncommitted): `C:\Users\liuyi\VSCodeProjecs\Kaggriculture\Kaggriculture\artifacts\local\bc-v1-E\best.pt`, variant E epoch 27, SHA-256 `F4B029D3E463ABA1DBD0544377D0D616E3DE94AA6CC469D3446F018DDDD8F6BF2`. It is intentionally absent from this isolated worktree because local artifacts/checkpoints are ignored.
- Retained tapes: [`executor_v07_fixed_plan/tape_ep98004787_s0_d10_l5.json`](executor_v07_fixed_plan/tape_ep98004787_s0_d10_l5.json), [`executor_v07_fixed_plan/tape_ep98093786_s0_d12_l7.json`](executor_v07_fixed_plan/tape_ep98093786_s0_d12_l7.json), [`executor_v07_fixed_plan/tape_ep98134768_s0_d5_l3.json`](executor_v07_fixed_plan/tape_ep98134768_s0_d5_l3.json). Each is schema-valid, covers its recording window, and carries internal `artifact_sha256` plus full engine/checkpoint/opening/replay provenance.

## Recommendation and caveats

Freeze V0.7 at `a7c826d` and use the default ON setting for the next RL
executor-factory swap/serious-training gate. Do not generalize beyond this
three-seed, two-seat PASS panel, do not cite expert-intent-only improvements
as acceptance, and do not claim illegal actions were proven absent.

# Stage 2.5 queue-ownership repair evaluation handoff

## Contract

Compare two candidate-only executor arms against the same frozen BC-E
opponent and the same stochastic P-final PPO policy:

1. existing persistent worker queues;
2. persistent worker queues plus `AgentConfig.queue_ownership_repair=True`.

Both arms retain `underfoot-first`, deadline-safe planting, and
deadline-safe hiring. The repair flag is default-off and is effective only
when persistent queues are enabled; it is never passed to the frozen
opponent. Schedule-informed hiring remains off. Use `E_LEGACY`, the standard
mixed opening, official `kaggle_environments==1.32.7`, four child processes,
and one bounded CPU thread per child.

The evaluator preserves both orientations and the identity
`episode_id = master_seed * (2 * len(full_seeds)) + 2 * seed_index + seat`.
Capture is passive: the primitive action is computed before executor/debug
snapshots are serialized. Do not compare a filtered or shortened seed list to
a full-panel run.

## Exact configurations

The notebook first runs seed `1470672056` with both seats and capture enabled,
then runs the 32-seed panel after the checkpoint-free targeted regression. The
panel is the original ordered 16-seed list followed by the known seed and 15
deterministic affine extensions:

```text
144368101, 309507, 615013, 918079, 1221109, 1524137, 1827169, 2130193,
2433221, 2736251, 3039283, 3342311, 3645341, 3948373, 4251401, 2112243121,
1470672056, 995106988, 1303793286, 521973470, 107449192, 768565387,
1370134739, 2090797777, 425789796, 1027359148, 1688475343, 261654734,
863224086, 1524340281, 97519672, 699089024
```

From the repository root, with actual checkpoint paths substituted:

```bash
COMMON='--backend official --e-history-version E_LEGACY --master-seed 25 --processes 4 --variants combined combined_wheat3 --underfoot-first --deadline-safe-planting --deadline-safe-hiring'
python -m tools.run_stage25_upkeep_sharded \
  --checkpoint "$PPO_CHECKPOINT" --e-checkpoint "$BC_E_CHECKPOINT" \
  --output-dir artifacts/local/stage25-known-persistent \
  --seeds 1470672056 --persistent-worker-queues --capture-dir artifacts/local/stage25-known-persistent-capture $COMMON
python -m tools.run_stage25_upkeep_sharded \
  --checkpoint "$PPO_CHECKPOINT" --e-checkpoint "$BC_E_CHECKPOINT" \
  --output-dir artifacts/local/stage25-known-repair \
  --seeds 1470672056 --persistent-worker-queues --queue-ownership-repair \
  --capture-dir artifacts/local/stage25-known-repair-capture $COMMON
```

The notebook contains the same commands for the full ordered panel, with
capture disabled for the panel arms. It pins evaluator/source commit
`9d0035a9cc9bbb04de4d0174c0ca15755a6b0703`;
the final notebook-only commit intentionally follows that source commit so the
clone contains all evaluator artifacts.

## Provenance and reporting

Every manifest records the ordered seeds, episode identities, source/patch
hashes, checkpoint hashes, backend provenance, arm flags, and child count.
`games.jsonl`, `comparison.json`, and `bootstrap_groups.json` retain paired
seat results. Capture/audit telemetry is not inferred from bank movement.

| requested surface | source mapping |
|---|---|
| candidate bank, opponent bank, margin | evaluator game row |
| hiring expense | executor `previous_labor.hire_cost`, when captured |
| missed maintenance | executor day `missed_maintenance` count, when captured |
| completed useful work | capture audit `cand_completed` observed-interaction heuristic |
| duplicate claims | capture audit `cand_coassigned_turns` |
| target abandonment | capture audit `cand_ended_unobserved` |
| movement between interactions | capture audit `cand_movement` worker-turn proxy; executor `foreman_counts.movement` when captured |

The sharded telemetry file always emits these canonical fields. If capture or
the audit does not expose a field, it reports `available: false`; it does not
turn missing data into zero. Movement is reported separately and is not a
sufficient success criterion.

## Local availability and limits

No PPO or BC-E checkpoint evaluation was run in this checkout: the checkpoint
artifacts and the official Kaggle engine are not locally available. This note
therefore makes no score, win-rate, or promotion claim. The local validation
is limited to evaluator/sharder propagation, identity preflight, notebook
JSON/structure validation, and focused checkpoint-free tests.

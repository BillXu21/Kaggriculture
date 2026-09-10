# Stage 2.5 starvation workload-visibility repair handoff

## Contract

This is a separately switchable, default-off executor repair layered on the
queue-ownership repair source. `starvation_workload_visibility_repair` keeps
the complete post-suppression workload visible to labor planning while the
existing starvation dispatch restriction remains in force for ordinary tile
work. Only starvation-boundary FEED (`consecutive_unfed >= 1`) is critical;
ordinary daily FEED is deferred. A deadline-critical WATER task is admitted
only when the forecast reserves feasible worker time for critical FEED first.
The reservation is diagnostic and conservative: an assignment does not prove
timely completion. WHEAT reserves, affordability, resource checks, terminal
deadlines, and same-worker plant/water continuations remain authoritative.

The restriction that remains is deliberate: expansion and ordinary tile work
stay suppressed during starvation unless the work is a feasible deadline-safe
WATER continuation. Resource-blocked FEED is reported as uncovered; it does
not erase the complete crop workload from diagnostics or hiring visibility.
The change does not enable schedule-informed hiring or add a new hiring
objective.

Opt-in diagnostics record complete versus eligible workload, starving animal
identities, feed coverage and reserved ETA, deferred deadlines/reasons, labor
forecasts before/after filtering, marginal hire costs, and survival work missed
at the boundary. Flag-off actions and diagnostics are unchanged.

## Isolated evaluation

Use the frozen BC-E opponent, official `kaggle_environments==1.32.7`,
`E_LEGACY`, the same P-final PPO checkpoint, common upkeep/wheat/underfoot /
deadline controls, and `--processes 4`. Keep both arms at:

```text
--persistent-worker-queues absent
--queue-ownership-repair absent
--schedule-informed-hiring absent
```

Control is the default executor. Treatment adds
`--starvation-workload-visibility-repair`. The ownership flag is equal (false)
on both sides; this comparison is not the queue-ownership ablation.

The ordered panel is:

```text
144368101, 309507, 615013, 918079, 1221109, 1524137, 1827169, 2130193,
2433221, 2736251, 3039283, 3342311, 3645341, 3948373, 4251401, 2112243121,
1470672056, 995106988, 1303793286, 521973470, 107449192, 768565387,
1370134739, 2090797777, 425789796, 1027359148, 1688475343, 261654734,
863224086, 1524340281, 97519672, 699089024
```

For the targeted interaction check, preserve those identities by passing the
full list plus `--game-filters 1470672056:0 1470672056:1` to both arms. Enable
capture for that check only. The current sharder may create one non-empty shard
for a one-seed filter; retain `--processes 4` and use four processes for the
subsequent full panel.

From the repository root, substitute real checkpoint paths and the pushed
source SHA:

```bash
COMMON='--backend official --e-history-version E_LEGACY --master-seed 25 --processes 4 --variants combined combined_wheat3 --underfoot-first --deadline-safe-planting --deadline-safe-hiring --seeds 144368101 309507 615013 918079 1221109 1524137 1827169 2130193 2433221 2736251 3039283 3342311 3645341 3948373 4251401 2112243121 1470672056 995106988 1303793286 521973470 107449192 768565387 1370134739 2090797777 425789796 1027359148 1688475343 261654734 863224086 1524340281 97519672 699089024'
python -m tools.run_stage25_upkeep_sharded --checkpoint "$PPO_CHECKPOINT" --e-checkpoint "$BC_E_CHECKPOINT" --output-dir artifacts/local/stage25-starvation-control --capture-dir artifacts/local/stage25-starvation-control-capture $COMMON
python -m tools.run_stage25_upkeep_sharded --checkpoint "$PPO_CHECKPOINT" --e-checkpoint "$BC_E_CHECKPOINT" --output-dir artifacts/local/stage25-starvation-repair --capture-dir artifacts/local/stage25-starvation-repair-capture --starvation-workload-visibility-repair $COMMON
```

Run the same two commands with the full seed list and
`--game-filters 1470672056:0 1470672056:1` for the targeted captures. Compare
earlier feasible maintenance coverage, critical-feed completion time,
late-hire bursts and marginal cost, crop-to-weed transitions, completed work,
both banks, and margin. Do not infer a fixed day-16 crop loss from task
visibility alone.

## Provenance and limits

Record the pushed source SHA, checkpoint hashes, ordered seeds, seat/orientation
identities, engine version, and manifest for every arm. The checkout has no
PPO/BC-E checkpoints or official engine, so no model result, score, crop-loss
repair, or promotion claim is made here. The local evidence is the focused
checkpoint-free test suite and preflight/manifest validation.

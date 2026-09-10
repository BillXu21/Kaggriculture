# Stage 2.5 schedule-informed hiring economic repair

This repair is an independent, default-off control layered on
`schedule_informed_hiring`. Enable both `--schedule-informed-hiring` and
`--schedule-hiring-economic-repair` for the repaired candidate. Existing
schedule-informed hiring behavior is unchanged when the second flag is absent.
The queue, queue-ownership, starvation-visibility, deadline, manager, crop
placement, and training controls remain independent.

## Decision contract

The pure seam evaluates a fresh resource ledger for every candidate worker
count, including the zero-hire baseline. It does not require every task to be
scheduled: feasible useful work is retained even when another task is blocked
by a missing seed, fertilizer, dependency, deadline, or capacity. Shed stock
and global seeds are consumed in candidate order; a resource cannot be spent
twice. Candidate diagnostics report completed keys, remaining capacity,
candidate hire cost, accepted/rejected reason, and residual resource/deadline
blockers.

Costs are deliberately mechanical: only worker actions after the current
market phase are counted; a new hand starts at the authoritative least-occupied
shed-access tile, then pays Manhattan travel, shed pickup actions, interaction,
and the required WATER follow-up for PLANT. Hire prices use
`replay_daily.constants.hire_cost(hires_today + index, hire_cost_mult)`, and
cash and the ten-order market cap are applied after candidate scoring. The
official engine confirms that worker actions precede market orders and that a
new hand acts on the next turn.

The benefit model is intentionally simple and conservative. A directly
completed maintenance task is worth `maintenance_benefit` (default 100), a
productive task 8, a manager task 3, and logistics 1. Optional/uncertain work
never justifies a hire. A task depending on another current task receives no
downstream bonus, and PLANT already includes its required WATER follow-up, so
that follow-up is not credited again. These are planning units, not claims of
market revenue or guaranteed crop/animal value; the full game remains the
authority for saved crops, saved animals, and productive output.

## Pure captured-state replay

First run the captured day-9 and day-14 observations through
`executor_v0.hiring.recommend_hires` and retain the returned JSON for each
turn. The replay must include the actual observation `farms`, `private` shed,
seeds, inventories, current worker positions, current task list, cash,
`hires_today`, and the active hire-cost multiplier. Compare the zero-hire
baseline and every candidate count; do not treat this as a full-game
counterfactual. The current checkout does not contain the audited seed
`1111877947` capture, so the replay is runnable once the preserved capture is
mounted, but no local day-9/day-14 claim is made here.

Example pure replay once `CAPTURE_JSON` contains one state per line:

```powershell
Get-Content $env:CAPTURE_JSON | python -c "import json,sys; from executor_v0.hiring import recommend_hires; [print(json.dumps(recommend_hires(row['obs'], row['seat'], row.get('tasks', []), available_cash=row.get('cash', 0), hires_today=row.get('hires_today'), hire_cost_mult=row.get('hire_cost_mult', 1), economic_repair=True).to_json_dict(), sort_keys=True)) for row in map(json.loads, sys.stdin)]"
```

## Frozen-BC evaluation handoff

Use the known seed `1111877947` in both orientations first. Keep persistent
queues OFF, keep all Packet-2 flags equal across arms, and preserve the full
seed/episode/opening identities. Compare these arms in order:

1. Existing `schedule_hiring` (`--schedule-informed-hiring` only).
2. Repaired `schedule_hiring` (add `--schedule-hiring-economic-repair`).
3. Control (schedule-informed hiring absent) versus repaired schedule hiring.

Set `PPO_CHECKPOINT` and `BC_E_CHECKPOINT` to immutable files and run from the
repository root. Pin every comparison arm to evaluator/source commit
`d892bf4dade5d5e9053e9ea7af57badb4f1abdd5` (the pushed hiring-repair head);
check out that exact commit before running any arm so the existing, repaired,
and control arms all evaluate the same source:

```powershell
git fetch origin d892bf4dade5d5e9053e9ea7af57badb4f1abdd5
git checkout --detach d892bf4dade5d5e9053e9ea7af57badb4f1abdd5
```

The sharder uses four bounded CPU workers and records the
source, patch, checkpoint, engine, seed, seat, and episode identities:

```powershell
$COMMON = @('--backend','official','--e-history-version','E_LEGACY','--master-seed','25','--processes','4','--variants','combined','combined_wheat3','--underfoot-first','--deadline-safe-planting','--deadline-safe-hiring','--seeds','1111877947')
python -m tools.run_stage25_upkeep_sharded --checkpoint $env:PPO_CHECKPOINT --e-checkpoint $env:BC_E_CHECKPOINT --output-dir artifacts/local/stage25-schedule-existing @COMMON --schedule-informed-hiring
python -m tools.run_stage25_upkeep_sharded --checkpoint $env:PPO_CHECKPOINT --e-checkpoint $env:BC_E_CHECKPOINT --output-dir artifacts/local/stage25-schedule-repaired @COMMON --schedule-informed-hiring --schedule-hiring-economic-repair
python -m tools.run_stage25_upkeep_sharded --checkpoint $env:PPO_CHECKPOINT --e-checkpoint $env:BC_E_CHECKPOINT --output-dir artifacts/local/stage25-schedule-control @COMMON
```

Then run the 32-seed panel with the same two seats per seed and the same three
arm configurations. Report hiring timing and expense, requested/submitted/
observed hires, useful completed work, residual blockers, crop/animal losses,
both banks, margin, and decision runtime. A submitted order is not an
observed hire: the latter must come from the next engine observation.

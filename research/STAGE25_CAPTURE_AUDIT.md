# Stage 2.5 paired replay capture + executor audit

Instrumentation for `research/STAGE25_UPKEEP_ABLATION.md`. No policy,
heuristic, priority, routing, or assignment behavior is changed by any of
this: default evaluation runs byte-identical games with capture off.

## Evaluator flags (opt-in; default off)

```bash
python -m tools.evaluate_stage25_upkeep \
  --checkpoint "$PPO_CHECKPOINT" --e-checkpoint "$BC_E_CHECKPOINT" \
  --seeds 144368101 2112243121 --master-seed 25 --backend official \
  --variants baseline care fertilizer combined \
  --output-dir results --capture-dir captures
```

- `--capture-dir`: enables per-game capture. Without it, no rollout, debug
  trace, executor snapshot, or replay call is made and output files are
  unchanged apart from two additive `games.jsonl` fields (`episode_id`,
  and `capture`/`capture_complete` only when capturing).
- `--game-filter SEED:SEAT` (repeatable): runs a subset while preserving
  original episode IDs. A shorter `--seeds` list instead **renumbers** games
  and must not be compared with the original panel.

Episode identity contract: with N seeds and master seed M, game
`seeds[index]` at seat `s` always uses `M * (2N) + 2*index + s`, for every
variant. Pairing key across arms is `(seed, seat, episode_id)`.

Capture mode sets `RunnerConfig(record_rollout=True,
record_debug_trace=True, record_executor_full_diagnostics=True,
record_official_replay=True)` and `record_turn_snapshot=True` on both
executors (both sides traced; the candidate is labeled by
`meta.candidate_seat`). These serializers are read-only: no extra RNG calls,
policy inference, or observation/plan/task mutation. Verified locally by
`tests/test_stage25_capture.py::test_capture_off_on_identical_actions_and_outcome`.

## Capture layout

`<capture-dir>/<variant>/episode_<id>_seed_<seed>_seat_<seat>/`:

| file | content |
| ---- | ------- |
| `meta.json` | pairing identity, outcome, provenance; `complete: true` written last |
| `debug_trace.json.gz` | canonical per-turn states + both seats' submitted actions + per-turn executor snapshots (existing `TraceRecorder` format) |
| `rollout.json.gz` | submitted primitive actions, sampled manager plans per seat/day, request digests, opening handoff |
| `executor_seat0/1.json.gz` | full per-day executor diagnostics (requested vs feasible plan, tasks, foreman counts, debts, hires, sells, economics) |
| `replay.json.gz` | official Kaggle replay (`env.toJSON()`; official backend only) |
| `status_history.json` | full official status history (official backend only) |
| `capture_error.json` | only on capture failure; earlier files are kept, the game result stands |

Task matching key: regenerated tasks use stable `key` strings (`KIND:y,x`
/ `SELL:…` / `BUY_…`). A disappearing key that reappears is a regenerated
equivalent; a new key is a genuinely new task. Reasons come only from
recorded foreman `reason`, `unassigned_reasons`, and generator `unresolved`
strings.

## Audit command

```bash
python -m tools.audit_stage25_capture \
  --capture-dir captures --output-dir audit --focus-pairs 4
```

Emits `audit.md` plus `games.csv`, `days.csv`, `pairs.csv`,
`divergences.csv`, `divergences.json` (and `skipped.json` if any capture
was incomplete). Report covers per-worker movement/productive/PASS splits,
completed-by-type (observed-interaction heuristic, documented in the module
docstring), assignment churn split by completion vs abandonment,
day-end debts, land/planting/economics trajectories, care/fertilizer
workload, paired first-divergences (action, state, manager plan), bank /
opponent-bank / margin deltas separately, and a combined-vs-fertilizer
section. All bottleneck candidates are trace-linked hypotheses; the report
states its reading cautions up front and uses no route optimizer or
counterfactual scheduler.

## Kaggle runbook

Use `notebooks/stage25_upkeep_capture_audit.ipynb` in the live session:
preflight checkpoints, single-game exact-identity capture check (stop on
mismatch), full 16-game panel with capture, audit, ZIP export. Same
working-root and checkpoint paths as the original notebook.

## Validation status

- Local (done): capture off/on equivalence on a scripted fast-engine game;
  executor snapshot neutrality; episode-ID/filter contract; writer
  round-trip + partial handling; audit fixture with known transitions.
- Kaggle (pending): real-checkpoint capture-on/off equivalence for one
  original game, then the 16-game panel and gameplay audit.

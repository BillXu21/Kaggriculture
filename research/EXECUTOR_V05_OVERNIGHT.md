# Executor V0.5 overnight optimization (issue #7)

Branch: `executor-v05-overnight`  |  Base SHA: `32fef4ac295e9addaf01cd339eee60a0ad14eaca`  |  Final SHA: `7319f31` (report/evidence commits follow `885adad`; see git log)

Scope discipline: executor-only mechanical compiler work under the frozen
manager. No manager/opening/BC changes. The BC-E checkpoint was **not used**
(unavailable locally); per the issue's revised methodology, the entire inner
loop ran checkpoint-free on expert-derived DailyPlans from raw replays.

## 1. Replay corpus manifest

Built automatically by `tools/replay_manifest.py` over
`C:\Users\liuyi\VSCodeProjecs\Kaggriculture\Kaggriculture\data\samples`
(raw files stay untracked; nothing bulk-committed):

- 19 raw 1.32.7 episode JSONs, all 720 steps, turnsPerDay=24.
- `98178196` (`datasamplesbc_e_mirror_98178196.json`, seed 0, Bill Xu mirror,
  rewards [42, 42]) -> classified `failure_specimen`; used ONLY as a bug
  specimen, never as a teacher.
- 18 elite games (rewards 48k-140k; teams incl. MiMi, Subramanya N,
  Ryo Hasegawa, Crop Dusta, junseok lee) -> classified `high_reward`.
- Regenerate: `python tools/replay_manifest.py <samples_dir> out.json`.

## 2. Engine-parity foundation for one-day slices

Before any evaluation, full-game action-trace reconstruction was validated:
resetting `FastKaggricultureEnv` with each replay's seed/config and replaying
both seats' recorded primitive actions reproduces the official observations
**exactly** for entire 719-turn episodes (4 episodes verified initially,
including the failure specimen). Only cosmetic field differences exist
(`age` vs `placed_day`, `remainingOverageTime`, zero-valued inventory keys).
This upgrades the fast Rust engine to the trusted slice-reconstruction
backend required by the issue.

Integration bug found by the harness immediately: `derive_animal` crashed
with `KeyError: 'placed_day'` on fast-engine animal tiles (they carry `age`),
making the executor silently all-PASS locally. Fixed to accept both
observation shapes and fail loudly otherwise (`baa8e86`).

## 3. One-day replay-slice harness

`tools/day_slice.py` + `tools/run_day_slice_suite.py`:

1. reset exact env with replay seed/config;
2. replay recorded joint actions through day-d hour 0;
3. verify reconstructed state against the raw replay boundary
   (`boundary_verified`, first-diff text);
4. extract the expert daily intent via `tools/expert_plan.py`
   (extractor-alignment canonical logic: end-of-day crop/animal targets,
   land count, CARE/FERTILIZE event counts attributed by tile-under-worker,
   sells grouped into the six bins);
5. switch only the tested seat to the candidate executor with that plan;
6. opponent stays on recorded actions for exactly 24 turns;
7. collect diagnostics; discard state (day d+1 re-reconstructs from scratch).

Metrics per slice: cash start/end, wealth start/end (cash + shed/carried at
observed market prices, animals at cost, crops/seeds at seed cost),
crops/animals/weeds/empty structures/unlocked counts, harvestable leftovers,
hires, foreman/market op families, full agent diagnostics.
Paired scenarios share identical reconstructed starts.

## 4. Verified pathologies and fixes (all reproduced first)

| # | Pathology (issue hypothesis) | Root cause confirmed | Fix |
|---|---|---|---|
| 1 | Weeds never cleaned | task scan skipped every weed shape; weeds permanently block PLANT/BUILD | weed tiles are reclaimable slot pools in both planners (DIG prerequisite wired into BUILD/PLANT chains); both `"WEED"` string and `{"kind":"WEED"}` forms recognized |
| 2 | Blanket WATER labor sink | `watered_today != True` => top-priority WATER for every plant | exact-mechanics classes: MUST (`consecutive_unwatered>=1`, incl. planting day) = MAINTENANCE; YIELD (single-harvest window / fertilized ongoing production-eve) = PRODUCTIVE; else no task |
| 3 | `[x,y]`/`[y,x]` anchor transposition | farmer `[x,y]` fed into `[y,x]` Manhattan math | layouts anchored to persistent hub `SHED_HUB_ANCHOR=(4,4)`; worker positions still converted explicitly in foreman |
| 4 | Layout churn from moving farmer anchor | anchor moved every turn | stable hub anchor; deterministic across turns/days |
| 5 | Crop/animal planners double-claimed tiles | independent scans of same board | `plan_day_layouts`: animals plan first, claimed tiles masked before crop reconciliation |
| 6 | Movement refused locked quadrants | `_legal_step` blocked what engine ops 1-4 allow unconditionally | bounds-only legality (pinned to rust `apply_unit_action`) |
| 7 | Same-turn SELL->HIRE ignored proceeds | hires priced off pre-market cash at hour 0 only | sequential queue accounting: running cash = money + queued sell revenue; hiring any hour |
| 8 | Hard 3-hire cap | arbitrary config cap | removed; workload-derived desired hands + maintenance-travel floor + exact fib affordability |
| 9 | Impossible market-order spam | buys had NO affordability check (BUY_LAND every hour etc.) | exact-cost gates (seed table, animal table, LAND_PRICES, official price model for products); skips logged as `unaffordable_market_orders` |
| 10 | BUILD before PLACE executable | builds emitted for deficits regardless of animal availability | BUILD emitted only when animal owned or affordable; else honest `build_deferred_no_animal` |
| 11 | CARE/FERTILIZE clipped to hour-0 assets | projection ignored same-day plan-implied animals/crops | eligibility = max(current, requested target), still clipped above planned totals |
| 12 | Diagnostics overwritten each turn | last-turn snapshot only | accumulated `unfinished_task_turns` / `missed_maintenance_turns` counters + snapshots |

## 5. Experiments accepted / rejected

Accepted (each with focused tests, smoke slices, then expanded set):

- movement legality fix (`48dca3d`);
- water urgency + coordinated hub layouts + weed reclamation + build gating (`924d966`);
- projection eligibility (`96c31ff`);
- sequential market accounting + any-hour workload hiring + buy gates + diagnostics (`620b43e`);
- maintenance-travel-aware hire floor (`885adad`).

Rejected and reverted:

- **Persistent task ownership hint** (foreman score bonus 500, then 40):
  mean wealth delta dropped 894 -> 864 on the smoke set regardless of bonus
  size; continuity never beat the churn cost here. Reverted cleanly;
  tests removed with it.
- **Travel-aware hiring over ALL tasks**: aggregate wealth gain fell
  +5.5% -> +1.5% (Fibonacci fees for 11-hand crews on low-priority work);
  replaced by the tempered maintenance-only floor.

## 6. Paired results (identical reconstructed day-starts)

Smoke set (14 slices: specimen d1/d3/d5 both seats + three elite replays):

- boundary verified 14/14 both sides; zero errors.
- WATER ops/day 16-23 -> typically 6-11 (mechanics-derived only).
- weeds_end totals improved (e.g. 98184881 d6: baseline 6 remaining vs 0);
  crops preserved/increased in most slices.

Expanded set (38 unique slices: specimen d2/d4 + every elite replay x two
mid-game days on alternating seats):

| metric | baseline (32fef4a+tools) | candidate r4 |
|---|---|---|
| mean one-day wealth delta | 1879 | **1981 (+5.4%)** |
| mean one-day cash delta | -16 | **+57** |
| wealth-better slices | - | **24/38** |
| weeds remaining at day end (total) | 53 | **22** |
| harvestable left at day end (total) | 1 | **0** |
| boundary verified | 38/38 | 38/38 |

Artifacts: `research/day_slice_baseline.json`,
`research/day_slice_candidate_r1.json`,
`research/day_slice_expanded_baseline.json` (label TRUE-baseline),
`research/day_slice_expanded_candidate.json` (label expanded-candidate-r4).

Worst remaining regressions vs baseline: `(98198569,d6,s1)` ~-627 and a few
~-100..300 day-6 seat-1 slices. Traced mechanism: genuinely saturated days
(crew capacity < interaction+travel demand of the expert plan) where the
baseline's blanket watering happened to keep plants alive while skipping all
build/fertilize/place work. The candidate does strictly more distinct work
but cannot cover everything; asset losses concentrate in overnight refreshes
of starved must-waters on the most crowded single day.

## 7. Remaining known executor failures

- One-worker-at-a-time route filling / batching not implemented; movement is
  still 50-67% of worker turns on scattered boards.
- Task reassignment churn between turns remains (ownership experiment failed;
  needs a route-level scheduler rather than a score patch).
- Wealth metric ignores yield units accumulated on tiles at refresh.
- BUY quantities are whole-deficit; JIT partial purchasing not implemented.
- BC-E-in-loop regression not run (checkpoint unavailable locally, per issue
  optional for the overnight loop).

## 8. Official-engine confirmation status

Official 1.32.7 Python engine is NOT installed locally
(`import kaggle_environments` fails). Confirmation performed instead:
exact full-game fast-engine reconstruction of four official replay JSONs
including the failure specimen (all 719 turns, both seats, state equality
modulo cosmetic field names). All mechanics fixes cite the vendored rust
source (`rust/kaggressive_env`... actual path `rust/kaggriculture_env/src/lib.rs`)
and MECHANICS.md sections. Per issue, official-engine acceptance remains a
promotion-time requirement and was not runnable in this environment.

## 9. Reproduction commands

```powershell
# worktree setup (already done)
git -C C:\Users\liuyi\VSCodeProjecs\Kaggriculture\Kaggriculture worktree add `
  ..\Kaggriculture-executor-v05 -b executor-v05-overnight origin/main

# unit/mechanics suites
python -m pytest tests/test_executor_v0.py tests/test_executor_v0_agent.py `
  tests/test_executor_v0_foreman.py tests/test_executor_v0_layout.py `
  tests/test_executor_v0_tasks.py tests/test_replay_daily.py `
  tests/test_replay_tools.py tests/test_day_slice.py -q `
  --basetemp="$env:TEMP\opencode\pt"

# manifest
python tools/replay_manifest.py `
  C:\Users\liuyi\VSCodeProjecs\Kaggriculture\Kaggriculture\data\samples out.json

# paired one-day suites (smoke 14 / expanded 38)
python tools/run_day_slice_suite.py --out research\smoke.json --label X
python tools/run_day_slice_suite.py --out research\exp.json --label X --set expanded
```

Baseline reproduction: check out `32fef4a` plus commits `31027b5`, `96a3d7f`,
`baa8e86` (tooling + lifecycle shape fix only), then run the same suite.

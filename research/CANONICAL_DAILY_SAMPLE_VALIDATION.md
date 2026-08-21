# Canonical Daily Sample Validation

Date: 2026-08-21
Engine: Kaggriculture `1.32.7`
Source partition: `kaggle/kaggriculture-episodes-2026-08-20`

## Artifacts and commands

The 15 local replays were extracted into the ignored artifact:

`data/canonical/2026-08-20-sample.jsonl`

- 900 records: 15 episodes × 30 days × 2 seats;
- 67,686,853 bytes;
- SHA-256: `0d9976703503da3aaa0dcc9a51c691df80195ba9864420346c90b9068699c2e7`.

Generation:

```text
python -m replay_daily extract --input data\samples\2026-08-20 --manifest data\samples\2026-08-20\manifest.csv --source-dataset kaggle/kaggriculture-episodes-2026-08-20 --partition-date 2026-08-20 --output data\canonical\2026-08-20-sample.jsonl
```

Determinism rerun used the same command with output
`data\temp\2026-08-20-sample-rerun.jsonl`; both outputs produced the same
SHA-256. Focused and full validation commands:

```text
python -m pytest -q tests/test_replay_daily.py
python -m pytest -q
git diff --check
```

A corpus-wide programmatic assertion pass read all 15 raw replays and the
canonical JSONL, checking the invariants below; it returned `ok: true` with no
errors.

## Corpus-wide results

- Every record is parseable JSONL, has `module_version == "1.32.7"`, and retains
  manifest `avg_score`/`min_score`.
- Every episode has both seats and days `0..29`; every day-0
  `previous_execution` is zeroed.
- Starts are explicit `hour == 0`; non-final days end at the next day’s hour 0;
  day 29 ends at terminal hour 23 with no day-30 observation.
- Recursive opponent-public scan covered 1,800 start/end views with no
  `shed`, `seeds`, `inventories`, or `private` payloads.
- End-board crop and animal counts, land targets, hire feedback/Fibonacci costs,
  fertilizer entries/counts, and ordered land events all reconciled.
- 13,283 exact SELL events reconciled into the six bins `0/4/8/12/16/20`.
- 68 land events, 2,020 fertilizer applications, and 1,086 duplicate-shop
  views were checked.
- 360,000 plant/animal lifecycle tile views matched the pinned 1.32.7 formulas.

The extractor uses the verified alignment rule: `steps[i][seat].action`
transforms `steps[i-1][seat].observation`, so the event belongs to the
preceding observation’s day/hour. An hour-23 action stored on the next-day
hour-0 step therefore remains in the prior day’s ledger.

## Representative inspection

Episode `94735084` day 0 for both seats starts at `[0, 0]`, ends at
`[1, 0, next_day_start]`, has zero previous execution, and retains the
hour-1 WHEAT sale of quantity 3.

- Hire example: episode `94735084`, seat 0, day 0 submitted 5 hires and
  realized 5 at Fibonacci total cost 12.
- Land example: seat 0, day 6 records `BUY_LAND` at hour 4 as `NE`; the end
  target reports expansion with new quadrant `NE`.
- Fertilizer example: seat 0, day 14 records two STRAWBERRY and two WHEAT
  applications with exact tile/hour entries.
- Duplicate shops: seat 0, day 20 preserves `ICE_CREAM_SHOP` twice in ordered
  `unlocked_shops` and reports count 2.
- Lifecycle examples: a WHEAT tile at age 1 reports one day to harvest; a
  SHEEP tile reports five days to next product under its verified schedule.
- Episode `94735084` day 29 is retained for both seats and ends at terminal
  `[29, 23]`.

No score cutoff, reactivity filter, training run, executor, or broader dataset
processing was introduced. Raw replays and generated JSONL remain ignored and
local.

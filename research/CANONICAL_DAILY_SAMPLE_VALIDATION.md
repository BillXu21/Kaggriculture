# Canonical Daily Sample Validation

Date: 2026-08-21
Engine: Kaggriculture `1.32.7`
Source partition: `kaggle/kaggriculture-episodes-2026-08-20`

## Artifacts and commands

The 15 local replays were extracted into the ignored artifact:

`data/canonical/2026-08-20-sample.parquet`

Parquet is now the production canonical physical format (one row per
`(episode, seat, day)` record, Zstandard compression, native nested Arrow
structs/lists; see `replay_daily/storage.py`). JSONL remains an optional
debug/inspection CLI output and is no longer maintained as a duplicate
artifact. Raw replays remain the source of truth.

- 900 records: 15 episodes × 30 days × 2 seats;
- regenerated at canonical schema v2 on 2026-08-22 (CARE-by-animal ledger);
  806,735 bytes;
- SHA-256 (v2): `F7176542FE34B72DCEFCF70799DEDC34F17D8DB2DBF372680BD0FEC597023441`.

Generation:

```text
python -m replay_daily extract --input data\samples\2026-08-20 --manifest data\samples\2026-08-20\manifest.csv --source-dataset kaggle/kaggriculture-episodes-2026-08-20 --partition-date 2026-08-20 --output data\canonical\2026-08-20-sample.parquet
```

Training-time access without this package:

```python
import pyarrow.parquet as pq

table = pq.read_table("data/canonical/2026-08-20-sample.parquet")
meta = table.column("metadata").to_pylist()      # episode/seat/scores/provenance
start = table.column("start").to_pylist()[0]     # boards/market/town state
sell_bins = table.column("targets").to_pylist()[0]["sell_quantity"]
tile = start["self"]["board"][0][0]              # tagged tile struct with derived lifecycle
```

`replay_daily.read_parquet(path)` reconstructs canonical logical records that
compare exactly equal to fresh extractor output.

## Schema v2 CARE regeneration (2026-08-22)

The artifact was regenerated at canonical schema v2 after the CARE-by-animal
correction (`events.care` ledger + `targets.care_by_animal`; readers/writers
fail loudly on v1/mixed processed data). All 900 records again compare with
exact Python equality against fresh extraction. Corpus-wide CARE results:

- 6,642 known-animal CARE events: COW 3,724, SHEEP 2,918, GOOSE 0 (no geese in
  the local elite sample; GOOSE attribution is covered by synthetic tests);
- 3,083 unknown CARE intents preserved with `animal: null` (empty pasture,
  empty/locked tiles) and never counted as a species;
- every record satisfies `targets.care_by_animal == events.care.by_animal`;
- CARE no longer appears in `worker_ops_other`;
- spot-checked raw adjacency: e.g. episode 94735084 seat 0 step 13 CARE at
  pre-action position `[3, 3]` on a SHEEP tile, hour 12, appears verbatim in
  the day ledger.

## Parquet full-sample parity and benchmark (2026-08-21)

Single run, `time.perf_counter`, single process. Environment: Python 3.13.1,
PyArrow 24.0.0, 20 logical CPUs, Windows.

| Metric | Value |
| --- | --- |
| Raw replays (15 × `.json`) | 465,360,390 bytes |
| Previous validated JSONL | 67,686,853 bytes |
| Parquet (zstd) | 795,154 bytes |
| Size reduction vs JSONL | 98.8% (ratio 0.012) |
| Extract 15 replays (parse + extraction) | 5.22 s — 172 rec/s, 89.1 MB/s raw |
| Parquet write | 4.14 s — 217 rec/s |
| Parquet read + logical reconstruction (`read_parquet`) | 20.42 s — 44 rec/s |
| Raw Arrow read (`pq.read_table`) | 0.59 s — 1,537 rec/s |
| JSONL read (before removal) | 1.21 s — 747 rec/s |

Projected single-process preprocessing throughput is roughly **0.3 hours per
100 GiB of raw replays** at the measured 89 MB/s, so replay-level parallelism
is not justified for the planned corpus scale; simplicity was preferred.
Full logical dict reconstruction is slower than raw Arrow reads because it
rebuilds Python dicts per record; training code should consume Arrow columns
directly and use `replay_daily.read_parquet` only when logical records are
needed. The large size reduction and clean nested round-trip removed any need
to evaluate NPZ or other alternatives.

Parity evidence: all 900 records reconstructed from Parquet compare with exact
Python equality against both freshly extracted in-memory records and the
previously validated JSONL artifact (verified before its removal). The only
earlier mismatch was a provenance string (`metadata.source_path` recorded as
repo-relative in the original validated run vs absolute in the first rerun);
regenerating with the original relative-path invocation produced exact equality,
confirming the difference was invocation provenance, not schema or content.

Corpus assertions re-run on the Parquet-reconstructed records:

- 15 × 2 × 30 = 900 unique `(episode_id, seat, day)` keys; days `0..29` for
  every seat of every episode; all `module_version == "1.32.7"`.
- Day 29 retained for both seats with terminal boundary and non-null final
  banks.
- All 13,283 exact SELL events reconciled into the six bins anchored at
  `0/4/8/12/16/20` (total quantity 1,130,784).
- 14,342 null lifecycle values preserved on start boards (never coerced to
  zero); BUY_LAND quadrant-null preservation is covered by focused synthetic
  tests (this sample contains no failed land purchase).
- 528 records carry duplicate town shops with preserved multiplicity.
- Recursive opponent-public privacy scan over all 1,800 start/end views found
  no `shed`/`seeds`/`inventories`/`private` payloads.

After parity verification the old ignored JSONL sample
(`data/canonical/2026-08-20-sample.jsonl`, 67,686,853 bytes) was deleted so
Parquet is the sole maintained local production artifact.

## Original JSONL validation record

The original validation pass generated
`data/canonical/2026-08-20-sample.jsonl` (67,686,853 bytes, SHA-256
`0d9976703503da3aaa0dcc9a51c691df80195ba9864420346c90b9068699c2e7`) via the
JSONL CLI output; that artifact has since been superseded by the Parquet file
above after exact-parity confirmation. Focused and full validation commands:

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
processing was introduced. Raw replays and generated canonical artifacts remain
ignored and local.

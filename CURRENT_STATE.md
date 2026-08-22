# Kaggriculture Current State

Last updated: 2026-08-22

## Snapshot

- Repository: `BillXu21/Kaggriculture`
- Phase: **first BC manager implemented; five-day canonical schema-v3 corpus regenerated and verified; adapter audit / private Kaggle dataset versioning are the immediate handoff before the first real BC run**
- Competitive training: not started
- Latest confirmed upstream package: `kaggle-environments 1.32.7`
- Training direction: **behavior cloning first, then PPO/RL refinement; scratch PPO remains a useful comparison because the manager horizon is only ~30 steps**
- Primary project goal: demonstrate a self-play/refinement pipeline that measurably improves a competent starting policy across multiple promotions; a medal is not the primary objective.

## Learned-Control Abstraction

The learned policy is a **once-per-day farm manager**, not a primitive 720-turn controller. The strategic horizon is therefore roughly 30 decisions per episode.

V0 manager ownership:

- crop composition / planting allocation;
- target animal counts;
- land expansion;
- fertilizer allocation by crop type;
- CARE allocation by animal type;
- daily selling intent with six 4-hour timing bins.

Deterministic executor ownership:

- exact worker assignment/routing/movement;
- hiring enough workers to execute requested work;
- watering and routine maintenance;
- mechanical harvesting;
- seed purchases implied by crop targets;
- animal purchases / structures / placement implied by animal targets;
- simple initial wheat/feed procurement;
- exact tile/animal choice for fertilizer and CARE;
- legal primitive execution of requested sell intent.

The executor should compile learned economic intent, not replace it with its own farm strategy.

See D-011 and `.agents/notes/implemented/2026-08-21-use-daily-manager-with-deterministic-executor.md`.

## Canonical Daily Replay Contract (D-018)

One canonical record exists per `(episode, seat, day)`.

Each record contains:

- explicit start-of-day state using replay `day`/`hour` boundaries;
- end-of-day / next-day state;
- a compact daily event ledger;
- complete replay / score provenance;
- seats canonicalized as `self` / `opponent`.

The canonical dataset remains richer than the first model tensors so later action/observation adapters can change without reparsing raw 720-turn JSON.

Important contracts:

- full 10x10 board plus crop/animal lifecycle state is retained;
- opponent PUBLIC state is retained even though V0 masks it;
- simulator worker positions are `[x, y]` while board/ledger coordinates are canonical `[y, x]`; tile-dependent lookups must use `tiles[y][x]` after `x, y = pos`;
- strategic labels include crop composition, animal counts, land state, fertilizer-by-crop, CARE-by-animal, and six-bin selling;
- exact event timing remains in the ledger;
- compressed nested Parquet is the production format;
- canonical schema versions are fail-loud semantic boundaries; incompatible processed files are regenerated from raw rather than silently migrated.

Current canonical schema version: **3**.

## Five-Day Schema-v3 Corpus

The complete elite 1.32.7 corpus for 2026-08-17 through 2026-08-21 has now been regenerated and verified on Kaggle.

| date | episodes | seat-days |
| --- | ---: | ---: |
| 2026-08-17 | 699 | 41,940 |
| 2026-08-18 | 697 | 41,820 |
| 2026-08-19 | 695 | 41,700 |
| 2026-08-20 | 698 | 41,880 |
| 2026-08-21 | 697 | 41,820 |
| **total** | **3,486** | **209,160** |

Final corpus checks:

- 5 Parquet files;
- 209,160 total rows;
- schema versions exactly `{3}`;
- regeneration wall time: 1.08 h;
- Kaggle output: `/kaggle/working/kaggriculture-canonical-v3`.

Full-corpus CARE attribution:

- COW: 1,309,686;
- SHEEP: 892,397;
- GOOSE: 14,996;
- unknown: 13,045 (~0.585% of CARE submissions).

Fertilizer unknown attribution totals 2,666 across all five partitions. The residual unknown tail is small and not currently a blocker for V0 BC; unknown labels remain honest rather than fabricated.

Detailed corpus note: `research/FIVE_DAY_V3_CORPUS.md`.

## First BC Manager (D-019)

Implemented under `bc_manager/`.

Architecture:

- stateless one-day-in / one-plan-out policy;
- 100 own-board tile tokens with a shared spatial tile encoder;
- MANAGER + SELF RESOURCE + MARKET + TOWN + LABOR + DAY context in one standard PyTorch TransformerEncoder;
- opponent PUBLIC board optional and off by default;
- own private shed/seeds/inventory included; opponent private state has no feature path;
- structured crop / animal / land / fertilizer / CARE / selling heads;
- sell presence plus masked log1p quantity regression using per-event 0..100 bounded intent in the BC adapter;
- seven group-balanced losses.

Default model:

- `d_model=128`
- 4 layers
- 4 heads
- FFN 384
- ~1,071,040 trainable parameters

Tiny CPU validation config is also implemented. Full suite after schema-v3 correction: **102 tests pass**; tiny-batch overfit and checkpoint reload are validated.

Default data protocol:

- train: 2026-08-17 through 2026-08-20;
- validation: 2026-08-21;
- `min_score >= 2950`;
- never random seat-day splitting;
- report train-only empirical day baseline beside model metrics;
- sparse nonzero diagnostics make tomato/goose/etc. collapse visible.

`bc_manager/README.md` now contains the schema-v3 command and correctly passes all five Parquets, including the held-out Aug-21 validation file.

No full five-day BC training run has been completed yet.

## Corpus Findings Relevant to BC

The earlier five-day audit found substantial strategy duplication:

- 6,972 seat trajectories;
- 2,342 unique exact crop/animal/land trajectories;
- ~66.4% exact duplicate rate;
- largest exact family: 786 trajectories.

This is why random seat-day validation is prohibited and why the day-only baseline matters.

Default `min_score >= 2950` previously yielded ~31,200 seat-days across all five partitions; the schema-v3 adapter audit should reconfirm the exact current train/validation counts before the first run.

Crop composition changed on ~86% of non-day0 decisions in the prior audit, supporting the once-per-day manager rather than a single episode-level strategy choice.

## Replay Selection (D-017)

Use Kaggle's daily **top-rated** episode datasets rather than broad/random ladder data.

Initial corpus rules:

- five recent complete post-patch 1.32.7 partitions beginning 2026-08-17;
- embedded `module_version == 1.32.7` is authoritative;
- retain both seats when the episode's `min_score` clears the configured threshold;
- preserve score/provenance metadata but never feed demonstrator identity/result metadata to the model;
- do not impose a reactivity/diversity filter until measurements justify it.

## Training Progression

1. **Current:** version/upload the verified schema-v3 corpus to the private Kaggle dataset and audit the exact `bc_manager.adapter` train/validation arrays.
2. Run the first ~1.071M date-held-out BC training and compare against the train-only day baseline.
3. Build the deterministic executor / foreman and evaluate BC closed-loop; teacher-forced accuracy alone is insufficient.
4. Compare BC-initialized PPO against scratch PPO under the same executor/opponents/budget.
5. Demonstrate improvement against a frozen opponent on held-out seeds.
6. Expand to a frozen opponent panel / cross-play.
7. Only after those stages learn reliably, introduce changing opponents/population/self-play and richer opponent/temporal/value machinery.

Do not add another layer of RL complexity until the simpler stationary problem underneath it demonstrably learns.

## Deterministic Executor: Next Research Subsystem

Implementation has not started.

Current direction to investigate after the first BC run is underway:

- infer useful compact crop/livestock layouts from elite replay board states;
- likely keep service-heavy livestock close to the shed and crops in the next compact region;
- preserve stable preferred slots rather than relayout the farm every day;
- task generation -> standing-on-useful-work actions -> small worker/task assignment problem -> primitive navigation;
- use simple assignment/routing first and only move toward VRP/facility-location optimization if measured executor completion is insufficient.

The manager should choose economic targets; the executor should minimize mechanical work needed to realize them.

## Current Engine / Economics

Latest confirmed upstream package: `kaggle-environments 1.32.7`.

Relevant 1.32.7 change:

- carrot, tomato, and egg use scarcity-side hinge curves;
- product value is state-dependent on shop demand, shared market inventory, opponent production, and time remaining;
- therefore no static product ranking should be encoded;
- naive inventory × spot-price reward shaping is unsafe because liquidation changes the price curve.

See `MECHANICS.md`, D-015, and D-016.

## Immediate Priorities

1. Finish private Kaggle dataset versioning with the five schema-v3 Parquets.
2. Audit the exact BC adapter outputs: train/validation counts, score ranges, CARE sparsity, bounded selling, and target ranges.
3. Launch the first real ~1.071M BC run in the foreground with date-held-out validation and a train-only day baseline.
4. While BC work runs, study elite spatial layouts / public executor approaches before implementing the deterministic foreman.

## Known Risks

- BC may mostly memorize common calendar scripts; the day-only baseline is the direct diagnostic.
- Sparse targets such as goose/tomato can collapse to zero without obvious aggregate-loss failure; keep nonzero recall visible.
- The daily abstraction may omit strategically important intra-day decisions; selling is the first explicit higher-frequency exception.
- The executor could accidentally become the strategist if it overrides manager economic intent.
- Opponent hidden inventory creates partial observability; richer opponent modeling is deferred until basic learning works.
- Non-transitive strategy matchups can make simple latest-checkpoint/Elo promotion misleading.
- Context loss across chats can waste major work; GitHub continuity remains mandatory.

## Do Not Forget

Before substantial work, read:

- `CURRENT_STATE.md`
- `DECISIONS.md`
- `MECHANICS.md`
- relevant `.agents/notes/implemented/`
- `research/RL_DESIGN.md`
- latest `HISTORY.md`

Before expensive runs, record the exact command/configuration, code+engine identity, data/version/filter, seeds/opponents where relevant, expected outputs, stop conditions, and recovery plan.

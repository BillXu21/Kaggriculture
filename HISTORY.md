# Kaggriculture Historical Record

This file is append-only except for correcting factual errors. New entries are added in reverse chronological order.

## 2026-08-22 — Canonical Schema v3: Official Worker `[x,y]` Tile-Lookup Correction

Correctness fix under D-018 (commit `e67f1b7`; no architecture change; D-019
untouched).

- **Root cause:** official 1.32.7 worker positions are `[x, y]` with board
  lookup `tiles[y][x]`, but `_events_from_action` unpacked them transposed
  (`y, x = pos[0], pos[1]`), so every tile-dependent worker event read the
  transposed tile. Affected: CARE animal attribution, FERTILIZE crop
  attribution, HARVEST item attribution, and DIG replaced-tile labels. PLANT
  is action-provided and was never affected.
- **Fix:** `x, y = int(pos[0]), int(pos[1])`; bounds checked in x/y;
  `tile = tiles[y][x]`; emitted ledger tile coordinates remain canonical
  `[y, x]`. Honest unknown behavior preserved: CARE stays `animal: null`
  unless the actual pre-action tile establishes GOOSE/COW/SHEEP; a decoy at
  the transposed tile can never be attributed.
- **Schema:** `SCHEMA_VERSION` bumped 2 → 3 (single authoritative constant).
  Parquet storage and the BC adapter expect v3 via the imported constant;
  v1/v2/mixed logical records and Parquet files fail loudly in both
  `replay_daily.storage` and the BC loader. No migration: regenerate from raw.
- **Asymmetric regression tests:** worker at `[x=2, y=5]`, actual tile only at
  `board[5][2]`, deliberately different decoy at `board[2][5]`, across CARE
  (species + honest unknown + hand actor + both seats), FERTILIZE, HARVEST,
  DIG, and PLANT coordinate semantics. Real-sample smoke expectation updated
  to the official convention. Full suite: 102 tests pass.
- **Local sample audit:** regenerated
  `data/canonical/2026-08-20-sample.parquet` at schema v3 (900 records,
  805,435 bytes, SHA-256
  `932617FF02EF7B5DF74C5AF2E766F3EC3423B3FAC24513992E218C0629F4054E`);
  exact read-back parity; every raw pre-action worker op reconciled against
  the emitted ledgers with zero mismatches. Corrected counts: CARE COW 5,614 /
  SHEEP 4,091 / GOOSE 20 / unknown 0 (9,725 entries); FERTILIZE 2,020 entries
  (known 2,011 = STRAWBERRY 1,960, WHEAT 27, TOMATO 24; unknown 9); DIG 889
  submissions; HARVEST 11,948 submissions with 11,905 item-bearing ledger
  entries (43 no-item submissions intentionally omitted under the existing
  item-only HARVEST ledger semantics). Total tile-dependent submissions:
  **24,582**.
- **Erratum:** the implementation commit report/message stated 14,582
  reconciled worker ops; the correct total is 24,582 (CARE 9,725 +
  HARVEST 11,948 + DIG 889 + FERTILIZE 2,020). The reconciliation itself was
  complete; only the reported aggregate was wrong. Recorded here without
  amending history.
- The v2 event labels were semantically wrong (transposed lookups), so no
  migration path exists or is desired; all processed corpora must be
  regenerated from raw replays at v3 before BC training.

## 2026-08-22 — First BC Manager: Adapter/Baseline, Tile Transformer, Training CLI

Implemented the complete first behavior-cloning stack over the canonical
schema-v2 records; D-019 published. No full training run (five-day v2 corpus
pending).

- **Compact BC data layer** (66fbaea): bc_manager/adapter.py reads
  schema-v2 Parquet directly with PyArrow (dotted-path projection, no logical
  reconstruction), verifies schema_version == 2, selects rows only by the
  date allowlist + equal min_score cutoff, and converts once into compact
  NumPy arrays (own/opponent-public boards, resource/market/town/labor/day
  features, count/CARE/sell targets). bc_manager/baseline.py fits a per-day
  empirical baseline on train rows only. Date-held-out splits; never random.
- **Tile Transformer + loss** (6b9db3b): stateless day/hour0 manager —
  shared tile encoder (kind/crop/animal embeddings, scaled lifecycle
  numerics with NaN indicators, bool/presence channels, row/col embeddings),
  MANAGER + 5 global tokens (106 sequence length), standard norm-first
  TransformerEncoder, structured heads for crop/animal/land/fertilizer/CARE
  counts plus sell presence and log1p quantity. Seven fixed-weight group
  losses; metadata keys rejected loudly; opponent PUBLIC board optional/off
  by default. Default config 1,071,040 trainable parameters; tiny CPU
  config for tests.
- **Training CLI** (86b8433): python -m bc_manager.cli — in-RAM tensor
  dataset, AdamW + gradient clipping + optional CUDA AMP, sparse diagnostics
  (exact/MAE/nonzero recall incl. per-animal GOOSE visibility) beside the
  train-only day baseline, early stopping, atomic best/last checkpoints that
  serialize model config and reload to equivalent eval outputs.
- **Validation:** 92 tests pass (65 data-layer, 16 model/loss, 11 training)
  including real forward/backward on the local 900-row v2 sample, genuine
  tiny-batch overfit (~770x loss reduction), checkpoint equivalence, and a
  synthetic two-date end-to-end CLI smoke. Full five-day training was not
  run; old v1 data fails loudly everywhere.
- **Decision publication:** D-019 added to DECISIONS.md; implemented note
  .agents/notes/implemented/2026-08-22-use-configurable-tile-transformer-for-initial-bc-manager.md;
  usage/handoff commands in bc_manager/README.md.

## 2026-08-22 — Canonical Schema v2: CARE-by-animal Correction

Logical extension/correction under D-018 (no redesign; D-017/D-018 unchanged).

- `events.care` now records every submitted CARE intent with its pre-action
  tile `[y, x]`, the animal identity established by that board tile (or
  `null`), and the exact primitive hour. CARE previously fell into the generic
  `worker_ops_other` aggregate with no species attribution.
- CARE takes no arguments: it targets the worker's own pre-action tile under
  the verified alignment rule (`steps[i].action` transforms `obs[i-1]`).
  Unknown/non-animal CARE stays `animal: null` and never increments a species.
- New derived target `targets.care_by_animal` exactly mirrors the known-animal
  daily counts for GOOSE/COW/SHEEP.
- `SCHEMA_VERSION` bumped 1 → 2 in `replay_daily/constants.py`. Parquet Arrow
  schema, conformance guards, normalization, reconstruction, and round-trip
  equality extended for CARE. Writers reject logical records with a foreign
  `schema_version`; `read_parquet`/JSONL readers fail loudly on v1 or
  mixed-version processed data. No migration machinery: regenerate from raw.
- Real-sample validation: all 15 local replays re-extracted to v2 Parquet
  (900 records, exact read-back parity). 6,642 known CARE events (COW 3,724,
  SHEEP 2,918) plus 3,083 unknown intents preserved as `animal: null`; GOOSE
  does not occur in the local elite sample and is covered synthetically.
  Regenerated artifact: `data/canonical/2026-08-20-sample.parquet`, 806,735
  bytes, SHA-256 `F7176542FE34B72DCEFCF70799DEDC34F17D8DB2DBF372680BD0FEC597023441`.
- 57 focused tests pass (13 new covering CARE attribution, hours, seats,
  privacy, merge/target mirror, round-trip with null animal, no double count,
  and v1/mixed rejection). Evidence in
  `research/CANONICAL_DAILY_SAMPLE_VALIDATION.md`.

## 2026-08-21 — Parquet Production Storage for Canonical Records

Adopted Zstandard-compressed Parquet as the production canonical physical
format under D-018; the logical `(episode, seat, day)` schema is unchanged.

- `replay_daily/storage.py` maps one logical record to one nested Arrow row;
  fail-loud conformance guards reject unknown canonical keys instead of
  silently dropping them, and bare string tile sentinels round-trip exactly.
- CLI `extract` defaults to `--format parquet`; JSONL remains an explicit
  debug/inspection output and is never written automatically.
- Full-sample validation: all 900 records from the 15-replay sample compare
  with exact Python equality across fresh extraction, the previously validated
  JSONL, and the new `data/canonical/2026-08-20-sample.parquet` (795,154 bytes,
  98.8% smaller than JSONL).
- Benchmark (single process): extraction ~89 MB/s of raw replay (~0.3 h per
  100 GiB projected), so no parallel preprocessing stage was justified; raw
  Arrow reads are faster than JSONL parsing. No NPZ evaluation needed.
- The old ignored JSONL sample was deleted after parity confirmation; raw
  replays remain the source of truth. Evidence in
  `research/CANONICAL_DAILY_SAMPLE_VALIDATION.md`. PyArrow dependency declared
  in `requirements.txt` (`pyarrow>=14`). 44 tests pass.

## 2026-08-21 — Canonical Daily Sample Validated

Completed the local 1.32.7 canonical replay foundation and generated the ignored
15-replay sample at `data/canonical/2026-08-20-sample.jsonl`.

- 900 records cover both seats and all 30 days for each replay;
- corpus-wide boundary, privacy, lifecycle, hire, land, fertilizer, shop, and
  six-window SELL checks passed;
- the output is deterministic across a second CLI run;
- representative findings are recorded in
  `research/CANONICAL_DAILY_SAMPLE_VALIDATION.md`.

Training and the deterministic executor remain unstarted; no broader dataset
processing was performed.

## 2026-08-16 — 1.32.7 Situational Resource Rebalance

### Upstream engine change

Reviewed and source-confirmed merged upstream PR `Kaggle/kaggle-environments#1399` (`Make underused resources situational`).

Upstream `pyproject.toml` now declares `kaggle-environments` version `1.32.7`.

PR metadata:

- head commit: `1fbd3b7571653434329d288dee9e068f54ff01c0`;
- merge commit: `28b6d8af3ce73926b3d0fda1410c1ddd8384ab8c`.

The host stated this should be the last balance change except game-breaking bugs. Live leaderboard rollout was announced but has not been independently server-locked in this repository.

### New hinge price curve

PR #1399 adds a new scarcity-side market function:

`u = x / T`

`hinge = u + 8 * max(0, u - 1)^2`

where `x = I0 - market_inventory` when a product is scarce.

Behavior:

- below `T`, the curve is linear in normalized scarcity;
- above `T`, a quadratic term creates a steep price spike;
- `hinge(T) = 1`, preserving the meaning of the market target parameter.

### Products changed

- carrot: scarcity curve `log`/0.20 → `hinge`/1.00, `T=450`, knee inventory 9550;
- tomato: `linear`/0.40 → `hinge`/0.40, `T=200`, knee inventory 9800;
- egg: `linear`/0.40 → `hinge`/0.40, `T=332`, knee inventory 9668;
- glut-side curves are unchanged.

Tomato and egg are therefore unchanged through their old linear knee and diverge only in deeper scarcity. Carrot changes more broadly because its scarcity target also increases from 0.20 to 1.00.

Explicit source test values include:

- carrot: 9550 → $70, 9400 → $113, 9100 → $385;
- tomato: 9800 → $84, 9700 → $144, 9500 → $552;
- egg: 9668 → $70, 9502 → $120, 9170 → $460.

### Random-shop interaction

Relevant shop demand noted by the PR:

- carrot: pet cafes and farmers markets; pet cafe is single-product and consumes double;
- tomato: pizza shops and farmers markets;
- egg: bakeries and brunch spots.

Because shops have been sampled with replacement since 1.32.6, duplicate demand can push these products through their scarcity knees.

Host-reported substantial-price-increase frequencies assuming **no production**:

- tomato: ~50% of games;
- carrot: ~26%;
- egg: ~22%.

These are recorded as host-reported statistics, not engine constants, and should be reproduced empirically under the locked local engine.

### Strategic interpretation

The change strengthens the RL-centered design rather than weakening it.

The new decision problem is conditional:

- detect that an episode is developing an unusual demand regime;
- estimate whether a crop/animal pivot can produce before the opportunity disappears;
- account for the fact that our own production/sales reduce scarcity;
- anticipate whether the opponent is already producing or will react;
- decide whether the expected competitive gain exceeds the opportunity cost of changing the farm plan.

This is especially relevant to end-game crop rotation and situational goose/egg production.

### Reward-design consequence

Added a durable guardrail against naive mark-to-market shaping.

Under the hinge curve, `quantity × current spot price` can hugely overstate realizable value because selling quantity moves the price back toward the knee. Future reward/potential design should use marginal-price-aware liquidation, time-to-sale constraints, exact/approximate simulation, or validated learned continuation value.

### RL observation consequence

Planned product entities should expose:

- current inventory/price;
- base, `I0`, `T`;
- curve shape and target parameters;
- normalized scarcity;
- signed distance to the knee;
- recent inventory/price velocity;
- shop multiplicity/known demand;
- own/opponent production pipeline;
- time remaining.

No static `GOOD_PRODUCT` feature or fixed product-priority table should be encoded.

### New pre-training studies

1. reproduce the host-reported no-production scarcity frequencies;
2. measure first knee-crossing time and maximum prices by shop composition;
3. estimate latest profitable pivot time for carrot, tomato, and goose/egg production;
4. measure how opponent production suppresses the opportunity;
5. test public deterministic baselines for 1.32.7 staleness;
6. verify reward potentials do not exploit temporary hinge spot prices.

### Files updated

- `CURRENT_STATE.md`
- `MECHANICS.md`
- `PLANS.md`
- `DECISIONS.md`
- `HISTORY.md`
- `research/RL_DESIGN.md`

## 2026-08-07 — 1.32.6 Town Rebalance and RL-Centered Planning

### Upstream engine change

Reviewed and source-confirmed merged upstream PR `Kaggle/kaggle-environments#1394` (`Kaggriculture town rebalance`).

The change:

- changes default `townCenterSellInterval` from 12 to 24 turns;
- therefore reduces default town-center consumption from twice/day to once/day;
- removes the old town-center demand schedule that increased to 2× after day 10 and 4× after day 20;
- samples town shops with replacement from the full shop table;
- allows duplicate shop names;
- makes each duplicate shop instance consume independently;
- caps total unlocked shop instances at 8 as before in effective maximum count.

Confirmed upstream package source snapshot `bded87b0d7879078c726a93a4884d044f79c4eed` identifies `kaggle-environments` as version `1.32.6`.

The live leaderboard rollout was announced but has not yet been independently locked to an observed server build in this repository.

### Strategic interpretation

The rebalance increases the value of adaptive economic behavior:

- weaker town-center demand means player-generated oversupply should persist longer;
- product gluts and opponent sale timing matter more;
- shop replacement sampling creates materially different per-episode demand regimes;
- duplicated shops can strongly favor particular product categories;
- fixed deterministic public schedules should become less universally optimal across seeds.

The town-shop observation must be treated as a multiset/count vector rather than a binary set.

### RL direction clarified

The project direction changed from "deterministic route first, learning later if useful" to an explicitly **RL-centered hybrid**.

The intended division of responsibility is now:

- learned policy owns production, resource allocation, task assignment, adaptation, and market strategy;
- deterministic infrastructure owns pathfinding, mechanical legality, task execution/persistence, and bookkeeping;
- candidate generation may remove impossible actions but should not encode strategic preferences by hiding mechanically valid actions.

Raw primitive movement PPO from scratch remains deferred. This is now understood as an action-abstraction decision, not a rejection of RL.

### Public RL discussion considered

A competitor reported poor results from standard PPO/SAC attempts because of:

- large observation/action spaces;
- long crop reward delays;
- catastrophic cascades from small logistical mistakes;
- difficulty learning exact watering/feed/seed timing through random exploration.

This was treated as evidence for imitation bootstrap and hierarchical action abstraction rather than evidence against RL.

### New RL design

Added `research/RL_DESIGN.md` covering:

- hierarchical worker-task intent actions;
- deterministic execution of selected intents;
- dedicated autoregressive market head;
- action masking rules;
- turn-level vs event-driven vs hybrid decision frequency;
- entity-based observation/model design;
- recurrent opponent-state inference;
- W/L/T terminal objective;
- potential-based reward shaping;
- auxiliary prediction losses;
- public-agent behavior cloning;
- PPO robustness training and population self-play;
- pre-training experiments for shop variance, market sensitivity, action abstraction, reward sanity, and memory.

### Reward planning

Current leading reward direction:

- final competitive objective aligned with win/tie/loss;
- avoid arbitrary positive maintenance rewards such as watering/harvesting bonuses;
- investigate potential-based shaping using liquidation/future economic value;
- use auxiliary prediction tasks for representation learning instead of silently changing the objective;
- compare `gamma=1.0` with values extremely close to one because the true objective is terminal.

### Demonstration/bootstrap plan

Strong deterministic public agents will be used as training data in addition to opponents:

1. archive exact agent/version provenance;
2. run over varied seeds/shop regimes/opponents;
3. collect state/action trajectories;
4. map primitive actions into intent-level labels;
5. behavior-clone initial competence;
6. fine-tune with PPO/self-play so the model can depart from fixed public scripts.

### Next-week planning agenda

While Pokémon work finishes and Kaggriculture has time to stabilize:

1. map exact 1.32.6 actions and RNG;
2. design worker-task candidate generation;
3. design market quantity/order representation;
4. decide policy decision frequency;
5. version the observation schema;
6. formalize potential functions/reward invariants;
7. design BC trajectory format;
8. design PPO/self-play curriculum;
9. estimate simulator/vectorization throughput;
10. define evaluation/promotion gates;
11. recheck upstream engine changes before implementation/training.

### Files updated

- `README.md`
- `CURRENT_STATE.md`
- `PLANS.md`
- `DECISIONS.md`
- `MECHANICS.md`
- `HISTORY.md`
- new `research/RL_DESIGN.md`

## 2026-08-06 — Repository Initialization

### Repository

- Created private GitHub repository: `BillXu21/Kaggriculture`.
- Initialized the default branch with project documentation.
- Established continuity files to reduce context loss between chats and agents.

### Strategic Assessment

- Current game structure appears highly deterministic.
- Physical farms are separate, with limited direct interaction.
- The shared market is the primary adversarial coupling mechanism.
- Strong public leaderboard entries are currently dominated by copies or variants of a few deterministic public notebooks.
- The project will remain in planning and mechanics-tracking mode while the engine and rules continue changing.

### Initial Architecture Hypothesis

The initial architecture hypothesis was:

1. deterministic production-route executor;
2. state-based validation and repair;
3. phase-level replanning;
4. opponent-aware market and production policy;
5. coherent expert selection;
6. optional optimization or learning at the macro level.

This was superseded/clarified on 2026-08-07: the project now intends RL to own meaningful strategic decisions, with deterministic code limited primarily to mechanics/execution.

### Research Findings Carried Into the Repository

The following findings were established before repository initialization and should be reverified against the exact live engine before implementation:

- Two players each manage a separate 10×10 farm.
- Matches span thirty days with twenty-four turns per day.
- Banked money determines final reward.
- Unsold inventory has no terminal value.
- Crop and animal schedules are largely deterministic.
- The market is shared and uses inventory-dependent pricing.
- Some daily events are driven by the episode seed.
- Public state exposes enough opponent farm information to support strategy fingerprinting and supply forecasting.
- Strong public strategies use mixed industrial production rather than simple single-crop loops.

### Public Baseline Direction

The first competitive reference should be a strong public deterministic route, preserved with:

- source URL or notebook identity;
- download date;
- immutable file hash;
- engine version assumptions;
- any local modifications;
- known performance evidence.

The project will not rely on redistribution-license concerns as a reason to avoid downloading publicly available Kaggle notebook artifacts, but provenance and third-party boundaries should still be tracked accurately.

### Compute and Workflow Lessons Imported From Pokémon TCG Work

- Chat-context loss can cause stale configuration reuse and wasted compute.
- Every expensive run must be specified in a durable file before execution.
- Current state must remain concise and authoritative.
- Full history must preserve failed experiments, commands, hashes, and output paths.
- Evaluation should not depend on a single seat, weak opponents, or unversioned artifacts.

### Files Established

- `README.md`
- `CURRENT_STATE.md`
- `PLANS.md`
- `HISTORY.md`
- `DECISIONS.md`
- `MECHANICS.md`
- `AGENTS.md`
- `research/README.md`
- `.gitignore`

### Next Actions At Initialization

1. Establish the exact current engine identity.
2. Archive important public notebooks and agents.
3. Catalog major strategy families.
4. Define the initial fixed-seed, seat-swapped evaluation protocol.
5. Delay competitive implementation until those contracts are recorded.

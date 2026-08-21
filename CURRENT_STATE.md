# Kaggriculture Current State

Last updated: 2026-08-21

## Snapshot

- Repository: `BillXu21/Kaggriculture`
- Phase: **implement and validate the canonical daily replay/BC dataset around a daily high-level manager abstraction**
- Competitive training: not started
- Latest confirmed upstream package: `kaggle-environments 1.32.7`
- Exact local/vendored engine lock: not yet established
- Training direction: **behavior cloning first, then PPO/RL refinement; scratch PPO remains an important comparison because the manager horizon is only ~30 steps**
- Primary competition goal: build a refinement pipeline that produces meaningful, measurable improvement over a competent starting policy across multiple promotions; a medal is not the primary objective.

## Current Learned-Control Abstraction

The initial learned policy is a **daily farm manager**, not a primitive 720-turn controller.

The manager acts approximately once per game day, reducing the strategic horizon to roughly 30 decisions per episode.

For V0, RL owns economic intent:

- crop planting/allocation changes;
- target animal counts;
- land expansion;
- fertilizer allocation;
- daily selling quantity plus a simple intra-day timing plan.

The exact model output parameterization can still change. The current decision locks the division of responsibility rather than a neural architecture.

The deterministic executor is assumed competent and initially owns:

- worker assignment/routing/movement;
- worker hiring needed for the requested workload;
- watering and routine maintenance;
- mechanical harvesting;
- seed purchases implied by the crop plan;
- animal purchase/placement/structures implied by animal targets;
- simple wheat/feed procurement;
- primitive execution of the requested sell plan within its selected timing window.

The executor should compile learned intent, not replace it with its own economic strategy.

Wheat management is the leading candidate to move from heuristic control to RL after the first manager learns. Selling is the first subsystem allowed to retain finer-than-daily timing; more reactive selling or a separate higher-frequency selling policy can be tested later.

See `.agents/notes/implemented/2026-08-21-use-daily-manager-with-deterministic-executor.md`.

## Canonical Daily Replay Contract

Replay preprocessing now has a stable intermediate target: one canonical record per `(episode, seat, day)`.

Each record contains:

- start-of-day state using explicit replay `day`/`hour` boundaries;
- end-of-day/next-day state;
- a compact daily event ledger;
- full score/version/source provenance;
- seats canonicalized as `self` and `opponent`.

The canonical dataset is deliberately richer than the first model's tensors so alternative BC/action encodings can be generated without reparsing the raw 720-turn JSON.

Retain the full 10x10 farm board with mechanically descriptive lifecycle state. Crop records should preserve raw age/growth information plus useful derived timing such as time to next harvest/output, harvestability, fertilizer state, and water/dry/weed state as available. Animal records should preserve production cooldown/time to next product, feed/starvation, care/bonus, and other relevant engine state.

Keep opponent public state in the canonical data even if the first experiment masks it.

For V0, derive high-level labels for:

- crop composition/targets;
- animal-count targets;
- land expansion/state;
- fertilizer applications by crop type;
- sell quantity per product in six windows anchored at `0, 4, 8, 12, 16, 20`, corresponding to hours `0-3`, `4-7`, `8-11`, `12-15`, `16-19`, `20-23`.

Preserve exact primitive sale hours in the event ledger so a future 24-turn/reactive selling policy remains possible.

Also retain compact daily aggregates for planting/digging, fertilizer, harvests, animal/seed/product purchases, land purchases, workers/hire cost, and sales. These are audit/future-extension data, not necessarily V0 BC targets.

See D-018 in `DECISIONS.md` and `.agents/notes/implemented/2026-08-21-canonical-daily-replay-record.md`.

## Behavior-Cloning Plan

BC should learn **daily realized management decisions**, not primitive movement traces.

Exact movement, worker assignments, watering routes, harvest routes, seed-buy commands, animal-buy commands, and worker hires do not need to be BC targets when they are implied by the manager's high-level plan.

For the initial pipeline, treat the strong replay agents' low-level execution as effectively perfect. Replay preprocessing can therefore begin before our own deterministic executor is complete.

BC does not need to be broadly adaptive to be useful. A competent stereotyped plan is an acceptable initialization if PPO can subsequently improve and diversify it. However, top private agents already appear to contain state-conditioned branches, so preprocessing must preserve shop/market context rather than collapse examples to day-indexed schedules.

## Replay Corpus Decision

Use Kaggle's daily **top-rated** episode datasets, not random ladder data. These partitions are deliberately filled from the highest-average-rated available games until the daily ~20 GiB cap, so the corpus is already strongly selected for competitive play.

Initial selection:

- use five recent complete daily partitions from 1.32.7 after allowing roughly two days for competitors to adjust to the balance patch; target **2026-08-17 onward**, adding later complete partitions as they become available;
- require embedded `module_version == 1.32.7` when parsing each replay;
- retain manifest `avg_score` and `min_score` and choose the actual elite cutoff empirically from the distribution; values around 3000 are plausible, but the threshold is not locked yet;
- train on **both seats** when `min_score` is above the selected cutoff, because this guarantees both demonstrations are strong without needing unavailable per-seat submission IDs/ratings;
- retain `episode_id`, partition date, `avg_score`, `min_score`, derived `max_score`, player/seat, seed, final reward, and source provenance in the preprocessed dataset;
- do not apply a reactivity/diversity filter initially. First measure whether the elite score-filtered corpus already contains enough shop/market-conditioned variation, then compare score-only versus optional reactivity filtering if useful.

The official 1.32.7 patch commit was `28b6d8af3ce73926b3d0fda1410c1ddd8384ab8c` (`Make underused resources situational (#1399)`), created `2026-08-15T01:24:24Z`, and bumped package `1.32.6 -> 1.32.7` while adding the carrot/tomato/egg scarcity-side `hinge` curves.

See D-017 in `DECISIONS.md`.

## Training Progression

Current high-level direction:

1. Implement the canonical daily replay extractor locally against the already-downloaded 1.32.7 examples.
2. Manually inspect `(episode, seat, day)` rows and verify boundaries, lifecycle timing, end-state targets, fertilizer labels, worker costs, and six-window sales.
3. Attach/download the selected five top-daily partitions on Kaggle and scale the one-time preprocessing there.
4. Inspect crop/animal/land/fertilizer/selling diversity and shop-conditioned reactivity in the resulting daily table.
5. Train a first BC manager on elite demonstrations.
6. Verify BC in closed-loop games using the eventual executor; do not rely only on teacher-forced accuracy.
7. Compare BC-initialized PPO against scratch PPO under the same executor/opponents/training budget.
8. Demonstrate RL improvement against a frozen/controlled opponent over held-out seeds, initially with opponent features optionally masked.
9. Expand to a frozen opponent panel and broad cross-play evaluation.
10. Only after those stages learn reliably, introduce richer opponent modeling, changing opponents/population/self-play, and additional learned control such as wheat or reactive selling.

The project should not add another layer of RL complexity until the simpler stationary problem underneath it demonstrably learns.

## Current Engine/Economics

1.32.6:

- town center consumes once/day at flat 1x;
- town shops are sampled with replacement;
- duplicate shops consume independently;
- total shop instances are capped at 8.

1.32.7:

- carrot, tomato, and egg use scarcity-side `hinge` curves;
- these products can become extremely valuable only in favorable randomized shop/scarcity regimes;
- selling adds supply and therefore erodes scarcity, so spot price times inventory is not realizable liquidation value;
- opponent production affects the same shared market even if the early learner does not explicitly model the opponent.

See `MECHANICS.md` for exact parameters and regression points.

## Useful External Research

- `diffmap/kaggicultureRL` contains a serious Rust batch simulator, replay/BC infrastructure, parity/fuzz tests, and feature-engineering ideas. Its engine is pinned to 1.32.6 and lacks the 1.32.7 `hinge` market shape, so it cannot be adopted unchanged.
- Current public RL discussion shows non-transitive matchup structure among strong route/meta-agents, reinforcing the need for cross-play evaluation rather than a single scalar strength metric.
- Community reports of failed primitive end-to-end RL reinforce the daily-manager choice: low-level execution failures can make strategic actions such as land expansion appear negative-value to PPO.

## Immediate Priorities

1. Have a local implementation produce canonical daily records from the five already-downloaded replay examples.
2. Manually sanity-check representative early/mid/late days, especially lifecycle timing and the `0/4/8/12/16/20` selling bins.
3. Freeze a serialization format for the canonical daily dataset only after those rows look mechanically correct; do not prematurely freeze model tensors.
4. Scale preprocessing to the selected top-daily Kaggle partitions and retain broad score/provenance metadata so training-time filtering stays cheap.
5. Audit the elite corpus for repeated behavioral lineages and shop/market-conditioned action variation; do not filter on reactivity until measurements justify it.
6. Train the first BC manager while the deterministic executor is designed in parallel.
7. Later evaluate whether to vendor/port the `diffmap/kaggicultureRL` Rust engine to 1.32.7 and require parity before PPO training.

## Future Control Alternatives Preserved by the Dataset

The canonical record intentionally keeps enough information to test later variants without raw replay reprocessing:

- absolute targets versus daily action deltas;
- explicit per-tile crop replacement;
- age-bucketed/tile-specific fertilizer control;
- learned wheat/feed economics;
- strategic harvesting when labor/storage/timing makes it matter;
- richer previous-day workload/completion feedback;
- opponent-board/unobserved-holdings features;
- 24-turn selling, a separate reactive seller, or multiple manager calls per day.

## Known Risks

- The daily action abstraction could hide strategically important intra-day decisions; selling is the first planned exception and cadence can be increased later.
- The executor could accidentally become the strategist if it changes requested economic intent rather than merely executing it.
- Daily labels derived from realized replay outcomes may occasionally reflect execution failures rather than intended plans; elite traces are expected to make this acceptable initially.
- BC may still memorize calendar scripts if the elite corpus contains too many near-identical tapes; this is acceptable as a bootstrap unless it prevents PPO from improving.
- Multiple submissions from the same player can have very different strength, so player-name-only replay filtering is unsafe.
- Observation state aliasing can create apparently contradictory PPO gradients.
- Non-transitive opponent strategies can make simple Elo/latest-checkpoint promotion misleading.
- Context loss across chats can waste major amounts of work; GitHub continuity remains mandatory.

## Do Not Forget

- Read `CURRENT_STATE.md`, `DECISIONS.md`, `MECHANICS.md`, relevant Agent Notes, `research/RL_DESIGN.md`, and the latest `HISTORY.md` entry before substantial work.
- `research/ACTION_OBSERVATION_V0.md` is an older scratch primitive-action design and is not authoritative.
- Preserve raw replay data/configs/hashes so representations and label extraction can change without reacquiring data.
- Before any expensive run, record exact command/config, code+engine hashes, replay/version/score filter, seeds, opponent pool, expected outputs, and stop conditions.

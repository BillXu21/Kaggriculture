# Kaggriculture Current State

Last updated: 2026-08-21

## Snapshot

- Repository: `BillXu21/Kaggriculture`
- Phase: **begin replay/BC pipeline around a daily high-level manager abstraction**
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

The exact output encoding is still open. The current decision locks the division of responsibility rather than a specific neural architecture or action vocabulary.

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

## Initial Observation Direction

The representation should support a compact daily manager state while preserving spatial information:

- day / season progress and relevant town timing;
- full own 10x10 farm board;
- own money, shed, seeds, land, animals, and fertilizer state;
- shared market inventory/prices and town shop state;
- simple previous-day execution feedback such as workers hired, labor cost, and completion/failure information;
- opponent public board/state may be supported because the model runs only about 30 times per episode.

For the cleanest first PPO experiment, opponent features may still be masked so the learner first proves it can improve own-farm management. Explicit opponent hidden-inventory inference and deliberate market attacks remain later-stage work.

## Behavior-Cloning Plan

BC should learn **daily realized management decisions**, not primitive movement traces.

For each replay day, derive high-level labels such as:

- crop additions/removals or resulting allocation changes;
- animal-count changes;
- land expansion;
- fertilizer allocation;
- quantities sold and their intra-day timing windows.

Exact movement, worker assignments, watering routes, harvest routes, seed-buy commands, animal-buy commands, and worker hires do not need to be BC targets for the daily manager.

For the initial pipeline, treat the strong replay agents' low-level execution as effectively perfect. Replay preprocessing can therefore begin before our own deterministic executor is complete.

BC does not need to be broadly adaptive to be useful. A competent stereotyped plan is an acceptable initialization if PPO can subsequently improve and diversify it. However, top private agents already appear to contain state-conditioned branches, so replay preprocessing should preserve shop/market context rather than collapsing examples to day-indexed schedules.

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

1. Download/attach the selected top-daily replay partitions and manifests.
2. Define/extract daily manager state-action examples while preserving score/provenance metadata.
3. Inspect crop/animal/land/fertilizer/selling diversity and shop-conditioned reactivity in the resulting daily table.
4. Train a first BC manager on elite demonstrations.
5. Verify BC in closed-loop games using the eventual executor; do not rely only on teacher-forced accuracy.
6. Compare BC-initialized PPO against scratch PPO under the same executor/opponents/training budget.
7. Demonstrate RL improvement against a frozen/controlled opponent over held-out seeds, initially with opponent features optionally masked.
8. Expand to a frozen opponent panel and broad cross-play evaluation.
9. Only after those stages learn reliably, introduce richer opponent modeling, changing opponents/population/self-play, and additional learned control such as wheat or reactive selling.

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

1. Work out the exact daily BC label semantics for crops, animal targets, fertilizer, land, and sell windows.
2. Build a small local replay extractor against the already-downloaded 1.32.7 examples and manually inspect the resulting `(episode, seat, day)` rows.
3. Attach/download the selected top-daily partitions on Kaggle and scale preprocessing there once the local schema is trusted.
4. Audit the elite corpus for repeated behavioral lineages and shop/market-conditioned action variation; do not filter on reactivity until measurements justify it.
5. Train the first BC manager while the deterministic executor is designed in parallel.
6. Later evaluate whether to vendor/port the `diffmap/kaggicultureRL` Rust engine to 1.32.7 and require parity before PPO training.

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

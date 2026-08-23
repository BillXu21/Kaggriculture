# Kaggriculture Current State

Last updated: 2026-08-22

## Snapshot

- Repository: `BillXu21/Kaggriculture`
- Phase: **first full BC manager run completed successfully; held-out Aug-21 evaluation materially beats the train-only day baseline; next gate is a minimal deterministic executor and real closed-loop games**
- Latest confirmed upstream package: `kaggle-environments 1.32.7`
- Canonical replay schema: **v3**
- Training direction: **BC -> closed-loop executor validation -> PPO/RL refinement**. Scratch PPO remains a useful later comparison because the manager horizon is only ~30 steps.
- Primary project goal: demonstrate a refinement/self-play pipeline that measurably improves a competent starting policy across multiple promotions.

## Learned-Control Contract

The learned policy is a **once-per-day farm manager**, not a primitive 720-turn controller.

V0 manager owns economic intent:

- crop composition / planting allocation;
- target animal counts;
- land expansion;
- fertilizer allocation by crop type;
- CARE allocation by animal type;
- six-bin daily selling intent.

The deterministic executor owns mechanics:

- exact worker assignment/routing/movement;
- enough hiring to realize requested work;
- watering and routine maintenance;
- mechanical harvesting;
- seed purchases implied by crop targets;
- animal structures/purchases/placement implied by animal targets;
- initial wheat/feed procurement;
- exact tile/animal choice for fertilizer and CARE;
- primitive execution of sell intent.

The executor is a strategy compiler, not a substitute economic policy. See D-011, D-018, and D-019.

## Canonical Corpus

Five elite post-patch 1.32.7 daily partitions were regenerated at schema v3:

| date | episodes | seat-days |
| --- | ---: | ---: |
| 2026-08-17 | 699 | 41,940 |
| 2026-08-18 | 697 | 41,820 |
| 2026-08-19 | 695 | 41,700 |
| 2026-08-20 | 698 | 41,880 |
| 2026-08-21 | 697 | 41,820 |
| **total** | **3,486** | **209,160** |

Key contracts:

- one row per `(episode, seat, day)`;
- simulator positions `[x,y]`, canonical board/event coordinates `[y,x]`, tile lookup `tiles[y][x]`;
- full own board/private resources plus opponent public state retained canonically;
- crop/animal/land/fertilizer/CARE/sell labels retained;
- nested Zstandard Parquet production format;
- schema versions are fail-loud semantic boundaries.

Full-corpus residual unknown attribution is small: CARE unknown 13,045 (~0.585%); fertilizer unknown 2,666. These are preserved honestly and are not a V0 blocker.

Dataset is mounted for training at:

`/kaggle/input/datasets/billll/kaggriculture-canonical-daily-1327`

See `research/FIVE_DAY_V3_CORPUS.md`.

## First BC Manager (D-019)

Architecture:

- stateless one-day-in / one-plan-out Transformer;
- 100 own-board tile tokens;
- MANAGER + SELF RESOURCE + MARKET + TOWN + LABOR + DAY context;
- own private shed/seeds/inventory included;
- opponent-public board optional and disabled in V0;
- structured crop / animal / land / fertilizer / CARE / sell heads;
- seven fixed-weight loss groups;
- default 1,071,040 trainable parameters.

The schema-v3 code path has 102 passing tests including forward/backward, tiny overfit, checkpoint reload, and CLI smoke.

## First Full BC Run

Reference run provenance:

- code at run start: `692bca50e8ba0b687e48fd970e67bbe17014f03f`;
- train dates: Aug17-20;
- held-out validation: Aug21;
- `min_score >= 2950`;
- train rows: 25,500;
- validation rows: 5,700;
- CUDA + AMP;
- batch 256, AdamW lr 3e-4, weight decay 1e-2;
- 30 epochs, dropout 0.1, no scheduler or tuning sweep;
- best epoch: 29;
- best validation total: **2.8889**;
- full 30 epochs: ~237 s (~3.65k examples/s).

The validation curve continued improving through late training rather than immediately overfitting.

### State-aware model vs train-only day baseline on Aug21

| group | model | day baseline |
| --- | ---: | ---: |
| crop exact accuracy | **0.7128** | 0.4752 |
| crop whole-vector exact | **0.2930** | 0.0598 |
| crop MAE | **1.2731** | 3.6217 |
| animal exact accuracy | **0.8267** | 0.4540 |
| animal whole-vector exact | **0.6619** | 0.1389 |
| animal MAE | **0.2681** | 1.6936 |
| fertilizer nonzero recall | **0.7522** | 0.4557 |
| CARE whole-vector exact | **0.5998** | 0.1754 |
| CARE MAE | **0.3296** | 1.6676 |
| land accuracy | **0.9912** | 0.9089 |
| sell presence accuracy | **0.9394** | 0.8923 |

Rare/shifted branches also demonstrate state conditioning rather than pure calendar imitation:

- tomato nonzero recall: **83.8% model vs 0% day baseline**;
- goose nonzero recall: **96.8% vs 0%**;
- goose CARE nonzero recall: **95.5% vs 0%**;
- wheat fertilizer nonzero recall: **60.4% vs 0%**;
- strawberry fertilizer nonzero recall: **85.6% vs 66.6%**.

Selling remains the clearest teacher-forced weakness: true positive-cell rate 11.21%, predicted 9.38%, positive recall 64.84%, positive-cell quantity log-MAE 0.3529. Presence accuracy is therefore not sufficient by itself.

Conclusion: **the D-019 representation experiment passes its intended diagnostic.** The state-aware model learns and materially beats a day-only calendar baseline. Do not immediately scale the network, add opponent features, rebalance losses, or sweep hyperparameters before closed-loop evaluation.

Detailed record: `research/FIRST_BC_V0_EVAL.md`.

## Immediate Gate: Closed-Loop V0

Teacher-forced BC accuracy is no longer the dominant uncertainty. The next meaningful test is a complete actual-game loop:

`live obs -> BC manager -> daily intent -> deterministic executor -> primitive 1.32.7 actions`

The first executor should be deliberately simple and measurable rather than globally optimal.

### Initial executor direction

1. **Live observation adapter** matching the BC feature semantics exactly; call policy at hour 0 and cache the daily plan.
2. **Stable layout preference** rather than daily relayout: service-heavy livestock near the shed, crops in the next compact slots, existing useful structures/plants sticky.
3. **Mechanical task generator** translating manager targets into build/plant/dig/water/feed/CARE/fertilize/harvest/collect/sell work.
4. **Simple worker foreman**: do useful work underfoot first, then assign workers to tasks by small Manhattan-cost matching, routing through the shed when inventory pickup is required; recompute each primitive turn.
5. **Closed-loop harness** against one frozen competent opponent on fixed seeds in both seats before any population/self-play complexity.

Do not begin with full multi-worker VRP, elaborate facility-location optimization, or strategic heuristics hidden inside the executor.

### Executor-compliance metrics

Log manager request separately from achieved mechanics:

- next-day crop target MAE;
- animal target MAE;
- land target hit rate;
- fertilizer requested/completed;
- CARE requested/completed;
- sell intent/submitted;
- missed watering/feed/maintenance;
- illegal/ineffective actions;
- idle worker turns;
- unfinished tasks at day boundary;
- final banks and paired W/L/T.

If compliance is poor, improve execution before blaming BC. If compliance is high and economic trajectories are poor, revisit the manager/action abstraction before adding RL complexity.

## Near-Term Sequence

1. Plan and implement the smallest complete deterministic executor/foreman.
2. Get `best.pt` through complete local 1.32.7 games without illegal-action cascades or deadlock.
3. Run a small paired fixed-seed, seat-swapped panel against one frozen competent opponent and inspect executor compliance.
4. Only then decide whether the next marginal improvement should be executor quality, BC retraining, opponent-public features, or PPO.
5. Later compare BC-initialized PPO against scratch PPO under the same executor/opponents/budget.
6. Expand to frozen opponent panels/cross-play before changing-population self-play.

## Known Risks

- closed-loop distribution shift may expose outputs that are individually plausible but compound poorly;
- a weak executor can make a good manager look bad;
- the executor can accidentally become the strategist if it overrides requested economic intent;
- sparse sell/fertilizer decisions remain less reliable than core crop/animal/land behavior;
- the daily abstraction may omit strategically important intra-day decisions; selling is already the first explicit higher-frequency exception;
- non-transitive matchups can make simple Elo/latest-checkpoint promotion misleading.

## Do Not Forget

Before substantial work, read `CURRENT_STATE.md`, `DECISIONS.md`, `MECHANICS.md`, the relevant implemented notes, `research/FIRST_BC_V0_EVAL.md`, and the latest history/research records.

Before expensive runs, record exact code/configuration, engine identity, data/version/filter, seeds/opponents, outputs, stop conditions, and recovery plan.

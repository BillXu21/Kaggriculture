# Kaggriculture Current State

Last updated: 2026-08-16

## Snapshot

- Repository: `BillXu21/Kaggriculture`
- Phase: **action/observation interface design before BC implementation**
- Competitive implementation/training: not started
- Latest confirmed upstream package: `kaggle-environments 1.32.7`
- Latest balance PR: `Kaggle/kaggle-environments#1399`, merged
- Host intent: 1.32.7 should be the last balance change except game-breaking bugs
- Exact local/vendored engine lock: not yet established
- Training direction: **behavior cloning first, then PPO refinement; self-play only after fixed-opponent/fixed-pool improvement is demonstrated**

## Current Engine/Economics

1.32.6:

- town center consumes once/day at flat 1x;
- town shops are sampled with replacement;
- duplicate shops consume independently;
- total shop instances capped at 8.

1.32.7:

- carrot, tomato, and egg use scarcity-side `hinge` curves;
- these products can become extremely valuable only in favorable randomized shop/scarcity regimes;
- policy must reason conditionally about shop counts, market inventory, production lead time, opponent supply, and turns remaining rather than use a static product ranking.

See `MECHANICS.md` for exact parameters and regression points.

## Training Strategy

The current plan is no longer scratch self-play.

1. Archive/run strong public agents under the current engine.
2. Collect raw observation -> exact engine action trajectories.
3. Behavior-clone a competent policy.
4. Verify strong **closed-loop** BC rollouts, not only teacher-forced accuracy.
5. PPO-refine against one frozen strong opponent over randomized seeds.
6. PPO-refine against a frozen versioned opponent panel.
7. Only then introduce slowly changing opponents/population/self-play.

The project must prove learning at each simpler stationary stage before adding moving-target RL complexity.

## V0 Action Interface

New leading design: **direct factorized primitive actions**, not a custom task/semi-MDP language for V0.

Reason: public demonstrations already provide exact primitive labels, so BC can learn navigation/logistics without random exploration. A task language would require reconstructing latent intent and adds wrapper complexity before we know it is needed.

V0:

- one policy decision per engine turn;
- workers decoded in stable order (farmer, then current hands);
- shared worker policy with autoregressive conditioning for joint constraints;
- factor worker action into opcode + conditional item/crop + quantity;
- ordered autoregressive market queue up to 10 actions;
- mechanical action masks only;
- quantity proposal: categorical 1..512 with dynamic feasibility masks, to be checked against public trace distributions.

Hierarchical/task actions are now a **V1 fallback/experiment** if primitive BC proves brittle after trajectory divergence.

Detailed contract: `research/ACTION_OBSERVATION_V0.md`.

## V0 Observation Interface

Use full actor-visible state plus mechanically derived features.

Core groups:

- time/season/town tick timing;
- both 10x10 farms as spatial tensors;
- own worker tokens including private carried inventory;
- own shed/seeds/cash/land/labor state;
- opponent public farm/cash/land/labor state only;
- per-product market tokens with inventory, price, curve constants, deltas, and normalized scarcity `(I0-inventory)/T`;
- town shop **count vector** and implied per-product demand;
- short history features (1/4/24-turn market and money deltas) before committing to recurrence.

Derived crop/animal lifecycle features such as time-to-yield, decay, escape risk, and fertilizer duration are allowed because they encode known mechanics rather than strategy.

## Immediate Priorities

1. Freeze exact 1.32.7 source/spec hashes and verify all action opcodes (notably README `DROP` vs compact JSON description).
2. Download/archive current strong public agents and inspect their actual action/quantity distributions.
3. Finalize V0 action masks and quantity vocabulary from data.
4. Finalize/version the observation feature schema.
5. Define BC trajectory storage so raw observations/actions are always preserved.
6. Define BC acceptance tests: tiny-set overfit, held-out imitation, closed-loop rollout, first-divergence/recovery analysis.
7. Only after BC works, design the first fixed-opponent PPO refinement experiment.

## Known Risks

- Primitive BC may imitate expert routes but fail to recover after small action errors because movement lacks an explicit persistent goal.
- Too much feature engineering could leak strategy into preprocessing; derived features should remain mechanical/state-descriptive.
- Action masks or post-hoc repairs can accidentally change the sampled policy action and invalidate PPO log-probabilities.
- BC can memorize turn-indexed scripts rather than condition on state.
- Hidden opponent inventory may eventually require recurrence or better history inference.
- Context loss across chats can waste major amounts of work/compute; continuity docs remain mandatory.

## Do Not Forget

- Read `CURRENT_STATE.md`, `DECISIONS.md`, `MECHANICS.md`, `research/ACTION_OBSERVATION_V0.md`, `research/RL_DESIGN.md`, and the latest `HISTORY.md` entry before substantial work.
- Preserve raw data/configs/hashes so representations can be changed without rerunning expensive collection.
- Before any expensive run, record exact command/config, code+engine hashes, seeds, opponent pool, expected outputs, and stop conditions.
- Update continuity docs before switching chats or starting the next major experiment.

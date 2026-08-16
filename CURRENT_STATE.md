# Kaggriculture Current State

Last updated: 2026-08-16

## Snapshot

- Repository: `BillXu21/Kaggriculture`
- Default branch: `main`
- Phase: engine stabilization + RL interface/reward planning
- Competitive agent implementation: not started
- Current best internal agent: none
- Current evaluation suite: not implemented
- Latest confirmed upstream package line: `kaggle-environments 1.32.7`
- Latest balance PR: `Kaggle/kaggle-environments#1399`, merged
- Live leaderboard rollout of 1.32.7: announced/rolling out; not independently server-verified in this repo
- Engine lock: not yet established locally or vendored
- Host statement: 1.32.7 is intended to be the last balance change except game-breaking bugs

## Major Recent Balance Changes

### 1.32.6 — town demand/randomization

Merged PR #1394:

- town center consumes once/day at flat 1× instead of twice/day with later 2×/4× ramps;
- town shops are sampled with replacement;
- duplicate shop instances consume independently;
- total shop instances are capped at 8.

This made the realized shop multiset a meaningful episode-specific economic regime and made player-generated sell pressure more persistent.

### 1.32.7 — situational carrot/tomato/egg scarcity

Merged PR #1399 (`Make underused resources situational`) changes the scarcity-side price curves for **carrot, tomato, and egg** to a new `hinge` shape.

The hinge uses normalized scarcity `u = x/T` and:

`hinge(u) = u + 8 * max(0, u - 1)^2`

where `x = I0 - market_inventory` on the scarcity side.

Interpretation:

- below the product-specific scarcity knee `T`, price behavior is relatively ordinary;
- once scarcity exceeds `T`, price rises quadratically and can become extremely large;
- high randomized shop demand with little/no production can therefore create temporary high-value opportunities.

Current parameters:

| Product | Base | T | Scarcity curve | Below target | Knee inventory `I0-T` |
|---|---:|---:|---|---:|---:|
| Carrot | 35 | 450 | hinge | 1.00 | 9,550 |
| Tomato | 60 | 200 | hinge | 0.40 | 9,800 |
| Egg | 50 | 332 | hinge | 0.40 | 9,668 |

Source examples from PR tests show the nonlinear effect after the knee:

- carrot: inventory 9550 → $70; 9400 → $113; 9100 → $385;
- tomato: 9800 → $84; 9700 → $144; 9500 → $552;
- egg: 9668 → $70; 9502 → $120; 9170 → $460.

Host-reported probability of reaching meaningful scarcity with **no production**, due to randomized shop demand:

- tomato: ~50% of games;
- carrot: ~26%;
- egg: ~22%.

Treat those percentages as host-reported balance statistics rather than engine constants.

## Strategic Interpretation

This strengthens the RL-centered direction.

A product can now be:

- mediocre in the unconditional average;
- extremely profitable in a subset of realized shop regimes;
- less attractive again if either player notices and starts producing it;
- especially valuable late enough that existing public fixed routes may not have time or flexibility to pivot correctly.

The policy therefore needs to reason about **opportunity conditional on state**, not a fixed global product ranking.

Important state for these decisions includes:

- shop-instance counts by type;
- current market inventory and price for every product;
- distance to each scarcity knee `T`;
- observed inventory/price velocity;
- own crop/animal production pipeline and time-to-yield;
- opponent visible production pipeline;
- turns remaining and pivot payback horizon.

The 1.32.7 change is particularly relevant to end-game crop rotation and goose construction decisions.

## Current Strategic Direction

The project remains intentionally **RL-centered**, but not raw primitive-action RL.

The learned policy should own meaningful decisions such as:

- production allocation and crop rotation;
- land/labor investment;
- crop/animal mix;
- task assignment;
- adaptation to randomized shop demand and scarcity opportunities;
- opponent-aware production pivots;
- market order selection and timing.

Deterministic code should primarily provide:

- mechanical legality/action masking;
- pathfinding and execution of selected worker intents;
- task persistence;
- state normalization/bookkeeping;
- minimal safety/recovery against meaningless invalid-action cascades.

The working design is hierarchical intent-level RL. Detailed design: [`research/RL_DESIGN.md`](research/RL_DESIGN.md).

## Reward / Training Direction

Current leading plan:

- terminal competitive objective aligned with W/L/T;
- investigate potential-based shaping rather than arbitrary maintenance rewards;
- auxiliary prediction heads for future market state, production, opponent behavior, and win probability;
- behavior-clone strong public traces for basic logistics competence;
- fine-tune with PPO against randomized seeds and competitive opponent pools;
- eventually use population/self-play if throughput and results justify it.

1.32.7 adds useful auxiliary targets such as predicting whether/when a product will cross its scarcity knee and estimating the value of a production pivot.

## Current Experiments

No training experiments are active.

Planning/measurement experiments to specify before training:

1. **Scarcity-regime frequency:** reproduce the host's tomato/carrot/egg opportunity frequencies over many seeds.
2. **Price-spike timing:** distribution of first knee crossing by shop composition and product.
3. **Pivot value:** latest day at which planting tomatoes/carrots or building/geese can still profitably exploit a spike.
4. **Opponent suppression:** how quickly an opponent producing the scarce resource removes the opportunity.
5. **Shop-regime variance:** outcome variance of frozen deterministic agents under 1.32.7.
6. **Action abstraction/reward/memory/throughput** studies from the prior RL plan.

## Immediate Priorities

1. Treat 1.32.7 as the current implementation target, while still checking for bug-fix releases.
2. Freeze/document exact 1.32.7 source, package version, commit, and hashes when implementation begins.
3. Add market-curve/knee features to the planned observation schema.
4. Enumerate exact worker and market action schemas.
5. Finalize the intent/task candidate generator without embedding strategy into it.
6. Choose market order/quantity representation.
7. Formalize reward/potential candidates and BC trajectory format.
8. Design PPO/self-play curriculum and evaluation gates.
9. Recheck upstream immediately before the first implementation/training packet.

## Known Risks

- Bug-fix engine releases after 1.32.7.
- Fixed public baselines being stale under the new situational-resource economics.
- RL overfitting product identity instead of learning conditional opportunity value.
- Candidate generation accidentally encoding a fixed product ranking.
- Reward shaping overvaluing mark-to-market spikes that cannot actually be harvested/sold in time.
- BC anchoring to pre-1.32.7 deterministic routes.
- Insufficient simulator throughput for large PPO runs.
- Hidden opponent inventory requiring memory/inference.
- Chat-context loss causing stale configurations or duplicated compute.

## Do Not Forget

- Read `CURRENT_STATE.md`, `PLANS.md`, `DECISIONS.md`, `MECHANICS.md`, `research/RL_DESIGN.md`, and the latest section of `HISTORY.md` before substantial work.
- Before any expensive run, record the exact command, engine identity, agent hash, seeds, opponent pool, expected outputs, and stop conditions.
- After a substantial session, update continuity documents before switching chats.
- Do not treat announced rollout as independently verified live-server identity.

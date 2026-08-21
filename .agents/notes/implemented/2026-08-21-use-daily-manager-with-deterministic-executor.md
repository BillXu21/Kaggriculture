# Agent Note: Use a Daily RL Manager With a Deterministic Executor

Status: implemented
Date: 2026-08-21

## Problem

Primitive Kaggriculture control creates a 720-turn episode in which the learner must jointly solve strategy, worker routing, maintenance, purchasing, and market execution. That is a poor first learning problem for the available compute and makes value learning and credit assignment unnecessarily difficult.

Strong public agents already demonstrate that low-level farm execution can be handled reliably with deterministic routes and heuristics. The first RL system should therefore spend model capacity on farm-management decisions rather than rediscovering movement and routine maintenance.

The project also wants to begin behavior-cloning work before the deterministic executor is fully implemented.

## Decision

Use a high-level **daily manager** as the initial learned policy. The manager acts approximately once per game day, giving an episode roughly 30 strategic decisions instead of 720 primitive decisions.

For V0, the learned manager should own the decisions that define economic intent:

- crop planting/allocation changes;
- target animal counts;
- land expansion;
- fertilizer allocation;
- a daily selling plan, including enough intra-day timing detail to react to scheduled town-shop consumption.

The exact output encoding is still to be designed. This note locks the division of responsibility, not a particular neural head or categorical vocabulary.

The deterministic executor is assumed to be competent for the purpose of designing and training the manager. It should execute requested intent when feasible and should not substitute its own economic strategy. Initially it may own:

- worker assignment, routing, and movement;
- worker hiring needed to execute the requested workload;
- watering and other routine maintenance;
- mechanical harvesting of ready production;
- seed purchases implied by the requested crop plan;
- animal purchases/placement and required structures implied by requested animal targets;
- simple wheat/feed procurement heuristics;
- exact primitive scheduling of the manager's requested sell plan within the selected timing window.

Worker count is therefore not an initial policy output. The manager may instead observe simple operational feedback from the previous day, such as workers hired, labor cost, and whether the requested work was completed. This lets RL learn the economic consequences of asking for too much work without first learning worker scheduling.

Harvesting is mechanical in V0. Strategic harvest delay can be revisited only if measurements show that it matters.

Wheat management is the leading candidate to move from heuristic control to RL after the basic daily manager learns, because wheat has both production/feed utility and market-dependent economic value.

Selling is the first subsystem allowed to retain finer-than-daily timing. The initial sell representation should use a small number of shop-aligned intra-day windows or an equivalent simple timing abstraction. More reactive selling, a separate small selling policy, or more frequent calls to the main policy are later extensions.

## Observation implications

The daily manager can use a relatively compact state while preserving the spatial farm:

- day / season progress and relevant timing;
- the full own 10x10 farm board;
- own money, shed, seeds, land, animals, and fertilizer state;
- shared market inventory/prices and town shop state;
- simple previous-day execution/labor feedback;
- the opponent's public 10x10 farm and other public state may be supported because inference occurs only about 30 times per episode.

For the cleanest first PPO experiment, opponent features may still be masked so the learner first proves it can improve own-farm management. Supporting opponent features in the representation does not require using them immediately.

## Behavior-cloning plan

Behavior cloning should target **daily realized management decisions**, not the 720 primitive worker-action sequence.

For each replay day, derive labels from the day's state transition and market actions, including as applicable:

- crop additions/removals or resulting crop-allocation changes;
- animal-count changes;
- land expansion;
- fertilizer use/allocation;
- quantities sold and their intra-day timing windows.

Exact movement, worker assignment, watering, harvesting routes, seed-buy commands, animal-buy commands, and worker hires do not need to be imitation targets for the daily manager.

For the initial pipeline, treat the strong public executor as effectively perfect and imitate realized daily outcomes. Do not block BC preprocessing on completion of our own executor implementation.

## Replay version requirement

Training data must use the post-balance **1.32.7** environment. The official `Make underused resources situational (#1399)` commit is `28b6d8af3ce73926b3d0fda1410c1ddd8384ab8c`; it was created at `2026-08-15T01:24:24Z` and changed the package version from 1.32.6 to 1.32.7 while introducing the carrot/tomato/egg hinge curves.

When replay metadata exposes the environment/version, filter on version rather than date. If the daily replay dataset does not make version metadata usable, use **2026-08-16 onward** as the conservative default partition cutoff so the transition day is excluded, then validate sampled episodes against 1.32.7 mechanics before training.

A large top-daily-episodes replay dataset is expected to provide ample post-patch demonstrations; dataset schema, exact filtering, and daily-label extraction are the next implementation/research packet.

## Alternatives considered

### Primitive 720-turn policy

Rejected as the initial approach. It forces RL/BC to spend capacity on movement and maintenance and greatly lengthens the credit-assignment horizon.

### Let RL directly choose worker count and every purchase

Deferred. Labor cost is economically meaningful, but worker scheduling and implied fixed-price purchases can initially be handled by the executor. They can be promoted to learned control later if the heuristic becomes a measurable strategic bottleneck.

### Keep animals entirely heuristic

Rejected. The number and timing of animals is a genuine capital-allocation/production decision, so target animal counts belong to the manager even if the executor handles the fixed-price purchase and placement mechanics.

### Run the entire manager 24 times per day from the beginning

Deferred. Daily control is the simpler learning experiment. Selling receives limited intra-day timing first because its economics can change around town-consumption ticks.

## Consequences

- Initial RL horizon drops from 720 primitive decisions to roughly 30 high-level decisions.
- BC data becomes much denser strategically: one replay yields about 30 management examples rather than hundreds of repetitive movement labels.
- Replay preprocessing can begin before the deterministic executor is finished.
- The executor must remain a compiler of learned intent rather than quietly becoming the farm strategist.
- The first implementation packet is now replay inspection/filtering and extraction of daily manager training examples from confirmed post-1.32.7 episodes.
- Output-space details, exact sell windows, observation feature widths, and neural architecture remain open design questions.

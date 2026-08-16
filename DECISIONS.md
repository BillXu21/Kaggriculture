# Kaggriculture Durable Decisions

This file records decisions that remain authoritative across chats and work sessions. A decision should include the reason, evidence, and conditions for revisiting it.

## D-001 — Use Planning-First Development

- Date: 2026-08-06
- Status: active
- Decision: Keep the project focused on research, mechanics tracking, public-baseline analysis, evaluation design, and RL interface/reward planning while the engine and rules remain unsettled.
- Rationale: Large implementation work built against a moving engine risks immediate invalidation and wasted compute.
- Revisit when: the engine has remained stable long enough to freeze a version and the first regression suite is defined.

## D-002 — Do Not Start With Raw Primitive-Action RL

- Date: 2026-08-06; clarified 2026-08-07
- Status: active
- Decision: Do not begin by asking PPO to learn unrestricted raw movement and all primitive mechanics from scratch.
- Rationale: The primitive action space is combinatorial, crop/animal rewards are delayed, and small logistical mistakes cascade. Learning shortest paths and basic legality is unnecessary credit-assignment burden.
- Preferred alternative: hierarchical intent/task RL in which the learned policy owns meaningful production, allocation, adaptation, and market decisions while deterministic infrastructure compiles intent into legal primitive execution.
- Revisit when: action-abstraction experiments show that primitive control provides important strategic capability that the intent interface cannot express.

## D-003 — Treat the Shared Market as the Main Interaction Channel

- Date: 2026-08-06
- Status: active
- Decision: Center adversarial analysis on market inventory, prices, order timing, town demand, and opponent production forecasts.
- Rationale: Farms are physically separate and currently have little or no direct tactical interaction. The 1.32.6 reduction in town demand makes player-driven market pressure more important; 1.32.7 adds conditional scarcity spikes that increase the value of opponent-aware production response.
- Revisit when: the engine or rules add meaningful direct interaction.

## D-004 — Use Seat-Swapped Fixed-Seed Evaluation

- Date: 2026-08-06
- Status: active
- Decision: Every serious head-to-head evaluation must use a fixed seed list and both seat assignments.
- Rationale: Seat effects and deterministic seed effects can otherwise masquerade as policy strength.
- Minimum output: paired results, seat-specific win rate, final-bank margin, runtime, engine identity, and immutable agent identities.
- Revisit when: never remove paired evaluation; only extend it.

## D-005 — Evaluate Against a Frozen Competitive Pool

- Date: 2026-08-06
- Status: active
- Decision: Built-in pass or random agents are only plumbing checks. Competitive decisions must use a versioned, frozen pool of strong public and internal agents.
- Rationale: Weak baselines cannot distinguish serious improvements.
- Revisit when: the pool should be expanded or rotated, but historical pools must remain reproducible.

## D-006 — Preserve Third-Party Provenance

- Date: 2026-08-06
- Status: active
- Decision: Public Kaggle notebooks and agents may be downloaded and analyzed, but their original identity, hash, source, date, and modification history must be preserved.
- Rationale: Provenance is required for reproducibility, debugging, and separating copied behavior from original work.
- Revisit when: never remove provenance requirements.

## D-007 — Keep Current State Short and History Long

- Date: 2026-08-06
- Status: active
- Decision: `CURRENT_STATE.md` contains only currently relevant facts and active work; `HISTORY.md` preserves the full chronological record.
- Rationale: New chats and agents need a compact authoritative entry point without losing detailed historical evidence.
- Revisit when: document size makes a split necessary, while preserving the same roles.

## D-008 — Record Expensive Runs Before Launch

- Date: 2026-08-06
- Status: active
- Decision: Before any expensive run, record the exact command/configuration, code and engine hashes, seeds, opponent pool, expected outputs, stop conditions, and recovery plan in durable project documentation.
- Rationale: Previous project work lost compute because active configurations were not reliably carried across chat boundaries.
- Revisit when: never remove; automation may enforce it later.

## D-009 — Prefer Source and Behavioral Tests Over Discussion Claims

- Date: 2026-08-06
- Status: active
- Decision: Mechanics are authoritative only when supported by current engine source or a controlled behavioral test.
- Rationale: Documentation and competition discussions may lag behind live implementation.
- Confidence labels: `CONFIRMED_SOURCE`, `CONFIRMED_EXPERIMENT`, `DISCUSSION_CLAIM`, `HOST_REPORTED_STAT`, `OUTDATED`, `UNKNOWN`.
- Revisit when: confidence labels may expand, but source priority remains.

## D-010 — No Codex Work Yet

- Date: 2026-08-06
- Status: active
- Decision: Do not spend Codex on implementation at the current stage.
- Rationale: The user did not have spare Codex budget and implementation was intentionally deferred.
- Revisit when: the user explicitly authorizes a bounded Codex packet.

## D-011 — RL Owns Strategy; Deterministic Code Owns Mechanics

- Date: 2026-08-07
- Status: active
- Decision: The intended competitive architecture is RL-centered. Deterministic infrastructure may enforce mechanical feasibility, pathfinding, task persistence, and bookkeeping, but it should not quietly encode the winning farm strategy.
- Rationale: The project goal is to learn adaptive behavior, especially under opponent-dependent markets and random shop demand, while avoiding wasted model capacity on deterministic navigation details.
- Practical test: candidate generation may remove impossible actions; it must not remove merely unprofitable or strategically unusual legal actions just because a heuristic dislikes them.
- Revisit when: measured results show that a different division of control produces stronger generalization without reducing learning to a cosmetic role.

## D-012 — Use Public Deterministic Agents as RL Demonstrations

- Date: 2026-08-07
- Status: active
- Decision: Strong public action-list agents should be treated as behavior-cloning/bootstrap data in addition to evaluation opponents.
- Rationale: They contain valuable precision-sensitive logistics behavior and can initialize a viable policy before RL fine-tuning.
- Guardrail: BC is initialization only; training must include varied seeds, shops, opponents, and perturbations so the model can depart from time-indexed public scripts. Prefer fresh 1.32.7 demonstrations and avoid treating pre-1.32.7 product choices as timeless optimal labels.
- Revisit when: experiments show imitation causes more harmful anchoring than training benefit.

## D-013 — Dense Reward Must Preserve the Competitive Objective

- Date: 2026-08-07
- Status: active design constraint
- Decision: Do not add arbitrary maintenance/event rewards merely to make PPO learn faster. Prefer potential-based shaping and auxiliary prediction losses whose relationship to the final objective is explicit.
- Rationale: Bonuses for watering, harvesting, feeding, or production can be reward-hacked and may optimize farm activity instead of winning.
- Current terminal target: investigate W/L/T `+1/0/-1`, with bank margin retained as a metric and possible controlled curriculum signal.
- Revisit when: a different reward is demonstrated to improve competitive win rate without introducing proxy-objective pathologies.

## D-014 — Encode Town Shops as a Multiset

- Date: 2026-08-07
- Status: active
- Decision: Any observation schema, analysis, or policy input must preserve duplicate town shop instances, normally as per-shop counts or explicit shop entities rather than binary unlocked flags.
- Rationale: Engine 1.32.6 samples shops with replacement and each duplicate instance consumes independently.
- Revisit when: only if the upstream engine removes replacement sampling.

## D-015 — Do Not Encode a Static Product Ranking

- Date: 2026-08-16
- Status: active
- Decision: The action generator, observation preprocessing, BC labels, and policy logic must not hard-code products such as carrot/tomato/egg as globally good or globally bad.
- Rationale: Engine 1.32.7 deliberately makes these resources conditionally valuable. Their value depends on realized shop demand, market scarcity, production lead time, opponent response, and turns remaining.
- Required policy information: expose market curve parameters/state, shop multiplicity, scarcity distance, own/opponent production pipelines, and time remaining so the learned policy can estimate opportunity value.
- Revisit when: only if the market contract changes so product value is no longer strongly state-dependent.

## D-016 — Do Not Use Naive Spot Mark-to-Market Value for Reward Shaping

- Date: 2026-08-16
- Status: active design constraint
- Decision: Do not value large inventories or future production as `quantity × current spot price` inside a dense reward/potential without accounting for price impact and realizability before season end.
- Rationale: 1.32.7 hinge curves can create very high spot prices under deep scarcity, but selling production moves inventory back toward/through the knee. A naive potential could reward fake paper wealth or self-induced price manipulation instead of realizable competitive value.
- Preferred alternatives: exact/approximate liquidation simulation, marginal-price-aware valuation, time-to-sale constraints, or a learned continuation-value estimate validated against terminal outcomes.
- Revisit when: a tested valuation is shown to preserve trajectory ranking and resist exploitation under nonlinear market impact.

# Kaggriculture Durable Decisions

This file records decisions that remain authoritative across chats and work sessions. A decision should include the reason, evidence, and conditions for revisiting it.

## D-001 — Use Planning-First Development

- Date: 2026-08-06
- Status: active
- Decision: Keep the project focused on research, mechanics tracking, public-baseline analysis, and evaluation design while the engine and rules remain unsettled.
- Rationale: Large implementation work built against a moving engine risks immediate invalidation and wasted compute.
- Revisit when: the engine has remained stable long enough to freeze a version and the first regression suite is defined.

## D-002 — Do Not Start With Primitive-Action RL

- Date: 2026-08-06
- Status: active
- Decision: Do not begin with end-to-end primitive-action reinforcement learning.
- Rationale: Most farm logistics are deterministic, current strong public agents are deterministic schedules, and the meaningful uncertainty and interaction occur at a higher level.
- Preferred alternative: deterministic route execution, closed-loop repair, opponent-aware macro planning, and optimization over coherent experts.
- Revisit when: structured approaches reach a measured ceiling against a strong frozen opponent pool.

## D-003 — Treat the Shared Market as the Main Interaction Channel

- Date: 2026-08-06
- Status: active
- Decision: Center adversarial analysis on market inventory, prices, order timing, town demand, and opponent production forecasts.
- Rationale: Farms are physically separate and currently have little or no direct tactical interaction.
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
- Confidence labels: `CONFIRMED_SOURCE`, `CONFIRMED_EXPERIMENT`, `DISCUSSION_CLAIM`, `OUTDATED`, `UNKNOWN`.
- Revisit when: confidence labels may expand, but source priority remains.

## D-010 — No Codex Work Yet

- Date: 2026-08-06
- Status: active
- Decision: Do not spend Codex on implementation at the current stage.
- Rationale: The user does not currently have spare Codex budget and implementation is intentionally deferred.
- Revisit when: the user explicitly authorizes a bounded Codex packet.

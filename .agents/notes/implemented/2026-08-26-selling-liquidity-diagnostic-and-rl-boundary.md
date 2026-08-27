# Selling Liquidity Diagnostic and RL Ownership Boundary

Date: 2026-08-26
Status: active decision note pending canonical `DECISIONS.md` sync after the local V0.7 Stage-1 commit is pushed.

## Evidence

Issue #12 Stage 1 added a config-gated literal `aggressive_sell_all` mode on local commit `bea7b2722667c6bbd73c86d8fd918d4c5561cc44`. Default V0.7 remains unchanged. The experiment uses the real promoted BC-E checkpoint, `standard_mixed` d0-d3 opening, PASS opponent, exact 1.32.7 fast backend path, and prior-debt suppression ON.

Frozen/default V0.7, 12 seeds x both seats (24 games):

- mean bank: 28,587.4
- median: 25,321.5
- minimum: 2
- maximum: 65,959
- games <1k: 3
- games <10k: 3

Literal aggressive sell-all on the same panel:

- mean bank: 60,230.3
- median: 58,334
- minimum: 51,692
- maximum: 69,070
- games <1k: 0
- games <10k: 0
- improved games: 24/24

Largest rescues include seed 7 seat 1 (`2 -> 51,929`) and seed 1019 (`76 -> 69,070` seat 0; `55 -> 58,268` seat 1).

Two identical Kaggle submissions of BC-E + V0.7 + literal aggressive sell-all are being used for external leaderboard verification. PASS-bank values are not treated as leaderboard predictions.

## Decision

The experiment is strong evidence that selling/liquidity is a dominant BC-E closed-loop manager-policy bottleneck. The farm policy can produce substantial value, but the learned sell behavior is too sparse/inconsistent to reliably convert inventory into operating cash.

Do **not** promote literal midgame sell-all into the permanent deterministic executor. Midgame sell timing and quantity remain learned manager strategy under the architecture boundary `learned manager -> deterministic executor`.

Keep aggressive sell-all as:

1. a diagnostic policy;
2. a local economic headroom/reference baseline;
3. an external leaderboard reference while the learned selling policy is repaired.

A final-action terminal liquidation rule remains conceptually mechanical because no future inventory-management decision remains after the final actionable transition.

## RL implications

Issue #9 PPO V0 currently freezes BC sell quantities and trains only sell presence. That simplification is now a likely capability bottleneck. Before serious selling-focused RL, add a proper trainable sell-quantity distribution with exact log-probability semantics rather than pretending the existing deterministic regression scalar is stochastic.

The older Issue #9 W/L/T-only first reward is also stale relative to the current economic-learning direction. The first bounded economic RL phase should use a terminal final-cash-dominated objective with fixed normalization and an explicit bankruptcy penalty, without hand-written dense watering/harvest/production rewards. Competitive outcome weighting can be increased after basic economic robustness is learned.

A high-value first curriculum is to hold most of BC-E fixed and let RL repair selling first, so the experiment tests whether RL can recover the roughly 60k PASS-bank liquidity baseline without destabilizing the rest of the copied farm policy. Full-manager unfreezing and competitive snapshot/self-play follow after that capability is demonstrated.

## Independent executor work

Issue #12 Stage 2 spawn/shed waiting remains independent. The aggressive-selling result does not prove the observed ~8-turn worker waits are acceptable. Patch only if passive diagnostics identify a reproducible mechanical executor defect.

Optional spare-capacity watering remains Stage 3 and must not preempt higher-priority work or recreate the rejected R4 worker-diversion failure.

## Revisit when

- both leaderboard submissions have enough games to estimate their actual competitive rating;
- a trainable sell-quantity PPO implementation is validated;
- selling-focused RL matches or exceeds the aggressive-sell reference without the hard-coded override;
- Stage 2 proves a mechanical executor defect that materially changes the economics of these trajectories.

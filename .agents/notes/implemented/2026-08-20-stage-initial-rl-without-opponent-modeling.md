# Agent Note: Stage Initial RL Without Opponent Modeling

Status: implemented
Date: 2026-08-20

## Problem

The project needs a first reinforcement-learning milestone that is simple enough to diagnose whether learning is actually occurring. Kaggriculture exposes a rich two-player state, and it is possible to infer or model opponent production, hidden holdings, and market pressure, but introducing that machinery immediately would make failures harder to attribute.

The main project goal is to demonstrate that a competent behavior-cloned policy can be improved meaningfully by RL/self-play over multiple promotions. Scratch RL is not the intended starting point because available compute is limited and prior competition experience showed that large self-play systems can consume substantial time before basic learning has been established.

## Decision

The initial RL scope will focus on managing the acting player's own farm while observing the shared town and market.

For the first learning milestone:

- initialize from behavior cloning of a competent public agent or public replay distribution;
- expose the policy to its own farm/private state plus the shared market and town state needed for economic decisions;
- do not require explicit opponent modeling, opponent hidden-inventory inference, or deliberate market attacks;
- use fixed or otherwise controlled opponents so their effects on the shared market remain part of the environment without making opponent exploitation the learning objective;
- first demonstrate that PPO/RL can improve the BC policy's own farming/economic management before adding richer adversarial reasoning.

Opponent-board features may be withheld from the earliest experiment if doing so makes the learning problem cleaner. This is a staging decision, not a claim that opponent information is unimportant to the eventual competitive agent.

## Alternatives considered

### Model the opponent from the start

Rejected for the first milestone. Public farm state plus market transitions can support detailed inference of opponent production and even approximate hidden holdings, but this adds partial-observation bookkeeping, strategic interaction, and additional failure modes before basic RL improvement has been demonstrated.

### Build explicit hidden-inventory accounting immediately

Rejected for now. A deterministic opponent inventory accountant may later be valuable, but it is unnecessary for answering the first question: can RL improve a competent BC farmer at managing its own production and economics?

### Train scratch self-play on the full two-player problem

Rejected as the starting path. It would require substantially more exploration and compute, and it would repeat the failure mode of building a complicated self-play system before proving learning on a simpler stationary problem.

## Consequences

- Early observation/action research should first ask what a strong policy needs to manage its own farm and respond to the public market/town state.
- Initial PPO evaluation should emphasize improvement against frozen opponents and held-out seeds rather than opponent exploitation.
- The shared market still provides realistic external variation because the opponent continues to affect prices and inventory.
- Failure to improve at this stage cannot be blamed on sophisticated opponent modeling or game-theoretic nonstationarity, which makes diagnosis cleaner.
- Opponent public-board features, inferred hidden holdings, counter-strategy features, and deliberate market manipulation remain later-stage extensions after basic BC-to-RL improvement is established.

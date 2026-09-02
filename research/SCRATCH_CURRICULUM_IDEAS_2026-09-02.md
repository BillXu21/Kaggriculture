# Scratch Curriculum / Exploration Ideas — 2026-09-02

Status: **future experiment ideas, not frozen implementation decisions**

Branch at time of this record: `codex/issue-37-fourth-quadrant-openings`

This note records follow-up ideas prompted by the true no-opening scratch PPO
experiment. It is intentionally speculative. The purpose is to preserve useful
curriculum/exploration directions for Stage 2.5+ without mixing them into the
clean no-opening baseline contract.

## 1. Empirical motivation from the no-opening scratch run

The present once-per-day manager can learn from a viable scripted opening, but
it did not bootstrap a useful d0 economy from scratch under the unchanged
Issue20 representation.

The u1/u10/u20 deterministic no-opening self/self panel was especially stark:

```text
u1:  mean/median/p10/min/max = 1509 / 1509 / 1509 / 1509 / 1509
u10: mean/median/p10/min/max = 1509 / 1509 / 1509 / 1509 / 1509
u20: mean/median/p10/min/max = 1509 / 1509 / 1509 / 1509 / 1509
```

All 32 deterministic games / 64 seat results at each checkpoint produced the
same $1509 bank. Parameter checks confirmed the mutable PPO weights did move:

```text
u1 -> u10  L2 delta ~1.98, mean |delta| ~0.00128, max |delta| ~0.02896
u10 -> u20 L2 delta ~3.45, mean |delta| ~0.00197, max |delta| ~0.06035
u1 -> u20  L2 delta ~4.38, mean |delta| ~0.00266, max |delta| ~0.06962
```

The stochastic rollouts were less degenerate but still poor: typical runs
remained around the low-thousands while occasional very high maxima carried the
mean. That pattern suggests the policy distribution may contain rare viable
sampled behaviors that PPO is failing to convert into the greedy mode.

The main future question is therefore not only "more updates?". It is how to
make the scratch search problem easier and how to retain rare useful behavior
once discovered.

## 2. Curriculum idea: temporarily mask land expansion

This is the strongest first masking candidate because the failed no-opening
policy repeatedly bought extra land while barely operating the starting
quadrant.

Possible Phase-1 rule:

```text
land ownership target = 1 quadrant
NE/SW/SE purchase decisions unavailable
```

Important PPO detail: a masked head should not merely be decoded and then
silently overwritten. While land is locked, either remove/freeze the land head
from trainable logprob/entropy accounting or otherwise make the mask explicit
in the policy objective. Do not train PPO against actions that cannot reach the
environment.

Bootstrap skill to learn first:

```text
use the initial quadrant
-> create productive squares
-> finance labor / maintenance
-> produce goods
-> realize revenue
-> reinvest coherently
```

After the policy can reliably operate one quadrant, resume the same lineage
with land expansion re-enabled.

Potential unlock criteria should be empirical rather than arbitrary. Candidates
include a sustained bank threshold, productive occupancy, productive-square-day
count, or production-value threshold over a fixed evaluation panel.

## 3. Curriculum idea: initially mask opponent-board inputs

A scratch manager does not need opponent-farm information to learn the first
own-farm production loop.

Possible early observation curriculum:

```text
own farm / inventory / labor / cash: visible
shared market / shop state:          visible
opponent farm board/state:           masked
```

The shared market should remain visible because it is part of the agent's own
economic environment. Opponent-specific farm information can be introduced
later once basic farm management works.

This is intended as input simplification, not a claim that opponent information
is unimportant for the final competitive policy.

## 4. Economic/action masks should be staged carefully

Do not broadly mask all selling: completing the basic production -> cash loop
requires ordinary selling behavior.

However, future fine-grained wheat buy/sell manipulation is a good candidate to
leave disabled during the earliest curriculum phase. Sophisticated inventory
cycling, price manipulation, and intraday liquidity management can be added
only after the manager can maintain a normal farm.

Product-specific action masks may also be useful as diversity tools, but should
be case-by-case rather than universal. Example:

```text
start from goose-heavy / goose-nation checkpoint
-> temporarily mask goose actions
-> force continuation to discover a different viable basin
-> later re-enable goose
```

This is better viewed as basin-forcing / diversity generation than as a general
fundamentals curriculum.

## 5. Short-horizon curriculum: first 10 days

Many failed no-opening trajectories are economically dead very early, so
running all the way to day 30 wastes rollout compute and makes credit assignment
harder.

A staged horizon is worth testing:

```text
Phase 1: d0-d10
Phase 2: d0-d15 or d0-d20
Phase 3: full d0-d30
```

But **raw d10 cash should not be the Phase-1 objective**. Raw bank would punish
reinvestment and can make "do nothing / hoard starting cash" look artificially
good.

The curriculum reward should instead measure whether the policy built a
working economic engine by the short horizon.

## 6. Candidate d10 reward: production value + productive-square-days

Preferred first concept:

```text
R10 = normalized cumulative production value
    + lambda * normalized productive-square-days
```

### Cumulative production value

Measure the total value of goods actually produced during d0-d10 using **fixed
reference prices**, not contemporaneous market prices.

Reasoning:

- reinvestment is not punished;
- the signal rewards the production loop directly;
- fixed reference prices avoid immediately teaching market manipulation;
- a product remains equally valuable for curriculum purposes regardless of
  temporary price crashes/spikes.

The reference-price table would need to be chosen and documented before an
implementation experiment. This note does not freeze exact prices.

### Productive-square-days

Prefer a cumulative capacity/utilization statistic over a single d10 snapshot:

```text
productive_square_days = sum(productive squares active on each day d0..d9)
```

This rewards getting capacity online early and prevents a policy from doing
nothing for most of the episode and then creating cheap productive tiles at the
last moment solely for the terminal bonus.

Production value should probably dominate the combined reward, with
productive-square-days acting as a smaller supporting term. Exact scaling is
left for an experiment design pass.

## 7. Economic net worth is worth logging, but not necessarily rewarding first

Another useful d10 diagnostic is conservative economic net worth:

```text
cash
+ conservative inventory liquidation value
+ conservative value of productive crops / animals / infrastructure
```

This is naturally less hostile to reinvestment than raw bank.

However, using net worth as the primary early reward risks subjective asset
valuation and can accidentally reward capital parking (for example buying land
if land is counted too generously). It is safer as a logged diagnostic at
first, especially until land is locked or asset valuations are validated.

## 8. Early-stop dead trajectories: possible compute optimization

A state-dependent early stop may save substantial compute if a farm is already
irrecoverable by d8-d10.

Possible future direction:

```text
if cash is extremely low
and there is effectively no productive capacity / recoverable inventory
and historical data shows such states almost never recover
-> terminate early with a pessimistic terminal value
```

Do not implement a loose heuristic immediately. A termination rule changes the
MDP and can become exploitable. First derive candidate dead-state criteria from
existing full-length trajectories and verify that states matching the rule have
negligible probability of meaningful recovery.

A fixed 10-day curriculum stage is cleaner than an unvalidated dynamic stop
condition.

## 9. Stochastic champion harvesting

The rare high maxima in otherwise poor scratch rollouts may contain useful
sampled strategies hidden inside a bad greedy policy.

Future diagnostic:

1. freeze one checkpoint;
2. run 128-256 stochastic self-play games;
3. retain the highest-return seats / episodes;
4. save their exact sampled manager plans and rollout provenance;
5. rerun the same stochastic identity to verify reproducibility;
6. test the same sampling stream across fresh environment seeds;
7. separately hold environment seed fixed while varying stochastic policy
   realization.

Interpretation:

```text
same stochastic realization works across many env seeds
-> viable stochastic subpolicy / "personality" may exist

only the original env seed works
-> likely environment/opponent lottery

many stochastic realizations work on one fixed env
-> useful probability mass exists but greedy mode is poor
```

The existing PPO adapter's row-aware deterministic seed hashing makes this type
of forensic sampling experiment feasible.

## 10. Elite self-imitation as a retention mechanism

If rare stochastic trajectories are genuinely competent, entropy alone solves
only the discovery half of the problem. PPO may still fail to concentrate
probability mass onto the useful actions.

Potential future addition:

```text
collect stochastic self-play
-> retain top-return trajectories / top percentile
-> apply a small auxiliary behavior-cloning loss on those self-generated actions
-> continue PPO
```

This is a form of elite/self-imitation learning: "the current policy randomly
found something that worked; make those decisions more probable."

It should remain auxiliary to PPO rather than replacing the actual economic
objective. Guard against overfitting one lucky trajectory by requiring either
repeatability, multiple elite episodes, or cross-seed competence.

## 11. Tentative curriculum progression

One possible sequence to test later:

```text
Phase 1 — bootstrap farm
  horizon:             d0-d10
  land expansion:      locked
  opponent board:      masked
  crops:               enabled
  animals:             enabled
  ordinary selling:    enabled
  advanced wheat econ: absent/masked
  reward:              fixed-price production value
                       + small productive-square-day bonus

Phase 2 — basic economy / expansion
  horizon:             d0-d15 or d0-d20
  land expansion:      unlock
  opponent board:      still masked initially
  reward:              mix bootstrap reward with longer-horizon economic value

Phase 3 — full competitive economy
  horizon:             d0-d30
  land:                enabled
  opponent board:      enabled
  advanced economics:  enabled as representation supports them
  reward:              transition toward true final-bank objective
```

This is only a candidate curriculum. The first experiment should change as few
things as possible.

## 12. Preferred first bounded curriculum ablation

The cleanest next masking experiment is probably **land lock only**:

```text
same true no-opening scratch PPO
same reward / horizon / observation representation
same learning rate and entropy
land count locked to 1
land head explicitly masked/frozen in PPO accounting
10 updates initially
```

This directly tests the most obvious pathological scratch action without
confounding the result with a new reward and several observation masks.

If land-lock alone produces a functioning one-quadrant farm, continue the same
lineage with land re-enabled. If it does not, the short-horizon production
curriculum becomes the next stronger intervention.

## 13. Relationship to the future multi-trainer population

These masks can later become useful sources of strategic diversity rather than
only training aids. Independent trainers could be given different temporary
constraints / curricula and then all restrictions removed before population
competition.

Examples:

```text
trainer A: normal actions
trainer B: land locked longer
trainer C: goose disabled temporarily
trainer D: crop-biased curriculum
trainer E: animal-family-biased curriculum
trainer F: different scratch / BC initialization
```

The purpose would be to create different strategic basins before the league's
replacement mechanism begins selecting among them.

None of these population variants are frozen yet.

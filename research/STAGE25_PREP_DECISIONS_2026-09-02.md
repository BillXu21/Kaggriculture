# Stage 2.5 Prep Decisions — 2026-09-02

Status: **working handoff record before Stage 2.5**

Branch at time of this record: `codex/issue-37-fourth-quadrant-openings`

No-opening implementation anchor:
`a83df0fcfafc549935c8470d13c6ea0696be3a6c`

This file records the current experimental conclusions and the decisions that
should survive notebook/session turnover. It is intentionally narrower than a
full Stage 2.5 design.

## 1. Current project objective

The immediate goal is not medal optimization. The current goal is to prove a
self-play pipeline can repeatedly improve a competent policy and to learn which
parts of the present action/observation interface prevent larger strategic
jumps.

The current-current economic scratch PPO experiment established an important
baseline:

- a scratch daily manager can bootstrap a viable strategy from the existing
  scripted opening;
- continued self-play can improve that strategy materially;
- improvement was strongest in lower-tail robustness and capital efficiency;
- the run remained in the same broad goose-heavy strategic basin rather than
  discovering a qualitatively different economy.

The u10 -> u40 continuation therefore counts as evidence that the current PPO
stack can find and optimize a local minimum. The next Stage 2.5 question is
primarily **diversity**, not merely more updates on one lineage.

## 2. Stage 2.5 working direction: multiple trainers, different minima

The preferred first Stage 2.5 target is multiple independent trainers that can
settle into different strategic basins.

Working direction, not yet a frozen league specification:

- several independent trainers with independent initialization / optimizer /
  RNG / rollout lineage;
- preserve strategically distinct historical policies rather than only the
  newest checkpoint;
- conservative population replacement, with a simple top-1 -> bottom-1 style
  rule preferred over aggressive population churn for the first version;
- fitness should not be pure W/L. Absolute bank, opponent results, and lower
  tail robustness all matter.

Do not over-design the full league before the remaining pre-2.5 experiments are
finished.

## 3. Fourth Quadrant investigation: what changed our interpretation

The sampled Fourth Quadrant replays should **not** be treated as one fixed
96-turn opening book.

The Issue #37 audit found:

- seat-0 sampled replays diverge from each other starting at d0h1 market
  behavior;
- seat-1 sampled replays also diverge starting at d0h1;
- the common d0h0 action is stable within a seat, but seat 0 and seat 1 have
  materially different d0h0 actions;
- later worker routing also diverges.

Current interpretation: Fourth Quadrant likely has an adaptive policy/planner
active essentially from turn 0. The literal `fourth_quadrant_s0` and
`fourth_quadrant_s1` traces are useful sampled trajectories, not robust
standalone openings.

A plausible architecture is:

```text
stable seat-conditioned d0h0
-> observe shared market / opponent / own state
-> adaptive d0h1-d3 economic control or replanning
-> continuation
```

No claim is made here about whether that control is neural, search-based,
optimization-based, a template library, or a hybrid.

## 4. Fast/official parity lesson from the edgy FQ trace

The FQ sampled opening exposed a parity-sensitive edge case.

With the intended $3000 starting money, fast and official already differed by
$1 after the first submitted opening action. Under this extremely capital-tight
trajectory, the small accounting difference changed later affordability and
quickly caused much larger economic/state divergence, including different hand
counts and farm layouts.

Decision:

```text
normal PPO / league training and broad diagnostics -> fast backend
razor-thin FQ-style opening reconstruction        -> official backend
final promising candidates                         -> official validation
```

Do not interpret the old fast FQ-vs-standard score as a trustworthy estimate
of the real opening's competitive value. Keep fast as the training engine, but
reopen exact parity work later if precise edge-economy experiments depend on
it.

## 5. FQ -> BC handoff diagnostic

One clean fast smoke completed a full game with the FQ seat-0 sampled trace for
d0-d3 and the current legacy BC-E + aggressive selling continuation from d4.

At the clean d4h0 handoff the FQ-side farm had approximately:

```text
cash:        $1
crops:       WHEAT 2, STRAWBERRY 5, MELON 10
animals:     COW 5, SHEEP 2
shed wheat:  3
land:        1 quadrant
```

BC immediately spent the final dollar on one hire. Executor diagnostics then
showed requested labor far above affordable labor and feed shortage on every
turn of the next days. The farm rapidly lost animals and productive capacity:

```text
d4: 5 cows + 2 sheep
d5: 4 cows + 2 sheep
d6: 1 cow  + 2 sheep
d8: 0 animals
```

The hybrid recovered only into a mediocre crop economy and finished around
$17.6k while the standard-opening BC opponent finished around $77.8k in that
smoke.

Interpretation: the FQ sampled state is highly leveraged. Stealing its assets
without stealing its continuation logic is not enough. The strategy likely
relies on aggressive intra-day liquidity management: inventory liquidation,
worker financing, feed purchases, and possibly rebuying/market cycling.

This is evidence against treating the FQ trajectory as a plug-in opening for
the current daily BC manager.

## 6. Stage 2.5 action-space candidate: finer WHEAT economic control

The current daily manager is probably too coarse to learn the most aggressive
zero-cash / near-zero-cash continuations.

The first Stage 2.5 microeconomic action worth testing is **finer learned wheat
buying and selling control**.

Why wheat first:

- it is both a product and the critical animal feed resource;
- it can act as emergency liquidity;
- buy/sell timing changes whether an animal-heavy farm survives;
- it directly interacts with labor financing and market manipulation-style
  strategies;
- the current frozen/coarse selling representation prevents the manager from
  deliberately expressing many of those sequences.

Do not add this control to the current no-opening baseline. That baseline is
specifically intended to measure the present representation unchanged.

## 7. Per-turn or sub-daily control is deferred

An extremely edgy opening may require decisions after each market/worker state
change rather than one global manager action per day.

The current daily-manager architecture cannot naturally express sequences such
as:

```text
sell inventory
-> observe cash / price
-> hire
-> observe
-> buy feed
-> observe
-> sell or rebalance again
```

Decision: **do not move to per-turn learned control in Stage 2.5**. Treat
sub-daily / event-driven / per-turn learned economic control as later Stage 3
or Stage 4 territory after the current hierarchy and population experiments
are better understood.

## 8. Immediate experiment: scratch self-play with no opening

Before Stage 2.5, run one clean experiment asking whether the existing daily
manager/executor representation can discover any competent opening from the
untouched initial state.

The no-opening implementation at
`a83df0fcfafc549935c8470d13c6ea0696be3a6c` adds:

- explicit `opening="none"`;
- direct executor actions from d0h0;
- `--opening` and `--manager-start-day` training CLI flags;
- manager horizon derived from `TOTAL_DAYS - manager_start_day`;
- no-opening provenance with no fake trace digest;
- matching parallel shard sizing and focused tests.

Experiment contract:

```text
old scratch experiment:
  d0-d3 standard_mixed
  d4-d29 scratch PPO

new baseline:
  d0-d29 scratch PPO
```

Keep every other meaningful Issue20 scratch setting unchanged:

- scratch initialization;
- `current_vs_current_economic`;
- terminal own-bank reward;
- bank baseline 3000;
- bank scale 50000;
- stochastic PPO collection;
- frozen legacy BC-E sell quantity;
- E_LEGACY;
- master seed 32032;
- 384 games/update;
- 96 rollout workers;
- 1 env / worker;
- 1 engine thread;
- fast native batch backend;
- policy-scope central inference;
- fixed inference batch size 32;
- wait 2 ms;
- 2 PPO epochs;
- minibatch 256;
- learning rate 1e-4;
- target KL 0.08;
- entropy coefficient remains the existing default 0.01;
- no promotion gate.

Run **20 updates** for the first read rather than 10. Ten updates may only show
bootstrap chaos; twenty gives the policy time to stabilize into an opening
basin if one exists.

With manager start day 0:

```text
30 manager decisions / seat
60 trainable rows / game
384 games / update
23,040 PPO rows / update
90 minibatches / epoch at minibatch size 256
```

Useful checkpoints are approximately u5, u10, u15, and u20. Do not judge the
experiment only by early u1-u3 behavior unless it is structurally broken.

Primary question:

> Can the current once-per-day manager + deterministic executor bootstrap a
> nontrivial economy from the untouched $3000 initial state with no scripted
> opening?

Interpretation guide:

```text
A. viable coherent opening emerges
   -> Stage 2.5 can seriously consider reducing dependence on scripted openings

B. survives but settles into a bad opening basin
   -> representation is capable, but exploration/diversity is weak

C. repeatedly burns/hoards capital and never bootstraps
   -> strong evidence Stage 2.5 needs finer economic controls before expecting
      RL to invent high-quality openings
```

## 9. Exact planned training invocation

```bash
python -m rl_manager.cli train \
  --e-checkpoint /kaggle/working/bc-v1-E/last.pt \
  --executor-factory executor_v0@stage-a-v1 \
  --backend fast \
  --master-seed 32032 \
  --init-mode scratch \
  --training-composition current_vs_current_economic \
  --reward-mode terminal_own_bank \
  --bank-reward-baseline 3000 \
  --bank-reward-scale 50000 \
  --opening none \
  --manager-start-day 0 \
  --num-workers 96 \
  --num-envs 1 \
  --num-threads 1 \
  --low-telemetry \
  --read-only-agent-observations \
  --batch-backend \
  --inference-batch-scope policy \
  --fixed-inference-batch-size 32 \
  --inference-batch-wait-ms 2 \
  --episodes-per-update 384 \
  --updates 20 \
  --promotion-every 0 \
  --epochs 2 \
  --minibatch-size 256 \
  --lr 1e-4 \
  --target-kl 0.08 \
  --e-history-version E_LEGACY \
  --output-dir /kaggle/working/issue20-scratch-no-opening-u20-v1 \
  --checkpoint /kaggle/working/issue20-scratch-no-opening-u20-v1/final.npz
```

The Kaggle TPU notebook generated for this run pins the exact commit above and
uses the Kaggle `GITHUB_TOKEN` secret through `GIT_ASKPASS` rather than placing
the token in the clone URL or argv.

## 10. What is deliberately not being changed yet

For this pre-2.5 baseline, do not simultaneously change:

- reward shaping;
- entropy coefficient / temperature;
- PPO learning rate;
- crop count representation;
- wheat control;
- sell quantity training;
- executor strategy;
- per-turn manager frequency;
- population/league scheduling.

The point is to leave Stage 2.5 with a clean answer about what the **current**
representation can do from d0 without a scripted opening.

# Kaggriculture Mechanics Ledger

Last updated: 2026-08-30

## Issue #33 Fast/Official Parity Audit (2026-08-30)

- `CONFIRMED_EXPERIMENT`: after rebuilding the native extension from the audit
  worktree, reset underlying canonical state and policy-visible canonical
  observations matched official 1.32.7 for seeds `7,17,42,123`.
- `CONFIRMED_EXPERIMENT`: eight seat-swapped current BC-E + `standard_mixed`
  opening + repaired executor + PASS traces, and one current-control vs
  current-control trace, had zero first divergence across all 719 accepted
  post-reset transitions. Exact saved official joint actions were replayed
  through both backends; policy feedback was not used during same-action replay.
- `CONFIRMED_EXPERIMENT`: the existing legal-ish corpus rerun covered 8 seeds,
  5,752 action pairs, 33 action families, and 29 day boundaries per episode
  with zero first divergence. Covered families include lifecycle, animal,
  market, hiring, overflow, malformed/no-op, town, RNG/day refresh, and
  terminal paths already defined by the corpus.
- No new mechanic divergence was found, so no fast-engine semantic correction
  or regression was added. This remains bounded evidence for exercised states,
  not universal parity. Native SHA-256 and artifact details are in
  `research/FAST_OFFICIAL_PARITY_AUDIT_ISSUE33.md`.

## Issue #25 Executor Evidence (2026-08-29)

- `PLACE` is a completion action for an already-owned animal, not a new
  capital commitment. Suppressing BUILD/BUY/LAND work must therefore preserve
  dependency-free placement into an existing compatible empty COOP/PASTURE;
  dependent placement remains blocked with its BUILD prerequisite.
- The foreman aligns private inventories as farmer then hands and routes a
  carried animal directly to PLACE. Empty structures are claimed once per
  generated task, and the next observation removes the placement task after
  the board contains the animal. These claims are covered by deterministic
  executor tests; no official-engine package is installed locally.
- Optional spare WATER remains the existing PASS-only layer. Production
  submission/RL default factories now enable it, while direct `AgentConfig`
  defaults remain false. A candidate WATER movement is allowed only when its
  target is reachable within the remaining same-day turns; mandatory hard
  WATER, FEED, HARVEST, manager, and logistics work still runs first.
- Bounded fast evidence with frozen BC-E, seeds `7,17`, both seats, and PASS
  measured OFF versus WATER-only final WEED tiles `53 -> 39`, PASS actions
  `1,178 -> 601`, and movement `10,581 -> 11,975`; no fallback or animal
  escapes occurred. The candidate's animal target and work-debt outcomes were
  mixed, so this does not establish competitive promotion or official parity.

## Issue #25 Executor Evidence (2026-08-29)

- `PLACE` is a completion action for an already-owned animal, not a new
  capital commitment. Suppressing BUILD/BUY/LAND work must therefore preserve
  dependency-free placement into an existing compatible empty COOP/PASTURE;
  dependent placement remains blocked with its BUILD prerequisite.
- The foreman aligns private inventories as farmer then hands and routes a
  carried animal directly to PLACE. Empty structures are claimed once per
  generated task, and the next observation removes the placement task after
  the board contains the animal. These claims are covered by deterministic
  executor tests; no official-engine package is installed locally.
- Optional spare WATER remains the existing PASS-only layer. Production
  submission/RL default factories now enable it, while direct `AgentConfig`
  defaults remain false. A candidate WATER movement is allowed only when its
  target is reachable within the remaining same-day turns; mandatory hard
  WATER, FEED, HARVEST, manager, and logistics work still runs first.

## Purpose

This file separates verified engine behavior from documentation, discussion claims, and stale assumptions.

Confidence labels:

- `CONFIRMED_SOURCE`: observed in a specific official source snapshot or merged upstream diff.
- `CONFIRMED_EXPERIMENT`: reproduced with a controlled behavioral test.
- `DISCUSSION_CLAIM`: reported publicly but not independently verified.
- `HOST_REPORTED_STAT`: quantitative statistic reported by the host but not an engine constant.
- `OUTDATED`: known to describe an older engine or contract.
- `UNKNOWN`: unresolved or insufficiently specified.

## Executor Continuation - Issue #28 (2026-08-29)

- `CONFIRMED_EXPERIMENT`: task generation already emits a maintenance WATER
  for a fresh observed plant, but global foreman matching can give that task to
  an earlier worker instead of the planter.
- `CONFIRMED_EXPERIMENT`: the executor now records exact submitted PLANT
  assignments, confirms the matching crop and `watered_today == False` at the
  immediate next observation, and binds the existing generated
  `water_must_weed_boundary` task to the same worker. The binding is independent
  for each worker and is removed after success or stale/invalid state.
- `CONFIRMED_EXPERIMENT`: active starvation retains the existing FEED-only
  dispatch preemption. Immediate watering is deferred during that safety state,
  not allowed to displace FEED.
- Official engine A/B was unavailable on the host. Fast-engine paired results
  are recorded in `HISTORY.md` and are not official-engine evidence.

## Aggressive Sell WHEAT Reserve - Issue #28 Follow-up (2026-08-30)

- `CONFIRMED_EXPERIMENT`: aggressive sell-all previously sold all shed WHEAT, bypassing the normal `feed["shed_reserve"]` guard; official 1.32.7 showed 6 escapes vs 0.
- `CONFIRMED_EXPERIMENT`: aggressive WHEAT now sells `shed[WHEAT] - min(shed[WHEAT], shed_reserve)` with `shed_reserve = max(0, unfed - carried_wheat)`; non-WHEAT products remain fully sold. The protected amount is counted in `feed_reserve_protected_units` and visible in turn traces.
- Fast sanity (7,17 both seats, aggressive+plant-water+spare-water) has 0 escapes and no WHEAT churn while starving; official retest is still required.

## Engine Identity

- Latest confirmed upstream package version: `1.32.7`
- Upstream `pyproject.toml` currently declares `version = "1.32.7"`
- 1.32.6 town-rebalance PR: `Kaggle/kaggle-environments#1394`
- 1.32.6 town-rebalance merge commit: `1fa13d78387eb3661b1e621a4f5df150e6c3b646`
- 1.32.7 situational-resources PR: `Kaggle/kaggle-environments#1399`
- 1.32.7 situational-resources merge commit: `28b6d8af3ce73926b3d0fda1410c1ddd8384ab8c`
- PR #1399 head commit: `1fbd3b7571653434329d288dee9e068f54ff01c0`
- Local package version: `1.32.7` installed in the repository `.venv` for the Stage 1 exact-archive verifier; dependency manifests remain unchanged
- Vendored source commit: not established
- Engine file SHA-256: not established
- Specification file SHA-256: not established
- Live Kaggle leaderboard server version: 1.32.7 rollout announced, not independently server-verified
- Host statement: 1.32.7 should be the last balance change except game-breaking bugs
- Status: **current source contract is known, but do not treat this repository as a complete engine lock until local source/spec hashes and behavioral tests are recorded**

## Rollout Process Boundary — Issue #17

- `CONFIRMED_EXPERIMENT`: the issue #17 coordinator uses Python `spawn`; the
  parent owns policy objects and JAX/libtpu, while workers own independent
  scalar engine/opening/executor state. Workers exchange encoded manager-day
  NumPy rows only; primitive engine state remains process-local.
- `CONFIRMED_EXPERIMENT`: worker startup rejects loaded `jax`, `jaxlib`,
  `torch_xla`, `bc_manager_jax`, or `optax`; the lazy `rl_manager` initializer
  permits importing the worker protocol without those modules.
- `CONFIRMED_EXPERIMENT`: deterministic truncated fast-engine runs with two
  workers and two environments per worker preserve episode seeds, result
  ordering, central request batching, and complete normalized trajectory rows.
- `UNKNOWN`: Kaggle TPU steady-state throughput/scaling for the real BC-E
  checkpoint. It must be measured with the fixed command in
  `research/RL_MANAGER_PARALLEL_ROLLOUTS.md`; local CPU/mock numbers are not a
  TPU claim.

## Central Inference Batching — Issue #17 Extension (2026-08-28)

- `CONFIRMED_EXPERIMENT`: the existing `policy_day` owner grouping remains the
  default. `RunnerConfig.inference_batch_scope="policy"` groups only by the
  immutable `PolicyIdentity`, so rows with different days can share a call;
  each encoded row is passed unchanged.
- `CONFIRMED_EXPERIMENT`: `fixed_inference_batch_size=B` dispatches sorted real
  requests in deterministic B-sized chunks. A short chunk is padded by
  repeating its first real encoded row, never by a synthetic observation.
  Padding receives deterministic `padding/...` row IDs for the row-aware seam;
  padding outputs are neither sent to workers nor appended to trajectories.
- Owner metrics now distinguish real requests/batch sizes from physical call
  sizes/rows and padding, and report aggregate occupancy, queue wait, and
  inference seconds. Existing `requests`, `batches`, and `batch_sizes` fields
  remain real-row-compatible.
- `CONFIRMED_EXPERIMENT`: `PPOBatchedPolicy.plan_batch_with_row_ids` hashes the
  immutable policy identity for the root key and folds in each stable row ID.
  Real stochastic action/logprob outputs are invariant to neighboring rows and
  padding in the focused CPU test. This is not a guarantee for policies that
  implement only the legacy `plan_batch` seam.
- `UNKNOWN`: TPU speedup, occupancy, and real-checkpoint behavior. The local
  mock/fast-engine smoke is correctness and observability evidence only.

## PPO/BC Initialization Diagnostic - Issue #22 (2026-08-29)

- `CONFIRMED_EXPERIMENT`: the same encoded E row passed through
  `JaxEPlanPolicy` and a fresh deterministic `PPOPolicy` produced identical
  action tensors at B=1 and at fixed padded B=24 on the local CPU.
- Before the bounded alignment, the first raw-only difference was
  `animal_logits[0,0,10]` at approximately `5.8e-11` for both batch sizes;
  decoded actions still matched. The cause was the PPO wrapper's eager/private
  forward path versus the BC wrapper's compiled public path, not decode
  thresholding or batch-shape behavior.
- The PPO wrapper now uses public compiled `forward` and
  `forward_with_representation` for host-facing inference. The enclosing
  jitted PPO loss/update continues to use its private computational seams.
- Evidence is local CPU only; no TPU numerical or performance claim follows.
## Multi-Trainer TPU Prototype — Issue #21 (2026-08-28)

- `CONFIRMED_EXPERIMENT`: `rl_manager/multitrainer_benchmark.py` keeps JAX and
  libtpu in one Python process, creates independent PPO state/optimizer/RNG
  trees, and assigns trainer `i` explicitly to `jax.devices()[i]`. It refuses
  to reuse a device when N exceeds visible devices.
- `CONFIRMED_EXPERIMENT`: placement diagnostics cover params, optimizer state,
  RNG, inference inputs/outputs, PPO action/index arrays, and PPO scalar arrays.
  First-call compile timing is synchronized separately from steady-state
  inference and PPO update timing; update dispatch is compared sequentially
  and dispatch-all-then-block.
- `CONFIRMED_EXPERIMENT`: `rl_manager/multitrainer.py` routes by exact immutable
  `PolicyIdentity` and rejects unknown or cross-trainer trainable rows. This
  seam is JAX-free so CPU workers remain accelerator-unaware.
- `UNKNOWN`: TPU utilization, async overlap, memory behavior, and scaling for
  N=1/2/4/8 with the real BC-E checkpoint. Local CPU output is plumbing only;
  run the commands in `research/RL_MULTI_TRAINER_TPU.md` before making a TPU
  claim.

## Submission Runtime Invariant — Stage 1 / Issue #13

- A submission is not validated by importing `make_agent` from the repository
  or by running a separately staged directory. The acceptance artifact is the
  exact archive produced by `tools/build_submission.py`, freshly extracted to
  an empty directory, and raw-loaded through
  `kaggle_environments.agent.get_last_callable` with repository-root
  `PYTHONPATH`/`sys.path` and import-origin fallback prohibited.
- The archive must carry the complete local runtime import closure used by the
  tracked BC-E entrypoint: `executor_v0`, `bc_manager`, `opening_book`,
  `oracle`, `replay_daily`, and `fast_env`; the lazy `fast_env.market` import
  is explicitly smoke-tested before the official game. Missing that package is
  a packaging failure, not evidence of a weak trajectory or a late bank change.
- Verification enables strict executor diagnostics through
  `KAGGRICULTURE_SUBMISSION_STRICT=1`, while the production template defaults
  to the existing `strict=False` all-PASS fallback and preserves the default
  non-aggressive executor mode outside this submission configuration.
- The pinned compatibility check is official Kaggriculture 1.32.7, seed 7,
  candidate seat 1 versus PASS, with every status in the full environment
  history required to be `ACTIVE` or `DONE`. The preserved pre-behavior
  reference at revision `8f716bec` ended at bank **54,439** with archive SHA
  `4ccfcf25d30465661c912626a5d029210897ec5855c3dc2b55db2cdfd1a7d6cf` and
  action fingerprint
  `516fab6d316b76e8b93fce3b4d185e49b2df53aa742be6558574563c1929dc40`.
- The accepted post-Stage-4 compatibility reference was built from source
  revision `11ecead2d5efe8bf87fc0da533c739e344d7eaa6`; it ends at bank
  **47,290** with archive SHA
  `c12218ac1010c894ed22fd065049a290d03555c9f44ad0d6cc667fa52ee13de2` and
  action fingerprint
  `a38bf47884e5e6e89c2d77f7aab07819f3559e898af40372942460693c8b6afc`.
- Stage 1 local archive evidence is recorded in `HISTORY.md`. The authorized
  BC-E input's verified SHA-256 is
  `f4b029d3e463aba1db0544377d0d616e3de94aa6cc469d3446f018dddd8f6bf2`.
- Exact current accepted result: archive SHA-256
  `c12218ac1010c894ed22fd065049a290d03555c9f44ad0d6cc667fa52ee13de2`,
  official provenance 1.32.7, 720 full-history entries, zero status anomalies,
  candidate seat-1 bank `47,290.0`, and deterministic action fingerprint
  `a38bf47884e5e6e89c2d77f7aab07819f3559e898af40372942460693c8b6afc`.
- Stage 5 accepts the retained Stage 4 lifecycle sequencing (`b9c88ff`): the
  isolated panel mean was `63,592.3` vs `60,778.1` (+`2,814.2`) and median
  `65,509.5` vs `60,956` (+`4,553.5`), with no `<1k`/`<10k` cases and no
  errors, status anomalies, unaffordable orders, or animal losses. The six
  negative cases are labor/cash tradeoffs rather than feed/starvation failures;
  no concrete feed exception or unconstrained cash forecast is justified.
  Stage 3 fertilizer retention remains rejected and reverted at `7204103`.

## Match Contract

| Mechanic | Current understanding | Confidence |
|---|---|---|
| Players | 2 | `CONFIRMED_SOURCE` |
| Farm size | Separate 10×10 farm per player | `CONFIRMED_SOURCE` |
| Farm quadrants | Four 5×5 quadrants | `CONFIRMED_SOURCE` |
| Match length | 30 days × 24 turns = 720 turns | `CONFIRMED_SOURCE` |
| Starting money | 3,000 | `CONFIRMED_SOURCE` |
| Starting labor | One farmer | `CONFIRMED_SOURCE` |
| Starting land | Northwest quadrant | `CONFIRMED_SOURCE` |
| Additional land costs | NE 1,000; SW 2,000; SE 4,000 | `CONFIRMED_SOURCE` |
| Worker actions | One action per worker each turn | `CONFIRMED_SOURCE` |
| Market orders | Up to 10 per turn | `CONFIRMED_SOURCE` |
| Default action timeout | Approximately 1 second | `CONFIRMED_SOURCE`; reverify server settings |
| Final reward | Final banked money | `CONFIRMED_SOURCE` |
| Terminal inventory value | Zero unless sold before termination | `CONFIRMED_SOURCE` |

## Observation and Hidden Information

### Publicly visible

- both players' banked money;
- farm tiles and visible contents;
- worker positions;
- unlocked quadrants;
- hire count;
- shared market inventory and prices;
- town shops, including duplicate shop names;
- current day and hour.

### Private to each player

- own shed contents;
- own seed inventory;
- inventories carried by own workers.

Opponent shed, seeds, and carried inventories are hidden.

Confidence: `CONFIRMED_SOURCE`.

## Worker Actions

Observed action names:

- movement: `NORTH`, `SOUTH`, `EAST`, `WEST`;
- idle: `PASS`;
- logistics: `PICKUP`, `DROP`, `PLACE`;
- crops: `PLANT`, `WATER`, `HARVEST`, `FERTILIZE`, `DIG`;
- structures: `BUILD_COOP`, `BUILD_PASTURE`;
- animals: `FEED`, `CARE`;
- fertilizer: `COLLECT_FERTILIZER`.

Invalid or illegal actions generally become silent no-ops rather than terminating the episode.

Confidence: `CONFIRMED_SOURCE`; exact argument schemas remain to be locked locally.

## Market Actions

Observed market action names:

- `BUY_SEED`;
- `BUY_PRODUCT`;
- `BUY_ANIMAL`;
- `SELL`;
- `HIRE`;
- `BUY_LAND`.

Confidence: `CONFIRMED_SOURCE`.

## Crop Snapshot

| Crop | Seed cost | First/mature yield timing | Recurrence | Maximum yield |
|---|---:|---|---|---:|
| Wheat | 10 | first day 2; max day 4 | none | 6 |
| Carrot | 20 | day 2; max day 3 | none | 4 |
| Tomato | 50 | first day 8 | every 1 day | 4 |
| Strawberry | 100 | first day 10 | every 2 days | 4 |
| Melon | 80 | first day 10; max day 12 | none | 6 |

Additional crop behavior:

- planting day counts as unwatered;
- two consecutive unwatered daily refreshes turn a plant into weed;
- one-shot crop yield depends on watering during its later growth window;
- fertilizer doubles the relevant yield increment and remains active for the current day plus the next two days;
- mature non-recurring crops decay after their lifespan;
- simultaneous planting requests can be atomic by crop: if requested quantity exceeds owned seeds, the same-turn planting group may fail.

Confidence: `CONFIRMED_SOURCE`.

## Animal Snapshot

| Animal | Cost | Structure | First yield | Recurrence | Maximum held output | Product |
|---|---:|---|---|---|---:|---|
| Goose | 300 | Coop | day 4 | daily | 4 | Egg |
| Cow | 400 | Pasture | day 8 | every 2 days | 6 | Milk |
| Sheep | 500 | Pasture | day 6 | every 3 days | 6 | Wool |

Additional behavior:

- animals consume wheat feed (1 carried WHEAT per FEED, once per day);
- two consecutive unfed daily refreshes cause escape while leaving the structure;
- base +1 production accrues on schedule even on an unfed production day;
- care plus feed creates a pending production bonus consumed only on a FED
  production day (an unfed production day still resets the pending bonus to 0,
  losing it); the total is capped at `max_held`;
- animals generate collectible fertilizer daily (`COLLECT_FERTILIZER`, once
  per day, unavailable on the placement day before the first refresh);
- `DIG` never removes a placed animal but clears empty structures;
- `BUY_ANIMAL` lands in the shed and partially fills when funds or shed
  capacity run out mid-order.

Confidence: `CONFIRMED_SOURCE`; locked to zero first divergence vs the pinned
official engine by `tests/test_oracle_animals.py` (12 tests, real-official
same-action replay).

## Labor and Logistics

- hired-hand daily prices follow Fibonacci: 1, 1, 2, 3, 5, 8, 13, ...;
- hired hands disappear at end of day;
- workers reset near the central shed at day end;
- carried inventory auto-drops to shed at day end;
- default shed capacity is 100;
- overflow is discarded;
- seeds are separate from shed product capacity;
- movement onto locked tiles is allowed, while tile actions there generally no-op.

Confidence: `CONFIRMED_SOURCE`.

## Shared Market

Products:

- wheat;
- carrot;
- tomato;
- strawberry;
- melon;
- egg;
- milk;
- wool;
- fertilizer.

Global constants:

- initial target inventory `I0 = 10,000`;
- price floor = 1.

Base prices:

| Product | Base |
|---|---:|
| Wheat | 25 |
| Carrot | 35 |
| Tomato | 60 |
| Strawberry | 120 |
| Melon | 250 |
| Egg | 50 |
| Milk | 160 |
| Wool | 200 |
| Fertilizer | 100 |

Behavior:

- price is a product-specific function of market inventory relative to `I0`;
- orders execute one unit at a time in lockstep;
- both players see the same pre-commit price per unit during simultaneous execution;
- order sequence and quantity matter;
- sales at the price floor may not add market supply;
- lower town-center demand from 1.32.6 makes player sell pressure persist more strongly.

Confidence: `CONFIRMED_SOURCE`.

## Market Price Functions — 1.32.7

General source formula:

- if `inventory < I0`, scarcity raises price;
- if `inventory > I0`, glut lowers price;
- price is rounded to nearest dollar and floored at $1;
- each resource has a calibration quantity `T`, shape function, and target amplitude.

Supported shapes now include:

- `linear`
- `sq`
- `sqrt`
- `log`
- `log10`
- `hinge`

### Hinge shape

Added by merged PR #1399.

For scarcity distance `x = I0 - inventory` and calibration quantity `T`:

`u = x / T`

`hinge(x, T) = u + 8 * max(0, u - 1)^2`

Properties:

- `hinge(T, T) = 1`;
- below the knee (`x <= T`) the function is linear in normalized scarcity;
- above the knee (`x > T`) a quadratic term causes a steep price increase;
- therefore a product can remain ordinary until demand creates real scarcity, then become exceptionally valuable.

### Current market curve table

| Resource | Base | T | Below func | Below target | Above func | Above target | P(I0-T) | P(I0+T) | P(I0+2T) |
|---|---:|---:|---|---:|---|---:|---:|---:|---:|
| Wheat | 25 | 400 | sqrt | 0.80 | log | 0.20 | 45 | 20 | 19 |
| Carrot | 35 | 450 | **hinge** | **1.00** | sqrt | 0.70 | 70 | 10 | 1 |
| Tomato | 60 | 200 | **hinge** | 0.40 | sqrt | 0.60 | 84 | 24 | 9 |
| Strawberry | 120 | 100 | sqrt | 0.70 | linear | 1.60 | 204 | 1 | 1 |
| Melon | 250 | 300 | log | 0.20 | sq | 3.60 | 300 | 1 | 1 |
| Egg | 50 | 332 | **hinge** | 0.40 | log | 0.20 | 70 | 40 | 39 |
| Milk | 160 | 122 | sqrt | 0.60 | linear | 1.60 | 256 | 1 | 1 |
| Wool | 200 | 105 | log | 0.20 | sq | 3.20 | 240 | 1 | 1 |
| Fertilizer | 100 | 200 | linear | 0.40 | linear | 0.40 | 140 | 60 | 20 |

### Source test points for hinge resources

These are explicit expected values added with PR #1399:

| Product | Market inventory | Price |
|---|---:|---:|
| Carrot | 10,000 | 35 |
| Carrot | 9,775 | 53 |
| Carrot | 9,550 | 70 |
| Carrot | 9,400 | 113 |
| Carrot | 9,100 | 385 |
| Tomato | 10,000 | 60 |
| Tomato | 9,900 | 72 |
| Tomato | 9,800 | 84 |
| Tomato | 9,700 | 144 |
| Tomato | 9,500 | 552 |
| Egg | 10,000 | 50 |
| Egg | 9,834 | 60 |
| Egg | 9,668 | 70 |
| Egg | 9,502 | 120 |
| Egg | 9,170 | 460 |

Important detail: tomato and egg keep the prior linear curve's `below_target = 0.40`, so behavior through the knee is unchanged from the old linear curve; the new behavior appears only beyond the knee. Carrot also changes `below_target` from 0.20 to 1.00, making scarcity more valuable even at the knee.

Confidence: `CONFIRMED_SOURCE` from merged PR #1399.

## Shop Demand Relevant to 1.32.7

Source notes from PR #1399:

- **Carrot** is consumed by pet cafes and farmers markets. Pet cafe is single-product, so its per-tick consumption is doubled.
- **Tomato** is consumed by pizza shops and farmers markets.
- **Egg** is consumed by bakeries and brunch spots.

Because shops are sampled with replacement since 1.32.6, repeated copies can drive one of these products past its scarcity knee if neither player supplies it.

Host-reported probability of significant price increase assuming **no production**:

- tomato: ~50% of games;
- carrot: ~26% of games;
- egg: ~22% of games.

Confidence for percentages: `HOST_REPORTED_STAT`; reproduce empirically once the local 1.32.7 engine is locked.

## Town Demand — Current Contract

### Town center

From merged PR #1394 / 1.32.6:

- default interval = 24 turns;
- with 24 turns/day, consumes once/day;
- removes one of each non-fertilizer product;
- flat for the whole season;
- old 2× after day 10 / 4× after day 20 schedule removed.

### Town shops

- unlock default every 3 days;
- draw uniformly from full shop table **with replacement**;
- duplicate shop names allowed;
- every duplicate instance consumes independently;
- unlocking stops at 8 total instances;
- shop consumption default every 4 turns;
- single-product shops consume at 2× per tick.

Consequences:

- town observation must preserve shop multiplicity;
- future demand composition remains stochastic;
- 1.32.7 turns some rare shop compositions into sharp scarcity opportunities rather than merely mild demand differences.

Confidence: `CONFIRMED_SOURCE`.

## Known Recent Engine Drift

### 2026-08-04 — shed-capacity enforcement

`BUY_PRODUCT` and `BUY_ANIMAL` respect shed capacity.

### 2026-08-07 / 1.32.6 — town rebalance

- town center 2 ticks/day → 1 tick/day;
- late-game town-center multipliers removed;
- shops sampled with replacement;
- duplicate shop instances consume independently;
- max shop instances = 8.

### 2026-08-16 / 1.32.7 — situational underused resources

Merged PR #1399:

- adds `hinge` market shape with gain 8;
- carrot scarcity curve: `log`/0.20 → `hinge`/1.00;
- tomato scarcity curve: `linear`/0.40 → `hinge`/0.40;
- egg scarcity curve: `linear`/0.40 → `hinge`/0.40;
- glut-side curves remain unchanged;
- intended to make carrot, tomato, and goose/egg production situationally viable under randomized shop demand.

## Fast-Engine Differential Parity — Worker/Ordering/Hiring/Market (2026-08-23)

First Stage-2b mechanics cluster proven against the real pinned official
1.32.7 engine with same-action replay and full canonical compare after every
turn (`tests/test_oracle_mechanics.py`, 27 scenarios; zero divergence). This
proves fast-engine parity for the exercised actions only — not full parity.

Confirmed exact behaviors (now `CONFIRMED_EXPERIMENT` via differential oracle,
in addition to `CONFIRMED_SOURCE`):

- worker inventory: no fixed slot count (item→quantity semantics); PICKUP n is
  unbounded and clamped only by shed stock; seed names are never carried;
  day-end carried inventories drop into the shed with overflow discarded;
- same-turn ordering: workers act before the market — a PICKUP frees shed room
  before a same-turn BUY, goods deposited this turn are sellable this turn,
  and a same-turn market buy can never be picked up this turn;
- hiring: Fibonacci prices 1, 1, 2, 3, 5, 8, ... scaled by `farmHandCostMult`;
  hire works at hour 0; a new hand cannot act on its hire turn but acts the
  next turn; unaffordable hires stop silently; `hires_today` resets at day end;
- market: up to 10 ordered slots per player per turn with silent truncation of
  extras; both players processed together per slot; HIRE/BUY_LAND are atomic
  and ignore extra arguments; BUY/SELL commit per unit with both players
  quoted from the same pre-commit inventory; orders abort mid-quantity on
  insufficient funds or shed capacity while later slots still run; order
  quantities are unbounded (resource-bounded only); a same-turn BUY 1 / SELL 1
  round-trip nets exactly zero.

Exact divergences found and fixed in the fast engine (each locked by a named
regression):

1. money observation decode trusted the raw f32 `normalize(10000)` round-trip,
   so any money change produced spurious canonical divergences (official
   2993.0 vs fast 2992.999755859375) — decode now rounds (`fast_env/api.py`);
2. market/PICKUP/PLACE quantities were clamped to `MAX_QUANTITY = 100`
   (BUY_SEED WHEAT 150 granted 100 seeds for 2000 money vs official 150 seeds
   for 1500) — clamps removed in favor of resource bounds, BUY_SEED cost
   widened to i64, and the official per-slot 100k lockstep iteration escape
   mirrored (`rust/kaggriculture_env/src/lib.rs`);
3. the wire translation raised ValueError for inputs the official engine
   silently ignores (11th market order, unknown unit ops, seed-name PICKUP,
   unknown PLANT crop, non-dict action, missing farmer, non-integer order
   quantity) — malformed actions now translate to no-op rows and hands/market
   lists truncate like the official interpreter (`fast_env/api.py`).

Former bounded deferral, now CLOSED: the >16 simultaneous hired hands gap was
removed by the exact-layout revision (MAX_HANDS=240; see the dedicated section
below). Submitted hand-action lists now truncate at 240 entries, which covers
every hand count the pinned default contract can reach.

Confidence: `CONFIRMED_EXPERIMENT` (differential oracle, pinned 1.32.7).

## Fast-Engine Exact Hand Capacity — MAX_HANDS=240 Layout Revision (2026-08-23)

The old fixed 16-slot layout was replaced by the exact default-contract
capacity. Derivation: official HIRE appends exactly one hand per atomic market
order (`_do_hire`), the market queue is truncated to
`maxMarketOrdersPerTurn = 10` orders per turn, `turnsPerDay = 24` turns make
one day, and `farm["hands"] = []` clears at every day reset — so the maximum
simultaneous hand count is exactly `10 * 24 = 240` under the pinned default
configuration. The generator now derives `max_hands` from these two schema
defaults (`scripts/generate_fast_protocol.py`); non-default configurations
with larger turnsPerDay/maxMarketOrdersPerTurn exceed this bound and remain
out of scope (fast engine supports boardSize=10 / maxMarketOrdersPerTurn=10
only, as before).

New constants and wire shapes (breaking layout change):

- `MAX_HANDS` 16 → 240; `ACTION_SLOTS` 27 → 251 (= farmer + 240 hands + 10
  market orders); market action rows moved from slot 17 to slot
  `MAX_HANDS + 1 = 241`; all Rust loops parameterized;
- `OBS_SIZE` 5630 → 8766 (+3136): only the two MAX_HANDS-scaled blocks move —
  hand positions block `2*(MAX_HANDS+1)` wide after offset 5280, and the
  per-hand inventory block `MAX_HANDS*12` wide; every other block and the
  three reserved gaps keep their original widths. Offsets:
  OBS_SHED 5319→5767, OBS_SEEDS 5331→5779, OBS_INVENTORY 5336→5784,
  OBS_ANIMAL_INVENTORY 5345→5793, OBS_HAND_INVENTORY 5348→5796,
  OBS_MARKET_INVENTORY 5540→8676, OBS_MARKET_PRICES 5549→8685,
  OBS_SHOPS 5560→8696;
- `MASK_SIZE` 3562 → 34026 (= `(MAX_HANDS+1)*136 + 10*125`);
- per-env buffer deltas (f32 obs ×2 players, i64 actions ×2 players, u8 masks
  ×2 players): observations 45,040 → 70,128 B (+25,088), actions 1,296 →
  12,048 B (+10,752), masks 7,124 → 68,052 B (+60,928);
- the extension module now exports `MAX_HANDS`, `ACTION_SLOTS`, `MASK_SIZE`
  alongside `OBS_SIZE`; preallocated `*_into` calls reject stale shapes with
  `ValueError`.

HIRE mask semantics locked to the official gate: HIRE in market slot 0 is
available iff `hand_count < MAX_HANDS AND money >= fib(hires_today)` with the
engine Fibonacci `fib(0)=fib(1)=1, fib(2)=2, ...`. Regression
(`tests/test_fast_env.py::test_hire_mask_matches_official_reachable_semantics`)
proves both sides: open at reset, closed at 23 hands where 75,024 spent of
100,000 leaves 24,976 < fib(23) = 46,368 for the next hire.

Evidence vs the real official 1.32.7 engine (`tests/test_oracle_hands.py`,
same-action replay with full canonical compare each turn, zero divergence):
exactly 16 hands (old boundary) with subsequent hand actions; a
startingMoney=100000 trace reaching 23 hands plus two turns of real hand
actions over all 23 slots; the exact 16→17 crossing turn; day-end reset from
23 hands with hires_today/inventory reset parity and next-day Fibonacci
restart at cost 1; and the fast HIRE mask bit equal to the official-reachable
gate evaluated from shared canonical state on every turn.

Scope notes: `bc_manager/constants.py::MAX_HANDS = 8` is a separate BC-manager
head-slot constant and intentionally unchanged. This revision proves parity
for exercised traces only; no full-episode or training-safety claim is made
until full 720-turn episodes pass through the oracle.

## Fast-Engine Differential Parity — Crop/Seed/Tile Lifecycle (2026-08-23)

Second Stage-2b mechanics cluster proven against the real pinned official
1.32.7 engine with same-action replay and full canonical compare after every
turn (`tests/test_oracle_crops.py`, 16 tests over 15 scenarios, 2,136 turn
pairs, 74 day boundaries; zero divergence). This proves fast-engine parity for
the exercised crop mechanics only — not full parity.

Confirmed exact behaviors (now `CONFIRMED_EXPERIMENT` via differential oracle,
in addition to `CONFIRMED_SOURCE`):

- `CROPS` table constants (seed cost / first_yield_day / max_yield_day /
  interval / max_yield / ongoing per crop);
- PLANT: tile must be empty and unlocked; the global seed pool is consumed
  only on success; fresh tiles get `consecutive_unwatered=1` (planting day
  counts as unwatered), 1 yield unit non-ongoing / 0 ongoing,
  `max_lifespan_step = -1` ongoing else `(day+max_yield_day+1)*24`,
  `fertilized_until_day=-1`, `watered_today=False`; invalid plants are silent
  no-ops; group PLANT demand counts across farmer AND every submitted hands
  entry (including entries beyond the hired-hand count) and blocks ALL of a
  short-supplied crop's requests to PASS;
- WATER: once per day; single-harvest bonus window
  `(max_yield_day+1)//2 .. max_yield_day` pays 2 when fertilized else 1;
- FERTILIZE: consumes 1 carried fertilizer; `fertilized_until_day =
  max(old, day+2)` — active on the application day plus the following two;
- HARVEST: silent no-op before `first_yield_day` even with positive yield;
  drains all units; non-ongoing crops are removed from the tile;
- `_daily_refresh_plants`: unwatered streak >=2 converts to WEED; ongoing
  interval accrual doubles only on watered days; `max_lifespan_step` is set
  when the production count reaches `max_yield`;
- `_decay_plants`: at `step >= max_lifespan_step` with `(step-mls)%2==0`,
  decrement yield UNCONDITIONALLY and convert to WEED when the result is <= 0;
- DIG clears plants/weeds/structures but never a placed animal; wrong-tile
  WATER/FERTILIZE/HARVEST are guarded no-ops.

Exact divergence found and fixed in the fast engine (locked by
`test_ongoing_tomato_daily_interval_harvest_survival_and_zero_yield_decay`):

1. `_decay_plants` gated the decrement on `yield > 0` and converted at
   `== 0`, while the official engine decrements unconditionally and converts
   at `<= 0`. An ongoing crop harvested down to exactly zero yield when its
   production completed stayed alive as a zero-yield PLANT forever instead of
   becoming a WEED at `max_lifespan_step`
   (`rust/kaggriculture_env/src/lib.rs`).

Confidence: `CONFIRMED_EXPERIMENT` (differential oracle, pinned 1.32.7).

## Fast-Engine Differential Parity — Town/World Updates, Day RNG, Reset, Terminal (2026-08-23)

Fourth Stage-2b mechanics cluster proven against the real pinned official
1.32.7 engine with same-action replay and full canonical compare after every
turn (`tests/test_oracle_town_world.py`, 10 tests over 10 scenarios,
~1,100 turn pairs including one 648-turn PASS-only season segment; zero
divergence, no engine changes required). This proves fast-engine parity for
the exercised town/world mechanics only — not full parity.

Confirmed exact behaviors (now `CONFIRMED_EXPERIMENT` via differential oracle,
in addition to `CONFIRMED_SOURCE`):

- shop unlock timing: exactly at end-of-day when `(day + 1) %
  townShopUnlockInterval == 0`, i.e. canonical steps 72/144/216/... for the
  default 3-day interval; nothing unlocks before day 3;
- draw-with-replacement multiplicity: seed 3 PASS-only draws BRUNCH_SPOT,
  FARMERS_MARKET, PET_CAFE, YARN_STORE, PIZZA_SHOP, FARMERS_MARKET,
  FARMERS_MARKET, BAKERY — three duplicate FARMERS_MARKET instances kept as
  ordered multiplicity; the 8-instance cap holds (no ninth unlock through
  day 27);
- consumption: every `townShopSellInterval == 4` steps each instance drains
  its product list once per unit of demand with multiplier 2 for
  single-product shops (PET_CAFE, YARN_STORE); every
  `townCenterSellInterval == 24` steps the town center drains one unit of
  each non-FERTILIZER product, firing at step 0 while the town is still
  empty; prices refresh visibly on the same transition;
- insufficient market stock: BUY_PRODUCT has no stock check (money and shed
  room only) and town consumption subtracts unconditionally, so configured
  money/shed capacity drive WHEAT stock below −20,000 with prices tracking
  the scarcity branch down there and never below `PRICE_FLOOR`; both engines
  agree exactly at negative stock;
- day-RNG stream: one `random.Random((seed * 1_000_003) ^ day)` per boundary
  shared across both farms in player order — weed draws over empty unlocked
  tiles row-major (farm 0 then farm 1), then the shop choice; a planted tile
  shifts the stream position until it converts to a WEED, and both engines
  keep identical boards and shop draws (`weedSpawnChance` 0 / default / 0.5
  all verified; zero chance never spawns; high chance spawns only on empty
  unlocked tiles);
- determinism: same seed + same actions reproduce bit-identical canonical
  state across fresh backend instances (reset-after-run repeatability); a
  different seed diverges in weed layout and shop draws wherever a draw
  occurs;
- day-boundary reset ordering: hands removed, carried inventories dropped to
  the shed (overflow discarded), `hires_today` reset (next hire costs
  Fibonacci 1 again), farmer returned to spawn, plant watering counters
  advanced — all before the new day's first action and before any shop
  unlock fires on the same boundary;
- terminal lifecycle: DONE + reward = final farm money land exactly at
  canonical step `episodeSteps - 1` (day/hour consistent); the official
  wrapper refuses post-terminal steps (`FailedPrecondition`) and the fast
  engine transitions nothing further; pre-terminal status anomalies stay
  invalid despite terminal DONE (offline status-history suite).

Confidence: `CONFIRMED_EXPERIMENT` (differential oracle, pinned 1.32.7).

## Fast-Engine Differential Parity - Full-Episode Legal-ish Corpus (2026-08-23)

Broad semantic acceptance campaign over complete default episodes
(decision D-022; generator `oracle/action_generator.py`, runner
`scripts/run_parity_corpus.py`, report `research/parity_corpus_report.json`):

- Episode contract (locked): a default "720-step episode" is ONE reset
  observation plus exactly 719 accepted primitive `step` calls; the terminal
  DONE transition lands at canonical step 719 = day 29 hour 23; 29
  day-boundary transitions per episode.
- Result: ZERO first divergence across seeds 0, 1, 2, 7, 17, 42, 123, 999 —
  initial state plus all 719 transitions per episode compared on the full
  canonical schema; official and fast terminal observations/rewards/statuses
  equal for both seats in every episode.
- Coverage: 33 action families, 28,508 attempted instances across the corpus
  union — movement, PASS, PICKUP/PLACE/DROP, PLANT/WATER/FERTILIZE/HARVEST/
  DIG, BUILD_COOP/BUILD_PASTURE, FEED/CARE/COLLECT_FERTILIZER (farmer and
  hands), BUY_SEED/BUY_PRODUCT/BUY_ANIMAL/SELL/HIRE/BUY_LAND, malformed
  market entries, unknown ops, missing/non-integer quantities, >10-order
  truncation bursts, extra hand slots, unaffordable orders; both seats every
  turn; 30 day cycles including day-RNG weed/shop draws and town consumption.
- No new engine mismatch was found at full-episode scale: the earlier
  cluster fixes (money f32 decode, MAX_QUANTITY clamp, wire no-op
  translation, zero-yield decay) held under broad legal-ish pressure.
- Repeatability: same generator seed reproduces an identical action trace;
  fast reset+replay reproduces identical canonical states/rewards/statuses.

Confidence: `CONFIRMED_EXPERIMENT` (bounded: parity proven for the states
these episodes reach, not universal proof).

## Randomness and Determinism

Current interpretation:

- most crop, animal, movement, labor, and market mechanics are deterministic given engine version and both action streams;
- weeds are seed-driven stochastic events;
- town-shop draws are seed-driven and sampled with replacement;
- current shop multiset is public, future draws are unknown;
- market prices are deterministic functions of inventory, so the stochasticity enters through shop draws and policies rather than random prices;
- 1.32.7 amplifies economic consequences of certain shop-draw tails without adding new RNG;
- opponent policy remains the other major source of strategic uncertainty.

Confidence: mostly `CONFIRMED_SOURCE`; exact RNG draw ordering still needs experimental mapping.

## Native Batched Fast Backend - Issue #16

- `fast_env.BatchedFastEnv(N, configuration)` owns N independent `GameState`s in one `RustBatchEnv`; `reset(seeds)` requires exactly N explicit unsigned 64-bit seeds and preserves per-environment RNG isolation.
- Every step uses one preallocated `[N, 2, ACTION_SLOTS, 3]` i64 action buffer and one native `step_into` call. Observation, reward, and status buffers are also reused; nested Python observations are refreshed after the native call.
- Public farms/market/town are decoded once per environment and attached to both seat views; private shed/seeds/carried inventories are decoded from the corresponding seat row only. Default observations match scalar-fast exactly. The adapter's `canonical_observations=True` mode emits executor-compatible planted/placed-day fields.
- `oracle.batched_backend.BatchedEngineBackend` is the framework-neutral rollout seam. `rl_manager.RunnerConfig(batch_backend=True)` uses one adapter per lockstep chunk, while the existing scalar runner remains the reference path.
- Evidence: `tests/test_fast_batched_env.py` and the native batch runner test prove fixed-seed scalar parity, terminal status/reward parity, action encoding parity, private-view isolation, and self-play trace/bank/status equality. Quantitative local measurements are recorded in `docs/benchmarks/ISSUE16_BATCHED_FASTENV.md`.

Confidence: `CONFIRMED_EXPERIMENT` for the exercised default-contract traces; no universal claim for unsupported non-default layout configurations.

## Required First Regression Tests

1. 720-turn episode termination and terminal reward.
2. Starting money, worker, land, and positions.
3. Land-purchase costs and unlock order.
4. Market-order count limit.
5. Simultaneous market lockstep and price updates.
6. Atomic planting failure when seeds are insufficient.
7. Watering, weed conversion, maturity, recurrence, and decay.
8. Fertilizer duration and yield effect.
9. Animal feeding, care bonus, output cap, fertilizer, and escape.
10. Day-end hand disappearance, worker reset, and carried-inventory drop.
11. Shed overflow and purchase-capacity behavior.
12. Terminal unsold inventory receives no reward.
13. Repeatability under identical seed and action streams.
14. Seat-swapped equivalence for symmetric agents.
15. Town center consumes exactly once/day at flat 1×.
16. Duplicate shop sampling and independent duplicate demand.
17. Eight-instance shop cap.
18. Hinge function exact values at/below/above `T`.
19. Carrot/tomato/egg source test prices listed above.
20. Other products' market curves remain unchanged across the 1.32.6 → 1.32.7 transition.
21. Empirically estimate no-production scarcity-event frequencies for carrot/tomato/egg across many seeds.

## Unresolved Questions

- Exact live leaderboard server version during/after rollout.
- Exact episode-seed construction and RNG draw order.
- Whether any server configuration differs from repository defaults.
- Exact care bonus in the live engine.
- Full consequences of price-floor sales.
- Exact operational definition used by the host for the reported 50%/26%/22% scarcity frequencies.
- How often rational opponent production suppresses hinge-price opportunities relative to no-production statistics.
- Whether bug-fix engine changes land after 1.32.7.

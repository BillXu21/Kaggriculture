# Coding-Agent Infrastructure Sprint

Date: 2026-08-23
Status: active planning backlog for the temporary high local coding-agent-usage window

## Purpose

Use the temporary increase in local coding-agent capacity on work that is:

- implementation-heavy;
- mechanically testable;
- very likely to be needed regardless of later strategic choices;
- separable from current policy quality;
- safe to build before PPO/self-play tuning.

Do **not** use this window to prematurely lock uncertain RL architecture, reward shaping, opponent strategy, or executor heuristics. The immediate competitive gate remains a real engine-executed BC-manager + executor game with useful compliance.

The sprint should produce reusable infrastructure while preserving the project rule:

> Never add another layer of RL complexity until learning has been demonstrated in the simpler stationary problem underneath it.

## Working rule

Prefer sequential bounded packets directly on `main` rather than a large branch tree. Every packet must have:

1. a narrow contract;
2. deterministic tests;
3. an explicit stop condition;
4. no hidden strategic assumptions;
5. documentation of simplifications/deferred upgrades/revisit triggers.

When two packets are independent enough to develop concurrently, avoid concurrent pushes to `main`; either run them sequentially or have the later agent start only after the earlier commit lands.

---

## Packet A — Evaluation / Match Runner Foundation

### Why now

This is needed immediately for executor validation, frozen-opponent evaluation, differential engine testing, self-play, league play, and promotion tests.

### Deliver

A small `evaluation/` package that can:

- run one or many games through the official `kaggle_environments` runner;
- adapt stateful one-argument agents to the actual Kaggle `(obs, config)` invocation contract;
- run either seat and paired seat-swapped matches;
- use explicit fixed seed lists;
- preserve per-turn status history so terminal `DONE` cannot hide earlier `ERROR`/`INVALID`/`TIMEOUT`;
- save compact JSONL match records;
- record engine version/config, exact agent identities, seed, seat, final banks/rewards, statuses, runtime, and optional executor diagnostics;
- support pass/random only as plumbing opponents;
- load frozen repo/public opponents through explicit adapters later;
- produce aggregate W/L/T, bank margin, seat split, error rate, and paired summaries.

### Stable interface

Keep the runner independent of PPO and independent of the custom fast engine. Define a small match-result schema that both official and fast backends can emit.

### Acceptance

- official 1.32.7 one-game smoke exercises a real stateful executor and produces non-empty diagnostics;
- paired seat-swapped smoke works;
- an intentionally failing agent is recorded as failure even if the final engine state becomes `DONE`;
- exact result JSON round-trips;
- deterministic seeds reproduce results.

### Deferred

- distributed cluster orchestration;
- Bayesian ratings;
- dashboards;
- large opponent catalogs.

---

## Packet B — Exact Fast Engine V0 + Differential Oracle

### Why now

PPO/self-play throughput will eventually be dominated by simulation. This is large mechanical work well suited to coding-agent capacity, but correctness must dominate speed.

### Design principle

Do **not** invent a new game. Port/factor the pinned 1.32.7 semantics into a lightweight local simulator with the official engine as the oracle.

Target pinned source:

- `kaggle-environments 1.32.7`;
- upstream merge commit `28b6d8af3ce73926b3d0fda1410c1ddd8384ab8c`.

### V0 architecture

Start scalar and exact before attempting vectorization:

- no framework renderer;
- no OpenSpiel/environment registry/import overhead;
- no JSON-schema validation in the hot loop;
- direct in-process Python state transition;
- same observations/actions/reward semantics required by our agents;
- deterministic seeded RNG matching the pinned engine.

Only after scalar differential parity is strong should the agent optimize object layout, copies, batching, or multiprocessing.

### Differential oracle

Build tests that execute the **same legal action trace** in official and fast engines and compare after every turn:

- day/hour/step;
- both farms money;
- boards/worker positions/unlocked land/hires;
- own private shed/seeds/inventories for each perspective;
- market inventory/prices;
- town shops;
- terminal status/reward.

Use:

- fixed hand-authored mechanic probes;
- random legal-ish traces;
- traces from real replay actions;
- seeds spanning shop/weed randomness;
- full 720-turn episodes.

Failure should print the first divergent field/turn/action.

### Benchmarks

Record official-vs-fast episodes/sec and turns/sec on representative CPU hardware. Speed is not an acceptance criterion until parity passes.

### Acceptance

- pinned mechanic probes pass;
- full-episode differential tests pass on a substantial fixed seed/action corpus;
- deterministic repeatability passes;
- fast engine emits the same match-result schema as Packet A;
- benchmark is recorded.

### Deferred

- NumPy/JAX/Torch vectorized simulator;
- Cython/Rust/C++ rewrite;
- speculative semantic shortcuts;
- approximate market/lifecycle models.

Revisit those only if scalar exact engine throughput remains insufficient.

---

## Packet C — Population / League Infrastructure

### Why now

The mechanics of keeping opponents, scheduling matches, promotion, and history are implementation-heavy but conceptually stable. They can be tested with toy agents before PPO exists.

### Deliver

A small `league/` package with:

#### Immutable agent identity

Each participant records:

- logical name/family;
- source commit;
- checkpoint path + checkpoint hash when applicable;
- model config / executor version;
- creation timestamp or training step;
- tags such as `champion`, `candidate`, `historical`, `baseline`.

Normal Git provenance is enough; do not build an enterprise artifact registry.

#### Population store

- champion;
- current candidate/latest;
- historical snapshots;
- frozen public/baseline agents;
- later targeted exploiters.

Metadata should be JSON/YAML and portable; large model files remain ordinary files/datasets/checkpoints.

#### Match scheduling

- deterministic seed generation;
- paired seat swapping;
- fixed evaluation panels;
- configurable opponent mixtures;
- resumable result files without duplicate matches.

#### Opponent sampling V0

Implement only simple policies initially:

- fixed list/weights;
- uniform population;
- latest/champion-heavy mixture.

Expose a clean sampler interface so PFSP or rating-aware sampling can be added later, but do not implement complex sampling until evidence supports it.

#### Promotion gate

Given candidate + champion + frozen panel:

- run paired fixed-seed matches;
- require configurable W/L/T or margin thresholds;
- report uncertainty/sample count;
- never silently promote on training reward alone.

Promotion itself should be an explicit caller action, not hidden inside evaluation.

#### Ratings

- record raw cross-play matrix first;
- provide simple Elo-style online summaries if useful;
- provide Bradley-Terry fitting for offline comparison because competition final ranking uses Bradley-Terry;
- ratings are diagnostics, not replacement for fixed paired gates.

### Acceptance

Use deterministic toy agents to verify:

- population snapshots/history;
- sampler determinism;
- no duplicate scheduled matches on resume;
- seat balance;
- promotion pass/fail cases;
- cross-play matrix + rating outputs.

### Deferred

- AlphaStar-style PFSP variants;
- exploitability search;
- automatic exploiter spawning;
- distributed league service;
- sophisticated matchmaking.

---

## Packet D — Rollout / Training Data Plane (Algorithm-Neutral)

### Why now

Self-play training will need a large amount of boring plumbing regardless of whether the first refinement loop is PPO exactly as currently expected.

### Deliver

A minimal `rl/rollout/` layer that can store manager-decision trajectories at the current once-per-day abstraction:

- observation tensors/features used by the manager;
- requested plan/action representation;
- old policy log-prob components when available;
- reward/terminal outcome;
- value estimate placeholder;
- episode/seed/seat/opponent identity;
- executor compliance diagnostics;
- policy/checkpoint identity.

Support deterministic serialization/chunking and concatenation. Keep the schema versioned.

Add generic utilities for:

- discounted returns;
- GAE with unit tests;
- advantage normalization;
- minibatch slicing/shuffling under explicit seed;
- train/eval split by episode/opponent/seed, never random rows across the same episode.

### Important boundary

Do **not** yet alter the BC network or commit to a PPO loss over the current factorized heads. The closed-loop executor test may still force changes to the plan/action contract. This packet is data-plane plumbing only.

### Acceptance

Synthetic trajectories reproduce hand-calculated returns/GAE; serialization round-trips; batching is deterministic; episode boundaries cannot leak.

---

## Packet E — PPO Policy Interface / Learner Skeleton (Only After Executor Gate)

This packet can be prepared as a design contract during the sprint, but implementation should start only after the real BC/executor loop demonstrates useful compliance.

Expected work:

- expose distributions/log-probs/entropy for current manager heads rather than only argmax decoding;
- add a value head;
- define how count heads and sell presence/quantity contribute to joint log-prob;
- PPO clipped objective, value loss, entropy diagnostics, KL/clip fraction;
- checkpoint loading from the successful BC weights;
- fixed-opponent stationary training first;
- no self-play until stationary PPO learning is demonstrated.

Hard rule: do not use the coding-agent sprint as justification to skip the stationary-learning gates from the Pokémon retrospective.

---

## Packet F — Self-Play Orchestrator

Implement after Packets A/C/D are stable; it can initially run fake policies without PPO.

Responsibilities:

- select opponent via league sampler;
- select deterministic seed/seat pair;
- run official or fast backend through one common match API;
- collect manager-level rollout records;
- attach immutable policy/opponent identities;
- write resumable chunks;
- periodically request candidate evaluation against champion/frozen panel;
- snapshot only at explicit cadence/criteria.

Keep learner and rollout workers separable so CPU simulation and GPU learning can be scaled independently later.

### Deferred

- asynchronous policy lag machinery;
- distributed RPC;
- elastic worker pools;
- complex replay prioritization.

Start with local multiprocessing/process pools only if profiling says one process is insufficient.

---

## Recommended sprint order

1. **Finish current executor engine smoke and establish that the agent actually plays.**
2. **Packet A — evaluation/match runner.** This immediately prevents false-positive smokes like the 720-step `DONE/DONE` case.
3. **Packet B — scalar exact fast engine + differential oracle.** Largest implementation-heavy packet and highest leverage for later RL.
4. **Packet C — population/league infrastructure.** Can be fully tested with toy agents.
5. **Packet D — rollout/data-plane utilities.** Algorithm-neutral and safe.
6. **Packet F — self-play orchestrator using fake/frozen policies.** Proves plumbing without learning.
7. **Packet E — PPO learner/model changes only after the executor compliance gate passes.**

Packets A–D/F are intentionally useful even if the first PPO formulation changes.

---

## What we should *not* spend the temporary usage burst on

- broad neural architecture sweeps;
- reward-shaping experiments;
- complex executor optimization before compliance is measured;
- opponent modeling before basic self-play works;
- PFSP/AlphaStar-style league sophistication before fixed-mixture self-play works;
- distributed systems before local throughput is benchmarked;
- vectorized approximate simulation before scalar exact differential parity;
- search/MCTS before stationary PPO has shown learning.

The coding-agent advantage should buy us **tested infrastructure and implementation breadth**, not speculative complexity.
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

## D-010 — Use Codex Only for Explicitly Authorized, Bounded Implementation Packets

- Date: 2026-08-06; superseded/clarified 2026-08-21
- Status: active
- Decision: Do not spend Codex speculatively. Use it only when the user explicitly authorizes a bounded implementation packet with a concrete stop condition; research/discussion and simple notebook glue should not automatically be delegated to Codex.
- Rationale: Codex budget is finite, and prior projects lost time to broad implementation/architecture churn before simpler learning assumptions were established. Bounded packets keep implementation aligned with validated decisions.
- Revisit when: Codex cost/availability or the project workflow changes materially.

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

## D-017 — Bootstrap BC From Elite Post-Patch Daily Replays

- Date: 2026-08-21
- Status: active
- Decision: Build the first behavior-cloning corpus from Kaggle's daily **top-rated** Kaggriculture episode datasets rather than broad/random ladder data. Use both seats from an episode when the manifest's `min_score` clears the chosen elite threshold, so both demonstrations are known to come from strong submissions even though per-seat submission IDs/ratings are unavailable.
- Initial window: use five recent complete 1.32.7 daily partitions after allowing roughly two days for competitors to adapt to the balance patch; target 2026-08-17 onward and add later daily partitions as they become available. Embedded `module_version == 1.32.7` remains the authoritative replay check.
- Data retention: preprocess broadly enough to preserve `episode_id`, date, `avg_score`, `min_score`, derived `max_score`, player/seat, seed, final reward, and the raw replay provenance so later experiments can change score thresholds or sampling without reacquiring the original data.
- Rationale: the top-daily archive is deliberately filled from the highest-average-rated games until its 20 GiB cap, so the main problem is selecting among strong demonstrations, not excluding obviously weak random play. Multiple submissions from one player can differ substantially in strength, making player-name filtering unreliable. A high `min_score` threshold avoids mixing in a weak seat without needing submission IDs.
- Reactivity: do not require a shop-reactivity/diversity filter for the first BC corpus. First measure whether elite trajectories already show useful state-conditioned branching; later compare score-only filtering against optional reactivity/diversity filtering if BC appears dominated by rigid tapes.
- Revisit when: the manifest schema changes, per-seat submission/rating metadata becomes available, or experiments show that a different replay-selection rule produces a materially better BC initialization.

## D-018 — Preserve a Canonical Daily Replay Dataset Before Model-Specific Encoding

- Date: 2026-08-21; clarified 2026-08-22
- Status: active
- Decision: Parse each selected replay once into a canonical `(episode, seat, day)` record containing start-of-day state, end-of-day state, a compact daily event ledger, and complete replay/score provenance. Treat this as the stable preprocessing boundary; derive BC tensors and alternative action encodings from it rather than reparsing raw 720-turn JSON for every experiment.
- Day boundaries: use explicit replay `day`/`hour` fields. Canonicalize seats as `self`/`opponent` so both sides share one representation.
- Board state: retain the full 10x10 board and mechanically descriptive lifecycle information. For crops preserve age/raw growth state plus derived time-to-next-harvest/output, harvestability, fertilizer state, and water/dry/weed state as available. For animals preserve production cooldown/time-to-next-product, feed/starvation, care/bonus, and fertilizer-related state as available.
- Daily strategic labels: derive crop composition targets, animal-count targets, land state/expansion, fertilizer applications by crop type, CARE applications by animal type, and sell quantities for every product across six windows anchored at hours `0, 4, 8, 12, 16, 20` (`0-3`, `4-7`, `8-11`, `12-15`, `16-19`, `20-23`). Preserve exact sale hours and exact fertilizer/CARE event timing in the ledger even when the learned manager uses aggregated daily targets.
- Strategic ownership boundary: fertilizer and CARE species/type allocation belong to the manager; exact tile/animal selection remains an executor responsibility. The canonical ledger therefore preserves crop/animal identity and exact event location/timing so richer future adapters can be derived without reparsing raw replays.
- Coordinate contract: simulator worker positions are `[x, y]`, while canonical board/event locations use row-major board coordinates `[y, x]`. Any tile-dependent worker-event attribution (including DIG/FERTILIZE/HARVEST/CARE) must index `tiles[y][x]` after unpacking `x, y = pos`; asymmetric-coordinate regression tests are required so transposition bugs cannot silently poison labels.
- Retention rule: keep opponent public state in the canonical dataset even when V0 masks it, and retain compact aggregates for plant/dig/fertilize/CARE/harvest, animal/seed/product buys, land buys, hires/costs, and sales. The canonical dataset should remain mechanically descriptive rather than contain hard-coded strategic profitability scores.
- Physical storage: use compressed Parquet as the production canonical format; JSONL is optional inspection/debug output only. Raw Kaggle replays remain the source of truth and processed files may be regenerated rather than duplicated indefinitely.
- Compatibility rule: canonical schema versions are fail-loud semantic compatibility boundaries. If a correction or extension changes the meaning/availability of training-relevant fields, bump the schema version and regenerate from raw replays rather than silently fabricating/migrating labels.
- Rationale: after collapsing each 720-turn game into roughly 30 daily decisions, processed-data throughput/RAM is unlikely to be difficult. The expensive part is repeatedly parsing huge raw replays. A rich intermediate representation lets observation/action experiments change cheaply while preserving auditability.
- Future alternatives intentionally preserved: delta rather than absolute crop/animal targets; explicit per-tile replacement control; age-bucketed/tile-specific fertilizer/CARE; learned wheat/feed economics; learned harvesting; richer workload feedback; opponent-aware inputs; and finer 24-turn or separate reactive selling policies.
- Detailed contract and alternatives: `.agents/notes/implemented/2026-08-21-canonical-daily-replay-record.md`.
- Revisit when: manually inspected replay rows show the contract loses strategically necessary information, engine state semantics change, or a later learned subsystem requires detail that cannot be derived from the retained start/end states and event ledger.

## D-019 - Use a Configurable Tile Transformer as the First BC Manager

- Date: 2026-08-22
- Status: active
- Decision: Implement the first learned manager as a stateless once-per-day (day/hour0) configurable PyTorch Transformer over [MANAGER, 100 own tile tokens, 5 compact global tokens] with structured heads for crop/animal/land/fertilizer-by-crop/CARE-by-animal counts, sell presence, and log1p sell quantity. The shared spatial tile encoder consumes the current canonical lifecycle/presence fields; own private shed/seeds/inventory and shared market/town/labor/day context are encoded in the same forward context. Opponent PUBLIC board tokens are optional and off by default; opponent private data has no feature path. Group-balanced loss (per-group mean CE/BCE/masked SmoothL1) prevents the 54 sell cells from dominating. Default ~1.071M parameters with a tiny CPU validation config; width/depth/head count remain ordinary configuration values so validation can use tiny models and later BC/PPO models can scale without a redesign.
- Validation protocol: date-held-out splits only (default train 2026-08-17..20, validation 2026-08-21) with configurable elite cutoff (default min_score >= 2950); train-split-only day baseline reported beside model metrics; sparse nonzero diagnostics make zero collapse visible.
- Selling adapter: preserve raw submitted sell events canonically, but cap each individual event to 0..100 only in the BC target adapter before six-bin aggregation; repeated events in one bin may sum above 100. Predict sell presence plus presence-masked log1p quantity rather than regress sentinel-like million-unit submitted quantities directly.
- Compute scope: ordinary batched inference is the intended optimization path (including both seats as separate batch rows during future self-play). Do not contort the architecture around Orbit-Wars-style double-sided public-state compute reuse; private per-seat state makes that optimization less attractive and current model sizes are small.
- Explicitly deferred: temporal/RNN/value heads, opponent inference, self-play, executor/legal-order mapping, and special symmetric compute reuse. The deterministic executor later retains exact tile/animal/routing details.
- Rationale: after D-017/D-018 the missing piece was a first learnable policy over the canonical daily records. A standard small Transformer on a stable compact interface is the cheapest genuine learning step; scaling is a config change, not a redesign. Flattened MLPs discard board structure; temporal and value machinery add complexity before the stationary problem learns.
- CARE attribution remains a canonical-schema correction under D-018 (`targets.care_by_animal`); this decision consumes it as one head and does not re-decide it.
- Detailed contract and alternatives: `.agents/notes/implemented/2026-08-22-use-configurable-tile-transformer-for-initial-bc-manager.md`.
- Revisit when: closed-loop evaluation shows the daily abstraction or output parameterization loses strategically necessary control, the state-aware model fails to beat the train-only day baseline, or PPO refinement requires value/temporal extensions.

## D-020 - Use a Pinned Official Same-Action Differential Oracle Before Broad Parity Work

- Date: 2026-08-23
- Status: active
- Decision: All fast-engine correctness claims route through `oracle/`: the official engine is exactly `kaggle-environments==1.32.7` (wheel SHA256 `2a1bb862ad2d6463080f80f6a766f46d94b53fd57168cfeddb9857fc3dbc4c8f`, interpreter files byte-matching upstream commit `28b6d8af3ce73926b3d0fda1410c1ddd8384ab8c`) enforced by a provenance guard that refuses to run otherwise. Replay submits the exact same action pair to both engines each turn BEFORE an immediate canonical full-state compare and stops at the first divergent field with seed/step/day/hour/path/values/actions context. The official backend imports `kaggle_environments` lazily; the fast hot path never imports Kaggle/OpenSpiel (fresh-process tested). Validity requires the FULL per-step status history to stay within {ACTIVE, DONE}; terminal DONE never masks earlier anomalies.
- Canonical comparison is exact, not shallow: step/day/hour; both farms (money, full 10x10 board incl. crop AND animal lifecycle, farmer/hand positions, hires_today, unlocked quadrants); both seats' private shed/seeds/farmer+hand inventories; market inventory/prices plus exact `params` when present; town shops with duplicate multiplicity; rewards; statuses.
- Fast API contract corrections folded in: wire unit operation ids translate to the Rust core's internal op codes (`UNIT_OP_CODES`), and observation decoding inverts the FIXED `generated_protocol::SEASON_STEPS = 720`, never the configurable `episodeSteps`.
- Rationale: Stage 2b broad mechanic/full-episode parity is only trustworthy on top of a reusable turn-level oracle with exact provenance; untranslated wire ops and wrong decode scale silently corrupted fast-vs-official comparisons.
- Scope boundary: passing traces prove parity only for exercised actions. Broad probes, random corpora, full 720-turn episodes, closed-loop A/B, and benchmarks remain Stage 2b; no full-parity or training-safety claim follows from this decision.
- Revisit when: the official pin moves (new engine version), the canonical schema needs fields the official observation does not expose, or Stage 2b evidence demands richer first-divergence context.

## D-021 - Size the Fast Engine to the Exact Default-Contract Hand Capacity (MAX_HANDS=240)

- Date: 2026-08-23
- Status: active
- Decision: Replace the fixed 16-slot hand layout with the exact default-contract capacity `max_hands = turnsPerDay * maxMarketOrdersPerTurn = 24 * 10 = 240`, derived in the protocol generator from the pinned schema defaults rather than hard-coded. Rationale: official HIRE appends exactly one hand per atomic market order, the market queue truncates to 10 orders/turn, a day is 24 turns, and hands clear at every day reset, so 240 simultaneous hands is the exact supremum under the default configuration — no silent truncation of officially reachable states remains.
- Breaking wire layout (single coordinated revision): `OBS_SIZE` 5630 -> 8766, `ACTION_SLOTS` 27 -> 251 with market action rows moved from slot 17 to slot `MAX_HANDS + 1 = 241`, `MASK_SIZE` 3562 -> 34026. Only the two MAX_HANDS-scaled observation blocks move; all other blocks and reserved gaps keep their widths. The extension exports `MAX_HANDS` / `ACTION_SLOTS` / `MASK_SIZE`; preallocated `*_into` buffers reject stale shapes loudly. Observations or models serialized against the old layout are incompatible and must be regenerated.
- HIRE mask gate locked to the official formula: available iff `hand_count < MAX_HANDS AND money >= fib(hires_today)` (engine Fibonacci fib(0)=fib(1)=1); regression-proven open at reset and closed at 23 hands / money 24976 < fib(23)=46368.
- Scope: valid for the pinned default configuration only; non-default turnsPerDay/maxMarketOrdersPerTurn above defaults exceed the bound and stay unsupported (as does boardSize != 10). `bc_manager/constants.py::MAX_HANDS = 8` is a separate BC-manager head-slot constant and intentionally unchanged.
- Evidence boundary: parity proven for exercised traces only (`tests/test_oracle_hands.py`, 5 real-official replays incl. exactly-16, 16->17 crossing, 23 hands + subsequent hand actions, day-end reset from 23 hands + rehire); no full-parity or training-safety claim until full 720-turn episodes pass through the oracle.
- Revisit when: the official engine changes hiring/market-order/day mechanics, a non-default configuration must be supported, or oracle evidence shows an officially reachable state the bound excludes.

## D-022 - Drive Broad Parity With a Deterministic State-Aware Legal-ISH Generator, Not Independent Policies

- Date: 2026-08-23
- Status: active
- Decision: The full-episode parity corpus is driven by `oracle/action_generator.py::LegalishActionGenerator`: one fixed `random.Random(generator_seed)` stream reads ONLY the pre-transition fast-engine observation pair and emits exactly ONE action pair per turn, which `run_same_action_replay` submits to BOTH engines before any comparison. No policy ever runs independently after a divergence; any first divergence stays attributable and reproducible from `(generator_seed, turn_index)` alone because the generator is deterministic given seed + engine states.
- "Legal-ish" deliberately includes the official silent-noop/partial-fill surface: malformed market entries, unknown ops, missing/non-integer quantities, order bursts beyond 10 (truncation), extra hand slots, and unaffordable orders are part of the covered contract, never rejected by the generator.
- Coverage is measured, not assumed: every attempted family increments a histogram published in the corpus report (`research/parity_corpus_report.json`); families not naturally reached must be reached by generator bias/targeted prefixes, never by weakening same-action semantics.
- Primitive-turn accounting (locked): a default "720-step episode" = ONE reset observation + exactly 719 accepted primitive `step` calls; terminal DONE lands at canonical step 719 = day 29 hour 23; 29 day-boundary transitions.
- Evidence boundary: zero first divergence over the fixed 8-seed corpus proves parity only for states those episodes reach — bounded coverage, not universal mathematical proof. Training-safety claims must cite the exact corpus result.

## D-023 - Add an Independent Stateful-Agent A/B Gate After Same-Action Parity

- Date: 2026-08-23
- Status: active
- Decision: Keep same-action replay as the primary engine-correctness gate, and add `oracle.run_closed_loop` as a secondary policy-interface gate. It must construct fresh official/fast backends and fresh agent instances for both seats, compare each corresponding presented observation before independent action computation, compare actions before stepping, then compare canonical next state/rewards/statuses immediately. The official full status history remains subject to anomaly validation.
- Deterministic fixture: use the existing stateful `executor_v0.ExecutorAgent` with a fixed nontrivial `DailyPlan` through `make_deterministic_executor_factory`; no mutable agent state may be shared across backends or seats. The narrow fast observation adapter only reconciles wire aliases/sparse maps that the existing executor cannot consume directly; it does not synchronize decisions.
- Evidence: `tests/test_oracle_closed_loop.py` and `research/closed_loop_ab_report.json` cover seeds 0, 7, and 42 for complete reset + 719-step episodes with `DONE/DONE`, equal rewards, and zero divergence; one repo-local `best.pt` checkpoint episode also passed. Deliberate observation/action drift tests stop before stepping and retain full first-failure context.
- Scope boundary: this proves deterministic interface/transition plumbing for the exercised fixed-plan and checkpoint paths, not competitive score, BC quality, universal engine parity, executor redesign, BC tuning, PPO, throughput, or benchmarks.

## D-024 - Release the GIL Around Native Batch Calls and Add an Optional Instance-Local Rayon Pool

- Date: 2026-08-23
- Status: active
- Decision: The Rust batch backend (`rust/kaggriculture_env/src/lib.rs`) releases the GIL for the whole native transition/observation pass of `reset`, `step`, `step_transition`, `observe_into`, `action_masks_into`, and `step_into` via `py.allow_threads`, and gains an optional instance-local Rayon thread pool selected per environment. Rationale: with the GIL held, one Python rollout worker monopolizes the interpreter and concurrent Python threads (timers, other workers in one process) starve; a configurable per-instance pool lets several batch environments in one process pin disjoint worker counts instead of oversubscribing Rayon's global pool.
- API: `FastKaggricultureEnv(configuration={"numThreads": N})` forwards to `RustBatchEnv(..., num_threads=N)`. `None`/omitted keeps the historical global-pool default; `N >= 1` builds a private pool (`kagg-batch-*` threads); `0` or any value `< 1` raises `ValueError` in both the Python wrapper and the Rust constructor. `num_threads()` reports the configured count, `0` = global-pool default.
- Parallel fan-out threshold unchanged: batch work parallelizes only at `>= 128` environments (`PARALLEL_MIN_ENVS`); below that a serial loop runs for both pool modes, so small-batch behavior is untouched.
- Safety boundary: validation and all Python-object work stay under the GIL; each released call operates on owned Rust state or on raw slices of exclusively borrowed caller-owned NumPy buffers (`PyReadwriteArray` exclusive borrow + released GIL means no Python code can alias the buffer during native work; PyO3's borrow guard rejects conflicting same-backend access loudly). Owned output arrays are allocated under the GIL, filled without it, then wrapped zero-copy.
- Determinism evidence: byte-identical trajectories across worker counts 1/2/4/default over 130 envs x 30 steps including the day boundary; buffer integrity under concurrent Python-thread pressure; real Python-thread progress (~84M counter ticks) during a long 512-env native call proving GIL release (`tests/test_batch_throughput_seam.py`, 5 tests).
- Explicit non-claims: no throughput/speedup/scaling claim until measured benchmarks exist. Target topology for future measurement is the Kaggle TPU v5 host (~96 CPU vCPUs) using process-level parallelism times in-process batch environments with one pinned pool per process/batch. Fused executor/day-step batching and distributed rollout remain deferred.
- Revisit when: benchmarks show contention between pool modes, a multi-process rollout launcher needs pool sizing defaults, or the official engine changes batch-call semantics.

## D-025 - Benchmark Issue #2 Honestly and Defer Observation-Writer Optimization to a Distinct Stage

- Date: 2026-08-23
- Status: active
- Decision: Issue #2 performance evidence comes from `scripts/benchmark_engine_throughput.py`: deterministic scripted traces (no policy), warmup-discarded median/min/max over repeated runs, batch throughput counted as env-transitions (N*steps), the diffmap reference built unmodified at `ef8bb3a` in an isolated temp venv and labeled 1.32.6 performance-only, and every derived rate validated (no NaN, no impossible numbers) before the report renders. Raw results are checked in at `docs/benchmarks/issue2_results.json`; the generated report is `docs/benchmarks/ISSUE2_THROUGHPUT.md`.
- Measured headline (i7-12700H laptop, Windows 11): scalar dict API 4.7x vs official 1.32.7 per full 720-step episode; native floor 341x vs official; default-pool multi-core scaling 2.87-2.89x at N>=512 vs 1 thread; serial below N=128 by design.
- Optimization decision: observation writing is 84% of steady large-batch step cost and the reference core (old 16-hand layout) is ~2.7x faster per transition than our MAX_HANDS=240 writer; this is the single bounded optimization candidate. It is explicitly deferred to a distinct correction stage - no semantic engine change happens in the benchmark stage, and no further tuning is warranted now because the engine already beats the official interpreter decisively at both scalar and batch levels.
- Non-claims: laptop numbers do not transfer to Kaggle TPU hosts; reference ratios are performance context, not parity statements; RSS deltas are allocator/pool upper bounds, never GameState sizes.
- Revisit when: the rollout topology changes, a fused day-step/observation-writer correction stage is scheduled, or the official engine changes the observation contract.

## D-026 - Gate BC V1 Promotion on the Fixed Paired Closed-Loop Panel, Never on Teacher-Forced or Coherence Metrics Alone

- Date: 2026-08-23
- Status: active
- Decision: No V0/J/E/JE variant (issue #6) may be promoted, declared a winner, or used to replace the production manager based on teacher-forced validation totals, coherence diagnostics, or any offline proxy alone. Promotion requires the fixed paired closed-loop panel run by `python -m bc_manager.ablation`: committed `standard_mixed` opening days 0-3 -> the tested BC checkpoint -> the unchanged `executor_v0.ExecutorAgent`, PASS responder opponent, seeds 7/17/42/123/2026 x seats {0,1} = 40 games behind the pinned official 1.32.7 provenance guard, ranked by closed-loop final-bank median then mean. Seed-17 collapse flags and seed-2026 retention vs V0 are reported beside raw banks; teacher-forced `validation_metrics.total` and coherence aggregates are prerequisites/diagnostics only.
- Rationale: teacher-forced metrics measure imitation of elite daily plans under teacher state distribution, not closed-loop bank outcomes; coherence diagnostics are lower-bound feasibility signals recorded but never enforced. Only the paired panel measures what promotion actually changes, and its fixed seeds/seats/opponent make variant comparisons attributable rather than anecdotal.
- Scope boundary: the PASS responder makes this a controlled ablation comparison between variants, not a competitive-strength measurement; competitive claims still require the frozen-opponent panels of the evaluation rules.
- Revisit when: the panel has produced real results and a successor gate (e.g. frozen competent-opponent panel or league entry) is defined, or the engine pin moves.

## D-027 - Port Only the Promoted BC-E Variant to JAX; Keep J/JE PyTorch-Only

- Date: 2026-08-24
- Status: active
- Decision: `bc_manager_jax` supports exactly model variants V0 and E. E is the four-way closed-loop ablation winner (median bank 25,873.0 vs V0 9,251.5), so it is the only new architecture/input ported (issue #8). The JAX variant lives OUTSIDE the frozen seven-field `ManagerConfig`, mirroring the torch checkpoint's top-level `model_variant` (absent -> V0) and the native NPZ's top-level metadata record; expected variants are checked strictly against stored metadata, never inferred from weight shapes; old V0 torch/native checkpoints keep loading unchanged. J/JE fail loudly as unsupported (`not supported by bc_manager_jax`) rather than surfacing as key/shape errors. E adds exactly the audited 14-channel `economic_context` float32 [B, 14] — built only by authoritative `bc_manager.economics` functions — concatenated after the six self-resource feature blocks before the same two-layer MLP; trunk/tokens/heads/loss/decode are untouched.
- Evidence boundary: local validation is tiny CPU-only — deterministic PyTorch E -> JAX E parity (worst max 6.855e-07 / mean 1.101e-07, all seven output groups), exact decode equality, loss groups within 9.5e-7, single-device JIT step, N=4 logical-CPU NamedSharding subprocess plus one bounded N=8 logical smoke. N=4/N=8 are forced host-CPU logical validation of sharding plumbing ONLY, never throughput/scaling evidence. The real BC-E checkpoint was absent locally; no real-checkpoint or TPU claim is made.
- Rationale: the JAX mirror exists to become the RL policy/value trunk on TPU hardware; porting the promoted variant now avoids converting a soon-dead V0 production path, while keeping the joint-decoder idea parked in PyTorch avoids freezing an unproven RL experiment into infrastructure.
- Revisit when: a later RL design adopts joint decision tokens (port J/JE then, behind the same variant seam), the real BC-E checkpoint becomes available for bounded parity, or Kaggle TPU capacity enables the documented benchmark.

## D-028 - Build RL Self-Play Around Manager-Day Trajectories, Identity-Grouped Batched Policy Calls, and an Injectable Executor Factory

- Date: 2026-08-24
- Status: active
- Decision: The RL harness (`rl_manager`, issue #9 Stage A) stores ONE trajectory row per manager decision/day/seat (d4..d29) — not per primitive turn — with the exact encoded action tensors of the issue-#8 decode plus PPO-ready logprob-group/logprob-total/value slots reserved from day one. All day-boundary manager requests across all lockstep environments are grouped by immutable policy identity and answered with exactly ONE batched policy call per (identity, day); there is never a per-environment accelerator call or a Python loop over examples inside JAX. Executors enter only through an injectable factory building fresh unmodified `executor_v0` agents fed by queued plans; the executor stays outside the RL gradient, and issue #7 can swap the factory without RL-semantics change.
- Semantics locked: days 0-3 belong solely to the committed `standard_mixed` opening; the first manager decision is d4h0; E encoding uses the authoritative `encode_live_inputs` stateless `economic_prev_start` path fed from runner-owned per-seat daily-start `(day, cash)` state (reset per episode/seat, never inferred from submitted orders); realized labor comes from observed `hires_today`; farm canonicalization uses only the public `EngineBackend.canonical_state()` seam. Rewards are terminal-only (+1/0/-1 on the final manager row). Trajectory serialization is strict-schema NPZ (schema version, count bounds, shapes/dtypes validated, `allow_pickle=False`) plus a JSON-safe sidecar; model-facing input arrays are pinned to the canonical E encoder spec derived by calling the encoder itself, never re-hardcoded.
- Evidence boundary: local validation is tiny-random-E plumbing ONLY — 47 passed + 1 skipped across 7 test files; two deterministic complete fast games (numThreads=1) with 52 transitions each, byte-equal rerun, and a [0,0] bankruptcy tie prove harness determinism, not policy quality. No PPO exists yet; no real-checkpoint rollout has run (`artifacts/local/bc-v1-E/best.pt` absent); the official-engine parity path is gated on the absent `kaggle_environments` dependency; the eventual many-CPU-worker topology is design only and UNMEASURED.
- Revisit when: Stage B (PPO updates) begins against this schema, issue #7 selects the executor (factory swap), or the real BC-E checkpoint enables real-policy rollouts.

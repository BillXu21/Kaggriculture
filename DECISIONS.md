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

## D-030: Executor V0.5 mechanical rules pinned from exact engine semantics (issue #7, 2026-08-24)

- Watering is classed by exact lifecycle mechanics: MUST only at the weed boundary (`consecutive_unwatered >= 1`, planting day included); YIELD for single-harvest window watering and fertilized ongoing production-eve; everything else deferred. Blanket daily watering is retired as a labor sink.
- Layouts anchor to the persistent shed hub `(4,4)` (canonical `[y,x]`), never the moving farmer; animal planning claims tiles before crop reconciliation via `plan_day_layouts`; WEED tiles are reclaimable slot pools behind DIG prerequisites.
- Movement legality is bounds-only: engine ops 1-4 move unconditionally, so locked quadrants/tiles never block walking.
- Market queue uses sequential within-turn cash: queued SELL revenue funds later HIRE/BUY candidates; every BUY is gated by exact whole-order cost; unaffordable skips are logged, not resubmitted blindly.
- Hiring is any-hour, workload-derived (crude per-worker divisor plus a maintenance-travel floor), with no arbitrary daily cap; hands are daily per engine reset semantics.
- CARE/FERTILIZE feasibility counts plan-implied same-day assets (max of current and requested targets).
- One-day replay-slice evaluation with boundary parity verification is the accepted inner-loop methodology; paired wealth (cash + inventories at market prices + assets at cost) is the primary comparison; unfinished-task count stays diagnostic-only.

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

## D-029 - Keep PPO V0 a Stationary-Opponent Smoke: Adapter Seam, Frozen Sell Quantities, and Guarded CLIs

- Date: 2026-08-24
- Status: active
- Decision: Issue #9 Stage B stays plumbing-only until issue #7 freezes the executor. The PPO policy reaches the Stage-A runner exclusively through `PPOBatchedPolicy` (`rl_manager/ppo_adapter.py`), which implements the framework-neutral `BatchedPlanPolicy` protocol: contiguous own-only E arrays plus an explicit string `prng_id` hashed to one root JAX key whose per-row decision seeds attach to row identity (never batch position); stochastic sampling for training, exact frozen-E argmax decode for eval; identity = name/version/params-fingerprint. Sell quantities always derive from the immutable frozen-E snapshot; the local smoke selects a deterministic evenly-spaced 2-8 row subset only AFTER GAE + advantage normalization over the complete candidate trajectory, and runs exactly ONE update call on it. Diagnostics artifacts are strictly JSON-safe (`allow_nan=False`) with null+reason for every genuinely missing quantity - never fabricated numbers. Train/eval CLIs require an existing real BC-E checkpoint and an explicit executor factory selection, default all worker/env/thread knobs to safe 1, expose exactly the fixed smoke(17,42,2026)/dev(200..263)/holdout(5000..5031) seed sets in both seat orientations, print the planned game count, and gate dev/holdout behind `--confirm-expensive`; tests never execute them.
- Evidence boundary: tiny live smoke uses ONE complete fast-engine game (tiny random-init E, numThreads=1), full-trajectory GAE, bit-equal stored-action logprob recompute before the update, ONE 4-row-minibatch update with finite metrics, bit-identical checkpoint resume, and pre/post deterministic eval equality. Combined sweep 130 passed + 4 skipped (~187 s). No policy-quality claim; official-engine parity and real-checkpoint paths remain gated skips.
- Rationale: correctness of batched RL plumbing is provable locally against a stationary opponent without optimizing quality against an executor that issue #7 may replace; explicit fail-loud guards prevent accidental expensive panels or implicit executor choices.
- Revisit when: issue #7 selects the executor (factory swap + first serious run), the real BC-E checkpoint lands at `artifacts/local/bc-v1-E/best.pt`, or multi-process rollout workers are actually implemented.

## D-037 — Reject R4 and Retain Prior-Debt Suppression as Bounded Heuristic Debt

- Date: 2026-08-25
- Status: active
- Decision: Reject/revert R4; retain the prior-debt suppression default **ON** in the frozen V0.7 executor. Keep the accepted survival WHEAT shed-room clamp at official/default `shed_capacity=100`, with survival work still ordered before hiring.
- Rationale: R4 failed the real BC-E fixed-plan 7d regression. The ON setting is a bounded survivability heuristic, not learned strategy and not a universal acceptance claim.
- Evidence: final no-R4 PASS panel, seeds 17/42/2026 x seats 0/1: ON banks `[17005,14961,23346,26587,56742,65959]`, mean `34100`, median `24966.5`, minimum `14961`; OFF banks `[265,265,30,33,0,0]`, mean `98.8`, median `31.5`, minimum `0`. Loss units are ON starvation `0` / overflow `12`, OFF starvation `38` / overflow `12`. The setting is bounded to this panel and remains architectural debt.
- Revisit when: the next RL executor-factory swap and serious training decision supplies broader evidence; do not generalize beyond this three-seed, two-seat PASS panel.

## D-038 — Validate the Exact Submission Archive Through Kaggle's Raw-Code Loader

- Date: 2026-08-27
- Status: active
- Decision: A Kaggriculture submission is not considered validated until the **exact built archive** is freshly extracted into an empty directory, isolated from the repository root, loaded through `kaggle_environments.agent.get_last_callable`, and run on the pinned official engine with strict/debug executor behavior and full status-history validation. The archive must carry the complete runtime import closure, including lazy dependencies such as `fast_env`.
- Rationale: the BC-E V0.7 submission omitted `fast_env`, while `executor_v0/agent.py::_buy_order_cost` lazily imported `fast_env.market`. Production `strict=False` swallowed the resulting `ModuleNotFoundError` and returned legal all-PASS actions repeatedly, producing apparent multi-day AFK behavior without useful Kaggle stderr/stdout. Source-worktree evaluation therefore did not prove submission equivalence.
- Current pinned post-Stage-4 reference: official `kaggle-environments==1.32.7`, seed 7, candidate seat 1 versus PASS, bank `47,290`, action fingerprint `a38bf47884e5e6e89c2d77f7aab07819f3559e898af40372942460693c8b6afc`, archive SHA-256 `c12218ac1010c894ed22fd065049a290d03555c9f44ad0d6cc667fa52ee13de2`, 719 candidate actions, zero status anomalies. The earlier pre-behavior reference at `54,439` remains historical evidence rather than the current compatibility target.
- Revisit when: the submission entrypoint/runtime import closure changes, the official engine pin moves, or the archive verifier itself changes semantics.

## D-039 — Keep Same-Tile Yield WATER Before HARVEST for Non-Ongoing Crops

- Date: 2026-08-27
- Status: active
- Decision: For non-ongoing crops, a same-tile yield-positive `WATER` is a mechanical dependency of `HARVEST`; another worker must not harvest first. The dependency is waived on the final actionable turn. This applies to WHEAT, CARROT, and MELON. Do not add a multi-day rule such as "always wait until max-yield day"; ongoing TOMATO/STRAWBERRY behavior is unchanged.
- Rationale: the executor could legally harvest at the first harvestable age while a same-day WATER would still increase eventual yield. The manager does not express primitive same-turn sequencing, so this is executor-owned mechanics rather than a strategic crop-timing heuristic.
- Evidence: fixed 12-seed x 2-seat PASS panel after the change: mean `63,592.3` versus actual pre-change `60,778.1` (+`2,814.2`), median `65,509.5` versus `60,956` (+`4,553.5`), min `47,290`, max `74,151`, with zero `<1k`/`<10k`, runtime/status errors, unaffordable orders, or animal losses. Six of 24 paired games declined, confirming economic consequences remain state-dependent even when sequencing is mechanically better.
- Revisit when: engine crop lifecycle semantics change, or broader evidence proves the dependency itself is mechanically incorrect; do not revert merely because individual economic trajectories move both directions.

## D-040 — Reject Blanket Fertilizer Retention; Treat Inventory Release as Strategy

- Date: 2026-08-27
- Status: active
- Decision: Do not add a blanket executor rule that retains FERTILIZER under the aggressive-selling reference. The diagnostic/reference aggressive-sell behavior continues to sell FERTILIZER; whether and when fertilizer should be retained or sold belongs to learned strategic control rather than a fixed executor reserve heuristic.
- Rationale: retaining fertilizer sounded locally sensible because it is an intermediate production input, but the isolated experiment destroyed liquidity: all 24 panel banks fell to zero, with `27,151` unaffordable orders, `1,728` feed-shortage/starvation turns, and `120` animals lost. Repairing this inside the executor would require a conditional cash governor/reserve-release policy, which crosses the D-011 strategy boundary.
- Scope: aggressive sell-all remains a diagnostic/reference policy, not the intended final selling strategy. The failed retention experiment is preserved in history rather than silently erased.
- Revisit when: the learned manager/selling policy controls state-dependent fertilizer retention, or new mechanics prove a mechanically mandatory reserve independent of economic preference.

## D-041 — Do Not Patch CARE/FERT Execution Without Accepted-Action Attribution

- Date: 2026-08-27
- Status: active design constraint
- Decision: Do not add CARE/FERTILIZE executor heuristics from current requested-versus-observed telemetry alone. First add accepted-action/completion observability capable of distinguishing manager infeasibility, inventory/buy failure, labor/dispatch shortfall, and successful actions whose effects are not represented by the current state metric.
- Evidence: pre-behavior audit counted CARE/COW requested `2,848`, assigned `1,749`, observed-state `0`; CARE/SHEEP requested `1,731`, assigned `1,158`, observed-state `0`; FERTILIZE/STRAWBERRY requested `1,470`, assigned `731`, observed-state `3,020`. There were `1,162` requested-positive/observed-zero rows, only `23` proven manager-infeasible cases, and `1,189` unresolved cases. The "observed" metric is state-derived, not an accepted-action ledger, so most shortfalls cannot be safely attributed.
- Rationale: patching an executor from ambiguous telemetry risks converting an observability gap into a strategic or mechanical heuristic with no proven causal defect.
- Revisit when: accepted action/completion telemetry is available and demonstrates a concrete executor-caused shortfall.

## D-042 — Idle Cleanup May Replace PASS Only; Reject Shared-Pool Optional Watering as a Valid Test

- Date: 2026-08-27
- Status: active design constraint; corrected experiment pending
- Decision: Optional idle cleanup must be strictly subordinate to normal dispatch: run the normal foreman first, preserve every worker's non-PASS normal action exactly, and allow cleanup to replace only a worker action that would otherwise be literal `PASS`. Cleanup must not contribute to hiring/workload, prior-day debt, manager completion, shortage buys, market orders, or expansion suppression, and is recomputed from scratch every primitive turn. Candidate cleanup order is weed DIG before safe optional WATER; promotion remains pending a corrected A/B/C panel.
- Rationale: the previous `optional_spare_watering` implementation appended OPTIONAL WATER tasks into the same foreman pool as normal work. Because foreman performs underfoot execution before global assignment, a worker standing on an optional-water tile could WATER instead of beginning higher-priority work elsewhere. Therefore the previous ON result did not isolate the intended hypothesis of consuming wasted labor.
- Evidence: official 12-seed x 2-seat A/B with aggressive sell-all and prior-debt suppression ON reproduced OFF mean `63,592.3`, median `65,509.5`, min `47,290`, max `74,151`; old optional watering ON scored mean `60,948.5`, median `61,114`, min `46,798`, max `69,673`, with 8 wins / 16 losses, mean paired delta `-2,643.8`, worst `-18,675`, best `+21,429`. The wide paired spread justifies investigating dispatch correctness rather than promoting or permanently rejecting the underlying cleanup idea.
- Day-boundary note: worker positions reset near the central shed at day end, hired hands disappear, and carried inventory auto-drops, so cleanup does not need an end-of-day return-home rule. Its opportunity cost is only within the current day's remaining turns, and normal dispatch must preempt it on the next primitive turn.
- Revisit when: the true PASS-only WATER and WEED-first+WATER panels complete. Promote only if normal non-PASS actions are proven unchanged and paired outcome/tail evidence is favorable.

## D-043 — Keep One JAX Owner and Spawn CPU Rollout Workers

- Date: 2026-08-27
- Status: active
- Decision: The parallel rollout coordinator keeps policy objects and all JAX/libtpu initialization in the parent process. It starts `spawn` workers that receive serializable episode/policy identities, build independent engine/opening/executor state, and exchange only manager-day encoded NumPy rows through bounded queues. Requests are canonically sorted by policy identity/day/episode/seat; trajectory shards and results are normalized before return.
- Rationale: the scalar rollout bottleneck is CPU executor/environment work, while multiple JAX-owning processes have caused TPU initialization conflicts. A simple local process boundary provides CPU overlap without changing executor or native-engine semantics.
- Guarantees: worker startup fails if accelerator modules are loaded; worker exceptions, dead exits, missing episodes, duplicate episodes, and duplicate trajectory rows fail loudly; the default executor factory is resolved inside each child rather than pickled from its nested implementation class. Stable row identifiers are available to future stochastic batched policies.
- Evidence boundary: deterministic truncated fast-engine smoke passed with two spawned workers and two environments per worker, including central request batching and trajectory normalization. TPU throughput and full real-checkpoint scaling remain unmeasured; use `research/RL_MANAGER_PARALLEL_ROLLOUTS.md`.
- Revisit when: issue #15 changes the executor factory contract, issue #16 supplies a native batched backend, or measured host profiling justifies a different transport.

# Kaggriculture Current State

Last updated: 2026-08-25

## Active Direction

The project is returning to the original learning-first boundary after a useful executor-debugging detour:

**freeze a mechanically credible two-layer stack -> inspect a few real BC-E games visually -> fix only obvious reproducible executor bugs -> audit manager action expressiveness -> run closed-loop self-play/PPO.**

The intended architecture remains:

`standard_mixed opening d0-d3 -> learned daily manager -> deterministic executor`

A learned tactical/middle controller may eventually sit between manager and router, but it is explicitly deferred until the current two-layer stack has been tested in self-play. Primitive-action RL is also deferred.

## Authoritative Main / Active Branches

- `main`: `3726e373c65b8221c4062138174898f6cf756119`.
- Issue #11 viewer branch: `issue-11-replay-debug-viewer`, validated tip `0f72bcd28ef20703718a8a16503b6776c4d4b046`; pushed, not merged.
- Issue #7 executor branch: `executor-v07-fixed-plan`, committed base work at `ed66981` plus local uncommitted R4/trace work under review; local-only until the V0.7 finish pass completes.
- Fast engine source/wheel provenance remains the V0.6-era pinned build (`fast-wheel-r3`, source `a6796a7ae18a13716e3c4f9498796c66971bc803`) unless the active branch records a successor.

## Real BC-E Checkpoint

The promoted E checkpoint is now available locally at:

`C:\Users\liuyi\VSCodeProjecs\Kaggriculture\Kaggriculture\artifacts\local\bc-v1-E\best.pt`

It is an external/local artifact and must remain ignored/uncommitted.

Checkpoint metadata carried from the Kaggle training run:

- model variant: E-own;
- epoch: 27;
- validation total: `2.910865758929336`.

This removes the previous blocker on real-policy executor validation, seed-17 diagnosis, real BC-E debug traces, and the first serious self-play baseline.

## Executor V0.6 Evidence

V0.6 added survival-oriented mechanics and corrected unfinished-work accounting:

- protect WHEAT needed by currently unfed animals from discretionary sells;
- starvation-boundary FEED preemption;
- survival WHEAT purchase before discretionary hiring/buys;
- partial affordable WHEAT buys;
- EOD work debt separated from temporary travel/waiting;
- pending-task diagnostics include claimed tasks still traveling/picking up/blocked.

Bounded paired evidence across 24 historical/known-collapse trajectories:

- mean bank delta V0.5 -> V0.6: `+18,362.7`;
- median bank delta: `+17,643.5`;
- better: `18 / 24`;
- animal drops: `116 -> 10`.

All remaining drops in that sample were seed 17: 5 on each seat.

Important caveat: V0.6 also suppresses expansion after **any** previous-day work debt. Diagnostics showed suppression active roughly 25-26 of the 26 manager-controlled days in representative games, often driven mostly by manager debt. This is a strategic-boundary violation candidate, not a trusted permanent invariant. The V0.7 finish pass must ablate the broad prior-debt veto against real BC-E rather than silently carrying it into RL.

## Multi-Day Executor Harness / R4

Issue #7 now has a reusable fixed-plan multi-day A/B harness with strict DailyPlan tapes and 3/5/7-day comparisons. Earlier expert-intent tapes are useful executor evidence but are **not BC-E evidence**; real BC-E-derived tapes are the next validation source.

Current narrow watering candidate R4 reserves an exact `water_must_weed_boundary` task for a worker already standing on that tile, preventing an earlier distant worker from stealing it. Preliminary expert-intent evidence:

- 3-day: survival debt `1 -> 0`, crops destroyed `1 -> 0`, movement `256 -> 228`, wealth `+173`;
- 5-day: survival debt `16 -> 4`, crops destroyed `18 -> 7`, movement `737 -> 721`, bank `+2,771`, wealth `+1,629`;
- 7-day: survival debt `101 -> 99`, crops destroyed `96 -> 93`, weeds `100 -> 95`, movement `671 -> 647`, bank `+540`, wealth `+185`, animal loss unchanged `2/2`.

Caveat: day 16 regressed locally; R4 is not promoted yet. The active finish pass must either promote it unchanged, make one narrow correctness repair, or reject it. No R5/R6 heuristic search.

## Debug Replay Viewer

Issue #11 built a canonical deterministic debug trace + local viewer with:

- board playback and controls;
- worker positions, inventories, and trails;
- crop watering/drought state;
- animal feed/starvation state;
- manager requested/feasible plans;
- generated tasks, assignments, targets, blocked work;
- market submitted/unaffordable/skipped orders;
- WHEAT reserve/shortage and EOD debt overlays.

Independent Codex validation classified it **READY TO MERGE LATER**. Exact base-vs-viewer-branch primitive actions, rewards/statuses, manager transitions, and final results matched under deterministic official-engine checks; trace-on/off also matched. The instrumentation is therefore treated as passive, but it should be integrated only after V0.7 behavior is finalized so `_act` conflicts are resolved deliberately.

## Near-Term Sequence

1. Finish **local-only V0.7 validation** with the real BC-E checkpoint:
   - review R4 and the day-16 regression;
   - record real BC-E 3/5/7-day plan-tape evidence;
   - diagnose seed 17;
   - ablate current broad prior-debt veto vs no prior-debt veto;
   - freeze the executor; do not start another heuristic search.
2. Integrate the validated issue #11 passive viewer instrumentation into the frozen V0.7 branch and prove trace-on/off action parity again.
3. Generate a tiny real BC-E trace set and visually inspect obvious mechanical failures only. Limit this to one or two bounded correction passes.
4. Audit the learned manager contract before serious PPO, especially whether crop/animal targets can **decrease**. Strategic contraction/abandonment must be representable by the learned policy rather than hidden in executor heuristics.
5. Run a closed-loop BC-E/self-play baseline with the executor frozen.
6. Run the existing PPO smoke/small development path. Revisit reward shaping, a tactical middle model, or richer action frequency only when self-play evidence requires it.

## Strategic Boundary

Executor responsibilities:

- exact legality/mechanics;
- pathing, assignment, prerequisites, bookkeeping;
- minimum-safe watering for crops the current strategy maintains;
- feeding existing animals and avoiding mechanically preventable escape;
- mechanically implied purchases and exact market/cash sequencing;
- faithful execution and diagnostics.

Manager/RL responsibilities:

- farm size and expansion pace;
- crop/animal mix;
- intentional contraction or crop abandonment;
- liquidity/cash reserve/recovery behavior;
- deciding whether an asset is worth maintaining;
- economic reaction to opponent and market regimes.

Diagnostic rule going forward:

> First ask whether the executor failed to execute a feasible strategy, or the manager chose an unsustainable strategy. Only the first should normally produce an executor patch.

## Explicitly Deferred

- alternative corner/serpentine crop-layout optimization;
- copying additional Top-1 replay tricks into deterministic heuristics;
- tactical/middle-layer RL;
- raw primitive movement RL;
- large executor parameter searches;
- broad 64+64 panels after small mechanical edits;
- serious PPO/self-play before V0.7 freeze and manager-contract audit.

## Known Non-Executor Infrastructure

- JAX E port and PPO/self-play plumbing already exist and were validated with tiny/random policy plumbing; no serious policy-quality claim yet.
- Canonical observation boundary remains mandatory for fast-engine rollouts; raw fast aliases such as `age` must never reach manager/executor/model code.
- Official 1.32.7 remains the semantic reference; fast engine remains the high-volume path after parity/provenance checks.

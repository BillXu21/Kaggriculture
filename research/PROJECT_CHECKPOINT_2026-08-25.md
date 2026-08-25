# Project Checkpoint — 2026-08-25

This note captures the executor/viewer/RL handoff before Executor V0.7 is finalized. It is intentionally a checkpoint rather than a final issue-#7 report.

## Why the project is re-centering

Recent executor work was useful, but visual inspection of elite games made it clear that there is a tactical layer between daily strategy and primitive movement. Examples include spatially coherent planting, deliberate crop abandonment after price collapse, and highly efficient minimum watering.

The key architectural conclusion is **not** to hand-code all elite tactics into the executor. The project remains learning-first:

`daily learned manager -> deterministic executor`

A learned middle/tactical model may later select lower-level coordinate tasks, but it is deferred until the current two-layer stack has been tested in self-play. Primitive movement RL is further deferred.

## Executor V0.6 bounded result

V0.6 survival changes substantially improved the catastrophic tail:

- paired mean bank delta across 24 historical/known-collapse trajectories: `+18,362.7`;
- paired median bank delta: `+17,643.5`;
- improved trajectories: `18/24`;
- animal drops: `116 -> 10`.

All remaining drops in that sample were seed 17, five per seat.

However, V0.6 also introduced a broad rule suppressing expansion after **any** previous-day EOD work debt. Representative trajectories showed suppression on almost every BC-managed day, often with manager debt dominating EOD debt. This means some of the spectacular bank improvement may come from the executor acting as a conservative second manager rather than simply executing BC-E better.

Two narrower substitutes were tested in issue #7 and rejected on fixed-plan safety evidence:

- current feed-only survival gate;
- current must-water expansion gate.

The broad veto therefore remains unresolved and must be ablated directly against no prior-debt veto under the real BC-E checkpoint.

## Work-debt interpretation

The authoritative metric is actual end-of-day unresolved tile work, not every temporary travel/waiting turn.

Current categories:

- survival;
- maintenance;
- productive;
- manager.

Raw manager debt is diagnostic and may scale with farm ambition. It must not automatically trigger strategic executor overrides. Survival/critical maintenance debt is much more relevant to mechanical correctness.

## Issue #11 — Debug replay viewer

Branch:

`issue-11-replay-debug-viewer`

Validated tip:

`0f72bcd28ef20703718a8a16503b6776c4d4b046`

The branch adds a canonical deterministic trace exporter and local viewer with board playback, worker inventories/trails, lifecycle state, manager plans, tasks/assignments, market decisions, feed state, and EOD debt.

Independent Codex validation recommendation: **READY TO MERGE LATER**.

Important validation evidence:

- exact base-vs-branch primitive action equality under deterministic official-engine checks;
- trace-enabled vs trace-disabled equality;
- canonical coordinate and inventory alignment validated;
- deterministic serialization validated;
- local server did not expose arbitrary repository files;
- instrumentation is post-decision/passive.

The branch should remain separate until V0.7 behavior is finalized because both issue #7 and #11 touch `executor_v0/agent.py`.

## Issue #7 — Multi-day V0.7 work

Branch:

`executor-v07-fixed-plan`

Current committed harness tip when this checkpoint was written:

`ed66981`

The branch adds a reusable multi-day fixed-plan A/B harness with:

- strict DailyPlan tapes;
- BC-E plan recorder;
- checkpoint-free expert-intent tape builder;
- deterministic 3/5/7-day comparisons;
- opt-in turn tracing.

Earlier expert-intent artifacts are valid executor-isolation evidence but must not be called BC-E evidence.

### R4 watering candidate

R4 narrowly reserves an exact `water_must_weed_boundary` task for a worker already standing on the tile, preventing an earlier distant worker from stealing it.

Preliminary expert-intent results:

| window | main effects |
| --- | --- |
| 3-day | survival debt `1->0`, crops destroyed `1->0`, movement `256->228`, wealth `+173` |
| 5-day | survival debt `16->4`, crops destroyed `18->7`, movement `737->721`, bank `+2771`, wealth `+1629` |
| 7-day | survival debt `101->99`, crops destroyed `96->93`, weeds `100->95`, movement `671->647`, bank `+540`, wealth `+185`, animal loss unchanged `2/2` |

Caveat: day 16 locally regressed, so R4 is not yet promoted.

The bounded finish rule is now: **promote R4 unchanged, make one narrow correctness repair, or reject it. No R5/R6/R7 heuristic loop.**

## Real BC-E checkpoint now available

Local path:

`C:\Users\liuyi\VSCodeProjecs\Kaggriculture\Kaggriculture\artifacts\local\bc-v1-E\best.pt`

Metadata:

- E-own;
- epoch 27;
- validation total `2.910865758929336`.

The checkpoint is ignored/local and must not be committed.

This removes the biggest evidence gap. It should now be used for:

- real BC-E 3/5/7-day plan tapes;
- seed-17 diagnosis;
- current-vs-no-prior-debt-veto ablation;
- real debug viewer traces;
- the first closed-loop BC-E/self-play baseline.

## Immediate freeze plan

The active Codex packet is intentionally bounded:

1. review R4 and day-16 causality;
2. record small real BC-E multi-day evidence;
3. diagnose seed 17;
4. ablate broad prior-debt veto vs no prior-debt veto;
5. define/freeze V0.7;
6. integrate issue #11 passive tracing around the finalized behavior;
7. prove trace-on/off parity again;
8. generate a few local real BC-E traces;
9. stop.

No PPO, self-play training, layout optimization, tactical model, primitive RL, or additional heuristic search belongs in that packet.

## Visual inspection phase after V0.7

Use the integrated viewer on a very small set:

- seed 17 seat 0;
- seed 17 seat 1;
- seed 42 seat 0;
- one healthy/high-bank trajectory.

Patch only obvious reproducible mechanical failures such as resource-present-but-not-delivered feed, task stealing, clearly wasteful pickup churn, or deadline work missed despite nearby capacity.

Do **not** encode strategic elite observations such as price-driven crop abandonment or product-specific farm choices into deterministic code.

Limit this phase to one or two correction passes.

## Manager contract audit before serious PPO

The most important known strategic-interface concern is contraction.

Elite replay inspection showed intentional abandonment of strawberries after their price was crashed. If the manager/projection contract prevents crop or animal targets from falling below current counts, then RL cannot express contraction and the executor will be pressured to hide the limitation.

Before serious PPO:

- audit whether crop targets can decrease;
- audit whether animal targets can decrease in a mechanically meaningful way;
- inspect all manager outputs for additive/monotone constraints that block useful strategy;
- keep asset-value/abandonment decisions in the manager, not executor heuristics.

## Planned return to RL

Once V0.7 and the manager contract are credible:

1. run a frozen real BC-E closed-loop/self-play baseline;
2. run the existing PPO plumbing smoke with the real checkpoint;
3. run a small candidate-vs-frozen-E development experiment;
4. inspect complete trajectories before changing reward/action architecture;
5. add shaping, richer decision frequency, or a tactical middle layer only when evidence identifies the missing capability.

## Durable diagnostic rule

> First ask whether the executor failed to execute a feasible strategy, or the manager chose an unsustainable strategy.

Only the first normally justifies an executor patch.

# Executor V0.6 — survival guardrails and real work debt

Date: 2026-08-24

Status: implemented on `main`; bounded fast-engine re-evaluation still required.

## Why this pass exists

The first broad fast-engine evaluation of E-own + Executor V0.5 exposed a
clear collapse tail despite zero fallback errors and zero opening divergence.
Across the combined PASS/self-play sample, final bank fell sharply as animal
losses increased. The worst bankrupt trajectories also spent long periods with
active purchases that were unaffordable. This is consistent with two distinct
problems:

1. the BC manager can enter cash-pressure states that are poorly represented
   in elite demonstrations and therefore need eventual self-play/RL recovery;
2. the executor must still satisfy mechanical survival invariants instead of
   allowing strategic mistakes to turn into avoidable asset destruction.

V0.6 addresses only (2) plus diagnostics. It deliberately does not teach the
BC how to manage liquidity and does not hide strategic overcommitment behind a
large economic recovery heuristic.

## Survival rules

Implemented in `executor_v0/agent.py`:

- **Protect current-day feed from WHEAT sells.** For every currently unfed
  animal, one WHEAT is reserved across carried inventory + shed stock. The
  manager's WHEAT sell request is clipped against only the unreserved shed
  amount.
- **Starvation-boundary FEED preempts all non-survival tile work.** If any
  animal has `consecutive_unfed >= 1` and is still unfed, workers dispatch only
  FEED work until the boundary clears. Market tasks remain visible so WHEAT
  can still be bought.
- **Feed shortage buys execute before hiring and discretionary purchases.** A
  survival WHEAT shortage gets the first claim on post-sell cash. If the whole
  shortage is unaffordable, the executor buys the largest affordable prefix
  instead of repeatedly buying nothing.
- **Pause new long-lived commitments under survival pressure or prior work
  debt.** BUILD_COOP / BUILD_PASTURE / PLACE / BUY_ANIMAL / BUY_LAND tasks are
  mechanically suppressed while current animals are starving / lack current
  feed, or when the preceding day ended with unresolved worker debt. The
  manager plan itself is left untouched and is reconsidered on later turns/
  days.
- **Maintenance-aware hiring still applies during starvation.** FEED remains
  included in the travel-aware maintenance capacity floor, but survival feed
  purchases reserve their cash before hire affordability is computed.

These are executor invariants, not learned strategy.

## Work-debt semantics

V0.5's `unfinished_task_turns` counted every turn that a task was unassigned.
That mislabeled ordinary travel/dependency waiting as failure even when the
work completed later the same day.

Diagnostics schema version 2 changes the interpretation:

- `pending_task_turns` / `pending_maintenance_turns` keep the old churn signal
  for debugging only;
- `end_of_day_work_debt` is the authoritative unfinished-work measure;
- on hour 23, a tile task is debt iff it still needs a worker action after the
  final primitive action. An interaction assigned on hour 23 is considered
  complete because worker actions execute before the daily refresh; movement,
  PICKUP, PASS, dependency-blocked, or unassigned work remains debt;
- debt is split into `survival`, `maintenance`, `productive`, and `manager`;
- compatibility keys `unfinished_tasks`, `missed_maintenance`,
  `unfinished_task_turns`, and `missed_maintenance_turns` now reflect only
  actual end-of-day debt (one count per task/day), not intermediate waiting.

This matches the intended operational definition: a sustainable farm should
finish the work it committed to; temporary waiting is not unfinished work.

## Targeted regression coverage

`tests/test_executor_v06_survival.py` covers:

- WHEAT sell reserve;
- starvation FEED preemption;
- feed-shortage purchase ordering and expansion suppression;
- temporary waiting resolved on hour 23 not becoming work debt;
- real hour-23 movement remaining debt and suppressing next-day expansion.

The tests were added with the implementation; a full repository test run and
closed-loop fast panel still need to be executed in an environment with the
compiled fast engine.

## Bounded evaluation plan

Do not immediately rerun hundreds of games.

1. Historical fixed panel: seeds `7,17,42,123,2026`, both seats vs PASS.
2. Collapse panel: reuse a small set of known bad trajectories from the first
   expanded run (for example PASS `1013/0`, `1022/1`, `1003/0`, `1026/1`,
   `1011/{0,1}`, plus a few bankrupt self-play seeds).
3. Acceptance priority:
   - zero fallback/opening divergence;
   - **zero animal escapes** on the bounded panel;
   - zero end-of-day survival debt, then drive total work debt toward zero;
   - materially reduce the <1000-bank collapse tail without destroying the
     historical center;
   - only after the bounded panel is sane, rerun the 32-seed PASS and 32-game
     self-play panels.

The remaining cash-pressure/recovery problem belongs to the later RL manager:
penalize persistent insolvency/uncollateralized maintenance liabilities rather
than spending itself, while terminal game outcome remains the primary signal.

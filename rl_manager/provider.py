"""Queued/precomputed plan provider behind the executor factory seam.

At each manager day start the runner encodes the acting seat exactly once,
batches the policy call, and queues the resulting `DailyPlan` here; the
unmodified `executor_v0.ExecutorAgent` then consumes it through the standard
`PlanProvider` protocol. Missing or double-consumed plans fail loudly.
"""

from __future__ import annotations

from typing import Mapping

from executor_v0.plan import DailyPlan


class QueuedPlanProvider:
    """PlanProvider fed by the runner's batched day-boundary decisions."""

    def __init__(self) -> None:
        self._queued: dict[int, DailyPlan] = {}
        self.consumed: dict[int, DailyPlan] = {}

    def queue(self, day: int, plan: DailyPlan) -> None:
        if isinstance(day, bool) or not isinstance(day, int):
            raise ValueError(f"day must be an int, got {day!r}")
        if not isinstance(plan, DailyPlan):
            raise ValueError(
                f"queue requires a DailyPlan, got {type(plan).__name__}")
        if day in self._queued or day in self.consumed:
            raise ValueError(f"a plan for day {day} was already queued")
        self._queued[day] = plan

    def daily_plan(
        self,
        obs: Mapping,
        seat: int,
        previous_execution: Mapping[str, int] | None = None,
    ) -> DailyPlan:
        day = int(obs["day"])
        plan = self._queued.pop(day, None)
        if plan is None:
            raise ValueError(
                f"no queued manager plan for seat {seat} day {day}; the "
                f"runner must batch and queue every manager decision before "
                f"the executor's first turn of that day")
        self.consumed[day] = plan
        return plan

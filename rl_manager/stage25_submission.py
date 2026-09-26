"""Native Stage 2.5 Kaggle submission runtime.

The runtime composes the existing standard-mixed opening, native deterministic
Stage 2.5 provider, and the canonical fixed-strip executor profile. The only
submission-owned state is observed realized labor, which the panel runner also
tracks outside the manager provider.
"""

from __future__ import annotations

from dataclasses import asdict, replace
import time
from pathlib import Path
from collections.abc import Mapping
from typing import Any

from executor_v0.strip_executor import StripExecutorController
from opening_book.agent import make_opening_agent
from replay_daily.constants import total_hire_cost
from rl_manager.executor_factory import make_stage25_executor_factory
from rl_manager.stage25_provider import Stage25PlanProvider
from rl_manager.stage25_submission_observation import (
    canonicalize_official_observation,
)

MANAGER_START_DAY = 4
OPENING_IDENTITY = "standard_mixed"


class RealizedLaborTracker:
    """Track per-seat labor using only observed daily hires counters."""

    def __init__(self, seat: int) -> None:
        self.seat = int(seat)
        self.last_observed_day: int | None = None
        self.max_hires_today = 0
        self.previous_execution = {"workers_hired": 0, "hire_cost": 0}

    def observe(self, observation: Mapping[str, Any]) -> None:
        day = int(observation["day"])
        farm = observation["farms"][self.seat]
        hires = int(farm.get("hires_today", 0) or 0)
        if self.last_observed_day is not None and day > self.last_observed_day:
            realized_hires = self.max_hires_today
            self.previous_execution = {
                "workers_hired": realized_hires,
                "hire_cost": int(total_hire_cost(realized_hires)),
            }
            self.max_hires_today = hires
        elif self.last_observed_day is None or day < self.last_observed_day:
            self.max_hires_today = hires
        else:
            self.max_hires_today = max(self.max_hires_today, hires)
        self.last_observed_day = day


class Stage25SubmissionAgent:
    """Stateful callable for one Kaggle game and one controlled seat."""

    def __init__(self, checkpoint: str | Path, *, seat: int) -> None:
        if isinstance(seat, bool) or int(seat) not in (0, 1):
            raise ValueError(f"seat must be 0 or 1, got {seat!r}")
        self.seat = int(seat)
        self.checkpoint = Path(checkpoint)
        factory = make_stage25_executor_factory()
        self.executor_profile = factory.effective_profile
        self.executor_config = replace(
            factory.strip_config, acting_seat=self.seat)
        self.provider = Stage25PlanProvider(
            episode_id="kaggle-stage25-submission",
            seat=self.seat,
            manager_start_day=MANAGER_START_DAY,
            native_checkpoint=self.checkpoint,
            mode="deterministic",
            seed=0,
        )
        self.controller = StripExecutorController(config=self.executor_config)
        self.opening = make_opening_agent(
            opening=OPENING_IDENTITY,
            downstream=self._stage25_action,
            seat=self.seat,
        )
        self.labor = RealizedLaborTracker(self.seat)
        self._plan_day: int | None = None
        self._plan: Any = None
        self.action_count = 0
        self.first_call_latency_s: float | None = None
        self.manager_inference_latencies_s: list[float] = []

    def _stage25_action(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        day = int(observation["day"])
        if self._plan_day != day:
            started = time.perf_counter()
            self._plan = self.provider.daily_plan(
                observation, self.seat, self.labor.previous_execution)
            self.manager_inference_latencies_s.append(
                time.perf_counter() - started)
            self._plan_day = day
        return self.controller.act(observation, self._plan).action_dict()

    def __call__(
        self, observation: Mapping[str, Any], configuration: Any = None
    ) -> dict[str, Any]:
        del configuration
        started = time.perf_counter()
        adapted = canonicalize_official_observation(observation)
        if int(adapted.get("player", -1)) != self.seat:
            raise ValueError(
                f"observation player {adapted.get('player')!r} does not match "
                f"configured seat {self.seat}")
        self.labor.observe(adapted)
        action = self.opening(adapted)
        self.action_count += 1
        if self.first_call_latency_s is None:
            self.first_call_latency_s = time.perf_counter() - started
        return action

    def diagnostics_json(self) -> dict[str, Any]:
        return {
            "seat": self.seat,
            "checkpoint": self.checkpoint.name,
            "mode": "deterministic",
            "opening": OPENING_IDENTITY,
            "manager_start_day": MANAGER_START_DAY,
            "last_observed_day": self.labor.last_observed_day,
            "max_hires_today": self.labor.max_hires_today,
            "previous_execution": dict(self.labor.previous_execution),
            "action_count": self.action_count,
            "first_call_latency_s": self.first_call_latency_s,
            "manager_inference_latency_s": list(
                self.manager_inference_latencies_s),
            "executor_profile": dict(self.executor_profile),
            "opening_diagnostics": self.opening.diagnostics_json(),
            "executor_diagnostics": self._executor_diagnostics(),
        }

    def _executor_diagnostics(self) -> dict[str, Any]:
        days = getattr(self.controller, "_daily", {})
        return {
            "config": asdict(self.executor_config),
            "days": dict(days),
            "provider": self.provider.diagnostics_json(),
        }


def make_stage25_submission_agent(
    checkpoint: str | Path, *, seat: int
) -> Stage25SubmissionAgent:
    """Create a deterministic native Stage 2.5 callable for one game seat."""
    return Stage25SubmissionAgent(checkpoint, seat=seat)

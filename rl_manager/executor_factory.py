"""Executor factory / version seam (issue #9, architecture req. on executors).

The executor stays entirely outside the RL gradient.  The legacy factory
builds the default :class:`executor_v0.agent.ExecutorAgent`; the explicit
Stage 2.5 factory adapts its injected plan provider to the fixed-strip
controller.  Both are parameterized by backend name/seat/configuration and an
injected plan provider, so swapping the factory does not change RL semantics.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, replace
from typing import Any, Mapping, Protocol

from rl_manager.provider import QueuedPlanProvider

__all__ = [
    "EXECUTOR_FACTORY_VERSION",
    "STAGE25_EXECUTOR_PROFILE_NAME",
    "STAGE25_EXECUTOR_PROFILE_VERSION",
    "RlExecutorFactory",
    "Stage25ExecutorFactory",
    "Stage25ExecutorProfile",
    "Stage25StripExecutorAgent",
    "make_default_executor_factory",
    "make_stage25_executor_factory",
]

EXECUTOR_FACTORY_VERSION = "executor_v0.make_agent(strict=True)@stage-a-v1"
STAGE25_EXECUTOR_PROFILE_VERSION = "strip_executor_v1@stage25-v1"
STAGE25_EXECUTOR_PROFILE_NAME = "stage25_strip_executor"


@dataclass(frozen=True)
class Stage25ExecutorProfile:
    """Versioned, introspectable fixed-strip settings for Stage 2.5."""

    strip_config: Any
    name: str = STAGE25_EXECUTOR_PROFILE_NAME
    version: str = STAGE25_EXECUTOR_PROFILE_VERSION

    def __post_init__(self) -> None:
        if getattr(self.strip_config, "aggressive_sell_all", None) is not True:
            raise ValueError(
                "Stage 2.5 strip profile requires aggressive_sell_all=True")

    def to_json_dict(self) -> dict[str, Any]:
        config = asdict(self.strip_config)
        config["acting_seat"] = "factory_injected_seat"
        return {
            "name": self.name,
            "version": self.version,
            "controller": "executor_v0.strip_executor.StripExecutorController",
            "strip_config": config,
            "aggressive_sell_all": bool(self.strip_config.aggressive_sell_all),
        }


def _validate_stage25_config(config: Any, strip_config_type: type) -> None:
    if not isinstance(config, strip_config_type):
        raise TypeError(
            "strip_config must be an executor_v0.strip_executor."
            "StripExecutorConfig instance")
    if config.aggressive_sell_all is not True:
        raise ValueError(
            "Stage 2.5 strip profile requires aggressive_sell_all=True")


class Stage25StripExecutorAgent:
    """Callable runner adapter around one fixed-strip controller.

    The provider remains authoritative for manager cadence and persistent-K
    lifecycle.  This adapter only retrieves one already-accepted ``DailyPlan``
    at a day boundary and reuses it for that day's primitive turns.
    """

    def __init__(self, *, provider: Any, seat: int, strip_config: Any,
                 profile: Mapping[str, Any],
                 materialize_diagnostics: bool = True) -> None:
        from executor_v0.strip_executor import StripExecutorController

        self.provider = provider
        self.seat = int(seat)
        self.materialize_diagnostics = bool(materialize_diagnostics)
        self.config = replace(strip_config, acting_seat=self.seat)
        self.controller = StripExecutorController(
            config=self.config,
            materialize_diagnostics=self.materialize_diagnostics)
        self.effective_profile = copy.deepcopy(dict(profile))
        self._day: int | None = None
        self._plan: Any | None = None
        self._days: dict[str, dict[str, Any]] = {}

    def _capture_current_day(self) -> None:
        if self.materialize_diagnostics or self._day is None:
            return
        day = str(self._day)
        if day not in self._days:
            # ``controller.diagnostics`` constructs a fresh snapshot from the
            # observation held for the last action of this day.  Capturing
            # before the next day starts preserves the historical snapshot
            # meaning without copying on every primitive turn.
            self._days[day] = self.controller.diagnostics

    def __call__(self, obs: Mapping[str, Any]) -> dict[str, Any]:
        day = int(obs["day"])
        if self._day != day:
            self._capture_current_day()
            self._plan = self.provider.daily_plan(obs, self.seat)
            self._day = day
        result = self.controller.act(obs, self._plan)
        if self.materialize_diagnostics:
            self._days[str(day)] = copy.deepcopy(result.diagnostics)
        return result.action_dict()

    def finalize_diagnostics(self, obs: Mapping[str, Any], seat: int) -> None:
        del obs, seat
        self._capture_current_day()

    def diagnostics_json(self) -> dict[str, Any]:
        days = copy.deepcopy(self._days)
        if not self.materialize_diagnostics and self._day is not None:
            day = str(self._day)
            if day not in days:
                days[day] = self.controller.diagnostics
        diagnostics = {
            "schema_version": 1,
            "seat": self.seat,
            "effective_profile": copy.deepcopy(self.effective_profile),
            "config": asdict(self.config),
            "days": days,
            "fallback_errors": [],
        }
        provider_diagnostics = getattr(self.provider, "diagnostics_json", None)
        if callable(provider_diagnostics):
            diagnostics["provider_diagnostics"] = provider_diagnostics()
        return diagnostics


@dataclass(frozen=True)
class Stage25ExecutorFactory:
    """Factory carrying the complete Stage 2.5 profile across rollouts."""

    profile: Stage25ExecutorProfile
    materialize_diagnostics: bool = True

    @property
    def name(self) -> str:
        return self.profile.name

    @property
    def version(self) -> str:
        return self.profile.version

    @property
    def agent_config(self) -> Any:
        """Deprecated compatibility alias for the registered factory wire."""
        return self.profile.strip_config

    @property
    def strip_config(self) -> Any:
        return self.profile.strip_config

    @property
    def effective_profile(self) -> dict[str, Any]:
        return self.profile.to_json_dict()

    def with_low_telemetry(self, enabled: bool) -> "Stage25ExecutorFactory":
        return replace(self, materialize_diagnostics=not bool(enabled))

    def create(
        self,
        *,
        backend_name: str,
        seat: int,
        configuration: Mapping[str, Any],
        provider: QueuedPlanProvider,
    ) -> object:
        del backend_name, configuration
        return Stage25StripExecutorAgent(
            provider=provider, seat=seat,
            strip_config=self.profile.strip_config,
            profile=self.profile.to_json_dict(),
            materialize_diagnostics=self.materialize_diagnostics,
        )


def make_stage25_executor_factory(
    strip_config: Any | None = None,
    *,
    low_telemetry: bool = False,
) -> RlExecutorFactory:
    """Build the explicit Stage 2.5 fixed-strip executor profile."""
    from executor_v0.strip_executor import StripExecutorConfig

    resolved_config = strip_config or StripExecutorConfig(
        aggressive_sell_all=True)
    _validate_stage25_config(resolved_config, StripExecutorConfig)
    return Stage25ExecutorFactory(
        profile=Stage25ExecutorProfile(strip_config=resolved_config),
        materialize_diagnostics=not bool(low_telemetry))


class RlExecutorFactory(Protocol):
    """Fresh executor agent per (backend, seat); never shared across games."""

    name: str
    version: str

    def create(
        self,
        *,
        backend_name: str,
        seat: int,
        configuration: Mapping[str, Any],
        provider: QueuedPlanProvider,
    ) -> object: ...


def make_default_executor_factory(
    agent_config: Any | None = None,
) -> RlExecutorFactory:
    """Default factory building strict unmodified `executor_v0` agents."""

    from executor_v0.agent import AgentConfig, make_agent

    resolved_config = agent_config or AgentConfig(
        strict=True, optional_spare_watering=True)
    if not isinstance(resolved_config, AgentConfig):
        raise TypeError(
            "agent_config must be an executor_v0.agent.AgentConfig instance")

    class _DefaultExecutorFactory:
        name = "executor_v0"
        version = EXECUTOR_FACTORY_VERSION
        agent_config = resolved_config

        def create(
            self,
            *,
            backend_name: str,
            seat: int,
            configuration: Mapping[str, Any],
            provider: QueuedPlanProvider,
        ) -> object:
            del backend_name, configuration  # engine-agnostic executor
            return make_agent(provider=provider, seat=seat,
                              config=resolved_config)

    return _DefaultExecutorFactory()

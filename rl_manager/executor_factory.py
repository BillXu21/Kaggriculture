"""Executor factory / version seam (issue #9, architecture req. on executors).

The executor stays entirely outside the RL gradient.  The legacy factory
continues to build the default `executor_v0.ExecutorAgent`; the explicit
Stage 2.5 factory selects a versioned configuration of the same executor
through this seam.  Both are parameterized by backend name/seat/configuration
and an injected plan provider, so swapping the factory does not change
RL-semantics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Protocol

from rl_manager.provider import QueuedPlanProvider

__all__ = [
    "EXECUTOR_FACTORY_VERSION",
    "STAGE25_EXECUTOR_PROFILE_NAME",
    "STAGE25_EXECUTOR_PROFILE_VERSION",
    "RlExecutorFactory",
    "Stage25ExecutorFactory",
    "Stage25ExecutorProfile",
    "make_default_executor_factory",
    "make_stage25_executor_factory",
]

EXECUTOR_FACTORY_VERSION = "executor_v0.make_agent(strict=True)@stage-a-v1"
STAGE25_EXECUTOR_PROFILE_VERSION = "stage25_executor_v1"
STAGE25_EXECUTOR_PROFILE_NAME = "stage25_executor"

# These are intentionally explicit rather than a best-effort overlay.  A
# Stage 2.5 rollout must be reproducible from its recorded profile, and an
# accidental false value here would turn the zero-valued transport scaffolds
# into missing mechanics.
_STAGE25_REQUIRED_TRUE = (
    "strict",
    "heuristic_care",
    "heuristic_fertilizer",
    "aggressive_sell_all",
    "suppress_expansion_from_prior_debt",
    "optional_spare_watering",
    "immediate_plant_water",
    "deadline_safe_planting",
    "deadline_safe_hiring",
    "persistent_worker_queues",
    "queue_ownership_repair",
    "batch_reserved_supplies",
    "underfoot_queue_insertion",
    "starvation_workload_visibility_repair",
)


@dataclass(frozen=True)
class Stage25ExecutorProfile:
    """Versioned, introspectable executor settings for Stage 2.5.

    The policy transport deliberately carries zero CARE/fertilizer/sell
    fields.  These settings supply the deterministic executor-side behavior
    without changing the policy action schema.  This is an executor profile,
    not a promoted strategy label or a claim about final policy quality.
    """

    agent_config: Any
    name: str = STAGE25_EXECUTOR_PROFILE_NAME
    version: str = STAGE25_EXECUTOR_PROFILE_VERSION

    def __post_init__(self) -> None:
        disabled = [name for name in _STAGE25_REQUIRED_TRUE
                    if not hasattr(self.agent_config, name)
                    or getattr(self.agent_config, name) is not True]
        if disabled:
            raise ValueError(
                "Stage 2.5 executor profile requires these settings enabled: "
                f"{disabled}; refusing to silently alter the profile")

    def to_json_dict(self) -> dict[str, Any]:
        config = asdict(self.agent_config)
        return {
            "name": self.name,
            "version": self.version,
            "agent_config": config,
            "required_true": list(_STAGE25_REQUIRED_TRUE),
            "strategic_protection": {
                "suppress_expansion_from_prior_debt": bool(
                    self.agent_config.suppress_expansion_from_prior_debt),
                "current_survival_expansion_veto": "executor_enforced",
            },
        }


def _validate_stage25_config(config: Any, agent_config_type: type) -> None:
    if not isinstance(config, agent_config_type):
        raise TypeError(
            "agent_config must be an executor_v0.agent.AgentConfig instance")
    missing = [name for name in _STAGE25_REQUIRED_TRUE
               if not hasattr(config, name)]
    if missing:
        raise ValueError(
            "Stage 2.5 executor profile is missing required settings: "
            f"{missing}")
    disabled = [name for name in _STAGE25_REQUIRED_TRUE
                if getattr(config, name) is not True]
    if disabled:
        raise ValueError(
            "Stage 2.5 executor profile requires these settings enabled: "
            f"{disabled}; refusing to silently alter the profile")


@dataclass(frozen=True)
class Stage25ExecutorFactory:
    """Factory carrying the complete Stage 2.5 profile across rollouts."""

    profile: Stage25ExecutorProfile

    @property
    def name(self) -> str:
        return self.profile.name

    @property
    def version(self) -> str:
        return self.profile.version

    @property
    def agent_config(self) -> Any:
        return self.profile.agent_config

    @property
    def effective_profile(self) -> dict[str, Any]:
        return self.profile.to_json_dict()

    def create(
        self,
        *,
        backend_name: str,
        seat: int,
        configuration: Mapping[str, Any],
        provider: QueuedPlanProvider,
    ) -> object:
        del backend_name, configuration
        from executor_v0.agent import make_agent

        return make_agent(
            provider=provider,
            seat=seat,
            config=self.profile.agent_config,
            profile=self.profile.to_json_dict(),
        )


def make_stage25_executor_factory(
    agent_config: Any | None = None,
) -> RlExecutorFactory:
    """Build the explicit Stage 2.5 executor profile.

    Passing a config is useful for an explicit telemetry-only variation, but
    required upkeep, liquidation, and expansion-protection settings must
    remain enabled.  The config is never silently patched.
    """
    from executor_v0.agent import AgentConfig

    resolved_config = agent_config or AgentConfig(
        strict=True,
        suppress_expansion_from_prior_debt=True,
        aggressive_sell_all=True,
        optional_spare_watering=True,
        immediate_plant_water=True,
        deadline_safe_planting=True,
        deadline_safe_hiring=True,
        persistent_worker_queues=True,
        queue_ownership_repair=True,
        batch_reserved_supplies=True,
        underfoot_queue_insertion=True,
        starvation_workload_visibility_repair=True,
        heuristic_care=True,
        heuristic_fertilizer=True,
    )
    _validate_stage25_config(resolved_config, AgentConfig)
    return Stage25ExecutorFactory(
        profile=Stage25ExecutorProfile(agent_config=resolved_config))


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

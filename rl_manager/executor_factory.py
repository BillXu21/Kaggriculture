"""Executor factory / version seam (issue #9, architecture req. on executors).

The executor stays entirely outside the RL gradient and is never edited: the
runner only ever builds unmodified `executor_v0.ExecutorAgent` instances
through this factory, parameterized by backend name/seat/configuration and an
injected plan provider. Issue #7 can swap V0 -> selected V0.5 by shipping a
different factory without any RL-semantics change.
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol

from rl_manager.provider import QueuedPlanProvider

EXECUTOR_FACTORY_VERSION = "executor_v0.make_agent(strict=True)@stage-a-v1"


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

    resolved_config = agent_config or AgentConfig(strict=True)
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

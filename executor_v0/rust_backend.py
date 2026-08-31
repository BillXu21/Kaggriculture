"""Explicit Rust V0 executor experiment seam.

The first prototype intentionally keeps :class:`ExecutorAgent` as the
semantic oracle.  The PyO3 object owns and calls that agent, which makes the
native boundary measurable without silently introducing a second heuristic.
If the extension is not built, the same opt-in factory uses a transparent
Python shim and remains exact but is not a native benchmark result.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .agent import AgentConfig, ExecutorAgent
from .manager import PlanProvider

RUST_EXECUTOR_FACTORY_VERSION = (
    "executor_v0.rust-v0-parity-adapter@stage-a-v0"
)


def _native_type() -> type[Any] | None:
    try:
        from fast_env._kaggriculture_env import RustExecutorV0
    except (ImportError, AttributeError):
        return None
    return RustExecutorV0


class RustExecutorV0:
    """Native-seam wrapper with exact Python executor behavior in V0."""

    def __init__(
        self,
        provider: PlanProvider,
        *,
        seat: int | None = None,
        config: AgentConfig | None = None,
    ) -> None:
        oracle = ExecutorAgent(provider, seat=seat, config=config)
        native = _native_type()
        self._oracle = oracle
        self._native = native(oracle) if native is not None else None

    def __call__(self, obs: Mapping[str, Any]) -> dict[str, Any]:
        if self._native is not None:
            return self._native(obs)
        return self._oracle(obs)

    @property
    def debug_trace_turn(self) -> dict[str, Any] | None:
        if self._native is not None:
            return self._native.debug_trace_turn
        return self._oracle.debug_trace_turn

    def diagnostics_json(self) -> dict[str, Any]:
        if self._native is not None:
            return self._native.diagnostics_json()
        return self._oracle.diagnostics_json()


class RustExecutorFactory:
    """Fresh per-seat/per-game factory for the opt-in Rust seam."""

    name = "executor_v0_rust"
    version = RUST_EXECUTOR_FACTORY_VERSION

    def __init__(self, agent_config: AgentConfig | None = None) -> None:
        self.agent_config = agent_config or AgentConfig(
            strict=True,
            record_turn_snapshot=False,
            optional_spare_watering=True,
        )

    def create(
        self,
        *,
        backend_name: str,
        seat: int,
        configuration: Mapping[str, Any],
        provider: PlanProvider,
    ) -> RustExecutorV0:
        del backend_name, configuration
        return RustExecutorV0(
            provider, seat=seat, config=self.agent_config)


def make_rust_executor_factory(
    agent_config: AgentConfig | None = None,
) -> RustExecutorFactory:
    return RustExecutorFactory(agent_config)


__all__ = [
    "RUST_EXECUTOR_FACTORY_VERSION",
    "RustExecutorFactory",
    "RustExecutorV0",
    "make_rust_executor_factory",
]

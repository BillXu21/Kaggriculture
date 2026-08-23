"""Narrow engine backend seam for deterministic policy/evaluation code.

``make_backend(name, configuration)`` returns one of two interchangeable
backends exposing the same minimal protocol:

- ``reset() -> observations``
- ``step(actions) -> (observations, rewards, statuses)``
- ``canonical_state() -> dict`` (see :mod:`oracle.canonical`)
- ``rewards`` / ``statuses`` properties
- ``name`` attribute

The ``"official"`` backend lazily imports ``kaggle_environments`` (oracle and
evaluation use only); the ``"fast"`` backend is the normal Stage-1 engine.
Stateful callable agents stay outside this seam: it deliberately carries no
agent logic — callers submit explicit action pairs.
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence

SUPPORTED_BACKENDS = ("official", "fast")


class EngineBackend(Protocol):
    name: str

    def reset(self) -> list[dict[str, Any]]: ...

    def step(
        self, actions: Sequence[Mapping[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[float], list[str]]: ...

    def canonical_state(self) -> dict[str, Any]: ...

    @property
    def rewards(self) -> list[float]: ...

    @property
    def statuses(self) -> list[str]: ...


class FastBackendAdapter:
    """Adapts ``fast_env.FastKaggricultureEnv`` to the backend protocol."""

    name = "fast"

    def __init__(self, configuration: Mapping[str, Any] | None = None) -> None:
        from fast_env import FastKaggricultureEnv  # lazy; keeps oracle import light

        self._env = FastKaggricultureEnv(configuration)
        self._rewards: list[float] = [0.0, 0.0]

    def reset(self) -> list[dict[str, Any]]:
        return self._env.reset()

    def observations(self) -> list[dict[str, Any]]:
        return self._env.state_snapshot()

    def step(
        self, actions: Sequence[Mapping[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[float], list[str]]:
        observations, rewards, statuses = self._env.step(actions)
        self._rewards = list(rewards)
        return observations, rewards, statuses

    def canonical_state(self) -> dict[str, Any]:
        from .canonical import canonical_state_fast

        return canonical_state_fast(
            self._env.state_snapshot(), self.rewards, self._env.statuses
        )

    @property
    def rewards(self) -> list[float]:
        return list(self._rewards)

    @property
    def statuses(self) -> list[str]:
        return self._env.statuses


def make_backend(name: str, configuration: Mapping[str, Any] | None = None) -> EngineBackend:
    if name == "fast":
        return FastBackendAdapter(configuration)
    if name == "official":
        from .official_backend import OfficialKaggricultureBackend

        return OfficialKaggricultureBackend(configuration)
    raise ValueError(f"unknown backend {name!r}; supported: {SUPPORTED_BACKENDS}")

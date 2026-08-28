"""Framework-neutral interface for native batched rollout engines."""

from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence

import numpy as np


class BatchedEngineBackend(Protocol):
    """One backend owning ``num_envs`` simultaneous two-seat games."""

    name: str
    num_envs: int

    def reset(self, seeds: Sequence[int]) -> list[list[dict[str, Any]]]: ...

    def step(
        self,
        actions: Sequence[Sequence[Mapping[str, Any]]],
    ) -> tuple[list[list[dict[str, Any]]], np.ndarray, np.ndarray]: ...

    def observations(self, index: int) -> list[dict[str, Any]]: ...

    def rewards(self, index: int) -> list[float]: ...

    def statuses(self, index: int) -> list[str]: ...


class FastBatchedBackendAdapter:
    """Adapter exposing :class:`fast_env.BatchedFastEnv` to rollout code."""

    name = "fast-batched"

    def __init__(
        self,
        num_envs: int,
        configuration: Mapping[str, Any] | None = None,
    ) -> None:
        from fast_env import BatchedFastEnv

        self._env = BatchedFastEnv(
            num_envs, configuration, canonical_observations=True
        )
        self.num_envs = self._env.num_envs

    def reset(self, seeds: Sequence[int]) -> list[list[dict[str, Any]]]:
        return self._env.reset(seeds)

    def step(
        self,
        actions: Sequence[Sequence[Mapping[str, Any]]],
    ) -> tuple[list[list[dict[str, Any]]], np.ndarray, np.ndarray]:
        return self._env.step(actions)

    def observations(self, index: int) -> list[dict[str, Any]]:
        return self._env.observations(index)

    def rewards(self, index: int) -> list[float]:
        return self._env.rewards(index)

    def statuses(self, index: int) -> list[str]:
        return self._env.statuses(index)

    @property
    def last_timing_seconds(self) -> Mapping[str, float]:
        return self._env.last_timing_seconds


def make_batched_backend(
    name: str,
    num_envs: int,
    configuration: Mapping[str, Any] | None = None,
) -> BatchedEngineBackend:
    if name in {"fast", "fast-batched"}:
        return FastBatchedBackendAdapter(num_envs, configuration)
    raise ValueError(
        f"unknown batched backend {name!r}; supported: ('fast', 'fast-batched')"
    )

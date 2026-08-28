"""Framework-neutral wire messages for local parallel rollouts.

This module is deliberately free of JAX, torch, executor, and backend
imports.  Instances cross ``multiprocessing`` queues and therefore contain
only pickle-stable scalar values, NumPy arrays, and small dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from rl_manager.types import PolicyIdentity, PolicyOutputs


@dataclass(frozen=True)
class EpisodeAssignment:
    """Serializable episode ownership and seat-policy snapshot."""

    episode_index: int
    seed: int
    composition: str
    seat_policy_identities: tuple[PolicyIdentity, PolicyIdentity]
    trainable_seats: tuple[int, ...]
    controlled_seat: int | None


@dataclass(frozen=True)
class InferenceRequest:
    """One manager-row request from a worker to the TPU owner."""

    request_id: str
    worker_id: int
    episode_index: int
    seat: int
    day: int
    policy_identity: PolicyIdentity
    prng_id: str
    inputs: Mapping[str, np.ndarray]
    queued_at: float


@dataclass(frozen=True)
class InferenceResponse:
    """One manager-row response routed by the request identifier."""

    request_id: str
    outputs: PolicyOutputs


@dataclass(frozen=True)
class WorkerTask:
    """Initial immutable task sent to one spawned rollout worker."""

    worker_id: int
    episodes: tuple[EpisodeAssignment, ...]
    runner_config: Any
    executor_factory: Any
    master_seed: int | None
    trajectory_capacity: int | None
    owner_pid: int


@dataclass(frozen=True)
class WorkerFinished:
    worker_id: int
    results: tuple[Any, ...]
    trajectory: Any | None


@dataclass(frozen=True)
class WorkerFailed:
    worker_id: int
    error_type: str
    error_message: str
    traceback: str


def policy_row_request_id(
    episode_index: int,
    seat: int,
    day: int,
    identity: PolicyIdentity,
) -> str:
    """Stable routing key independent of worker assignment or arrival order."""
    return (f"episode={int(episode_index)}/seat={int(seat)}/day={int(day)}"
            f"/policy={identity.identity_id()}")

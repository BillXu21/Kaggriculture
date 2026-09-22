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
from rl_manager.stage25_types import Stage25BehaviorIdentity, Stage25PolicyOutputs


@dataclass(frozen=True)
class EpisodeAssignment:
    """Serializable episode ownership and seat-policy snapshot."""

    episode_index: int
    seed: int
    composition: str
    seat_policy_identities: tuple[PolicyIdentity, PolicyIdentity]
    trainable_seats: tuple[int, ...]
    controlled_seat: int | None
    # Explicit framework-neutral curriculum payload for each seat.  This is
    # transported separately from the identity fingerprint; workers never
    # infer curriculum contents from a hash.
    stage25_curricula: tuple[Mapping[str, Any] | None,
                             Mapping[str, Any] | None] = (None, None)
    stage25_validation_modes: tuple[str, str] = ("strict", "strict")


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
class Stage25RequestIdentity:
    """Immutable identity for one Stage 2.5 manager boundary.

    The behavior snapshot is part of the routing identity.  Keeping it on
    the wire (rather than looking it up from a mutable parent registry) makes
    stale or cross-run responses reject before any provider state is touched.
    """

    episode_index: int
    seat: int
    day: int
    behavior_identity: Stage25BehaviorIdentity

    def __post_init__(self) -> None:
        for name in ("episode_index", "seat", "day"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an int")
        if self.episode_index < 0 or self.day < 0:
            raise ValueError("episode_index and day must be nonnegative")
        if self.seat not in (0, 1):
            raise ValueError("seat must be 0 or 1")
        if not isinstance(self.behavior_identity, Stage25BehaviorIdentity):
            raise TypeError("behavior_identity must be Stage25BehaviorIdentity")

    @property
    def request_id(self) -> str:
        return (f"episode={self.episode_index}/seat={self.seat}/day={self.day}"
                f"/behavior={self.behavior_identity.identity_id()}")

    @property
    def row_id(self) -> str:
        """Stable row token used by stochastic parent inference."""
        return self.request_id


@dataclass(frozen=True)
class Stage25InferenceRequest:
    """Framework-neutral parent-inference request.

    ``crop_capacity`` is explicitly the pre-decision K ledger.  Physical
    context and support are copied by the producer and are never recomputed
    by the parent from policy outputs.
    """

    identity: Stage25RequestIdentity
    worker_id: int
    prng_id: str
    inputs: Mapping[str, np.ndarray]
    crop_capacity: np.ndarray
    physical_context: Any
    queued_at: float
    support: Any | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        if "crop_capacity" in self.inputs:
            raise ValueError(
                "Stage 2.5 request inputs must not duplicate crop_capacity")
        capacity = np.asarray(self.crop_capacity)
        if capacity.shape not in ((5,), (1, 5)) or not np.issubdtype(
                capacity.dtype, np.integer):
            raise ValueError("crop_capacity must be an integer [5] or [1, 5] array")
        copied = np.array(capacity, dtype=np.int16, copy=True)
        copied.setflags(write=False)
        object.__setattr__(self, "crop_capacity", copied)

    @property
    def request_id(self) -> str:
        return self.identity.request_id


@dataclass(frozen=True)
class Stage25InferenceResponse:
    """Response carrying the exact request identity and behavior snapshot."""

    request_id: str
    identity: Stage25RequestIdentity
    outputs: Stage25PolicyOutputs

    def __post_init__(self) -> None:
        if self.request_id != self.identity.request_id:
            raise ValueError("Stage 2.5 response request_id disagrees with identity")
        if self.outputs.policy_identity != self.identity.behavior_identity:
            raise ValueError(
                "Stage 2.5 response behavior identity disagrees with request")


@dataclass(frozen=True)
class Stage25BootstrapRequest:
    """Value-only next-state request for a truncated manager trajectory."""

    identity: Stage25RequestIdentity
    worker_id: int
    inputs: Mapping[str, np.ndarray]
    crop_capacity: np.ndarray
    physical_context: Any
    queued_at: float

    def __post_init__(self) -> None:
        if "crop_capacity" in self.inputs:
            raise ValueError(
                "Stage 2.5 bootstrap inputs must not duplicate crop_capacity")
        capacity = np.asarray(self.crop_capacity)
        if capacity.shape not in ((5,), (1, 5)) or not np.issubdtype(
                capacity.dtype, np.integer):
            raise ValueError("bootstrap crop_capacity must be an integer [5] or [1, 5] array")
        copied = np.array(capacity, dtype=np.int16, copy=True)
        copied.setflags(write=False)
        object.__setattr__(self, "crop_capacity", copied)

    @property
    def request_id(self) -> str:
        return self.identity.request_id + "/bootstrap"


@dataclass(frozen=True)
class Stage25BootstrapResponse:
    """Identity-bound scalar value for truncation bootstrapping."""

    request_id: str
    identity: Stage25RequestIdentity
    value: np.ndarray
    policy_identity: Stage25BehaviorIdentity

    def __post_init__(self) -> None:
        value = np.asarray(self.value)
        if value.shape != () or value.dtype != np.dtype(np.float32):
            raise ValueError("bootstrap value must be a float32 scalar ndarray")
        if self.request_id != self.identity.request_id + "/bootstrap":
            raise ValueError("bootstrap response request_id disagrees with identity")
        if self.policy_identity != self.identity.behavior_identity:
            raise ValueError("bootstrap response behavior identity disagrees with request")


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
    stage25_trajectory_capacity: int | None = None


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

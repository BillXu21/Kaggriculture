"""Framework-neutral identity routing for independent PPO trainers.

The parent process owns trainers and accelerator placement. Rollout workers
continue to send only immutable policy identities and NumPy rows; they never
need to know which accelerator device serves a policy.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

from rl_manager.parallel_protocol import InferenceRequest
from rl_manager.trajectory import TransitionMetadata
from rl_manager.types import PolicyIdentity


class TrainerIdentityRouter:
    """Route inference and trainable trajectory rows by exact policy identity."""

    def __init__(self, trainers: Mapping[PolicyIdentity, Any]) -> None:
        if not trainers:
            raise ValueError("at least one trainer identity is required")
        self._trainers = dict(trainers)

    def group_requests(
        self,
        requests: Sequence[InferenceRequest],
    ) -> list[tuple[PolicyIdentity, Any, tuple[InferenceRequest, ...]]]:
        """Return deterministic identity-homogeneous owner dispatch groups."""
        grouped: dict[PolicyIdentity, list[InferenceRequest]] = defaultdict(list)
        request_ids: set[str] = set()
        for request in requests:
            if request.request_id in request_ids:
                raise ValueError(f"duplicate inference request {request.request_id!r}")
            request_ids.add(request.request_id)
            if request.policy_identity not in self._trainers:
                raise ValueError(
                    "no trainer registered for policy identity "
                    f"{request.policy_identity.identity_id()}")
            grouped[request.policy_identity].append(request)

        routed = []
        for identity in sorted(grouped, key=lambda item: (
                item.identity_id(), item.version, item.fingerprint)):
            rows = tuple(sorted(grouped[identity], key=lambda request: (
                request.episode_index, request.seat, request.day,
                request.request_id)))
            routed.append((identity, self._trainers[identity], rows))
        return routed

    def partition_trainable_rows(
        self,
        arrays: Mapping[str, np.ndarray],
        metadata: Sequence[TransitionMetadata],
    ) -> dict[PolicyIdentity, dict[str, np.ndarray]]:
        """Partition trainable trajectory arrays without cross-policy rows."""
        if "trainable" not in arrays:
            raise ValueError("trajectory arrays are missing 'trainable'")
        row_count = int(np.asarray(arrays["trainable"]).shape[0])
        if len(metadata) != row_count:
            raise ValueError(
                f"trajectory metadata count {len(metadata)} != array rows "
                f"{row_count}")
        for name, array in arrays.items():
            if np.asarray(array).shape[0] != row_count:
                raise ValueError(
                    f"trajectory array {name!r} has "
                    f"{np.asarray(array).shape[0]} rows, expected {row_count}")

        rows_by_identity: dict[PolicyIdentity, list[int]] = {
            identity: [] for identity in self._trainers
        }
        identity_by_fields = {
            (identity.identity_id(), identity.version, identity.fingerprint): identity
            for identity in self._trainers
        }
        trainable = np.asarray(arrays["trainable"]).astype(bool)
        for row, record in enumerate(metadata):
            if record.index != row:
                raise ValueError(
                    f"trajectory metadata index {record.index} != row {row}")
            if bool(record.trainable) != bool(trainable[row]):
                raise ValueError(
                    f"trajectory trainable flag disagrees at row {row}")
            if not trainable[row]:
                continue
            fields = (record.policy_id, record.policy_version,
                      record.policy_fingerprint)
            identity = identity_by_fields.get(fields)
            if identity is None:
                raise ValueError(
                    f"trainable trajectory row {row} belongs to unknown policy "
                    f"{record.policy_id!r}")
            rows_by_identity[identity].append(row)

        return {
            identity: {
                name: np.ascontiguousarray(np.asarray(array)[rows])
                for name, array in arrays.items()
            }
            for identity, rows in rows_by_identity.items()
        }


__all__ = ["TrainerIdentityRouter"]

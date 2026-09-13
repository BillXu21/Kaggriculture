"""Framework-neutral Stage 2.5 inference contracts.

This module is safe to import in spawned CPU rollout workers.  In particular,
it must not import JAX, torch, executors, or environment backends.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping

import numpy as np

from bc_manager.economics import normalize_e_history_version


STAGE25_POLICY_SCHEMA_VERSION = "stage25_policy_v1"
STAGE25_PHYSICAL_SUPPORT_VERSION = "stage25_physical_v1"
STAGE25_RNG_NAMESPACE = "stage25/rollout/v1"


def stage25_rng_namespace(identity: "Stage25BehaviorIdentity", seed: int = 0) -> str:
    """Return the one rollout RNG namespace shared by local and parallel paths."""
    if not isinstance(identity, Stage25BehaviorIdentity):
        raise TypeError("identity must be Stage25BehaviorIdentity")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    return (f"{STAGE25_RNG_NAMESPACE}/seed={int(seed)}"
            f"/behavior={identity.identity_id()}")


def stage25_row_token(row_id: object) -> int:
    """Map one immutable logical row identity to a stable policy token."""
    if isinstance(row_id, (str, np.str_)):
        text = "s:" + str(row_id)
    elif isinstance(row_id, (int, np.integer)) and not isinstance(
            row_id, (bool, np.bool_)):
        text = "i:" + str(int(row_id))
    else:
        raise ValueError("row_ids must contain non-empty strings or integers")
    if text == "s:":
        raise ValueError("row_ids must not contain empty strings")
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4],
                          "little") % (2**31 - 1)


@dataclass(frozen=True)
class Stage25BehaviorIdentity:
    """Immutable identity for one exact Stage 2.5 behavior distribution."""

    name: str
    version: str
    parameter_fingerprint: str
    observation_schema_version: str
    policy_schema_version: str
    e_history_version: str
    curriculum_version: str
    curriculum_fingerprint: str
    physical_support_version: str = STAGE25_PHYSICAL_SUPPORT_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "e_history_version",
            normalize_e_history_version(self.e_history_version),
        )
        for field_name in (
            "name",
            "version",
            "parameter_fingerprint",
            "observation_schema_version",
            "policy_schema_version",
            "curriculum_version",
            "curriculum_fingerprint",
            "physical_support_version",
        ):
            if not str(getattr(self, field_name)):
                raise ValueError(f"{field_name} must be non-empty")

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            self.to_json_dict(include_identity=False),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def identity_id(self) -> str:
        return f"{self.name}@{self.version}:{self.fingerprint[:12]}"

    def to_json_dict(self, *, include_identity: bool = True) -> dict[str, str]:
        result = {
            "name": self.name,
            "version": self.version,
            "parameter_fingerprint": self.parameter_fingerprint,
            "observation_schema_version": self.observation_schema_version,
            "policy_schema_version": self.policy_schema_version,
            "e_history_version": self.e_history_version,
            "curriculum_version": self.curriculum_version,
            "curriculum_fingerprint": self.curriculum_fingerprint,
            "physical_support_version": self.physical_support_version,
        }
        if include_identity:
            result["behavior_fingerprint"] = self.fingerprint
            result["identity_id"] = self.identity_id()
        return result


@dataclass(frozen=True)
class Stage25PolicyOutputs:
    """Batched nine-head policy result returned by the parent inference owner."""

    classes: np.ndarray
    component_logprobs: np.ndarray
    joint_logprob: np.ndarray
    value: np.ndarray
    decoded_goals: np.ndarray
    valid: np.ndarray
    policy_identity: Stage25BehaviorIdentity
    batch_size: int

    def __post_init__(self) -> None:
        batch = int(self.batch_size)
        expected = {
            "classes": ((batch, 9), np.dtype(np.int16)),
            "component_logprobs": ((batch, 9), np.dtype(np.float32)),
            "joint_logprob": ((batch,), np.dtype(np.float32)),
            "value": ((batch,), np.dtype(np.float32)),
            "decoded_goals": ((batch, 5), np.dtype(np.int16)),
            "valid": ((batch,), np.dtype(np.bool_)),
        }
        for name, (shape, dtype) in expected.items():
            array = np.asarray(getattr(self, name))
            if array.shape != shape or array.dtype != dtype:
                raise ValueError(
                    f"{name} must have shape {shape} and dtype {dtype}, got "
                    f"{array.shape} and {array.dtype}"
                )


Stage25Inputs = Mapping[str, np.ndarray]

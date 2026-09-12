"""Strict, framework-free Stage 2.5 manager-day trajectories.

Stage 2.5 rollout inference is owned by the parent process.  This module is
the persistence boundary for the resulting rows and deliberately imports no
JAX, Torch, executor, or runner code.  The format is a versioned numeric NPZ
plus a JSON sidecar; arbitrary Python objects are never serialized.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from bc_manager.constants import (
    BOARD_SIZE,
    CROP_ORDER,
    PRODUCT_ORDER,
    QUADRANT_ORDER,
    RESOURCE_ORDER,
    SHOP_VOCAB,
)
from bc_manager.model_spec import BOARD_BOOL_FIELDS, BOARD_NUMERIC_FIELDS
from rl_manager.stage25_mechanics import ACTION_CLASS_COUNTS, ACTION_ORDER, ACTION_SCHEMA_VERSION
from rl_manager.stage25_types import Stage25BehaviorIdentity, Stage25PolicyOutputs


STAGE25_TRAJECTORY_SCHEMA_VERSION = "stage25_trajectory_v1"
TRAJECTORY_SCHEMA_VERSION = STAGE25_TRAJECTORY_SCHEMA_VERSION
STAGE25_OBSERVATION_SCHEMA_VERSION = "stage25_corrected_e_own_only_v1"
STAGE25_PERSISTENT_LEDGER_VERSION = "stage25_crop_capacity_ledger_v1"
STAGE25_PHYSICAL_SUPPORT_VERSION = ACTION_SCHEMA_VERSION
STAGE25_MANAGER_START_DAY = 4
STAGE25_MANAGER_END_DAY = 29
STAGE25_ACTION_COUNT = 9

_PROVENANCE_KEYS = ("policy", "curriculum", "history", "physical", "executor")
_ROW_SCALAR_FIELDS = {
    "episode_id": (np.dtype(np.int64), ()),
    "seed": (np.dtype(np.int64), ()),
    "seat": (np.dtype(np.int8), ()),
    "day": (np.dtype(np.int16), ()),
    "row_index": (np.dtype(np.int64), ()),
    "classes": (np.dtype(np.int16), (STAGE25_ACTION_COUNT,)),
    "component_logprobs": (np.dtype(np.float32), (STAGE25_ACTION_COUNT,)),
    "joint_logprob": (np.dtype(np.float32), ()),
    "value": (np.dtype(np.float32), ()),
    "reward": (np.dtype(np.float32), ()),
    "terminated": (np.dtype(np.uint8), ()),
    "truncated": (np.dtype(np.uint8), ()),
    "bootstrap_value": (np.dtype(np.float32), ()),
    "valid": (np.dtype(np.uint8), ()),
    "reward_patched": (np.dtype(np.uint8), ()),
    "bootstrap_patched": (np.dtype(np.uint8), ()),
    "trainable": (np.dtype(np.uint8), ()),
}


def stage25_input_spec() -> dict[str, tuple[tuple[int, ...], np.dtype]]:
    """Return the exact persisted, pre-decision corrected-E input schema."""
    return {
        "board_kind": ((BOARD_SIZE,), np.dtype(np.int16)),
        "board_crop": ((BOARD_SIZE,), np.dtype(np.int8)),
        "board_animal": ((BOARD_SIZE,), np.dtype(np.int8)),
        "board_numeric": ((BOARD_SIZE, len(BOARD_NUMERIC_FIELDS)), np.dtype(np.float32)),
        "board_bool": ((BOARD_SIZE, len(BOARD_BOOL_FIELDS)), np.dtype(np.bool_)),
        "board_mask": ((BOARD_SIZE, 4), np.dtype(np.uint8)),
        "scalars": ((4,), np.dtype(np.float32)),
        "shed_counts": ((len(RESOURCE_ORDER),), np.dtype(np.int32)),
        "seed_counts": ((len(CROP_ORDER),), np.dtype(np.int32)),
        "carried_counts": ((len(RESOURCE_ORDER),), np.dtype(np.int32)),
        "unlocked": ((len(QUADRANT_ORDER),), np.dtype(np.uint8)),
        "market_inventory": ((len(PRODUCT_ORDER),), np.dtype(np.int32)),
        "market_prices": ((len(PRODUCT_ORDER),), np.dtype(np.float32)),
        "shop_counts": ((len(SHOP_VOCAB),), np.dtype(np.int32)),
        "day": ((), np.dtype(np.int16)),
        "days_remaining": ((), np.dtype(np.int16)),
        "economic_context": ((14,), np.dtype(np.float32)),
        "crop_capacity": ((len(CROP_ORDER),), np.dtype(np.int16)),
    }


INPUT_SPEC = stage25_input_spec()


def _require_exact_array(value: Any, shape: tuple[int, ...], dtype: np.dtype, name: str) -> np.ndarray:
    if shape == () and isinstance(value, np.generic):
        value = np.asarray(value)
    if not isinstance(value, np.ndarray):
        raise ValueError(f"{name} must be a numpy.ndarray with shape {shape} and dtype {dtype}")
    if value.dtype != dtype or value.shape not in (shape, (1,) + shape):
        raise ValueError(
            f"{name} must have shape {shape} or {(1,) + shape} and dtype {dtype}; "
            f"got {value.shape}/{value.dtype}"
        )
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    return value[0] if value.shape == (1,) + shape else value


def _require_scalar(value: Any, dtype: np.dtype, name: str) -> np.ndarray:
    if isinstance(value, np.generic):
        value = np.asarray(value)
    if not isinstance(value, np.ndarray) or value.shape != () or value.dtype != dtype:
        raise ValueError(f"{name} must be a scalar ndarray with dtype {dtype}; got {getattr(value, 'shape', None)}/{getattr(value, 'dtype', None)}")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    return value


def _require_patch_scalar(value: Any, name: str) -> np.ndarray:
    """Accept the ergonomic Python float spelling, but persist float32 only."""
    if isinstance(value, float):
        value = np.asarray(value, dtype=np.float32)
    return _require_scalar(value, np.dtype(np.float32), name)


def _finite(array: np.ndarray, name: str) -> None:
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or Inf")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not np.isfinite(value):
            raise ValueError("JSON metadata contains NaN or Inf")
        return value
    raise TypeError(f"metadata contains unsupported value {type(value).__name__}")


def _identity_json(identity: Stage25BehaviorIdentity, name: str) -> dict[str, str]:
    if not isinstance(identity, Stage25BehaviorIdentity):
        raise ValueError(f"{name} must be Stage25BehaviorIdentity")
    return identity.to_json_dict()


def _identity_from_json(value: Any, name: str) -> Stage25BehaviorIdentity:
    if not isinstance(value, Mapping):
        raise ValueError(f"sidecar {name} is missing or is not an object")
    fields = {
        "name", "version", "parameter_fingerprint", "observation_schema_version",
        "policy_schema_version", "e_history_version", "curriculum_version",
        "curriculum_fingerprint", "physical_support_version",
    }
    if not fields.issubset(value):
        raise ValueError(f"sidecar {name} is missing identity fields {sorted(fields - set(value))}")
    identity = Stage25BehaviorIdentity(**{key: str(value[key]) for key in fields})
    if value.get("behavior_fingerprint") != identity.fingerprint or value.get("identity_id") != identity.identity_id():
        raise ValueError(f"sidecar {name} fingerprint does not match identity fields")
    return identity


def _provenance(value: Mapping[str, Any], learner: Stage25BehaviorIdentity, opponent: Stage25BehaviorIdentity) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("provenance must be a mapping")
    result = dict(value)
    result.setdefault("policy", {"learner": learner.to_json_dict(), "opponent": opponent.to_json_dict()})
    result.setdefault("curriculum", {"learner": learner.curriculum_version, "opponent": opponent.curriculum_version})
    result.setdefault("history", {"learner": learner.e_history_version, "opponent": opponent.e_history_version})
    result.setdefault("physical", {"learner": learner.physical_support_version, "opponent": opponent.physical_support_version})
    executor = result.get("executor")
    if executor is None or executor == "" or (isinstance(executor, Mapping) and not executor):
        raise ValueError("provenance must include a non-empty executor identity")
    missing = sorted(set(_PROVENANCE_KEYS) - set(result))
    if missing:
        raise ValueError(f"provenance is missing identities {missing}")
    return _jsonable(result)


@dataclass(frozen=True)
class Stage25TrajectoryRow:
    """One exact pre-decision manager row, before terminal patching."""

    episode_id: int
    seat: int
    day: int
    inputs: Mapping[str, np.ndarray]
    classes: np.ndarray
    component_logprobs: np.ndarray
    joint_logprob: np.ndarray
    value: np.ndarray
    learner_identity: Stage25BehaviorIdentity
    opponent_identity: Stage25BehaviorIdentity
    provenance: Mapping[str, Any]
    seed: int = 0
    trainable: bool = True
    reward: np.ndarray = np.float32(0.0)
    terminated: bool = False
    truncated: bool = False
    bootstrap_value: np.ndarray = np.float32(0.0)
    reward_patched: bool = False
    bootstrap_patched: bool = False
    row_id: str | None = None
    valid: bool = True

    @property
    def episode_index(self) -> int:
        return self.episode_id


class Stage25TrajectoryBuffer:
    """Bounded append-only rows with explicit end-of-episode patching."""

    def __init__(self, capacity: int, *, input_spec: Mapping[str, Any] | None = None) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError(f"capacity must be a positive int, got {capacity!r}")
        if input_spec is not None and dict(input_spec) != INPUT_SPEC:
            raise ValueError("Stage 2.5 trajectory input_spec does not match the canonical pre-decision schema")
        self.capacity = capacity
        self._count = 0
        self._arrays: dict[str, np.ndarray] = {
            name: np.zeros((capacity,) + shape, dtype=dtype)
            for name, (dtype, shape) in _ROW_SCALAR_FIELDS.items()
        }
        self._arrays.update({f"input_{name}": np.zeros((capacity,) + shape, dtype=dtype)
                             for name, (shape, dtype) in INPUT_SPEC.items()})
        self._row_ids: list[str] = []
        self._learners: list[Stage25BehaviorIdentity] = []
        self._opponents: list[Stage25BehaviorIdentity] = []
        self._provenance_records: list[dict[str, Any]] = []
        self._keys: set[tuple[int, int, int]] = set()
        self._closed_outgoing: set[tuple[int, int, int]] = set()

    def __len__(self) -> int:
        return self._count

    @property
    def rows(self) -> tuple[Stage25TrajectoryRow, ...]:
        return tuple(self.iter_rows())

    def iter_rows(self):
        for index in range(self._count):
            inputs = {name: self._arrays[f"input_{name}"][index].copy() for name in INPUT_SPEC}
            yield Stage25TrajectoryRow(
                episode_id=int(self._arrays["episode_id"][index]),
                seed=int(self._arrays["seed"][index]),
                seat=int(self._arrays["seat"][index]), day=int(self._arrays["day"][index]),
                inputs=inputs, classes=self._arrays["classes"][index].copy(),
                component_logprobs=self._arrays["component_logprobs"][index].copy(),
                joint_logprob=self._arrays["joint_logprob"][index].copy(),
                value=self._arrays["value"][index].copy(),
                learner_identity=self._learners[index], opponent_identity=self._opponents[index],
                provenance=self._provenance_records[index],
                trainable=bool(self._arrays["trainable"][index]),
                reward=self._arrays["reward"][index].copy(),
                terminated=bool(self._arrays["terminated"][index]),
                truncated=bool(self._arrays["truncated"][index]),
                bootstrap_value=self._arrays["bootstrap_value"][index].copy(),
                reward_patched=bool(self._arrays["reward_patched"][index]),
                bootstrap_patched=bool(self._arrays["bootstrap_patched"][index]),
                row_id=self._row_ids[index],
            )

    def append(self, row: Stage25TrajectoryRow | None = None, **kwargs: Any) -> int:
        """Append a validated row and return its contiguous row index."""
        if row is None:
            outputs = kwargs.pop("outputs", kwargs.pop("policy_outputs", None))
            if outputs is not None:
                opponent = kwargs.pop("opponent_identity", None)
                if not isinstance(outputs, Stage25PolicyOutputs) or outputs.batch_size != 1:
                    raise ValueError("outputs must be a one-row Stage25PolicyOutputs")
                if opponent is None:
                    raise ValueError("opponent_identity is required with outputs")
                row = row_from_policy_outputs(
                    episode_id=kwargs.pop("episode_id"), seat=kwargs.pop("seat"),
                    day=kwargs.pop("day"), inputs=kwargs.pop("inputs"),
                    outputs=outputs, opponent_identity=opponent,
                    provenance=kwargs.pop("provenance"), row_id=kwargs.pop("row_id", None))
                if kwargs:
                    raise ValueError(f"unexpected append arguments {sorted(kwargs)}")
            else:
                if "episode_id" not in kwargs and "episode_index" in kwargs:
                    kwargs["episode_id"] = kwargs.pop("episode_index")
                row = Stage25TrajectoryRow(**kwargs)
        if not isinstance(row, Stage25TrajectoryRow):
            raise ValueError("append expects a Stage25TrajectoryRow")
        if self._count >= self.capacity:
            raise ValueError(f"trajectory buffer at explicit capacity {self.capacity}")
        if isinstance(row.episode_id, bool) or not isinstance(row.episode_id, (int, np.integer)):
            raise ValueError("episode_id must be an integer")
        if isinstance(row.seed, bool) or not isinstance(row.seed, (int, np.integer)):
            raise ValueError("seed must be an integer")
        if isinstance(row.seat, bool) or not isinstance(row.seat, (int, np.integer)) or not 0 <= int(row.seat) <= 127:
            raise ValueError("seat must be an integer in [0, 127]")
        if isinstance(row.day, bool) or not isinstance(row.day, (int, np.integer)) or not STAGE25_MANAGER_START_DAY <= int(row.day) <= STAGE25_MANAGER_END_DAY:
            raise ValueError(f"day must be in [{STAGE25_MANAGER_START_DAY}, {STAGE25_MANAGER_END_DAY}]")
        key = (int(row.episode_id), int(row.seat), int(row.day))
        if key in self._keys:
            raise ValueError(f"duplicate episode/seat/day identity {key}")
        if any(self._arrays["terminated"][i] or self._arrays["truncated"][i]
               for i in range(self._count)
               if (int(self._arrays["episode_id"][i]), int(self._arrays["seat"][i])) == key[:2]):
            raise ValueError("cannot append after an episode/seat end patch")
        if not isinstance(row.valid, (bool, np.bool_)) or not bool(row.valid):
            raise ValueError("invalid Stage 2.5 policy row cannot enter a trajectory")
        classes = _require_exact_array(row.classes, (STAGE25_ACTION_COUNT,), np.dtype(np.int16), "classes")
        if np.any(classes < 0) or np.any(classes >= np.asarray(ACTION_CLASS_COUNTS, dtype=np.int16)):
            raise ValueError("classes contain an out-of-vocabulary action")
        component = _require_exact_array(row.component_logprobs, (STAGE25_ACTION_COUNT,), np.dtype(np.float32), "component_logprobs")
        joint = _require_scalar(row.joint_logprob, np.dtype(np.float32), "joint_logprob")
        value = _require_scalar(row.value, np.dtype(np.float32), "value")
        _finite(component, "component_logprobs")
        _finite(joint, "joint_logprob")
        _finite(value, "value")
        if np.all(component == 0.0) and float(joint) == 0.0:
            raise ValueError("diagnostic zero likelihood placeholder cannot enter a trajectory")
        if not np.isclose(float(joint), float(np.sum(component, dtype=np.float32)), atol=1e-5, rtol=1e-5):
            raise ValueError("joint_logprob must equal the sum of component_logprobs")
        if not isinstance(row.inputs, Mapping) or set(row.inputs) != set(INPUT_SPEC):
            raise ValueError("inputs must contain exactly the canonical own-only keys plus crop_capacity")
        for name, (shape, dtype) in INPUT_SPEC.items():
            normalized = _require_exact_array(row.inputs[name], shape, dtype, f"input {name!r}")
            if np.issubdtype(dtype, np.floating):
                if name == "board_numeric":
                    if np.any(np.isinf(normalized)):
                        raise ValueError(f"input {name!r} contains Inf")
                else:
                    _finite(normalized, f"input {name!r}")
            self._arrays[f"input_{name}"][self._count] = normalized
        learner_json = _identity_json(row.learner_identity, "learner_identity")
        opponent_json = _identity_json(row.opponent_identity, "opponent_identity")
        del learner_json, opponent_json
        provenance = _provenance(row.provenance, row.learner_identity, row.opponent_identity)
        row_id = row.row_id if row.row_id is not None else f"episode={int(row.episode_id)}/seat={int(row.seat)}/day={int(row.day)}"
        if not isinstance(row_id, str) or not row_id:
            raise ValueError("row_id must be a non-empty string")
        if row_id in self._row_ids:
            raise ValueError(f"duplicate row_id {row_id!r}")
        index = self._count
        self._arrays["episode_id"][index] = int(row.episode_id)
        self._arrays["seed"][index] = int(row.seed)
        self._arrays["seat"][index] = int(row.seat)
        self._arrays["day"][index] = int(row.day)
        self._arrays["row_index"][index] = index
        self._arrays["classes"][index] = classes
        self._arrays["component_logprobs"][index] = component
        self._arrays["joint_logprob"][index] = joint
        self._arrays["value"][index] = value
        self._arrays["valid"][index] = 1
        if not isinstance(row.trainable, (bool, np.bool_)):
            raise ValueError("trainable must be boolean")
        self._arrays["trainable"][index] = int(bool(row.trainable))
        reward = _require_patch_scalar(row.reward, "reward")
        bootstrap_value = _require_patch_scalar(
            row.bootstrap_value, "bootstrap_value")
        _finite(reward, "reward")
        _finite(bootstrap_value, "bootstrap_value")
        if not isinstance(row.terminated, (bool, np.bool_)) or not isinstance(
                row.truncated, (bool, np.bool_)):
            raise ValueError("terminated and truncated must be boolean")
        if bool(row.terminated) and bool(row.truncated):
            raise ValueError("a row cannot be both terminated and truncated")
        if not isinstance(row.reward_patched, (bool, np.bool_)) or not isinstance(
                row.bootstrap_patched, (bool, np.bool_)):
            raise ValueError("patch flags must be boolean")
        if bool(row.reward_patched) != bool(row.terminated):
            raise ValueError("reward_patched must match terminated")
        if bool(row.truncated) and not bool(row.bootstrap_patched):
            raise ValueError("truncated rows require bootstrap_patched")
        if bool(row.terminated) and bool(row.bootstrap_patched):
            raise ValueError("terminated rows cannot have bootstrap_patched")
        if not bool(row.terminated) and float(reward) != 0.0:
            raise ValueError("unpatched rows must have zero reward")
        if not bool(row.bootstrap_patched) and float(bootstrap_value) != 0.0:
            raise ValueError("unpatched rows must have zero bootstrap_value")
        self._arrays["reward"][index] = reward
        self._arrays["terminated"][index] = int(bool(row.terminated))
        self._arrays["truncated"][index] = int(bool(row.truncated))
        self._arrays["bootstrap_value"][index] = bootstrap_value
        self._arrays["reward_patched"][index] = int(bool(row.reward_patched))
        self._arrays["bootstrap_patched"][index] = int(bool(row.bootstrap_patched))
        self._row_ids.append(row_id)
        self._learners.append(row.learner_identity)
        self._opponents.append(row.opponent_identity)
        self._provenance_records.append(provenance)
        self._keys.add(key)
        self._count += 1
        return index

    def _require_index(self, index: int) -> None:
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < self._count:
            raise ValueError(f"index {index!r} outside filled range [0, {self._count})")

    def close_outgoing(
        self, *, episode_index: int, seat: int, next_day: int,
        next_inputs: Mapping[str, np.ndarray], next_crop_capacity: np.ndarray,
        terminal: bool = False, truncated: bool = False,
    ) -> None:
        """Close the prior manager transition before its successor is added.

        The successor row carries the next predictive inputs and K, so this
        boundary records only the lifecycle invariant and validates that
        successor payload. Terminal closure remains the explicit patch below.
        """
        if terminal or truncated:
            raise ValueError("non-terminal close_outgoing cannot be terminal")
        if not isinstance(next_inputs, Mapping) or set(next_inputs) != set(INPUT_SPEC):
            raise ValueError("next_inputs must match the Stage 2.5 input schema")
        for name, (shape, dtype) in INPUT_SPEC.items():
            value = next_inputs[name]
            if shape == () and isinstance(value, np.generic):
                value = np.asarray(value)
            normalized = _require_exact_array(value, shape, dtype,
                                              f"next input {name!r}")
            if np.issubdtype(dtype, np.floating):
                if name == "board_numeric":
                    if np.any(np.isinf(normalized)):
                        raise ValueError("next board_numeric contains Inf")
                else:
                    _finite(normalized, f"next input {name!r}")
        capacity = np.asarray(next_crop_capacity)
        if capacity.shape != (5,) or capacity.dtype != np.dtype(np.int16):
            raise ValueError("next_crop_capacity must be int16 [5]")
        key = (int(episode_index), int(seat), int(next_day))
        previous = [candidate for candidate in self._keys
                    if candidate[:2] == key[:2] and candidate[2] < key[2]]
        if previous:
            self._closed_outgoing.add(max(previous, key=lambda item: item[2]))

    def _require_open(self, index: int) -> None:
        self._require_index(index)
        if self._arrays["terminated"][index] or self._arrays["truncated"][index]:
            raise ValueError(f"row {index} already has a terminal/truncation patch")

    def patch_terminal(self, index: int, reward: Any, terminated: bool = True) -> None:
        self._require_open(index)
        if not isinstance(terminated, (bool, np.bool_)) or not bool(terminated):
            raise ValueError("patch_terminal requires terminated=True")
        reward = _require_patch_scalar(reward, "reward")
        _finite(reward, "reward")
        episode_seat = (int(self._arrays["episode_id"][index]), int(self._arrays["seat"][index]))
        if any(i > index and
               (int(self._arrays["episode_id"][i]), int(self._arrays["seat"][i])) == episode_seat
               for i in range(self._count)):
            raise ValueError("terminal patch must be on the final manager row for the episode/seat")
        if any(self._arrays["terminated"][i] or self._arrays["truncated"][i]
               for i in range(self._count) if i != index and
               (int(self._arrays["episode_id"][i]), int(self._arrays["seat"][i])) == episode_seat):
            raise ValueError("episode/seat already has an end patch")
        self._arrays["reward"][index] = reward
        self._arrays["terminated"][index] = 1
        self._arrays["reward_patched"][index] = 1

    def patch_truncated(self, index: int, bootstrap_value: Any = None) -> None:
        self._require_open(index)
        episode_seat = (int(self._arrays["episode_id"][index]), int(self._arrays["seat"][index]))
        if any(i > index and
               (int(self._arrays["episode_id"][i]), int(self._arrays["seat"][i])) == episode_seat
               for i in range(self._count)):
            raise ValueError("truncation patch must be on the final manager row for the episode/seat")
        if any(self._arrays["terminated"][i] or self._arrays["truncated"][i]
               for i in range(self._count) if i != index and
               (int(self._arrays["episode_id"][i]), int(self._arrays["seat"][i])) == episode_seat):
            raise ValueError("episode/seat already has an end patch")
        self._arrays["truncated"][index] = 1
        if bootstrap_value is not None:
            self.patch_bootstrap(index, bootstrap_value)

    patch_truncation = patch_truncated

    def patch_bootstrap(self, index: int, bootstrap_value: Any) -> None:
        self._require_index(index)
        if self._arrays["terminated"][index]:
            raise ValueError("terminated rows cannot receive a bootstrap value")
        bootstrap_value = _require_patch_scalar(bootstrap_value, "bootstrap_value")
        _finite(bootstrap_value, "bootstrap_value")
        self._arrays["bootstrap_value"][index] = bootstrap_value
        self._arrays["bootstrap_patched"][index] = 1

    def finalize(self) -> dict[str, np.ndarray]:
        result = {}
        for name, array in self._arrays.items():
            value = np.asarray(array[:self._count]).copy()
            value.setflags(write=False)
            result[name] = value
        return result

    def save(self, path: str | Path, *, run_metadata: Mapping[str, Any] | None = None) -> Path:
        if any(self._arrays["truncated"][i] and not self._arrays["bootstrap_patched"][i] for i in range(self._count)):
            raise ValueError("every truncated row requires a patched bootstrap_value")
        base = Path(path)
        base.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {
            "schema_version": np.frombuffer(STAGE25_TRAJECTORY_SCHEMA_VERSION.encode("utf-8"), dtype=np.uint8),
            "count": np.asarray(self._count, dtype=np.int64),
            "capacity": np.asarray(self.capacity, dtype=np.int64),
        }
        arrays.update(self.finalize())
        np.savez_compressed(str(base) + ".npz", **arrays)
        sidecar = {
            "schema_version": STAGE25_TRAJECTORY_SCHEMA_VERSION,
            "npz_schema_version": STAGE25_TRAJECTORY_SCHEMA_VERSION,
            "count": self._count, "capacity": self.capacity,
            "observation_schema_version": STAGE25_OBSERVATION_SCHEMA_VERSION,
            "persistent_ledger_version": STAGE25_PERSISTENT_LEDGER_VERSION,
            "physical_support_version": STAGE25_PHYSICAL_SUPPORT_VERSION,
            "action_schema_version": ACTION_SCHEMA_VERSION,
            "action_order": list(ACTION_ORDER),
            "action_class_counts": list(ACTION_CLASS_COUNTS),
            "input_schema": {name: {"shape": list(shape), "dtype": dtype.str}
                             for name, (shape, dtype) in INPUT_SPEC.items()},
            "run_metadata": _jsonable(dict(run_metadata or {})),
            "rows": [
                {"index": i, "row_id": self._row_ids[i],
                 "episode_id": int(self._arrays["episode_id"][i]),
                 "seed": int(self._arrays["seed"][i]),
                 "seat": int(self._arrays["seat"][i]), "day": int(self._arrays["day"][i]),
                 "trainable": bool(self._arrays["trainable"][i]),
                 "learner_identity": self._learners[i].to_json_dict(),
                 "opponent_identity": self._opponents[i].to_json_dict(),
                 "provenance": self._provenance_records[i]}
                for i in range(self._count)
            ],
        }
        Path(str(base) + ".json").write_text(
            json.dumps(sidecar, sort_keys=True, indent=1, allow_nan=False), encoding="utf-8")
        return base

    @classmethod
    def load(cls, path: str | Path) -> tuple["Stage25TrajectoryBuffer", dict[str, Any]]:
        base = Path(path)
        try:
            with np.load(str(base) + ".npz", allow_pickle=False) as data:
                expected_version = np.frombuffer(STAGE25_TRAJECTORY_SCHEMA_VERSION.encode("utf-8"), dtype=np.uint8)
                if "schema_version" not in data or data["schema_version"].dtype != np.dtype(np.uint8) or not np.array_equal(data["schema_version"], expected_version):
                    raise ValueError("unsupported Stage 2.5 trajectory schema version")
                for key, (dtype, shape) in {"count": (np.dtype(np.int64), ()), "capacity": (np.dtype(np.int64), ())}.items():
                    if key not in data or data[key].dtype != dtype or data[key].shape != shape:
                        raise ValueError(f"npz {key} has invalid shape/dtype")
                count, capacity = int(data["count"]), int(data["capacity"])
                if capacity < 1 or count < 0 or count > capacity:
                    raise ValueError("invalid trajectory count/capacity")
                expected_keys = {"schema_version", "count", "capacity", * _ROW_SCALAR_FIELDS,
                                 *(f"input_{name}" for name in INPUT_SPEC)}
                if set(data.files) != expected_keys:
                    missing = sorted(expected_keys - set(data.files))
                    extra = sorted(set(data.files) - expected_keys)
                    raise ValueError(f"npz schema keys mismatch; missing={missing}, extra={extra}")
                buffer = cls(capacity)
                for name, (dtype, shape) in _ROW_SCALAR_FIELDS.items():
                    loaded = data[name]
                    expected_shape = (count,) + shape
                    if loaded.dtype != dtype or loaded.shape != expected_shape:
                        raise ValueError(f"npz array {name!r} mismatch: {loaded.shape}/{loaded.dtype}")
                    buffer._arrays[name][:count] = loaded
                for name, (shape, dtype) in INPUT_SPEC.items():
                    loaded = data[f"input_{name}"]
                    if loaded.dtype != dtype or loaded.shape != (count,) + shape:
                        raise ValueError(f"npz input {name!r} mismatch: {loaded.shape}/{loaded.dtype}")
                    buffer._arrays[f"input_{name}"][:count] = loaded
                buffer._count = count
                classes = buffer._arrays["classes"][:count]
                if np.any(classes < 0) or np.any(classes >= np.asarray(ACTION_CLASS_COUNTS, dtype=np.int16)):
                    raise ValueError("persisted classes contain an out-of-vocabulary action")
                component = buffer._arrays["component_logprobs"][:count]
                joint = buffer._arrays["joint_logprob"][:count]
                if not np.all(np.isfinite(component)) or not np.all(np.isfinite(joint)):
                    raise ValueError("persisted likelihoods contain NaN or Inf")
                if np.any(np.all(component == 0.0, axis=1) & (joint == 0.0)):
                    raise ValueError("persisted diagnostic zero likelihood placeholder")
                if not np.allclose(joint, np.sum(component, axis=1, dtype=np.float32), atol=1e-5, rtol=1e-5):
                    raise ValueError("persisted joint_logprob does not equal component sum")
                for name in ("value", "reward", "bootstrap_value"):
                    if not np.all(np.isfinite(buffer._arrays[name][:count])):
                        raise ValueError(f"persisted {name} contains NaN or Inf")
                for name in ("valid", "terminated", "truncated", "reward_patched", "bootstrap_patched", "trainable"):
                    if np.any(~np.isin(buffer._arrays[name][:count], (0, 1))):
                        raise ValueError(f"persisted {name} is not boolean-valued")
                for name, (shape, dtype) in INPUT_SPEC.items():
                    if (name == "board_numeric" and
                            np.any(np.isinf(buffer._arrays[f"input_{name}"][:count]))):
                        raise ValueError(f"persisted input {name!r} contains Inf")
                    if (name != "board_numeric" and
                            np.issubdtype(dtype, np.floating) and
                            not np.all(np.isfinite(buffer._arrays[f"input_{name}"][:count]))):
                        raise ValueError(f"persisted input {name!r} contains NaN or Inf")
                if np.any(buffer._arrays["day"][:count] < STAGE25_MANAGER_START_DAY) or np.any(buffer._arrays["day"][:count] > STAGE25_MANAGER_END_DAY):
                    raise ValueError("persisted row contains a pre-opening or out-of-range manager day")
                if np.any(buffer._arrays["terminated"][:count] & buffer._arrays["truncated"][:count]):
                    raise ValueError("persisted row cannot be both terminated and truncated")
                if np.any(buffer._arrays["terminated"][:count] & (1 - buffer._arrays["reward_patched"][:count])):
                    raise ValueError("persisted terminated row is missing reward patch")
                if np.any(buffer._arrays["truncated"][:count] & (1 - buffer._arrays["bootstrap_patched"][:count])):
                    raise ValueError("persisted truncated row is missing bootstrap patch")
                if np.any(buffer._arrays["terminated"][:count] & buffer._arrays["bootstrap_patched"][:count]):
                    raise ValueError("persisted terminated row has a bootstrap patch")
                if np.any((1 - buffer._arrays["terminated"][:count]) & buffer._arrays["reward_patched"][:count]):
                    raise ValueError("persisted non-terminal row has a reward patch")
                if np.any(buffer._arrays["terminated"][:count] & (buffer._arrays["bootstrap_value"][:count] != 0.0)):
                    raise ValueError("persisted terminated row has nonzero bootstrap value")
                if not np.array_equal(buffer._arrays["row_index"][:count], np.arange(count, dtype=np.int64)):
                    raise ValueError("persisted row_index values are not contiguous")
        except (KeyError, OSError, ValueError) as exc:
            if isinstance(exc, ValueError):
                raise
            raise ValueError(f"invalid Stage 2.5 trajectory NPZ: {exc}") from exc
        try:
            sidecar = json.loads(Path(str(base) + ".json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid Stage 2.5 trajectory JSON sidecar: {exc}") from exc
        if not isinstance(sidecar, Mapping):
            raise ValueError("trajectory JSON sidecar must be an object")
        if sidecar.get("schema_version") != STAGE25_TRAJECTORY_SCHEMA_VERSION or sidecar.get("npz_schema_version") != STAGE25_TRAJECTORY_SCHEMA_VERSION:
            raise ValueError("sidecar/NPZ schema version mismatch")
        if sidecar.get("count") != count or sidecar.get("capacity") != capacity:
            raise ValueError("sidecar count/capacity mismatch")
        if sidecar.get("observation_schema_version") != STAGE25_OBSERVATION_SCHEMA_VERSION \
                or sidecar.get("persistent_ledger_version") != STAGE25_PERSISTENT_LEDGER_VERSION \
                or sidecar.get("physical_support_version") != STAGE25_PHYSICAL_SUPPORT_VERSION \
                or sidecar.get("action_schema_version") != ACTION_SCHEMA_VERSION \
                or sidecar.get("action_order") != list(ACTION_ORDER) \
                or sidecar.get("action_class_counts") != list(ACTION_CLASS_COUNTS):
            raise ValueError("sidecar Stage 2.5 contract version mismatch")
        if sidecar.get("input_schema") != {name: {"shape": list(shape), "dtype": dtype.str} for name, (shape, dtype) in INPUT_SPEC.items()}:
            raise ValueError("sidecar input schema mismatch")
        rows = sidecar.get("rows")
        if not isinstance(rows, list) or len(rows) != count:
            raise ValueError("sidecar row count mismatch")
        for i, record in enumerate(rows):
            if not isinstance(record, Mapping) or record.get("index") != i:
                raise ValueError("sidecar row indices are not contiguous")
            if (record.get("episode_id"), record.get("seed"), record.get("seat"), record.get("day")) != (
                    int(buffer._arrays["episode_id"][i]), int(buffer._arrays["seed"][i]),
                    int(buffer._arrays["seat"][i]), int(buffer._arrays["day"][i])):
                raise ValueError(f"sidecar row identity mismatch at index {i}")
            if not isinstance(record.get("trainable"), bool):
                raise ValueError("sidecar trainable must be boolean")
            if int(record["trainable"]) != int(buffer._arrays["trainable"][i]):
                raise ValueError(f"sidecar trainable mismatch at index {i}")
            buffer._row_ids.append(record.get("row_id"))
            if not isinstance(buffer._row_ids[-1], str) or not buffer._row_ids[-1]:
                raise ValueError("sidecar row_id must be a non-empty string")
            buffer._learners.append(_identity_from_json(record.get("learner_identity"), "learner_identity"))
            buffer._opponents.append(_identity_from_json(record.get("opponent_identity"), "opponent_identity"))
            buffer._provenance_records.append(_provenance(record.get("provenance"), buffer._learners[-1], buffer._opponents[-1]))
            key = (int(buffer._arrays["episode_id"][i]), int(buffer._arrays["seat"][i]), int(buffer._arrays["day"][i]))
            if key in buffer._keys or buffer._row_ids[-1] in buffer._row_ids[:-1]:
                raise ValueError("duplicate persisted row identity")
            if (buffer._arrays["terminated"][i] or buffer._arrays["truncated"][i]) \
                    and any((int(buffer._arrays["episode_id"][j]), int(buffer._arrays["seat"][j])) == key[:2]
                            and (buffer._arrays["terminated"][j] or buffer._arrays["truncated"][j])
                            for j in range(i)):
                raise ValueError("episode/seat has more than one end patch")
            buffer._keys.add(key)
        if np.any(buffer._arrays["valid"][:count] != 1):
            raise ValueError("invalid rows are not loadable trajectories")
        if np.any(buffer._arrays["terminated"] & buffer._arrays["truncated"]):
            raise ValueError("row cannot be both terminated and truncated")
        if np.any(buffer._arrays["truncated"] & (1 - buffer._arrays["bootstrap_patched"])):
            raise ValueError("truncated row is missing bootstrap patch")
        return buffer, sidecar


def load_stage25_trajectory(path: str | Path) -> tuple[Stage25TrajectoryBuffer, dict[str, Any]]:
    return Stage25TrajectoryBuffer.load(path)


def row_from_policy_outputs(
    *, episode_id: int, seat: int, day: int, inputs: Mapping[str, np.ndarray],
    outputs: Stage25PolicyOutputs, opponent_identity: Stage25BehaviorIdentity,
    provenance: Mapping[str, Any], row_id: str | None = None,
    seed: int = 0, trainable: bool = True,
) -> Stage25TrajectoryRow:
    """Convert a one-row parent inference result into a persistence row."""
    if not isinstance(outputs, Stage25PolicyOutputs) or outputs.batch_size != 1:
        raise ValueError("outputs must be a one-row Stage25PolicyOutputs")
    if not bool(outputs.valid[0]):
        raise ValueError("invalid policy output cannot enter a trajectory")
    return Stage25TrajectoryRow(
        episode_id=episode_id, seat=seat, day=day, inputs=inputs,
        classes=outputs.classes[0], component_logprobs=outputs.component_logprobs[0],
        joint_logprob=outputs.joint_logprob[0], value=outputs.value[0],
        learner_identity=outputs.policy_identity, opponent_identity=opponent_identity,
        provenance=provenance, seed=seed, trainable=trainable, row_id=row_id,
    )


@dataclass(frozen=True)
class Stage25TeacherForceAudit:
    rows_checked: int
    mismatches: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.mismatches


def audit_teacher_force(
    rows: Stage25TrajectoryBuffer | Sequence[Stage25TrajectoryRow],
    teacher_force: Callable[[Mapping[str, np.ndarray], np.ndarray], Any],
    *, atol: float = 1e-5,
) -> Stage25TeacherForceAudit:
    """Audit stored likelihoods through a caller-supplied unchanged-weight evaluator.

    The callback is intentionally dependency-injected: callers can close over
    native JAX parameters, while this module remains importable in CPU workers
    that have no JAX installation.  It must return a Stage25PolicyOutputs or a
    mapping containing ``classes``, ``component_logprobs``, ``joint_logprob``,
    and ``value`` for one row.
    """
    items = rows.iter_rows() if isinstance(rows, Stage25TrajectoryBuffer) else iter(rows)
    mismatches: list[str] = []
    checked = 0
    for checked, row in enumerate(items, start=1):
        batched_inputs = {name: np.asarray(value)[None, ...]
                          for name, value in row.inputs.items()}
        result = teacher_force(batched_inputs, row.classes[None, :])
        if isinstance(result, Stage25PolicyOutputs):
            if result.policy_identity != row.learner_identity:
                mismatches.append(f"row {checked - 1}: policy identity changed")
            result = {name: getattr(result, name) for name in ("classes", "component_logprobs", "joint_logprob", "value")}
        if not isinstance(result, Mapping):
            mismatches.append(f"row {checked - 1}: evaluator did not return a mapping")
            continue
        try:
            classes = np.asarray(result["classes"])[0]
            component = np.asarray(result["component_logprobs"])[0]
            joint = float(np.asarray(result["joint_logprob"])[0])
            value = float(np.asarray(result["value"])[0])
            if not np.array_equal(classes, row.classes):
                mismatches.append(f"row {checked - 1}: classes changed")
            if not np.allclose(component, row.component_logprobs, atol=atol, rtol=atol):
                mismatches.append(f"row {checked - 1}: component_logprobs changed")
            if not np.isclose(joint, float(row.joint_logprob), atol=atol, rtol=atol):
                mismatches.append(f"row {checked - 1}: joint_logprob changed")
            if not np.isclose(value, float(row.value), atol=atol, rtol=atol):
                mismatches.append(f"row {checked - 1}: value changed")
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            mismatches.append(f"row {checked - 1}: malformed evaluator output ({exc})")
    return Stage25TeacherForceAudit(checked, tuple(mismatches))


audit_teacher_forced_rows = audit_teacher_force


__all__ = [
    "INPUT_SPEC", "STAGE25_ACTION_COUNT", "STAGE25_MANAGER_END_DAY",
    "STAGE25_MANAGER_START_DAY", "STAGE25_OBSERVATION_SCHEMA_VERSION",
    "STAGE25_PERSISTENT_LEDGER_VERSION", "STAGE25_PHYSICAL_SUPPORT_VERSION",
    "STAGE25_TRAJECTORY_SCHEMA_VERSION", "TRAJECTORY_SCHEMA_VERSION",
    "Stage25TeacherForceAudit", "Stage25TrajectoryBuffer", "Stage25TrajectoryRow",
    "audit_teacher_force", "audit_teacher_forced_rows", "load_stage25_trajectory",
    "row_from_policy_outputs", "stage25_input_spec",
]

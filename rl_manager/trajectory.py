"""Compact manager-day trajectory buffer + strict NPZ serialization (issue #9 A1).

One transition per manager decision/day/seat (d4..d29). The training buffer
is a set of preallocated contiguous NumPy arrays with explicit capacity
(append fails loudly on overflow); rich diagnostics live in an optional JSON
sidecar so the training arrays stay compact. Round-trip preserves every
array, action, reward, and metadata record.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from bc_manager.live import encode_live_inputs
from bc_manager.economics import (
    E_HISTORY_CORRECTED_V1,
    E_HISTORY_LEGACY,
    normalize_e_history_version,
)

from rl_manager.decode import ACTION_TENSOR_SHAPES, LOGPROB_GROUPS

TRAJECTORY_SCHEMA_VERSION = 2
_LEGACY_TRAJECTORY_SCHEMA_VERSION = 1

_TRACE_DIGEST_LEN = 32  # sha256 raw bytes


@dataclass(frozen=True)
class Transition:
    """One manager decision/day for one seat (pre-terminal patching)."""

    episode_index: int
    seed: int
    seat: int
    day: int
    trainable: bool
    inputs: Mapping[str, np.ndarray]  # own-only E model-facing arrays, [1, ...]
    action_tensors: Mapping[str, np.ndarray]
    logprob_groups: Mapping[str, np.ndarray]
    logprob_total: float
    value: float
    trace_digest: bytes  # patched after the day's primitive turns complete
    truncated: bool = False


@dataclass
class TransitionMetadata:
    """Sidecar metadata aligned 1:1 with buffer rows (never model input)."""

    index: int
    episode_index: int
    seed: int
    seat: int
    day: int
    policy_id: str
    policy_version: str
    policy_fingerprint: str
    opponent_id: str
    trainable: bool
    plan_json: dict[str, Any]
    executor_day_diagnostics: dict[str, Any] = field(default_factory=dict)
    trace_digest_hex: str | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "episode_index": self.episode_index,
            "seed": self.seed,
            "seat": self.seat,
            "day": self.day,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "policy_fingerprint": self.policy_fingerprint,
            "opponent_id": self.opponent_id,
            "trainable": self.trainable,
            "plan_json": self.plan_json,
            "executor_day_diagnostics": self.executor_day_diagnostics,
            "trace_digest_hex": self.trace_digest_hex,
        }


_SCALAR_FIELDS: dict[str, tuple[type, tuple]] = {
    "episode_index": (np.int32, ()),
    "seed": (np.int64, ()),
    "seat": (np.int8, ()),
    "day": (np.int16, ()),
    "trainable": (np.uint8, ()),
    "reward": (np.float32, ()),
    "terminated": (np.uint8, ()),
    "truncated": (np.uint8, ()),
    "valid": (np.uint8, ()),
    "logprob_total": (np.float32, ()),
    "value": (np.float32, ()),
}

_ACTION_DTYPES = {
    "crop": np.int16, "animal": np.int16, "land": np.int16,
    "fertilizer": np.int16, "care": np.int16,
    "sell_presence": np.uint8, "sell_quantity": np.int16,
}


class TrajectoryBuffer:
    """Preallocated compact trajectory storage with strict schema."""

    def __init__(
        self,
        capacity: int,
        input_spec: Mapping[str, tuple[int, ...] | tuple[tuple[int, ...], type]],
        *,
        e_history_version: str = E_HISTORY_CORRECTED_V1,
    ) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) \
                or capacity < 1:
            raise ValueError(f"capacity must be a positive int, got {capacity!r}")
        self.capacity = int(capacity)
        self.e_history_version = normalize_e_history_version(e_history_version)
        self._count = 0
        self._arrays: dict[str, np.ndarray] = {}

        for name, (dtype, shape) in _SCALAR_FIELDS.items():
            self._arrays[name] = np.zeros((capacity,) + shape, dtype=dtype)
        for group in LOGPROB_GROUPS:
            self._arrays[f"logprob_{group}"] = np.zeros(
                (capacity,), dtype=np.float32)
        for name, shape in ACTION_TENSOR_SHAPES.items():
            self._arrays[f"action_{name}"] = np.zeros(
                (capacity,) + shape, dtype=_ACTION_DTYPES[name])
        self._arrays["trace_digest"] = np.zeros(
            (capacity, _TRACE_DIGEST_LEN), dtype=np.uint8)

        # Model-facing input arrays: exact shapes/dtypes pinned by the
        # canonical own-only E spec derived from the authoritative encoder.
        canonical = e_input_spec()
        if set(input_spec.keys()) != set(canonical.keys()):
            raise ValueError(
                f"input_spec keys {sorted(input_spec.keys())} != canonical "
                f"own-only E spec keys {sorted(canonical.keys())}; the strict "
                f"trajectory schema is pinned to the authoritative encoder")
        self.input_spec: dict[str, tuple[tuple[int, ...], type]] = {}
        for key, spec in input_spec.items():
            if not (isinstance(spec, tuple) and len(spec) == 2
                    and isinstance(spec[0], tuple)):
                raise ValueError(
                    f"input_spec[{key!r}] must be ((dim, ...), dtype), got "
                    f"{spec!r}")
            shape, dtype = spec
            expected_shape, expected_dtype = canonical[key]
            if tuple(shape) != tuple(expected_shape) \
                    or dtype is not expected_dtype:
                raise ValueError(
                    f"input_spec[{key!r}] must be "
                    f"{(tuple(expected_shape), expected_dtype)} per the "
                    f"canonical E encoder, got {spec!r}")
            self.input_spec[key] = (tuple(shape), dtype)
        for key, (shape, dtype) in self.input_spec.items():
            self._arrays[f"input_{key}"] = np.zeros((capacity,) + shape,
                                                    dtype=dtype)

        self.sidecar_records: list[TransitionMetadata] = []

    # ------------------------------------------------------------- append
    def __len__(self) -> int:
        return self._count

    def append(self, transition: Transition,
               metadata: TransitionMetadata) -> None:
        if self._count >= self.capacity:
            raise ValueError(
                f"trajectory buffer at explicit capacity {self.capacity}; "
                f"refusing to append transition {self._count} (fail loud, "
                f"never silently truncate)")
        index = self._count
        expected_keys = set(self.input_spec.keys())
        actual_keys = set(transition.inputs.keys())
        if actual_keys != expected_keys:
            raise ValueError(
                f"transition input keys mismatch; missing="
                f"{sorted(expected_keys - actual_keys)}, unexpected="
                f"{sorted(actual_keys - expected_keys)}")
        for key, (shape, dtype) in self.input_spec.items():
            value = np.asarray(transition.inputs[key])
            if value.shape != (1,) + shape:
                raise ValueError(
                    f"input {key!r} must have shape {(1,) + shape}, got "
                    f"{value.shape}")
            # Drop the leading batch row before storing: scalar-shaped
            # inputs land in a 0-d slot and must be assigned as scalars.
            self._arrays[f"input_{key}"][index] = \
                value.astype(dtype).reshape(shape)
        for name, shape in ACTION_TENSOR_SHAPES.items():
            value = np.asarray(transition.action_tensors[name])
            if value.shape != (1,) + shape:
                raise ValueError(
                    f"action {name!r} must have shape {(1,) + shape}, got "
                    f"{value.shape}")
            # Drop the leading batch row: scalar actions land in 0-d slots.
            self._arrays[f"action_{name}"][index] = \
                value.astype(_ACTION_DTYPES[name]).reshape(shape)
        for group in LOGPROB_GROUPS:
            self._arrays[f"logprob_{group}"][index] = float(
                transition.logprob_groups[group])
        digest = np.frombuffer(transition.trace_digest, dtype=np.uint8)
        if digest.shape != (_TRACE_DIGEST_LEN,):
            raise ValueError(
                f"trace_digest must be {_TRACE_DIGEST_LEN} sha256 bytes")
        self._arrays["trace_digest"][index] = digest

        self._arrays["episode_index"][index] = transition.episode_index
        self._arrays["seed"][index] = transition.seed
        self._arrays["seat"][index] = transition.seat
        self._arrays["day"][index] = transition.day
        self._arrays["trainable"][index] = int(transition.trainable)
        self._arrays["logprob_total"][index] = float(transition.logprob_total)
        self._arrays["value"][index] = float(transition.value)
        self._arrays["reward"][index] = 0.0
        self._arrays["terminated"][index] = 0
        self._arrays["truncated"][index] = int(transition.truncated)
        self._arrays["valid"][index] = 1

        metadata.index = index
        metadata.trace_digest_hex = transition.trace_digest.hex()
        self.sidecar_records.append(metadata)
        self._count += 1

    # ------------------------------------------------------------- patching
    def patch_trace_digest(self, index: int, digest: bytes) -> None:
        self._require_index(index)
        raw = np.frombuffer(digest, dtype=np.uint8)
        if raw.shape != (_TRACE_DIGEST_LEN,):
            raise ValueError(f"digest must be {_TRACE_DIGEST_LEN} bytes")
        self._arrays["trace_digest"][index] = raw
        self.sidecar_records[index].trace_digest_hex = digest.hex()

    def patch_terminal(self, index: int, reward: float,
                       terminated: bool) -> None:
        """Assign the terminal reward to the final manager transition only."""
        self._require_index(index)
        self._arrays["reward"][index] = float(reward)
        self._arrays["terminated"][index] = int(terminated)

    def patch_truncated(self, index: int) -> None:
        """Flag a transition whose episode ended by turn budget, not DONE."""
        self._require_index(index)
        self._arrays["truncated"][index] = 1

    def _require_index(self, index: int) -> None:
        if isinstance(index, bool) or not isinstance(index, int) \
                or not 0 <= index < self._count:
            raise ValueError(f"index {index!r} outside filled range "
                             f"[0, {self._count})")

    # ------------------------------------------------------------ finalize
    def finalize(self) -> dict[str, np.ndarray]:
        """Read-only views over the full preallocated capacity arrays.

        Unfilled rows stay zero-initialized and untouched; the filled
        transition count is `len(buffer)` (and the explicit `count` field in
        the serialized NPZ). Consumers slice `[:count]` for training batches.
        """
        out = {}
        for name, array in self._arrays.items():
            view = array[:]
            view.setflags(write=False)  # view-only flag; base stays writable
            out[name] = view
        return out

    # -------------------------------------------------------- serialization
    def save(self, path: str | Path,
             run_metadata: Mapping[str, Any] | None = None) -> Path:
        """Write `<path>.npz` (strict schema) + `<path>.json` sidecar."""
        base = Path(path)
        base.parent.mkdir(parents=True, exist_ok=True)
        arrays = {"schema_version": np.asarray(TRAJECTORY_SCHEMA_VERSION,
                                               dtype=np.int32),
                  "count": np.asarray(self._count, dtype=np.int32),
                  "capacity": np.asarray(self.capacity, dtype=np.int32),
                  "e_history_version": np.frombuffer(
                      self.e_history_version.encode("utf-8"), dtype=np.uint8)}
        # Only the filled rows are serialized so the NPZ stays compact;
        # capacity-sized zero tail is reconstructed on load.
        arrays.update({name: array[:self._count]
                       for name, array in self.finalize().items()})
        np.savez_compressed(str(base) + ".npz", **arrays)
        sidecar = {
            "schema_version": TRAJECTORY_SCHEMA_VERSION,
            "npz_schema_version": TRAJECTORY_SCHEMA_VERSION,
            "count": self._count,
            "capacity": self.capacity,
            "e_history_version": self.e_history_version,
            "run_metadata": dict(run_metadata or {}),
            "transitions": [record.to_json_dict()
                            for record in self.sidecar_records],
        }
        # allow_nan=False: the sidecar must stay strictly JSON-safe; any
        # NaN/Inf leak fails loudly at write time instead of poisoning the
        # artifact for downstream JSON consumers.
        Path(str(base) + ".json").write_text(
            json.dumps(sidecar, sort_keys=True, indent=1, allow_nan=False),
            encoding="utf-8")
        return base

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_e_history_version: str | None = E_HISTORY_CORRECTED_V1,
    ) -> tuple["TrajectoryBuffer", dict[str, Any]]:
        """Round-trip load; validates schema, history version, and count bounds.

        Schema v1 is accepted only when the caller explicitly requests
        ``E_LEGACY``; corrected loading remains the default.
        """
        base = Path(path)
        with np.load(str(base) + ".npz", allow_pickle=False) as data:
            schema = int(data["schema_version"])
            if schema not in (TRAJECTORY_SCHEMA_VERSION,
                              _LEGACY_TRAJECTORY_SCHEMA_VERSION):
                raise ValueError(
                    f"trajectory schema version {schema} != supported "
                    f"{TRAJECTORY_SCHEMA_VERSION} or legacy "
                    f"{_LEGACY_TRAJECTORY_SCHEMA_VERSION}")
            count = int(data["count"])
            capacity = int(data["capacity"])
            if schema == _LEGACY_TRAJECTORY_SCHEMA_VERSION:
                e_history_version = E_HISTORY_LEGACY
            else:
                try:
                    e_history_version = normalize_e_history_version(
                        data["e_history_version"].tobytes().decode("utf-8"))
                except (KeyError, UnicodeDecodeError, ValueError) as exc:
                    raise ValueError(
                        "trajectory is missing a valid e_history_version") from exc
            if (expected_e_history_version is not None
                    and e_history_version != normalize_e_history_version(
                        expected_e_history_version)):
                raise ValueError(
                    f"trajectory e_history_version {e_history_version!r} does "
                    f"not match requested "
                    f"{normalize_e_history_version(expected_e_history_version)!r}")
            input_spec: dict[str, tuple[tuple[int, ...], type]] = {}
            for key in data.files:
                if not key.startswith("input_"):
                    continue
                array = data[key]
                input_spec[key[len("input_"):]] = (
                    tuple(array.shape[1:]), array.dtype.type)
            expected_core = set(arrayspec_core_keys()) | {
                f"input_{key}" for key in input_spec}
            missing = expected_core - set(data.files)
            if missing:
                raise ValueError(f"npz missing schema arrays {sorted(missing)}")
            buffer = cls(capacity=capacity, input_spec=input_spec,
                         e_history_version=e_history_version)
            for key in data.files:
                if key in ("schema_version", "count", "capacity",
                           "e_history_version"):
                    continue
                if key not in buffer._arrays:
                    raise ValueError(f"unexpected npz array {key!r}")
                expected = buffer._arrays[key][:count]
                loaded = data[key]
                if loaded.shape != expected.shape or loaded.dtype != expected.dtype:
                    raise ValueError(
                        f"npz array {key!r} mismatch: "
                        f"{loaded.shape}/{loaded.dtype} vs "
                        f"{expected.shape}/{expected.dtype}")
                buffer._arrays[key][:count] = loaded
            buffer._count = count
        sidecar = json.loads(
            Path(str(base) + ".json").read_text(encoding="utf-8"))
        if int(sidecar.get("npz_schema_version", -1)) != schema:
            raise ValueError("sidecar/npz schema version mismatch")
        if len(sidecar["transitions"]) != count:
            raise ValueError("sidecar transition count != npz count")
        return buffer, sidecar


def arrayspec_core_keys() -> list[str]:
    """All non-input npz keys of the strict schema (for load validation)."""
    keys = list(_SCALAR_FIELDS) + [f"logprob_{g}" for g in LOGPROB_GROUPS]
    keys += [f"action_{name}" for name in ACTION_TENSOR_SHAPES]
    keys.append("trace_digest")
    return keys


def e_input_spec() -> dict[str, tuple[tuple[int, ...], type]]:
    """Canonical own-only E model-facing input spec (shapes + dtypes).

    Derived through `bc_manager.live.encode_live_inputs` itself (never
    re-hardcoded) so the trajectory buffer cannot drift from the encoder.
    """
    empty_tiles = [[None] * 10 for _ in range(10)]
    farm = {
        "farmer": [0, 0], "hands": [], "hires_today": 0, "money": 3000.0,
        "tiles": empty_tiles, "unlocked_quadrants": ["NW"],
    }
    obs = {
        "day": 0, "hour": 0, "step": 0, "player": 0,
        "farms": [farm, dict(farm)],
        "market": {"inventory": {}, "prices": {}},
        "town": {"unlocked_shops": []},
        "private": {"shed": {}, "seeds": {}, "inventories": [{}]},
    }
    inputs = encode_live_inputs(
        obs, 0, {"workers_hired": 0, "hire_cost": 0}, step=0,
        economic_prev_start=(0, 3000.0))
    if "economic_context" not in inputs:
        raise ValueError("E input spec requires the economic_context channel")
    return {key: (value.shape[1:], value.dtype.type)
            for key, value in inputs.items()}


def load_trajectory(
    path: str | Path,
    *,
    expected_e_history_version: str | None = E_HISTORY_CORRECTED_V1,
) -> tuple[TrajectoryBuffer, dict[str, Any]]:
    """Module-level alias matching `TrajectoryBuffer.load`."""
    return TrajectoryBuffer.load(
        path, expected_e_history_version=expected_e_history_version)

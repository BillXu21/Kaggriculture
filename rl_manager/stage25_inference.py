"""Parent-owned native Stage 2.5 inference.

This is the only rollout-facing owner of a native Stage 2.5 parameter tree.
It deliberately sits above :mod:`rl_manager.stage25_policy`: the policy owns
the compiled nine-head JAX computation, while this module owns checkpoint
loading, provenance, row-stable randomness, and the NumPy contract at the
parent/worker boundary.

CPU rollout workers must not import this module.  In particular, the existing
worker-side provider keeps its native policy import lazy so importing the
provider alone remains JAX-free.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from bc_manager.economics import E_HISTORY_CORRECTED_V1, normalize_e_history_version
from rl_manager.stage25_checkpoint import (
    ARCHITECTURE_VERSION,
    OBSERVATION_SCHEMA_VERSION,
    OBSERVATION_VOCABULARY,
    PHYSICAL_SUPPORT_VERSION,
    load_stage25_inference_checkpoint,
)
from rl_manager.stage25_config import Stage25CurriculumConfig
from rl_manager.stage25_mechanics import (
    ACTION_CLASS_COUNTS,
    PhysicalContext,
    animal_target_support_mask,
    crop_delta_support_mask,
    land_target_support_mask,
    physical_crop_capacity,
)
from rl_manager.stage25_policy import (
    Stage25ModelConfig,
    _host_inputs,
    evaluate_actions as _evaluate_actions,
    greedy_act,
    stochastic_act,
)
from rl_manager.stage25_types import (
    STAGE25_PHYSICAL_SUPPORT_VERSION,
    STAGE25_POLICY_SCHEMA_VERSION,
    Stage25BehaviorIdentity,
    Stage25PolicyOutputs,
    stage25_rng_namespace,
)


__all__ = [
    "Stage25InferenceAdapter", "Stage25ParentInferenceAdapter",
    "Stage25NativeInference", "parameter_fingerprint",
    "curriculum_fingerprint", "teacher_force", "evaluate_actions",
    "audit_unchanged_weights",
]


_INTEGER_INPUTS = frozenset({
    "board_kind", "board_crop", "board_animal", "board_mask", "shed_counts",
    "seed_counts", "carried_counts", "unlocked", "market_inventory",
    "shop_counts", "day", "days_remaining", "crop_capacity",
})
_FLOAT_INPUTS = frozenset(OBSERVATION_VOCABULARY) - _INTEGER_INPUTS
_BOOLEAN_INPUTS = frozenset({"board_bool"})
_OUTPUT_FLOAT_FIELDS = ("component_logprobs", "joint_logprob", "value")
_AUDIT_ATOL = 1.0e-6
_AUDIT_RTOL = 1.0e-6


def _normalise_mode(mode: str, deterministic: bool) -> str:
    if deterministic:
        mode = "deterministic"
    if mode in ("sample", "stochastic"):
        return "stochastic"
    if mode in ("greedy", "deterministic"):
        return "deterministic"
    raise ValueError("mode must be 'stochastic' or 'deterministic'")


def _jsonable_curriculum(value: Stage25CurriculumConfig | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, Mapping):
        value = Stage25CurriculumConfig(**dict(value))
    if not isinstance(value, Stage25CurriculumConfig):
        raise TypeError("curriculum must be Stage25CurriculumConfig or a mapping")
    return dataclasses.asdict(value)


def curriculum_fingerprint(curriculum: Stage25CurriculumConfig | Mapping[str, Any]) -> str:
    """Return the stable fingerprint of the effective curriculum config."""
    payload = json.dumps(_jsonable_curriculum(curriculum), sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def parameter_fingerprint(params: Mapping[str, Any]) -> str:
    """Hash every parameter path, shape, dtype, and raw value exactly once."""
    if not isinstance(params, Mapping):
        raise TypeError("params must be a mapping pytree")
    flat, _ = jax.tree_util.tree_flatten_with_path(params)
    digest = hashlib.sha256()
    for path, leaf in flat:
        array = np.asarray(leaf)
        if array.dtype.hasobject:
            raise ValueError("parameter leaves cannot have object dtype")
        digest.update("/".join(str(item) for item in path).encode("utf-8"))
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(str(array.shape).encode("utf-8"))
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _as_curriculum(value: Any) -> Stage25CurriculumConfig:
    if isinstance(value, Stage25CurriculumConfig):
        return value
    if isinstance(value, Mapping):
        return Stage25CurriculumConfig(**dict(value))
    raise TypeError("curriculum must be Stage25CurriculumConfig or a mapping")


def _config_from_metadata(meta: Mapping[str, Any]) -> Stage25ModelConfig:
    # Keep this reconstruction local so the adapter does not need to retain a
    # second config source.  The checkpoint loader has already validated the
    # payload and its parameter manifest.
    from bc_manager_jax.model import ManagerConfig

    payload = dict(meta.get("config", meta.get("model_config", {})))
    if not payload:
        raise ValueError("native checkpoint metadata has no model config")
    manager_payload = payload.get("manager_config")
    if not isinstance(manager_payload, Mapping):
        raise ValueError("native checkpoint metadata has no manager_config")
    payload["manager_config"] = ManagerConfig(**dict(manager_payload))
    payload["curriculum"] = _as_curriculum(payload.get("curriculum", {}))
    return Stage25ModelConfig(**payload)


def _validate_inputs(inputs: Mapping[str, Any]) -> int:
    if not isinstance(inputs, Mapping) or not inputs:
        raise ValueError("inputs must be a non-empty mapping of batched arrays")
    unknown = sorted(set(inputs) - set(OBSERVATION_VOCABULARY))
    if unknown:
        raise ValueError(f"inputs contain unknown or non-own-only keys {unknown}")
    missing = sorted(set(OBSERVATION_VOCABULARY) - set(inputs))
    if missing:
        raise ValueError(f"inputs are missing required arrays {missing}")
    batch: int | None = None
    for name, value in inputs.items():
        array = np.asarray(value)
        if array.ndim == 0:
            raise ValueError(f"input {name!r} must be batched, got scalar")
        if batch is None:
            batch = int(array.shape[0])
        elif int(array.shape[0]) != batch:
            raise ValueError(
                f"input {name!r} has {array.shape[0]} rows, expected {batch}")
        valid_dtype = (np.issubdtype(array.dtype, np.integer)
                       if name in _INTEGER_INPUTS else
                       np.issubdtype(array.dtype, np.bool_)
                       if name in _BOOLEAN_INPUTS else
                       np.issubdtype(array.dtype, np.floating))
        if array.dtype.hasobject or not valid_dtype:
            expected = ("an integer" if name in _INTEGER_INPUTS else
                        "a boolean" if name in _BOOLEAN_INPUTS else "a float")
            raise ValueError(f"input {name!r} must have {expected} dtype")
        # Official canonical observations use NaN sentinels for nullable board
        # timing channels; the unchanged JAX encoder maps those channels
        # through its explicit NaN indicator/safe value path.
        if (name != "board_numeric" and
                np.issubdtype(array.dtype, np.floating) and
                not np.all(np.isfinite(array))):
            raise ValueError(f"input {name!r} contains non-finite values")
    if batch is None or batch < 1:
        raise ValueError("inputs must contain at least one row")
    return batch


def _normalise_contexts(
    contexts: Sequence[PhysicalContext], batch: int,
) -> tuple[PhysicalContext, ...]:
    if isinstance(contexts, (str, bytes)):
        raise ValueError("physical_contexts must be a sequence of contexts")
    try:
        values = tuple(contexts)
    except TypeError as exc:
        raise ValueError("physical_contexts must be a sequence of contexts") from exc
    if len(values) != batch:
        raise ValueError(
            f"physical_contexts has {len(values)} rows, expected {batch}")
    normalised = []
    for index, context in enumerate(values):
        if isinstance(context, PhysicalContext):
            normalised.append(context)
            continue
        try:
            normalised.append(PhysicalContext(
                observed_land=int(context.observed_land),
                crop_build_cells_by_land=tuple(context.crop_build_cells_by_land),
                placed_animals=tuple(context.placed_animals),
                reusable_empty_coops=int(context.reusable_empty_coops),
                reusable_empty_pastures=int(context.reusable_empty_pastures),
                unplaced_animals=tuple(context.unplaced_animals),
            ))
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid physical context at row {index}") from exc
    return tuple(normalised)


def _validate_context_consistency(
    inputs: Mapping[str, Any], contexts: Sequence[PhysicalContext],
    crop_capacity: Any, supports: Sequence[Any] | None = None,
) -> None:
    """Check that explicit physical payloads agree with canonical inputs.

    Curriculum can remove physically supported classes, so supplied support
    masks are checked as subsets of the physical masks rather than compared
    for exact equality.
    """
    unlocked = np.asarray(inputs["unlocked"])
    board_animals = np.asarray(inputs["board_animal"])
    capacities = np.asarray(crop_capacity).reshape(len(contexts), 5)
    for row, context in enumerate(contexts):
        observed_land = int(np.count_nonzero(unlocked[row]))
        placed = tuple(
            int(np.count_nonzero(board_animals[row] == species + 1))
            for species in range(3))
        if context.observed_land != observed_land:
            raise ValueError(
                f"physical context row {row} observed_land disagrees with inputs")
        if context.placed_animals != placed:
            raise ValueError(
                f"physical context row {row} placed_animals disagrees with inputs")
        if supports is None:
            continue
        support = supports[row]
        expected_land = land_target_support_mask(context.observed_land)
        if any(bool(value) and not expected_land[index]
               for index, value in enumerate(support["land"])):
            raise ValueError(f"physical support row {row} allows shrinking land")
        for species in range(3):
            expected = animal_target_support_mask(
                context, context.observed_land, species,
                context.placed_animals[:species])
            actual = support["animals"][species]
            if any(bool(value) and not expected[index]
                   for index, value in enumerate(actual)):
                raise ValueError(
                    f"physical support row {row} allows an infeasible animal target")
        total_capacity = max(
            physical_crop_capacity(context, context.observed_land,
                                   context.placed_animals), 0)
        for crop, goal in enumerate(capacities[row]):
            expected = crop_delta_support_mask(int(goal), total_capacity)
            actual = support["crops"][crop]
            if any(bool(value) and not expected[index]
                   for index, value in enumerate(actual)):
                raise ValueError(
                    f"physical support row {row} allows an infeasible crop delta")


def _row_token(value: Any) -> int:
    if isinstance(value, (str, np.str_)):
        text = "s:" + str(value)
    elif isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        text = "i:" + str(int(value))
    else:
        raise ValueError("row_ids must contain non-empty strings or integers")
    if text in ("s:",):
        raise ValueError("row_ids must not contain empty strings")
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "little") % (2**31 - 1)


def _normalise_row_ids(row_ids: Sequence[Any], batch: int) -> np.ndarray:
    if isinstance(row_ids, (str, bytes)):
        raise ValueError("row_ids must be a sequence, not one scalar")
    try:
        values = tuple(row_ids)
    except TypeError as exc:
        raise ValueError("row_ids must be a sequence") from exc
    if len(values) != batch:
        raise ValueError(f"row_ids has {len(values)} rows, expected {batch}")
    if len(set(values)) != len(values):
        raise ValueError("row_ids must be unique within one inference batch")
    tokens = np.asarray([_row_token(value) for value in values], dtype=np.int32)
    return tokens


def _canonical_prng_id(prng_id: str, identity: Stage25BehaviorIdentity,
                       seed: int = 0) -> str:
    if not isinstance(prng_id, str) or not prng_id:
        raise ValueError("prng_id must be a non-empty string")
    canonical = stage25_rng_namespace(identity, seed)
    accepted = (canonical,
                f"stage25/policy={identity.identity_id()}",
                f"stage25/behavior={identity.identity_id()}",
                f"parallel/policy={identity.identity_id()}")
    # Older direct adapter callers may provide an opaque rollout namespace;
    # preserve it as supplied.  The runner and parent dispatch both use the
    # canonical namespace above, while only known topology aliases normalize.
    return canonical if prng_id in accepted else prng_id


def _root_key(prng_id: str, identity: Stage25BehaviorIdentity,
              seed: int = 0) -> jax.Array:
    canonical = _canonical_prng_id(prng_id, identity, seed)
    digest = hashlib.sha256(
        (identity.fingerprint + "|" + canonical).encode("utf-8")).digest()
    return jax.random.PRNGKey(int.from_bytes(digest[:4], "little"))


class Stage25InferenceAdapter:
    """Immutable parent-side owner of one exact native Stage 2.5 snapshot."""

    @classmethod
    def from_checkpoint(cls, path: str | Path, **kwargs: Any) -> "Stage25InferenceAdapter":
        return cls(path, **kwargs)

    @classmethod
    def from_params(cls, params: Mapping[str, Any], config: Stage25ModelConfig,
                    **kwargs: Any) -> "Stage25InferenceAdapter":
        return cls(params, config, **kwargs)

    def __init__(
        self,
        checkpoint_or_params: str | Path | Mapping[str, Any] | None = None,
        config: Stage25ModelConfig | None = None,
        *,
        checkpoint_path: str | Path | None = None,
        params: Mapping[str, Any] | None = None,
        provider_curriculum: Stage25CurriculumConfig | Mapping[str, Any] | None = None,
        curriculum: Stage25CurriculumConfig | Mapping[str, Any] | None = None,
        e_history_version: str | None = None,
        name: str = "stage25_parent",
        version: str = "native-v1",
        seed: int | None = None,
        mode: str = "stochastic",
        deterministic: bool = False,
    ) -> None:
        sources = [value is not None for value in
                   (checkpoint_or_params, checkpoint_path, params)]
        if sum(sources) > 1:
            raise ValueError("provide exactly one checkpoint or parameter source")
        if provider_curriculum is not None and curriculum is not None:
            if _as_curriculum(provider_curriculum) != _as_curriculum(curriculum):
                raise ValueError("provider_curriculum and curriculum disagree")
        requested_curriculum = (provider_curriculum if provider_curriculum is not None
                                else curriculum)
        source = checkpoint_or_params if checkpoint_or_params is not None else (
            checkpoint_path if checkpoint_path is not None else params)
        if source is None:
            raise ValueError("a native checkpoint or in-memory params is required")

        metadata: dict[str, Any] = {}
        if isinstance(source, (str, Path)):
            loaded, metadata = load_stage25_inference_checkpoint(
                source, config=config,
                seed=seed,
                expected_e_history_version=e_history_version,
            )
            self.params = loaded
            self.checkpoint_path = Path(source)
            if config is None:
                config = _config_from_metadata(metadata)
        else:
            if not isinstance(source, Mapping):
                raise TypeError("parameter source must be a checkpoint path or mapping")
            if config is None:
                raise ValueError("config is required for in-memory params")
            self.params = jax.tree_util.tree_map(lambda leaf: jnp.asarray(leaf), source)
            self.checkpoint_path = None

        if not isinstance(config, Stage25ModelConfig):
            raise TypeError("config must be Stage25ModelConfig")
        self.config = config
        self.seed = int(metadata.get("seed", 0 if seed is None else seed))
        self._mode = _normalise_mode(mode, deterministic)
        checkpoint_curriculum = _as_curriculum(
            metadata.get("curriculum", config.curriculum))
        effective = checkpoint_curriculum if requested_curriculum is None \
            else _as_curriculum(requested_curriculum)
        if effective != checkpoint_curriculum:
            raise ValueError(
                "provider curriculum does not match the checkpoint curriculum")
        if config.curriculum != effective:
            raise ValueError("config curriculum does not match effective curriculum")
        self.curriculum = effective
        self.e_history_version = normalize_e_history_version(
            metadata.get("e_history_version", e_history_version or
                        E_HISTORY_CORRECTED_V1))
        if e_history_version is not None and self.e_history_version != normalize_e_history_version(e_history_version):
            raise ValueError("requested E-history version does not match checkpoint")
        physical_version = metadata.get("physical_support_version",
                                        STAGE25_PHYSICAL_SUPPORT_VERSION)
        observation_schema = metadata.get("observation_schema_version",
                                          OBSERVATION_SCHEMA_VERSION)
        policy_schema = metadata.get("architecture_version",
                                     STAGE25_POLICY_SCHEMA_VERSION)
        if observation_schema != OBSERVATION_SCHEMA_VERSION:
            raise ValueError("checkpoint observation schema is incompatible")
        if policy_schema != ARCHITECTURE_VERSION or \
                policy_schema != STAGE25_POLICY_SCHEMA_VERSION:
            raise ValueError("checkpoint policy schema is incompatible")
        if physical_version != STAGE25_PHYSICAL_SUPPORT_VERSION:
            raise ValueError("checkpoint physical-support version is incompatible")
        if PHYSICAL_SUPPORT_VERSION != STAGE25_PHYSICAL_SUPPORT_VERSION:
            raise ValueError("native physical-support constants disagree")

        self._parameter_fingerprint = parameter_fingerprint(self.params)
        self.identity = Stage25BehaviorIdentity(
            name=name, version=version,
            parameter_fingerprint=self._parameter_fingerprint,
            observation_schema_version=observation_schema,
            policy_schema_version=policy_schema,
            e_history_version=self.e_history_version,
            curriculum_version=self.curriculum.version,
            curriculum_fingerprint=curriculum_fingerprint(self.curriculum),
            physical_support_version=physical_version,
        )
        self.call_count = 0
        self.batch_size_history: list[int] = []

    @property
    def policy_identity(self) -> Stage25BehaviorIdentity:
        return self.identity

    @property
    def behavior_identity(self) -> Stage25BehaviorIdentity:
        return self.identity

    @property
    def deterministic(self) -> bool:
        return self._mode == "deterministic"

    def bind_provider_curriculum(self, curriculum: Stage25CurriculumConfig | Mapping[str, Any]) -> None:
        """Validate a provider's effective curriculum before any sampling."""
        requested = _as_curriculum(curriculum)
        if requested != self.curriculum:
            raise ValueError("provider curriculum does not match checkpoint curriculum")

    bind_curriculum = bind_provider_curriculum

    def load_config(self) -> Stage25ModelConfig:
        """Return the already-loaded config without a worker-side import."""
        return self.config

    def _prepare(
        self, inputs: Mapping[str, Any], physical_contexts: Sequence[PhysicalContext] | None,
    ) -> tuple[int, dict[str, Any], jax.Array, tuple[PhysicalContext, ...] | None]:
        batch = _validate_inputs(inputs)
        prepared, capacity, policy_batch = _host_inputs(inputs, self.config)
        if policy_batch != batch:
            raise ValueError("policy input preparation changed the batch size")
        contexts = None
        if physical_contexts is not None:
            contexts = _normalise_contexts(physical_contexts, batch)
            _validate_context_consistency(inputs, contexts, capacity)
        return batch, prepared, capacity, contexts

    def _outputs(self, result: Mapping[str, Any], batch: int) -> Stage25PolicyOutputs:
        arrays = {
            "classes": np.asarray(result["classes"], dtype=np.int16),
            "component_logprobs": np.asarray(result["component_logprobs"], dtype=np.float32),
            "joint_logprob": np.asarray(result["joint_logprob"], dtype=np.float32),
            "value": np.asarray(result["value"], dtype=np.float32),
            "decoded_goals": np.asarray(result["decoded_goals"], dtype=np.int16),
            "valid": np.asarray(result["valid"], dtype=np.bool_),
        }
        output = Stage25PolicyOutputs(**arrays, policy_identity=self.identity,
                                      batch_size=batch)
        return output

    def plan_batch(
        self, inputs: Mapping[str, Any], prng_id: str, *,
        physical_contexts: Sequence[PhysicalContext] | None = None,
    ) -> Stage25PolicyOutputs:
        batch = _validate_inputs(inputs)
        return self.plan_batch_with_row_ids(
            inputs, tuple(range(batch)), prng_id,
            physical_contexts=physical_contexts)

    def plan_batch_with_row_ids(
        self, inputs: Mapping[str, Any], row_ids: Sequence[Any], prng_id: str,
        *, physical_contexts: Sequence[PhysicalContext] | None = None,
    ) -> Stage25PolicyOutputs:
        batch, prepared, capacity, contexts = self._prepare(inputs, physical_contexts)
        ids = _normalise_row_ids(row_ids, batch)
        if self.deterministic:
            result = greedy_act(
                self.params, prepared, self.config,
                physical_contexts=contexts, crop_capacity=capacity,
                row_ids=ids)
        else:
            root = _root_key(prng_id, self.identity, self.seed)
            keys = np.asarray(jnp.stack([
                jax.random.fold_in(root, int(row_id)) for row_id in ids
            ]), dtype=np.uint32)
            result = stochastic_act(
                self.params, prepared, self.config, rng_keys=keys,
                physical_contexts=contexts, crop_capacity=capacity, row_ids=ids,
                reject_invalid=False)
        output = self._outputs(result, batch)
        if not bool(np.all(output.valid)):
            raise ValueError("native stochastic inference produced unsupported physical actions")
        self.call_count += 1
        self.batch_size_history.append(batch)
        return output

    def infer_batch(
        self, inputs: Mapping[str, Any], crop_capacity: Any = None,
        physical_contexts: Sequence[PhysicalContext] | None = None,
        supports: Sequence[Any] | None = None,
        row_ids: Sequence[Any] | None = None, prng_id: str | None = None,
    ) -> Stage25PolicyOutputs:
        """Compatibility seam for runner callers carrying context separately.

        ``supports`` is accepted only as a row-count contract.  The native
        policy recomputes the support intersection from the immutable
        physical contexts and bound curriculum; an external mask must never
        silently change behavior identity or action likelihoods.
        """
        if not isinstance(inputs, Mapping):
            raise ValueError("inputs must be a mapping")
        merged = dict(inputs)
        if crop_capacity is not None:
            if "crop_capacity" in merged and not np.array_equal(
                    np.asarray(merged["crop_capacity"]), np.asarray(crop_capacity)):
                raise ValueError("crop_capacity disagrees between inputs and context")
            merged["crop_capacity"] = crop_capacity
        batch = _validate_inputs(merged)
        support_rows = None if supports is None else tuple(supports)
        if support_rows is not None:
            if len(support_rows) != batch:
                raise ValueError("supports must contain one row per input")
            if physical_contexts is not None:
                contexts = _normalise_contexts(physical_contexts, batch)
                _validate_context_consistency(
                    merged, contexts, merged["crop_capacity"], support_rows)
        if prng_id is None:
            raise ValueError("prng_id is required")
        if row_ids is None:
            return self.plan_batch(merged, prng_id,
                                   physical_contexts=physical_contexts)
        return self.plan_batch_with_row_ids(
            merged, row_ids, prng_id, physical_contexts=physical_contexts)

    stage25_infer_batch = infer_batch
    plan_batch_with_context = infer_batch

    def greedy_batch(
        self, inputs: Mapping[str, Any], *, physical_contexts: Sequence[PhysicalContext] | None = None,
    ) -> Stage25PolicyOutputs:
        batch, prepared, capacity, contexts = self._prepare(inputs, physical_contexts)
        result = greedy_act(
            self.params, prepared, self.config, physical_contexts=contexts,
            crop_capacity=capacity, row_ids=np.arange(batch, dtype=np.int32))
        output = self._outputs(result, batch)
        if not bool(np.all(output.valid)):
            raise ValueError("native greedy inference produced unsupported physical actions")
        self.call_count += 1
        self.batch_size_history.append(batch)
        return output

    def teacher_force(
        self, inputs: Mapping[str, Any], classes: Any = None, *, actions: Any = None,
        physical_contexts: Sequence[PhysicalContext] | None = None,
        row_ids: Sequence[Any] | None = None, reject_invalid: bool = True,
    ) -> Stage25PolicyOutputs:
        if classes is None:
            classes = actions
        elif actions is not None:
            raise ValueError("provide classes or actions, not both")
        if classes is None:
            raise ValueError("teacher_force requires stored classes")
        batch, prepared, capacity, contexts = self._prepare(inputs, physical_contexts)
        actions = np.asarray(classes)
        if actions.shape != (batch, len(ACTION_CLASS_COUNTS)):
            raise ValueError(f"classes must have shape {(batch, len(ACTION_CLASS_COUNTS))}")
        if not np.issubdtype(actions.dtype, np.integer):
            raise ValueError("classes must have an integer dtype")
        if np.any(actions < np.iinfo(np.int16).min) or np.any(actions > np.iinfo(np.int16).max):
            raise ValueError("classes cannot be represented by int16")
        ids = (np.arange(batch, dtype=np.int32) if row_ids is None
               else _normalise_row_ids(row_ids, batch))
        result = _evaluate_actions(
            self.params, prepared, self.config, actions=actions,
            physical_contexts=contexts, crop_capacity=capacity, row_ids=ids,
            reject_invalid=reject_invalid)
        return self._outputs(result, batch)

    def evaluate_actions(self, inputs: Mapping[str, Any], classes: Any = None, *,
                         actions: Any = None,
                         physical_contexts: Sequence[PhysicalContext] | None = None,
                         row_ids: Sequence[Any] | None = None,
                         reject_invalid: bool = True) -> Stage25PolicyOutputs:
        return self.teacher_force(
            inputs, classes, actions=actions,
            physical_contexts=physical_contexts, row_ids=row_ids,
            reject_invalid=reject_invalid)

    def bootstrap_value(
        self, inputs: Mapping[str, Any], *,
        physical_contexts: Sequence[PhysicalContext] | None = None,
        crop_capacity: Any = None, row_ids: Sequence[Any] | None = None,
        prng_id: str | None = None,
    ) -> np.ndarray:
        """Evaluate only the critic at a truncation next state.

        The dummy class sequence is deliberately never returned or accepted by
        a provider.  ``_policy_core`` computes the value from the encoded
        observation before decoding heads, so teacher forcing is sufficient to
        obtain the bootstrap while preserving the no-extra-plan rule.
        """
        # The runner carries the same namespace on decision and value-only
        # requests.  Critic evaluation is deterministic and does not consume
        # that namespace, but accepting it preserves the Packet 5A transport
        # contract without accidentally sampling a replacement plan.
        del prng_id
        merged = dict(inputs)
        if crop_capacity is not None:
            if "crop_capacity" in merged and not np.array_equal(
                    np.asarray(merged["crop_capacity"]), np.asarray(crop_capacity)):
                raise ValueError("crop_capacity disagrees between inputs and context")
            merged["crop_capacity"] = crop_capacity
        batch, prepared, capacity, contexts = self._prepare(merged, physical_contexts)
        classes = np.zeros((batch, len(ACTION_CLASS_COUNTS)), dtype=np.int16)
        ids = (np.arange(batch, dtype=np.int32) if row_ids is None
               else _normalise_row_ids(row_ids, batch))
        result = _evaluate_actions(
            self.params, prepared, self.config, actions=classes,
            physical_contexts=contexts, crop_capacity=capacity,
            row_ids=ids, reject_invalid=False)
        return np.asarray(result["value"], dtype=np.float32)

    def audit_unchanged_weights(
        self, inputs: Mapping[str, Any] | Stage25PolicyOutputs,
        rollout_outputs: Stage25PolicyOutputs | Mapping[str, Any] | None = None,
        *, physical_contexts: Sequence[PhysicalContext] | None = None,
        row_ids: Sequence[Any] | None = None,
        atol: float = _AUDIT_ATOL, rtol: float = _AUDIT_RTOL,
    ) -> dict[str, Any]:
        """Re-evaluate stored classes and compare every rollout value."""
        if isinstance(inputs, Stage25PolicyOutputs):
            inputs, rollout_outputs = rollout_outputs, inputs
        if not isinstance(inputs, Mapping) or not isinstance(rollout_outputs, Stage25PolicyOutputs):
            raise TypeError("audit requires (inputs, Stage25PolicyOutputs)")
        if rollout_outputs.policy_identity != self.identity:
            raise ValueError("rollout outputs belong to a different policy identity")
        fresh = self.teacher_force(
            inputs, rollout_outputs.classes, physical_contexts=physical_contexts,
            row_ids=row_ids, reject_invalid=True)
        max_errors: dict[str, float] = {}
        for name in ("classes", "decoded_goals", "valid"):
            left = np.asarray(getattr(rollout_outputs, name))
            right = np.asarray(getattr(fresh, name))
            if not np.array_equal(left, right):
                raise AssertionError(f"unchanged-weight audit mismatch in {name}")
            max_errors[name] = 0.0
        for name in _OUTPUT_FLOAT_FIELDS:
            left = np.asarray(getattr(rollout_outputs, name), dtype=np.float32)
            right = np.asarray(getattr(fresh, name), dtype=np.float32)
            difference = float(np.max(np.abs(left - right))) if left.size else 0.0
            max_errors[name] = difference
            if not np.allclose(left, right, atol=atol, rtol=rtol, equal_nan=False):
                raise AssertionError(
                    f"unchanged-weight audit mismatch in {name}: max error {difference}")
        return {"ok": True, "identity": self.identity.identity_id(),
                "max_abs_error": max_errors, "output": fresh}


Stage25ParentInferenceAdapter = Stage25InferenceAdapter
Stage25NativeInference = Stage25InferenceAdapter


def teacher_force(adapter: Stage25InferenceAdapter, inputs: Mapping[str, Any], classes: Any = None, **kwargs: Any) -> Stage25PolicyOutputs:
    return adapter.teacher_force(inputs, classes, **kwargs)


def evaluate_actions(adapter: Stage25InferenceAdapter, inputs: Mapping[str, Any], classes: Any = None, **kwargs: Any) -> Stage25PolicyOutputs:
    return adapter.teacher_force(inputs, classes, **kwargs)


def audit_unchanged_weights(adapter: Stage25InferenceAdapter, inputs: Mapping[str, Any], rollout_outputs: Stage25PolicyOutputs, **kwargs: Any) -> dict[str, Any]:
    return adapter.audit_unchanged_weights(inputs, rollout_outputs, **kwargs)

"""Native JAX Stage 2.5 encoder/decoder/value policy.

The policy intentionally has a narrow parameter contract.  Its encoder is
the own-only E representation from :mod:`bc_manager_jax.model`; the decoder
is a single autoregressive recurrence over the nine physical actions.  No
economic reward or legacy manager head is part of this module.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from bc_manager.constants import (
    BOARD_SIZE,
    CROP_ORDER,
    TILE_KIND_IDS,
)
from bc_manager_jax.model import (
    ManagerConfig,
    _Dropout,
    _manager_representation,
    empty_params as _empty_encoder_params,
    init_train_params as _init_encoder_train_params,
    tiny_manager_config,
    validate_inputs as _validate_encoder_inputs,
)
from rl_manager.stage25_config import Stage25CurriculumConfig
from rl_manager.stage25_mechanics import (
    ACTION_CLASS_COUNTS,
    ACTION_ORDER,
)


__all__ = [
    "ACTION_ORDER", "ACTION_CLASS_COUNTS", "Stage25ModelConfig",
    "tiny_stage25_config", "small_stage25_config", "large_stage25_config",
    "init_stage25_params", "parameter_spec", "stage25_parameter_count",
    "stochastic_act", "greedy_act", "evaluate_actions",
]


_N_ACTIONS = len(ACTION_ORDER)
_N_CROPS = len(CROP_ORDER)
_N_ANIMALS = 3
_N_LAND = 4
_N_ANIMAL_CLASSES = 101
_N_CROP_CLASSES = 201
_CROP_DELTA_OFFSET = 100
_MAX_LOGIT = 1.0e30
_TILE_PRESENT = 0
_ANIMAL_PRESENT = 2
_KIND_PLANT = TILE_KIND_IDS["PLANT"]
_KIND_COOP = TILE_KIND_IDS["COOP"]
_KIND_PASTURE = TILE_KIND_IDS["PASTURE"]
_KIND_WEED = TILE_KIND_IDS["WEED"]
_KIND_LOCKED = TILE_KIND_IDS["LOCKED"]


class _ShapeSpec:
    """Small shape leaf that is iterable and exposes a ``size``."""

    __slots__ = ("shape", "size")

    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = tuple(shape)
        self.size = int(np.prod(self.shape))

    def __iter__(self):
        return iter(self.shape)


@dataclass(frozen=True)
class Stage25ModelConfig:
    """Static policy configuration.

    ``manager_config`` selects the established JAX encoder dimensions.  The
    convenience constructors use the existing tiny/default manager configs,
    while the decoder remains the same width ``D`` as that encoder.
    """

    d_model: int = 128
    num_layers: int = 4
    num_heads: int = 4
    ffn_dim: int = 384
    dropout: float = 0.0
    manager_config: ManagerConfig | None = field(default=None, repr=False,
                                                  compare=False)
    curriculum: Stage25CurriculumConfig = field(
        default_factory=Stage25CurriculumConfig)
    crop_prior_tau: float = 25.0
    output_init_scale: float = 0.02
    # Aliases retained for callers using the names from the implementation
    # brief.  ``None`` means use crop_prior_tau/output_init_scale.
    crop_bias_tau: float | None = None
    state_relative_scale_init: float = 0.08

    def __post_init__(self) -> None:
        supplied_manager = self.manager_config
        if supplied_manager is not None:
            if supplied_manager.include_opponent_board:
                raise ValueError(
                    "Stage25ModelConfig requires the own-only encoder")
            object.__setattr__(self, "d_model", supplied_manager.d_model)
            object.__setattr__(self, "num_layers", supplied_manager.num_layers)
            object.__setattr__(self, "num_heads", supplied_manager.num_heads)
            object.__setattr__(self, "ffn_dim", supplied_manager.ffn_dim)
            object.__setattr__(self, "dropout", supplied_manager.dropout)
        resolved_manager = ManagerConfig(
            d_model=self.d_model, num_layers=self.num_layers,
            num_heads=self.num_heads, ffn_dim=self.ffn_dim,
            dropout=self.dropout)
        object.__setattr__(self, "manager_config", resolved_manager)
        tau = self.crop_prior_tau if self.crop_bias_tau is None \
            else self.crop_bias_tau
        if not math.isfinite(float(tau)) or float(tau) <= 0.0:
            raise ValueError("crop_prior_tau must be finite and positive")
        if not math.isfinite(float(self.output_init_scale)) \
                or float(self.output_init_scale) <= 0.0:
            raise ValueError("output_init_scale must be finite and positive")
        if not math.isfinite(float(self.state_relative_scale_init)) \
                or float(self.state_relative_scale_init) <= 0.0:
            raise ValueError(
                "state_relative_scale_init must be finite and positive")

    @classmethod
    def tiny(cls, **overrides: Any) -> "Stage25ModelConfig":
        mc = tiny_manager_config()
        values = {"d_model": mc.d_model, "num_layers": mc.num_layers,
                  "num_heads": mc.num_heads, "ffn_dim": mc.ffn_dim,
                  "dropout": mc.dropout}
        supplied_mc = overrides.pop("manager_config", None)
        if supplied_mc is not None:
            values.update({"d_model": supplied_mc.d_model,
                           "num_layers": supplied_mc.num_layers,
                           "num_heads": supplied_mc.num_heads,
                           "ffn_dim": supplied_mc.ffn_dim,
                           "dropout": supplied_mc.dropout})
        values.update(overrides)
        return cls(**values)

    @classmethod
    def small(cls, **overrides: Any) -> "Stage25ModelConfig":
        values = {"d_model": 128, "num_layers": 4, "num_heads": 4,
                  "ffn_dim": 384, "dropout": 0.0}
        values.update(overrides)
        return cls(**values)

    @classmethod
    def large(cls, **overrides: Any) -> "Stage25ModelConfig":
        values = {"d_model": 256, "num_layers": 7, "num_heads": 8,
                  "ffn_dim": 1024, "dropout": 0.0}
        values.update(overrides)
        return cls(**values)


def tiny_stage25_config(**overrides: Any) -> Stage25ModelConfig:
    return Stage25ModelConfig.tiny(**overrides)


def small_stage25_config(**overrides: Any) -> Stage25ModelConfig:
    return Stage25ModelConfig.small(**overrides)


def large_stage25_config(**overrides: Any) -> Stage25ModelConfig:
    return Stage25ModelConfig.large(**overrides)


def _linear(in_dim: int, out_dim: int) -> dict[str, jax.Array]:
    return {
        "kernel": jnp.zeros((in_dim, out_dim), dtype=jnp.float32),
        "bias": jnp.zeros((out_dim,), dtype=jnp.float32),
    }


def _encoder_only(params: Mapping[str, Any]) -> dict[str, Any]:
    """Copy only the leaves used by ``_manager_representation``."""
    names = (
        "manager_token", "role_embedding", "tile_encoder",
        "global_encoders", "encoder", "encoder_norm",
    )
    # A full bc_manager_jax tree has these at its root; a policy tree may
    # wrap the same representation under ``encoder``.
    source = params if "manager_token" in params else params.get("encoder", params)
    missing = [name for name in names if name not in source]
    if missing:
        raise ValueError(f"encoder params are missing {missing}")
    return {name: source[name] for name in names}


def _empty_params(config: Stage25ModelConfig) -> dict[str, Any]:
    d = config.d_model
    encoder = _encoder_only(_empty_encoder_params(
        config.manager_config, model_variant="E"))
    projections = tuple(_linear(d, count)
                        for count in ACTION_CLASS_COUNTS)
    embeddings = tuple(jnp.zeros((count, d), dtype=jnp.float32)
                       for count in ACTION_CLASS_COUNTS)
    return {
        "encoder": encoder,
        "capacity_conditioning": jnp.zeros((_N_CROPS, d), jnp.float32),
        "recurrent_decoder": {
            "Wz": jnp.zeros((d, d), jnp.float32),
            "Wh": jnp.zeros((d, d), jnp.float32),
            "bias": jnp.zeros((d,), jnp.float32),
            "decision_embedding": jnp.zeros((_N_ACTIONS, d), jnp.float32),
            "state_relative_scale": jnp.zeros((_N_LAND,), jnp.float32),
        },
        "output_projections": projections,
        "action_embeddings": embeddings,
        "value_head": _linear(d, 1),
    }


def _assert_projection_embedding_count(params: Mapping[str, Any], d: int) -> int:
    total = 0
    projections = params["output_projections"]
    embeddings = params["action_embeddings"]
    if len(projections) != _N_ACTIONS or len(embeddings) != _N_ACTIONS:
        raise AssertionError("Stage 2.5 requires exactly nine action pairs")
    for index, (projection, embedding, count) in enumerate(
            zip(projections, embeddings, ACTION_CLASS_COUNTS)):
        kernel = projection["kernel"]
        bias = projection["bias"]
        if tuple(kernel.shape) != (d, count):
            raise AssertionError(
                f"output projection {index} has shape {kernel.shape}, "
                f"expected {(d, count)}")
        if tuple(bias.shape) != (count,) or tuple(embedding.shape) != (count, d):
            raise AssertionError(f"action pair {index} has incorrect shape")
        total += int(np.prod(kernel.shape)) + int(np.prod(bias.shape)) \
            + int(np.prod(embedding.shape))
    expected = sum(ACTION_CLASS_COUNTS) * (2 * d + 1)
    if total != expected:
        raise AssertionError(
            f"output+embedding parameter count {total} != {expected}")
    return total


def parameter_spec(config: Stage25ModelConfig) -> dict[str, Any]:
    """Return a zero-valued pytree with the exact parameter structure."""
    if not isinstance(config, Stage25ModelConfig):
        raise TypeError("parameter_spec expects Stage25ModelConfig")
    params = _empty_params(config)
    _assert_projection_embedding_count(params, config.d_model)
    return jax.tree_util.tree_map(
        lambda value: _ShapeSpec(tuple(value.shape)), params)


def stage25_parameter_count(
        params_or_config: Mapping[str, Any] | Stage25ModelConfig) -> int:
    """Count actual projection/bias/embedding leaves, asserting the contract."""
    if isinstance(params_or_config, Stage25ModelConfig):
        params = _empty_params(params_or_config)
        d = params_or_config.d_model
    else:
        params = params_or_config
        try:
            d = int(params["recurrent_decoder"]["Wz"].shape[0])
        except (KeyError, AttributeError, TypeError) as exc:
            raise ValueError("invalid Stage 2.5 parameter tree") from exc
    # The explicit decoder count is checked from actual leaves.  The public
    # count covers the complete native tree, matching parameter_spec.
    _assert_projection_embedding_count(params, d)
    return int(sum(np.prod(leaf.shape)
                   for leaf in jax.tree_util.tree_leaves(params)))


def init_stage25_params(
        config: Stage25ModelConfig,
        seed: int = 0,
        encoder_params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Initialize a native Stage 2.5 pytree.

    An optional full E checkpoint or encoder-only subtree can seed the single
    stored encoder.  Legacy head leaves are discarded at this boundary.
    """
    if not isinstance(config, Stage25ModelConfig):
        raise TypeError("config must be Stage25ModelConfig")
    source = (_init_encoder_train_params(config.manager_config, seed=int(seed),
                                         model_variant="E")
              if encoder_params is None else encoder_params)
    encoder = _encoder_only(source)

    d = config.d_model
    key = jax.random.PRNGKey(int(seed) + 101)
    leaves = jax.random.split(key, 6 + 2 * _N_ACTIONS)
    def normal(k: jax.Array, shape: tuple[int, ...]) -> jax.Array:
        return jax.random.normal(k, shape, dtype=jnp.float32) \
            * float(config.output_init_scale)

    projections = []
    embeddings = []
    for i, count in enumerate(ACTION_CLASS_COUNTS):
        projections.append({
            "kernel": normal(leaves[6 + 2 * i], (d, count)),
            "bias": jnp.zeros((count,), jnp.float32),
        })
        embeddings.append(normal(leaves[7 + 2 * i], (count, d)))
    # Crop logits start with the requested -abs(delta)/tau prior.
    tau = config.crop_prior_tau if config.crop_bias_tau is None \
        else config.crop_bias_tau
    crop_bias = -jnp.abs(jnp.arange(-100, 101, dtype=jnp.float32)) / float(tau)
    projections = [dict(item) for item in projections]
    for index in range(4, _N_ACTIONS):
        projections[index]["bias"] = crop_bias
    params = {
        "encoder": encoder,
        "capacity_conditioning": normal(leaves[0], (_N_CROPS, d)),
        "recurrent_decoder": {
            "Wz": normal(leaves[1], (d, d)),
            "Wh": normal(leaves[2], (d, d)),
            "bias": jnp.zeros((d,), jnp.float32),
            "decision_embedding": normal(leaves[3], (_N_ACTIONS, d)),
            "state_relative_scale": jnp.full(
                (_N_LAND,), float(config.state_relative_scale_init),
                dtype=jnp.float32),
        },
        "output_projections": tuple(projections),
        "action_embeddings": tuple(embeddings),
        "value_head": {
            "kernel": normal(leaves[4], (d, 1)),
            "bias": jnp.zeros((1,), jnp.float32),
        },
    }
    stage25_parameter_count(params)
    return params


def _lookup_extra(inputs: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in inputs:
            return inputs[name]
    return None


def _host_inputs(
        inputs: Mapping[str, Any], config: Stage25ModelConfig,
        crop_capacity: Any = None,
) -> tuple[dict[str, jax.Array], jax.Array, jax.Array, int]:
    if not isinstance(inputs, Mapping):
        raise ValueError("inputs must be a mapping")
    capacity = (crop_capacity if crop_capacity is not None else
                _lookup_extra(inputs, "crop_capacity", "capacity"))
    if capacity is None:
        raise ValueError("inputs must contain crop_capacity")
    base = {key: value for key, value in inputs.items()
            if key not in ("crop_capacity", "capacity", "crop_goals",
                           "previous_crop_goals", "prior_crop_goals",
                           "row_ids")}
    _validate_encoder_inputs(base, config.manager_config, model_variant="E")
    prepared = {
        key: (jnp.asarray(value, dtype=jnp.int32)
              if key in {"board_kind", "board_crop", "board_animal",
                         "board_mask", "shed_counts", "carried_counts",
                         "unlocked", "seed_counts", "market_inventory",
                         "shop_counts", "day", "days_remaining"}
              else jnp.asarray(value, dtype=jnp.float32))
        for key, value in base.items()
    }
    # Economics are allowed as encoder context, but never enter the physical
    # support equations below.  This preserves the corrected-E representation
    # while keeping permanent feasibility independent of prices or money.
    b = int(prepared["board_kind"].shape[0])
    board = prepared["board_kind"]
    if tuple(board.shape) != (b, BOARD_SIZE):
        raise ValueError("board_kind must have shape [B, 100]")
    if tuple(prepared["board_animal"].shape) != tuple(board.shape):
        raise ValueError("board_animal must match board_kind")
    if tuple(prepared["board_mask"].shape) != (b, BOARD_SIZE, 4):
        raise ValueError("board_mask must have shape [B, 100, 4]")
    unlocked = np.asarray(prepared["unlocked"])
    if unlocked.shape != (b, _N_LAND) or not np.all(np.isin(unlocked, (0, 1))):
        raise ValueError("unlocked must have shape [B, 4] and contain 0/1")
    prefix = np.asarray(unlocked, dtype=np.int32)
    if np.any(prefix[:, 1:] > prefix[:, :-1]):
        raise ValueError("unlocked must be a canonical land prefix")

    capacity_array = np.asarray(capacity, dtype=np.float32)
    if capacity_array.ndim == 1:
        if capacity_array.shape != (b,):
            raise ValueError("crop_capacity must have shape [B] or [B, 5]")
        capacity_array = np.repeat(capacity_array[:, None], _N_CROPS, axis=1)
    if capacity_array.shape != (b, _N_CROPS) \
            or not np.all(np.isfinite(capacity_array)):
        raise ValueError("crop_capacity must have finite shape [B, 5]")

    goals = _lookup_extra(inputs, "crop_goals", "previous_crop_goals",
                          "prior_crop_goals")
    if goals is None:
        # A useful boundary default for live encoded boards; callers training
        # persistent goals should pass them explicitly.
        crop = np.asarray(prepared["board_crop"])
        goals_array = np.stack([
            np.sum((crop == index + 1).astype(np.int32), axis=1)
            for index in range(_N_CROPS)
        ], axis=1)
    else:
        goals_array = np.asarray(goals, dtype=np.int32)
    if goals_array.shape != (b, _N_CROPS) \
            or np.any(goals_array < 0) or np.any(goals_array > 100):
        raise ValueError("crop_goals must have shape [B, 5] and lie in [0, 100]")
    return prepared, jnp.asarray(capacity_array), jnp.asarray(goals_array), b


def _host_contexts(contexts: Any, batch: int) -> tuple[jax.Array, ...] | None:
    if contexts is None:
        return None
    if len(contexts) != batch:
        raise ValueError("physical_contexts must contain one context per row")
    fields = ("observed_land", "crop_build_cells_by_land", "placed_animals",
              "reusable_empty_coops", "reusable_empty_pastures",
              "unplaced_animals")
    values = []
    for field_name in fields:
        values.append(np.asarray([getattr(context, field_name)
                                  for context in contexts], dtype=np.int32))
    return tuple(jnp.asarray(value) for value in values)


def _masked_distribution(logits: jax.Array,
                         mask: jax.Array) -> tuple[jax.Array, ...]:
    any_valid = jnp.any(mask, axis=-1)
    safe = jnp.where(mask, logits, -_MAX_LOGIT)
    max_logit = jnp.max(safe, axis=-1, keepdims=True)
    exp_logits = jnp.where(mask, jnp.exp(safe - max_logit), 0.0)
    denom = jnp.sum(exp_logits, axis=-1, keepdims=True)
    denom = jnp.where(any_valid[:, None], denom, 1.0)
    probs = exp_logits / denom
    log_norm = max_logit + jnp.log(denom)
    log_probs = jnp.where(mask, logits - log_norm, 0.0)
    entropy = -jnp.sum(jnp.where(mask, probs * log_probs, 0.0), axis=-1)
    return probs, log_probs, entropy, any_valid


def _physical_context_jax(inputs: Mapping[str, jax.Array]) -> tuple[jax.Array, ...]:
    kind = inputs["board_kind"]
    animal = inputs["board_animal"]
    mask = inputs["board_mask"] > 0
    unlocked = inputs["unlocked"] > 0
    positions = jnp.arange(BOARD_SIZE)
    rows = positions // 10
    cols = positions % 10
    quadrant = jnp.where(
        rows < 5, jnp.where(cols < 5, 0, 1),
        jnp.where(cols < 5, 2, 3))
    in_unlocked = unlocked[:, quadrant]
    tile_present = mask[..., _TILE_PRESENT]
    animal_present = mask[..., _ANIMAL_PRESENT]
    compatible = (~tile_present) | (kind == _KIND_PLANT) | (kind == _KIND_WEED)
    locked_future = (kind == _KIND_LOCKED) & (~in_unlocked)
    cells = jnp.stack([
        jnp.sum(((quadrant < target) &
                 (jnp.where(in_unlocked, compatible, locked_future))), axis=1)
        for target in range(1, _N_LAND + 1)
    ], axis=1).astype(jnp.int32)
    structures = (kind == _KIND_COOP) | (kind == _KIND_PASTURE)
    occupied = in_unlocked & structures & animal_present
    placed = jnp.stack([
        jnp.sum(occupied & (animal == index + 1), axis=1)
        for index in range(_N_ANIMALS)
    ], axis=1).astype(jnp.int32)
    empty_coops = jnp.sum(in_unlocked & (kind == _KIND_COOP) &
                          (~animal_present), axis=1).astype(jnp.int32)
    empty_pastures = jnp.sum(in_unlocked & (kind == _KIND_PASTURE) &
                             (~animal_present), axis=1).astype(jnp.int32)
    unplaced = (inputs["shed_counts"][..., -_N_ANIMALS:] +
                inputs["carried_counts"][..., -_N_ANIMALS:]).astype(jnp.int32)
    observed_land = jnp.sum(unlocked, axis=1).astype(jnp.int32)
    return (observed_land, cells, placed, empty_coops, empty_pastures,
            unplaced)


def _curriculum_masks(
        config: Stage25ModelConfig, observed_land: jax.Array,
        placed: jax.Array, land_mask: jax.Array,
        animal_mask: jax.Array, crop_mask: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    curriculum = config.curriculum
    if not curriculum.enabled:
        return land_mask, animal_mask, crop_mask
    land_cap = curriculum.max_land_expansion_per_decision
    if land_cap is not None:
        land_mask = land_mask & (
            jnp.arange(1, 5)[None, :] <= observed_land[:, None] + land_cap)
    animal_cap = curriculum.max_animal_additions_per_species_per_decision
    if animal_cap is not None:
        animal_mask = animal_mask & (
            jnp.arange(101)[None, :] <=
            placed[:, :, None] + animal_cap)
    crop_cap = curriculum.max_positive_crop_delta
    if crop_cap is not None:
        crop_mask = crop_mask & (
            (jnp.arange(-100, 101)[None, :] <= 0) |
            (jnp.arange(-100, 101)[None, :] <= crop_cap))
    return land_mask, animal_mask, crop_mask


def _policy_core(
        params: Mapping[str, Any], inputs: Mapping[str, jax.Array],
        crop_capacity: jax.Array, crop_goals: jax.Array,
        rng_keys: jax.Array, supplied_actions: jax.Array,
        context_values: tuple[jax.Array, ...], row_ids: jax.Array,
        config: Stage25ModelConfig, mode: str, use_explicit_context: bool,
) -> dict[str, jax.Array | dict[str, jax.Array]]:
    """One jitted core shared by stochastic, greedy, and evaluation paths."""
    z = _manager_representation(params["encoder"], inputs,
                                config.manager_config, _Dropout(0.0, None), "E")
    z = z + (crop_capacity / 100.0) @ params["capacity_conditioning"]
    value = (z @ params["value_head"]["kernel"] +
             params["value_head"]["bias"])[:, 0]
    derived_context = _physical_context_jax(inputs)
    if use_explicit_context:
        observed_land, cells, placed, empty_coops, empty_pastures, unplaced = \
            context_values
    else:
        observed_land, cells, placed, empty_coops, empty_pastures, unplaced = \
            derived_context
    b = z.shape[0]
    h = jnp.zeros((b, config.d_model), dtype=jnp.float32)
    animal_targets = placed
    decoded_goals = jnp.zeros((b, _N_CROPS), dtype=jnp.int32)
    classes = []
    component_logprobs = []
    entropies = []
    validity = jnp.ones((b,), dtype=jnp.bool_)
    support_counts = []
    all_logits = []
    all_masks = []
    total_capacity = jnp.zeros((b,), dtype=jnp.int32)

    for step in range(_N_ACTIONS):
        decoder = params["recurrent_decoder"]
        q = z @ decoder["Wz"]
        h = jax.nn.gelu(q + h @ decoder["Wh"] +
                        decoder["decision_embedding"][step] + decoder["bias"],
                        approximate=False)
        projection = params["output_projections"][step]
        logits = h @ projection["kernel"] + projection["bias"]
        if step == 0:
            targets = jnp.arange(1, 5)
            support = targets[None, :] >= observed_land[:, None]
            support, _, _ = _curriculum_masks(
                config, observed_land, placed, support,
                jnp.ones((b, _N_ANIMALS, _N_ANIMAL_CLASSES), bool),
                jnp.ones((b, _N_CROPS, _N_CROP_CLASSES), bool))
            scale = decoder["state_relative_scale"][0]
            logits = logits - scale * (targets[None, :] - observed_land[:, None])
        elif step < 4:
            species = step - 1
            land_class_valid = (classes[0] >= 0) & (classes[0] < _N_LAND)
            land_target = jnp.where(land_class_valid, classes[0] + 1, 1)
            batch_index = jnp.arange(b)
            base = animal_targets
            candidate = jnp.broadcast_to(base[:, None, :],
                                         (b, _N_ANIMAL_CLASSES, _N_ANIMALS))
            for prior in range(species):
                candidate = candidate.at[:, :, prior].set(
                    classes[1 + prior][:, None])
            candidate = candidate.at[:, :, species].set(
                jnp.arange(_N_ANIMAL_CLASSES)[None, :])
            land_cells = cells[batch_index, land_target - 1]
            goose_need = jnp.maximum(
                0, candidate[:, :, 0] - placed[:, 0, None] -
                empty_coops[:, None])
            pasture_need = jnp.maximum(
                0, candidate[:, :, 1] - placed[:, 1, None] +
                candidate[:, :, 2] - placed[:, 2, None] -
                empty_pastures[:, None])
            feasible = (jnp.all(candidate >= placed[:, None, :], axis=-1) &
                        (land_cells[:, None] - goose_need - pasture_need >= 0))
            support = feasible
            _, animal_support, _ = _curriculum_masks(
                config, observed_land, placed,
                jnp.ones((b, _N_LAND), bool),
                jnp.ones((b, _N_ANIMALS, _N_ANIMAL_CLASSES), bool),
                jnp.ones((b, _N_CROPS, _N_CROP_CLASSES), bool))
            support = support & animal_support[:, species, :]
            scale = decoder["state_relative_scale"][step]
            logits = logits - scale * (
                jnp.arange(_N_ANIMAL_CLASSES)[None, :] - placed[:, species, None])
        else:
            crop_index = step - 4
            land_class_valid = (classes[0] >= 0) & (classes[0] < _N_LAND)
            land_target = jnp.where(land_class_valid, classes[0] + 1, 1)
            batch_index = jnp.arange(b)
            selected_animals = jnp.stack(classes[1:4], axis=1)
            goose_need = jnp.maximum(
                0, selected_animals[:, 0] - placed[:, 0] - empty_coops)
            pasture_need = jnp.maximum(
                0, selected_animals[:, 1] - placed[:, 1] +
                selected_animals[:, 2] - placed[:, 2] - empty_pastures)
            # The five supplied values are the pre-decision crop capacities;
            # crop support consumes their shared total without reserving
            # capacity for future crop heads.
            total_capacity = jnp.sum(crop_capacity, axis=1).astype(jnp.int32)
            residual = total_capacity - jnp.sum(decoded_goals, axis=1)
            goals = crop_goals[:, crop_index]
            deltas = jnp.arange(-100, 101)
            lower = jnp.maximum(-100, -goals)
            upper = jnp.minimum(100, residual) - goals
            support = (deltas[None, :] >= lower[:, None]) & \
                (deltas[None, :] <= upper[:, None])
            _, _, crop_support = _curriculum_masks(
                config, observed_land, placed,
                jnp.ones((b, _N_LAND), bool),
                jnp.ones((b, _N_ANIMALS, _N_ANIMAL_CLASSES), bool),
                jnp.ones((b, _N_CROPS, _N_CROP_CLASSES), bool))
            support = support & crop_support[:, crop_index, :]

        probs, log_probs, entropy, any_valid = _masked_distribution(logits, support)
        if mode == "sample":
            key = jax.vmap(lambda row_key, row_id: jax.random.fold_in(
                jax.random.fold_in(row_key, row_id), step))(rng_keys, row_ids)
            sample_logits = jnp.where(
                support, jnp.log(jnp.maximum(probs, 1e-30)), -_MAX_LOGIT)
            sampled = jax.vmap(jax.random.categorical)(key, sample_logits)
            # A malformed context must never turn into a random action. Keep
            # a deterministic sentinel for diagnostics while validity stays
            # false and the host wrapper raises explicitly.
            selected = jnp.where(any_valid, sampled, 0)
        elif mode == "greedy":
            selected = jnp.argmax(jnp.where(support, logits, -_MAX_LOGIT), axis=-1)
        else:
            selected = supplied_actions[:, step]
        class_count = ACTION_CLASS_COUNTS[step]
        chosen_valid = (selected >= 0) & (selected < class_count)
        # Safe indexing for an invalid teacher-forced class is only a
        # diagnostic sentinel; validity remains false and the host wrapper
        # rejects it. Sampled/greedy classes are already in range.
        selected_safe = jnp.where(chosen_valid, selected, 0).astype(jnp.int32)
        chosen_valid = chosen_valid & any_valid & jnp.take_along_axis(
            support, selected_safe[:, None], axis=1)[:, 0]
        selected_logprob = jnp.take_along_axis(
            log_probs, selected_safe[:, None], axis=1)[:, 0]
        selected_logprob = jnp.where(chosen_valid, selected_logprob, 0.0)
        classes.append(selected_safe)
        component_logprobs.append(selected_logprob)
        entropies.append(entropy)
        support_counts.append(jnp.sum(support, axis=1))
        all_logits.append(jnp.pad(logits, ((0, 0),
                                           (0, 201 - class_count))))
        all_masks.append(jnp.pad(support, ((0, 0),
                                           (0, 201 - class_count))))
        validity = validity & chosen_valid & any_valid
        if step < 4:
            if step == 0:
                animal_targets = animal_targets
            elif step < 4:
                animal_targets = animal_targets.at[:, step - 1].set(selected_safe)
        else:
            crop_index = step - 4
            updated = crop_goals[:, crop_index] + selected_safe - _CROP_DELTA_OFFSET
            decoded_goals = decoded_goals.at[:, crop_index].set(updated)
        embedding = params["action_embeddings"][step]
        h = h + embedding[selected_safe]

    component_logprobs_array = jnp.stack(component_logprobs, axis=1)
    classes_array = jnp.stack(classes, axis=1)
    entropy_array = jnp.stack(entropies, axis=1)
    return {
        "classes": classes_array,
        "component_logprobs": component_logprobs_array,
        "joint_logprob": jnp.sum(component_logprobs_array, axis=1),
        "conditional_entropies": entropy_array,
        "prefix_entropy_surrogate": jnp.sum(entropy_array, axis=1),
        "logits": jnp.stack(all_logits, axis=1),
        "masks": jnp.stack(all_masks, axis=1),
        "decoded_goals": decoded_goals,
        "value": value,
        "validity": validity,
        "valid": validity,
        "diagnostics": {
            "observed_land": observed_land,
            "placed_animals": placed,
            "unplaced_animals": unplaced,
            "total_capacity": total_capacity,
            "support_counts": jnp.stack(support_counts, axis=1),
            "all_invalid_steps": jnp.stack(
                [jnp.asarray(x == 0) for x in support_counts], axis=1),
        },
    }


_stage25_jit = jax.jit(
    _policy_core,
    static_argnames=("config", "mode", "use_explicit_context"))


def _call_policy(
        params: Mapping[str, Any], inputs: Mapping[str, Any],
        config: Stage25ModelConfig, *, mode: str,
        rng_keys: Any = None, actions: Any = None,
        physical_contexts: Any = None, crop_capacity: Any = None,
        row_ids: Any = None,
        reject_invalid: bool = True,
) -> dict[str, Any]:
    prepared, capacity, goals, batch = _host_inputs(
        inputs, config, crop_capacity=crop_capacity)
    if rng_keys is None:
        keys = jnp.zeros((batch, 2), dtype=jnp.uint32)
    else:
        keys = jnp.asarray(rng_keys, dtype=jnp.uint32)
        if tuple(keys.shape) != (batch, 2):
            raise ValueError("rng_keys must have shape [B, 2]")
    context_values = _host_contexts(physical_contexts, batch)
    use_explicit_context = context_values is not None
    if context_values is None:
        context_values = tuple(jnp.zeros((batch,) if index in (0, 3, 4)
                                         else (batch, 3) if index in (2, 5)
                                         else (batch, 4,), dtype=jnp.int32)
                               for index in range(6))
    if row_ids is None:
        row_id_array = jnp.zeros((batch,), dtype=jnp.int32)
    else:
        row_id_array = jnp.asarray(row_ids, dtype=jnp.int32)
        if tuple(row_id_array.shape) != (batch,):
            raise ValueError("row_ids must have shape [B]")
    if actions is None:
        supplied = jnp.zeros((batch, _N_ACTIONS), dtype=jnp.int32)
    else:
        supplied_array = np.asarray(actions)
        if supplied_array.shape != (batch, _N_ACTIONS):
            raise ValueError("actions must have shape [B, 9]")
        if not np.issubdtype(supplied_array.dtype, np.integer):
            if (not np.issubdtype(supplied_array.dtype, np.floating)
                    or not np.all(np.isfinite(supplied_array))
                    or not np.all(supplied_array == np.floor(supplied_array))):
                raise ValueError("actions must contain integer class indices")
        supplied = jnp.asarray(supplied_array, dtype=jnp.int32)
        if reject_invalid and (np.any(supplied_array < 0) or
                               np.any(supplied_array >=
                                      np.asarray(ACTION_CLASS_COUNTS))):
            raise ValueError("actions contain a class outside its vocabulary")
    result = _stage25_jit(params, prepared, capacity, goals, keys, supplied,
                          context_values, row_id_array, config, mode,
                          use_explicit_context)
    if actions is not None and reject_invalid:
        if not bool(np.all(np.asarray(result["validity"]))):
            raise ValueError("actions contain a physically unsupported class")
    return result


def stochastic_act(
        params: Mapping[str, Any], inputs: Mapping[str, Any],
        config: Stage25ModelConfig, rng: Any = None, rng_keys: Any = None,
        *, reject_invalid: bool = True,
        physical_contexts: Any = None, crop_capacity: Any = None,
        row_ids: Any = None,
) -> dict[str, Any]:
    """Sample the nine classes using one stable [B, 2] key per row."""
    if rng_keys is None:
        rng_keys = rng
    return _call_policy(params, inputs, config, mode="sample",
                        rng_keys=rng_keys, reject_invalid=reject_invalid,
                        physical_contexts=physical_contexts,
                        crop_capacity=crop_capacity, row_ids=row_ids)


def greedy_act(
        params: Mapping[str, Any], inputs: Mapping[str, Any],
        config: Stage25ModelConfig,
        *, physical_contexts: Any = None, crop_capacity: Any = None,
        row_ids: Any = None,
) -> dict[str, Any]:
    """Greedy decode using the exact evaluation support masks."""
    return _call_policy(params, inputs, config, mode="greedy",
                        physical_contexts=physical_contexts,
                        crop_capacity=crop_capacity, row_ids=row_ids)


def evaluate_actions(
        params: Mapping[str, Any], inputs: Mapping[str, Any],
        config: Stage25ModelConfig | Any = None, actions: Any = None,
        classes: Any = None,
        *, reject_invalid: bool = True, physical_contexts: Any = None,
        crop_capacity: Any = None, row_ids: Any = None,
) -> dict[str, Any]:
    """Evaluate supplied classes with shared teacher-forced recurrence."""
    # Accept both ``(params, inputs, config, actions)`` and the natural
    # ``(params, inputs, actions, config)`` positional spelling.
    if not isinstance(config, Stage25ModelConfig):
        config, actions = actions, config
    if not isinstance(config, Stage25ModelConfig):
        raise TypeError("config must be Stage25ModelConfig")
    if actions is None:
        actions = classes
    if actions is None:
        raise ValueError("evaluate_actions requires classes/actions")
    return _call_policy(params, inputs, config, mode="eval", actions=actions,
                        reject_invalid=reject_invalid,
                        physical_contexts=physical_contexts,
                        crop_capacity=crop_capacity, row_ids=row_ids)

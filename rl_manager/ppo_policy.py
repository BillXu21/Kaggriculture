"""PPO V0 policy over the issue-#8 JAX E manager (issue #9 B1/B2/B3).

Architecture decisions implemented here (packet root decisions 3/4/6/7/9):

- The policy trunk is a MUTABLE copy of the E base params; the frozen E
  snapshot is kept as a separate immutable tree and is never optimized.
- A small independently-initialized value head maps the final manager
  representation (via the additive `bc_manager_jax` representation seam,
  never a duplicated Transformer) to one scalar per row.
- Sell quantities are ALWAYS computed from the full immutable frozen-E
  snapshot with the exact issue-#8 rounding rule; sell quantity has no
  logprob slot, and the mutable base sell-quantity head receives exactly
  zero optimizer updates (masked AdamW: no gradient step AND no decay).
- Action distribution given the manager representation: 17 conditionally
  independent categoricals (5 crop 0..100, 3 animal 0..100, 1 land stored
  0..3 / decoded 1..4, 5 fertilizer 0..100, 3 care 0..100) plus 54
  sell-presence Bernoullis. No masks, no autoregression. Joint logprob is
  the raw SUM of all component logprobs; entropy uses the raw sum with six
  group means reported.
- Sampling vmaps over explicit per-row decision seeds (fold_in of one root
  key); there is no Python loop over examples inside JAX and no dependence
  on env scheduling order. Deterministic mode is argmax / logit>0 and
  reproduces the frozen JAX-E decode exactly before any policy drift.
"""

from __future__ import annotations

import dataclasses
from typing import Mapping

import jax
import jax.numpy as jnp
import numpy as np
import optax

from bc_manager.model import (
    NUM_ANIMALS,
    NUM_CROPS,
    NUM_LAND_CLASSES,
    NUM_PRODUCTS,
    SELL_BIN_COUNT,
    SELL_PRESENCE_CELLS,
)
from bc_manager_jax.model import (
    ECONOMIC_CONTEXT_KEY,
    ManagerConfig,
    OWN_INPUT_KEYS,
    _forward_eval,
    _forward_eval_with_representation,
    _prepare_inputs,
    empty_params,
    resolve_model_variant,
)

from rl_manager.decode import ACTION_TENSOR_SHAPES

#: Land is stored internally as index 0..3 and decoded to plan value +1.
LAND_INDEX_OFFSET = 1

#: Categorical group -> component count (issue #9 B2 list).
CATEGORICAL_GROUP_SIZES: dict[str, int] = {
    "crop": NUM_CROPS,
    "animal": NUM_ANIMALS,
    "land": NUM_LAND_CLASSES,
    "fertilizer": NUM_CROPS,
    "care": NUM_ANIMALS,
}

#: All stochastic PPO groups: five categorical groups + sell presence.
PPO_GROUPS: tuple[str, ...] = (
    "crop", "animal", "land", "fertilizer", "care", "sell_presence")


@dataclasses.dataclass(frozen=True)
class PPOConfig:
    """Conventional inspectable PPO plumbing defaults (issue #9 B5).

    These are NOT tuned hyperparameters; every field is configurable and
    the defaults mirror the issue text (clip .2, value coef .5, entropy
    coef .01, global grad clip 1.0, gamma .99, lambda .95). KL-to-frozen
    is exposed but disabled by default.
    """

    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    gradient_clip: float = 1.0
    lr: float = 3e-4
    weight_decay: float = 1e-2
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    epochs: int = 4
    minibatch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    normalize_advantages: bool = True
    advantage_epsilon: float = 1e-8
    kl_to_frozen_coef: float = 0.0
    value_init_scale: float = 0.01

    def __post_init__(self) -> None:
        if self.clip_eps <= 0.0:
            raise ValueError(f"clip_eps must be positive, got {self.clip_eps}")
        if self.value_coef < 0.0 or self.entropy_coef < 0.0:
            raise ValueError("value_coef/entropy_coef must be >= 0")
        if self.gradient_clip <= 0.0:
            raise ValueError(
                f"gradient_clip must be positive, got {self.gradient_clip}")
        if self.lr <= 0.0:
            raise ValueError(f"lr must be positive, got {self.lr}")
        if self.weight_decay < 0.0:
            raise ValueError(
                f"weight_decay must be >= 0, got {self.weight_decay}")
        if not 0.0 <= self.beta1 < 1.0 or not 0.0 <= self.beta2 < 1.0:
            raise ValueError("betas must be in [0, 1)")
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")
        if self.minibatch_size < 1:
            raise ValueError(
                f"minibatch_size must be >= 1, got {self.minibatch_size}")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError(f"gamma must be in [0, 1], got {self.gamma}")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError(
                f"gae_lambda must be in [0, 1], got {self.gae_lambda}")
        if self.kl_to_frozen_coef < 0.0:
            raise ValueError("kl_to_frozen_coef must be >= 0")
        if self.value_init_scale < 0.0:
            raise ValueError("value_init_scale must be >= 0")


# ------------------------------------------------------ distribution math


def distribution_logits(outputs: Mapping[str, jax.Array]) -> dict[str, jax.Array]:
    """Flatten raw head outputs into per-component logits.

    Categorical logits keep their `[B, K, classes]` layout; the 54
    sell-presence Bernoulli logits are flattened to `[B, 54]`.
    """
    b = outputs["crop_logits"].shape[0]
    return {
        "crop": outputs["crop_logits"],
        "animal": outputs["animal_logits"],
        "land": outputs["land_logits"],
        "fertilizer": outputs["fertilizer_logits"],
        "care": outputs["care_logits"],
        "sell_presence": jnp.reshape(outputs["sell_presence_logits"],
                                     (b, SELL_PRESENCE_CELLS)),
    }


def _sum_components(per_row: jax.Array) -> jax.Array:
    """Sum component axes into [B]; handles both [B, K, ...] and [B]."""
    return jnp.sum(per_row.reshape(per_row.shape[0], -1), axis=-1)


def categorical_logprobs(logits: jax.Array,
                         actions: jax.Array) -> jax.Array:
    """Per-row summed categorical logprob over the component axis.

    `logits` [B, K, classes] (or [B, classes] for the single land
    categorical), `actions` matching minus the class axis -> [B].
    """
    logp = jax.nn.log_softmax(logits, axis=-1)
    gathered = jnp.take_along_axis(logp, actions[..., None], axis=-1)[..., 0]
    return _sum_components(gathered)


def categorical_entropy(logits: jax.Array) -> jax.Array:
    """Per-row summed categorical entropy over the component axis."""
    logp = jax.nn.log_softmax(logits, axis=-1)
    return _sum_components(-jnp.sum(jnp.exp(logp) * logp, axis=-1))


def bernoulli_logprobs(logits: jax.Array, bits: jax.Array) -> jax.Array:
    """Per-row summed Bernoulli logprob over the cell axis."""
    lp1 = jax.nn.log_sigmoid(logits)
    lp0 = jax.nn.log_sigmoid(-logits)
    return jnp.sum(bits * lp1 + (1.0 - bits) * lp0, axis=-1)


def bernoulli_entropy(logits: jax.Array) -> jax.Array:
    """Per-row summed Bernoulli entropy over the cell axis."""
    lp1 = jax.nn.log_sigmoid(logits)
    lp0 = jax.nn.log_sigmoid(-logits)
    p1 = jnp.exp(lp1)
    return -jnp.sum(p1 * lp1 + (1.0 - p1) * lp0, axis=-1)


def categorical_kl_to_frozen(logits: jax.Array,
                             frozen_logits: jax.Array) -> jax.Array:
    """Per-row summed analytic KL(current || frozen) for categoricals."""
    logp = jax.nn.log_softmax(logits, axis=-1)
    logq = jax.nn.log_softmax(frozen_logits, axis=-1)
    return _sum_components(jnp.exp(logp) * (logp - logq))


def bernoulli_kl_to_frozen(logits: jax.Array,
                           frozen_logits: jax.Array) -> jax.Array:
    """Per-row summed analytic KL(current || frozen) for Bernoullis."""
    lp = jax.nn.log_sigmoid(logits)
    lq = jax.nn.log_sigmoid(frozen_logits)
    p = jnp.exp(lp)
    q = jnp.exp(lq)
    return jnp.sum(p * (lp - lq) + (1.0 - p) * (jnp.log1p(-p) - jnp.log1p(-q)),
                   axis=-1)


def action_index_tensors(action_tensors: Mapping[str, object],
                         batch_size: int) -> dict[str, jax.Array]:
    """Stage-A action tensors -> internal index tensors (validated eagerly).

    Land plan values 1..4 become indices 0..3; sell presence [B, 6, 9]
    flattens to [B, 54] bits. Ranges are checked loudly on host.
    """
    def checked(name: str, array: np.ndarray, high: int) -> np.ndarray:
        if np.any(array < 0) or np.any(array > high):
            raise ValueError(
                f"action {name!r} outside valid range [0, {high}]")
        return array

    crop = np.asarray(action_tensors["crop"], dtype=np.int64)
    animal = np.asarray(action_tensors["animal"], dtype=np.int64)
    land = np.asarray(action_tensors["land"], dtype=np.int64) - LAND_INDEX_OFFSET
    fertilizer = np.asarray(action_tensors["fertilizer"], dtype=np.int64)
    care = np.asarray(action_tensors["care"], dtype=np.int64)
    presence = np.asarray(action_tensors["sell_presence"], dtype=np.int64)
    if crop.shape != (batch_size, NUM_CROPS) \
            or animal.shape != (batch_size, NUM_ANIMALS) \
            or fertilizer.shape != (batch_size, NUM_CROPS) \
            or care.shape != (batch_size, NUM_ANIMALS):
        raise ValueError("action tensor shapes do not match the Stage-A schema")
    if land.shape != (batch_size,) or \
            presence.shape != (batch_size, NUM_PRODUCTS, SELL_BIN_COUNT):
        raise ValueError("action tensor shapes do not match the Stage-A schema")
    checked("crop", crop, 100)
    checked("animal", animal, 100)
    checked("land", land, NUM_LAND_CLASSES - 1)
    checked("fertilizer", fertilizer, 100)
    checked("care", care, 100)
    checked("sell_presence", presence, 1)
    return {
        "crop": jnp.asarray(crop, dtype=jnp.int32),
        "animal": jnp.asarray(animal, dtype=jnp.int32),
        "land": jnp.asarray(land, dtype=jnp.int32),
        "fertilizer": jnp.asarray(fertilizer, dtype=jnp.int32),
        "care": jnp.asarray(care, dtype=jnp.int32),
        "sell_presence": jnp.asarray(
            presence.reshape(batch_size, SELL_PRESENCE_CELLS),
            dtype=jnp.int32),
    }


def group_logprob_and_entropy(
        logits: dict[str, jax.Array],
        indices: dict[str, jax.Array]) -> dict[str, jax.Array]:
    """Per-group logprob/entropy ([B] each) plus raw-sum totals."""
    groups_logprob = {
        "crop": categorical_logprobs(logits["crop"], indices["crop"]),
        "animal": categorical_logprobs(logits["animal"], indices["animal"]),
        "land": categorical_logprobs(logits["land"], indices["land"]),
        "fertilizer": categorical_logprobs(logits["fertilizer"],
                                           indices["fertilizer"]),
        "care": categorical_logprobs(logits["care"], indices["care"]),
        "sell_presence": bernoulli_logprobs(logits["sell_presence"],
                                            indices["sell_presence"]),
    }
    groups_entropy = {
        "crop": categorical_entropy(logits["crop"]),
        "animal": categorical_entropy(logits["animal"]),
        "land": categorical_entropy(logits["land"]),
        "fertilizer": categorical_entropy(logits["fertilizer"]),
        "care": categorical_entropy(logits["care"]),
        "sell_presence": bernoulli_entropy(logits["sell_presence"]),
    }
    total_logprob = groups_logprob["crop"] + groups_logprob["animal"] \
        + groups_logprob["land"] + groups_logprob["fertilizer"] \
        + groups_logprob["care"] + groups_logprob["sell_presence"]
    total_entropy = groups_entropy["crop"] + groups_entropy["animal"] \
        + groups_entropy["land"] + groups_entropy["fertilizer"] \
        + groups_entropy["care"] + groups_entropy["sell_presence"]
    return {"logprob_groups": groups_logprob,
            "entropy_groups": groups_entropy,
            "logprob_total": total_logprob,
            "entropy_total": total_entropy}


def kl_to_frozen(logits: dict[str, jax.Array],
                 frozen_logits: dict[str, jax.Array]) -> jax.Array:
    """Per-row summed analytic KL(current || frozen) across all groups."""
    total = categorical_kl_to_frozen(logits["crop"], frozen_logits["crop"])
    for name in ("animal", "land", "fertilizer", "care"):
        total += categorical_kl_to_frozen(logits[name], frozen_logits[name])
    total += bernoulli_kl_to_frozen(logits["sell_presence"],
                                    frozen_logits["sell_presence"])
    return total


# ------------------------------------------------------------- sampling


def deterministic_action_indices(
        logits: dict[str, jax.Array]) -> dict[str, jax.Array]:
    """Argmax counts / land argmax / presence logit>0 — exact frozen decode."""
    return {
        "crop": jnp.argmax(logits["crop"], axis=-1),
        "animal": jnp.argmax(logits["animal"], axis=-1),
        "land": jnp.argmax(logits["land"], axis=-1),
        "fertilizer": jnp.argmax(logits["fertilizer"], axis=-1),
        "care": jnp.argmax(logits["care"], axis=-1),
        "sell_presence": (logits["sell_presence"] > 0.0).astype(jnp.int32),
    }


def sample_action_indices(logits: dict[str, jax.Array],
                          rng: jax.Array,
                          decision_seeds: jax.Array | None = None
                          ) -> dict[str, jax.Array]:
    """Vmapped stochastic sampling keyed by explicit per-row decision seeds.

    `rng` is the root key; each row's key is `fold_in(rng, decision_seed)`
    so samples attach to row identity (seed), never batch position or env
    scheduling order. One vmap over rows; no Python example loop.
    """
    seeds = (jnp.arange(logits["crop"].shape[0], dtype=jnp.uint32)
             if decision_seeds is None else jnp.asarray(
                 decision_seeds, dtype=jnp.uint32))
    if seeds.ndim != 1 or seeds.shape[0] != logits["crop"].shape[0]:
        raise ValueError(
            "decision_seeds must be one uint32 seed per policy row")
    keys = jax.vmap(lambda s: jax.random.fold_in(rng, s))(seeds)

    def one(key, crop_l, animal_l, land_l, fert_l, care_l, sell_l):
        k = jax.random.split(key, 6)
        return (jax.random.categorical(k[0], crop_l),
                jax.random.categorical(k[1], animal_l),
                jax.random.categorical(k[2], land_l),
                jax.random.categorical(k[3], fert_l),
                jax.random.categorical(k[4], care_l),
                jax.random.bernoulli(
                    k[5], jax.nn.sigmoid(sell_l)).astype(jnp.int32))

    (crop, animal, land, fert, care, sell) = jax.vmap(one)(
        keys, logits["crop"], logits["animal"], logits["land"],
        logits["fertilizer"], logits["care"], logits["sell_presence"])
    return {"crop": crop, "animal": animal, "land": land,
            "fertilizer": fert, "care": care, "sell_presence": sell}


# ------------------------------------------------------- params assembly


def value_head_template(config: ManagerConfig) -> dict[str, jax.Array]:
    """Canonical value-head shape spec: kernel [d_model, 1], bias [1]."""
    return {"kernel": jnp.zeros((config.d_model, 1), dtype=jnp.float32),
            "bias": jnp.zeros((1,), dtype=jnp.float32)}


def combined_params_template(config: ManagerConfig,
                             model_variant: str = "E") -> dict:
    """Canonical combined PPO param tree: mutable base + value head."""
    return {"base": empty_params(config, model_variant),
            "value": value_head_template(config)}


def init_value_head(config: ManagerConfig, key: jax.Array,
                    scale: float) -> dict[str, jax.Array]:
    """Independently small value-head init; never perturbs the trunk."""
    kernel = jax.random.normal(key, (config.d_model, 1),
                               dtype=jnp.float32) * scale
    return {"kernel": kernel, "bias": jnp.zeros((1,), dtype=jnp.float32)}


def _leaf_path(tokens) -> list[str]:
    return [str(getattr(entry, "key", None)
                if getattr(entry, "key", None) is not None else entry.idx)
            for entry in tokens]


def frozen_leaf_mask(combined_params: Mapping) -> dict:
    """Boolean pytree: False exactly at the mutable base sell-quantity head.

    Masked leaves receive EXACTLY zero optimizer updates — no gradient step
    AND no AdamW decoupled weight decay (optax.masked zeroes the whole
    update for those leaves).
    """
    flat, treedef = jax.tree_util.tree_flatten_with_path(combined_params)
    flags = []
    for tokens, _leaf in flat:
        parts = _leaf_path(tokens)
        # Only the mutable base's heads/sell_quantity subtree is frozen;
        # the value head and everything else trains.
        flags.append(not ("base" in parts and "sell_quantity" in parts))
    return jax.tree_util.tree_unflatten(treedef, flags)


def make_ppo_optimizer(ppo_config: PPOConfig, mask: Mapping):
    """Global-norm clip BEFORE masked AdamW (decay masked out too)."""
    return optax.chain(
        optax.clip_by_global_norm(ppo_config.gradient_clip),
        optax.masked(
            optax.adamw(learning_rate=ppo_config.lr, b1=ppo_config.beta1,
                        b2=ppo_config.beta2, eps=ppo_config.eps,
                        weight_decay=ppo_config.weight_decay),
            mask))


def value_from_representation(representation: jax.Array,
                              value_params: Mapping) -> jax.Array:
    """One scalar value per row from the final manager representation."""
    return jnp.squeeze(
        representation @ value_params["kernel"] + value_params["bias"],
        axis=-1)


def enforce_own_only_e_inputs(inputs: Mapping[str, np.ndarray]) -> None:
    """Own-only E contract at the RL seam (mirrors JaxEPlanPolicy)."""
    unknown = sorted(set(inputs.keys()) - set(OWN_INPUT_KEYS)
                     - {ECONOMIC_CONTEXT_KEY})
    if unknown:
        raise ValueError(
            f"own-only E contract violated: unknown/leaked input keys "
            f"{unknown}; opponent-public arrays and metadata must never "
            f"reach the E policy")


class PPOPolicy:
    """PPO V0 policy: mutable E trunk/value head + immutable frozen snapshot."""

    def __init__(self, frozen_params: Mapping, config: ManagerConfig, *,
                 seed: int, model_variant: str = "E",
                 ppo_config: PPOConfig | None = None) -> None:
        variant = resolve_model_variant(model_variant)
        if variant != "E":
            raise ValueError(
                "PPO V0 is the own-only E contract; got variant "
                f"{variant!r}")
        self._variant = variant
        self._config = config
        self.ppo_config = ppo_config or PPOConfig()
        # Immutable frozen snapshot: independent copies, never written to.
        self.frozen_params = jax.tree_util.tree_map(
            lambda leaf: jnp.array(leaf), frozen_params)
        value_key, rng = jax.random.split(jax.random.PRNGKey(seed))
        # Mutable copy of the E base params + independent small value head.
        mutable_base = jax.tree_util.tree_map(
            lambda leaf: jnp.array(leaf), frozen_params)
        self.params = {
            "base": mutable_base,
            "value": init_value_head(config, value_key,
                                     self.ppo_config.value_init_scale),
        }
        self.rng = rng

    # -------------------------------------------------------------- act
    def act(self, inputs: Mapping[str, np.ndarray], *,
            deterministic: bool = False,
            rng: jax.Array | None = None,
            decision_seeds: jax.Array | None = None
            ) -> dict[str, np.ndarray]:
        """Batched action sampling/decoding for one contiguous request batch.

        Deterministic mode reproduces the frozen JAX-E decode exactly while
        the policy equals its initialization. Quantities always come from
        the immutable frozen snapshot gated by the (sampled or argmax)
        presence bits, using the exact issue-#8 round-half-up rule.
        """
        enforce_own_only_e_inputs(inputs)
        if not deterministic and rng is None:
            raise ValueError("stochastic act() requires an explicit rng key")
        prepared = _prepare_inputs(inputs)
        frozen_outputs = _forward_eval(self.frozen_params, prepared,
                                       self._config, self._variant)
        mut_outputs, representation = _forward_eval_with_representation(
            self.params["base"], prepared, self._config, self._variant)
        logits = distribution_logits(mut_outputs)
        if deterministic:
            indices = deterministic_action_indices(logits)
        else:
            indices = sample_action_indices(logits, rng, decision_seeds)

        batch = int(prepared["board_kind"].shape[0])
        presence_bits = np.asarray(indices["sell_presence"], dtype=np.int64) \
            .reshape(batch, NUM_PRODUCTS, SELL_BIN_COUNT)
        # Frozen quantities: exact issue-#8 rounding from the immutable
        # snapshot, gated by THIS policy's presence decision.
        quantity_log1p = np.asarray(frozen_outputs["sell_quantity_log1p"])
        quantity = np.floor(
            np.expm1(np.clip(quantity_log1p, 0.0, None)) + 0.5)
        sell_quantity = np.where(presence_bits.astype(bool), quantity, 0) \
            .astype(np.int16)
        action_tensors = {
            "crop": np.asarray(indices["crop"], dtype=np.int16),
            "animal": np.asarray(indices["animal"], dtype=np.int16),
            "land": np.asarray(indices["land"], dtype=np.int16)
                    + LAND_INDEX_OFFSET,
            "fertilizer": np.asarray(indices["fertilizer"], dtype=np.int16),
            "care": np.asarray(indices["care"], dtype=np.int16),
            "sell_presence": presence_bits.astype(np.uint8),
            "sell_quantity": sell_quantity,
        }
        stats = group_logprob_and_entropy(logits, indices)
        value = np.asarray(value_from_representation(
            representation, self.params["value"]), dtype=np.float32)
        return {
            "action_tensors": action_tensors,
            "logprob_groups": {name: np.asarray(array, dtype=np.float32)
                               for name, array
                               in stats["logprob_groups"].items()},
            "logprob_total": np.asarray(stats["logprob_total"],
                                        dtype=np.float32),
            "entropy_groups": {name: np.asarray(array, dtype=np.float32)
                               for name, array
                               in stats["entropy_groups"].items()},
            "entropy_total": np.asarray(stats["entropy_total"],
                                        dtype=np.float32),
            "value": value,
            "batch_size": batch,
        }

    # ------------------------------------------------ evaluate actions
    def evaluate_actions(self, inputs: Mapping[str, np.ndarray],
                         action_tensors: Mapping[str, object]) -> dict:
        """Exact stored-action logprob recomputation under current params."""
        enforce_own_only_e_inputs(inputs)
        prepared = _prepare_inputs(inputs)
        mut_outputs, representation = _forward_eval_with_representation(
            self.params["base"], prepared, self._config, self._variant)
        logits = distribution_logits(mut_outputs)
        batch = int(prepared["board_kind"].shape[0])
        indices = action_index_tensors(action_tensors, batch)
        stats = group_logprob_and_entropy(logits, indices)
        value = np.asarray(value_from_representation(
            representation, self.params["value"]), dtype=np.float32)
        return {
            "logprob_groups": {name: np.asarray(array, dtype=np.float32)
                               for name, array
                               in stats["logprob_groups"].items()},
            "logprob_total": np.asarray(stats["logprob_total"],
                                        dtype=np.float32),
            "entropy_groups": {name: np.asarray(array, dtype=np.float32)
                               for name, array
                               in stats["entropy_groups"].items()},
            "entropy_total": np.asarray(stats["entropy_total"],
                                        dtype=np.float32),
            "value": value,
        }

    # -------------------------------------------------- frozen helpers
    def frozen_decode(self, inputs: Mapping[str, np.ndarray]) -> dict:
        """Frozen-snapshot forward outputs (host NumPy) for diagnostics."""
        enforce_own_only_e_inputs(inputs)
        prepared = _prepare_inputs(inputs)
        outputs = _forward_eval(self.frozen_params, prepared, self._config,
                                self._variant)
        return {name: np.asarray(array) for name, array in outputs.items()}

    def parity_check_deterministic(self, inputs: Mapping[str, np.ndarray],
                                   expected_action_tensors: Mapping) -> None:
        """Fail loud unless deterministic decode matches an exact reference."""
        result = self.act(inputs, deterministic=True)["action_tensors"]
        for name in ACTION_TENSOR_SHAPES:
            if not np.array_equal(result[name],
                                  np.asarray(expected_action_tensors[name])):
                raise ValueError(
                    f"deterministic decode drifted from frozen JAX-E decode "
                    f"in action tensor {name!r}")


__all__ = [
    "CATEGORICAL_GROUP_SIZES",
    "LAND_INDEX_OFFSET",
    "PPOConfig",
    "PPO_GROUPS",
    "PPOPolicy",
    "action_index_tensors",
    "bernoulli_entropy",
    "bernoulli_kl_to_frozen",
    "bernoulli_logprobs",
    "categorical_entropy",
    "categorical_kl_to_frozen",
    "categorical_logprobs",
    "combined_params_template",
    "deterministic_action_indices",
    "distribution_logits",
    "enforce_own_only_e_inputs",
    "frozen_leaf_mask",
    "group_logprob_and_entropy",
    "init_value_head",
    "kl_to_frozen",
    "make_ppo_optimizer",
    "sample_action_indices",
    "value_from_representation",
    "value_head_template",
]

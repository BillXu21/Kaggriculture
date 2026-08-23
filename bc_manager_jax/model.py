"""Pure-JAX mirror of `bc_manager.model.DailyManagerTransformer`.

A faithful, eval-exact reimplementation of the PyTorch daily-manager tile
Transformer: shared tile encoder over the own 100-tile board (plus optional
opponent PUBLIC board with role embeddings), five compact global tokens, a
norm-first GELU Transformer encoder, and seven structured output heads.

Parameter mapping contract (PyTorch -> JAX):

- Every `nn.Linear` weight `[out, in]` becomes a JAX `kernel [in, out]`
  (explicit transpose done once in `bc_manager_jax.checkpoint`); bias maps
  unchanged.
- The packed self-attention `in_proj_weight [3d, d]` / `in_proj_bias [3d]`
  stay PACKED as `qkv_kernel [d, 3d]` / `qkv_bias [3d]`; the chunk order is
  q | k | v exactly as `F.multi_head_attention_forward` expects.
- Embeddings, LayerNorm weight/bias, and head parameters map unchanged.
- `manager_token [1, 1, d]` is stored squeezed as `(d,)`.
- Non-persistent buffers (`numeric_scales`) are recomputed from constants.

Eval-mode forward is numerically exact against the PyTorch module (see
`tests/test_bc_manager_jax_parity.py`). Dropout uses the same placement as
the PyTorch modules (after tile/global MLP GELU, on attention weights, after
attention output projection, after FFN second linear) and is disabled unless
`training=True` with an explicit PRNG key. There is no train step here;
that is stage 2.
"""

from __future__ import annotations

import dataclasses
from typing import Mapping

import jax
import jax.numpy as jnp
import numpy as np

from bc_manager.constants import (
    BOARD_BOOL_FIELDS,
    BOARD_NUMERIC_FIELDS,
    BOARD_SIZE,
    MAX_HANDS,
    RESOURCE_ORDER,
    SHOP_VOCAB,
    TOTAL_DAYS,
)
from bc_manager.model import (
    BOARD_NUMERIC_SCALES,
    BOARD_SIDE,
    GLOBAL_TOKEN_NAMES,
    NULLABLE_TIMING_CHANNELS,
    NUM_ANIMALS,
    NUM_ANIMAL_IDS,
    NUM_CROPS,
    NUM_CROP_IDS,
    NUM_LAND_CLASSES,
    NUM_PRODUCTS,
    NUM_TILE_KINDS,
    OPPONENT_PUBLIC_INPUT_KEYS,
    OWN_INPUT_KEYS,
    SELF_RESOURCE_DIM,
    SELL_BIN_COUNT,
    SELL_PRESENCE_CELLS,
    MARKET_DIM,
    TOWN_DIM,
    LABOR_DIM,
)

_LAYER_NORM_EPS = 1e-5  # torch.nn.LayerNorm default; must match source


@dataclasses.dataclass(frozen=True)
class ManagerConfig:
    """Static configuration mirroring `bc_manager.model.ManagerConfig`."""

    d_model: int = 128
    num_layers: int = 4
    num_heads: int = 4
    ffn_dim: int = 384
    dropout: float = 0.1
    count_max: int = 100
    include_opponent_board: bool = False

    def __post_init__(self) -> None:
        if self.d_model <= 0 or self.num_layers <= 0 or self.num_heads <= 0 \
                or self.ffn_dim <= 0:
            raise ValueError(
                "d_model/num_layers/num_heads/ffn_dim must be positive")
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model={self.d_model} must be divisible by "
                f"num_heads={self.num_heads}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.count_max < 1:
            raise ValueError(f"count_max must be >= 1, got {self.count_max}")

    @property
    def count_classes(self) -> int:
        return self.count_max + 1

    @property
    def token_count(self) -> int:
        opponent_tiles = BOARD_SIZE if self.include_opponent_board else 0
        return 1 + BOARD_SIZE + len(GLOBAL_TOKEN_NAMES) + opponent_tiles


def tiny_manager_config(**overrides) -> ManagerConfig:
    """Fast CPU test configuration (mirrors `bc_manager.model`)."""
    params = dict(d_model=16, num_layers=1, num_heads=1, ffn_dim=32,
                  dropout=0.0, include_opponent_board=False)
    params.update(overrides)
    return ManagerConfig(**params)


# --------------------------------------------------------------- params


def _linear(in_dim: int, out_dim: int) -> dict[str, jax.Array]:
    return {"kernel": jnp.zeros((in_dim, out_dim), dtype=jnp.float32),
            "bias": jnp.zeros((out_dim,), dtype=jnp.float32)}


def empty_params(config: ManagerConfig) -> dict:
    """Zero-initialized parameter pytree; also the canonical shape spec."""
    d = config.d_model
    feature_dim = (
        5 * d  # kind + crop + animal + row + col embeddings
        + len(BOARD_NUMERIC_FIELDS) + len(NULLABLE_TIMING_CHANNELS)
        + len(BOARD_BOOL_FIELDS)
        + 4  # presence mask channels
    )

    def mlp(in_dim: int) -> dict[str, dict[str, jax.Array]]:
        return {"0": _linear(in_dim, d), "3": _linear(d, d)}

    layers = []
    for _ in range(config.num_layers):
        layers.append({
            "qkv_kernel": jnp.zeros((d, 3 * d), dtype=jnp.float32),
            "qkv_bias": jnp.zeros((3 * d,), dtype=jnp.float32),
            "out_kernel": jnp.zeros((d, d), dtype=jnp.float32),
            "out_bias": jnp.zeros((d,), dtype=jnp.float32),
            "linear1": _linear(d, config.ffn_dim),
            "linear2": _linear(config.ffn_dim, d),
            "norm1_weight": jnp.zeros((d,), dtype=jnp.float32),
            "norm1_bias": jnp.zeros((d,), dtype=jnp.float32),
            "norm2_weight": jnp.zeros((d,), dtype=jnp.float32),
            "norm2_bias": jnp.zeros((d,), dtype=jnp.float32),
        })

    c = config.count_classes
    return {
        "manager_token": jnp.zeros((d,), dtype=jnp.float32),
        "role_embedding": jnp.zeros((2, d), dtype=jnp.float32),
        "tile_encoder": {
            "kind_embedding": jnp.zeros((NUM_TILE_KINDS, d), jnp.float32),
            "crop_embedding": jnp.zeros((NUM_CROP_IDS, d), jnp.float32),
            "animal_embedding": jnp.zeros((NUM_ANIMAL_IDS, d), jnp.float32),
            "row_embedding": jnp.zeros((BOARD_SIDE, d), jnp.float32),
            "col_embedding": jnp.zeros((BOARD_SIDE, d), jnp.float32),
            "project": {"0": _linear(feature_dim, d), "3": _linear(d, d)},
        },
        "global_encoders": {
            "day_embedding": jnp.zeros((TOTAL_DAYS, d), jnp.float32),
            "self_resource": mlp(SELF_RESOURCE_DIM),
            "market": mlp(MARKET_DIM),
            "town": mlp(TOWN_DIM),
            "labor": mlp(LABOR_DIM),
            "day_scalar": _linear(2, d),
        },
        "encoder": {"layers": layers},
        "encoder_norm": {
            "weight": jnp.zeros((d,), dtype=jnp.float32),
            "bias": jnp.zeros((d,), dtype=jnp.float32),
        },
        "heads": {
            "crop": _linear(d, NUM_CROPS * c),
            "animal": _linear(d, NUM_ANIMALS * c),
            "land": _linear(d, NUM_LAND_CLASSES),
            "fertilizer": _linear(d, NUM_CROPS * c),
            "care": _linear(d, NUM_ANIMALS * c),
            "sell_presence": _linear(d, SELL_PRESENCE_CELLS),
            "sell_quantity": _linear(d, SELL_PRESENCE_CELLS),
        },
    }


def init_params(config: ManagerConfig, seed: int) -> dict:
    """Random parameter init (N(0, 0.02^2)); NOT matching torch's exact
    initialization distributions — use checkpoint conversion for parity."""
    spec = empty_params(config)
    flat, treedef = jax.tree_util.tree_flatten(spec)
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, len(flat))
    values = [jax.random.normal(k, x.shape, x.dtype) * 0.02
              for k, x in zip(keys, flat)]
    return jax.tree_util.tree_unflatten(treedef, values)


# ------------------------------------------------------------- primitives


def _sign_log1p(x: jax.Array) -> jax.Array:
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))


def _layer_norm(x: jax.Array, weight: jax.Array, bias: jax.Array) -> jax.Array:
    mu = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x - mu), axis=-1, keepdims=True)
    return (x - mu) / jnp.sqrt(var + _LAYER_NORM_EPS) * weight + bias


def _linear_apply(x: jax.Array, lin: Mapping[str, jax.Array]) -> jax.Array:
    return x @ lin["kernel"] + lin["bias"]


class _Dropout:
    """JIT-safe dropout with fixed site enumeration matching torch placement."""

    def __init__(self, rate: float, rng: jax.Array | None) -> None:
        self.rate = rate
        self.rng = rng
        self._site = 0

    def __call__(self, x: jax.Array) -> jax.Array:
        if self.rate == 0.0 or self.rng is None:
            return x
        site_key = jax.random.fold_in(self.rng, self._site)
        self._site += 1
        mask = jax.random.bernoulli(site_key, 1.0 - self.rate, x.shape)
        return jnp.where(mask, x / (1.0 - self.rate), 0.0)


# ---------------------------------------------------------------- encoder


def _encode_tiles(tile_params: Mapping, dropout: _Dropout,
                  kind: jax.Array, crop: jax.Array, animal: jax.Array,
                  numeric: jax.Array, boolean: jax.Array,
                  mask: jax.Array, role_vector: jax.Array) -> jax.Array:
    positions = jnp.arange(BOARD_SIZE)
    rows = positions // BOARD_SIDE
    cols = positions % BOARD_SIDE
    batch = kind.shape[0]
    row_features = jnp.broadcast_to(
        tile_params["row_embedding"][rows], (batch, BOARD_SIZE,
                                             tile_params["row_embedding"]
                                             .shape[-1]))
    col_features = jnp.broadcast_to(
        tile_params["col_embedding"][cols], (batch, BOARD_SIZE,
                                             tile_params["col_embedding"]
                                             .shape[-1]))

    nullable = numeric[..., list(NULLABLE_TIMING_CHANNELS)]
    nan_indicator = jnp.isnan(nullable).astype(numeric.dtype)
    scales = jnp.asarray(np.asarray(BOARD_NUMERIC_SCALES), numeric.dtype)
    safe_numeric = jnp.nan_to_num(numeric / scales, nan=0.0,
                                  posinf=8.0, neginf=-8.0)
    safe_numeric = jnp.clip(safe_numeric, -8.0, 8.0)

    features = jnp.concatenate([
        tile_params["kind_embedding"][kind],
        tile_params["crop_embedding"][crop],
        tile_params["animal_embedding"][animal],
        row_features,
        col_features,
        safe_numeric,
        nan_indicator,
        boolean.astype(numeric.dtype) * 2.0 - 1.0,
        mask.astype(numeric.dtype),
    ], axis=-1)
    hidden = jax.nn.gelu(_linear_apply(features, tile_params["project"]["0"]),
                         approximate=False)
    hidden = dropout(hidden)
    out = _linear_apply(hidden, tile_params["project"]["3"])
    return out + role_vector


def _global_tokens(global_params: Mapping, dropout: _Dropout,
                   batch: Mapping[str, jax.Array]) -> dict[str, jax.Array]:
    scalars = batch["scalars"].astype(jnp.float32)
    money = jnp.clip(_sign_log1p(scalars[:, 0:1] * 1e-4), -8.0, 8.0)
    hires_today = scalars[:, 1:2] / float(MAX_HANDS)
    shed = _sign_log1p(batch["shed_counts"].astype(jnp.float32))
    seeds = _sign_log1p(batch["seed_counts"].astype(jnp.float32))
    carried = _sign_log1p(batch["carried_counts"].astype(jnp.float32))
    unlocked = batch["unlocked"].astype(jnp.float32)
    self_features = jnp.concatenate(
        [money, shed, seeds, carried, unlocked, hires_today], axis=-1)

    market_inventory = _sign_log1p(
        batch["market_inventory"].astype(jnp.float32))
    market_prices = jnp.clip(
        batch["market_prices"].astype(jnp.float32) * 0.01, -8.0, 8.0)

    town = _sign_log1p(batch["shop_counts"].astype(jnp.float32))

    labor_features = jnp.stack([
        hires_today[:, 0],
        scalars[:, 2] / float(MAX_HANDS),
        jnp.clip(scalars[:, 3] * 0.01, -8.0, 8.0),
    ], axis=-1)

    day = jnp.clip(batch["day"].astype(jnp.int32), 0, TOTAL_DAYS - 1)
    days_remaining = batch["days_remaining"].astype(jnp.float32)
    normalized = jnp.stack([
        day.astype(jnp.float32) / float(TOTAL_DAYS - 1),
        jnp.clip(days_remaining / float(TOTAL_DAYS - 1), 0.0, 1.0),
    ], axis=-1)

    def mlp(name: str, features: jax.Array) -> jax.Array:
        hidden = jax.nn.gelu(
            _linear_apply(features, global_params[name]["0"]),
            approximate=False)
        return _linear_apply(dropout(hidden), global_params[name]["3"])

    return {
        "self_resource": mlp("self_resource", self_features),
        "market": mlp("market",
                      jnp.concatenate([market_inventory, market_prices],
                                      axis=-1)),
        "town": mlp("town", town),
        "labor": mlp("labor", labor_features),
        "day": global_params["day_embedding"][day]
               + _linear_apply(normalized, global_params["day_scalar"]),
    }


def _encoder_layer(layer: Mapping, x: jax.Array, config: ManagerConfig,
                   dropout: _Dropout) -> jax.Array:
    """Norm-first self-attention block, exact torch layout (q|k|v packed)."""
    d = config.d_model
    num_heads = config.num_heads
    head_dim = d // num_heads
    b, t, _ = x.shape

    normed = _layer_norm(x, layer["norm1_weight"], layer["norm1_bias"])
    qkv = normed @ layer["qkv_kernel"] + layer["qkv_bias"]
    q, k, v = jnp.split(qkv, 3, axis=-1)  # packed layout is exactly q|k|v
    # [B, T, H, hd] -> [B, H, T, hd]
    q = q.reshape(b, t, num_heads, head_dim).transpose(0, 2, 1, 3)
    k = k.reshape(b, t, num_heads, head_dim).transpose(0, 2, 1, 3)
    v = v.reshape(b, t, num_heads, head_dim).transpose(0, 2, 1, 3)
    scores = q @ k.transpose(0, 1, 3, 2) * (head_dim ** -0.5)
    weights = jax.nn.softmax(scores, axis=-1)
    weights = dropout(weights)  # torch attention dropout_p placement
    attended = (weights @ v).transpose(0, 2, 1, 3).reshape(b, t, d)
    attn_out = attended @ layer["out_kernel"] + layer["out_bias"]
    x = x + dropout(attn_out)  # torch dropout1 placement

    normed = _layer_norm(x, layer["norm2_weight"], layer["norm2_bias"])
    hidden = jax.nn.gelu(_linear_apply(normed, layer["linear1"]),
                         approximate=False)
    ffn_out = _linear_apply(hidden, layer["linear2"])
    return x + dropout(ffn_out)  # torch dropout2 placement


# ------------------------------------------------------------- validation


def validate_inputs(inputs: Mapping[str, object],
                    config: ManagerConfig) -> None:
    """Reject missing/unexpected adapter-array keys loudly (metadata can
    never leak into features). Non-jitted boundary; call before forward."""
    keys = set(inputs.keys())
    unknown = sorted(keys - OWN_INPUT_KEYS - OPPONENT_PUBLIC_INPUT_KEYS)
    if unknown:
        raise ValueError(
            f"unknown input keys {unknown}; only adapter predictive arrays "
            f"are accepted — result metadata (names/scores/banks/partition/"
            f"source) must never reach the model")
    allowed = set(OWN_INPUT_KEYS)
    if config.include_opponent_board:
        allowed |= OPPONENT_PUBLIC_INPUT_KEYS
    missing = sorted(allowed - keys)
    if missing:
        raise ValueError(
            f"missing required input keys {missing}; expected adapter "
            f"arrays {sorted(allowed)}")


def _as_float(array: object) -> jax.Array:
    return jnp.asarray(array, dtype=jnp.float32)


def _as_int(array: object) -> jax.Array:
    return jnp.asarray(array, dtype=jnp.int32)


def _prepare_inputs(inputs: Mapping[str, object]) -> dict[str, jax.Array]:
    int_keys = {
        "board_kind", "board_crop", "board_animal", "board_mask", "shed_counts",
        "seed_counts", "carried_counts", "unlocked", "market_inventory",
        "shop_counts", "day", "days_remaining",
        "opp_board_kind", "opp_board_crop", "opp_board_animal",
        "opp_board_mask", "opp_unlocked",
    }
    prepared: dict[str, jax.Array] = {}
    for key, value in inputs.items():
        prepared[key] = (_as_int(value) if key in int_keys
                         else _as_float(value))
    return prepared


# ---------------------------------------------------------------- forward


def _forward_core(params: Mapping, inputs: Mapping[str, jax.Array],
                  config: ManagerConfig, dropout: _Dropout) -> dict[str, jax.Array]:
    b = inputs["board_kind"].shape[0]
    role_vectors = params["role_embedding"]

    own_tiles = _encode_tiles(
        params["tile_encoder"], dropout,
        inputs["board_kind"], inputs["board_crop"], inputs["board_animal"],
        inputs["board_numeric"], inputs["board_bool"], inputs["board_mask"],
        role_vectors[0])
    parts = [
        jnp.broadcast_to(params["manager_token"], (b, 1, config.d_model)),
        own_tiles,
    ]
    if config.include_opponent_board:
        opp_tiles = _encode_tiles(
            params["tile_encoder"], dropout,
            inputs["opp_board_kind"], inputs["opp_board_crop"],
            inputs["opp_board_animal"], inputs["opp_board_numeric"],
            inputs["opp_board_bool"], inputs["opp_board_mask"],
            role_vectors[1])
        parts.append(opp_tiles)
    globals_ = _global_tokens(params["global_encoders"], dropout, inputs)
    parts.extend(globals_[name][:, None, :] for name in GLOBAL_TOKEN_NAMES)

    hidden = jnp.concatenate(parts, axis=1)
    for layer in params["encoder"]["layers"]:
        hidden = _encoder_layer(layer, hidden, config, dropout)
    manager = _layer_norm(hidden[:, 0], params["encoder_norm"]["weight"],
                          params["encoder_norm"]["bias"])

    c = config.count_classes
    heads = params["heads"]

    def head(name: str) -> jax.Array:
        return _linear_apply(manager, heads[name])

    return {
        "crop_logits": head("crop").reshape(b, NUM_CROPS, c),
        "animal_logits": head("animal").reshape(b, NUM_ANIMALS, c),
        "land_logits": head("land"),
        "fertilizer_logits": head("fertilizer").reshape(b, NUM_CROPS, c),
        "care_logits": head("care").reshape(b, NUM_ANIMALS, c),
        "sell_presence_logits":
            head("sell_presence").reshape(b, NUM_PRODUCTS, SELL_BIN_COUNT),
        "sell_quantity_log1p":
            head("sell_quantity").reshape(b, NUM_PRODUCTS, SELL_BIN_COUNT),
    }


def _forward_eval(params: Mapping, inputs: Mapping[str, jax.Array],
                  config: ManagerConfig) -> dict[str, jax.Array]:
    return _forward_core(params, inputs, config,
                         _Dropout(config.dropout, None))


# ManagerConfig is a frozen (hashable) dataclass; passing it as a static
# argument retraces only when the architecture actually changes.
_forward_jit = jax.jit(_forward_eval, static_argnames="config")


def forward(params: Mapping, inputs: Mapping[str, object],
            config: ManagerConfig, *, training: bool = False,
            rng: jax.Array | None = None) -> dict[str, jax.Array]:
    """Eval-exact forward pass over adapter predictive arrays.

    `training=True` enables the torch-placed dropout sites and requires
    `rng`. Input-key validation happens eagerly at this non-jitted boundary.
    """
    validate_inputs(inputs, config)
    if training and rng is None:
        raise ValueError("training=True requires an explicit rng key")
    if training and config.dropout == 0.0:
        training = False
    prepared = _prepare_inputs(inputs)
    if training:
        return _forward_core(params, prepared, config, _Dropout(
            config.dropout, rng))
    return _forward_jit(params, prepared, config)


# ------------------------------------------------------- inference helpers


def predict_counts(count_logits: jax.Array) -> jax.Array:
    """Argmax count prediction [B, K] in 0..count_max."""
    return jnp.argmax(count_logits, axis=-1)


def predict_land(land_logits: jax.Array) -> jax.Array:
    """Land class argmax + 1 -> unlocked quadrant count 1..4."""
    return jnp.argmax(land_logits, axis=-1) + 1


def predict_sells(presence_logits: jax.Array,
                  quantity_log1p: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Sigmoid presence and nonnegative expm1 quantity predictions."""
    presence = jax.nn.sigmoid(presence_logits)
    quantity = jnp.clip(
        jnp.expm1(jnp.clip(quantity_log1p, min=0.0)), min=0.0)
    return presence, quantity

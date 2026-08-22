"""Stateless daily-manager tile Transformer over the compact schema-v3 arrays.

One implementation, one forward pass per day: a shared tile encoder embeds
the own 100-tile board (and optionally the opponent PUBLIC board), five
compact global tokens summarize resources/market/town/labor/day, and a
standard `nn.TransformerEncoder` (batch_first, norm_first, GELU) produces
one MANAGER representation consumed by the structured output heads.

The model consumes ONLY the adapter predictive input arrays
(`bc_manager.adapter.table_to_arrays` "inputs"). Result metadata (names,
scores, final banks, partition identity, source paths) is rejected loudly so
it can never leak into features. There is no temporal sequence, RNN, value
head, opponent inference, or custom attention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn

from .constants import (
    ANIMAL_ORDER,
    BOARD_BOOL_FIELDS,
    BOARD_NUMERIC_FIELDS,
    BOARD_SIZE,
    CROP_ORDER,
    MAX_HANDS,
    PRODUCT_ORDER,
    QUADRANT_ORDER,
    RESOURCE_ORDER,
    SELL_BIN_COUNT,
    SHOP_VOCAB,
    TILE_KIND_IDS,
    TOTAL_DAYS,
)

NUM_CROPS = len(CROP_ORDER)                      # 5
NUM_ANIMALS = len(ANIMAL_ORDER)                  # 3
NUM_PRODUCTS = len(PRODUCT_ORDER)                # 9
NUM_LAND_CLASSES = len(QUADRANT_ORDER)           # 4 -> counts 1..4
SELL_PRESENCE_CELLS = NUM_PRODUCTS * SELL_BIN_COUNT  # 54

NUM_TILE_KINDS = max(TILE_KIND_IDS.values()) + 1     # 8
NUM_CROP_IDS = len(CROP_ORDER) + 2                   # absent + vocab + UNKNOWN
NUM_ANIMAL_IDS = len(ANIMAL_ORDER) + 2               # absent + vocab + UNKNOWN

BOARD_SIDE = 10
assert BOARD_SIDE * BOARD_SIDE == BOARD_SIZE

# Fixed stable scaling for board numeric channels (order =
# BOARD_NUMERIC_FIELDS). Days are divided by episode length; small counters
# by small constants. Nullable timing NaNs get explicit indicator channels.
BOARD_NUMERIC_SCALES = (
    float(TOTAL_DAYS),   # planted_day
    float(TOTAL_DAYS),   # placed_day
    10.0,                # yield_units
    200.0,               # max_lifespan_step
    float(TOTAL_DAYS),   # fertilized_until_day
    7.0,                 # consecutive_unwatered
    7.0,                 # consecutive_unfed
    5.0,                 # pending_care_bonus
    float(TOTAL_DAYS),   # age_days
    float(TOTAL_DAYS),   # days_until_next_harvest
    float(TOTAL_DAYS),   # days_until_next_product
)
NULLABLE_TIMING_CHANNELS = (
    BOARD_NUMERIC_FIELDS.index("days_until_next_harvest"),
    BOARD_NUMERIC_FIELDS.index("days_until_next_product"),
)

OWN_INPUT_KEYS = frozenset({
    "board_kind", "board_crop", "board_animal", "board_numeric",
    "board_bool", "board_mask", "scalars", "shed_counts", "seed_counts",
    "carried_counts", "unlocked", "market_inventory", "market_prices",
    "shop_counts", "day", "days_remaining",
})
OPPONENT_PUBLIC_INPUT_KEYS = frozenset({
    "opp_board_kind", "opp_board_crop", "opp_board_animal",
    "opp_board_numeric", "opp_board_bool", "opp_board_mask",
    "opp_scalars", "opp_unlocked",
})

GLOBAL_TOKEN_NAMES = ("self_resource", "market", "town", "labor", "day")


@dataclass
class ManagerConfig:
    """Configuration for `DailyManagerTransformer`.

    The default configuration is the intended training size; the tiny CPU
    configuration is for fast tests only.
    """

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
    def own_tile_tokens(self) -> int:
        return BOARD_SIZE

    @property
    def opponent_tile_tokens(self) -> int:
        return BOARD_SIZE if self.include_opponent_board else 0

    @property
    def global_token_count(self) -> int:
        return len(GLOBAL_TOKEN_NAMES)

    @property
    def token_count(self) -> int:
        """MANAGER + own tiles + global tokens (+ opponent PUBLIC tiles)."""
        return (1 + self.own_tile_tokens + self.global_token_count
                + self.opponent_tile_tokens)


def tiny_manager_config(**overrides) -> ManagerConfig:
    """Fast CPU test configuration."""
    params = dict(d_model=16, num_layers=1, num_heads=1, ffn_dim=32,
                  dropout=0.0, include_opponent_board=False)
    params.update(overrides)
    return ManagerConfig(**params)


def _sign_log1p(x: Tensor) -> Tensor:
    return torch.sign(x) * torch.log1p(x.abs())


class TileEncoder(nn.Module):
    """Shared encoder for one 100-tile board (own or opponent PUBLIC)."""

    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.kind_embedding = nn.Embedding(NUM_TILE_KINDS, d_model)
        self.crop_embedding = nn.Embedding(NUM_CROP_IDS, d_model)
        self.animal_embedding = nn.Embedding(NUM_ANIMAL_IDS, d_model)
        self.row_embedding = nn.Embedding(BOARD_SIDE, d_model)
        self.col_embedding = nn.Embedding(BOARD_SIDE, d_model)
        numeric_dim = len(BOARD_NUMERIC_FIELDS) + len(NULLABLE_TIMING_CHANNELS)
        feature_dim = (
            d_model  # kind
            + d_model  # crop
            + d_model  # animal
            + d_model  # row
            + d_model  # col
            + numeric_dim
            + len(BOARD_BOOL_FIELDS)
            + 4  # presence mask channels
        )
        self.register_buffer(
            "numeric_scales",
            torch.tensor(BOARD_NUMERIC_SCALES, dtype=torch.float32),
            persistent=False)
        self.project = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

    def forward(self, kind: Tensor, crop: Tensor, animal: Tensor,
                numeric: Tensor, boolean: Tensor, mask: Tensor) -> Tensor:
        b = kind.shape[0]
        positions = torch.arange(BOARD_SIZE, device=kind.device)
        rows = positions // BOARD_SIDE
        cols = positions % BOARD_SIDE

        nullable = numeric[..., NULLABLE_TIMING_CHANNELS]
        nan_indicator = torch.isnan(nullable).to(numeric.dtype)
        safe_numeric = torch.nan_to_num(numeric / self.numeric_scales,
                                        nan=0.0, posinf=8.0, neginf=-8.0)
        safe_numeric = torch.clamp(safe_numeric, -8.0, 8.0)

        features = torch.cat([
            self.kind_embedding(kind.long()),
            self.crop_embedding(crop.long()),
            self.animal_embedding(animal.long()),
            self.row_embedding(rows).unsqueeze(0).expand(b, -1, -1),
            self.col_embedding(cols).unsqueeze(0).expand(b, -1, -1),
            safe_numeric,
            nan_indicator,
            boolean.to(torch.float32) * 2.0 - 1.0,
            mask.to(torch.float32),
        ], dim=-1)
        return self.project(features)


def _global_mlp(in_dim: int, d_model: int, dropout: float) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_dim, d_model),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(d_model, d_model),
    )


SELF_RESOURCE_DIM = 1 + len(RESOURCE_ORDER) + len(CROP_ORDER) \
    + len(RESOURCE_ORDER) + len(QUADRANT_ORDER) + 1
MARKET_DIM = 2 * NUM_PRODUCTS
TOWN_DIM = len(SHOP_VOCAB)
LABOR_DIM = 3


class GlobalEncoders(nn.Module):
    """Five compact global tokens; scalars are never tokenized individually."""

    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.day_embedding = nn.Embedding(TOTAL_DAYS, d_model)
        self.self_resource = _global_mlp(SELF_RESOURCE_DIM, d_model, dropout)
        self.market = _global_mlp(MARKET_DIM, d_model, dropout)
        self.town = _global_mlp(TOWN_DIM, d_model, dropout)
        self.labor = _global_mlp(LABOR_DIM, d_model, dropout)
        self.day_scalar = nn.Linear(2, d_model)

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        scalars = batch["scalars"].to(torch.float32)
        money = torch.clamp(_sign_log1p(scalars[:, 0:1] * 1e-4), -8.0, 8.0)
        hires_today = scalars[:, 1:2] / float(MAX_HANDS)
        shed = _sign_log1p(batch["shed_counts"].to(torch.float32))
        seeds = _sign_log1p(batch["seed_counts"].to(torch.float32))
        carried = _sign_log1p(batch["carried_counts"].to(torch.float32))
        unlocked = batch["unlocked"].to(torch.float32)
        self_features = torch.cat(
            [money, shed, seeds, carried, unlocked, hires_today], dim=-1)

        market_inventory = _sign_log1p(
            batch["market_inventory"].to(torch.float32))
        market_prices = torch.clamp(
            batch["market_prices"].to(torch.float32) * 0.01, -8.0, 8.0)

        town = _sign_log1p(batch["shop_counts"].to(torch.float32))

        labor_features = torch.stack([
            hires_today[:, 0],
            scalars[:, 2] / float(MAX_HANDS),
            torch.clamp(scalars[:, 3] * 0.01, -8.0, 8.0),
        ], dim=-1)

        day = batch["day"].long().clamp(0, TOTAL_DAYS - 1)
        days_remaining = batch["days_remaining"].to(torch.float32)
        normalized = torch.stack([
            day.to(torch.float32) / float(TOTAL_DAYS - 1),
            torch.clamp(days_remaining / float(TOTAL_DAYS - 1), 0.0, 1.0),
        ], dim=-1)

        return {
            "self_resource": self.self_resource(self_features),
            "market": self.market(
                torch.cat([market_inventory, market_prices], dim=-1)),
            "town": self.town(town),
            "labor": self.labor(labor_features),
            "day": self.day_embedding(day) + self.day_scalar(normalized),
        }


class DailyManagerTransformer(nn.Module):
    """Stateless daily manager: tile Transformer with structured heads."""

    def __init__(self, config: ManagerConfig | None = None) -> None:
        super().__init__()
        self.config = config if config is not None else ManagerConfig()
        d = self.config.d_model
        dropout = self.config.dropout

        self.manager_token = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        # Role embedding distinguishes own vs opponent PUBLIC tiles; it is
        # only applied when include_opponent_board is enabled.
        self.role_embedding = nn.Embedding(2, d)
        self.tile_encoder = TileEncoder(d, dropout)
        self.global_encoders = GlobalEncoders(d, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=self.config.num_heads,
            dim_feedforward=self.config.ffn_dim, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.config.num_layers,
            enable_nested_tensor=False)
        self.encoder_norm = nn.LayerNorm(d)

        c = self.config.count_classes
        self.crop_head = nn.Linear(d, NUM_CROPS * c)
        self.animal_head = nn.Linear(d, NUM_ANIMALS * c)
        self.land_head = nn.Linear(d, NUM_LAND_CLASSES)
        self.fertilizer_head = nn.Linear(d, NUM_CROPS * c)
        self.care_head = nn.Linear(d, NUM_ANIMALS * c)
        self.sell_presence_head = nn.Linear(d, SELL_PRESENCE_CELLS)
        self.sell_quantity_head = nn.Linear(d, SELL_PRESENCE_CELLS)

    @property
    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ------------------------------------------------------------- inputs

    def _validate_inputs(self, batch: Mapping[str, Tensor]) -> None:
        keys = set(batch.keys())
        allowed = set(OWN_INPUT_KEYS)
        if self.config.include_opponent_board:
            allowed |= OPPONENT_PUBLIC_INPUT_KEYS
        unknown = sorted(keys - OWN_INPUT_KEYS - OPPONENT_PUBLIC_INPUT_KEYS)
        if unknown:
            raise ValueError(
                f"unknown input keys {unknown}; only adapter predictive "
                f"arrays {sorted(allowed)} are accepted — result metadata "
                f"(names/scores/banks/partition/source) must never reach "
                f"the model")
        missing = sorted(allowed - keys)
        if missing:
            raise ValueError(
                f"missing required input keys {missing}; expected adapter "
                f"arrays {sorted(allowed)}")

    def _encode_tiles(self, batch: Mapping[str, Tensor], prefix: str,
                      role: int) -> Tensor:
        tokens = self.tile_encoder(
            batch[prefix + "kind"], batch[prefix + "crop"],
            batch[prefix + "animal"], batch[prefix + "numeric"],
            batch[prefix + "bool"], batch[prefix + "mask"])
        role_id = torch.tensor(role, device=tokens.device)
        return tokens + self.role_embedding(role_id)

    # ------------------------------------------------------------ forward

    def forward(self, batch: Mapping[str, Tensor]) -> dict[str, Tensor]:
        self._validate_inputs(batch)
        b = batch["board_kind"].shape[0]

        own_tiles = self._encode_tiles(batch, "board_", role=0)
        parts = [self.manager_token.expand(b, -1, -1), own_tiles]
        if self.config.include_opponent_board:
            parts.append(self._encode_tiles(batch, "opp_board_", role=1))
        globals_ = self.global_encoders(dict(batch))
        parts.extend(globals_[name].unsqueeze(1) for name in GLOBAL_TOKEN_NAMES)

        hidden = self.encoder(torch.cat(parts, dim=1))
        manager = self.encoder_norm(hidden[:, 0])

        c = self.config.count_classes
        outputs = {
            "crop_logits":
                self.crop_head(manager).view(b, NUM_CROPS, c),
            "animal_logits":
                self.animal_head(manager).view(b, NUM_ANIMALS, c),
            "land_logits": self.land_head(manager),
            "fertilizer_logits":
                self.fertilizer_head(manager).view(b, NUM_CROPS, c),
            "care_logits":
                self.care_head(manager).view(b, NUM_ANIMALS, c),
            "sell_presence_logits":
                self.sell_presence_head(manager).view(b, NUM_PRODUCTS,
                                                      SELL_BIN_COUNT),
            "sell_quantity_log1p":
                self.sell_quantity_head(manager).view(b, NUM_PRODUCTS,
                                                      SELL_BIN_COUNT),
        }
        return outputs


# ------------------------------------------------------- inference helpers


@torch.no_grad()
def predict_counts(count_logits: Tensor) -> Tensor:
    """Argmax count prediction [B, K] in 0..count_max."""
    return count_logits.argmax(dim=-1)


@torch.no_grad()
def predict_land(land_logits: Tensor) -> Tensor:
    """Land class argmax + 1 -> unlocked quadrant count 1..4."""
    return land_logits.argmax(dim=-1) + 1


@torch.no_grad()
def predict_sells(presence_logits: Tensor,
                  quantity_log1p: Tensor) -> tuple[Tensor, Tensor]:
    """Sigmoid presence and nonnegative expm1 quantity predictions."""
    presence = torch.sigmoid(presence_logits)
    quantity = torch.expm1(torch.clamp(quantity_log1p, min=0.0)).clamp_min(0.0)
    return presence, quantity

"""Framework-free model dimensions and input-schema definitions.

This module is shared by the PyTorch reference model and native JAX policies.
Keeping these lightweight definitions here prevents JAX inference imports from
loading Torch while preserving the established V0/E tensor contracts.
"""

from __future__ import annotations

from .constants import (
    ANIMAL_ORDER,
    BOARD_BOOL_FIELDS,
    BOARD_NUMERIC_FIELDS,
    BOARD_SIZE,
    CROP_ORDER,
    PRODUCT_ORDER,
    QUADRANT_ORDER,
    RESOURCE_ORDER,
    SELL_BIN_COUNT,
    SHOP_VOCAB,
    TOTAL_DAYS,
    TILE_KIND_IDS,
)


NUM_CROPS = len(CROP_ORDER)
NUM_ANIMALS = len(ANIMAL_ORDER)
NUM_PRODUCTS = len(PRODUCT_ORDER)
NUM_LAND_CLASSES = len(QUADRANT_ORDER)
SELL_PRESENCE_CELLS = NUM_PRODUCTS * SELL_BIN_COUNT

NUM_TILE_KINDS = max(TILE_KIND_IDS.values()) + 1
NUM_CROP_IDS = len(CROP_ORDER) + 2
NUM_ANIMAL_IDS = len(ANIMAL_ORDER) + 2

BOARD_SIDE = 10
assert BOARD_SIDE * BOARD_SIDE == BOARD_SIZE

BOARD_NUMERIC_SCALES = (
    float(TOTAL_DAYS),
    float(TOTAL_DAYS),
    10.0,
    200.0,
    float(TOTAL_DAYS),
    7.0,
    7.0,
    5.0,
    float(TOTAL_DAYS),
    float(TOTAL_DAYS),
    float(TOTAL_DAYS),
)
NULLABLE_TIMING_CHANNELS = (
    BOARD_NUMERIC_FIELDS.index("days_until_next_harvest"),
    BOARD_NUMERIC_FIELDS.index("days_until_next_product"),
)
TILE_NUMERIC_FEATURE_DIM = len(BOARD_NUMERIC_FIELDS) + len(
    NULLABLE_TIMING_CHANNELS)
TILE_NON_EMBEDDING_FEATURE_DIM = (
    TILE_NUMERIC_FEATURE_DIM + len(BOARD_BOOL_FIELDS) + 4
)


def tile_feature_dim(d_model: int) -> int:
    """Return the compact tile encoder input width for ``d_model``."""
    return 5 * int(d_model) + TILE_NON_EMBEDDING_FEATURE_DIM

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
DECISION_TOKEN_NAMES = ("crop", "animal", "land", "fertilizer", "care",
                        "sell")
DECISION_SELL_INDEX = DECISION_TOKEN_NAMES.index("sell")

# Shared global-token input widths. Keep these here so native JAX imports do
# not need to import the Torch implementation merely to build parameter trees.
SELF_RESOURCE_DIM = (
    1 + len(RESOURCE_ORDER) + len(CROP_ORDER) + len(RESOURCE_ORDER)
    + len(QUADRANT_ORDER) + 1
)
MARKET_DIM = 2 * NUM_PRODUCTS
TOWN_DIM = len(SHOP_VOCAB)
LABOR_DIM = 3

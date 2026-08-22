"""Compact-array BC data layer for the daily farm manager.

`bc_manager.adapter`   schema-v2 Parquet -> compact NumPy arrays, date/score
                       selection (never random splits)
`bc_manager.metrics`   sparse-target NumPy metrics
`bc_manager.baseline`  train-split-only empirical day baseline

The model/loss/training packet consumes these interfaces; nothing here
depends on torch.
"""

from .adapter import (
    SchemaVersionError,
    aggregate_sells,
    build_targets,
    load_dataset,
    load_selected_table,
    load_train_val,
    table_to_arrays,
)
from .baseline import DayBaseline, evaluate_baseline
from .constants import (
    ANIMAL_IDS,
    ANIMAL_ORDER,
    ANIMAL_UNKNOWN_ID,
    BOARD_BOOL_FIELDS,
    BOARD_DERIVED_FIELD_ORDER,
    BOARD_DERIVED_PRESENT_BIT,
    BOARD_MASK_FIELDS,
    BOARD_NUMERIC_FIELDS,
    CROP_IDS,
    CROP_ORDER,
    CROP_UNKNOWN_ID,
    MIN_SCORE_DEFAULT,
    PRODUCT_ORDER,
    QUADRANT_ORDER,
    RESOURCE_ORDER,
    SHOPS,
    SELL_QUANTITY_CAP,
    SHOP_VOCAB,
    TILE_KIND_IDS,
    TRAIN_DATES_DEFAULT,
    VAL_DATES_DEFAULT,
    bound_sell_quantity,
    board_field_present,
    sell_bin_index,
)
from .metrics import (
    exact_accuracy,
    exact_match_rate,
    group_metrics,
    mae,
    nonzero_recall,
    pred_nonzero_rate,
    sell_metrics,
    true_nonzero_rate,
)

__all__ = [
    "SchemaVersionError",
    "aggregate_sells",
    "build_targets",
    "load_dataset",
    "load_selected_table",
    "load_train_val",
    "table_to_arrays",
    "DayBaseline",
    "evaluate_baseline",
    "ANIMAL_ORDER",
    "ANIMAL_IDS",
    "ANIMAL_UNKNOWN_ID",
    "BOARD_BOOL_FIELDS",
    "BOARD_DERIVED_FIELD_ORDER",
    "BOARD_DERIVED_PRESENT_BIT",
    "BOARD_MASK_FIELDS",
    "BOARD_NUMERIC_FIELDS",
    "CROP_ORDER",
    "CROP_IDS",
    "CROP_UNKNOWN_ID",
    "MIN_SCORE_DEFAULT",
    "PRODUCT_ORDER",
    "QUADRANT_ORDER",
    "RESOURCE_ORDER",
    "SHOPS",
    "SELL_QUANTITY_CAP",
    "SHOP_VOCAB",
    "TILE_KIND_IDS",
    "TRAIN_DATES_DEFAULT",
    "VAL_DATES_DEFAULT",
    "bound_sell_quantity",
    "board_field_present",
    "sell_bin_index",
    "exact_accuracy",
    "exact_match_rate",
    "group_metrics",
    "mae",
    "nonzero_recall",
    "pred_nonzero_rate",
    "sell_metrics",
    "true_nonzero_rate",
]

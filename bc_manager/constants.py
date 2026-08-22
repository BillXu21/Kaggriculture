"""BC data constants and compact-array layout for the daily farm manager.

Single source of truth for categorical vocabs, array channel ordering, and
split defaults consumed by `bc_manager.adapter`, `bc_manager.metrics`, and
`bc_manager.baseline`. Engine vocabularies are imported from
`replay_daily.constants` so the canonical schema stays authoritative.
"""

from replay_daily.constants import ANIMALS, CROPS, PRODUCTS

# ---------------------------------------------------------------- vocabs

# Fixed channel order for all count targets and product-keyed features.
CROP_ORDER = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
ANIMAL_ORDER = ("GOOSE", "COW", "SHEEP")
PRODUCT_ORDER = tuple(PRODUCTS)  # nine canonical products
assert CROP_ORDER == tuple(CROPS)

# Shed/carried inventories may hold products and (unplaced) live animals.
RESOURCE_ORDER = PRODUCT_ORDER + ANIMAL_ORDER

# Town shops observed in canonical data plus an explicit UNKNOWN channel for
# future shop names. Unknown shop names map to UNKNOWN instead of failing.
SHOPS = (
    "BAKERY", "BRUNCH_SPOT", "FARMERS_MARKET", "ICE_CREAM_SHOP",
    "PET_CAFE", "PIZZA_SHOP", "SMOOTHIE_SHOP", "YARN_STORE",
)
SHOP_UNKNOWN = "UNKNOWN"
SHOP_VOCAB = SHOPS + (SHOP_UNKNOWN,)

QUADRANT_ORDER = ("NW", "NE", "SW", "SE")

# Board tile-kind vocabulary (physical tagged-struct `tile_kind` values).
TILE_EMPTY = "EMPTY"
TILE_KIND_IDS = {
    "EMPTY": 0,
    "PLANT": 1,
    "COOP": 2,
    "PASTURE": 3,
    "WEED": 4,
    # Bare string sentinels (logical "LOCKED"/"WEED" strings).
    "LOCKED": 5,
    "BARE_OTHER": 6,
    # Any unrecognized kind maps here instead of failing.
    "UNKNOWN": 7,
}
BARE_STRING_KIND_IDS = {"LOCKED": TILE_KIND_IDS["LOCKED"]}

# Crop/animal ids: 0 = absent, 1..k = vocabulary, last = UNKNOWN.
CROP_IDS = {name: i + 1 for i, name in enumerate(CROP_ORDER)}
CROP_UNKNOWN_ID = len(CROP_ORDER) + 1
ANIMAL_IDS = {name: i + 1 for i, name in enumerate(ANIMAL_ORDER)}
ANIMAL_UNKNOWN_ID = len(ANIMAL_ORDER) + 1

# ---------------------------------------------------- board array layout

# Numeric tile channels (raw + derived lifecycle ints), fixed order.
BOARD_NUMERIC_FIELDS = (
    "planted_day", "placed_day", "yield_units", "max_lifespan_step",
    "fertilized_until_day", "consecutive_unwatered", "consecutive_unfed",
    "pending_care_bonus",
    "age_days", "days_until_next_harvest", "days_until_next_product",
)
# Derived timing fields that are nullable *within* a present derived struct;
# null there is encoded as NaN (absent struct stays 0.0 + mask 0).
BOARD_NULLABLE_TIMING = ("days_until_next_harvest", "days_until_next_product")

# Boolean tile channels, fixed order.
BOARD_BOOL_FIELDS = (
    "watered_today", "fed_today", "cared_today", "fertilizer_available",
    "currently_harvestable", "fertilizer_active", "past_lifespan",
    "starving",
)

# Presence-mask channels distinguishing absent vs filled groups.
BOARD_MASK_FIELDS = ("tile_present", "plant_present", "animal_present",
                     "derived_present")

# `replay_daily.storage` preserves logical key presence in this bit layout.
# Keeping the offsets here lets the adapter distinguish absent keys from
# present-but-null lifecycle values without reconstructing logical records.
BOARD_TILE_FIELD_ORDER = (
    "crop", "animal", "planted_day", "placed_day", "yield_units",
    "max_lifespan_step", "fertilized_until_day", "consecutive_unwatered",
    "watered_today", "fed_today", "cared_today", "consecutive_unfed",
    "fertilizer_available", "pending_care_bonus",
)
BOARD_DERIVED_FIELD_ORDER = (
    "age_days", "currently_harvestable", "days_until_next_harvest",
    "days_until_next_product", "fertilizer_active", "past_lifespan",
    "starving",
)
BOARD_DERIVED_PRESENT_BIT = len(BOARD_TILE_FIELD_ORDER)


def board_field_present(present_mask: int, field: str) -> bool:
    """Return whether a canonical tile key was present before Arrow encoding."""
    if field in BOARD_TILE_FIELD_ORDER:
        bit = BOARD_TILE_FIELD_ORDER.index(field)
    elif field in BOARD_DERIVED_FIELD_ORDER:
        bit = BOARD_DERIVED_PRESENT_BIT + 1 + BOARD_DERIVED_FIELD_ORDER.index(field)
    else:
        raise KeyError(field)
    return bool(int(present_mask) & (1 << bit))

BOARD_SIZE = 100  # 10x10, row-major y*10+x
BOARD_NUMERIC_FILL = 0.0

# ------------------------------------------------------------ labor/labor

MAX_HANDS = 8  # hand slots after the farmer; more raises a clear error

# --------------------------------------------------------------- selling

SELL_QUANTITY_CAP = 100
SELL_BIN_COUNT = 6  # floor(hour/4) for hours 0..23


def sell_bin_index(hour: int) -> int:
    """Six intraday bins anchored at 0/4/8/12/16/20."""
    hour = int(hour)
    if not 0 <= hour < 24:
        raise ValueError(f"sell hour must be in [0, 23], got {hour}")
    return hour // 4


def bound_sell_quantity(quantity: int) -> int:
    """min(max(int(quantity), 0), 100); repeated events may still exceed 100."""
    return min(max(int(quantity), 0), SELL_QUANTITY_CAP)


# ---------------------------------------------------------- split policy

TRAIN_DATES_DEFAULT = ("2026-08-17", "2026-08-18", "2026-08-19", "2026-08-20")
VAL_DATES_DEFAULT = ("2026-08-21",)
MIN_SCORE_DEFAULT = 2950.0

TOTAL_DAYS = 30  # confirmed 30-day episodes; days_remaining = 29 - day

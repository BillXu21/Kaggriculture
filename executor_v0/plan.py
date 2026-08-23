"""Typed daily plan for the V0 BC executor (issue #1 section 2).

A `DailyPlan` is the complete once-per-day manager decision, stored in fixed
canonical orders (`bc_manager.constants.CROP_ORDER` / `ANIMAL_ORDER`,
`replay_daily.constants.PRODUCTS`, sell-bin anchors 0/4/8/12/16/20) as plain
Python integer tuples so instances are immutable, copy-safe, deterministically
ordered, and JSON-serializable. No tile-specific applications, routes, value
heads, or strategic extras.
"""

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any

from bc_manager.constants import ANIMAL_ORDER, CROP_ORDER
from replay_daily.constants import PRODUCTS

__all__ = ["DailyPlan", "SELL_BIN_ANCHORS"]

SELL_BIN_ANCHORS = (0, 4, 8, 12, 16, 20)
_LAND_MIN, _LAND_MAX = 1, 4


def _require_count(value: Any, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int,)):
        raise ValueError(f"{what} must be an integer, got {value!r}")
    if not math.isfinite(value):
        raise ValueError(f"{what} must be finite, got {value!r}")
    if value < 0:
        raise ValueError(f"{what} must be nonnegative, got {int(value)}")
    return int(value)


def _validated_vector(mapping: Mapping[str, Any], order: tuple[str, ...],
                      what: str) -> tuple[int, ...]:
    if not isinstance(mapping, Mapping):
        raise ValueError(
            f"{what} must be a mapping keyed by {list(order)}, got "
            f"{type(mapping).__name__}")
    unknown = sorted(set(mapping) - set(order))
    missing = sorted(set(order) - set(mapping))
    if unknown or missing:
        raise ValueError(
            f"{what} key mismatch; unknown={unknown}, missing={missing}; "
            f"expected exactly {list(order)}")
    return tuple(_require_count(mapping[name], f"{what}[{name!r}]")
                 for name in order)


@dataclass(frozen=True)
class DailyPlan:
    """One day's manager decision in fixed canonical orders.

    Fields store positional integer tuples (never dicts) so instances are
    immutable and trivially copy-safe; dict views are available via the
    ``*_dict`` properties and always iterate in canonical vocabulary order.
    """

    crop_targets: tuple[int, ...]     # CROP_ORDER
    animal_targets: tuple[int, ...]   # ANIMAL_ORDER
    land_count: int                   # resulting unlocked quadrant count 1..4
    fertilizer_by_crop: tuple[int, ...]  # by CROP_ORDER
    care_by_animal: tuple[int, ...]   # by ANIMAL_ORDER
    sell_quantities: tuple[tuple[int, ...], ...]  # [PRODUCT_ORDER][bin index]

    @classmethod
    def create(
        cls,
        *,
        crop_targets: Mapping[str, int],
        animal_targets: Mapping[str, int],
        land_count: int,
        fertilizer_by_crop: Mapping[str, int],
        care_by_animal: Mapping[str, int],
        sell_quantities: Mapping[str, Mapping[str, int]],
    ) -> "DailyPlan":
        """Validate mappings and normalize into canonical positional order."""
        crops = _validated_vector(crop_targets, CROP_ORDER, "crop_targets")
        animals = _validated_vector(animal_targets, ANIMAL_ORDER,
                                    "animal_targets")
        land = _require_count(land_count, "land_count")
        if not _LAND_MIN <= land <= _LAND_MAX:
            raise ValueError(
                f"land_count must be in [{_LAND_MIN}, {_LAND_MAX}], got {land}")
        fertilizer = _validated_vector(fertilizer_by_crop, CROP_ORDER,
                                       "fertilizer_by_crop")
        care = _validated_vector(care_by_animal, ANIMAL_ORDER,
                                 "care_by_animal")

        if not isinstance(sell_quantities, Mapping):
            raise ValueError(
                "sell_quantities must map each product to its six bin "
                "quantities")
        unknown_products = sorted(set(sell_quantities) - set(PRODUCTS))
        missing_products = sorted(set(PRODUCTS) - set(sell_quantities))
        if unknown_products or missing_products:
            raise ValueError(
                f"sell_quantities product mismatch; "
                f"unknown={unknown_products}, missing={missing_products}")
        rows: list[tuple[int, ...]] = []
        for product in PRODUCTS:
            bins = sell_quantities[product]
            if not isinstance(bins, Mapping):
                raise ValueError(
                    f"sell_quantities[{product!r}] must map sell-bin anchors "
                    f"{list(SELL_BIN_ANCHORS)} to quantities")
            unknown_bins = sorted(set(bins) - set(SELL_BIN_ANCHORS))
            missing_bins = sorted(set(SELL_BIN_ANCHORS) - set(bins))
            if unknown_bins or missing_bins:
                raise ValueError(
                    f"sell_quantities[{product!r}] anchor mismatch; "
                    f"unknown={unknown_bins}, missing={missing_bins}; expected "
                    f"exactly {list(SELL_BIN_ANCHORS)}")
            rows.append(tuple(
                _require_count(bins[anchor],
                               f"sell_quantities[{product!r}][{anchor}]")
                for anchor in SELL_BIN_ANCHORS))

        return cls(crop_targets=crops, animal_targets=animals,
                   land_count=land, fertilizer_by_crop=fertilizer,
                   care_by_animal=care, sell_quantities=tuple(rows))

    @property
    def crop_targets_dict(self) -> dict[str, int]:
        return dict(zip(CROP_ORDER, self.crop_targets))

    @property
    def animal_targets_dict(self) -> dict[str, int]:
        return dict(zip(ANIMAL_ORDER, self.animal_targets))

    @property
    def fertilizer_by_crop_dict(self) -> dict[str, int]:
        return dict(zip(CROP_ORDER, self.fertilizer_by_crop))

    @property
    def care_by_animal_dict(self) -> dict[str, int]:
        return dict(zip(ANIMAL_ORDER, self.care_by_animal))

    @property
    def sell_quantities_dict(self) -> dict[str, dict[str, int]]:
        """{str(anchor): {product: quantity}} in canonical order."""
        return {
            str(anchor): {product: self.sell_quantities[product_index][bin_index]
                          for product_index, product in enumerate(PRODUCTS)}
            for bin_index, anchor in enumerate(SELL_BIN_ANCHORS)
        }

    def to_json_dict(self) -> dict[str, Any]:
        """Deterministic, JSON-serializable representation (canonical order)."""
        return {
            "crop_targets": self.crop_targets_dict,
            "animal_targets": self.animal_targets_dict,
            "land_count": self.land_count,
            "fertilizer_by_crop": self.fertilizer_by_crop_dict,
            "care_by_animal": self.care_by_animal_dict,
            "sell_quantities": self.sell_quantities_dict,
        }

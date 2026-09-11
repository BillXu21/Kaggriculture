"""Framework-free Stage 2.5 curriculum and crop-shortfall configuration."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from numbers import Real


CURRICULUM_CONFIG_VERSION = "stage25_curriculum_v1"
SHORTFALL_CONFIG_VERSION = "stage25_crop_shortfall_v1"

__all__ = [
    "CURRICULUM_CONFIG_VERSION",
    "SHORTFALL_CONFIG_VERSION",
    "Stage25CurriculumConfig",
    "CropShortfallConfig",
    "apply_land_curriculum",
    "apply_animal_curriculum",
    "apply_crop_curriculum",
    "crop_shortfall_penalty",
]


def _nonnegative_int(value: object, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{what} must be a nonnegative integer, got {value!r}")
    return value


def _optional_nonnegative_int(value: object, what: str) -> int | None:
    if value is None:
        return None
    return _nonnegative_int(value, what)


@dataclass(frozen=True)
class Stage25CurriculumConfig:
    """Optional caps intersected with permanent Packet 1A physical support."""

    version: str = CURRICULUM_CONFIG_VERSION
    enabled: bool = False
    max_positive_crop_delta: int | None = None
    max_land_expansion_per_decision: int | None = None
    max_animal_additions_per_species_per_decision: int | None = None

    def __post_init__(self) -> None:
        if self.version != CURRICULUM_CONFIG_VERSION:
            raise ValueError(
                f"unsupported curriculum version {self.version!r}; expected "
                f"{CURRICULUM_CONFIG_VERSION!r}")
        if not isinstance(self.enabled, bool):
            raise ValueError(f"enabled must be bool, got {self.enabled!r}")
        for name in (
            "max_positive_crop_delta",
            "max_land_expansion_per_decision",
            "max_animal_additions_per_species_per_decision",
        ):
            object.__setattr__(
                self, name, _optional_nonnegative_int(getattr(self, name), name))


@dataclass(frozen=True)
class CropShortfallConfig:
    """One shared crop-capacity shortfall allowance and penalty scale."""

    version: str = SHORTFALL_CONFIG_VERSION
    crop_shortfall_tolerance: int = 5
    crop_shortfall_coef: float = 0.0

    def __post_init__(self) -> None:
        if self.version != SHORTFALL_CONFIG_VERSION:
            raise ValueError(
                f"unsupported shortfall version {self.version!r}; expected "
                f"{SHORTFALL_CONFIG_VERSION!r}")
        object.__setattr__(self, "crop_shortfall_tolerance", _nonnegative_int(
            self.crop_shortfall_tolerance, "crop_shortfall_tolerance"))
        coef = self.crop_shortfall_coef
        if isinstance(coef, bool) or not isinstance(coef, Real):
            raise ValueError(
                f"crop_shortfall_coef must be a finite nonnegative number, "
                f"got {coef!r}")
        parsed = float(coef)
        if not math.isfinite(parsed) or parsed < 0.0:
            raise ValueError(
                f"crop_shortfall_coef must be a finite nonnegative number, "
                f"got {coef!r}")
        object.__setattr__(self, "crop_shortfall_coef", parsed)


def _physical_mask(mask: Sequence[bool], size: int, what: str) -> tuple[bool, ...]:
    result = tuple(mask)
    if len(result) != size or any(not isinstance(value, bool) for value in result):
        raise ValueError(f"{what} must contain exactly {size} bool values")
    if not any(result):
        raise ValueError(f"{what} must retain at least one physical action")
    return result


def _nonempty_intersection(
    physical: tuple[bool, ...], allowed: Sequence[bool], what: str,
) -> tuple[bool, ...]:
    result = tuple(p and a for p, a in zip(physical, allowed))
    if not any(result):
        raise ValueError(f"{what} curriculum intersection removed all actions")
    return result


def apply_land_curriculum(
    physical_support: Sequence[bool],
    observed_land: int,
    config: Stage25CurriculumConfig,
) -> tuple[bool, ...]:
    """Cap absolute land targets relative to current observed land."""
    physical = _physical_mask(physical_support, 4, "land physical support")
    observed = _nonnegative_int(observed_land, "observed_land")
    if not 1 <= observed <= 4:
        raise ValueError(f"observed_land must be in [1, 4], got {observed}")
    cap = config.max_land_expansion_per_decision
    if not config.enabled or cap is None:
        return physical
    return _nonempty_intersection(
        physical, (target <= observed + cap for target in range(1, 5)), "land")


def apply_animal_curriculum(
    physical_support: Sequence[bool],
    observed_placed: int,
    config: Stage25CurriculumConfig,
) -> tuple[bool, ...]:
    """Cap an absolute species target relative to its observed placed count."""
    physical = _physical_mask(physical_support, 101, "animal physical support")
    observed = _nonnegative_int(observed_placed, "observed_placed")
    if observed > 100:
        raise ValueError(f"observed_placed must be in [0, 100], got {observed}")
    cap = config.max_animal_additions_per_species_per_decision
    if not config.enabled or cap is None:
        return physical
    return _nonempty_intersection(
        physical, (target <= observed + cap for target in range(101)), "animal")


def apply_crop_curriculum(
    physical_support: Sequence[bool],
    config: Stage25CurriculumConfig,
) -> tuple[bool, ...]:
    """Cap positive deltas while leaving every physical contraction available."""
    physical = _physical_mask(physical_support, 201, "crop physical support")
    cap = config.max_positive_crop_delta
    if not config.enabled or cap is None:
        return physical
    return _nonempty_intersection(
        physical,
        (delta <= 0 or delta <= cap for delta in range(-100, 101)),
        "crop",
    )


def crop_shortfall_penalty(
    U_crop: int,
    config: CropShortfallConfig = CropShortfallConfig(),
) -> float:
    """Return ``-coef * max(0, U_crop - tolerance)`` for measured shortfall."""
    shortfall = _nonnegative_int(U_crop, "U_crop")
    if config.crop_shortfall_coef == 0.0:
        return 0.0
    excess = max(0, shortfall - config.crop_shortfall_tolerance)
    return -config.crop_shortfall_coef * excess

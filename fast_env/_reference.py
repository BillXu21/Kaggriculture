"""Independent Python observation decoder used by parity tests and benchmarks.

This module intentionally preserves the pre-optimization decode loops.  It is
not imported by the production facades; keeping it here gives tests and
benchmarks a trustworthy semantic oracle without putting a second decode on
the rollout path.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .api import (
    ANIMALS,
    CROPS,
    MAX_HANDS,
    OBS_FARM_BASE,
    OBS_HAND_INVENTORY,
    OBS_HAND_POSITIONS,
    OBS_INVENTORY,
    OBS_MARKET_INVENTORY,
    OBS_MARKET_PRICES,
    OBS_SEEDS,
    OBS_SHED,
    OBS_SHOPS,
    PRODUCTS,
    SEASON_STEPS,
    SHOPS,
    _round,
)


def _reference_tile(
    raw: np.ndarray,
    day: int,
    *,
    canonical: bool = False,
) -> Any:
    if raw[1] > 0.5:
        return "LOCKED"
    if raw[2] > 0.5:
        crop = CROPS[next(index for index in range(5) if raw[7 + index] > 0.5)]
        age = _round(raw[14] * 30.0)
        result = {
            "kind": "PLANT", "crop": crop, "age": age, "planted_day": day - age,
            "max_lifespan_step": _round(raw[16] * SEASON_STEPS),
            "yield_units": _round(raw[15] * 100.0),
            "watered_today": bool(raw[17] > 0.5),
            "consecutive_unwatered": _round(raw[18] * 2.0),
            "fertilized_until_day": _round(raw[19] * 30.0),
        }
        if canonical:
            del result["age"]
        return result
    if raw[3] > 0.5:
        return {"kind": "WEED"}
    if raw[4] > 0.5 or raw[5] > 0.5:
        result: dict[str, Any] = {
            "kind": "COOP" if raw[4] > 0.5 else "PASTURE"
        }
        if raw[11] > 0.5:
            animal = ANIMALS[
                next(index for index in range(3) if raw[12 + index] > 0.5)
            ]
            age = _round(raw[20] * 30.0)
            result.update({
                "animal": animal, "yield_units": _round(raw[15] * 100.0),
                "age": age, "fed_today": bool(raw[21] > 0.5),
                "consecutive_unfed": _round(raw[22] * 2.0),
                "cared_today": bool(raw[23] > 0.5),
                "fertilizer_available": bool(raw[24] * 100.0 > 0.5),
                "pending_care_bonus": _round(raw[25] * 100.0),
            })
            if canonical:
                result["placed_day"] = day - age
                del result["age"]
        return result
    return None


def _reference_inventory(raw: np.ndarray, start: int) -> dict[str, int]:
    return {
        name: _round(raw[start + index] * 100.0)
        for index, name in enumerate(PRODUCTS + ANIMALS)
    }


def decode_public(
    raw: np.ndarray,
    configuration: Mapping[str, Any],
    *,
    canonical_farms: bool = False,
) -> dict[str, Any]:
    turns_per_day = int(configuration["turnsPerDay"])
    step = _round(raw[0] * SEASON_STEPS)
    day = step // turns_per_day
    hour = step % turns_per_day
    farms: list[dict[str, Any]] = []
    for farm_index in range(2):
        position = 7 + farm_index * 6
        hands_position = OBS_HAND_POSITIONS + farm_index * (MAX_HANDS + 1)
        hand_count = max(
            0,
            min(MAX_HANDS, _round(raw[hands_position] * float(MAX_HANDS))),
        )
        tiles = [
            _reference_tile(
                raw[
                    OBS_FARM_BASE + farm_index * 2600 + index * 26:
                    OBS_FARM_BASE + farm_index * 2600 + index * 26 + 26
                ],
                day,
                canonical=canonical_farms,
            )
            for index in range(100)
        ]
        farms.append({
            "money": float(_round(raw[5 + farm_index] * 10000.0)),
            "tiles": [
                tiles[row * 10:(row + 1) * 10]
                for row in range(10)
            ],
            "farmer": [
                _round(raw[position + 1] * 9.0),
                _round(raw[position + 2] * 9.0),
            ],
            "hands": [[
                (_round(raw[hands_position + 1 + hand] * 100.0 - 1.0) % 10),
                (_round(raw[hands_position + 1 + hand] * 100.0 - 1.0) // 10),
            ] for hand in range(hand_count)],
            "unlocked_quadrants": [
                name for index, name in enumerate(("NW", "NE", "SW", "SE"))
                if raw[19 + farm_index * 4 + index] > 0.5
            ],
            "hires_today": _round(
                raw[position + 3] * float(MAX_HANDS)
            ),
        })
    shops = [
        SHOPS[_round(raw[OBS_SHOPS + slot] * 8.0) - 1]
        for slot in range(8)
        if raw[OBS_SHOPS + slot] > 0.0
    ]
    return {
        "farms": farms,
        "market": {
            "inventory": {
                name: _round(raw[OBS_MARKET_INVENTORY + index] * 10000.0)
                for index, name in enumerate(PRODUCTS)
            },
            "prices": {
                name: _round(raw[OBS_MARKET_PRICES + index] * 1000.0)
                for index, name in enumerate(PRODUCTS)
            },
        },
        "town": {"unlocked_shops": shops},
        "day": day,
        "hour": hour,
        "step": step,
        "remainingOverageTime": 60,
    }


def decode_private(raw: np.ndarray, hand_count: int) -> dict[str, Any]:
    inventories = [_reference_inventory(raw, OBS_INVENTORY)]
    inventories.extend(
        _reference_inventory(raw, OBS_HAND_INVENTORY + hand * 12)
        for hand in range(hand_count)
    )
    return {
        "shed": _reference_inventory(raw, OBS_SHED),
        "seeds": {
            name: _round(raw[OBS_SEEDS + index] * 100.0)
            for index, name in enumerate(CROPS)
        },
        "inventories": inventories,
    }


def decode_observation_pair(
    raw: np.ndarray,
    configuration: Mapping[str, Any],
    *,
    canonical_farms: bool = False,
    prepared_kinds: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    del prepared_kinds
    public = decode_public(
        raw[0], configuration, canonical_farms=canonical_farms
    )
    observations = []
    for player in range(2):
        observation = dict(public)
        observation["player"] = player
        observation["private"] = decode_private(
            raw[player], len(public["farms"][player]["hands"])
        )
        observations.append(observation)
    return observations

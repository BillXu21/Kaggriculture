"""Scalar Python facade over the adapted Rust game core."""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

import numpy as np

from ._kaggriculture_env import RustBatchEnv

PRODUCTS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER")
CROPS = PRODUCTS[:5]
ANIMALS = ("GOOSE", "COW", "SHEEP")
SHOPS = ("BAKERY", "BRUNCH_SPOT", "FARMERS_MARKET", "ICE_CREAM_SHOP", "PET_CAFE", "PIZZA_SHOP", "SMOOTHIE_SHOP", "YARN_STORE")
UNIT_IDS = {name: index for index, name in enumerate(("PASS", "NORTH", "SOUTH", "EAST", "WEST", "PICKUP", "PLACE", "DROP", "PLANT", "WATER", "HARVEST", "FERTILIZE", "BUILD_COOP", "BUILD_PASTURE", "FEED", "COLLECT_FERTILIZER", "CARE", "DIG"))}
MARKET_IDS = {name: index for index, name in enumerate(("PASS", "BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL", "HIRE", "BUY_LAND"))}
DEFAULT_CONFIGURATION: dict[str, Any] = {
    "episodeSteps": 720,
    "boardSize": 10,
    "startingMoney": 3000,
    "maxMarketOrdersPerTurn": 10,
    "turnsPerDay": 24,
    "shedCapacity": 100,
    "weedSpawnChance": 0.005,
    "townShopUnlockInterval": 3,
    "townShopSellInterval": 4,
    "townCenterSellInterval": 24,
    "farmHandCostMult": 1,
    "marketParams": {},
    "seed": 0,
}


def _as_int(value: Any, label: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be an integer") from error


def _unit_row(entry: Sequence[Any]) -> tuple[int, int, int]:
    if not entry:
        raise ValueError("unit action must not be empty")
    operation = str(entry[0])
    if operation not in UNIT_IDS:
        raise ValueError(f"unknown unit operation: {operation}")
    target = 0
    quantity = 0
    if operation == "PLANT":
        if len(entry) < 2 or entry[1] not in CROPS:
            raise ValueError(f"invalid PLANT action: {entry!r}")
        target = CROPS.index(entry[1])
    elif operation == "PICKUP":
        if len(entry) < 2:
            raise ValueError(f"invalid PICKUP action: {entry!r}")
        if entry[1] in ANIMALS:
            target = ANIMALS.index(entry[1]) + 1
        elif entry[1] in PRODUCTS:
            target = PRODUCTS.index(entry[1]) + 4
        else:
            raise ValueError(f"unknown PICKUP item: {entry[1]}")
        quantity = _as_int(entry[2], "PICKUP quantity") if len(entry) > 2 else 1
    elif operation == "PLACE":
        if len(entry) < 2:
            raise ValueError(f"invalid PLACE action: {entry!r}")
        if entry[1] in ANIMALS:
            target = ANIMALS.index(entry[1])
        elif entry[1] in PRODUCTS:
            target = PRODUCTS.index(entry[1]) + 4
        else:
            raise ValueError(f"unknown PLACE item: {entry[1]}")
        quantity = _as_int(entry[2], "PLACE quantity") if len(entry) > 2 else 1
    return UNIT_IDS[operation], target, quantity


def _market_row(entry: Sequence[Any]) -> tuple[int, int, int]:
    if not entry or entry[0] == "PASS":
        return 0, 0, 0
    operation = str(entry[0])
    if operation not in MARKET_IDS:
        return 0, 0, 0
    if operation in {"HIRE", "BUY_LAND"}:
        return MARKET_IDS[operation], 0, 1
    if len(entry) < 3:
        return 0, 0, 0
    target_name = entry[1]
    if operation == "BUY_ANIMAL":
        if target_name not in ANIMALS:
            return 0, 0, 0
        target = ANIMALS.index(target_name)
    elif operation in {"BUY_SEED"}:
        if target_name not in CROPS:
            return 0, 0, 0
        target = CROPS.index(target_name)
    elif target_name in PRODUCTS:
        target = PRODUCTS.index(target_name)
    else:
        return 0, 0, 0
    return MARKET_IDS[operation], target, _as_int(entry[2], f"{operation} quantity")


def _encode_actions(actions: Sequence[Mapping[str, Any]]) -> np.ndarray:
    if len(actions) != 2:
        raise ValueError("actions must contain exactly two player action dictionaries")
    encoded = np.zeros((1, 2, 27, 3), dtype=np.int64)
    for player, action in enumerate(actions):
        if not isinstance(action, Mapping) or "farmer" not in action:
            raise ValueError(f"player {player} action must contain farmer")
        encoded[0, player, 0] = _unit_row(action["farmer"])
        hands = action.get("hands", [])
        if len(hands) > 16:
            raise ValueError("at most 16 hand actions are supported")
        for index, hand in enumerate(hands, start=1):
            encoded[0, player, index] = _unit_row(hand)
        market = action.get("market", [])
        if len(market) > 10:
            raise ValueError("at most 10 market orders are supported")
        for index, order in enumerate(market):
            encoded[0, player, 17 + index] = _market_row(order)
    return encoded


def _round(value: float) -> int:
    return int(round(float(value)))


def _tile(raw: np.ndarray, day: int, episode_steps: int) -> Any:
    if raw[1] > 0.5:
        return "LOCKED"
    if raw[2] > 0.5:
        crop = CROPS[next(index for index in range(5) if raw[7 + index] > 0.5)]
        age = _round(raw[14] * 30.0)
        return {
            "kind": "PLANT", "crop": crop, "age": age, "planted_day": day - age,
            "max_lifespan_step": _round(raw[16] * episode_steps),
            "yield_units": _round(raw[15] * 100.0),
            "watered_today": bool(raw[17] > 0.5),
            "consecutive_unwatered": _round(raw[18] * 2.0),
            "fertilized_until_day": _round(raw[19] * 30.0),
        }
    if raw[3] > 0.5:
        return {"kind": "WEED"}
    if raw[4] > 0.5 or raw[5] > 0.5:
        result: dict[str, Any] = {"kind": "COOP" if raw[4] > 0.5 else "PASTURE"}
        if raw[11] > 0.5:
            animal = ANIMALS[next(index for index in range(3) if raw[12 + index] > 0.5)]
            result.update({
                "animal": animal, "yield_units": _round(raw[15] * 100.0),
                "age": _round(raw[20] * 30.0), "fed_today": bool(raw[21] > 0.5),
                "consecutive_unfed": _round(raw[22] * 2.0),
                "cared_today": bool(raw[23] > 0.5),
                "fertilizer_available": bool(raw[24] * 100.0 > 0.5),
                "pending_care_bonus": _round(raw[25] * 100.0),
            })
        return result
    return None


def _inventory(raw: np.ndarray, start: int) -> dict[str, int]:
    return {name: _round(raw[start + index] * 100.0) for index, name in enumerate(PRODUCTS + ANIMALS)}


def _decode_observation(raw: np.ndarray, player: int, configuration: Mapping[str, Any]) -> dict[str, Any]:
    episode_steps = int(configuration["episodeSteps"])
    turns_per_day = int(configuration["turnsPerDay"])
    step = _round(raw[0] * episode_steps)
    day = step // turns_per_day
    hour = step % turns_per_day
    farms: list[dict[str, Any]] = []
    for farm_index in range(2):
        position = 7 + farm_index * 6
        hands_position = 5280 + farm_index * 17
        hand_count = max(0, min(16, _round(raw[hands_position] * 16.0)))
        tiles = [
            _tile(raw[62 + farm_index * 2600 + index * 26:62 + farm_index * 2600 + index * 26 + 26], day, episode_steps)
            for index in range(100)
        ]
        farms.append({
            "money": float(raw[5 + farm_index] * 10000.0),
            "tiles": [tiles[row * 10:(row + 1) * 10] for row in range(10)],
            "farmer": [_round(raw[position + 1] * 9.0), _round(raw[position + 2] * 9.0)],
            "hands": [[
                (_round(raw[hands_position + 1 + hand] * 100.0 - 1.0) % 10),
                (_round(raw[hands_position + 1 + hand] * 100.0 - 1.0) // 10),
            ] for hand in range(hand_count)],
            "unlocked_quadrants": [
                name for index, name in enumerate(("NW", "NE", "SW", "SE"))
                if raw[19 + farm_index * 4 + index] > 0.5
            ],
            "hires_today": _round(raw[position + 3] * 16.0),
        })
    private_base = 5319
    shed = _inventory(raw, private_base)
    seeds = {name: _round(raw[5331 + index] * 100.0) for index, name in enumerate(CROPS)}
    inventories = [_inventory(raw, 5336)]
    hand_count = len(farms[player]["hands"])
    inventories.extend(_inventory(raw, 5348 + hand * 12) for hand in range(hand_count))
    shops = [SHOPS[_round(raw[5560 + slot] * 8.0) - 1] for slot in range(8) if raw[5560 + slot] > 0.0]
    return {
        "player": player,
        "farms": farms,
        "private": {"shed": shed, "seeds": seeds, "inventories": inventories},
        "market": {
            "inventory": {name: _round(raw[5540 + index] * 10000.0) for index, name in enumerate(PRODUCTS)},
            "prices": {name: _round(raw[5549 + index] * 1000.0) for index, name in enumerate(PRODUCTS)},
        },
        "town": {"unlocked_shops": shops},
        "day": day, "hour": hour, "step": step,
        "remainingOverageTime": 60,
    }


class FastKaggricultureEnv:
    """One exact two-seat scalar episode without Kaggle registry imports."""

    def __init__(self, configuration: Mapping[str, Any] | None = None) -> None:
        self.configuration = dict(DEFAULT_CONFIGURATION)
        if configuration:
            self.configuration.update(configuration)
        if int(self.configuration["boardSize"]) != 10:
            raise ValueError("fast engine supports boardSize=10 only")
        if int(self.configuration["maxMarketOrdersPerTurn"]) != 10:
            raise ValueError("fast engine supports maxMarketOrdersPerTurn=10 only")
        self._seed = _as_int(self.configuration.get("seed", 0), "seed")
        self._backend = RustBatchEnv(
            1,
            int(self.configuration["episodeSteps"]),
            int(self.configuration["turnsPerDay"]),
            float(self.configuration["weedSpawnChance"]),
            int(self.configuration["townCenterSellInterval"]),
            int(self.configuration["townShopSellInterval"]),
            int(self.configuration["townShopUnlockInterval"]),
            float(self.configuration["startingMoney"]),
            10,
            int(self.configuration["shedCapacity"]),
            json.dumps(self.configuration.get("marketParams", {}), sort_keys=True),
            int(self.configuration["farmHandCostMult"]),
            "",
        )
        self._observations: list[dict[str, Any]] = []
        self._statuses = ["ACTIVE", "ACTIVE"]

    def _decode(self, observations: np.ndarray) -> list[dict[str, Any]]:
        self._observations = [
            _decode_observation(observations[0, player], player, self.configuration)
            for player in range(2)
        ]
        return self._observations

    def reset(self) -> list[dict[str, Any]]:
        observations, statuses = self._backend.reset(np.asarray([self._seed], dtype=np.uint64))
        self._statuses = ["DONE" if bool(value) else "ACTIVE" for value in statuses[0]]
        return self._decode(observations)

    def step(self, actions: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[float], list[str]]:
        observations, rewards, statuses = self._backend.step(_encode_actions(actions))
        self._statuses = ["DONE" if bool(value) else "ACTIVE" for value in statuses[0]]
        return self._decode(observations), [float(value) for value in rewards[0]], list(self._statuses)

    def state_snapshot(self) -> list[dict[str, Any]]:
        """Return the latest fully decoded public/private state seam."""
        if not self._observations:
            return self.reset()
        return self._observations

    @property
    def statuses(self) -> list[str]:
        return list(self._statuses)

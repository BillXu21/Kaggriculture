"""Scalar Python facade over the adapted Rust game core."""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

import numpy as np

from ._kaggriculture_env import ACTION_SLOTS, MAX_HANDS, RustBatchEnv

PRODUCTS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER")
CROPS = PRODUCTS[:5]
ANIMALS = ("GOOSE", "COW", "SHEEP")
SHOPS = ("BAKERY", "BRUNCH_SPOT", "FARMERS_MARKET", "ICE_CREAM_SHOP", "PET_CAFE", "PIZZA_SHOP", "SMOOTHIE_SHOP", "YARN_STORE")
UNIT_IDS = {name: index for index, name in enumerate(("PASS", "NORTH", "SOUTH", "EAST", "WEST", "PICKUP", "PLACE", "DROP", "PLANT", "WATER", "HARVEST", "FERTILIZE", "BUILD_COOP", "BUILD_PASTURE", "FEED", "COLLECT_FERTILIZER", "CARE", "DIG"))}
# Operation codes the Rust core's `apply_unit_action` actually dispatches on
# (rust/kaggriculture_env/src/lib.rs). These intentionally differ from the
# wire vocabulary order in `UNIT_IDS`; sending wire ids untranslated made
# e.g. "PLANT" (wire 8) land in the core's BUILD-structure arm (internal 8).
UNIT_OP_CODES = {
    "PASS": 0, "NORTH": 1, "SOUTH": 2, "EAST": 3, "WEST": 4,
    "PICKUP": 11, "PLACE": 9, "DROP": 15, "PLANT": 5, "WATER": 10,
    "HARVEST": 6, "FERTILIZE": 12, "BUILD_COOP": 8, "BUILD_PASTURE": 8,
    "FEED": 7, "COLLECT_FERTILIZER": 14, "CARE": 13, "DIG": 17,
}
MARKET_IDS = {name: index for index, name in enumerate(("PASS", "BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL", "HIRE", "BUY_LAND"))}
# The Rust observation writer normalizes the primitive step and plant
# lifespan by this FIXED season length (generated_protocol::SEASON_STEPS),
# not by the configured episodeSteps; decoding must invert exactly that
# constant or step/day/hour are wrong whenever episodeSteps != 720.
SEASON_STEPS = 720
# Exact default-contract hand capacity (generated_protocol::MAX_HANDS):
# official HIRE is one hand per atomic market order, the market queue is
# truncated to maxMarketOrdersPerTurn=10 per turn, a day lasts turnsPerDay=24
# turns, and hands clear at every day reset -- so at most 10*24=240 hands can
# ever exist simultaneously under the pinned default configuration.
MARKET_ACTION_START = MAX_HANDS + 1
# Fixed observation layout offsets (must mirror the Rust writer exactly; the
# authoritative source is scripts/generate_fast_protocol.py). Only the two
# MAX_HANDS-scaled blocks move; the reserved gaps are fixed paddings.
OBS_FARM_BASE = 62
OBS_HAND_POSITIONS = 5280
OBS_SHED = OBS_HAND_POSITIONS + 2 * (MAX_HANDS + 1) + 5
OBS_SEEDS = OBS_SHED + 12
OBS_INVENTORY = OBS_SEEDS + 5
OBS_ANIMAL_INVENTORY = OBS_INVENTORY + 9
OBS_HAND_INVENTORY = OBS_ANIMAL_INVENTORY + 3
OBS_MARKET_INVENTORY = OBS_HAND_INVENTORY + MAX_HANDS * 12
OBS_MARKET_PRICES = OBS_MARKET_INVENTORY + 9
OBS_SHOPS = OBS_MARKET_PRICES + 9 + 2
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


NOOP_ROW = (0, 0, 0)


def _unit_row(entry: Sequence[Any]) -> tuple[int, int, int]:
    # Official interpreter contract: any malformed / unknown unit action is a
    # silent no-op (`_apply_unit_action` returns without mutating state), so
    # the wire translation must never reject an action the official engine
    # would quietly ignore -- it translates it to the no-op row instead.
    if not isinstance(entry, (list, tuple)) or not entry:
        return NOOP_ROW
    operation = str(entry[0])
    if operation not in UNIT_OP_CODES:
        return NOOP_ROW
    target = 0
    quantity = 0
    if operation == "PLANT":
        if len(entry) < 2 or entry[1] not in CROPS:
            return NOOP_ROW
        target = CROPS.index(entry[1])
    elif operation == "BUILD_COOP":
        target = 0
    elif operation == "BUILD_PASTURE":
        target = 1
    elif operation == "PICKUP":
        if len(entry) < 2:
            return NOOP_ROW
        if entry[1] in ANIMALS:
            target = ANIMALS.index(entry[1]) + 1
        elif entry[1] in PRODUCTS:
            target = PRODUCTS.index(entry[1]) + 4
        else:
            # Seed names and other non-shed items can never be carried
            # (official PICKUP looks them up in the shed and finds nothing).
            return NOOP_ROW
        try:
            quantity = _as_int(entry[2], "PICKUP quantity") if len(entry) > 2 else 1
        except ValueError:
            return NOOP_ROW
    elif operation == "PLACE":
        if len(entry) < 2:
            return NOOP_ROW
        if entry[1] in ANIMALS:
            target = ANIMALS.index(entry[1])
        elif entry[1] in PRODUCTS:
            target = PRODUCTS.index(entry[1]) + 4
        else:
            return NOOP_ROW
        try:
            quantity = _as_int(entry[2], "PLACE quantity") if len(entry) > 2 else 1
        except ValueError:
            return NOOP_ROW
    return UNIT_OP_CODES[operation], target, quantity


def _market_row(entry: Sequence[Any]) -> tuple[int, int, int]:
    # Official `_parse_order` returns None (order skipped) for anything
    # malformed, including non-integer or non-positive quantities.
    if not isinstance(entry, (list, tuple)) or not entry or entry[0] == "PASS":
        return NOOP_ROW
    operation = str(entry[0])
    if operation not in MARKET_IDS:
        return NOOP_ROW
    if operation in {"HIRE", "BUY_LAND"}:
        return MARKET_IDS[operation], 0, 1
    if len(entry) < 3:
        return NOOP_ROW
    target_name = entry[1]
    if operation == "BUY_ANIMAL":
        if target_name not in ANIMALS:
            return NOOP_ROW
        target = ANIMALS.index(target_name)
    elif operation in {"BUY_SEED"}:
        if target_name not in CROPS:
            return NOOP_ROW
        target = CROPS.index(target_name)
    elif target_name in PRODUCTS:
        target = PRODUCTS.index(target_name)
    else:
        return NOOP_ROW
    try:
        quantity = _as_int(entry[2], f"{operation} quantity")
    except ValueError:
        return NOOP_ROW
    return MARKET_IDS[operation], target, quantity


def _encode_actions(actions: Sequence[Mapping[str, Any]]) -> np.ndarray:
    if len(actions) != 2:
        raise ValueError("actions must contain exactly two player action dictionaries")
    encoded = np.zeros((1, 2, ACTION_SLOTS, 3), dtype=np.int64)
    for player, action in enumerate(actions):
        # Official interpreter: a non-dict action is treated as {} and every
        # missing component defaults to PASS / [] / [].
        if not isinstance(action, Mapping):
            action = {}
        # Official defaults; a missing "farmer" acts as ["PASS"].
        farmer = action.get("farmer", ["PASS"])
        # Extra submitted hand actions beyond the representable core slots are
        # silent no-ops officially; MAX_HANDS=240 covers every hand count the
        # default contract can reach (10 orders/turn * 24 turns/day).
        hands = action.get("hands", [])[:MAX_HANDS]
        # Official truncates the market queue to maxMarketOrdersPerTurn (10).
        market = action.get("market", [])[:10]
        encoded[0, player, 0] = _unit_row(farmer)
        for index, hand in enumerate(hands, start=1):
            encoded[0, player, index] = _unit_row(hand)
        for index, order in enumerate(market):
            encoded[0, player, MARKET_ACTION_START + index] = _market_row(order)
    return encoded


def _round(value: float) -> int:
    return int(round(float(value)))


def _tile(raw: np.ndarray, day: int) -> Any:
    if raw[1] > 0.5:
        return "LOCKED"
    if raw[2] > 0.5:
        crop = CROPS[next(index for index in range(5) if raw[7 + index] > 0.5)]
        age = _round(raw[14] * 30.0)
        return {
            "kind": "PLANT", "crop": crop, "age": age, "planted_day": day - age,
            "max_lifespan_step": _round(raw[16] * SEASON_STEPS),
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
    turns_per_day = int(configuration["turnsPerDay"])
    step = _round(raw[0] * SEASON_STEPS)
    day = step // turns_per_day
    hour = step % turns_per_day
    farms: list[dict[str, Any]] = []
    for farm_index in range(2):
        position = 7 + farm_index * 6
        hands_position = OBS_HAND_POSITIONS + farm_index * (MAX_HANDS + 1)
        hand_count = max(0, min(MAX_HANDS, _round(raw[hands_position] * float(MAX_HANDS))))
        tiles = [
            _tile(raw[62 + farm_index * 2600 + index * 26:62 + farm_index * 2600 + index * 26 + 26], day)
            for index in range(100)
        ]
        farms.append({
            # Money is always an exact integer in the official engine
            # (integer starting value, integer prices only); recovering it
            # via rounding removes the f32 normalize(10000) round-trip noise
            # that otherwise shows up as spurious canonical divergences.
            "money": float(_round(raw[5 + farm_index] * 10000.0)),
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
            "hires_today": _round(raw[position + 3] * float(MAX_HANDS)),
        })
    shed = _inventory(raw, OBS_SHED)
    seeds = {name: _round(raw[OBS_SEEDS + index] * 100.0) for index, name in enumerate(CROPS)}
    inventories = [_inventory(raw, OBS_INVENTORY)]
    hand_count = len(farms[player]["hands"])
    inventories.extend(_inventory(raw, OBS_HAND_INVENTORY + hand * 12) for hand in range(hand_count))
    shops = [SHOPS[_round(raw[OBS_SHOPS + slot] * 8.0) - 1] for slot in range(8) if raw[OBS_SHOPS + slot] > 0.0]
    return {
        "player": player,
        "farms": farms,
        "private": {"shed": shed, "seeds": seeds, "inventories": inventories},
        "market": {
            "inventory": {name: _round(raw[OBS_MARKET_INVENTORY + index] * 10000.0) for index, name in enumerate(PRODUCTS)},
            "prices": {name: _round(raw[OBS_MARKET_PRICES + index] * 1000.0) for index, name in enumerate(PRODUCTS)},
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
        # Optional instance-local Rayon worker count for the Rust batch
        # backend. None keeps the historical default (Rayon's global pool);
        # an explicit positive count builds a private pool so several batch
        # environments in one process never oversubscribe each other.
        raw_num_threads = self.configuration.get("numThreads")
        if raw_num_threads is None:
            num_threads = None
        else:
            num_threads = _as_int(raw_num_threads, "numThreads")
            if num_threads < 1:
                raise ValueError("numThreads must be a positive integer")
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
            num_threads,
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

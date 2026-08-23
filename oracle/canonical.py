"""Canonical full-state representation and field-path-level deep diff.

Both engines are mapped onto ONE canonical schema that mirrors the official
1.32.7 observation/state shape. Canonicalization never discards order,
multiplicity, privacy, lifecycle, reward, or status information:

- ``town.unlocked_shops`` keeps duplicates and unlock order;
- ``farms[*].tiles`` keeps board row/col order;
- per-seat ``privates[*]`` keep shed/seeds/farmer+hand inventories separate;
- lifecycle fields are compared under their official names; the fast engine's
  derived ``age`` is converted to the official ``placed_day``/
  ``planted_day`` form (bijective given the current day, so no information is
  lost);
- officially-sparse unit inventories are zero-filled over the fixed item key
  set (official semantics guarantee absent == 0 and never negative).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

PRODUCTS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER")
CROPS = PRODUCTS[:5]
ANIMALS = ("GOOSE", "COW", "SHEEP")
SHED_ITEMS = PRODUCTS + ANIMALS

MISSING = object()


@dataclass(frozen=True)
class FieldDiff:
    """One concrete difference at an exact field path."""

    path: str
    official_value: Any
    fast_value: Any

    def render(self, limit: int = 200) -> str:
        return (
            f"{self.path}: official={_render(self.official_value, limit)} "
            f"fast={_render(self.fast_value, limit)}"
        )


def _render(value: Any, limit: int = 200) -> str:
    if value is MISSING:
        return "<missing>"
    text = json.dumps(value, sort_keys=True, default=str)
    if len(text) > limit:
        text = text[: limit - 3] + "..."
    return text


def _plain(value: Any) -> Any:
    """Convert official Struct wrappers into plain dicts/lists."""
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    try:
        return {key: _plain(item) for key, item in value.items()}
    except AttributeError:
        return value


def _zero_filled(mapping: Mapping[str, Any], keys: Sequence[str]) -> dict[str, int]:
    return {key: int(mapping.get(key, 0)) for key in keys}


def _canonical_tile(tile: Any, day: int, *, from_fast: bool) -> Any:
    if not isinstance(tile, dict):
        return tile  # None (empty) or "LOCKED"
    kind = tile.get("kind")
    if kind == "PLANT":
        canonical = {
            "kind": "PLANT",
            "crop": tile["crop"],
            "planted_day": int(tile["planted_day"]),
            "max_lifespan_step": int(tile["max_lifespan_step"]),
            "yield_units": int(tile["yield_units"]),
            "watered_today": bool(tile["watered_today"]),
            "consecutive_unwatered": int(tile["consecutive_unwatered"]),
            "fertilized_until_day": int(tile["fertilized_until_day"]),
        }
        if from_fast:
            # Fast encodes planted_day redundantly as age; official form wins.
            if "age" in tile and "planted_day" not in tile:
                canonical["planted_day"] = int(day) - int(tile["age"])
        return canonical
    if kind in ("COOP", "PASTURE"):
        canonical: dict[str, Any] = {"kind": kind}
        if "animal" in tile:
            placed_day = tile.get("placed_day")
            if placed_day is None and "age" in tile:
                placed_day = int(day) - int(tile["age"])
            canonical.update({
                "animal": tile["animal"],
                "placed_day": int(placed_day),
                "yield_units": int(tile["yield_units"]),
                "consecutive_unfed": int(tile["consecutive_unfed"]),
                "fed_today": bool(tile["fed_today"]),
                "cared_today": bool(tile["cared_today"]),
                "fertilizer_available": bool(tile["fertilizer_available"]),
                "pending_care_bonus": int(tile["pending_care_bonus"]),
            })
        return canonical
    # "WEED" and any other dict tile compare verbatim.
    return {key: _plain(value) for key, value in sorted(tile.items())}


def _canonical_farm(farm: Mapping[str, Any], day: int, *, from_fast: bool) -> dict[str, Any]:
    tiles = farm["tiles"]
    return {
        "money": float(farm["money"]),
        "tiles": [
            [_canonical_tile(tiles[y][x], day, from_fast=from_fast) for x in range(len(tiles[y]))]
            for y in range(len(tiles))
        ],
        "farmer": [int(farm["farmer"][0]), int(farm["farmer"][1])],
        "hands": [[int(hand[0]), int(hand[1])] for hand in farm["hands"]],
        "unlocked_quadrants": [str(name) for name in farm["unlocked_quadrants"]],
        "hires_today": int(farm["hires_today"]),
    }


def canonical_state_official(
    env: Any,
    *,
    rewards: Sequence[float] | None = None,
    statuses: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Canonical full state from an official ``kaggle_environments`` env."""
    state = env.state
    obs0 = state[0].observation
    day = int(obs0.day)
    farms = [_canonical_farm(_plain(farm), day, from_fast=False) for farm in obs0.farms]
    privates = []
    for agent in state:
        private = _plain(agent.observation.private)
        privates.append({
            "shed": _zero_filled(private["shed"], SHED_ITEMS),
            "seeds": _zero_filled(private["seeds"], CROPS),
            "inventories": [
                _zero_filled(inv, SHED_ITEMS) for inv in private["inventories"]
            ],
        })
    market_plain = _plain(obs0.market)
    market: dict[str, Any] = {
        "inventory": {name: int(market_plain["inventory"][name]) for name in PRODUCTS},
        "prices": {name: int(market_plain["prices"][name]) for name in PRODUCTS},
    }
    if "params" in market_plain:
        market["params"] = _plain(market_plain["params"])
    return {
        "step": int(obs0.step),
        "day": day,
        "hour": int(obs0.hour),
        "farms": farms,
        "privates": privates,
        "market": market,
        "town": {"unlocked_shops": [str(s) for s in _plain(obs0.town)["unlocked_shops"]]},
        "rewards": [float(reward) for reward in (rewards if rewards is not None else [a.reward or 0.0 for a in state])],
        "statuses": [str(statuses[i] if statuses else state[i].status) for i in range(len(state))],
    }


def canonical_state_fast(
    observations: Sequence[Mapping[str, Any]],
    rewards: Sequence[float],
    statuses: Sequence[str],
) -> dict[str, Any]:
    """Canonical full state from the fast engine's decoded observation pair."""
    obs0 = observations[0]
    day = int(obs0["day"])
    farms = [_canonical_farm(observation["farms"][index], day, from_fast=True)
             for index, observation in enumerate(observations)]
    privates = []
    for observation in observations:
        private = observation["private"]
        privates.append({
            "shed": _zero_filled(private["shed"], SHED_ITEMS),
            "seeds": _zero_filled(private["seeds"], CROPS),
            "inventories": [_zero_filled(inv, SHED_ITEMS) for inv in private["inventories"]],
        })
    market_source = obs0["market"]
    market: dict[str, Any] = {
        "inventory": {name: int(market_source["inventory"][name]) for name in PRODUCTS},
        "prices": {name: int(market_source["prices"][name]) for name in PRODUCTS},
    }
    if "params" in market_source:
        market["params"] = _plain(market_source["params"])
    return {
        "step": int(obs0["step"]),
        "day": day,
        "hour": int(obs0["hour"]),
        "farms": farms,
        "privates": privates,
        "market": market,
        "town": {"unlocked_shops": [str(s) for s in obs0["town"]["unlocked_shops"]]},
        "rewards": [float(reward) for reward in rewards],
        "statuses": [str(status) for status in statuses],
    }


def _values_equal(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return left == right
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _values_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _values_equal(a, b) for a, b in zip(left, right)
        )
    if left is MISSING or right is MISSING:
        return False
    return type(left) is type(right) and left == right


def deep_diff(official: Any, fast: Any, path: str = "state") -> list[FieldDiff]:
    """Ordered field-path differences between two canonical states."""
    diffs: list[FieldDiff] = []

    def walk(left: Any, right: Any, current: str) -> None:
        if isinstance(left, dict) and isinstance(right, dict):
            for key in sorted(set(left) | set(right)):
                child = f"{current}.{key}" if current else str(key)
                if key not in left:
                    diffs.append(FieldDiff(child, MISSING, right[key]))
                elif key not in right:
                    diffs.append(FieldDiff(child, left[key], MISSING))
                else:
                    walk(left[key], right[key], child)
            return
        if isinstance(left, list) and isinstance(right, list):
            if len(left) != len(right):
                diffs.append(FieldDiff(
                    f"{current} (len {len(left)} vs {len(right)})",
                    _lengths(left), _lengths(right),
                ))
            for index in range(min(len(left), len(right))):
                walk(left[index], right[index], f"{current}[{index}]")
            return
        if not _values_equal(left, right):
            diffs.append(FieldDiff(current, left, right))

    walk(official, fast, path)
    return diffs


def _lengths(values: list[Any]) -> Any:
    return {"len": len(values)}

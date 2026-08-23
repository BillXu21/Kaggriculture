"""Deterministic state-aware legal-ish action-pair generator for the
same-action differential corpus.

Design contract (decision D-022):

- ONE fixed ``random.Random(generator_seed)`` stream drives every choice;
- each turn the generator reads ONLY the pre-transition fast-engine
  observation pair supplied by the replay harness and chooses exactly ONE
  action pair;
- that exact pair is what ``oracle.run_same_action_replay`` submits to BOTH
  engines before any comparison — the generator never runs policies
  independently after a divergence and never sees official-side state;
- "legal-ish" means well-formed under the official 1.32.7 wire contract
  (silent no-op / partial-fill semantics included deliberately: malformed
  entries, unknown ops, missing/non-integer quantities, order truncation
  beyond 10 market orders, and extra hand slots are part of the covered
  surface);
- the generator is fully deterministic given ``(generator_seed, turn,
  observations)``: replaying the same seed against the same engine states
  reproduces the identical trace, so any first divergence stays attributable
  and reproducible from ``(generator_seed, turn_index)`` alone.

Coverage: every submitted action family increments a counter exposed via
:attr:`LegalishActionGenerator.coverage`; the corpus report publishes the
histogram so uncovered families are visible instead of silently assumed.
"""

from __future__ import annotations

import random
from typing import Any, Mapping, Sequence

# Duplicated verbatim from fast_env/oracle.canonical to keep this module
# importable without the compiled extension (mirrors oracle.canonical).
PRODUCTS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER")
CROPS = PRODUCTS[:5]
ANIMALS = ("GOOSE", "COW", "SHEEP")
MOVES = ("NORTH", "SOUTH", "EAST", "WEST")
STRUCTURE_FOR_ANIMAL = {"GOOSE": "COOP", "COW": "PASTURE", "SHEEP": "PASTURE"}
ANIMAL_COST = {"GOOSE": 300, "COW": 400, "SHEEP": 500}
SHED_POSITION = (4, 4)
MAX_MARKET_ORDERS = 10

Observation = Mapping[str, Any]
ActionPair = list[dict[str, Any]]


class LegalishActionGenerator:
    """Fixed-RNG reflex policy producing one action pair per turn."""

    def __init__(self, seed: int) -> None:
        self._seed = int(seed)
        self._rng = random.Random(self._seed)
        self._coverage: dict[str, int] = {}

    # ------------------------------------------------------------------ API

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def coverage(self) -> dict[str, int]:
        """Histogram of attempted action families (attempts, not successes)."""
        return dict(self._coverage)

    def next_pair(
        self, turn: int, observations: Sequence[Observation]
    ) -> ActionPair:
        """Choose ONE action pair from the pre-transition observations."""
        del turn  # turn-independent state; kept for the ActionSource protocol
        return [
            self._seat_action(player, observations[player]) for player in range(2)
        ]

    # -------------------------------------------------------------- internals

    def _mark(self, family: str) -> None:
        self._coverage[family] = self._coverage.get(family, 0) + 1

    def _seat_action(self, player: int, obs: Observation) -> dict[str, Any]:
        farm = obs["farms"][player]
        private = obs["private"]
        action: dict[str, Any] = {
            "farmer": self._farmer_action(farm, private, int(obs["step"])),
            "hands": self._hand_actions(farm, private),
            "market": self._market_orders(obs, player),
        }
        return action

    # ---------------------------------------------------------- market orders

    def _market_orders(self, obs: Observation, player: int) -> list[list[Any]]:
        rng = self._rng
        money = float(obs["farms"][player]["money"])
        shed = obs["private"]["shed"]
        orders: list[list[Any]] = []

        # Early economy bootstrap so plant/feed/sell chains can start.
        if obs["step"] < 4:
            crop = rng.choice(CROPS)
            orders.append(["BUY_SEED", crop, rng.randint(2, 6)])
            self._mark("market.BUY_SEED")
        if obs["step"] < 6:
            orders.append(["BUY_PRODUCT", "WHEAT", rng.randint(4, 12)])
            self._mark("market.BUY_PRODUCT")

        if rng.random() < 0.08:
            orders.append(["HIRE"])
            self._mark("market.HIRE")
        if rng.random() < 0.04:
            orders.append(["BUY_LAND"])
            self._mark("market.BUY_LAND")
        if rng.random() < 0.50:
            animal = rng.choice(ANIMALS)
            quantity = rng.randint(1, 2)
            if money >= ANIMAL_COST[animal] * quantity:
                orders.append(["BUY_ANIMAL", animal, quantity])
                self._mark("market.BUY_ANIMAL")
        if rng.random() < 0.20:
            product = rng.choice(PRODUCTS[:-1])
            quantity = rng.randint(1, 4)
            # Selling from an empty shed exercises the official
            # clamp/no-op path legally.
            orders.append(["SELL", product, quantity])
            self._mark("market.SELL")
        if rng.random() < 0.10:
            # FERTILIZER included deliberately: stocked shed fertilizer is
            # what lets the FERTILIZE unit family be attempted downstream.
            orders.append(["BUY_PRODUCT", rng.choice(PRODUCTS), rng.randint(1, 4)])
            self._mark("market.BUY_PRODUCT")

        # Malformed / no-op market surface (official skips these silently).
        roll = rng.random()
        if roll < 0.06:
            orders.append(["BOGUS_OP", "WHEAT", 1])
            self._mark("market.malformed")
        elif roll < 0.11:
            orders.append(["BUY_SEED", "WHEAT"])  # missing quantity
            self._mark("market.malformed")
        elif roll < 0.16:
            orders.append(["BUY_SEED", "WHEAT", "two"])  # non-integer quantity
            self._mark("market.malformed")
        elif roll < 0.20:
            orders.append(["SELL", "UNICORN", 1])  # unknown item
            self._mark("market.malformed")

        # Order-truncation coverage: bursts beyond maxMarketOrdersPerTurn=10.
        if rng.random() < 0.05:
            orders.extend([["BUY_SEED", "WHEAT", 1]] * 12)
            self._mark("market.order_truncation")

        return orders

    # ------------------------------------------------------------ unit actions

    def _farmer_action(
        self,
        farm: Mapping[str, Any],
        private: Mapping[str, Any],
        step: int,
    ) -> list[Any]:
        rng = self._rng
        position = (int(farm["farmer"][0]), int(farm["farmer"][1]))
        carried = private["inventories"][0]
        shed = private["shed"]
        tile = farm["tiles"][position[1]][position[0]]

        if isinstance(tile, dict):
            kind = tile.get("kind")
            if kind == "PLANT":
                if not tile["watered_today"]:
                    self._mark("unit.WATER")
                    return ["WATER"]
                if int(tile["max_lifespan_step"]) - step <= 24 or rng.random() < 0.30:
                    self._mark("unit.HARVEST")
                    return ["HARVEST"]
                if shed["FERTILIZER"] > 0 and rng.random() < 0.35:
                    self._mark("unit.FERTILIZE")
                    return ["FERTILIZE"]
            elif kind == "WEED":
                self._mark("unit.DIG")
                return ["DIG"]
            elif kind in ("COOP", "PASTURE") and "animal" in tile:
                if not tile["fed_today"] and carried["WHEAT"] > 0:
                    self._mark("unit.FEED")
                    return ["FEED"]
                if not tile["cared_today"]:
                    self._mark("unit.CARE")
                    return ["CARE"]
                if tile["fertilizer_available"] and rng.random() < 0.70:
                    self._mark("unit.COLLECT_FERTILIZER")
                    return ["COLLECT_FERTILIZER"]
                if rng.random() < 0.40:
                    self._mark("unit.PASS")
                    return ["PASS"]
            elif kind in ("COOP", "PASTURE"):
                animal = self._carried_animal(carried)
                if animal is not None and STRUCTURE_FOR_ANIMAL[animal] == kind:
                    self._mark("unit.PLACE")
                    return ["PLACE", animal]
        elif tile is None:
            seeds = private["seeds"]
            owned = [crop for crop in CROPS if seeds[crop] > 0]
            if owned and rng.random() < 0.55:
                crop = rng.choice(owned)
                self._mark("unit.PLANT")
                return ["PLANT", crop]
            animal = self._carried_animal(carried)
            if animal is not None and rng.random() < 0.60:
                self._mark("unit.BUILD_" + STRUCTURE_FOR_ANIMAL[animal])
                return ["BUILD_" + STRUCTURE_FOR_ANIMAL[animal]]
            if rng.random() < 0.02:
                build = rng.choice(("BUILD_COOP", "BUILD_PASTURE"))
                self._mark("unit." + build)
                return [build]

        # Logistics: pick up wheat for feeding, fertilizer, or shed animals.
        if rng.random() < 0.35:
            wanted = self._wanted_pickup(private, carried)
            if wanted is not None:
                if position == SHED_POSITION:
                    self._mark("unit.PICKUP")
                    return ["PICKUP", wanted, rng.randint(1, 2)]
                return self._step_toward(position, SHED_POSITION)

        # Drop carried products at the shed occasionally.
        if (
            position == SHED_POSITION
            and any(carried[name] > 0 for name in PRODUCTS[:-1])
            and rng.random() < 0.30
        ):
            self._mark("unit.DROP")
            return ["DROP"]

        if rng.random() < 0.15:
            self._mark("unit.PASS")
            return ["PASS"]

        target = self._interest_target(farm)
        if target is None or rng.random() < 0.35:
            self._mark("unit.move_random")
            return [rng.choice(MOVES)]
        self._mark("unit.move_toward")
        return self._step_toward(position, target)

    def _hand_actions(
        self, farm: Mapping[str, Any], private: Mapping[str, Any]
    ) -> list[list[Any]]:
        rng = self._rng
        shed = private["shed"]
        actions: list[list[Any]] = []
        # Structures with a live animal needing ANY attention (feed, care,
        # or fertilizer collection) — not just unfed ones.
        attended = self._animal_structures(farm)
        for index, hand_position in enumerate(farm["hands"]):
            position = (int(hand_position[0]), int(hand_position[1]))
            carried = private["inventories"][index + 1]
            tile = farm["tiles"][position[1]][position[0]]

            if isinstance(tile, dict) and tile.get("kind") in ("COOP", "PASTURE"):
                if "animal" in tile:
                    if not tile["fed_today"] and carried["WHEAT"] > 0:
                        self._mark("hand.FEED")
                        actions.append(["FEED"])
                        continue
                    if not tile["cared_today"]:
                        self._mark("hand.CARE")
                        actions.append(["CARE"])
                        continue
                    if tile["fertilizer_available"] and rng.random() < 0.60:
                        self._mark("hand.COLLECT_FERTILIZER")
                        actions.append(["COLLECT_FERTILIZER"])
                        continue
                else:
                    animal = self._carried_animal(carried)
                    if animal is not None and STRUCTURE_FOR_ANIMAL[animal] == tile["kind"]:
                        self._mark("hand.PLACE")
                        actions.append(["PLACE", animal])
                        continue

            if isinstance(tile, dict) and tile.get("kind") == "WEED":
                self._mark("hand.DIG")
                actions.append(["DIG"])
                continue

            if attended:
                target = min(attended, key=lambda t: (_manhattan(position, t), t[0], t[1]))
                need_feed = self._tile_at(farm, target).get("fed_today") is False
                if need_feed and carried["WHEAT"] == 0:
                    if shed["WHEAT"] > 0:
                        if position == SHED_POSITION:
                            self._mark("hand.PICKUP")
                            actions.append(["PICKUP", "WHEAT", rng.randint(1, 2)])
                        else:
                            self._mark("hand.move_toward")
                            actions.append(self._step_toward(position, SHED_POSITION))
                        continue
                if position == target:
                    if need_feed and carried["WHEAT"] > 0:
                        self._mark("hand.FEED")
                        actions.append(["FEED"])
                    elif rng.random() < 0.50:
                        self._mark("hand.PASS")
                        actions.append(["PASS"])
                    else:
                        self._mark("hand.move_random")
                        actions.append([rng.choice(MOVES)])
                else:
                    self._mark("hand.move_toward")
                    actions.append(self._step_toward(position, target))
                continue

            if rng.random() < 0.20:
                self._mark("hand.PASS")
                actions.append(["PASS"])
            else:
                self._mark("hand.move_random")
                actions.append([rng.choice(MOVES)])

        # Extra hand slots beyond the live hand count are silent no-ops
        # officially; occasionally submit them for truncation coverage.
        if farm["hands"] and rng.random() < 0.08:
            actions.append(["PASS"])
            self._mark("hand.extra_slot")
        return actions

    # ---------------------------------------------------------------- helpers

    def _wanted_pickup(
        self, private: Mapping[str, Any], carried: Mapping[str, int]
    ) -> str | None:
        shed = private["shed"]
        # Shed animals first: they enable the BUILD/PLACE chain.
        for animal in ANIMALS:
            if shed[animal] > 0:
                return animal
        if carried["WHEAT"] == 0 and shed["WHEAT"] > 0:
            return "WHEAT"
        if carried["FERTILIZER"] == 0 and shed["FERTILIZER"] > 0:
            return "FERTILIZER"
        return None

    @staticmethod
    def _carried_animal(carried: Mapping[str, int]) -> str | None:
        for animal in ANIMALS:
            if carried[animal] > 0:
                return animal
        return None

    @staticmethod
    def _animal_structures(farm: Mapping[str, Any]) -> list[tuple[int, int]]:
        found: list[tuple[int, int]] = []
        for y, row in enumerate(farm["tiles"]):
            for x, tile in enumerate(row):
                if (
                    isinstance(tile, dict)
                    and tile.get("kind") in ("COOP", "PASTURE")
                    and "animal" in tile
                ):
                    found.append((x, y))
        return found

    @staticmethod
    def _tile_at(farm: Mapping[str, Any], position: tuple[int, int]) -> Mapping[str, Any]:
        tile = farm["tiles"][position[1]][position[0]]
        return tile if isinstance(tile, dict) else {}

    def _interest_target(self, farm: Mapping[str, Any]) -> tuple[int, int] | None:
        """Nearest weed, else a random animal structure / thirsty plant."""
        rng = self._rng
        position = (int(farm["farmer"][0]), int(farm["farmer"][1]))
        weeds: list[tuple[int, int]] = []
        plants: list[tuple[int, int]] = []
        for y, row in enumerate(farm["tiles"]):
            for x, tile in enumerate(row):
                if isinstance(tile, dict):
                    if tile.get("kind") == "WEED":
                        weeds.append((x, y))
                    elif tile.get("kind") == "PLANT" and not tile["watered_today"]:
                        plants.append((x, y))
        if weeds:
            return min(weeds, key=lambda t: (_manhattan(position, t), t[0], t[1]))
        structures = self._animal_structures(farm)
        if structures and rng.random() < 0.50:
            return structures[rng.randrange(len(structures))]
        if plants:
            return plants[rng.randrange(len(plants))]
        return None

    def _step_toward(
        self, position: tuple[int, int], target: tuple[int, int]
    ) -> list[Any]:
        dx = target[0] - position[0]
        dy = target[1] - position[1]
        if dx == 0 and dy == 0:
            return ["PASS"]
        if abs(dy) >= abs(dx):
            return ["NORTH" if dy < 0 else "SOUTH"]
        return ["WEST" if dx < 0 else "EAST"]


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

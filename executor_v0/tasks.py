"""Explicit V0 task records and deterministic per-turn generation (issue #1 §5).

Small frozen dataclasses plus one pure generator; no graph, no framework, no
stale-task persistence. Tasks are regenerated from the actual observation
every primitive turn; the generator never pretends a task was completed and
never mutates its inputs.

Priority groups (issue #1 urgency ordering):

- ``MAINTENANCE``: hard deadline/mechanic-upkeep work (WATER, FEED,
  COLLECT_FERTILIZER when the canonical raw ``fertilizer_available`` field
  is True) — two dry days kill a plant, unfed animals escape.
- ``PRODUCTIVE``: revenue work (HARVEST of actually harvestable tiles).
- ``MANAGER``: manager-directed changes (DIG/PLANT reconciliation,
  BUILD/PLACE deficits, CARE/FERTILIZE allocations, BUY_LAND).
- ``LOGISTICS``: cleanup/logistics (SELL in the current bin, BUY_* shortages
  implied by active tasks only).
- ``OPTIONAL``: config-gated spare-capacity work; never part of generated
  manager workload or mandatory accounting.

Final tie-break is the stable task key; the foreman later adds distance.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

from bc_manager.constants import ANIMAL_ORDER, CROP_ORDER
from executor_v0.layout import (
    SHED_HUB_ANCHOR,
    AnimalLayoutResult,
    CropReconciliationResult,
    DayLayoutResult,
    plan_animal_layout,
    plan_day_layouts,
    quadrant_of,
    reconcile_crops,
    tile_role,
)
from executor_v0.plan import DailyPlan
from replay_daily.constants import ANIMALS, CROPS, LAND_ORDER, PRODUCTS
from replay_daily.lifecycle import canonical_board

__all__ = [
    "Priority",
    "Task",
    "GenerationResult",
    "generate_optional_idle_cleanup_tasks",
    "generate_optional_water_tasks",
    "generate_tasks",
]

_FEED_ITEM = "WHEAT"
_FERTILIZER_ITEM = "FERTILIZER"
_TOTAL_DAYS = 30
_TURNS_PER_DAY = 24
_FINAL_ACTIONABLE_STEP = _TOTAL_DAYS * _TURNS_PER_DAY - 2


class Priority(IntEnum):
    MAINTENANCE = 0
    PRODUCTIVE = 1
    MANAGER = 2
    LOGISTICS = 3
    OPTIONAL = 4


@dataclass(frozen=True)
class Task:
    """One unit of pending mechanical work; JSON-safe and orderable."""

    key: str
    kind: str                                  # engine opcode-style kind
    priority: Priority
    tile: tuple[int, int] | None = None        # canonical [y, x] or None
    quantity: int = 1
    required_item: str | None = None
    deadline_hour: int | None = None
    depends_on: tuple[str, ...] = ()
    crop: str | None = None
    animal: str | None = None
    product: str | None = None
    source: str = ""                           # free-form origin note

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "kind": self.kind,
            "priority": self.priority.name,
            "tile": list(self.tile) if self.tile is not None else None,
            "quantity": self.quantity,
            "required_item": self.required_item,
            "deadline_hour": self.deadline_hour,
            "depends_on": list(self.depends_on),
            "crop": self.crop,
            "animal": self.animal,
            "product": self.product,
            "source": self.source,
        }

    @property
    def sort_key(self) -> tuple[int, int, str]:
        return (
            int(self.priority),
            self.deadline_hour if self.deadline_hour is not None else 1 << 30,
            self.key,
        )


@dataclass(frozen=True)
class GenerationResult:
    tasks: tuple[Task, ...]
    unresolved: tuple[str, ...] = ()

    def sorted_tasks(self) -> tuple[Task, ...]:
        return tuple(sorted(self.tasks, key=lambda t: t.sort_key))


# ------------------------------------------------------------------ helpers


def _validate_obs(obs: Mapping, seat: int) -> None:
    if seat not in (0, 1):
        raise ValueError(f"seat must be 0 or 1, got {seat!r}")
    for key in ("farms", "day", "hour"):
        if key not in obs:
            raise ValueError(f"observation missing required field {key!r}")
    day, hour = obs["day"], obs["hour"]
    if not isinstance(day, int) or isinstance(day, bool) \
            or not 0 <= day < _TOTAL_DAYS:
        raise ValueError(f"obs['day'] must be an integer in [0, 29], got "
                         f"{day!r}")
    if not isinstance(hour, int) or isinstance(hour, bool) \
            or not 0 <= hour < 24:
        raise ValueError(f"obs['hour'] must be an integer in [0, 23], got "
                         f"{hour!r}")
    farms = obs["farms"]
    if not isinstance(farms, (list, tuple)) or len(farms) < 2:
        raise ValueError("obs['farms'] must hold both seats")
    tiles = farms[seat]["tiles"]
    if len(tiles) != 10 or any(len(row) != 10 for row in tiles):
        raise ValueError("own board must be 10x10")


def _canonical_own_state(obs: Mapping, seat: int) -> dict[str, Any]:
    """Canonicalize own board/state exactly like replay_daily.extractor."""
    day = int(obs["day"])
    step = int(obs.get("step", 0))
    farm = obs["farms"][seat]
    private = obs.get("private") or {}
    return {
        "board": canonical_board(farm["tiles"], day, step),
        "unlocked_quadrants": list(farm["unlocked_quadrants"]),
        "farmer": list(farm["farmer"]),
        "money": farm["money"],
        "shed": dict(private.get("shed") or {}),
        "seeds": dict(private.get("seeds") or {}),
        "inventories": [dict(inv) for inv in (private.get("inventories") or [])],
    }


def _available(state: Mapping, item: str) -> int:
    """Shed plus all carried inventories; own private state only."""
    total = int(state["shed"].get(item, 0))
    for inv in state["inventories"]:
        total += int(inv.get(item, 0))
    return total


def _plan_counts(mapping: Mapping[str, int], order) -> dict[str, int]:
    return {name: int(mapping.get(name, 0)) for name in order}


def _tile_at(board, coord):
    y, x = coord
    return board[y][x]


def _water_urgency(tile: Mapping) -> str | None:
    """Exact-mechanics watering class for one PLANT tile.

    Pinned 1.32.7 semantics (rust engine ``apply_unit_action`` op 10 +
    ``_daily_refresh_plants``; MECHANICS.md section 7):

    - planting day counts as unwatered (fresh tiles get
      ``consecutive_unwatered=1``);
    - a second consecutive unwatered daily refresh converts the plant to a
      WEED (crop destroyed);
    - WATER on a single-harvest crop (interval 0) inside its late growth
      window adds +1 yield (+2 when fertilized) up to ``max_yield``;
    - ongoing-crop production at the next morning refresh yields +2 instead
      of +1 only when the plant was watered today AND fertilizer is active;
    - every other unwatered plant loses nothing observable today.

    Returns ``"must"`` (weed boundary), ``"yield"`` (measurable production
    gain), or ``None`` (safe to defer -- watering it would only reset the
    drought counter, which must not starve harvest/maintenance labor).
    """
    if tile.get("watered_today") is True:
        return None
    if int(tile.get("consecutive_unwatered") or 0) >= 1:
        return "must"
    data = CROPS.get(tile.get("crop"))
    derived = tile.get("derived") or {}
    if data is None:
        return None
    age = int(derived.get("age_days") or 0)
    if data["interval"] == 0:
        window_start = (data["max_yield_day"] + 1) // 2
        if (window_start <= age <= data["max_yield_day"]
                and int(tile.get("yield_units") or 0) < data["max_yield"]):
            return "yield"
        return None
    # Ongoing crop: watering matters tomorrow only when tomorrow's refresh is
    # a production day AND fertilizer is active (+2 vs +1); age increments
    # before the production check at the refresh.
    if not derived.get("fertilizer_active"):
        return None
    dsf = (age + 1) - data["first_yield_day"]
    if dsf >= 0 and dsf % data["interval"] == 0             and dsf // data["interval"] + 1 <= data["max_yield"]:
        return "yield"
    return None


def _at_terminal_action_horizon(obs: Mapping) -> bool:
    """Whether this action is the last one that can precede terminal state.

    The pinned engine starts at step 0 and marks the state done after the
    action at step 718 advances it to terminal step 719.  A missing step is
    reconstructed from the validated day/hour pair for replay-shaped inputs.
    """
    day = int(obs["day"])
    hour = int(obs["hour"])
    step = int(obs.get("step", day * _TURNS_PER_DAY + hour))
    return step >= _FINAL_ACTIONABLE_STEP


def generate_optional_idle_cleanup_tasks(
    obs: Mapping,
    seat: int,
    mode: str = "weed_water",
) -> tuple[Task, ...]:
    """Return safe PASS-only cleanup candidates for the requested mode.

    Cleanup is intentionally separate from ``generate_tasks`` so it cannot
    contribute to manager workload, shortage purchases, or work debt.  The
    caller may only assign these candidates to workers whose normal action was
    PASS; they are never a normal foreman dispatch pool. ``water_only`` emits
    only optional WATER tasks, while ``weed_water`` emits weed DIG tasks first
    and then optional WATER tasks.
    """
    if mode not in ("none", "water_only", "weed_water"):
        raise ValueError(f"unsupported optional cleanup mode: {mode!r}")
    if mode == "none":
        return ()
    _validate_obs(obs, seat)
    try:
        state = _canonical_own_state(obs, seat)
    except (KeyError, TypeError, ValueError):
        # Optional work must never turn malformed source data into a task.
        return ()
    unlocked = set(state["unlocked_quadrants"])
    optional: list[Task] = []
    for y, row in enumerate(state["board"]):
        for x, tile in enumerate(row):
            coord = (y, x)
            if quadrant_of(y, x) not in unlocked:
                continue
            if tile_role(tile) == "weed":
                if mode != "weed_water":
                    continue
                optional.append(Task(
                    key=f"DIG_CLEANUP:{y},{x}", kind="DIG",
                    priority=Priority.OPTIONAL, tile=coord, crop="WEED",
                    source="dig_cleanup"))
                continue
            if not isinstance(tile, Mapping) or tile.get("kind") != "PLANT":
                continue
            watered_today = tile.get("watered_today")
            consecutive_unwatered = tile.get("consecutive_unwatered")
            if not isinstance(watered_today, bool) or watered_today \
                    or not isinstance(consecutive_unwatered, int) \
                    or isinstance(consecutive_unwatered, bool) \
                    or consecutive_unwatered != 0:
                continue
            crop = tile.get("crop")
            derived = tile.get("derived")
            if crop not in CROPS or not isinstance(derived, Mapping) \
                    or not isinstance(derived.get("age_days"), int) \
                    or isinstance(derived.get("age_days"), bool):
                continue
            if _water_urgency(tile) is not None:
                continue
            optional.append(Task(
                key=f"WATER_OPTIONAL:{y},{x}", kind="WATER",
                priority=Priority.OPTIONAL, tile=coord, crop=crop,
                source="water_optional_spare"))
    return tuple(sorted(
        optional,
        key=lambda task: (0 if task.kind == "DIG" else 1, task.key),
    ))


def generate_optional_water_tasks(obs: Mapping, seat: int) -> tuple[Task, ...]:
    """Return the distinct water-only PASS cleanup candidate set."""
    return generate_optional_idle_cleanup_tasks(obs, seat, mode="water_only")


# --------------------------------------------------------------- generation


def generate_tasks(
    obs: Mapping,
    seat: int,
    *,
    feasible_plan: DailyPlan,
    remaining_sells: Mapping[str, int],
    reconcile_result: CropReconciliationResult | None = None,
    animal_layout_result: AnimalLayoutResult | None = None,
) -> GenerationResult:
    """Regenerate the full V0 task set from the current observation.

    Pure: inputs are never mutated. ``reconcile_result`` /
    ``animal_layout_result`` may be supplied by a caller that already ran the
    stage-3 planners (recommended for consistency); otherwise they are
    recomputed from this observation with the farmer position as anchor.
    """
    _validate_obs(obs, seat)
    for product in remaining_sells:
        if product not in PRODUCTS:
            raise ValueError(
                f"remaining_sells has unknown product {product!r}; expected "
                f"one of {list(PRODUCTS)}")
    state = _canonical_own_state(obs, seat)
    board = state["board"]
    unlocked = state["unlocked_quadrants"]
    # Layout/proximity anchor is the persistent central logistics hub, never
    # the moving farmer: raw farmer coordinates are [x, y] while canonical
    # task tiles are [y, x] (the old transposition skewed every proximity
    # ordering), and anchoring layouts to the farmer made target slots churn
    # all day as workers walked around (issue #7).
    anchor = SHED_HUB_ANCHOR
    hour = int(obs["hour"])
    unresolved: list[str] = []
    tasks: list[Task] = []

    # Existing on-board animals always count toward the feasible targets;
    # the layout planner slots positive deficits only (never duplicates).
    current_animals = {name: 0 for name in ANIMAL_ORDER}
    for row in board:
        for tile in row:
            if isinstance(tile, Mapping) and tile.get("animal") \
                    in current_animals:
                current_animals[tile["animal"]] += 1

    if reconcile_result is None and animal_layout_result is None:
        day_layout: DayLayoutResult | None = plan_day_layouts(
            board, unlocked_quadrants=unlocked,
            crop_targets=feasible_plan.crop_targets_dict,
            animals_needed={
                name: max(0, feasible_plan.animal_targets_dict[name]
                          - current_animals[name])
                for name in ANIMAL_ORDER
            },
            anchor=anchor)
        assert day_layout is not None
        reconcile_result = day_layout.crops
        animal_layout_result = day_layout.animals
    else:
        if reconcile_result is None:
            reconcile_result = reconcile_crops(
                board, unlocked_quadrants=unlocked,
                crop_targets=feasible_plan.crop_targets_dict, anchor=anchor)
        if animal_layout_result is None:
            animal_layout_result = plan_animal_layout(
                board, unlocked_quadrants=unlocked,
                animals_needed={
                    name: max(0, feasible_plan.animal_targets_dict[name]
                              - current_animals[name])
                    for name in ANIMAL_ORDER
                },
                anchor=anchor)

    # ---- scan the canonical board once -------------------------------
    water_must_targets: list[tuple[int, int]] = []
    water_yield_targets: list[tuple[int, int]] = []
    harvest_plant_targets: list[tuple[int, int]] = []
    harvest_animal_targets: list[tuple[int, int]] = []
    feed_targets: list[tuple[int, int]] = []
    collect_targets: list[tuple[int, int]] = []
    care_eligible: dict[str, list[tuple[int, int]]] = {
        name: [] for name in ANIMAL_ORDER}
    fert_eligible: dict[str, list[tuple[int, int]]] = {
        crop: [] for crop in CROP_ORDER}
    terminal_action_horizon = _at_terminal_action_horizon(obs)

    for y, row in enumerate(board):
        for x, tile in enumerate(row):
            coord = (y, x)
            if not isinstance(tile, Mapping):
                continue  # empty / locked / weed sentinels
            if "animal" in tile:
                species = tile["animal"]
                if tile.get("yield_units", 0) > 0:
                    harvest_animal_targets.append(coord)
                if tile.get("fed_today") is not True:
                    feed_targets.append(coord)
                # Collectible fertilizer readiness comes from the canonical
                # raw `fertilizer_available` field; missing/unknown is never
                # treated as available.
                if tile.get("fertilizer_available") is True:
                    collect_targets.append(coord)
                if species in care_eligible \
                        and tile.get("cared_today") is not True:
                    care_eligible[species].append(coord)
            elif tile.get("kind") == "PLANT":
                urgency = _water_urgency(tile)
                if urgency == "must":
                    water_must_targets.append(coord)
                elif urgency == "yield":
                    water_yield_targets.append(coord)
                derived = tile.get("derived") or {}
                if derived.get("currently_harvestable"):
                    harvest_plant_targets.append(coord)
                crop = tile.get("crop")
                fertilizer_available = tile.get("fertilizer_available")
                if fertilizer_available is None:
                    fertilizer_available = not derived.get("fertilizer_active")
                if crop in fert_eligible and fertilizer_available:
                    fert_eligible[crop].append(coord)

    def by_proximity(coords_list):
        return sorted(coords_list,
                      key=lambda c: (abs(c[0] - anchor[0]) + abs(c[1] - anchor[1]),
                                     c[0], c[1]))

    # ---- MAINTENANCE --------------------------------------------------
    # Only weed-boundary watering is hard-deadline maintenance; yield-window
    # watering is productive work; all other unwatered plants are safely
    # deferred (blanket daily WATER was a major labor sink, issue #7).
    for coord in by_proximity(water_must_targets):
        tasks.append(Task(key=f"WATER:{coord[0]},{coord[1]}", kind="WATER",
                          priority=Priority.MAINTENANCE, tile=coord,
                          crop=_tile_at(board, coord)["crop"],
                          source="water_must_weed_boundary"))
    for coord in by_proximity(water_yield_targets):
        tasks.append(Task(key=f"WATER:{coord[0]},{coord[1]}", kind="WATER",
                          priority=Priority.PRODUCTIVE, tile=coord,
                          crop=_tile_at(board, coord)["crop"],
                          source="water_yield_window"))
    for coord in by_proximity(feed_targets):
        tasks.append(Task(key=f"FEED:{coord[0]},{coord[1]}", kind="FEED",
                          priority=Priority.MAINTENANCE, tile=coord,
                          animal=_tile_at(board, coord)["animal"],
                          required_item=_FEED_ITEM, quantity=1,
                          source="mechanical"))
    for coord in by_proximity(collect_targets):
        tasks.append(Task(
            key=f"COLLECT_FERTILIZER:{coord[0]},{coord[1]}",
            kind="COLLECT_FERTILIZER", priority=Priority.MAINTENANCE,
            tile=coord, animal=_tile_at(board, coord)["animal"],
            source="mechanical"))

    # ---- PRODUCTIVE ---------------------------------------------------
    for coord in by_proximity(harvest_plant_targets):
        tile = _tile_at(board, coord)
        crop_data = CROPS.get(tile.get("crop"))
        waits_for_water = (
            coord in water_yield_targets
            and crop_data is not None
            and not crop_data["ongoing"]
            and not terminal_action_horizon
        )
        tasks.append(Task(key=f"HARVEST:{coord[0]},{coord[1]}",
                          kind="HARVEST", priority=Priority.PRODUCTIVE,
                          tile=coord, crop=tile["crop"],
                          depends_on=(f"WATER:{coord[0]},{coord[1]}",)
                          if waits_for_water else (),
                          source="mechanical"))
    for coord in by_proximity(harvest_animal_targets):
        tasks.append(Task(key=f"HARVEST:{coord[0]},{coord[1]}",
                          kind="HARVEST", priority=Priority.PRODUCTIVE,
                          tile=coord, animal=_tile_at(board, coord)["animal"],
                          source="mechanical"))

    # ---- MANAGER: crop reconciliation ---------------------------------
    for dig in reconcile_result.digs:
        tasks.append(Task(key=f"DIG:{dig.coord[0]},{dig.coord[1]}",
                          kind="DIG", priority=Priority.MANAGER,
                          tile=dig.coord, crop=dig.crop,
                          source="manager_reconciliation"))
    for intent in reconcile_result.plants:
        depends = []
        if any(d.coord == intent.coord for d in reconcile_result.digs):
            depends.append(f"DIG:{intent.coord[0]},{intent.coord[1]}")
        # Planting consumes the GLOBAL own seed pool (`private.seeds[crop]`)
        # atomically at the engine; seeds are never picked up or carried.
        # Seed sufficiency is enforced/reserved by the foreman per turn and
        # shortages surface as BUY_SEED logistics tasks below.
        tasks.append(Task(
            key=f"PLANT:{intent.crop}:{intent.coord[0]},{intent.coord[1]}",
            kind="PLANT", priority=Priority.MANAGER, tile=intent.coord,
            crop=intent.crop,
            depends_on=tuple(depends),
            source="manager_reconciliation"))
    for crop, count in reconcile_result.unresolved_deficits:
        unresolved.append(f"crop_deficit_unresolved:{crop}:{count}")

    # ---- MANAGER: animal deficits --------------------------------------
    # Prerequisite gating (issue #7): a BUILD is only emitted when its PLACE
    # can plausibly execute today -- the animal is already owned (shed or
    # carried) or affordable at current cash. Otherwise the structure would
    # sit empty; the deficit resurfaces automatically once cash/animals
    # change because tasks regenerate every turn.
    place_keys_by_animal: dict[str, list[str]] = {}
    for slot in animal_layout_result.placements:
        if slot.source == "empty_structure":
            place_key = f"PLACE:{slot.animal}:{slot.coord[0]},{slot.coord[1]}"
            tasks.append(Task(key=place_key, kind="PLACE",
                              priority=Priority.MANAGER, tile=slot.coord,
                              animal=slot.animal, required_item=slot.animal,
                              quantity=1, source="manager_target"))
            place_keys_by_animal.setdefault(slot.animal, []).append(place_key)
            continue
        animal_owned = _available(state, slot.animal) > 0
        animal_affordable = float(state["money"])             >= float(ANIMALS[slot.animal]["cost"])
        if not (animal_owned or animal_affordable):
            unresolved.append(
                f"build_deferred_no_animal:{slot.animal}:"
                f"{slot.coord[0]},{slot.coord[1]}")
            continue
        build_deps: list[str] = []
        if slot.source == "weed_reclaim":
            dig_key = f"DIG:{slot.coord[0]},{slot.coord[1]}"
            tasks.append(Task(key=dig_key, kind="DIG",
                              priority=Priority.MANAGER, tile=slot.coord,
                              crop="WEED", source="weed_reclaim"))
            build_deps.append(dig_key)
        build_kind = ("BUILD_COOP" if slot.structure == "COOP"
                      else "BUILD_PASTURE")
        build_key = f"{build_kind}:{slot.coord[0]},{slot.coord[1]}"
        tasks.append(Task(key=build_key, kind=build_kind,
                          priority=Priority.MANAGER, tile=slot.coord,
                          depends_on=tuple(build_deps),
                          source=("manager_layout_conversion"
                                  if slot.source == "crop_sacrifice"
                                  else ("weed_reclaim"
                                        if slot.source == "weed_reclaim"
                                        else "manager_layout"))))
        place_key = f"PLACE:{slot.animal}:{slot.coord[0]},{slot.coord[1]}"
        tasks.append(Task(key=place_key, kind="PLACE",
                          priority=Priority.MANAGER, tile=slot.coord,
                          animal=slot.animal, required_item=slot.animal,
                          quantity=1, depends_on=(build_key,),
                          source="manager_target"))
        place_keys_by_animal.setdefault(slot.animal, []).append(place_key)
    for animal, count in animal_layout_result.unresolved:
        unresolved.append(f"animal_deficit_unresolved:{animal}:{count}")

    # ---- MANAGER: CARE / FERTILIZE exact allocations --------------------
    care_requests = _plan_counts(feasible_plan.care_by_animal_dict,
                                 ANIMAL_ORDER)
    for species in ANIMAL_ORDER:
        budget = care_requests[species]
        for coord in by_proximity(care_eligible[species])[:budget]:
            tasks.append(Task(key=f"CARE:{species}:{coord[0]},{coord[1]}",
                              kind="CARE", priority=Priority.MANAGER,
                              tile=coord, animal=species, quantity=1,
                              source="manager_allocation"))
    fert_requests = _plan_counts(feasible_plan.fertilizer_by_crop_dict,
                                 CROP_ORDER)
    for crop in CROP_ORDER:
        budget = fert_requests[crop]
        for coord in by_proximity(fert_eligible[crop])[:budget]:
            tasks.append(Task(key=f"FERTILIZE:{crop}:{coord[0]},{coord[1]}",
                              kind="FERTILIZE", priority=Priority.MANAGER,
                              tile=coord, crop=crop,
                              required_item=_FERTILIZER_ITEM, quantity=1,
                              source="manager_allocation"))

    # ---- MANAGER: land -------------------------------------------------
    current_land = len(unlocked)
    if feasible_plan.land_count > current_land:
        if current_land >= len(LAND_ORDER) + 1:
            unresolved.append("land_request_exceeds_mechanical_maximum")
        else:
            next_quadrant = LAND_ORDER[current_land - 1]
            tasks.append(Task(key=f"BUY_LAND:{next_quadrant}", kind="BUY_LAND",
                              priority=Priority.MANAGER, tile=None,
                              product=next_quadrant, quantity=1,
                              source="manager_target"))

    # ---- LOGISTICS: sells in the current bin only -----------------------
    bin_anchor = (hour // 4) * 4
    for product in PRODUCTS:
        remaining = int(remaining_sells.get(product, 0))
        if remaining > 0:
            tasks.append(Task(
                key=f"SELL:{product}:{bin_anchor}", kind="SELL",
                priority=Priority.LOGISTICS, tile=None, product=product,
                quantity=remaining, deadline_hour=bin_anchor + 3,
                source=f"sell_bin_{bin_anchor}"))

    # ---- LOGISTICS: purchases implied by active tasks only --------------
    seed_demand: dict[str, int] = {}
    item_demand: dict[str, int] = {}
    animal_demand: dict[str, int] = {}
    for task in tasks:
        if task.kind == "PLANT":
            seed_demand[task.crop] = seed_demand.get(task.crop, 0) + 1
        elif task.required_item == _FERTILIZER_ITEM:
            item_demand[_FERTILIZER_ITEM] = \
                item_demand.get(_FERTILIZER_ITEM, 0) + 1
        elif task.kind == "FEED":
            item_demand[_FEED_ITEM] = item_demand.get(_FEED_ITEM, 0) + 1
        elif task.kind == "PLACE":
            animal_demand[task.animal] = animal_demand.get(task.animal, 0) + 1
    for crop, demand in sorted(seed_demand.items()):
        shortage = demand - int(state["seeds"].get(crop, 0))
        if shortage > 0:
            tasks.append(Task(key=f"BUY_SEED:{crop}", kind="BUY_SEED",
                              priority=Priority.LOGISTICS, tile=None,
                              crop=crop, quantity=shortage,
                              required_item=crop, source="task_shortage"))
    for item, demand in sorted(item_demand.items()):
        shortage = demand - _available(state, item)
        if shortage > 0:
            tasks.append(Task(key=f"BUY_PRODUCT:{item}", kind="BUY_PRODUCT",
                              priority=Priority.LOGISTICS, tile=None,
                              product=item, quantity=shortage,
                              required_item=item, source="task_shortage"))
    for animal, demand in sorted(animal_demand.items()):
        shortage = demand - _available(state, animal)
        if shortage > 0:
            tasks.append(Task(key=f"BUY_ANIMAL:{animal}", kind="BUY_ANIMAL",
                              priority=Priority.LOGISTICS, tile=None,
                              animal=animal, quantity=shortage,
                              required_item=animal, source="task_shortage"))

    # Unique keys are a contract; duplicates would break dependency semantics.
    keys = [t.key for t in tasks]
    if len(keys) != len(set(keys)):
        raise RuntimeError(f"duplicate task keys generated: {keys}")

    return GenerationResult(tasks=tuple(tasks), unresolved=tuple(unresolved))

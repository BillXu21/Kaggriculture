"""Replay -> canonical daily record extraction for Kaggriculture 1.32.7.

Action alignment (verified against actual downloaded replays, not framework
recollection): `steps[i][seat].action` is the submitted action that transformed
`steps[i-1][seat].observation` into `steps[i][seat].observation`. Every event is
therefore attributed to the day/hour of observation `i-1`. `steps[0]` holds the
default no-op action and never yields events.

Day boundaries come exclusively from explicit observation `day`/`hour` fields:
day start = first observation with `day == d, hour == 0`; end = first
observation with `day == d+1, hour == 0`, else the final terminal observation.
"""

import csv
from typing import Any

from .constants import (
    ANIMALS,
    ENGINE_VERSION,
    FARM_HAND_COST_MULT_DEFAULT,
    LAND_ORDER,
    PRODUCTS,
    SCHEMA_VERSION,
    SELL_BIN_ANCHORS,
    sell_bin,
    total_hire_cost,
)
from .lifecycle import canonical_board


class VersionMismatch(Exception):
    """Raised when a replay does not report module_version == 1.32.7."""


# Keys that must never appear under an opponent subtree.
OPPONENT_PRIVATE_KEYS = ("shed", "seeds", "inventories", "private")


def load_manifest(path: str) -> dict[int, dict[str, Any]]:
    """Load a daily-partition manifest.csv keyed by integer episode_id."""
    rows: dict[int, dict[str, Any]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[int(row["episode_id"])] = row
    return rows


def _tile_kind_for_ledger(tile: Any) -> str:
    if tile is None:
        return "empty"
    if isinstance(tile, str):
        return tile
    if tile.get("kind") == "PLANT":
        return f"PLANT:{tile.get('crop')}"
    if "animal" in tile:
        return f"{tile.get('kind')}:{tile['animal']}"
    return str(tile.get("kind", "unknown"))


def _harvest_item(tile: dict[str, Any]) -> str | None:
    if tile.get("kind") == "PLANT":
        return tile.get("crop")
    if "animal" in tile:
        return ANIMALS[tile["animal"]]["product"]
    return None


def _observed_land_additions(
    pre_obs: dict[str, Any], post_obs: dict[str, Any] | None, seat: int,
) -> list[str]:
    """Return validated quadrants added by this action step, if any.

    Land orders are submitted intents; only the adjacent observation transition
    can establish which next quadrants were actually unlocked. Any non-prefix,
    reordered, or otherwise unexpected transition is treated as unknown.
    """
    if post_obs is None:
        return []
    before = list(pre_obs["farms"][seat].get("unlocked_quadrants") or [])
    after = list(post_obs["farms"][seat].get("unlocked_quadrants") or [])
    if len(after) < len(before) or after[:len(before)] != before:
        return []
    additions = after[len(before):]
    expected = LAND_ORDER[len(before) - 1:len(before) - 1 + len(additions)]
    return additions if additions == expected else []


def empty_events() -> dict[str, Any]:
    return {
        "plants": {},
        "digs": {"total": 0, "replaced": {}},
        "fertilizer_applications": {"by_crop": {}, "entries": []},
        "harvests": {"by_item": {}, "entries": []},
        "buys": {"seeds": {}, "products": {}, "animals": {}},
        "land_purchases": [],
        "hires": {"submitted": 0},
        "sells": [],
        "market_events_ordered": [],
        "worker_ops_other": {},
    }


def _events_from_action(
    action: dict[str, Any], pre_obs: dict[str, Any], post_obs: dict[str, Any] | None,
    seat: int, hour: int,
) -> dict[str, Any]:
    """Compact ledger of one turn's submitted intent, attributed to `pre_obs`."""
    ev = empty_events()
    farm = pre_obs["farms"][seat]
    tiles = farm["tiles"]

    actor_positions: list[list[int]] = [farm["farmer"]]
    actor_positions.extend(farm.get("hands") or [])

    # Schema: "farmer": [op, *args], "hands": [[op, *args], ...] — exactly one
    # op per actor per turn.
    worker_ops: list[list[Any]] = [action.get("farmer") or []]
    worker_ops.extend(action.get("hands") or [])

    for pos, op in zip(actor_positions, worker_ops):
        if not pos or len(pos) < 2:
            continue
        y, x = int(pos[0]), int(pos[1])
        if not (0 <= y < len(tiles) and 0 <= x < len(tiles[y])):
            continue
        if not isinstance(op, list):
            op = [op]
        if not op:
            continue
        name = op[0]
        args = op[1:]
        if name in (None, "PASS"):
            continue
        if name == "PLANT" and args:
            crop = args[0]
            ev["plants"][crop] = ev["plants"].get(crop, 0) + 1
        elif name == "DIG":
            replaced = _tile_kind_for_ledger(tiles[y][x])
            ev["digs"]["total"] += 1
            ev["digs"]["replaced"][replaced] = ev["digs"]["replaced"].get(replaced, 0) + 1
        elif name == "FERTILIZE":
            tile = tiles[y][x]
            crop = tile.get("crop") if isinstance(tile, dict) else None
            key = crop if crop else "unknown"
            ev["fertilizer_applications"]["by_crop"][key] = \
                ev["fertilizer_applications"]["by_crop"].get(key, 0) + 1
            ev["fertilizer_applications"]["entries"].append(
                {"tile": [y, x], "crop": crop, "hour": hour})
        elif name == "HARVEST":
            tile = tiles[y][x]
            item = _harvest_item(tile) if isinstance(tile, dict) else None
            if item:
                ev["harvests"]["by_item"][item] = ev["harvests"]["by_item"].get(item, 0) + 1
                ev["harvests"]["entries"].append({"tile": [y, x], "item": item, "hour": hour})
        else:
            ev["worker_ops_other"][name] = ev["worker_ops_other"].get(name, 0) + 1

    observed_land_additions = _observed_land_additions(pre_obs, post_obs, seat)
    observed_land_index = 0
    for order in action.get("market") or []:
        if not isinstance(order, list) or not order:
            continue
        op = order[0]
        # Ordered audit trail with exact attribution hour: [op, *args, hour].
        ev["market_events_ordered"].append([*order, hour])
        if op == "BUY_SEED" and len(order) > 2:
            ev["buys"]["seeds"][order[1]] = ev["buys"]["seeds"].get(order[1], 0) + order[2]
        elif op == "BUY_PRODUCT" and len(order) > 2:
            ev["buys"]["products"][order[1]] = \
                ev["buys"]["products"].get(order[1], 0) + order[2]
        elif op == "BUY_ANIMAL" and len(order) > 2:
            ev["buys"]["animals"][order[1]] = \
                ev["buys"]["animals"].get(order[1], 0) + order[2]
        elif op == "SELL" and len(order) > 2:
            ev["sells"].append({"product": order[1], "quantity": order[2], "hour": hour})
        elif op == "HIRE":
            ev["hires"]["submitted"] += 1
        elif op == "BUY_LAND":
            quadrant = (
                observed_land_additions[observed_land_index]
                if observed_land_index < len(observed_land_additions) else None
            )
            if quadrant is not None:
                observed_land_index += 1
            ev["land_purchases"].append({"quadrant": quadrant, "hour": hour})
    return ev


def _merge_events(target: dict[str, Any], extra: dict[str, Any]) -> None:
    for crop, n in extra["plants"].items():
        target["plants"][crop] = target["plants"].get(crop, 0) + n
    target["digs"]["total"] += extra["digs"]["total"]
    for k, n in extra["digs"]["replaced"].items():
        target["digs"]["replaced"][k] = target["digs"]["replaced"].get(k, 0) + n
    for k, n in extra["fertilizer_applications"]["by_crop"].items():
        target["fertilizer_applications"]["by_crop"][k] = \
            target["fertilizer_applications"]["by_crop"].get(k, 0) + n
    target["fertilizer_applications"]["entries"].extend(extra["fertilizer_applications"]["entries"])
    for k, n in extra["harvests"]["by_item"].items():
        target["harvests"]["by_item"][k] = target["harvests"]["by_item"].get(k, 0) + n
    target["harvests"]["entries"].extend(extra["harvests"]["entries"])
    for group in ("seeds", "products", "animals"):
        for k, n in extra["buys"][group].items():
            target["buys"][group][k] = target["buys"][group].get(k, 0) + n
    target["land_purchases"].extend(extra["land_purchases"])
    target["hires"]["submitted"] += extra["hires"]["submitted"]
    target["sells"].extend(extra["sells"])
    target["market_events_ordered"].extend(extra["market_events_ordered"])
    for k, n in extra["worker_ops_other"].items():
        target["worker_ops_other"][k] = target["worker_ops_other"].get(k, 0) + n


def _canonical_town(town: dict[str, Any]) -> dict[str, Any]:
    shops = list((town or {}).get("unlocked_shops") or [])
    counts: dict[str, int] = {}
    for s in shops:
        counts[s] = counts.get(s, 0) + 1
    return {"unlocked_shops": shops, "shop_counts": counts}


def _self_state(obs: dict[str, Any], seat: int, current_day: int, current_step: int) -> dict[str, Any]:
    farm = obs["farms"][seat]
    private = obs.get("private") or {}
    return {
        "money": farm["money"],
        "board": canonical_board(farm["tiles"], current_day, current_step),
        "unlocked_quadrants": list(farm["unlocked_quadrants"]),
        "farmer": list(farm["farmer"]),
        "hands": [list(h) for h in (farm.get("hands") or [])],
        "hires_today": farm["hires_today"],
        "shed": dict(private.get("shed") or {}),
        "seeds": dict(private.get("seeds") or {}),
        "inventories": [dict(inv) for inv in (private.get("inventories") or [])],
    }


def _opponent_public_state(
    obs: dict[str, Any], seat: int, current_day: int, current_step: int,
) -> dict[str, Any]:
    other = 1 - seat
    farm = obs["farms"][other]
    # Public farm data only; opponent shed/seeds/inventories are never read here.
    return {
        "money": farm["money"],
        "board": canonical_board(farm["tiles"], current_day, current_step),
        "unlocked_quadrants": list(farm["unlocked_quadrants"]),
        "farmer": list(farm["farmer"]),
        "hands": [list(h) for h in (farm.get("hands") or [])],
        "hires_today": farm["hires_today"],
    }


def _shared_state(obs: dict[str, Any]) -> dict[str, Any]:
    market = obs.get("market") or {}
    return {
        "market": {
            "inventory": dict(market.get("inventory") or {}),
            "prices": dict(market.get("prices") or {}),
        },
        "town": _canonical_town(obs.get("town") or {}),
    }


def _crop_composition(board: list[list[Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in board:
        for tile in row:
            if isinstance(tile, dict) and tile.get("kind") == "PLANT":
                counts[tile["crop"]] = counts.get(tile["crop"], 0) + 1
    return counts


def _animal_counts(board: list[list[Any]]) -> dict[str, int]:
    counts = {name: 0 for name in ("GOOSE", "COW", "SHEEP")}
    for row in board:
        for tile in row:
            if isinstance(tile, dict) and "animal" in tile:
                counts[tile["animal"]] = counts.get(tile["animal"], 0) + 1
    return counts


def _sell_quantity_bins(sells: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    bins = {str(anchor): {p: 0 for p in PRODUCTS} for anchor in SELL_BIN_ANCHORS}
    for sale in sells:
        key = str(sell_bin(int(sale["hour"])))
        bins[key][sale["product"]] += int(sale["quantity"])
    return bins


def extract_replay(
    replay: dict[str, Any],
    *,
    source_dataset: str | None = None,
    partition_date: str | None = None,
    source_path: str | None = None,
    manifest_row: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Extract one canonical record per (episode, seat, day). Parses input once."""
    module_version = replay.get("module_version")
    if module_version != ENGINE_VERSION:
        raise VersionMismatch(
            f"expected module_version {ENGINE_VERSION!r}, got {module_version!r}"
        )

    steps = replay["steps"]
    config = replay.get("configuration") or {}
    hire_mult = int(config.get("farmHandCostMult", FARM_HAND_COST_MULT_DEFAULT))
    info = replay.get("info") or {}
    episode_id = info.get("EpisodeId")
    team_names = info.get("TeamNames")
    if not team_names:
        team_names = [a.get("Name") for a in (info.get("Agents") or [])]
    seed = info.get("seed")
    rewards = replay.get("rewards")

    def _score(name: str) -> float | None:
        if manifest_row is None:
            return None
        raw = manifest_row.get(name)
        return float(raw) if raw not in (None, "") else None

    avg_score = _score("avg_score")
    min_score = _score("min_score")
    max_score = _score("max_score")
    sum_score = _score("sum_score")
    if source_dataset is None and manifest_row is not None:
        source_dataset = manifest_row.get("source_dataset")
    if partition_date is None and manifest_row is not None:
        partition_date = manifest_row.get("partition_date") or manifest_row.get("date")

    records: list[dict[str, Any]] = []
    n_seats = len(steps[0])
    last_index = len(steps) - 1

    for seat in range(n_seats):
        observations = [steps[i][seat].get("observation") for i in range(len(steps))]

        # Explicit day/hour scan only; the outer step index is never used modulo.
        first_start_by_day: dict[int, int] = {}
        last_obs_by_day: dict[int, int] = {}
        for i, o in enumerate(observations):
            d, h = o["day"], o["hour"]
            if h == 0 and d not in first_start_by_day:
                first_start_by_day[d] = i
            last_obs_by_day[d] = i
        days = sorted(first_start_by_day)

        for d in days:
            start_i = first_start_by_day[d]
            if (d + 1) in first_start_by_day:
                end_i = first_start_by_day[d + 1]
                boundary = "next_day_start"
            else:
                end_i = last_index
                boundary = "terminal"

            start_obs = observations[start_i]
            end_obs = observations[end_i]

            # Previous-day realized hires are read from the observed hires_today
            # sequence, not from submitted HIRE intents. The state resets at the
            # next morning refresh; cost follows the exact Fibonacci sequence.
            previous_execution = {"workers_hired": 0, "hire_cost": 0}
            prev_day = d - 1
            if prev_day in first_start_by_day:
                previous_execution = previous_execution_of_day(
                    observations, first_start_by_day, last_obs_by_day,
                    seat, prev_day, hire_mult,
                )

            events = empty_events()
            # Actions at steps (start_i, end_i] act on observations inside day d;
            # the initial default action at steps[0] is never an event.
            for i in range(max(start_i + 1, 1), end_i + 1):
                pre_obs = observations[i - 1]
                if pre_obs["day"] != d:
                    continue
                action = steps[i][seat].get("action")
                if not action:
                    continue
                _merge_events(
                    events,
                    _events_from_action(
                        action, pre_obs, observations[i], seat, pre_obs["hour"]
                    ),
                )
            events["hires"]["realized"] = previous_execution_of_day(
                observations, first_start_by_day, last_obs_by_day, seat, d, hire_mult,
            )

            # Seat 1 observations in compiled replays omit `step`; fall back to
            # the outer index only for lifecycle decay timing. Day boundaries
            # remain exclusively based on explicit day/hour fields.
            start_step = int(start_obs.get("step", start_i))
            end_step = int(end_obs.get("step", end_i))
            start_self = _self_state(start_obs, seat, start_obs["day"], start_step)
            end_self = _self_state(end_obs, seat, end_obs["day"], end_step)

            targets = {
                "crop_composition_end": _crop_composition(end_self["board"]),
                "animal_counts_end": _animal_counts(end_self["board"]),
                "unlocked_quadrants_end": list(end_self["unlocked_quadrants"]),
                "land_expansion": {
                    "expanded": end_self["unlocked_quadrants"] != start_self["unlocked_quadrants"],
                    "new_quadrants": [
                        q for q in end_self["unlocked_quadrants"]
                        if q not in start_self["unlocked_quadrants"]
                    ],
                },
                "fertilizer_by_crop": dict(events["fertilizer_applications"]["by_crop"]),
                "sell_quantity": _sell_quantity_bins(events["sells"]),
            }

            metadata = {
                "episode_id": episode_id,
                "source_dataset": source_dataset,
                "partition_date": partition_date,
                "source_path": source_path,
                "seat": seat,
                "player": team_names[seat] if team_names and len(team_names) > seat else None,
                "opponent": team_names[1 - seat] if team_names and len(team_names) > 1 - seat else None,
                "seed": seed,
                "module_version": module_version,
                "avg_score": avg_score,
                "min_score": min_score,
                "max_score": max_score,
                "sum_score": sum_score,
                "final_rewards": list(rewards) if rewards is not None else None,
                "final_bank_self": rewards[seat] if rewards and len(rewards) > seat else None,
                "final_bank_opponent": rewards[1 - seat] if rewards and len(rewards) > 1 - seat else None,
            }

            records.append({
                "schema_version": SCHEMA_VERSION,
                "metadata": metadata,
                "day": d,
                "start": {
                    "day": start_obs["day"],
                    "hour": start_obs["hour"],
                    "self": start_self,
                    "opponent_public": _opponent_public_state(
                        start_obs, seat, start_obs["day"], start_step),
                    **_shared_state(start_obs),
                    "previous_execution": previous_execution,
                },
                "events": events,
                "targets": targets,
                "end": {
                    "boundary": boundary,
                    "day": end_obs["day"],
                    "hour": end_obs["hour"],
                    "self": end_self,
                    "opponent_public": _opponent_public_state(
                        end_obs, seat, end_obs["day"], end_step),
                    **_shared_state(end_obs),
                },
            })
    return records


def previous_execution_of_day(
    observations: list[dict[str, Any]],
    first_start_by_day: dict[int, int],
    last_obs_by_day: dict[int, int],
    seat: int,
    day: int,
    hire_mult: int,
) -> dict[str, int]:
    """Realized hires/cost for `day` itself (used as next day's feedback mirror).

    `hires_today` is the authoritative successful-hire counter. Summing the
    cost for each observed increase also handles sparse or irregular replay
    observations without treating submitted HIRE orders as successful.
    """
    if day not in first_start_by_day:
        return {"workers_hired": 0, "hire_cost": 0}
    lo = first_start_by_day[day]
    hi = last_obs_by_day[day]
    hired = 0
    cost = 0
    for i in range(lo, hi + 1):
        observed = int(observations[i]["farms"][seat].get("hires_today", 0))
        if observed <= hired:
            continue
        cost += total_hire_cost(observed, hire_mult) - total_hire_cost(hired, hire_mult)
        hired = observed
    return {"workers_hired": hired, "hire_cost": cost}

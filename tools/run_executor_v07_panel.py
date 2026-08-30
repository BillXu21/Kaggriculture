"""Deterministic local full-game evaluator for the V0.7 executor stack.

The production path is deliberately small: one fresh checkpoint-backed
``ExecutorAgent`` is wrapped by one fresh built-in opening agent for each
seed/seat pair, while the other seat submits only PASS actions.  The module
records canonical post-transition diagnostics, never observations, and writes
one content-addressed JSON document.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import math
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
import subprocess
import sys
from typing import Any

from executor_v0.agent import AgentConfig, make_agent
from opening_book.agent import make_opening_agent
from opening_book.trace import canonical_json_bytes, validate_action
from oracle.backend import make_backend
from oracle.closed_loop import _executor_observation
from rl_manager.provenance import backend_provenance, opening_provenance

__all__ = [
    "EvaluatorError",
    "run_game",
    "run_panel",
    "main",
]

SCHEMA_VERSION = 2
ARTIFACT_TYPE = "executor_v07_full_game_evaluator"
DEFAULT_OPENING = "standard_mixed"
DEFAULT_OPPONENT = "PASS"
DEFAULT_MAX_TRANSITIONS = 719
EXPECTED_ENGINE_STATUSES = frozenset({"ACTIVE", "DONE"})
ANIMAL_NAMES = frozenset({"GOOSE", "COW", "SHEEP"})

ProviderFactory = Callable[[Path], Any]
BackendFactory = Callable[[str, Mapping[str, Any]], Any]


class EvaluatorError(RuntimeError):
    """Raised when a requested evaluation cannot produce a valid artifact."""


def _require_int(value: Any, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise EvaluatorError(f"{name} must be an integer, got {value!r}")
    if minimum is not None and value < minimum:
        raise EvaluatorError(f"{name} must be >= {minimum}, got {value!r}")
    return value


def _require_seat(seat: Any) -> int:
    seat = _require_int(seat, "seat")
    if seat not in (0, 1):
        raise EvaluatorError(f"seat must be 0 or 1, got {seat!r}")
    return seat


def _require_seed(seed: Any) -> int:
    return _require_int(seed, "seed", minimum=0)


def _require_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvaluatorError(f"{name} must be a non-empty string")
    return value


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_metadata(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    from bc_manager.training import checkpoint_model_variant, load_checkpoint

    payload = load_checkpoint(path)
    variant = checkpoint_model_variant(payload)
    if variant != "E":
        raise EvaluatorError(
            f"checkpoint variant mismatch: expected 'E', got {variant!r}: {path}"
        )
    epoch = payload.get("epoch")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise EvaluatorError(f"checkpoint epoch is missing or invalid: {epoch!r}")
    return {
        "path": str(path),
        "sha256": _checkpoint_sha256(path),
        "variant": variant,
        "epoch": epoch,
    }


def _source_repo_sha(repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise EvaluatorError(f"unable to determine source repo SHA in {repo_root}") from exc
    sha = result.stdout.strip()
    if len(sha) != 40 or any(char not in "0123456789abcdef" for char in sha.lower()):
        raise EvaluatorError(f"git rev-parse returned an invalid source SHA: {sha!r}")
    return sha


def _ensure_finite(value: Any, path: str = "artifact") -> Any:
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise EvaluatorError(f"non-finite JSON value at {path}: {value!r}")
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _ensure_finite(item, f"{path}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _ensure_finite(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise EvaluatorError(f"non-JSON value at {path}: {type(value).__name__}")


def _canonical_json(value: Any) -> bytes:
    return canonical_json_bytes(_ensure_finite(value))


def _write_artifact(document: Mapping[str, Any], output_path: Path, *, overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"output already exists: {output_path}; pass overwrite=True or --overwrite"
        )
    payload = _canonical_json(document) + b"\n"
    output_path.write_bytes(payload)


def _pass_action(observation: Mapping[str, Any], seat: int) -> dict[str, Any]:
    farms = observation.get("farms")
    if not isinstance(farms, list) or len(farms) <= seat or not isinstance(farms[seat], Mapping):
        raise EvaluatorError(f"observation has no farm for PASS seat {seat}")
    hands = farms[seat].get("hands") or []
    if not isinstance(hands, list):
        raise EvaluatorError(f"observation hands for PASS seat {seat} are not a list")
    return {"farmer": ["PASS"], "hands": [["PASS"] for _ in hands], "market": []}


def _animal_snapshot(state: Mapping[str, Any], seat: int) -> dict[str, Any]:
    farms = state.get("farms")
    privates = state.get("privates")
    if not isinstance(farms, list) or len(farms) <= seat:
        raise EvaluatorError("canonical state has no tested-seat farm")
    farm = farms[seat]
    if not isinstance(farm, Mapping):
        raise EvaluatorError("canonical tested-seat farm is not a mapping")

    board_animals: list[dict[str, Any]] = []
    for y, row in enumerate(farm.get("tiles") or []):
        if not isinstance(row, list):
            continue
        for x, tile in enumerate(row):
            if not isinstance(tile, Mapping) or tile.get("animal") not in ANIMAL_NAMES:
                continue
            board_animals.append({
                "coord": [y, x],
                "kind": tile.get("kind"),
                "animal": tile.get("animal"),
                "fed_today": tile.get("fed_today"),
                "consecutive_unfed": tile.get("consecutive_unfed"),
            })
    board_animals.sort(key=lambda item: (item["coord"][0], item["coord"][1], item["animal"]))

    private = privates[seat] if isinstance(privates, list) and len(privates) > seat else {}
    private = private if isinstance(private, Mapping) else {}
    shed = private.get("shed") if isinstance(private.get("shed"), Mapping) else {}
    inventories = private.get("inventories") if isinstance(private.get("inventories"), list) else []
    carried_animals = {
        animal: sum(int(inv.get(animal, 0) or 0) for inv in inventories if isinstance(inv, Mapping))
        for animal in sorted(ANIMAL_NAMES)
    }
    board_total = len(board_animals)
    shed_total = sum(int(shed.get(animal, 0) or 0) for animal in ANIMAL_NAMES)
    carried_total = sum(carried_animals.values())
    carried_wheat = sum(int(inv.get("WHEAT", 0) or 0) for inv in inventories if isinstance(inv, Mapping))
    shed_wheat = int(shed.get("WHEAT", 0) or 0)
    market = state.get("market") if isinstance(state.get("market"), Mapping) else {}
    inventory = market.get("inventory") if isinstance(market.get("inventory"), Mapping) else {}
    prices = market.get("prices") if isinstance(market.get("prices"), Mapping) else {}
    return {
        "money": farm.get("money"),
        "total_animals": board_total + shed_total + carried_total,
        "board_animals": board_animals,
        "shed_animals": {animal: int(shed.get(animal, 0) or 0) for animal in sorted(ANIMAL_NAMES)},
        "carried_animals": carried_animals,
        "shed_wheat": shed_wheat,
        "carried_wheat": carried_wheat,
        "available_wheat": shed_wheat + carried_wheat,
        "market_wheat_inventory": int(inventory.get("WHEAT", 0) or 0),
        "market_wheat_price": int(prices.get("WHEAT", 0) or 0),
    }


def _farm_evaluation_snapshot(state: Mapping[str, Any], seat: int) -> dict[str, Any]:
    """Extract bounded own-farm outcome counters without retaining states."""
    farms = state.get("farms") or []
    farm = farms[seat] if len(farms) > seat and isinstance(farms[seat], Mapping) else {}
    animal_counts = {animal: 0 for animal in sorted(ANIMAL_NAMES)}
    crop_counts: dict[str, int] = {}
    weed_count = 0
    for row in farm.get("tiles") or []:
        for tile in row if isinstance(row, list) else ():
            if isinstance(tile, Mapping) and tile.get("animal") in animal_counts:
                animal_counts[tile["animal"]] += 1
            elif isinstance(tile, Mapping) and tile.get("kind") == "PLANT":
                crop = str(tile.get("crop"))
                crop_counts[crop] = crop_counts.get(crop, 0) + 1
            elif tile == "WEED" or (
                    isinstance(tile, Mapping) and tile.get("kind") == "WEED"):
                weed_count += 1
    privates = state.get("privates") or []
    private = privates[seat] if len(privates) > seat and isinstance(privates[seat], Mapping) else {}
    shed = private.get("shed") if isinstance(private.get("shed"), Mapping) else {}
    inventories = private.get("inventories") if isinstance(private.get("inventories"), list) else []
    inventory_animals = {
        animal: int(shed.get(animal, 0) or 0) + sum(
            int(inv.get(animal, 0) or 0) for inv in inventories
            if isinstance(inv, Mapping)
        )
        for animal in sorted(ANIMAL_NAMES)
    }
    return {
        "board_animals": animal_counts,
        "board_crops": crop_counts,
        "weed_tiles": weed_count,
        "inventory_animals": inventory_animals,
    }


def _trace_for_turn(diagnostics: Any, day: int, hour: int) -> dict[str, Any] | None:
    if not isinstance(diagnostics, Mapping):
        return None
    days = diagnostics.get("days")
    if not isinstance(days, Mapping):
        return None
    record = days.get(str(day))
    if not isinstance(record, Mapping):
        return None
    trace = record.get("turn_trace")
    if not isinstance(trace, list):
        return None
    for entry in trace:
        if isinstance(entry, Mapping) and int(entry.get("hour", -1)) == hour:
            return copy.deepcopy(dict(entry))
    return None


def _wheat_animal_orders(action: Mapping[str, Any]) -> list[Any]:
    orders = action.get("market")
    if not isinstance(orders, list):
        return []
    return [
        copy.deepcopy(order)
        for order in orders
        if isinstance(order, list)
        and len(order) > 1
        and order[1] in ("WHEAT", "GOOSE", "COW", "SHEEP")
    ]


def _submitted_wheat_quantities(action: Mapping[str, Any]) -> tuple[int, int]:
    """Return submitted WHEAT BUY_PRODUCT and SELL quantities."""
    buy = 0
    sell = 0
    orders = action.get("market")
    if not isinstance(orders, list):
        return buy, sell
    for order in orders:
        if not isinstance(order, list) or len(order) < 3:
            continue
        if order[1] != "WHEAT":
            continue
        quantity = order[2]
        if isinstance(quantity, bool) or not isinstance(quantity, int):
            continue
        if order[0] == "BUY_PRODUCT":
            buy += quantity
        elif order[0] == "SELL":
            sell += quantity
    return buy, sell


def _wheat_fill_ledger(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    submitted_buy: int,
    submitted_sell: int,
) -> dict[str, Any]:
    """Infer fills only when the fixed PASS market delta identifies them."""
    market_delta = after["market_wheat_inventory"] - before["market_wheat_inventory"]
    if submitted_buy and submitted_sell:
        return {
            "inferred_wheat_buy_fill_quantity": None,
            "inferred_wheat_sell_fill_quantity": None,
            "wheat_fill_attribution": None,
            "wheat_fill_reason": "buy_and_sell_net_market_delta_is_ambiguous",
        }

    if submitted_buy:
        fill = -market_delta
        if 0 <= fill <= submitted_buy:
            return {
                "inferred_wheat_buy_fill_quantity": fill,
                "inferred_wheat_sell_fill_quantity": 0,
                "wheat_fill_attribution": "exact_fixed_pass_market_delta",
                "wheat_fill_reason": None,
            }
        reason = "market_delta_not_consistent_with_submitted_buy"
    elif submitted_sell:
        fill = market_delta
        if 0 <= fill <= submitted_sell:
            return {
                "inferred_wheat_buy_fill_quantity": 0,
                "inferred_wheat_sell_fill_quantity": fill,
                "wheat_fill_attribution": "exact_fixed_pass_market_delta",
                "wheat_fill_reason": None,
            }
        reason = "market_delta_not_consistent_with_submitted_sell"
    elif market_delta == 0:
        return {
            "inferred_wheat_buy_fill_quantity": 0,
            "inferred_wheat_sell_fill_quantity": 0,
            "wheat_fill_attribution": "exact_fixed_pass_market_delta",
            "wheat_fill_reason": None,
        }
    else:
        reason = "market_changed_without_submitted_wheat_order"

    return {
        "inferred_wheat_buy_fill_quantity": None,
        "inferred_wheat_sell_fill_quantity": None,
        "wheat_fill_attribution": None,
        "wheat_fill_reason": reason,
    }


def _timeline_entry(
    before_state: Mapping[str, Any],
    after_state: Mapping[str, Any],
    action: Mapping[str, Any],
    seat: int,
    transition: int,
    turn_trace: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    before = _animal_snapshot(before_state, seat)
    after = _animal_snapshot(after_state, seat)
    submitted_buy, submitted_sell = _submitted_wheat_quantities(action)
    wheat_fill = _wheat_fill_ledger(before, after, submitted_buy, submitted_sell)
    day = int(before_state.get("day", 0))
    hour = int(before_state.get("hour", 0))
    entry: dict[str, Any] = {
        "transition": transition,
        "step": int(after_state.get("step", transition)),
        "day": day,
        "hour": hour,
        "money_before": before["money"],
        "money_after": after["money"],
        "money_delta": after["money"] - before["money"],
        "animal_total_before": before["total_animals"],
        "animal_total_after": after["total_animals"],
        "board_animals": after["board_animals"],
        "shed_wheat_before": before["shed_wheat"],
        "shed_wheat": after["shed_wheat"],
        "shed_wheat_after": after["shed_wheat"],
        "shed_wheat_delta": after["shed_wheat"] - before["shed_wheat"],
        "carried_wheat_before": before["carried_wheat"],
        "carried_wheat": after["carried_wheat"],
        "carried_wheat_after": after["carried_wheat"],
        "carried_wheat_delta": after["carried_wheat"] - before["carried_wheat"],
        "available_wheat_before": before["available_wheat"],
        "available_wheat": after["available_wheat"],
        "available_wheat_after": after["available_wheat"],
        "available_wheat_delta": after["available_wheat"] - before["available_wheat"],
        "market_wheat_inventory_before": before["market_wheat_inventory"],
        "market_wheat_inventory": after["market_wheat_inventory"],
        "market_wheat_inventory_after": after["market_wheat_inventory"],
        "market_wheat_inventory_delta": (
            after["market_wheat_inventory"] - before["market_wheat_inventory"]
        ),
        "market_wheat_price": after["market_wheat_price"],
        "submitted_wheat_buy_quantity": submitted_buy,
        "submitted_wheat_sell_quantity": submitted_sell,
        **wheat_fill,
        "submitted_tested_seat_market_orders": _wheat_animal_orders(action),
        "turn_trace": turn_trace,
        "feed_shortage": (
            bool((turn_trace.get("feed") or {}).get("shortage"))
            if turn_trace is not None else None
        ),
        "starvation": (
            bool((turn_trace.get("feed") or {}).get("starving"))
            if turn_trace is not None else None
        ),
        "starvation_preemption": (
            bool((turn_trace.get("feed") or {}).get("starving"))
            if turn_trace is not None else None
        ),
        "expansion_suppression_reasons": (
            list((turn_trace.get("expansion") or {}).get("reasons") or [])
            if turn_trace is not None else None
        ),
    }
    event = None
    if after["total_animals"] < before["total_animals"]:
        event = {
            "transition": transition,
            "step": entry["step"],
            "day": day,
            "hour": hour,
            "before": before["total_animals"],
            "after": after["total_animals"],
            "decrease": before["total_animals"] - after["total_animals"],
            "action": copy.deepcopy(dict(action)),
            "nearby": {
                "before_board_animals": before["board_animals"],
                "after_board_animals": after["board_animals"],
                "before_available_wheat": before["available_wheat"],
                "after_available_wheat": after["available_wheat"],
                "market_wheat_inventory": after["market_wheat_inventory"],
                "market_wheat_price": after["market_wheat_price"],
                "turn_trace": turn_trace,
            },
        }
        entry["animal_total_decrease"] = {
            "before": before["total_animals"],
            "after": after["total_animals"],
            "decrease": before["total_animals"] - after["total_animals"],
        }
    return entry, event


def _daily_summary(timeline: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for entry in timeline:
        grouped[int(entry["day"])].append(entry)
    result = []
    for day, entries in sorted(grouped.items()):
        drops = [entry for entry in entries if "animal_total_decrease" in entry]
        traces = [entry.get("turn_trace") for entry in entries if isinstance(entry.get("turn_trace"), Mapping)]
        feed_shortage = sum(1 for trace in traces if (trace.get("feed") or {}).get("shortage"))
        starving = sum(1 for trace in traces if (trace.get("feed") or {}).get("starving"))
        suppressed = sum(
            1 for trace in traces
            if (trace.get("expansion") or {}).get("suppressed_current")
        )
        result.append({
            "day": day,
            "transitions": len(entries),
            "start_animal_total": entries[0]["animal_total_before"],
            "end_animal_total": entries[-1]["animal_total_after"],
            "animal_decrease_events": len(drops),
            "feed_shortage_turns": feed_shortage,
            "starvation_turns": starving,
            "expansion_suppressed_turns": suppressed,
        })
    return result


def _action_trace_digest(trace: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(_canonical_json(list(trace))).hexdigest()


def _build_downstream(
    checkpoint_path: Path,
    seat: int,
    config: AgentConfig,
    provider_factory: ProviderFactory | None,
) -> tuple[Any, str]:
    if provider_factory is None:
        downstream = make_agent(checkpoint=str(checkpoint_path), seat=seat, config=config)
    else:
        provider = provider_factory(checkpoint_path)
        downstream = make_agent(provider=provider, seat=seat, config=config)
    provider = getattr(downstream, "provider", None)
    variant = getattr(provider, "model_variant", None)
    if provider_factory is None and variant != "E":
        raise EvaluatorError(f"real checkpoint provider variant mismatch: {variant!r}")
    return downstream, str(variant or "injected")


def _run_one(
    checkpoint_path: Path,
    checkpoint_info: Mapping[str, Any],
    seed: int,
    seat: int,
    *,
    backend_name: str,
    opening: str,
    opponent: str,
    prior_debt_suppression: bool,
    turn_trace: bool,
    aggressive_sell_all: bool,
    immediate_plant_water: bool,
    optional_idle_cleanup: bool,
    optional_spare_watering: bool,
    max_transitions: int,
    backend_factory: BackendFactory,
    provider_factory: ProviderFactory | None,
) -> dict[str, Any]:
    configuration = {"seed": seed}
    backend = backend_factory(backend_name, dict(configuration))
    if not hasattr(backend, "reset") or not hasattr(backend, "step"):
        raise EvaluatorError("backend factory did not return the canonical backend protocol")
    config = AgentConfig(
        strict=True,
        turn_trace=turn_trace,
        suppress_expansion_from_prior_debt=prior_debt_suppression,
        aggressive_sell_all=aggressive_sell_all,
        immediate_plant_water=immediate_plant_water,
        optional_idle_cleanup=optional_idle_cleanup,
        optional_spare_watering=optional_spare_watering,
    )
    cleanup_mode = config.cleanup_mode
    downstream, provider_variant = _build_downstream(
        checkpoint_path, seat, config, provider_factory
    )
    opening_agent = make_opening_agent(opening=opening, downstream=downstream, seat=seat)

    observations = backend.reset()
    if not isinstance(observations, list) or len(observations) != 2:
        raise EvaluatorError("backend reset must return two observations")
    statuses = list(getattr(backend, "statuses", []))
    if len(statuses) != 2 or any(status not in EXPECTED_ENGINE_STATUSES for status in statuses):
        raise EvaluatorError(f"invalid backend statuses after reset: {statuses!r}")

    timeline: list[dict[str, Any]] = []
    animal_events: list[dict[str, Any]] = []
    tested_trace: list[dict[str, Any]] = []
    opponent_trace: list[dict[str, Any]] = []
    purchased_animals = {animal: 0 for animal in sorted(ANIMAL_NAMES)}
    placed_animals = {animal: 0 for animal in sorted(ANIMAL_NAMES)}
    weed_tiles_created = 0
    action_counts: dict[str, int] = {}
    transitions = 0
    terminal = False

    for transition in range(1, max_transitions + 1):
        before_state = copy.deepcopy(backend.canonical_state())
        current_step = int(before_state.get("step", transition - 1))
        tested_observation = _executor_observation(
            copy.deepcopy(observations[seat]), from_fast=backend_name == "fast"
        )
        opponent_observation = _executor_observation(
            copy.deepcopy(observations[1 - seat]), from_fast=backend_name == "fast"
        )
        tested_action = opening_agent(tested_observation)
        if not isinstance(tested_action, Mapping):
            raise EvaluatorError(f"executor returned {type(tested_action).__name__}")
        tested_action = copy.deepcopy(dict(tested_action))
        validate_action(tested_action, label=f"tested action transition {transition}")
        for order in tested_action.get("market") or []:
            if len(order) >= 3 and order[0] == "BUY_ANIMAL" \
                    and order[1] in purchased_animals:
                purchased_animals[order[1]] += int(order[2])
        for action in [tested_action.get("farmer", []),
                       *(tested_action.get("hands") or [])]:
            if action:
                action_counts[action[0]] = action_counts.get(action[0], 0) + 1
        opponent_action = _pass_action(opponent_observation, 1 - seat)
        validate_action(opponent_action, label=f"PASS action transition {transition}")
        pair: list[Mapping[str, Any]] = [dict(tested_action), dict(opponent_action)]
        if seat == 1:
            pair = [dict(opponent_action), dict(tested_action)]

        diagnostic = getattr(downstream, "diagnostics_json", None)
        trace_entry = _trace_for_turn(diagnostic() if callable(diagnostic) else None,
                                      int(before_state.get("day", 0)),
                                      int(before_state.get("hour", 0))) if turn_trace else None
        tested_trace.append({
            "transition": transition,
            "step": current_step,
            "day": int(before_state.get("day", 0)),
            "hour": int(before_state.get("hour", 0)),
            "action": copy.deepcopy(tested_action),
        })
        opponent_trace.append({
            "transition": transition,
            "step": current_step,
            "day": int(before_state.get("day", 0)),
            "hour": int(before_state.get("hour", 0)),
            "action": copy.deepcopy(opponent_action),
        })

        observations, _, statuses = backend.step(copy.deepcopy(pair))
        statuses = list(statuses)
        if len(statuses) != 2 or any(status not in EXPECTED_ENGINE_STATUSES for status in statuses):
            raise EvaluatorError(f"invalid backend statuses at transition {transition}: {statuses!r}")
        validate_history = getattr(backend, "validate_status_history", None)
        if callable(validate_history):
            validate_history()
        after_state = copy.deepcopy(backend.canonical_state())
        before_farm = _farm_evaluation_snapshot(before_state, seat)
        after_farm = _farm_evaluation_snapshot(after_state, seat)
        for animal in placed_animals:
            placed_animals[animal] += max(
                0, after_farm["board_animals"].get(animal, 0)
                - before_farm["board_animals"].get(animal, 0)
            )
        weed_tiles_created += max(
            0, after_farm["weed_tiles"] - before_farm["weed_tiles"])
        entry, event = _timeline_entry(
            before_state, after_state, tested_action, seat, transition, trace_entry
        )
        timeline.append(entry)
        if event is not None:
            animal_events.append(event)
        transitions = transition
        terminal = statuses == ["DONE", "DONE"]
        if terminal:
            break

    final_state = copy.deepcopy(backend.canonical_state())
    if max_transitions == DEFAULT_MAX_TRANSITIONS and not terminal:
        raise EvaluatorError(
            f"incomplete episode: reached {transitions} transitions with statuses {statuses!r}"
        )
    if terminal and max_transitions == DEFAULT_MAX_TRANSITIONS and int(final_state.get("step", -1)) != 719:
        raise EvaluatorError(
            f"episode terminated at step {final_state.get('step')!r}, expected full step 719"
        )

    opening_diagnostics = opening_agent.diagnostics_json()
    if opening_diagnostics.get("divergence", {}).get("occurred") or opening_diagnostics.get("fallback_active"):
        raise EvaluatorError(f"opening divergence/fallback: {opening_diagnostics}")
    if max_transitions == DEFAULT_MAX_TRANSITIONS:
        if not opening_diagnostics.get("handoff", {}).get("clean_d4h0_handoff"):
            raise EvaluatorError(f"opening did not cleanly hand off at d4h0: {opening_diagnostics}")

    executor_diagnostics = downstream.diagnostics_json() if callable(getattr(downstream, "diagnostics_json", None)) else {}
    final_farm = (final_state.get("farms") or [])[seat]
    final_bank = final_farm.get("money") if isinstance(final_farm, Mapping) else None
    final_farm_metrics = _farm_evaluation_snapshot(final_state, seat)
    final_day = executor_diagnostics.get("days", {}).get(str(final_state.get("day")), {})
    target_animals = (final_day.get("feasible") or {}).get("animal_targets", {})
    target_error = {
        animal: int(final_farm_metrics["board_animals"].get(animal, 0))
        - int(target_animals.get(animal, 0))
        for animal in sorted(ANIMAL_NAMES)
    }
    work_debt = sum(
        len((record.get("end_of_day_work_debt") or {}).get("all") or [])
        for record in (executor_diagnostics.get("days") or {}).values()
    )
    final_status = "complete" if terminal else "partial"
    game: dict[str, Any] = {
        "seed": seed,
        "seat": seat,
        "opening": opening,
        "opponent": opponent,
        "prior_debt_suppression": prior_debt_suppression,
        "turn_trace": turn_trace,
        "aggressive_sell_all": aggressive_sell_all,
        "immediate_plant_water": immediate_plant_water,
        "optional_idle_cleanup": optional_idle_cleanup or optional_spare_watering,
        "optional_spare_watering": optional_spare_watering,
        "cleanup_mode": cleanup_mode,
        "cleanup_metrics": executor_diagnostics.get("cleanup_metrics", {}),
        "evaluation_metrics": {
            "purchased_animals": purchased_animals,
            "placed_animals": placed_animals,
            "animals_left_in_inventories": final_farm_metrics["inventory_animals"],
            "final_animal_target_error": target_error,
            "animal_escapes": sum(
                int(event.get("decrease", 0)) for event in animal_events),
            "weed_tiles_created": weed_tiles_created,
            "final_weed_tiles": final_farm_metrics["weed_tiles"],
            "action_counts": action_counts,
            "worker_pass_count": action_counts.get("PASS", 0),
            "movement_count": sum(action_counts.get(name, 0) for name in (
                "NORTH", "SOUTH", "EAST", "WEST")),
            "unfinished_work_debt": work_debt,
            "fallback_errors": len(executor_diagnostics.get("fallback_errors", [])),
        },
        "status": final_status,
        "transitions": transitions,
        "final": {
            "step": final_state.get("step"),
            "day": final_state.get("day"),
            "hour": final_state.get("hour"),
            "bank": final_bank,
            "result": list(getattr(backend, "rewards", [])),
            "statuses": list(getattr(backend, "statuses", statuses)),
        },
        "opening_diagnostics": opening_diagnostics,
        "executor_diagnostics": executor_diagnostics,
        "provider": {"identity": "executor_v0.CheckpointPlanProvider", "variant": provider_variant},
        "timeline": timeline,
        "animal_total_decrease_events": animal_events,
        "daily_summary": _daily_summary(timeline),
        "tested_action_trace_sha256": _action_trace_digest(tested_trace),
        "opponent_action_trace_sha256": _action_trace_digest(opponent_trace),
    }
    game["deterministic_rerun_identity"] = hashlib.sha256(_canonical_json({
        "seed": seed,
        "seat": seat,
        "opening": opening,
        "opponent": opponent,
        "prior_debt_suppression": prior_debt_suppression,
        "aggressive_sell_all": aggressive_sell_all,
        "immediate_plant_water": immediate_plant_water,
        "optional_idle_cleanup": optional_idle_cleanup or optional_spare_watering,
        "optional_spare_watering": optional_spare_watering,
        "cleanup_mode": cleanup_mode,
        "tested_action_trace_sha256": game["tested_action_trace_sha256"],
        "final": game["final"],
    })).hexdigest()
    return _ensure_finite(game)


def run_game(
    checkpoint_path: str | Path,
    seed: int,
    seat: int,
    *,
    backend: str = "fast",
    opening: str = DEFAULT_OPENING,
    opponent: str = DEFAULT_OPPONENT,
    prior_debt_suppression: bool = True,
    turn_trace: bool = False,
    aggressive_sell_all: bool = False,
    immediate_plant_water: bool = True,
    optional_idle_cleanup: bool = False,
    optional_spare_watering: bool = False,
    max_transitions: int = DEFAULT_MAX_TRANSITIONS,
    backend_factory: BackendFactory | None = None,
    provider_factory: ProviderFactory | None = None,
) -> dict[str, Any]:
    """Run one seed/seat game and return its deterministic game document."""
    path = Path(checkpoint_path)
    seed = _require_seed(seed)
    seat = _require_seat(seat)
    if backend not in ("fast", "official"):
        raise EvaluatorError(f"backend must be 'fast' or 'official', got {backend!r}")
    if opening != DEFAULT_OPENING:
        raise EvaluatorError(f"opening must be {DEFAULT_OPENING!r}, got {opening!r}")
    if opponent != DEFAULT_OPPONENT:
        raise EvaluatorError("this evaluator supports only the fixed PASS opponent")
    if not isinstance(prior_debt_suppression, bool) \
            or not isinstance(turn_trace, bool) \
            or not isinstance(aggressive_sell_all, bool) \
            or not isinstance(immediate_plant_water, bool) \
            or not isinstance(optional_idle_cleanup, bool) \
            or not isinstance(optional_spare_watering, bool):
        raise EvaluatorError(
            "prior_debt_suppression, turn_trace, aggressive_sell_all, "
            "immediate_plant_water, and "
            "optional_idle_cleanup, optional_spare_watering "
            "must be booleans"
        )
    max_transitions = _require_int(max_transitions, "max_transitions", minimum=0)
    if max_transitions > DEFAULT_MAX_TRANSITIONS:
        raise EvaluatorError(f"max_transitions must be <= {DEFAULT_MAX_TRANSITIONS}")
    if provider_factory is None:
        checkpoint_info = _checkpoint_metadata(path)
    else:
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {path}")
        checkpoint_info = {
            "path": str(path), "sha256": _checkpoint_sha256(path),
            "variant": "injected", "epoch": None,
        }
    return _run_one(
        path, checkpoint_info, seed, seat,
        backend_name=backend,
        opening=opening,
        opponent=opponent,
        prior_debt_suppression=prior_debt_suppression,
        turn_trace=turn_trace,
        aggressive_sell_all=aggressive_sell_all,
        immediate_plant_water=immediate_plant_water,
        optional_idle_cleanup=optional_idle_cleanup,
        optional_spare_watering=optional_spare_watering,
        max_transitions=max_transitions,
        backend_factory=backend_factory or (lambda name, config: make_backend(name, config)),
        provider_factory=provider_factory,
    )


def run_panel(
    checkpoint_path: str | Path,
    seeds: Sequence[int],
    seats: Sequence[int],
    *,
    backend: str = "fast",
    opening: str = DEFAULT_OPENING,
    opponent: str = DEFAULT_OPPONENT,
    prior_debt_suppression: bool = True,
    turn_trace: bool = False,
    aggressive_sell_all: bool = False,
    immediate_plant_water: bool = True,
    optional_idle_cleanup: bool = False,
    optional_spare_watering: bool = False,
    max_transitions: int = DEFAULT_MAX_TRANSITIONS,
    output_path: str | Path | None = None,
    label: str = "executor-v07-local-full-game",
    source: str = "local",
    overwrite: bool = False,
    backend_factory: BackendFactory | None = None,
    provider_factory: ProviderFactory | None = None,
) -> dict[str, Any]:
    """Run the Cartesian product of ``seeds`` and ``seats`` and optionally save it."""
    path = Path(checkpoint_path)
    label = _require_text(label, "label")
    source = _require_text(source, "source")
    seed_list = [_require_seed(seed) for seed in seeds]
    seat_list = [_require_seat(seat) for seat in seats]
    if not seed_list or not seat_list:
        raise EvaluatorError("seeds and seats must each contain at least one value")
    if len(set(seed_list)) != len(seed_list) or len(set(seat_list)) != len(seat_list):
        raise EvaluatorError("duplicate seeds or seats are not allowed")
    if output_path is not None and Path(output_path).exists() and not overwrite:
        raise FileExistsError(
            f"output already exists: {output_path}; pass overwrite=True or --overwrite"
        )
    max_transitions = _require_int(max_transitions, "max_transitions", minimum=0)
    if max_transitions > DEFAULT_MAX_TRANSITIONS:
        raise EvaluatorError(f"max_transitions must be <= {DEFAULT_MAX_TRANSITIONS}")

    if provider_factory is None:
        checkpoint_info = _checkpoint_metadata(path)
    else:
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {path}")
        checkpoint_info = {
            "path": str(path), "sha256": _checkpoint_sha256(path),
            "variant": "injected", "epoch": None,
        }
    if backend not in ("fast", "official"):
        raise EvaluatorError(f"backend must be 'fast' or 'official', got {backend!r}")
    if opening != DEFAULT_OPENING:
        raise EvaluatorError(f"opening must be {DEFAULT_OPENING!r}, got {opening!r}")
    if opponent != DEFAULT_OPPONENT:
        raise EvaluatorError("this evaluator supports only the fixed PASS opponent")
    if not isinstance(prior_debt_suppression, bool) \
            or not isinstance(turn_trace, bool) \
            or not isinstance(aggressive_sell_all, bool) \
            or not isinstance(immediate_plant_water, bool) \
            or not isinstance(optional_idle_cleanup, bool) \
            or not isinstance(optional_spare_watering, bool):
        raise EvaluatorError(
            "prior_debt_suppression, turn_trace, aggressive_sell_all, "
            "immediate_plant_water, and "
            "optional_idle_cleanup, optional_spare_watering "
            "must be booleans"
        )
    cleanup_mode = (
        "weed_water" if optional_idle_cleanup else
        "water_only" if optional_spare_watering else "none"
    )
    repo_root = Path(__file__).resolve().parents[1]
    games = [
        _run_one(
            path, checkpoint_info, seed, seat,
            backend_name=backend,
            opening=opening,
            opponent=opponent,
            prior_debt_suppression=prior_debt_suppression,
            turn_trace=turn_trace,
            aggressive_sell_all=aggressive_sell_all,
            immediate_plant_water=immediate_plant_water,
            optional_idle_cleanup=optional_idle_cleanup,
            optional_spare_watering=optional_spare_watering,
            max_transitions=max_transitions,
            backend_factory=backend_factory or (lambda name, config: make_backend(name, config)),
            provider_factory=provider_factory,
        )
        for seed in seed_list
        for seat in seat_list
    ]
    configuration = {
        "seed_selection": seed_list,
        "seat_selection": seat_list,
        "aggressive_sell_all": aggressive_sell_all,
        "immediate_plant_water": immediate_plant_water,
        "optional_idle_cleanup": optional_idle_cleanup or optional_spare_watering,
        "optional_spare_watering": optional_spare_watering,
        "cleanup_mode": (
            "weed_water" if optional_idle_cleanup else
            "water_only" if optional_spare_watering else "none"
        ),
    }
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "label": label,
        "source": source,
        "source_provenance": {
            "repo_sha": _source_repo_sha(repo_root),
            "tool": "tools.run_executor_v07_panel",
            "immediate_plant_water": immediate_plant_water,
            "optional_idle_cleanup": optional_idle_cleanup or optional_spare_watering,
            "optional_spare_watering": optional_spare_watering,
            "cleanup_mode": cleanup_mode,
        },
        "checkpoint": dict(checkpoint_info),
        "backend": backend_provenance(backend, {"seed": "per_game"}),
        "engine": {"name": "kaggriculture", "version": "1.32.7", "max_transitions": max_transitions},
        "opening": opening_provenance(opening),
        "opponent": {"identity": DEFAULT_OPPONENT, "policy": "fixed_pass_current_hands"},
        "request": {
            "seeds": seed_list,
            "seats": seat_list,
            "opening": opening,
            "backend": backend,
            "prior_debt_suppression": prior_debt_suppression,
            "turn_trace": turn_trace,
            "aggressive_sell_all": aggressive_sell_all,
            "immediate_plant_water": immediate_plant_water,
            "optional_idle_cleanup": optional_idle_cleanup or optional_spare_watering,
            "optional_spare_watering": optional_spare_watering,
            "cleanup_mode": cleanup_mode,
            "max_transitions": max_transitions,
            "configuration": configuration,
        },
        "games": games,
    }
    unsigned = _ensure_finite(document)
    unsigned["artifact_sha256"] = hashlib.sha256(_canonical_json(unsigned)).hexdigest()
    document = _ensure_finite(unsigned)
    if output_path is not None:
        _write_artifact(document, Path(output_path), overwrite=overwrite)
    return document


def _csv_ints(values: Sequence[str] | None, name: str) -> list[int]:
    result: list[int] = []
    for value in values or []:
        for part in value.split(","):
            if not part.strip():
                raise EvaluatorError(f"empty value in {name}")
            try:
                result.append(int(part.strip()))
            except ValueError as exc:
                raise EvaluatorError(f"invalid integer in {name}: {part!r}") from exc
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", dest="seed_values", action="append")
    parser.add_argument("--seeds", dest="seed_lists", action="append")
    parser.add_argument("--seat", dest="seat_values", action="append")
    parser.add_argument("--seats", dest="seat_lists", action="append")
    parser.add_argument("--backend", choices=("fast", "official"), default="fast")
    parser.add_argument("--opening", default=DEFAULT_OPENING, choices=(DEFAULT_OPENING,))
    parser.add_argument("--opponent", default=DEFAULT_OPPONENT, choices=(DEFAULT_OPPONENT,))
    parser.add_argument("--prior-debt-suppression", choices=("on", "off"), default="on")
    parser.add_argument("--turn-trace", action="store_true")
    parser.add_argument(
        "--aggressive-sell-all", action="store_true",
        help="enable the experimental literal full-shed sell override",
    )
    parser.add_argument(
        "--immediate-plant-water", choices=("on", "off"), default="on",
        help="water a confirmed newly planted tile with its planting worker next turn",
    )
    parser.add_argument(
        "--optional-idle-cleanup", action="store_true",
        help="enable weed-first then water PASS-only cleanup",
    )
    parser.add_argument(
        "--optional-spare-watering", action="store_true",
        help="enable water-only PASS cleanup (legacy flag name); --optional-idle-cleanup supersedes it",
    )
    parser.add_argument("--max-transitions", type=int, default=DEFAULT_MAX_TRANSITIONS)
    parser.add_argument("--output", required=True)
    parser.add_argument("--label", default="executor-v07-local-full-game")
    parser.add_argument("--source", default="local")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        seeds = _csv_ints((args.seed_values or []) + (args.seed_lists or []), "seeds")
        seats = _csv_ints((args.seat_values or []) + (args.seat_lists or []), "seats")
        if not seeds or not seats:
            raise EvaluatorError("at least one --seed/--seeds and --seat/--seats value is required")
        run_panel(
            args.checkpoint,
            seeds,
            seats,
            backend=args.backend,
            opening=args.opening,
            opponent=args.opponent,
            prior_debt_suppression=args.prior_debt_suppression == "on",
            turn_trace=args.turn_trace,
            aggressive_sell_all=args.aggressive_sell_all,
            immediate_plant_water=args.immediate_plant_water == "on",
            optional_idle_cleanup=args.optional_idle_cleanup,
            optional_spare_watering=args.optional_spare_watering,
            max_transitions=args.max_transitions,
            output_path=args.output,
            label=args.label,
            source=args.source,
            overwrite=args.overwrite,
        )
    except (EvaluatorError, FileExistsError, FileNotFoundError, ValueError, OSError) as exc:
        print(f"run_executor_v07_panel: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

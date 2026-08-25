"""One-arm, fixed-plan multi-day executor evaluation runner.

The runner reconstructs a replay boundary from the seed/configuration, checks
that boundary against the recorded observation, then gives the unmodified
deterministic executor a :class:`FixedPlanTapeProvider` while the opponent's
recorded primitive actions remain fixed.  It deliberately emits one artifact;
comparison of two such artifacts belongs to a separate stage.
"""

from __future__ import annotations

import argparse
import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

from oracle.backend import SUPPORTED_BACKENDS, make_backend
from oracle.closed_loop import _executor_observation
from replay_daily.lifecycle import canonical_board

from tools.day_slice import (
    _farm_metrics,
    first_diff,
    normalize_obs,
)
from tools.fixed_plan_tape import FixedPlanTape, FixedPlanTapeProvider
from tools.replay_io import episode_configuration, episode_id, load_replay

__all__ = [
    "FIXED_PLAN_RUNNER_SCHEMA_VERSION",
    "BoundaryMismatchError",
    "FixedPlanRunError",
    "MultiDayFixedPlanResult",
    "run_multiday_fixed_plan",
]


FIXED_PLAN_RUNNER_SCHEMA_VERSION = 1
_WINDOW_LENGTHS = frozenset((3, 5, 7))
_TURNS_PER_DAY = 24
_EXPECTED_STATUSES = frozenset(("ACTIVE", "DONE"))
_MOVEMENT_OPS = frozenset(("NORTH", "SOUTH", "EAST", "WEST"))
_CANONICAL_INTERACTION_OPS = frozenset({
    "WATER", "HARVEST", "DIG", "PLANT", "BUILD_COOP", "BUILD_PASTURE",
    "PLACE", "FEED", "CARE", "FERTILIZE", "COLLECT_FERTILIZER",
})


class FixedPlanRunError(ValueError):
    """Raised when a fixed-plan run cannot satisfy its frozen-input contract."""


class BoundaryMismatchError(FixedPlanRunError):
    """Raised at the first replay/engine boundary difference."""


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"artifact value is not JSON-safe: {type(value).__name__}")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(_json_safe(value))).hexdigest()


def _require_nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise FixedPlanRunError(f"{label} must be a non-empty string")
    return value


def _load_replay_input(value: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    if isinstance(value, (str, os.PathLike)):
        path = Path(value)
        try:
            raw = path.read_bytes()
        except OSError as error:
            raise FixedPlanRunError(f"cannot read replay {path}: {error}") from error
        replay = load_replay(path)
        provenance = {
            "kind": "file",
            "path": str(path),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        return replay, provenance
    if not isinstance(value, Mapping):
        raise FixedPlanRunError(
            f"replay must be a path or mapping, got {type(value).__name__}"
        )
    replay = copy.deepcopy(dict(value))
    return replay, {"kind": "object", "sha256": _sha256(replay)}


def _load_tape_input(value: Any) -> tuple[FixedPlanTape, dict[str, Any]]:
    if isinstance(value, FixedPlanTape):
        return value, {"kind": "object"}
    if isinstance(value, (str, os.PathLike)):
        path = Path(value)
        return FixedPlanTape.load(path), {"kind": "file", "path": str(path)}
    raise FixedPlanRunError(
        f"fixed_plan_tape must be a FixedPlanTape or path, got {type(value).__name__}"
    )


def _step_actions(steps: Sequence[Any], index: int) -> list[dict[str, Any]]:
    try:
        entries = steps[index]
    except IndexError as error:
        raise FixedPlanRunError(
            f"replay is missing step {index}, required by the requested window"
        ) from error
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)) \
            or len(entries) != 2:
        raise FixedPlanRunError(f"replay step {index} must contain two seat entries")
    actions: list[dict[str, Any]] = []
    for seat, entry in enumerate(entries):
        if not isinstance(entry, Mapping) or "action" not in entry:
            raise FixedPlanRunError(
                f"replay step {index} seat {seat} is missing a primitive action"
            )
        action = entry["action"]
        if not isinstance(action, Mapping):
            raise FixedPlanRunError(
                f"replay step {index} seat {seat} action must be a mapping"
            )
        actions.append(copy.deepcopy(dict(action)))
    return actions


def _boundary_observation(steps: Sequence[Any], step: int, seat: int) -> Mapping[str, Any]:
    try:
        entry = steps[step][seat]
    except (IndexError, TypeError, KeyError) as error:
        raise FixedPlanRunError(
            f"replay step {step} seat {seat} is missing a boundary observation"
        ) from error
    if not isinstance(entry, Mapping) or not isinstance(entry.get("observation"), Mapping):
        raise FixedPlanRunError(
            f"replay step {step} seat {seat} has no mapping observation"
        )
    return entry["observation"]


def _verify_boundary(
    observations: Sequence[Mapping[str, Any]],
    steps: Sequence[Any],
    step: int,
) -> dict[str, Any]:
    checked: list[dict[str, Any]] = []
    for seat in (0, 1):
        diff = first_diff(
            normalize_obs(observations[seat]),
            normalize_obs(_boundary_observation(steps, step, seat)),
        )
        item = {"seat": seat, "verified": diff is None, "first_diff": diff}
        checked.append(item)
        if diff is not None:
            raise BoundaryMismatchError(
                f"replay boundary mismatch at step {step} seat {seat}: {diff}"
            )
    return {"verified": True, "step": step, "seats": checked}


def _unit_actions(action: Mapping[str, Any]) -> list[tuple[int, Any]]:
    result: list[tuple[int, Any]] = [(0, action.get("farmer"))]
    result.extend((index, value) for index, value in enumerate(action.get("hands") or [], 1))
    return result


def _op(value: Any) -> str | None:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and value:
        return str(value[0])
    return None


def _worker_coord(obs: Mapping[str, Any], seat: int, worker_index: int) -> tuple[int, int] | None:
    farm = obs["farms"][seat]
    positions = [farm.get("farmer")]
    positions.extend(farm.get("hands") or [])
    if worker_index >= len(positions):
        return None
    position = positions[worker_index]
    if not isinstance(position, Sequence) or len(position) != 2:
        return None
    # Observation positions are [x, y]; lifecycle boards are [y][x].
    return int(position[1]), int(position[0])


def _tile(obs: Mapping[str, Any], seat: int, coord: tuple[int, int]) -> Any:
    board = canonical_board(
        obs["farms"][seat]["tiles"], int(obs["day"]), int(obs.get("step", 0))
    )
    y, x = coord
    if not (0 <= y < len(board) and 0 <= x < len(board[y])):
        return None
    return board[y][x]


def _same_identity(left: Any, right: Any) -> bool:
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        return False
    if left.get("kind") != right.get("kind"):
        return False
    if left.get("kind") == "PLANT":
        return left.get("crop") == right.get("crop")
    if "animal" in left or "animal" in right:
        return left.get("animal") == right.get("animal")
    return True


def _update_transition_metrics(
    day_record: dict[str, Any],
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    seat: int,
    action: Mapping[str, Any],
) -> None:
    old_board = canonical_board(
        before["farms"][seat]["tiles"], int(before["day"]), int(before.get("step", 0))
    )
    new_board = canonical_board(
        after["farms"][seat]["tiles"], int(after["day"]), int(after.get("step", 0))
    )
    refresh = int(after.get("day", 0)) != int(before.get("day", 0))
    old_animal_count = sum(
        1 for row in old_board for tile in row
        if isinstance(tile, Mapping) and "animal" in tile
    )
    new_animal_count = sum(
        1 for row in new_board for tile in row
        if isinstance(tile, Mapping) and "animal" in tile
    )
    if old_animal_count > new_animal_count:
        day_record["animal_count_decreases_observed"] += (
            old_animal_count - new_animal_count
        )
    loss_evidence_before = len(day_record["animal_loss_evidence"])
    for y in range(min(len(old_board), len(new_board))):
        for x in range(min(len(old_board[y]), len(new_board[y]))):
            old = old_board[y][x]
            new = new_board[y][x]
            if isinstance(new, Mapping) and new.get("kind") == "WEED" \
                    and not (isinstance(old, Mapping) and old.get("kind") == "WEED"):
                day_record["weeds_created"] += 1
            if isinstance(old, Mapping) and old.get("kind") == "PLANT" \
                    and isinstance(new, Mapping) and new.get("kind") == "WEED":
                day_record["crops_destroyed_observed"] += 1
                day_record["crop_destroyed_evidence"].append({
                    "coord": [y, x],
                    "crop": old.get("crop"),
                    "reason": "plant_to_weed_transition",
                })
            if not refresh or not isinstance(old, Mapping) or "animal" not in old:
                continue
            stable_structure = isinstance(new, Mapping) \
                and new.get("kind") == old.get("kind") \
                and "animal" not in new
            if stable_structure:
                starving = int(old.get("consecutive_unfed") or 0) >= 1
                evidence = {
                    "coord": [y, x],
                    "species": old.get("animal"),
                    "structure": old.get("kind"),
                    "consecutive_unfed_before_refresh": int(
                        old.get("consecutive_unfed") or 0),
                    "kind": "escape_evidence" if starving else "loss_at_refresh",
                }
                day_record["animal_loss_evidence"].append(evidence)
                if starving:
                    day_record["animal_escape_evidence"].append(evidence)
    stable_losses = len(day_record["animal_loss_evidence"]) - loss_evidence_before
    day_record["animal_count_decreases_without_stable_evidence"] += max(
        0, old_animal_count - new_animal_count - stable_losses
    )

    water_coords_seen: set[tuple[int, int]] = set()
    for worker_index, worker_action in _unit_actions(action):
        operation = _op(worker_action)
        if operation == "HARVEST":
            coord = _worker_coord(before, seat, worker_index)
            if coord is None:
                day_record["harvested_units"]["unmeasured_actions"] += 1
                continue
            old = _tile(before, seat, coord)
            new = _tile(after, seat, coord)
            old_units = int(old.get("yield_units") or 0) if isinstance(old, Mapping) else 0
            new_units = int(new.get("yield_units") or 0) if isinstance(new, Mapping) else 0
            if old_units >= new_units and _same_identity(old, new):
                day_record["harvested_units"]["observed"] += old_units - new_units
            else:
                day_record["harvested_units"]["unmeasured_actions"] += 1
        if operation != "WATER":
            continue
        coord = _worker_coord(before, seat, worker_index)
        classification = "redundant_or_unjustified"
        crop = None
        pre_state: dict[str, Any] = {}
        if coord is not None:
            pre_tile = _tile(before, seat, coord)
            if isinstance(pre_tile, Mapping):
                crop = pre_tile.get("crop")
                pre_state = {
                    key: pre_tile.get(key)
                    for key in (
                        "kind", "crop", "watered_today", "consecutive_unwatered",
                        "yield_units", "fertilized_until_day",
                    )
                }
                if coord in water_coords_seen:
                    classification = "redundant_or_unjustified"
                elif pre_tile.get("kind") == "PLANT":
                    from executor_v0.tasks import _water_urgency

                    urgency = _water_urgency(pre_tile)
                    classification = {
                        "must": "weed_prevention",
                        "yield": "yield_useful",
                    }.get(urgency, classification)
                water_coords_seen.add(coord)
        day_record["water_interactions"][classification] += 1
        day_record["water_interaction_evidence"].append({
            "worker": worker_index,
            "coord": list(coord) if coord is not None else None,
            "classification": classification,
            "crop": crop,
            "pre_action": pre_state,
        })


def _new_day_metrics(obs: Mapping[str, Any], seat: int) -> dict[str, Any]:
    metrics = _farm_metrics(obs, seat)
    return {
        "cash_start": metrics["cash"],
        "wealth_start": metrics["wealth"],
        "crops_start": metrics["crops"],
        "animals_start": metrics["animals"],
        "weeds_start": metrics["weeds"],
        "harvestable_start": metrics["harvestable"],
        "water_interactions": {
            "weed_prevention": 0,
            "yield_useful": 0,
            "redundant_or_unjustified": 0,
        },
        "water_interaction_evidence": [],
        "weeds_created": 0,
        "crops_destroyed_observed": 0,
        "crop_destroyed_evidence": [],
        "animal_loss_evidence": [],
        "animal_escape_evidence": [],
        "animal_count_decreases_observed": 0,
        "animal_count_decreases_without_stable_evidence": 0,
        "harvested_units": {"observed": 0, "unmeasured_actions": 0},
        "action_turn_counts": {
            "movement": 0, "pickup": 0, "interaction": 0, "pass": 0,
        },
    }


def _action_turn_counts(action: Mapping[str, Any]) -> dict[str, int]:
    counts = {"movement": 0, "pickup": 0, "interaction": 0, "pass": 0}
    for _, worker_action in _unit_actions(action):
        operation = _op(worker_action)
        if operation in _MOVEMENT_OPS:
            counts["movement"] += 1
        elif operation == "PICKUP":
            counts["pickup"] += 1
        elif operation == "PASS":
            counts["pass"] += 1
        elif operation is not None:
            counts["interaction"] += 1
    return counts


def _trace_hash(actions: list[Any]) -> str:
    return _sha256(actions)


def _finalize_day(
    start: Mapping[str, Any],
    end: Mapping[str, Any],
    day_record: dict[str, Any],
    diagnostics_record: Mapping[str, Any],
    day: int,
    seat: int,
) -> dict[str, Any]:
    end_metrics = _farm_metrics(end, seat)
    result = dict(day_record)
    result.update({
        "day": day,
        "cash_end": end_metrics["cash"],
        "wealth_end": end_metrics["wealth"],
        "cash_delta": end_metrics["cash"] - day_record["cash_start"],
        "wealth_delta": end_metrics["wealth"] - day_record["wealth_start"],
        "crops_end": end_metrics["crops"],
        "animals_end": end_metrics["animals"],
        "weeds_end": end_metrics["weeds"],
        "harvestable_end": end_metrics["harvestable"],
        "diagnostics": dict(diagnostics_record),
        "plan_targets": {
            "requested": diagnostics_record.get("requested"),
            "feasible": diagnostics_record.get("feasible"),
            "achieved_current": diagnostics_record.get("achieved_current"),
            "achieved_final": diagnostics_record.get("achieved_final"),
        },
        "projection_changes": diagnostics_record.get("projection_changes", {}),
        "eod_work_debt": diagnostics_record.get(
            "end_of_day_work_debt",
            {"all": [], "survival": [], "maintenance": [], "productive": [], "manager": []},
        ),
        "pending_task_turns": diagnostics_record.get("pending_task_turns", {}),
        "foreman_counts": diagnostics_record.get("foreman_counts", {}),
        "survival": diagnostics_record.get("survival", {}),
        "hires": diagnostics_record.get("hires", {}),
        "unaffordable_orders": diagnostics_record.get("unaffordable_market_orders", []),
    })
    result["harvested_units"]["unavailable_reason"] = (
        "stable pre/post tile identity or yield transition was unavailable"
        if result["harvested_units"]["unmeasured_actions"]
        else None
    )
    return _json_safe(result)


def _artifact_document(
    *,
    label: str,
    executor_provenance: str,
    replay_provenance: Mapping[str, Any],
    tape_provenance: Mapping[str, Any],
    tape_fingerprint: str,
    replay: Mapping[str, Any],
    config: Mapping[str, Any],
    seat: int,
    backend: str,
    boundary: Mapping[str, Any],
    days: list[dict[str, Any]],
    diagnostics: Mapping[str, Any],
    opponent_trace: list[Any],
    tested_trace: list[Any],
    turn_trace: bool,
) -> dict[str, Any]:
    window = tape_provenance["recording_window"]
    turns = len(tested_trace)
    totals = {
        "turns": turns,
        "days": len(days),
        "cash_delta": sum(float(day["cash_delta"]) for day in days),
        "wealth_delta": sum(float(day["wealth_delta"]) for day in days),
        "weeds_created": sum(int(day["weeds_created"]) for day in days),
        "crops_destroyed_observed": sum(
            int(day["crops_destroyed_observed"]) for day in days),
        "water_interactions": {
            category: sum(int(day["water_interactions"][category]) for day in days)
            for category in (
                "weed_prevention", "yield_useful", "redundant_or_unjustified",
            )
        },
        "harvested_units_observed": sum(
            int(day["harvested_units"]["observed"]) for day in days),
        "harvested_units_unmeasured_actions": sum(
            int(day["harvested_units"]["unmeasured_actions"]) for day in days),
        "animal_loss_evidence": sum(len(day["animal_loss_evidence"]) for day in days),
        "animal_escape_evidence": sum(len(day["animal_escape_evidence"]) for day in days),
        "animal_count_decreases_observed": sum(
            int(day["animal_count_decreases_observed"]) for day in days),
        "animal_count_decreases_without_stable_evidence": sum(
            int(day["animal_count_decreases_without_stable_evidence"]) for day in days),
    }
    document = {
        "schema_version": FIXED_PLAN_RUNNER_SCHEMA_VERSION,
        "label": label,
        "executor_provenance": executor_provenance,
        "tape_fingerprint": tape_fingerprint,
        "tape": {
            "fingerprint": tape_fingerprint,
            "provenance": _json_safe(tape_provenance),
        },
        "replay": {
            **dict(replay_provenance),
            "episode_id": int(episode_id(replay)),
            "seed": int(config["seed"]),
            "configuration_sha256": _sha256(config),
            "configuration": _json_safe(config),
        },
        "seat": seat,
        "run_config": {"turn_trace": turn_trace},
        "window": {
            "start_day": int(window["start_day"]),
            "end_day": int(window["end_day"]),
            "length": len(days),
            "turns_per_day": _TURNS_PER_DAY,
            "turns": turns,
        },
        "backend": {
            "name": backend,
            "tape_provenance": _json_safe(tape_provenance["backend"]),
            "engine": _json_safe(tape_provenance["engine"]),
            "source_repo_sha": tape_provenance["source_repo_sha"],
        },
        "boundary": _json_safe(boundary),
        "strategy_inputs": {
            "fixed_plan_provider": True,
            "live_manager_invocations": 0,
            "opening_source": "replay_prefix",
            "opponent_trace_sha256": _trace_hash(opponent_trace),
            "tested_action_trace_sha256": _trace_hash(tested_trace),
        },
        "days": days,
        "totals": totals,
        "executor_diagnostics": _json_safe(diagnostics),
    }
    document["artifact_sha256"] = hashlib.sha256(
        _canonical_json_bytes(document)
    ).hexdigest()
    return _json_safe(document)


@dataclass(frozen=True)
class MultiDayFixedPlanResult:
    """Deterministic versioned one-arm artifact returned by the library API."""

    document: dict[str, Any]

    @property
    def artifact_sha256(self) -> str:
        return str(self.document["artifact_sha256"])

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.document)

    def to_json(self) -> str:
        return _canonical_json_bytes(self.document).decode("utf-8")

    def save(self, path: str | os.PathLike[str]) -> Path:
        output = Path(path)
        output.write_text(self.to_json() + "\n", encoding="utf-8")
        return output


def run_multiday_fixed_plan(
    replay_path_or_replay: Any,
    fixed_plan_tape: FixedPlanTape | str | os.PathLike[str],
    seat: int,
    window_length: int,
    *,
    backend: str = "fast",
    output_path: str | os.PathLike[str] | None = None,
    label: str | None = None,
    executor_provenance: str | None = None,
    turn_trace: bool = False,
) -> MultiDayFixedPlanResult:
    """Execute one immutable tape window against a fixed replay opponent.

    The tape recording window supplies the absolute start/end day.  The
    explicit ``window_length`` must equal that inclusive tape window and must
    be 3, 5, or 7; no alternate start-day or live manager path is accepted.
    ``turn_trace`` is opt-in diagnostic capture and does not alter executor
    decisions.
    """
    if isinstance(window_length, bool) or window_length not in _WINDOW_LENGTHS:
        raise FixedPlanRunError(
            f"window_length must be one of {sorted(_WINDOW_LENGTHS)}, got {window_length!r}"
        )
    if isinstance(seat, bool) or seat not in (0, 1):
        raise FixedPlanRunError(f"seat must be 0 or 1, got {seat!r}")
    if backend not in SUPPORTED_BACKENDS:
        raise FixedPlanRunError(
            f"backend must be one of {SUPPORTED_BACKENDS}, got {backend!r}"
        )
    if not isinstance(turn_trace, bool):
        raise FixedPlanRunError(f"turn_trace must be a boolean, got {turn_trace!r}")
    label = _require_nonempty_string(label, "label")
    executor_provenance = _require_nonempty_string(
        executor_provenance, "executor_provenance"
    )

    replay, replay_provenance = _load_replay_input(replay_path_or_replay)
    tape, tape_source = _load_tape_input(fixed_plan_tape)
    tape_provenance = tape.provenance
    recording_window = tape_provenance["recording_window"]
    recorded_length = recording_window["end_day"] - recording_window["start_day"] + 1
    if recorded_length != window_length:
        raise FixedPlanRunError(
            "window_length does not match tape recording window: "
            f"requested {window_length}, tape covers {recorded_length} days"
        )
    if tape_provenance["seat"] != seat:
        raise FixedPlanRunError(
            f"seat mismatch: tape records seat {tape_provenance['seat']}, requested {seat}"
        )
    if tape_provenance["backend"]["name"] != backend:
        raise FixedPlanRunError(
            f"backend mismatch: tape records {tape_provenance['backend']['name']!r}, "
            f"requested {backend!r}"
        )
    config = episode_configuration(replay)
    if int(tape_provenance["seed"]) != int(config["seed"]):
        raise FixedPlanRunError(
            f"seed mismatch: tape records {tape_provenance['seed']}, "
            f"replay records {config['seed']}"
        )
    steps = replay.get("steps")
    if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)):
        raise FixedPlanRunError("replay steps must be a sequence")
    start_step = int(recording_window["start_day"]) * _TURNS_PER_DAY
    end_step = start_step + window_length * _TURNS_PER_DAY
    if start_step < 0 or end_step >= len(steps):
        raise FixedPlanRunError(
            f"replay steps do not cover required boundary/window steps "
            f"{start_step}..{end_step}; available 0..{len(steps) - 1}"
        )

    backend_instance = make_backend(backend, config)
    observations = backend_instance.reset()
    for prefix_step in range(1, start_step + 1):
        observations, _, statuses = backend_instance.step(_step_actions(steps, prefix_step))
        if any(status not in _EXPECTED_STATUSES for status in statuses):
            raise FixedPlanRunError(
                f"unexpected backend status during replay prefix at step {prefix_step}: {statuses}"
            )
    boundary = _verify_boundary(observations, steps, start_step)

    from executor_v0.agent import AgentConfig, make_agent

    agent = make_agent(
        provider=FixedPlanTapeProvider(tape),
        seat=seat,
        config=AgentConfig(strict=True, turn_trace=turn_trace),
    )
    opponent = 1 - seat
    day_records: list[dict[str, Any]] = []
    action_trace: list[Any] = []
    opponent_trace: list[Any] = []
    day_starts: list[Mapping[str, Any]] = []
    day_accumulators: list[dict[str, Any]] = []
    for day_offset in range(window_length):
        day_starts.append(copy.deepcopy(observations[seat]))
        day_accumulators.append(_new_day_metrics(observations[seat], seat))
        for hour in range(_TURNS_PER_DAY):
            before = copy.deepcopy(observations)
            executor_input = _executor_observation(
                before[seat], from_fast=backend == "fast"
            )
            tested_action = agent(executor_input)
            if not isinstance(tested_action, Mapping):
                raise FixedPlanRunError(
                    f"executor returned {type(tested_action).__name__}, expected a mapping"
                )
            tested_action = copy.deepcopy(dict(tested_action))
            transition_step = start_step + day_offset * _TURNS_PER_DAY + hour + 1
            recorded_actions = _step_actions(steps, transition_step)
            opponent_action = recorded_actions[opponent]
            pair: list[Mapping[str, Any] | None] = [None, None]
            pair[seat] = tested_action
            pair[opponent] = copy.deepcopy(opponent_action)
            action_trace.append(copy.deepcopy(tested_action))
            opponent_trace.append(copy.deepcopy(opponent_action))
            observations, _, statuses = backend_instance.step(pair)  # type: ignore[arg-type]
            if any(status not in _EXPECTED_STATUSES for status in statuses):
                raise FixedPlanRunError(
                    f"unexpected backend status at window turn {len(action_trace) - 1}: {statuses}"
                )
            if all(status == "DONE" for status in statuses) \
                    and transition_step != end_step:
                raise FixedPlanRunError(
                    f"backend reached DONE before the requested window ended at step "
                    f"{transition_step}; required end step is {end_step}"
                )
            action_counts = _action_turn_counts(tested_action)
            for name, count in action_counts.items():
                day_accumulators[day_offset]["action_turn_counts"][name] += count
            _update_transition_metrics(
                day_accumulators[day_offset], before[seat], observations[seat],
                seat, tested_action
            )

        day_records.append(day_accumulators[day_offset])

    diagnostics = agent.diagnostics_json()
    diagnostic_days = diagnostics.get("days", {}) if isinstance(diagnostics, Mapping) else {}
    finalized_days: list[dict[str, Any]] = []
    for offset, day_record in enumerate(day_records):
        day = int(recording_window["start_day"]) + offset
        diagnostic_record = diagnostic_days.get(str(day), {})
        if not isinstance(diagnostic_record, Mapping):
            diagnostic_record = {}
        finalized_days.append(_finalize_day(
            day_starts[offset],
            observations[seat] if offset == window_length - 1 else day_starts[offset + 1],
            day_record, diagnostic_record, day, seat,
        ))

    document = _artifact_document(
        label=label,
        executor_provenance=executor_provenance,
        replay_provenance=replay_provenance,
        tape_provenance={**tape_provenance, "source": tape_source},
        tape_fingerprint=tape.artifact_sha256,
        replay=replay,
        config=config,
        seat=seat,
        backend=backend,
        boundary=boundary,
        days=finalized_days,
        diagnostics=diagnostics,
        opponent_trace=opponent_trace,
        tested_trace=action_trace,
        turn_trace=turn_trace,
    )
    result = MultiDayFixedPlanResult(document)
    if output_path is not None:
        result.save(output_path)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", required=True)
    parser.add_argument("--tape", required=True)
    parser.add_argument("--seat", required=True, type=int, choices=(0, 1))
    parser.add_argument("--window-length", required=True, type=int, choices=(3, 5, 7))
    parser.add_argument("--backend", default="fast", choices=SUPPORTED_BACKENDS)
    parser.add_argument("--output", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--executor-provenance", required=True)
    parser.add_argument(
        "--turn-trace",
        action="store_true",
        help="enable bounded per-turn executor causality diagnostics",
    )
    args = parser.parse_args(argv)
    result = run_multiday_fixed_plan(
        args.replay,
        args.tape,
        args.seat,
        args.window_length,
        backend=args.backend,
        output_path=args.output,
        label=args.label,
        executor_provenance=args.executor_provenance,
        turn_trace=args.turn_trace,
    )
    print(result.to_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Build checkpoint-free expert-intent fixed-plan tape artifacts.

This utility compiles canonical :class:`DailyPlan` values observed in a raw
replay.  It does not run a manager, executor, opponent, or engine, and its
artifacts are explicitly benchmark inputs rather than BC-E promotion evidence.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

# Script-mode execution puts ``tools/`` on sys.path instead of the repository
# root; restore the root before importing repository packages.
if __package__ in (None, ""):
    _REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)

from replay_daily.constants import ENGINE_VERSION
from tools import expert_plan
from tools.fixed_plan_tape import FixedPlanTape
from tools.replay_io import episode_configuration, episode_id, load_replay

__all__ = [
    "ExpertPlanTapeBuildError",
    "build_expert_plan_tape",
    "main",
]


_WINDOW_LENGTHS = frozenset((3, 5, 7))
_REPLAY_VERSION = ENGINE_VERSION
_TURNS_PER_DAY = 24


class ExpertPlanTapeBuildError(ValueError):
    """Raised when an expert-intent tape cannot be built safely."""


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_canonical_json_bytes(value))


def _require_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ExpertPlanTapeBuildError(f"{label} must be an integer, got {value!r}")
    return value


def _require_nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ExpertPlanTapeBuildError(f"{label} must be a non-empty string")
    return value


def _json_safe(value: Any, label: str) -> Any:
    try:
        encoded = _canonical_json_bytes(value)
        return json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise ExpertPlanTapeBuildError(f"{label} must be JSON-safe: {error}") from error


def _load_replay(value: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    if isinstance(value, (str, os.PathLike)):
        path = Path(value)
        try:
            raw = path.read_bytes()
        except OSError as error:
            raise ExpertPlanTapeBuildError(
                f"cannot read replay source {path}: {error}"
            ) from error
        try:
            replay = load_replay(path)
        except ValueError as error:
            raise ExpertPlanTapeBuildError(str(error)) from error
        return replay, {
            "kind": "file",
            "path": str(path),
            "name": path.name,
            "sha256": _sha256_bytes(raw),
        }
    if not isinstance(value, Mapping):
        raise ExpertPlanTapeBuildError(
            f"replay must be a path or mapping, got {type(value).__name__}"
        )
    replay = copy.deepcopy(dict(value))
    return replay, {
        "kind": "object",
        "name": None,
        "path": None,
        "sha256": _sha256_json(replay),
    }


def _validate_replay(replay: Mapping[str, Any], seat: int, start_day: int,
                     end_day: int) -> tuple[dict[str, Any], list[Any]]:
    version = replay.get("module_version")
    if version != _REPLAY_VERSION:
        raise ExpertPlanTapeBuildError(
            f"unsupported replay module_version {version!r}; "
            f"expected {_REPLAY_VERSION!r}"
        )
    if replay.get("name") not in (None, "kaggriculture"):
        raise ExpertPlanTapeBuildError(
            f"unsupported replay name {replay.get('name')!r}; expected 'kaggriculture'"
        )
    try:
        config = episode_configuration(dict(replay))
    except (TypeError, ValueError) as error:
        raise ExpertPlanTapeBuildError(f"invalid replay configuration: {error}") from error

    steps = replay.get("steps")
    if not isinstance(steps, list):
        raise ExpertPlanTapeBuildError("replay steps must be a list")
    last_required_index = (end_day + 1) * _TURNS_PER_DAY
    if end_day >= int(config["episodeSteps"]) // _TURNS_PER_DAY:
        raise ExpertPlanTapeBuildError(
            f"recording window {start_day}..{end_day} is outside episode bounds"
        )
    if last_required_index >= len(steps):
        raise ExpertPlanTapeBuildError(
            f"replay does not contain the requested window through step "
            f"{last_required_index}"
        )

    for day in range(start_day, end_day + 1):
        for hour in range(_TURNS_PER_DAY):
            observation_index = day * _TURNS_PER_DAY + hour
            _step_entry(steps, observation_index, seat, require_observation=True)
            _step_entry(steps, observation_index, 1 - seat, require_observation=True)
            action_index = observation_index + 1
            _step_entry(steps, action_index, seat, require_action=True)
            _step_entry(steps, action_index, 1 - seat, require_action=True)
    return config, steps


def _step_entry(steps: list[Any], index: int, seat: int, *,
                require_observation: bool = False,
                require_action: bool = False) -> Mapping[str, Any]:
    try:
        entries = steps[index]
        entry = entries[seat]
    except (IndexError, KeyError, TypeError) as error:
        raise ExpertPlanTapeBuildError(
            f"replay step {index} is missing seat {seat}"
        ) from error
    if not isinstance(entry, Mapping):
        raise ExpertPlanTapeBuildError(
            f"replay step {index} seat {seat} must be a mapping"
        )
    if require_observation:
        observation = entry.get("observation")
        if not isinstance(observation, Mapping):
            raise ExpertPlanTapeBuildError(
                f"replay step {index} seat {seat} is missing an observation"
            )
        expected_day, expected_hour = divmod(index, _TURNS_PER_DAY)
        if (observation.get("day"), observation.get("hour")) != (
            expected_day, expected_hour
        ):
            raise ExpertPlanTapeBuildError(
                f"replay step {index} seat {seat} has malformed day/hour "
                f"{observation.get('day')!r}/{observation.get('hour')!r}"
            )
    if require_action:
        action = entry.get("action")
        if not isinstance(action, Mapping):
            raise ExpertPlanTapeBuildError(
                f"replay step {index} seat {seat} is missing a primitive action"
            )
    return entry


def _opponent_trace(replay_steps: list[Any], seat: int, start_day: int,
                    end_day: int) -> tuple[list[dict[str, Any]], str]:
    opponent = 1 - seat
    trace: list[dict[str, Any]] = []
    for day in range(start_day, end_day + 1):
        for hour in range(_TURNS_PER_DAY):
            index = day * _TURNS_PER_DAY + hour + 1
            action = copy.deepcopy(_step_entry(
                replay_steps, index, opponent, require_action=True
            )["action"])
            trace.append({"day": day, "hour": hour, "action": action})
    return trace, _sha256_json(trace)


def _named_provenance(value: Mapping[str, Any] | None, *, name: str,
                      version: str, label: str) -> dict[str, Any]:
    if value is None:
        return {"name": name, "version": version}
    if not isinstance(value, Mapping):
        raise ExpertPlanTapeBuildError(f"{label} must be a mapping")
    result = dict(value)
    result.setdefault("name", name)
    result.setdefault("version", version)
    return _json_safe(result, label)


def build_expert_plan_tape(
    replay_path_or_replay: Any,
    *,
    seat: int,
    start_day: int,
    length: int,
    backend_name: str,
    source_repo_sha: str,
    opening_identity: str,
    output_path: str | Path,
    backend_version: str = ENGINE_VERSION,
    engine_name: str = "kaggriculture",
    engine_version: str = ENGINE_VERSION,
    backend_provenance_data: Mapping[str, Any] | None = None,
    engine_provenance_data: Mapping[str, Any] | None = None,
    opening_provenance_data: Mapping[str, Any] | None = None,
    label: str | None = None,
    force: bool = False,
) -> FixedPlanTape:
    """Compile one contiguous 3/5/7-day expert-intent tape.

    ``extract_daily_plan`` is called once for each absolute day.  The replay is
    only read; the resulting artifact contains canonical plans and JSON-safe
    provenance, never observations or model outputs.
    """
    seat = _require_int(seat, "seat")
    if seat not in (0, 1):
        raise ExpertPlanTapeBuildError(f"seat must be 0 or 1, got {seat}")
    start_day = _require_int(start_day, "start_day")
    if start_day < 0:
        raise ExpertPlanTapeBuildError(
            f"start_day must be non-negative, got {start_day}"
        )
    length = _require_int(length, "length")
    if length not in _WINDOW_LENGTHS:
        raise ExpertPlanTapeBuildError("length must be one of 3, 5, or 7")
    end_day = start_day + length - 1
    backend_name = _require_nonempty(backend_name, "backend_name")
    backend_version = _require_nonempty(backend_version, "backend_version")
    engine_name = _require_nonempty(engine_name, "engine_name")
    engine_version = _require_nonempty(engine_version, "engine_version")
    source_repo_sha = _require_nonempty(source_repo_sha, "source_repo_sha")
    opening_identity = _require_nonempty(opening_identity, "opening_identity")
    if label is not None:
        label = _require_nonempty(label, "label")

    output = Path(output_path)
    if output.exists() and not force:
        raise FileExistsError(
            f"expert plan tape output already exists: {output}; use --force to overwrite"
        )

    replay, replay_source = _load_replay(replay_path_or_replay)
    config, steps = _validate_replay(replay, seat, start_day, end_day)
    try:
        episode = episode_id(replay)
    except (TypeError, ValueError) as error:
        raise ExpertPlanTapeBuildError(f"invalid replay episode id: {error}") from error

    plans = [
        (day, expert_plan.extract_daily_plan(replay, seat, day))
        for day in range(start_day, end_day + 1)
    ]
    opponent_trace, opponent_digest = _opponent_trace(
        steps, seat, start_day, end_day
    )
    del opponent_trace  # only its deterministic digest belongs in provenance

    if opening_provenance_data is None:
        try:
            from rl_manager.provenance import opening_provenance

            opening = opening_provenance(opening_identity)
        except (ImportError, KeyError, OSError, ValueError) as error:
            raise ExpertPlanTapeBuildError(
                f"cannot derive opening provenance for {opening_identity!r}: {error}"
            ) from error
    else:
        opening = _json_safe(opening_provenance_data, "opening_provenance_data")
    if not isinstance(opening, Mapping):
        raise ExpertPlanTapeBuildError("opening_provenance_data must be a mapping")

    provenance: dict[str, Any] = {
        "manager": "expert-replay-intent",
        "checkpoint": "none:checkpoint-free",
        "model_variant": "expert-replay-v1",
        "seed": _require_int(config["seed"], "replay seed"),
        "seat": seat,
        "opening_identity": opening_identity,
        "opening_provenance": dict(opening),
        "source_repo_sha": source_repo_sha,
        "backend": _named_provenance(
            backend_provenance_data,
            name=backend_name,
            version=backend_version,
            label="backend_provenance_data",
        ),
        "engine": _named_provenance(
            engine_provenance_data,
            name=engine_name,
            version=engine_version,
            label="engine_provenance_data",
        ),
        "recording_window": {"start_day": start_day, "end_day": end_day},
        "replay": {
            "episode_id": episode,
            "source": replay_source["path"],
            "name": replay_source["name"],
            "source_sha256": replay_source["sha256"],
            "opponent_seat": 1 - seat,
            "opponent_trace_sha256": opponent_digest,
        },
        "opponent_trace_sha256": opponent_digest,
        "limitations": [
            "checkpoint-free expert-intent benchmark; not BC-E promotion evidence",
            "no live manager, executor, or opponent execution",
        ],
    }
    if label is not None:
        provenance["label"] = label

    tape = FixedPlanTape.create(plans=plans, provenance=provenance)
    tape.save(output)
    return tape


def _parse_json_object(text: str, label: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} must be valid JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a checkpoint-free expert-intent fixed-plan tape."
    )
    parser.add_argument("--replay", required=True)
    parser.add_argument("--seat", required=True, type=int, choices=(0, 1))
    parser.add_argument("--start-day", required=True, type=int)
    parser.add_argument("--length", required=True, type=int, choices=(3, 5, 7))
    parser.add_argument("--backend", required=True)
    parser.add_argument("--backend-version", default=ENGINE_VERSION)
    parser.add_argument("--engine-name", default="kaggriculture")
    parser.add_argument("--engine-version", default=ENGINE_VERSION)
    parser.add_argument("--opening", required=True)
    parser.add_argument("--opening-provenance")
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-repo-sha", required=True)
    parser.add_argument("--label")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    opening = (
        _parse_json_object(args.opening_provenance, "opening-provenance")
        if args.opening_provenance
        else None
    )
    tape = build_expert_plan_tape(
        args.replay,
        seat=args.seat,
        start_day=args.start_day,
        length=args.length,
        backend_name=args.backend,
        backend_version=args.backend_version,
        engine_name=args.engine_name,
        engine_version=args.engine_version,
        source_repo_sha=args.source_repo_sha,
        opening_identity=args.opening,
        opening_provenance_data=opening,
        output_path=args.output,
        label=args.label,
        force=args.force,
    )
    print(json.dumps({
        "output": str(args.output),
        "artifact_sha256": tape.artifact_sha256,
        "kind": "expert-intent-benchmark",
        "bc_e_promotion_evidence": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

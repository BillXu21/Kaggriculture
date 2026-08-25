"""Record one frozen BC-E executor trajectory into a fixed-plan tape.

The recorder owns the only live manager/executor run.  It reconstructs the
requested replay boundary in the selected backend, verifies that boundary,
then lets an injected or checkpoint-backed provider drive one reference
executor while the opponent is always taken from the replay primitive trace.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence

# Script-mode execution puts ``tools/`` on sys.path instead of the repository
# root; restore the root before importing repository packages.
if __package__ in (None, ""):
    _REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)

from executor_v0.agent import AgentConfig, ExecutorAgent
from executor_v0.manager import CheckpointPlanProvider, PlanProvider
from executor_v0.plan import DailyPlan
from opening_book.trace import validate_action
from oracle.backend import EngineBackend, make_backend
from replay_daily.constants import ENGINE_VERSION
from rl_manager.provenance import (
    backend_provenance,
    canonical_json,
    opening_provenance,
    sha256_hex,
)

from tools.day_slice import first_diff, normalize_obs
from tools.fixed_plan_tape import FixedPlanTape
from tools.replay_io import episode_configuration, episode_id, load_replay

__all__ = [
    "FixedPlanTapeRecordingError",
    "ReplayBoundaryMismatch",
    "RecorderConfig",
    "record_fixed_plan_tape",
    "main",
]


class FixedPlanTapeRecordingError(ValueError):
    """Raised when a fixed-plan recording cannot be completed safely."""


class ReplayBoundaryMismatch(FixedPlanTapeRecordingError):
    """Raised when the backend prefix does not match the recorded replay."""


@dataclass(frozen=True)
class RecorderConfig:
    """Explicit reference-executor provenance and runtime configuration."""

    name: str
    version: str
    configuration: Mapping[str, Any]


ExecutorFactory = Callable[[PlanProvider, int, AgentConfig], Callable[[Mapping], Mapping]]


def _require_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FixedPlanTapeRecordingError(f"{label} must be an integer, got {value!r}")
    return value


def _require_nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise FixedPlanTapeRecordingError(f"{label} must be a non-empty string")
    return value


def _json_safe(value: Any, label: str) -> Any:
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
        return json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise FixedPlanTapeRecordingError(f"{label} must be JSON-safe: {error}") from error


def _load_replay_with_digest(value: Any) -> tuple[dict[str, Any], str, str | None]:
    if isinstance(value, (str, os.PathLike)):
        path = Path(value)
        try:
            raw = path.read_bytes()
        except OSError as error:
            raise FixedPlanTapeRecordingError(
                f"cannot read replay source {path}: {error}"
            ) from error
        replay = load_replay(str(path))
        return replay, hashlib.sha256(raw).hexdigest(), str(path)
    if not isinstance(value, Mapping):
        raise FixedPlanTapeRecordingError(
            "replay_path_or_replay must be a replay path or mapping"
        )
    replay = copy.deepcopy(dict(value))
    return replay, sha256_hex(canonical_json(replay)), None


def _canonical_observations(
    backend: EngineBackend,
    observations: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Expose backend-canonical farms while retaining each private view."""
    canonical = backend.canonical_state()
    farms = canonical.get("farms")
    if not isinstance(farms, list) or len(farms) != 2:
        raise FixedPlanTapeRecordingError(
            "backend canonical_state must contain two farms"
        )
    adapted: list[dict[str, Any]] = []
    for obs in observations:
        view = copy.deepcopy(dict(obs))
        view["farms"] = copy.deepcopy(farms)
        if "step" not in view:
            view["step"] = int(view["day"]) * 24 + int(view.get("hour", 0))
        adapted.append(view)
    return adapted


def _step_entry(replay: Mapping[str, Any], index: int, seat: int) -> Mapping[str, Any]:
    steps = replay.get("steps")
    try:
        entry = steps[index][seat]
    except (IndexError, KeyError, TypeError) as error:
        raise FixedPlanTapeRecordingError(
            f"replay step {index} is missing seat {seat}"
        ) from error
    if not isinstance(entry, Mapping) or "action" not in entry:
        raise FixedPlanTapeRecordingError(
            f"replay step {index} seat {seat} is missing an action"
        )
    return entry


def _opponent_trace(
    replay: Mapping[str, Any], seat: int, start_day: int, end_day: int
) -> tuple[list[dict[str, Any]], str]:
    opponent = 1 - seat
    trace: list[dict[str, Any]] = []
    for absolute_day in range(start_day, end_day + 1):
        for hour in range(24):
            index = absolute_day * 24 + hour + 1
            raw_action = _step_entry(replay, index, opponent)["action"]
            action = copy.deepcopy(raw_action)
            validate_action(action, label=f"replay step {index} opponent")
            trace.append({"day": absolute_day, "hour": hour, "action": action})
    return trace, sha256_hex(canonical_json(trace))


def _default_executor_factory(
    provider: PlanProvider, seat: int, config: AgentConfig
) -> ExecutorAgent:
    return ExecutorAgent(provider, seat=seat, config=config)


def _executor_provenance(
    value: RecorderConfig | Mapping[str, Any],
) -> dict[str, Any]:
    if isinstance(value, RecorderConfig):
        result = {
            "name": value.name,
            "version": value.version,
            "configuration": dict(value.configuration),
        }
    elif isinstance(value, Mapping):
        result = dict(value)
    else:
        raise FixedPlanTapeRecordingError(
            "reference_executor_provenance must be RecorderConfig or mapping"
        )
    _require_nonempty(result.get("name"), "reference_executor_provenance.name")
    _require_nonempty(result.get("version"), "reference_executor_provenance.version")
    if "configuration" not in result or not isinstance(result["configuration"], Mapping):
        raise FixedPlanTapeRecordingError(
            "reference_executor_provenance.configuration must be a mapping"
        )
    return _json_safe(result, "reference_executor_provenance")


def _checkpoint_fields(
    provider: PlanProvider,
    checkpoint_path: str | Path | None,
    checkpoint_sha256: str | None,
    model_variant: str | None,
) -> tuple[str, str, str]:
    detected_variant = model_variant or getattr(provider, "model_variant", "E")
    detected_variant = _require_nonempty(detected_variant, "model_variant")
    if detected_variant != "E":
        raise FixedPlanTapeRecordingError(
            f"fixed-plan recording requires BC-E model variant 'E', got {detected_variant!r}"
        )
    if checkpoint_path is None:
        return "injected-plan-provider", "injected-plan-provider", detected_variant
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"BC manager checkpoint not found: {path}; supply the real D-019 best.pt path"
        )
    digest = _require_nonempty(checkpoint_sha256, "checkpoint_sha256")
    return f"{path.name}:sha256={digest}", str(path), detected_variant


def record_fixed_plan_tape(
    replay_path_or_replay: Any,
    plan_provider: PlanProvider | None = None,
    *,
    provider: PlanProvider | None = None,
    seat: int,
    start_day: int,
    length: int,
    backend_name: str,
    opening_identity: str,
    output_path: str | Path,
    source_repo_sha: str,
    reference_executor_provenance: RecorderConfig | Mapping[str, Any],
    opening_provenance_data: Mapping[str, Any] | None = None,
    checkpoint_path: str | Path | None = None,
    checkpoint_sha256: str | None = None,
    model_variant: str | None = None,
    manager_identity: str = "executor_v0.CheckpointPlanProvider",
    reference_executor_factory: ExecutorFactory | None = None,
    reference_executor_config: AgentConfig | None = None,
    force: bool = False,
) -> FixedPlanTape:
    """Record absolute-day plans from one deterministic reference trajectory."""
    if plan_provider is not None and provider is not None:
        raise FixedPlanTapeRecordingError("provide only one of plan_provider or provider")
    plan_provider = plan_provider or provider
    if plan_provider is None or not callable(getattr(plan_provider, "daily_plan", None)):
        raise FixedPlanTapeRecordingError("an injected PlanProvider is required")
    seat = _require_int(seat, "seat")
    if seat not in (0, 1):
        raise FixedPlanTapeRecordingError(f"seat must be 0 or 1, got {seat}")
    start_day = _require_int(start_day, "start_day")
    length = _require_int(length, "length")
    if start_day < 0:
        raise FixedPlanTapeRecordingError(f"start_day must be non-negative, got {start_day}")
    if length not in (3, 5, 7):
        raise FixedPlanTapeRecordingError("length must be one of 3, 5, or 7")
    end_day = start_day + length - 1
    backend_name = _require_nonempty(backend_name, "backend_name")
    opening_identity = _require_nonempty(opening_identity, "opening_identity")
    source_repo_sha = _require_nonempty(source_repo_sha, "source_repo_sha")
    manager_identity = _require_nonempty(manager_identity, "manager_identity")
    output = Path(output_path)
    if output.exists() and not force:
        raise FileExistsError(
            f"fixed-plan tape output already exists: {output}; use --force to overwrite"
        )

    replay, replay_digest, replay_source = _load_replay_with_digest(replay_path_or_replay)
    config = episode_configuration(replay)
    steps = replay.get("steps")
    if not isinstance(steps, list):
        raise FixedPlanTapeRecordingError("replay steps must be a list")
    episode_steps = _require_int(config["episodeSteps"], "configuration.episodeSteps")
    if end_day >= episode_steps // 24:
        raise FixedPlanTapeRecordingError(
            f"recording window {start_day}..{end_day} is outside the episode bounds"
        )
    last_required_index = (end_day + 1) * 24
    if last_required_index >= len(steps):
        raise FixedPlanTapeRecordingError(
            f"replay does not contain the requested window through step {last_required_index}"
        )

    opening = (
        _json_safe(opening_provenance_data, "opening_provenance_data")
        if opening_provenance_data is not None
        else _json_safe(opening_provenance(opening_identity), "opening_provenance")
    )
    executor_provenance = _executor_provenance(reference_executor_provenance)
    checkpoint, checkpoint_location, detected_variant = _checkpoint_fields(
        plan_provider, checkpoint_path, checkpoint_sha256, model_variant
    )
    executor_config = reference_executor_config or AgentConfig(strict=True)
    if not isinstance(executor_config, AgentConfig):
        raise FixedPlanTapeRecordingError("reference_executor_config must be AgentConfig")

    backend_configuration = dict(config)
    backend = make_backend(backend_name, backend_configuration)
    live_obs = _canonical_observations(backend, backend.reset())
    boundary_step = start_day * 24
    for index in range(1, boundary_step + 1):
        pair = [
            copy.deepcopy(_step_entry(replay, index, 0)["action"]),
            copy.deepcopy(_step_entry(replay, index, 1)["action"]),
        ]
        validate_action(pair[0], label=f"replay step {index} seat 0")
        validate_action(pair[1], label=f"replay step {index} seat 1")
        live_obs = _canonical_observations(backend, backend.step(pair)[0])

    recorded_boundary = _step_entry(replay, boundary_step, seat).get("observation")
    if not isinstance(recorded_boundary, Mapping):
        raise ReplayBoundaryMismatch(
            f"replay boundary step {boundary_step} seat {seat} has no observation"
        )
    boundary_diff = first_diff(
        normalize_obs(live_obs[seat]), normalize_obs(recorded_boundary)
    )
    if boundary_diff is not None:
        raise ReplayBoundaryMismatch(
            f"replay boundary mismatch at day {start_day}, seat {seat}: {boundary_diff}"
        )

    opponent_trace, opponent_digest = _opponent_trace(replay, seat, start_day, end_day)
    captured: dict[int, DailyPlan] = {}

    class RecordingProvider:
        def daily_plan(self, obs, requested_seat, previous_execution=None):
            day = _require_int(int(obs["day"]), "provider observation day")
            if day in captured:
                raise FixedPlanTapeRecordingError(f"duplicate manager plan for day {day}")
            plan = plan_provider.daily_plan(
                copy.deepcopy(obs), requested_seat,
                copy.deepcopy(previous_execution) if previous_execution is not None else None,
            )
            if not isinstance(plan, DailyPlan):
                raise FixedPlanTapeRecordingError(
                    f"manager returned {type(plan).__name__}, expected DailyPlan"
                )
            captured[day] = plan
            return plan

        def diagnostics_json(self):
            diagnostics = getattr(plan_provider, "diagnostics_json", None)
            return diagnostics() if callable(diagnostics) else {}

    recording_provider = RecordingProvider()
    factory = reference_executor_factory or _default_executor_factory
    executor = factory(recording_provider, seat, executor_config)
    if not callable(executor):
        raise FixedPlanTapeRecordingError("reference executor factory did not return a callable")

    opponent = 1 - seat
    for trace_index, trace_entry in enumerate(opponent_trace):
        action = executor(copy.deepcopy(live_obs[seat]))
        if not isinstance(action, Mapping):
            raise FixedPlanTapeRecordingError(
                f"reference executor returned {type(action).__name__} at trace index {trace_index}"
            )
        pair: list[Mapping[str, Any] | None] = [None, None]
        pair[seat] = copy.deepcopy(dict(action))
        pair[opponent] = copy.deepcopy(trace_entry["action"])
        live_obs = _canonical_observations(backend, backend.step(pair)[0])
        if backend.statuses == ["DONE", "DONE"] and trace_index + 1 < len(opponent_trace):
            raise FixedPlanTapeRecordingError(
                f"reference executor terminated before recording day {trace_entry['day']}"
            )

    diagnostics = getattr(executor, "diagnostics_json", None)
    if callable(diagnostics):
        snapshot = diagnostics()
        fallback_errors = snapshot.get("fallback_errors", []) if isinstance(snapshot, Mapping) else []
        if fallback_errors:
            raise FixedPlanTapeRecordingError(
                f"reference executor manager fallback/error: {fallback_errors[-1]}"
            )

    expected_days = set(range(start_day, end_day + 1))
    actual_days = set(captured)
    if actual_days != expected_days:
        raise FixedPlanTapeRecordingError(
            f"manager plans must cover exactly {sorted(expected_days)}; "
            f"missing={sorted(expected_days - actual_days)}, extra={sorted(actual_days - expected_days)}"
        )

    backend_meta = backend_provenance(backend_name, backend_configuration)
    provenance = {
        "manager": manager_identity,
        "checkpoint": checkpoint,
        "checkpoint_path": checkpoint_location,
        "checkpoint_name": Path(checkpoint_location).name if checkpoint_location else None,
        "checkpoint_sha256": checkpoint_sha256,
        "model_variant": detected_variant,
        "seed": int(config["seed"]),
        "seat": seat,
        "opening_identity": opening_identity,
        "opening_provenance": opening,
        "source_repo_sha": source_repo_sha,
        "backend": {
            "name": backend_name,
            "version": ENGINE_VERSION,
            "configuration": backend_configuration,
            "provenance": backend_meta,
        },
        "engine": {"name": "kaggriculture", "version": ENGINE_VERSION},
        "recording_window": {"start_day": start_day, "end_day": end_day},
        "replay": {
            "episode_id": episode_id(replay),
            "source": replay_source,
            "source_sha256": replay_digest,
            "opponent_seat": opponent,
            "opponent_trace_sha256": opponent_digest,
        },
        "opponent_trace_sha256": opponent_digest,
        "reference_executor": executor_provenance,
    }
    tape = FixedPlanTape.create(
        plans=[(day, captured[day]) for day in sorted(captured)],
        provenance=provenance,
    )
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
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    parser = argparse.ArgumentParser(description="Record a frozen BC-E fixed-plan tape.")
    parser.add_argument("--replay", required=True)
    parser.add_argument("--seat", required=True, type=int, choices=(0, 1))
    parser.add_argument("--start-day", required=True, type=int)
    parser.add_argument("--length", required=True, type=int, choices=(3, 5, 7))
    parser.add_argument("--backend", required=True, choices=("fast", "official"))
    parser.add_argument("--opening", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-repo-sha", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reference-executor-name", required=True)
    parser.add_argument("--reference-executor-version", required=True)
    parser.add_argument("--reference-executor-config", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"BC manager checkpoint not found: {checkpoint_path}; supply the real D-019 best.pt path"
        )
    checkpoint_sha = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    provider = CheckpointPlanProvider(checkpoint_path, device="cpu")
    if provider.model_variant != "E":
        raise FixedPlanTapeRecordingError(
            f"fixed-plan recording requires BC-E model variant 'E', got {provider.model_variant!r}"
        )
    config_data = _parse_json_object(args.reference_executor_config, "reference-executor-config")
    executor_config = AgentConfig(**config_data)
    provenance = RecorderConfig(
        name=args.reference_executor_name,
        version=args.reference_executor_version,
        configuration=config_data,
    )
    tape = record_fixed_plan_tape(
        args.replay,
        provider,
        seat=args.seat,
        start_day=args.start_day,
        length=args.length,
        backend_name=args.backend,
        opening_identity=args.opening,
        output_path=args.output,
        source_repo_sha=args.source_repo_sha,
        reference_executor_provenance=provenance,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha,
        model_variant=provider.model_variant,
        reference_executor_config=executor_config,
        force=args.force,
    )
    print(json.dumps({"output": str(args.output), "artifact_sha256": tape.artifact_sha256}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

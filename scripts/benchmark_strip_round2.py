"""Profile Stage 2.5 strip controller and wrapper telemetry regions on CPU."""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
import time
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence
from unittest.mock import patch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from executor_v0 import strip_work as work_module  # noqa: E402
from executor_v0.strip_executor import StripExecutorController  # noqa: E402
from executor_v0.strip_market import MarketBootstrapState  # noqa: E402
from executor_v0.strip_routes import StripRoute  # noqa: E402
from executor_v0.strip_supply import (  # noqa: E402
    RouteSupplyPlan,
    RouteSupplyState,
)
from rl_manager import executor_factory as factory_module  # noqa: E402
from rl_manager.stage25_provider import Stage25PlanProvider  # noqa: E402
from scripts.benchmark_strip_round1 import (  # noqa: E402
    DEFAULT_SEEDS,
    action_trace,
    run_games as run_round1_games,
)
from rl_manager.trajectory import TrajectoryBuffer, e_input_spec  # noqa: E402


@dataclass
class RegionMetrics:
    controller_seconds: list[float] = field(default_factory=list)
    wrapper_seconds: list[float] = field(default_factory=list)
    work_build_seconds: list[float] = field(default_factory=list)
    controller_diagnostic_seconds: list[float] = field(default_factory=list)
    wrapper_deepcopy_seconds: list[float] = field(default_factory=list)
    full_diagnostic_deepcopy_seconds: list[float] = field(default_factory=list)
    executor_diagnostics_seconds: list[float] = field(default_factory=list)
    provider_diagnostics_seconds: list[float] = field(default_factory=list)
    work_diagnostic_seconds: list[float] = field(default_factory=list)
    diagnostic_json_seconds: list[float] = field(default_factory=list)
    diagnostic_json_calls: dict[str, int] = field(default_factory=dict)
    full_diagnostic_copy_count: int = 0
    diagnostic_sample: dict[str, Any] | None = None
    work_plan_samples: list[dict[str, Any]] = field(default_factory=list)
    _work_plan_sample_keys: set[tuple[int, int]] = field(default_factory=set)
    _wrapper_depth: int = 0
    _controller_diagnostic_depth: int = 0
    _executor_diagnostics_depth: int = 0


def run_games(
    seeds: Sequence[int], *, max_turns: int, low_telemetry: bool = False
):
    buffer = TrajectoryBuffer(
        capacity=max(512, len(seeds) * 128), input_spec=e_input_spec())
    return run_round1_games(
        seeds,
        max_turns=max_turns,
        low_telemetry=low_telemetry,
        trajectory_buffer=buffer,
    )


def _timed_ms(values: list[float]) -> dict[str, float | int]:
    total = sum(values)
    calls = len(values)
    return {
        "calls": calls,
        "total_seconds": total,
        "mean_ms": total * 1000.0 / calls if calls else 0.0,
    }


def _first_value_diff(
        before: Any, after: Any, path: str = "$",
) -> dict[str, Any] | None:
    if isinstance(before, dict) and isinstance(after, dict):
        for key in sorted(set(before) | set(after)):
            child_path = f"{path}.{key}"
            if key not in before or key not in after:
                return {
                    "path": child_path,
                    "before": before.get(key),
                    "after": after.get(key),
                }
            mismatch = _first_value_diff(before[key], after[key], child_path)
            if mismatch is not None:
                return mismatch
        return None
    if isinstance(before, list) and isinstance(after, list):
        for index, (left, right) in enumerate(zip(before, after)):
            mismatch = _first_value_diff(left, right, f"{path}[{index}]")
            if mismatch is not None:
                return mismatch
        if len(before) != len(after):
            return {
                "path": f"{path}.length",
                "before": len(before),
                "after": len(after),
            }
        return None
    if before != after:
        return {"path": path, "before": before, "after": after}
    return None


def first_mismatch(before: list[Any], after: list[Any]) -> dict[str, Any] | None:
    # Compare the values callers actually serialize.  Work-plan JSON helpers
    # may contain tuples, which json.dumps renders as arrays; comparing those
    # Python objects directly creates a false mismatch against parsed JSON.
    before = json.loads(json.dumps(before, allow_nan=False))
    after = json.loads(json.dumps(after, allow_nan=False))
    for index, (left, right) in enumerate(zip(before, after)):
        if left != right:
            return {
                "index": index,
                "identity": {
                    key: left.get(key, right.get(key))
                    for key in ("seed", "seat", "step", "day", "hour")
                    if key in left or key in right
                } if isinstance(left, dict) and isinstance(right, dict) else {},
                "difference": _first_value_diff(left, right),
            }
    if len(before) != len(after):
        return {
            "index": min(len(before), len(after)),
            "before": before[len(after)] if len(before) > len(after) else None,
            "after": after[len(before)] if len(after) > len(before) else None,
            "before_length": len(before),
            "after_length": len(after),
        }
    return None


@contextmanager
def instrument(metrics: RegionMetrics) -> Iterator[None]:
    original_wrapper = factory_module.Stage25StripExecutorAgent.__call__
    original_controller = StripExecutorController.act
    original_build = StripExecutorController._build_work_plan
    original_controller_diagnostics = StripExecutorController._diagnostics
    original_work_diagnostics = work_module._diagnostics
    original_deepcopy = copy.deepcopy
    original_executor_diagnostics = (
        factory_module.Stage25StripExecutorAgent.diagnostics_json)
    original_provider_diagnostics = Stage25PlanProvider.diagnostics_json

    def timed_wrapper(agent, obs):
        started = time.perf_counter()
        metrics._wrapper_depth += 1
        try:
            return original_wrapper(agent, obs)
        finally:
            metrics._wrapper_depth -= 1
            metrics.wrapper_seconds.append(time.perf_counter() - started)

    def timed_controller(controller, obs, plan):
        started = time.perf_counter()
        try:
            return original_controller(controller, obs, plan)
        finally:
            metrics.controller_seconds.append(time.perf_counter() - started)

    def timed_build(controller, obs, plan):
        started = time.perf_counter()
        try:
            result = original_build(controller, obs, plan)
        finally:
            metrics.work_build_seconds.append(time.perf_counter() - started)
        sample_key = (int(controller.config.acting_seat), int(obs.get("step", 0)))
        if (len(metrics.work_plan_samples) < 24
                and sample_key not in metrics._work_plan_sample_keys):
            metrics._work_plan_sample_keys.add(sample_key)
            metrics.work_plan_samples.append({
                "seat": int(controller.config.acting_seat),
                "step": int(obs.get("step", 0)),
                "day": int(obs.get("day", 0)),
                "hour": int(obs.get("hour", 0)),
                "plan": result.to_json_dict(),
            })
        return result

    def timed_controller_diagnostics(controller):
        started = time.perf_counter()
        metrics._controller_diagnostic_depth += 1
        try:
            return original_controller_diagnostics(controller)
        finally:
            metrics._controller_diagnostic_depth -= 1
            metrics.controller_diagnostic_seconds.append(
                time.perf_counter() - started)

    def timed_work_diagnostics(*args, **kwargs):
        started = time.perf_counter()
        try:
            return original_work_diagnostics(*args, **kwargs)
        finally:
            metrics.work_diagnostic_seconds.append(time.perf_counter() - started)

    def timed_deepcopy(value, memo=None, _nil=[]):  # noqa: B006 - copy signature
        if not metrics._wrapper_depth and not metrics._executor_diagnostics_depth:
            return original_deepcopy(value, memo, _nil)
        started = time.perf_counter()
        is_full_diagnostic = (
            isinstance(value, dict)
            and "routes_finalized" in value
            and "market_diagnostics" in value
        )
        try:
            return original_deepcopy(value, memo, _nil)
        finally:
            elapsed = time.perf_counter() - started
            if metrics._wrapper_depth:
                metrics.wrapper_deepcopy_seconds.append(elapsed)
                if is_full_diagnostic:
                    metrics.full_diagnostic_deepcopy_seconds.append(elapsed)
                    metrics.full_diagnostic_copy_count += 1
                    if metrics.diagnostic_sample is None:
                        metrics.diagnostic_sample = value

    def timed_executor_diagnostics(agent):
        started = time.perf_counter()
        metrics._executor_diagnostics_depth += 1
        try:
            return original_executor_diagnostics(agent)
        finally:
            metrics._executor_diagnostics_depth -= 1
            metrics.executor_diagnostics_seconds.append(time.perf_counter() - started)

    def timed_provider_diagnostics(provider):
        started = time.perf_counter()
        try:
            return original_provider_diagnostics(provider)
        finally:
            metrics.provider_diagnostics_seconds.append(time.perf_counter() - started)

    def json_timer(owner: str, original):
        def timed(value):
            if not metrics._controller_diagnostic_depth:
                return original(value)
            started = time.perf_counter()
            try:
                return original(value)
            finally:
                metrics.diagnostic_json_seconds.append(time.perf_counter() - started)
                metrics.diagnostic_json_calls[owner] = (
                    metrics.diagnostic_json_calls.get(owner, 0) + 1)

        return timed

    with ExitStack() as stack:
        stack.enter_context(patch.object(
            factory_module.Stage25StripExecutorAgent, "__call__", timed_wrapper))
        stack.enter_context(patch.object(
            StripExecutorController, "act", timed_controller))
        stack.enter_context(patch.object(
            StripExecutorController, "_build_work_plan", timed_build))
        stack.enter_context(patch.object(
            StripExecutorController, "_diagnostics", timed_controller_diagnostics))
        stack.enter_context(patch.object(
            work_module, "_diagnostics", timed_work_diagnostics))
        stack.enter_context(patch.object(
            factory_module.Stage25StripExecutorAgent,
            "diagnostics_json", timed_executor_diagnostics))
        stack.enter_context(patch.object(
            Stage25PlanProvider, "diagnostics_json", timed_provider_diagnostics))
        stack.enter_context(patch.object(copy, "deepcopy", timed_deepcopy))
        for owner, cls in (
            ("route", StripRoute),
            ("supply_plan", RouteSupplyPlan),
            ("supply_state", RouteSupplyState),
            ("market_state", MarketBootstrapState),
        ):
            stack.enter_context(patch.object(
                cls,
                "to_json_dict",
                json_timer(owner, cls.to_json_dict),
            ))
        yield


def summarize(metrics: RegionMetrics, *, games: int) -> dict[str, Any]:
    controller = _timed_ms(metrics.controller_seconds)
    wrapper = _timed_ms(metrics.wrapper_seconds)
    build = _timed_ms(metrics.work_build_seconds)
    controller_diagnostics = _timed_ms(metrics.controller_diagnostic_seconds)
    wrapper_deepcopy = _timed_ms(metrics.wrapper_deepcopy_seconds)
    work_diagnostics = _timed_ms(metrics.work_diagnostic_seconds)
    diagnostic_json = _timed_ms(metrics.diagnostic_json_seconds)
    full_diagnostic_deepcopy = _timed_ms(
        metrics.full_diagnostic_deepcopy_seconds)
    executor_diagnostics = _timed_ms(metrics.executor_diagnostics_seconds)
    provider_diagnostics = _timed_ms(metrics.provider_diagnostics_seconds)
    controller_total = float(controller["total_seconds"])
    wrapper_total = float(wrapper["total_seconds"])
    return {
        "controller": controller,
        "wrapper": wrapper,
        "work_plan_build": build,
        "controller_diagnostics": controller_diagnostics,
        "wrapper_deepcopy": wrapper_deepcopy,
        "full_diagnostic_deepcopy": full_diagnostic_deepcopy,
        "full_diagnostic_copy_count": metrics.full_diagnostic_copy_count,
        "executor_diagnostics_json": executor_diagnostics,
        "provider_diagnostics_json": provider_diagnostics,
        "work_plan_diagnostics": work_diagnostics,
        "diagnostic_to_json": {
            **diagnostic_json,
            "calls_by_type": dict(sorted(metrics.diagnostic_json_calls.items())),
        },
        "controller_ms_per_call_median": (
            statistics.median(metrics.controller_seconds) * 1000.0
            if metrics.controller_seconds else 0.0
        ),
        "wrapper_ms_per_call_median": (
            statistics.median(metrics.wrapper_seconds) * 1000.0
            if metrics.wrapper_seconds else 0.0
        ),
        "wrapper_ms_per_two_seat_game": (
            wrapper_total * 1000.0 / games if games else 0.0
        ),
        "work_plan_percent_of_controller": (
            100.0 * float(build["total_seconds"]) / controller_total
            if controller_total else 0.0
        ),
        "controller_diagnostics_percent_of_controller": (
            100.0 * float(controller_diagnostics["total_seconds"])
            / controller_total if controller_total else 0.0
        ),
        "wrapper_deepcopy_percent_of_wrapper": (
            100.0 * float(wrapper_deepcopy["total_seconds"]) / wrapper_total
            if wrapper_total else 0.0
        ),
        "outside_builder_percent_of_controller": (
            100.0 * (controller_total - float(build["total_seconds"]))
            / controller_total if controller_total else 0.0
        ),
    }


def _object_graph_size(value: Any) -> tuple[int, int]:
    pending = [value]
    seen: set[int] = set()
    total_bytes = 0
    while pending:
        current = pending.pop()
        identity = id(current)
        if identity in seen:
            continue
        seen.add(identity)
        total_bytes += sys.getsizeof(current)
        if isinstance(current, dict):
            pending.extend(current.keys())
            pending.extend(current.values())
        elif isinstance(current, (list, tuple, set, frozenset)):
            pending.extend(current)
    return len(seen), total_bytes


def _trace_plans(results) -> list[dict[str, Any]]:
    plans = []
    for result in results:
        if result.rollout is None:
            continue
        for (seat, day), plan in sorted(result.rollout.plans.items()):
            plans.append({
                "seed": int(result.seed),
                "seat": int(seat),
                "day": int(day),
                "plan": plan,
            })
    return plans


def _work_plan_without_diagnostics(
        samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for sample in samples:
        record = dict(sample)
        plan = dict(sample["plan"])
        plan.pop("diagnostics", None)
        record["plan"] = plan
        normalized.append(record)
    return normalized


def _game_outcomes(results) -> list[dict[str, Any]]:
    return [{
        "seed": int(result.seed),
        "statuses": list(result.statuses),
        "final_banks": list(result.final_banks),
        "transitions": int(result.transitions),
        "terminated": bool(result.terminated),
        "trace_digest": result.trace_digest,
    } for result in results]


def _diagnostic_snapshot_summary(metrics: RegionMetrics) -> dict[str, int] | None:
    if metrics.diagnostic_sample is None:
        return None
    snapshot = metrics.diagnostic_sample
    object_count, estimated_bytes = _object_graph_size(snapshot)
    return {
        "object_count": object_count,
        "estimated_python_bytes": estimated_bytes,
        "json_bytes": len(json.dumps(
            snapshot, allow_nan=False, separators=(",", ":")).encode("utf-8")),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-turns", type=int, default=720)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-extension-dir", type=Path)
    parser.add_argument("--low-telemetry", action="store_true")
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args(argv)
    if args.repeats < 3:
        parser.error("--repeats must be at least 3")
    native_extension = None
    if args.native_extension_dir is not None:
        import hashlib
        import fast_env

        native_dir = args.native_extension_dir.resolve()
        fast_env.__path__.append(str(native_dir))
        native_files = tuple(native_dir.glob("_kaggriculture_env*.pyd"))
        if not native_files:
            parser.error(f"no Windows fast engine extension found in {native_dir}")
        native_extension = {
            "path": str(native_files[0]),
            "sha256": hashlib.sha256(native_files[0].read_bytes()).hexdigest(),
        }
    if any(name == "jax" or name.startswith("jax.") for name in sys.modules):
        parser.error("JAX was imported before the CPU-only benchmark started")

    run_games(args.seeds[:1], max_turns=min(args.max_turns, 96),
              low_telemetry=args.low_telemetry)
    aggregate = RegionMetrics()
    retained_actions = None
    mismatches = []
    repetitions = []
    for _ in range(args.repeats):
        current = RegionMetrics()
        with instrument(current):
            results = run_games(
                args.seeds, max_turns=args.max_turns,
                low_telemetry=args.low_telemetry)
        current_actions = action_trace(args.seeds, results)
        if retained_actions is None:
            retained_actions = current_actions
        else:
            mismatch = first_mismatch(retained_actions, current_actions)
            if mismatch is not None:
                mismatches.append(mismatch)
        repetitions.append(summarize(current, games=len(results)))
        for name in (
            "controller_seconds",
            "wrapper_seconds",
            "work_build_seconds",
            "controller_diagnostic_seconds",
            "wrapper_deepcopy_seconds",
            "work_diagnostic_seconds",
            "diagnostic_json_seconds",
            "full_diagnostic_deepcopy_seconds",
            "executor_diagnostics_seconds",
            "provider_diagnostics_seconds",
        ):
            getattr(aggregate, name).extend(getattr(current, name))
        for name, count in current.diagnostic_json_calls.items():
            aggregate.diagnostic_json_calls[name] = (
                aggregate.diagnostic_json_calls.get(name, 0) + count)
        aggregate.full_diagnostic_copy_count += current.full_diagnostic_copy_count
        if aggregate.diagnostic_sample is None:
            aggregate.diagnostic_sample = current.diagnostic_sample
        if not aggregate.work_plan_samples:
            aggregate.work_plan_samples = current.work_plan_samples

    payload = {
        "schema_version": 1,
        "config": {
            "seeds": args.seeds,
            "repeats": args.repeats,
            "max_turns": args.max_turns,
            "backend": "fast-scalar-cpu",
            "num_threads": 1,
            "jax": False,
            "telemetry": "low" if args.low_telemetry else "full",
            "native_extension": native_extension,
        },
        "summary": summarize(
            aggregate, games=len(args.seeds) * args.repeats),
        "repetitions": repetitions,
        "repeat_action_mismatches": mismatches,
        "action_trace": retained_actions,
        "manager_plan_trace": _trace_plans(results),
        "game_outcomes": _game_outcomes(results),
        "work_plan_samples": aggregate.work_plan_samples,
        "normalized_work_plan_samples": _work_plan_without_diagnostics(
            aggregate.work_plan_samples),
        "full_diagnostic_snapshot_estimate": _diagnostic_snapshot_summary(aggregate),
    }
    if args.compare is not None:
        baseline = json.loads(args.compare.read_text(encoding="utf-8"))
        baseline_config = baseline.get("config", {})
        for key, expected in (
            ("seeds", args.seeds),
            ("max_turns", args.max_turns),
            ("backend", "fast-scalar-cpu"),
            ("num_threads", 1),
        ):
            if key in baseline_config and baseline_config[key] != expected:
                parser.error(
                    f"--compare config mismatch for {key}: "
                    f"{baseline_config[key]!r} != {expected!r}")
        baseline_normalized_work = baseline.get("normalized_work_plan_samples")
        if baseline_normalized_work is None and "work_plan_samples" in baseline:
            baseline_normalized_work = _work_plan_without_diagnostics(
                baseline["work_plan_samples"])
        parity_sources = {
            "action_trace": baseline.get("action_trace"),
            "manager_plan_trace": baseline.get("manager_plan_trace"),
            "game_outcomes": baseline.get("game_outcomes"),
            "work_plan_samples": (
                None if args.low_telemetry
                else baseline.get("work_plan_samples")
            ),
            "normalized_work_plan_samples": baseline_normalized_work,
        }
        payload["parity"] = {}
        for key, before in parity_sources.items():
            after = payload.get(key)
            available = before is not None
            payload["parity"][key] = {
                "available": available,
                "equal": (
                    json.loads(json.dumps(before, allow_nan=False))
                    == json.loads(json.dumps(after, allow_nan=False))
                    if available else None
                ),
                "first_mismatch": (
                    first_mismatch(before, after)
                    if available and after is not None else None
                ),
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "config": payload["config"],
        "summary": payload["summary"],
        "repeat_action_mismatches": mismatches,
        "parity": payload.get("parity"),
        "output": str(args.output),
    }, indent=2, sort_keys=True, allow_nan=False))
    failed_parity = any(
        result["available"] and not result["equal"]
        for result in payload.get("parity", {}).values())
    return 3 if mismatches else 2 if failed_parity else 0


if __name__ == "__main__":
    raise SystemExit(main())

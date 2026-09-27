"""Deterministic CPU-only benchmark and parity harness for strip Round 1.

The harness deliberately uses a scripted NumPy policy and the scalar fast
backend: it never imports JAX or loads model weights.  It retains the complete
ordered action trace for the measured seeds plus normalized work-plan samples
from representative manager-day calls.
"""

from __future__ import annotations

import argparse
import json
import hashlib
import statistics
import subprocess
import sys
import time
from collections.abc import Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from executor_v0.plan import DailyPlan  # noqa: E402
from executor_v0.strip_executor import StripExecutorController  # noqa: E402
from rl_manager.decode import ACTION_TENSOR_SHAPES  # noqa: E402
from rl_manager.executor_factory import make_stage25_executor_factory  # noqa: E402
from rl_manager.runner import RunnerConfig, SelfPlayRunner, build_episode_spec  # noqa: E402
from rl_manager.types import E_VS_E, PolicyIdentity, PolicyOutputs  # noqa: E402


DEFAULT_SEEDS = (17, 42, 2026)


class ScriptedPolicy:
    """A deterministic, accelerator-free manager policy for executor timing."""

    def __init__(self) -> None:
        self.identity = PolicyIdentity(
            name="strip-round1-scripted",
            version="v1",
            fingerprint="strip-round1-scripted-v1",
        )

    def plan_batch(self, inputs, prng_id):
        del prng_id
        batch_size = int(np.asarray(inputs["board_kind"]).shape[0])
        action_tensors = {
            name: np.zeros((batch_size,) + shape, dtype=np.int16)
            for name, shape in ACTION_TENSOR_SHAPES.items()
        }
        action_tensors["land"] = np.ones((batch_size,), dtype=np.int16)
        zeros = np.zeros(batch_size, dtype=np.float32)
        return PolicyOutputs(
            action_tensors=action_tensors,
            logprob_groups={
                group: zeros.copy()
                for group in (
                    "crop",
                    "animal",
                    "land",
                    "fertilizer",
                    "care",
                    "sell_presence",
                )
            },
            logprob_total=zeros.copy(),
            value=zeros.copy(),
            batch_size=batch_size,
        )


@dataclass
class Metrics:
    strip_seconds: list[float] = field(default_factory=list)
    work_build_seconds: list[float] = field(default_factory=list)
    sales_view_accesses: int = 0
    animal_layout_calls: int = 0
    zero_deficit_animal_layout_calls: int = 0
    zero_deficit_animal_fast_path_hits: int = 0
    animal_layout_candidate_checks: int = 0
    finalization_calls: int = 0
    finalization_work_builds: int = 0
    _finalization_depth: int = 0
    _work_build_depth: int = 0
    _animal_layout_depth: int = 0
    row_key_lookups: int = 0
    row_key_object_constructions: int = 0
    _row_key_ids: set[int] = field(default_factory=set)
    _row_key_references: list[Any] = field(default_factory=list)
    _sample_keys: set[tuple[int, int]] = field(default_factory=set)
    work_plan_samples: list[dict[str, Any]] = field(default_factory=list)


@contextmanager
def instrument(metrics: Metrics, *, sample_limit: int = 0):
    import executor_v0.layout as layout_module
    import executor_v0.strip_work as work_module

    original_act = StripExecutorController.act
    original_build = StripExecutorController._build_work_plan
    original_finalize = StripExecutorController._finalize_day
    original_sales_view = DailyPlan.sell_quantities_dict.fget
    original_animal_layout = layout_module.plan_animal_layout
    original_in_unlocked = layout_module._in_unlocked
    original_row_key = work_module.row_key_for_tile
    precomputed_row_key_ids = {
        id(row_key)
        for row in getattr(work_module, "_ROW_KEYS", ())
        for row_key in row
    }

    def timed_act(controller, obs, plan):
        started = time.perf_counter()
        try:
            return original_act(controller, obs, plan)
        finally:
            metrics.strip_seconds.append(time.perf_counter() - started)

    def timed_build(controller, obs, plan):
        started = time.perf_counter()
        metrics._work_build_depth += 1
        try:
            result = original_build(controller, obs, plan)
        finally:
            metrics._work_build_depth -= 1
            metrics.work_build_seconds.append(time.perf_counter() - started)
            if metrics._finalization_depth:
                metrics.finalization_work_builds += 1
        sample_key = (
            int(controller.config.acting_seat),
            int(obs.get("step", 0)),
        )
        if (sample_limit
                and len(metrics.work_plan_samples) < sample_limit
                and sample_key not in metrics._sample_keys):
            metrics._sample_keys.add(sample_key)
            metrics.work_plan_samples.append({
                "seat": int(controller.config.acting_seat),
                "step": int(obs.get("step", 0)),
                "day": int(obs.get("day", 0)),
                "hour": int(obs.get("hour", 0)),
                "plan": result.to_json_dict(),
            })
        return result

    def timed_finalize(controller, obs, plan, *args, **kwargs):
        metrics.finalization_calls += 1
        metrics._finalization_depth += 1
        try:
            return original_finalize(controller, obs, plan, *args, **kwargs)
        finally:
            metrics._finalization_depth -= 1

    def counted_sales_view(plan):
        if metrics._work_build_depth:
            metrics.sales_view_accesses += 1
        return original_sales_view(plan)

    def counted_animal_layout(*args, **kwargs):
        metrics.animal_layout_calls += 1
        needed = kwargs["animals_needed"]
        zero_deficit = all(
            int(needed.get(animal, 0)) == 0
            for animal in ("GOOSE", "COW", "SHEEP")
        )
        if zero_deficit:
            metrics.zero_deficit_animal_layout_calls += 1
        checks_before = metrics.animal_layout_candidate_checks
        metrics._animal_layout_depth += 1
        try:
            result = original_animal_layout(*args, **kwargs)
        finally:
            metrics._animal_layout_depth -= 1
        if (zero_deficit
                and checks_before == metrics.animal_layout_candidate_checks):
            metrics.zero_deficit_animal_fast_path_hits += 1
        return result

    def counted_in_unlocked(*args, **kwargs):
        if metrics._animal_layout_depth:
            metrics.animal_layout_candidate_checks += 1
        return original_in_unlocked(*args, **kwargs)

    def counted_row_key(tile):
        row_key = original_row_key(tile)
        metrics.row_key_lookups += 1
        # Keep references so CPython cannot reuse an id during the sample.
        metrics._row_key_references.append(row_key)
        key_id = id(row_key)
        if (key_id not in precomputed_row_key_ids
                and key_id not in metrics._row_key_ids):
            metrics._row_key_ids.add(key_id)
            metrics.row_key_object_constructions += 1
        return row_key

    with ExitStack() as stack:
        stack.enter_context(patch.object(StripExecutorController, "act", timed_act))
        stack.enter_context(patch.object(
            StripExecutorController, "_build_work_plan", timed_build))
        stack.enter_context(patch.object(
            StripExecutorController, "_finalize_day", timed_finalize))
        stack.enter_context(patch.object(
            DailyPlan, "sell_quantities_dict", property(counted_sales_view)))
        stack.enter_context(patch.object(
            layout_module, "plan_animal_layout", counted_animal_layout))
        stack.enter_context(patch.object(
            layout_module, "_in_unlocked", counted_in_unlocked))
        stack.enter_context(patch.object(
            work_module, "row_key_for_tile", counted_row_key))
        yield


def run_games(
    seeds: Sequence[int], *, max_turns: int, low_telemetry: bool = False,
    trajectory_buffer: Any | None = None,
):
    policy = ScriptedPolicy()
    runner = SelfPlayRunner(
        RunnerConfig(
            backend_name="fast",
            backend_configuration={"seed": 0, "numThreads": 1},
            max_turns=max_turns,
            num_envs=1,
            record_rollout=True,
            low_telemetry=low_telemetry,
        ),
        trajectory_buffer=trajectory_buffer,
        executor_factory=make_stage25_executor_factory(),
        master_seed=25,
    )
    specs = [
        build_episode_spec(index, int(seed), E_VS_E, policy, policy)
        for index, seed in enumerate(seeds)
    ]
    return runner.run(specs)


def action_trace(seeds: Sequence[int], results) -> list[dict[str, Any]]:
    trace = []
    for seed, result in zip(seeds, results):
        for step, day, hour, action0, action1 in result.rollout.joint_actions:
            trace.append({
                "seed": int(seed),
                "seat": 0,
                "step": int(step),
                "day": int(day),
                "hour": int(hour),
                "action": action0,
            })
            trace.append({
                "seed": int(seed),
                "seat": 1,
                "step": int(step),
                "day": int(day),
                "hour": int(hour),
                "action": action1,
            })
    return trace


def summarize(metrics: Metrics, *, games: int) -> dict[str, Any]:
    strip_ms = [seconds * 1000.0 for seconds in metrics.strip_seconds]
    build_seconds = sum(metrics.work_build_seconds)
    builds = len(metrics.work_build_seconds)
    return {
        "strip_calls": len(strip_ms),
        "work_plan_builds": builds,
        "build_strip_work_plan_seconds": build_seconds,
        "build_strip_work_plan_ms_per_build": (
            build_seconds * 1000.0 / builds if builds else 0.0
        ),
        "strip_ms_per_call_mean": statistics.fmean(strip_ms) if strip_ms else 0.0,
        "strip_ms_per_call_median": statistics.median(strip_ms) if strip_ms else 0.0,
        "strip_ms_per_game": sum(strip_ms) / games if games else 0.0,
        "sales_view_accesses": metrics.sales_view_accesses,
        "sales_view_accesses_per_build": (
            metrics.sales_view_accesses / builds if builds else 0.0
        ),
        "animal_layout_calls": metrics.animal_layout_calls,
        "zero_deficit_animal_layout_calls": metrics.zero_deficit_animal_layout_calls,
        "zero_deficit_animal_fast_path_hits": (
            metrics.zero_deficit_animal_fast_path_hits),
        "animal_layout_candidate_checks": metrics.animal_layout_candidate_checks,
        "row_key_lookups": metrics.row_key_lookups,
        "row_key_object_constructions": metrics.row_key_object_constructions,
        "finalization_calls": metrics.finalization_calls,
        "finalization_work_builds": metrics.finalization_work_builds,
    }


def first_mismatch(before: list[Any], after: list[Any]) -> dict[str, Any] | None:
    for index, (left, right) in enumerate(zip(before, after)):
        if left != right:
            return {"index": index, "before": left, "after": right}
    if len(before) != len(after):
        return {
            "index": min(len(before), len(after)),
            "before": before[min(len(before), len(after)):] or None,
            "after": after[min(len(before), len(after)):] or None,
        }
    return None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-turns", type=int, default=720)
    parser.add_argument("--sample-limit", type=int, default=24)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--native-extension-dir", type=Path)
    args = parser.parse_args(argv)
    if args.repeats < 3:
        parser.error("--repeats must be at least 3")
    native_extension = None
    if args.native_extension_dir is not None:
        import fast_env

        native_dir = args.native_extension_dir.resolve()
        if not native_dir.is_dir():
            parser.error(f"native extension directory does not exist: {native_dir}")
        fast_env.__path__.append(str(native_dir))
        native_files = tuple(native_dir.glob("_kaggriculture_env*.pyd"))
        if not native_files:
            parser.error(f"no Windows fast engine extension found in {native_dir}")
        native_extension = {
            "path": str(native_files[0]),
            "sha256": hashlib.sha256(native_files[0].read_bytes()).hexdigest(),
        }
    if any(name == "jax" or name.startswith("jax.") for name in sys.modules):
        parser.error("JAX was imported before the CPU-only strip benchmark started")

    # Warm imports/native bindings and capture representative normalized plans
    # outside the measured repetitions.
    run_games(args.seeds[:1], max_turns=min(args.max_turns, 96))
    sample_metrics = Metrics()
    with instrument(sample_metrics, sample_limit=args.sample_limit):
        run_games(args.seeds[:1], max_turns=min(args.max_turns, 145))
    normalized_work_plan_samples = json.loads(json.dumps(
        sample_metrics.work_plan_samples,
        sort_keys=True,
        allow_nan=False,
    ))

    repetitions = []
    retained_actions = None
    repeat_action_parity = []
    repeat_action_mismatches = []
    aggregate = Metrics()
    for _ in range(args.repeats):
        current = Metrics()
        started = time.perf_counter()
        with instrument(current):
            results = run_games(args.seeds, max_turns=args.max_turns)
        wall_seconds = time.perf_counter() - started
        repetitions.append({
            "wall_seconds": wall_seconds,
            **summarize(current, games=len(results)),
        })
        aggregate.strip_seconds.extend(current.strip_seconds)
        aggregate.work_build_seconds.extend(current.work_build_seconds)
        aggregate.sales_view_accesses += current.sales_view_accesses
        aggregate.animal_layout_calls += current.animal_layout_calls
        aggregate.zero_deficit_animal_layout_calls += (
            current.zero_deficit_animal_layout_calls)
        aggregate.zero_deficit_animal_fast_path_hits += (
            current.zero_deficit_animal_fast_path_hits)
        aggregate.animal_layout_candidate_checks += (
            current.animal_layout_candidate_checks)
        aggregate.row_key_lookups += current.row_key_lookups
        aggregate.row_key_object_constructions += (
            current.row_key_object_constructions)
        aggregate.finalization_calls += current.finalization_calls
        aggregate.finalization_work_builds += current.finalization_work_builds
        current_actions = action_trace(args.seeds, results)
        if retained_actions is None:
            retained_actions = current_actions
        else:
            mismatch = first_mismatch(retained_actions, current_actions)
            repeat_action_parity.append(mismatch is None)
            if mismatch is not None:
                repeat_action_mismatches.append(mismatch)

    payload: dict[str, Any] = {
        "schema_version": 1,
        "source_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True
        ).strip(),
        "config": {
            "seeds": args.seeds,
            "repeats": args.repeats,
            "max_turns": args.max_turns,
            "backend": "fast-scalar-cpu",
            "num_threads": 1,
            "jax": False,
            "native_extension": native_extension,
        },
        "summary": summarize(
            aggregate, games=len(args.seeds) * args.repeats),
        "repetitions": repetitions,
        "repeat_action_trace_parity": repeat_action_parity,
        "repeat_action_mismatches": repeat_action_mismatches,
        "action_trace": retained_actions,
        "work_plan_samples": normalized_work_plan_samples,
    }
    exit_code = 0
    if repeat_action_mismatches:
        exit_code = 3
    if args.compare is not None:
        baseline = json.loads(args.compare.read_text(encoding="utf-8"))
        action_mismatch = first_mismatch(
            baseline["action_trace"], payload["action_trace"])
        work_plan_mismatch = first_mismatch(
            baseline["work_plan_samples"], payload["work_plan_samples"])
        payload["parity"] = {
            "actions_equal": action_mismatch is None,
            "work_plans_equal": work_plan_mismatch is None,
            "first_action_mismatch": action_mismatch,
            "first_work_plan_mismatch": work_plan_mismatch,
        }
        if action_mismatch is not None or work_plan_mismatch is not None:
            exit_code = 2

    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text + "\n", encoding="utf-8")
    print(json.dumps({
        "source_sha": payload["source_sha"],
        "config": payload["config"],
        "summary": payload["summary"],
        "repetitions": payload["repetitions"],
        "parity": payload.get("parity"),
        "output": str(args.output),
    }, indent=2, sort_keys=True, allow_nan=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

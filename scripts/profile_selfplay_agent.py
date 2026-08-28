"""Profile the CPU-side self-play agent/executor path on complete games.

The profiler is intentionally external to ``SelfPlayRunner``. It wraps named
hot-path call sites at runtime, runs fixed-seed BC-E self-play, and emits JSON
with wall time, runner timing buckets, manually instrumented sub-buckets, and
optional cProfile function rows.

Example::

    python scripts/profile_selfplay_agent.py \
      --checkpoint C:/path/to/bc-v1-E/best.pt --num-envs 1 2
"""

from __future__ import annotations

import argparse
import cProfile
import copy as stdlib_copy
import functools
import json
import os
import pstats
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

# ``python scripts/...`` puts ``scripts`` rather than the repository root at
# the front of sys.path on some Python versions.
def _root_argument() -> Path | None:
    for index, value in enumerate(sys.argv[1:]):
        if value == "--repo-root" and index + 2 <= len(sys.argv[1:]):
            return Path(sys.argv[index + 2])
    return None


REPOSITORY_ROOT = _root_argument() or Path(
    os.environ.get("KAGGRICULTURE_PROFILE_ROOT", Path(__file__).resolve().parents[1]))
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from bc_manager_jax.checkpoint import load_torch_checkpoint  # noqa: E402
from bc_manager_jax.model import ManagerConfig  # noqa: E402
from rl_manager.policy import JaxEPlanPolicy  # noqa: E402
from rl_manager.runner import (  # noqa: E402
    RunnerConfig,
    SelfPlayRunner,
    build_episode_spec,
)
from rl_manager.types import E_VS_E  # noqa: E402


DEFAULT_SEEDS = (17, 42, 2026, 7, 123, 1013, 1022, 1003)


class Timings:
    def __init__(self) -> None:
        self.seconds: dict[str, float] = defaultdict(float)
        self.calls: dict[str, int] = defaultdict(int)

    def wrapper(self, name: str, function: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(function)
        def timed(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                self.seconds[name] += time.perf_counter() - start
                self.calls[name] += 1

        return timed

    def rows(self, primitive_turns: int) -> dict[str, dict[str, float | int]]:
        return {
            name: {
                "seconds": self.seconds[name],
                "calls": self.calls[name],
                "milliseconds_per_turn": (
                    self.seconds[name] * 1000.0 / primitive_turns
                    if primitive_turns else 0.0
                ),
            }
            for name in sorted(self.seconds)
        }


@contextmanager
def instrument_hot_path(timings: Timings):
    """Wrap profiler-only call sites without changing production code."""
    import executor_v0.agent as agent_module
    import executor_v0.tasks as tasks_module
    import opening_book.agent as opening_module
    import rl_manager.runner as runner_module

    targets = (
        (opening_module.OpeningAgent, "__call__", "opening_wrapper"),
        (opening_module.OpeningAgent, "_delegate", "opening_delegate"),
        (agent_module.ExecutorAgent, "__call__", "executor_call"),
        (agent_module.ExecutorAgent, "_new_day", "executor_new_day"),
        (agent_module.ExecutorAgent, "_sell_candidates", "market_sell"),
        (agent_module.ExecutorAgent, "_hire_orders", "market_hire"),
        (agent_module.ExecutorAgent, "_buy_order_cost", "market_buy_cost"),
        (agent_module.ExecutorAgent, "_record_cleanup_telemetry", "bookkeeping_cleanup"),
        (agent_module.ExecutorAgent, "_pending_task_keys", "bookkeeping_pending"),
        (agent_module.ExecutorAgent, "_build_debug_trace_turn", "diagnostics_turn_snapshot"),
    )
    with ExitStack() as stack:
        for owner, attribute, name in targets:
            original = getattr(owner, attribute)
            wrapped = timings.wrapper(name, original)
            descriptor = owner.__dict__.get(attribute)
            if isinstance(descriptor, staticmethod):
                wrapped = staticmethod(wrapped)
            stack.enter_context(patch.object(owner, attribute, wrapped))

        for module, attribute, name in (
            (agent_module, "generate_tasks", "task_generation"),
            (agent_module, "run_foreman", "foreman_dispatch"),
            (agent_module, "canonical_board", "canonical_board_agent"),
            (tasks_module, "canonical_board", "canonical_board_tasks"),
            (tasks_module, "plan_day_layouts", "layout_day_planning"),
        ):
            original = getattr(module, attribute)
            stack.enter_context(patch.object(
                module, attribute, timings.wrapper(name, original)))

        # Isolate only runner-owned observation/trace deep copies. Replacing
        # the runner's module reference avoids modifying the stdlib copy module.
        copy_proxy = SimpleNamespace(
            deepcopy=timings.wrapper(
                "runner_deepcopy", stdlib_copy.deepcopy))
        stack.enter_context(patch.object(runner_module, "copy", copy_proxy))
        yield


def _profile_rows(profile: cProfile.Profile, limit: int) -> list[dict[str, Any]]:
    stats = pstats.Stats(profile)
    rows = []
    for (filename, line, function), values in sorted(
            stats.stats.items(), key=lambda item: item[1][3], reverse=True)[:limit]:
        primitive_calls, total_calls, self_seconds, cumulative_seconds, _callers = values
        rows.append({
            "file": filename,
            "line": line,
            "function": function,
            "primitive_calls": primitive_calls,
            "total_calls": total_calls,
            "self_seconds": self_seconds,
            "cumulative_seconds": cumulative_seconds,
        })
    return rows


def run_case(
    policy: JaxEPlanPolicy,
    *,
    num_envs: int,
    seeds: Sequence[int],
    include_cprofile: bool,
    profile_limit: int,
) -> dict[str, Any]:
    selected_seeds = tuple(int(seed) for seed in seeds[:num_envs])
    if len(selected_seeds) != num_envs:
        raise ValueError(f"need at least {num_envs} fixed seeds")
    runner = SelfPlayRunner(RunnerConfig(
        backend_name="fast",
        backend_configuration={"seed": 0, "numThreads": 1},
        num_envs=num_envs,
    ))
    specs = [
        build_episode_spec(index, seed, E_VS_E, policy, policy)
        for index, seed in enumerate(selected_seeds)
    ]
    timings = Timings()
    profile = cProfile.Profile() if include_cprofile else None
    start = time.perf_counter()
    with instrument_hot_path(timings):
        if profile is not None:
            profile.enable()
        results = runner.run(specs)
        if profile is not None:
            profile.disable()
    steady_seconds = time.perf_counter() - start
    primitive_turns = sum(719 if result.terminated else 0 for result in results)
    agent_seconds = runner.timing_totals["agent_actions"]
    return {
        "num_envs": num_envs,
        "seeds": list(selected_seeds),
        "games": len(results),
        "steady_seconds": steady_seconds,
        "games_per_second": len(results) / steady_seconds,
        "primitive_turns": primitive_turns,
        "primitive_turns_per_second": primitive_turns / steady_seconds,
        "agent_actions_seconds": agent_seconds,
        "agent_actions_share": agent_seconds / steady_seconds,
        "ordinary_executor_seconds": (
            timings.seconds["executor_call"]
            - timings.seconds["executor_new_day"]
        ),
        "runner_timing_seconds": dict(runner.timing_totals),
        "manual_timing": timings.rows(primitive_turns),
        "final_banks": [result.final_banks for result in results],
        "statuses": [result.statuses for result in results],
        "trace_digests": [result.trace_digest for result in results],
        # The default runner factory is strict, so any fallback would abort the
        # run rather than be silently recorded in this profiler.
        "fallback_errors": 0,
        "cprofile": _profile_rows(profile, profile_limit) if profile else [],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--profile-limit", type=int, default=40)
    parser.add_argument("--no-cprofile", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if any(value < 1 for value in args.num_envs):
        parser.error("--num-envs values must be >= 1")
    if max(args.num_envs) > len(args.seeds):
        parser.error("--seeds must contain at least max(--num-envs) values")

    # The root is resolved before package imports so this same helper can
    # profile a clean base worktree without copying or editing it.
    params, metadata = load_torch_checkpoint(args.checkpoint)
    config = ManagerConfig(**metadata["model_config"])
    policy = JaxEPlanPolicy(params, config, name="profile_bc_e")

    # Compile and initialize lazy imports outside measured complete games.
    warmup = run_case(
        policy, num_envs=1, seeds=args.seeds,
        include_cprofile=False, profile_limit=0)
    cases = [
        run_case(
            policy, num_envs=value, seeds=args.seeds,
            include_cprofile=not args.no_cprofile, profile_limit=args.profile_limit)
        for value in args.num_envs
    ]
    payload = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint.resolve()),
        "warmup": {
            "steady_seconds": warmup["steady_seconds"],
            "trace_digests": warmup["trace_digests"],
        },
        "cases": cases,
    }
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

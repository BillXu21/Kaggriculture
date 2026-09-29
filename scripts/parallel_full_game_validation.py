"""Run checkpoint-backed Stage 2.5 full-game matches concurrently.

Each spawned worker owns one independent 720-step fast-engine episode. A
literal built-in opening handles days 0-3, then the native Stage 2.5 policy
plans once per day from day 4 through the end of the game.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import hashlib
import importlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Iterable, Mapping, Sequence

from bc_manager.constants import TOTAL_DAYS
from evaluation.agent_match import MatchResult, run_match
from opening_book.agent import make_opening_agent
from executor_v0.strip_executor import StripExecutorConfig
from rl_manager.executor_factory import (
    STAGE25_EXECUTOR_PROFILE_VERSION,
    make_stage25_executor_factory,
)
from rl_manager.provenance import (
    backend_provenance,
    opening_provenance,
)
from rl_manager.stage25_provider import Stage25PlanProvider

FULL_GAME_TURNS = 719  # Reset is step 0 in a 720-step episode.
EPISODE_STEPS = FULL_GAME_TURNS + 1
MANAGER_START_DAY = 4
MANAGER_ACTIVE_DAYS = TOTAL_DAYS - MANAGER_START_DAY
OPENING_TURNS = MANAGER_START_DAY * 24
OPENING_NAME = "standard_mixed"
DEFAULT_SEEDS = (7,)
DEFAULT_WORKERS = 2
# "symmetric": the same checkpoint and executor configuration drive both seats,
# so one seed is one game and no seat-swap duplicate is needed.
# "pass": controller B is the PASS baseline; both seat orientations are run.
OPPONENT_SYMETRIC = "symmetric"
OPPONENT_PASS = "pass"
OPPONENT_MODES = (OPPONENT_SYMETRIC, OPPONENT_PASS)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_checkpoint_path(value: str | Path) -> Path:
    """Require one existing Stage 2.5 native inference checkpoint."""
    try:
        path = Path(value).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"checkpoint path is unavailable: {value!r}") from exc
    if not path.is_file():
        raise ValueError(f"checkpoint path is not a file: {path}")
    if path.suffix.lower() != ".npz":
        raise ValueError(
            "checkpoint must be a Stage 2.5 native inference .npz file")
    return path


class _StripController:
    observation_mode = "canonical"

    def __init__(
        self,
        *,
        seat: int,
        configuration: Mapping[str, Any],
        provider: Stage25PlanProvider,
        executor_factory: Any,
    ) -> None:
        self._agent = executor_factory.create(
            backend_name="fast",
            seat=seat,
            configuration=configuration,
            provider=provider,
        )
        self._opening = make_opening_agent(
            OPENING_NAME, downstream=self._agent, seat=seat)
        self._actions = 0
        self._observed_days: set[int] = set()
        self._manager_days: set[int] = set()
        self._sell_orders_by_product: dict[str, int] = {}

    def act(self, observation: Mapping[str, Any]) -> Mapping[str, Any]:
        day = int(observation["day"])
        hour = int(observation["hour"])
        action = self._opening(observation)
        self._actions += 1
        self._observed_days.add(day)
        if day >= MANAGER_START_DAY and hour == 0:
            self._manager_days.add(day)
        for order in (action.get("market") or ()):
            if order and order[0] == "SELL" and len(order) > 1:
                product = str(order[1])
                self._sell_orders_by_product[product] = (
                    self._sell_orders_by_product.get(product, 0) + 1
                )
        return action

    __call__ = act

    def diagnostics(self) -> dict[str, Any]:
        return {
            "opening": self._opening.diagnostics_json(),
            "executor": {
                "agent": self._agent.diagnostics_json(),
                "validation": {
                    "primitive_actions": self._actions,
                    "observed_days": sorted(self._observed_days),
                    "stage25_manager_days": sorted(self._manager_days),
                    "automatic_sell_orders_by_product": dict(
                        self._sell_orders_by_product
                    ),
                },
            },
        }

    def close(self) -> None:
        return None


class _StripControllerFactory:
    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        checkpoint_sha256: str,
        episode_index: int,
        seed: int,
        enable_row_claim_board: bool = False,
    ) -> None:
        self._checkpoint_path = Path(checkpoint_path)
        self._episode_index = episode_index
        self._seed = seed
        self._enable_row_claim_board = bool(enable_row_claim_board)
        self._strip_config = StripExecutorConfig(
            aggressive_sell_all=True,
            enable_row_claim_board=self._enable_row_claim_board,
        )
        self._executor_factory = make_stage25_executor_factory(self._strip_config)
        opening = opening_provenance(OPENING_NAME)
        self.provenance = {
            "display_name": "Stage 2.5 native checkpoint + strip executor",
            "kind": "stage25_native_inference_checkpoint",
            "identity": f"stage25-native-checkpoint-sha256:{checkpoint_sha256}",
            "checkpoint": {
                "path": str(self._checkpoint_path),
                "sha256": checkpoint_sha256,
            },
            "policy": {
                "provider": "rl_manager.stage25_provider.Stage25PlanProvider",
                "mode": "deterministic",
                "manager_start_day": MANAGER_START_DAY,
            },
            "opening": opening,
            "executor_profile": {
                **self._executor_factory.effective_profile,
                "version": STAGE25_EXECUTOR_PROFILE_VERSION,
                "enable_row_claim_board": self._enable_row_claim_board,
            },
            "execution_mode": "in_process",
        }

    def create(
        self, *, seat: int, configuration: Mapping[str, Any]
    ) -> _StripController:
        provider = Stage25PlanProvider(
            episode_id=(
                f"parallel-full-game-{self._episode_index}-seed-{self._seed}"
            ),
            seat=seat,
            manager_start_day=MANAGER_START_DAY,
            native_checkpoint=self._checkpoint_path,
            mode="deterministic",
            seed=self._seed,
        )
        # Load and strictly validate the native checkpoint during controller
        # construction, before spending a full game on an invalid artifact.
        provider.effective_curriculum()
        return _StripController(
            seat=seat,
            configuration=configuration,
            provider=provider,
            executor_factory=self._executor_factory,
        )


class _PassController:
    observation_mode = "raw"

    def act(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        seat = int(observation["player"])
        farm = observation["farms"][seat]
        return {
            "farmer": ["PASS"],
            "hands": [["PASS"] for _ in farm.get("hands", ())],
            "market": [],
        }

    __call__ = act

    def close(self) -> None:
        return None


class _PassControllerFactory:
    @property
    def provenance(self) -> Mapping[str, Any]:
        return {
            "display_name": "PASS",
            "kind": "pass",
            "execution_mode": "in_process",
            "identity": "PASS",
        }

    def create(
        self, *, seat: int, configuration: Mapping[str, Any]
    ) -> _PassController:
        del seat, configuration
        return _PassController()


def parse_seed_spec(value: str) -> tuple[int, ...]:
    """Parse comma-separated seeds and inclusive ascending ranges."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("at least one seed is required")
    seeds: list[int] = []
    try:
        for raw_token in value.split(","):
            token = raw_token.strip()
            if not token:
                raise ValueError("empty seed entry")
            if ".." in token:
                parts = token.split("..")
                if len(parts) != 2:
                    raise ValueError("seed ranges must use START..END")
                start, stop = (int(part) for part in parts)
                if stop < start:
                    raise ValueError("seed ranges must be ascending")
                seeds.extend(range(start, stop + 1))
            else:
                seeds.append(int(token))
    except ValueError as exc:
        raise ValueError(f"invalid seed specification {value!r}: {exc}") from exc
    if not seeds:
        raise ValueError("at least one seed is required")
    if any(seed < 0 for seed in seeds):
        raise ValueError("seeds must be nonnegative integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seed specification contains duplicates")
    return tuple(seeds)


def _validation_failures(
    *,
    terminated: bool,
    turns: int,
    statuses: Sequence[str],
    controller_errors: Sequence[Mapping[str, Any]],
    backend_errors: Sequence[Mapping[str, Any]],
    primitive_actions: int,
    interaction_turns: int,
    observed_days: Sequence[int],
    manager_days: Sequence[int],
    opening_diagnostics: Mapping[str, Any] | None,
) -> list[str]:
    failures: list[str] = []
    if not terminated:
        failures.append("game did not terminate cleanly")
    if turns != FULL_GAME_TURNS:
        failures.append(f"expected {FULL_GAME_TURNS} turns, got {turns}")
    if list(statuses) != ["DONE", "DONE"]:
        failures.append(f"expected terminal statuses, got {list(statuses)!r}")
    if controller_errors:
        failures.append("controller errors were recorded")
    if backend_errors:
        failures.append("backend errors were recorded")
    if primitive_actions != turns:
        failures.append(
            f"strip action count {primitive_actions} did not match turns {turns}"
        )
    if interaction_turns < 1:
        failures.append("strip executor completed no gameplay interaction")
    expected_observed_days = list(range(TOTAL_DAYS))
    if list(observed_days) != expected_observed_days:
        failures.append(
            f"expected observations across days {expected_observed_days}, "
            f"got {list(observed_days)}")
    expected_manager_days = list(range(MANAGER_START_DAY, TOTAL_DAYS))
    if list(manager_days) != expected_manager_days:
        failures.append(
            f"expected Stage 2.5 manager days {expected_manager_days}, "
            f"got {list(manager_days)}")
    if not isinstance(opening_diagnostics, Mapping):
        failures.append("opening diagnostics are missing")
    else:
        divergence = opening_diagnostics.get("divergence", {})
        handoff = opening_diagnostics.get("handoff", {})
        if int(opening_diagnostics.get("turns_replayed", -1)) != OPENING_TURNS:
            failures.append(
                f"opening replayed {opening_diagnostics.get('turns_replayed')!r} "
                f"turns, expected {OPENING_TURNS}")
        if not isinstance(divergence, Mapping) or divergence.get("occurred"):
            failures.append("opening trace diverged")
        if (not isinstance(handoff, Mapping)
                or handoff.get("turn") != [MANAGER_START_DAY, 0]
                or not handoff.get("clean_d4h0_handoff")):
            failures.append("opening did not hand off cleanly at day 4 hour 0")
    return failures


def _strip_detail_by_seat(result: MatchResult) -> dict[int, Mapping[str, Any]]:
    """Map every seat carrying strip-controller diagnostics to its detail."""
    details: dict[int, Mapping[str, Any]] = {}
    for item in result.executor_diagnostics:
        seat = item.get("seat")
        detail = item.get("detail")
        if (isinstance(seat, int) and not isinstance(seat, bool)
                and isinstance(detail, Mapping)
                and isinstance(detail.get("validation"), Mapping)):
            details[seat] = detail
    return details


def _seat_validation(
    result: MatchResult, detail: Mapping[str, Any],
    opening_detail: Mapping[str, Any] | None,
) -> tuple[list[str], dict[str, Any]]:
    """Validate one strip seat and return (failures, summary counters)."""
    validation = detail.get("validation", {})
    agent_diagnostics = detail.get("agent", {})
    days = agent_diagnostics.get("days", {})
    interaction_turns = sum(
        int(route.get("interaction_turns", 0))
        for day in days.values()
        for route in day.get("route_diagnostics", ())
    )
    primitive_actions = int(validation.get("primitive_actions", 0))
    observed_days = list(validation.get("observed_days", ()))
    manager_days = list(validation.get("stage25_manager_days", ()))
    failures = _validation_failures(
        terminated=result.terminated,
        turns=result.turns,
        statuses=result.statuses,
        controller_errors=result.controller_errors,
        backend_errors=result.backend_errors,
        primitive_actions=primitive_actions,
        interaction_turns=interaction_turns,
        observed_days=observed_days,
        manager_days=manager_days,
        opening_diagnostics=opening_detail,
    )
    return failures, {
        "active_days": len(observed_days),
        "stage25_manager_days": len(manager_days),
        "primitive_actions": primitive_actions,
        "interaction_turns": interaction_turns,
        "automatic_sell_orders_by_product": dict(
            validation.get("automatic_sell_orders_by_product", {})),
    }


def _summarize_match(
    result: MatchResult, *, worker_pid: int, require_both_seats: bool = False,
) -> dict[str, Any]:
    details = _strip_detail_by_seat(result)
    opening_by_seat: dict[int, Mapping[str, Any]] = {}
    for item in result.opening_diagnostics:
        seat = item.get("seat")
        detail = item.get("detail")
        if isinstance(seat, int) and not isinstance(seat, bool) \
                and isinstance(detail, Mapping):
            opening_by_seat[seat] = detail

    primary = result.controller_a_seat
    strip_detail = details.get(primary)
    opening_detail = opening_by_seat.get(primary)

    if strip_detail is None:
        failures = ["no strip-controller diagnostics for the primary seat"]
        counters = {
            "active_days": 0,
            "stage25_manager_days": 0,
            "primitive_actions": 0,
            "interaction_turns": 0,
            "automatic_sell_orders_by_product": {},
        }
    else:
        failures, counters = _seat_validation(
            result, strip_detail, opening_detail)

    by_seat: dict[str, Any] = {}
    all_failures: list[str] = []
    for seat in sorted(details):
        seat_failures, seat_counters = _seat_validation(
            result, details[seat], opening_by_seat.get(seat))
        by_seat[str(seat)] = {
            "passed": not seat_failures,
            "failures": seat_failures,
            **seat_counters,
        }
        all_failures.extend(
            f"seat {seat}: {failure}" for failure in seat_failures)

    if require_both_seats:
        missing = [seat for seat in (0, 1) if seat not in details]
        for seat in missing:
            all_failures.append(
                f"seat {seat}: missing strip-controller diagnostics")
        if len(details) < 2:
            all_failures.append(
                "symmetric validation requires both seats to run the checkpoint")

    failures = all_failures if require_both_seats else failures

    return {
        "episode_index": result.episode_index,
        "seed": result.seed,
        "orientation": f"checkpoint_seat_{result.controller_a_seat}_vs_pass",
        "controller_a_seat": result.controller_a_seat,
        "final_banks": result.final_banks,
        "margin": result.margin,
        "winner_seat": result.winner_seat,
        "outcome": result.outcome,
        "statuses": result.statuses,
        "terminated": result.terminated,
        "turns": result.turns,
        "trace_digest": result.trace_digest,
        "runtime_seconds": result.runtime_seconds,
        "controller_errors": result.controller_errors,
        "backend_errors": result.backend_errors,
        "opening": opening_detail,
        "validation": {
            "passed": not failures,
            "failures": failures,
            "strips_by_seat": by_seat,
            **counters,
        },
        "worker_pid": worker_pid,
    }


def _run_full_game_task(
    task: tuple[int, int, int], checkpoint_path: str, checkpoint_sha256: str,
    *, enable_row_claim_board: bool = False, opponent: str = OPPONENT_SYMETRIC,
) -> dict[str, Any]:
    episode_index, seed, controller_a_seat = task
    started = time.perf_counter()
    backend_configuration = {
        "seed": seed,
        "numThreads": 1,
        "episodeSteps": EPISODE_STEPS,
    }
    strip_factory = _StripControllerFactory(
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        episode_index=episode_index,
        seed=seed,
        enable_row_claim_board=enable_row_claim_board,
    )
    if opponent == OPPONENT_SYMETRIC:
        # Identical checkpoint and executor configuration on both seats, so
        # the seat-swap duplicate carries no extra information.
        controller_b: Any = _StripControllerFactory(
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_sha256,
            episode_index=episode_index,
            seed=seed,
            enable_row_claim_board=enable_row_claim_board,
        )
    else:
        controller_b = _PassControllerFactory()
    result = run_match(
        strip_factory,
        controller_b,
        seed=seed,
        controller_a_seat=controller_a_seat,
        backend_name="fast",
        backend_configuration=backend_configuration,
        max_turns=FULL_GAME_TURNS,
        episode_index=episode_index,
    )
    record = _summarize_match(
        result,
        worker_pid=os.getpid(),
        require_both_seats=opponent == OPPONENT_SYMETRIC,
    )
    record["worker_wall_seconds"] = time.perf_counter() - started
    return record


def _fast_engine_provenance() -> dict[str, Any]:
    module = importlib.import_module("fast_env")
    if not hasattr(module, "FastKaggricultureEnv"):
        raise RuntimeError("fast_env does not expose FastKaggricultureEnv")
    engine = backend_provenance(
        "fast",
        {"seed": "per_game", "numThreads": 1, "episodeSteps": EPISODE_STEPS},
    )
    module_path = engine.get("engine_module")
    if not module_path or not Path(module_path).is_file():
        raise RuntimeError("could not identify the loaded fast-engine module")
    engine["engine_module_sha256"] = _sha256_file(module_path)
    return engine


def _source_provenance() -> dict[str, Any]:
    repository_root = Path(__file__).resolve().parents[1]

    def git_value(*arguments: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *arguments],
                cwd=repository_root,
                capture_output=True,
                check=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip() or None

    source_paths = {
        "evaluation_agent_match": repository_root / "evaluation" / "agent_match.py",
        "rl_manager_executor_factory": (
            repository_root / "rl_manager" / "executor_factory.py"
        ),
        "rl_manager_stage25_provider": (
            repository_root / "rl_manager" / "stage25_provider.py"
        ),
        "rl_manager_stage25_checkpoint": (
            repository_root / "rl_manager" / "stage25_checkpoint.py"
        ),
        "rl_manager_stage25_policy": (
            repository_root / "rl_manager" / "stage25_policy.py"
        ),
        "opening_agent": repository_root / "opening_book" / "agent.py",
        "opening_trace": repository_root / "opening_book" / "trace.py",
        "oracle_backend": repository_root / "oracle" / "backend.py",
        "oracle_canonical": repository_root / "oracle" / "canonical.py",
    }
    return {
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_branch": git_value("branch", "--show-current"),
        "harness_sha256": _sha256_file(__file__),
        "executor_v0_source_sha256": _python_tree_sha256(
            repository_root / "executor_v0"
        ),
        "source_sha256": {
            name: _sha256_file(path) for name, path in source_paths.items()
        },
    }


def _python_tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


def _task_iter(
    seeds: Sequence[int], *, both_seats: bool = False
) -> Iterable[tuple[int, int, int]]:
    """Yield (episode_index, seed, controller_a_seat) tasks.

    Symmetric panels run one game per seed with the same checkpoint and
    executor configuration on both seats, so no seat-swap duplicate is
    emitted. ``both_seats`` is reserved for explicitly asymmetric panels.
    """
    seats = (0, 1) if both_seats else (0,)
    episode_index = 0
    for seed in seeds:
        for controller_a_seat in seats:
            yield episode_index, seed, controller_a_seat
            episode_index += 1


def run_validation(
    *, checkpoint_path: str | Path,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    workers: int = DEFAULT_WORKERS,
    row_claim: bool = False,
    opponent: str = OPPONENT_SYMETRIC,
) -> dict[str, Any]:
    """Run one full game per seed using one explicit native policy checkpoint.

    ``opponent`` selects the panel shape. The default symmetric panel runs the
    same checkpoint and executor configuration on both seats, so one seed is
    exactly one game. The ``pass`` panel is the explicitly asymmetric mode and
    runs both seat orientations against the PASS baseline.
    """
    if opponent not in OPPONENT_MODES:
        raise ValueError(
            f"opponent must be one of {OPPONENT_MODES!r}, got {opponent!r}")
    checkpoint = _resolve_checkpoint_path(checkpoint_path)
    checkpoint_sha256 = _sha256_file(checkpoint)
    normalized_seeds = tuple(seeds)
    if not normalized_seeds:
        raise ValueError("at least one seed is required")
    if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
           for seed in normalized_seeds):
        raise ValueError("seeds must be nonnegative integers")
    if len(set(normalized_seeds)) != len(normalized_seeds):
        raise ValueError("seeds must be unique")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")

    both_seats = opponent == OPPONENT_PASS
    expected_games = len(normalized_seeds) * (2 if both_seats else 1)
    workers_used = min(workers, expected_games)
    game_results: list[dict[str, Any]] = []
    worker_errors: list[dict[str, Any]] = []
    started = time.perf_counter()
    specs = iter(_task_iter(normalized_seeds, both_seats=both_seats))

    with ProcessPoolExecutor(
        max_workers=workers_used,
        mp_context=mp.get_context("spawn"),
    ) as pool:
        pending: dict[Any, tuple[int, int, int]] = {}

        def fill_workers() -> None:
            while len(pending) < workers_used:
                try:
                    task = next(specs)
                except StopIteration:
                    break
                try:
                    pending[pool.submit(
                        _run_full_game_task, task, str(checkpoint),
                        checkpoint_sha256,
                        enable_row_claim_board=row_claim, opponent=opponent,
                    )] = task
                except Exception as exc:  # noqa: BLE001 - retain per-game failure.
                    index, seed, seat = task
                    worker_errors.append({
                        "episode_index": index,
                        "seed": seed,
                        "controller_a_seat": seat,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    })

        fill_workers()
        while pending:
            completed, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in sorted(completed, key=lambda item: pending[item][0]):
                task = pending.pop(future)
                try:
                    game_results.append(future.result())
                except Exception as exc:  # noqa: BLE001 - report game boundary.
                    index, seed, seat = task
                    worker_errors.append({
                        "episode_index": index,
                        "seed": seed,
                        "controller_a_seat": seat,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    })
            fill_workers()

    game_results.sort(key=lambda result: result["episode_index"])
    worker_errors.sort(key=lambda error: error["episode_index"])
    passed_games = sum(bool(game["validation"]["passed"])
                       for game in game_results)
    outcomes = {
        result: sum(game["outcome"] == result for game in game_results)
        for result in ("W", "L", "T")
    }
    worker_pids = sorted({game["worker_pid"] for game in game_results})
    parallel_workers_observed = len(worker_pids)
    all_passed = (
        len(game_results) == expected_games
        and not worker_errors
        and passed_games == expected_games
        and (workers_used == 1 or parallel_workers_observed > 1)
    )
    factory = _StripControllerFactory(
        checkpoint_path=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        episode_index=0,
        seed=normalized_seeds[0],
        enable_row_claim_board=row_claim,
    )
    if opponent == OPPONENT_SYMETRIC:
        controller_b_provenance: Mapping[str, Any] = dict(factory.provenance)
        orientations = ["symmetric_same_policy_both_seats"]
    else:
        controller_b_provenance = dict(_PassControllerFactory().provenance)
        orientations = [
            "checkpoint_seat_0_vs_pass", "checkpoint_seat_1_vs_pass",
        ]
    return {
        "schema_version": 3,
        "validation": "stage25_native_checkpoint_parallel_full_game_v1",
        "status": "ok" if all_passed else "failed",
        "checkpoint_backed": True,
        "policy_quality_claim": False,
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": checkpoint_sha256,
            "format": "stage25_native_inference_npz",
        },
        "executor_settings": {
            "enable_row_claim_board": bool(row_claim),
            "aggressive_sell_all": True,
            "opponent": opponent,
            "symmetric": opponent == OPPONENT_SYMETRIC,
        },
        "engine_provenance": _fast_engine_provenance(),
        "source_provenance": _source_provenance(),
        "controller_a": factory.provenance,
        "controller_b": controller_b_provenance,
        "panel": {
            "seeds": list(normalized_seeds),
            "orientations": orientations,
            "opponent": opponent,
            "expected_games": expected_games,
            "completed_games": len(game_results),
            "full_game_turns": FULL_GAME_TURNS,
            "episode_steps": EPISODE_STEPS,
            "workers_requested": workers,
            "workers_used": workers_used,
            "worker_pids": worker_pids,
            "parallel_workers_observed": parallel_workers_observed,
            "parallelism": "multiprocessing.spawn",
            "wall_seconds": time.perf_counter() - started,
        },
        "summary": {
            "passed_games": passed_games,
            "failed_games": len(game_results) - passed_games + len(worker_errors),
            "outcomes_from_strip_controller": outcomes,
        },
        "worker_errors": worker_errors,
        "games": game_results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run checkpoint-backed parallel full-game Stage 2.5 validation"
        ),
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="trained Stage 2.5 native inference checkpoint (.npz)",
    )
    parser.add_argument(
        "--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS),
        help="comma-separated seeds or inclusive ranges such as 7,10..12",
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    row_claim_group = parser.add_mutually_exclusive_group()
    row_claim_group.add_argument(
        "--row-claim", dest="row_claim", action="store_true", default=False,
        help="enable the opt-in row-claim scheduler in the strip executor",
    )
    row_claim_group.add_argument(
        "--no-row-claim", dest="row_claim", action="store_false",
        help="disable the row-claim scheduler (default)",
    )
    parser.add_argument(
        "--opponent", choices=OPPONENT_MODES, default=OPPONENT_SYMETRIC,
        help=(
            "symmetric (default) runs the same checkpoint and executor "
            "configuration on both seats, so one seed is one game; pass runs "
            "both seat orientations against the PASS baseline"
        ),
    )
    args = parser.parse_args(argv)
    try:
        report = run_validation(
            checkpoint_path=args.checkpoint,
            seeds=parse_seed_spec(args.seeds),
            workers=args.workers,
            row_claim=args.row_claim,
            opponent=args.opponent,
        )
    except Exception as exc:  # noqa: BLE001 - CLI returns a stable JSON failure.
        print(json.dumps({
            "schema_version": 3,
            "validation": "stage25_native_checkpoint_parallel_full_game_v1",
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }, sort_keys=True, allow_nan=False))
        return 1
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())

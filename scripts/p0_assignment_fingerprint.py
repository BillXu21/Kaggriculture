"""Measure value-identical strip assignment work on canonical seed 41003."""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import executor_v0.strip_executor as se  # noqa: E402
import executor_v0.strip_hiring as sh  # noqa: E402
import executor_v0.strip_routes as sr  # noqa: E402
import scripts.parallel_full_game_validation as validation  # noqa: E402

CHECKPOINT = Path(
    r"C:\Users\liuyi\VSCodeProjecs\Kaggriculture"
    r"\stage25_bc_7m_best_inference.npz"
)
SEED = 41003

CALLS: list[dict[str, Any]] = []
SEEN: dict[Any, tuple[str, int]] = {}
SCOPED_SEEN: dict[Any, tuple[str, int]] = {}
FRONTIER_STACK: list[list[float]] = []
CONTROLLER_IDS: dict[Any, int] = {}


def _fingerprint(args: tuple[Any, ...], kwargs: dict[str, Any]):
    candidates, positions = args[:2]
    return sr.route_assignment_fingerprint(
        candidates,
        positions,
        assignment_hour=kwargs["assignment_hour"],
        remaining_action_slots=kwargs.get("remaining_action_slots"),
        worker_action_slots=kwargs.get("worker_action_slots"),
        worker_inventories=kwargs.get("worker_inventories"),
        shed_stock=kwargs.get("shed_stock"),
        global_resources=kwargs.get("global_resources"),
        enable_row_helpers=kwargs.get("enable_row_helpers", True),
    )


def _context_from_frame(frame):
    controller = None
    observation = None
    current = frame
    while current is not None:
        local_values = current.f_locals
        if observation is None and isinstance(local_values.get("obs"), Mapping):
            observation = local_values["obs"]
        candidate = local_values.get("self")
        if candidate is not None and hasattr(candidate, "_day"):
            controller = candidate
        current = current.f_back
    if controller is None:
        controller_id = None
        day = None if observation is None else int(observation.get("day", -1))
    else:
        if controller not in CONTROLLER_IDS:
            CONTROLLER_IDS[controller] = len(CONTROLLER_IDS)
        controller_id = CONTROLLER_IDS[controller]
        day = (
            int(observation.get("day", -1))
            if observation is not None
            else int(getattr(controller, "_day", -1))
        )
    return controller_id, day


def _register(site: str, fingerprint: Any, seconds: float, scope: Any) -> int:
    prior = SEEN.get(fingerprint)
    if prior is None:
        SEEN[fingerprint] = (site, len(CALLS))
    scoped_key = (scope, fingerprint)
    scoped_prior = SCOPED_SEEN.get(scoped_key)
    if scoped_prior is None:
        SCOPED_SEEN[scoped_key] = (site, len(CALLS))
    CALLS.append({
        "site": site,
        "fingerprint": fingerprint,
        "seconds": seconds,
        "scope": scope,
        "duplicate": prior is not None,
        "duplicate_in_scope": scoped_prior is not None,
        "first_site": prior[0] if prior else None,
        "first_site_in_scope": scoped_prior[0] if scoped_prior else None,
    })
    return len(CALLS) - 1


def _wrap_hiring_single(original):
    def timed(candidates, positions, **kwargs):
        candidates = tuple(candidates)
        started = time.perf_counter()
        try:
            result = original(candidates, positions, **kwargs)
        finally:
            elapsed = time.perf_counter() - started
        caller = sys._getframe(1)
        line = caller.f_lineno
        site = (
            "hiring_single_overload" if line < 449 else
            "hiring_single_helper" if line < 511 else
            "hiring_single_final_assignment"
        )
        _register(
            site,
            _fingerprint((candidates, positions), kwargs),
            elapsed,
            _context_from_frame(sys._getframe(1)),
        )
        return result

    return timed


def _wrap_frontier(original):
    def timed(candidates, positions, *, worker_counts, **kwargs):
        candidates = tuple(candidates)
        counts = tuple(worker_counts)
        started = time.perf_counter()
        nested_answer_seconds: list[float] = []
        FRONTIER_STACK.append(nested_answer_seconds)
        try:
            result = original(
                candidates, positions, worker_counts=counts, **kwargs
            )
        finally:
            elapsed = time.perf_counter() - started
            FRONTIER_STACK.pop()
        answer_fingerprints = []
        ordered_workers = tuple(sorted(positions))
        for count in sorted(set(counts)):
            prefix = ordered_workers[:count]
            answer_fingerprints.append(sr.route_assignment_fingerprint(
                candidates,
                {worker: positions[worker] for worker in prefix},
                assignment_hour=kwargs["assignment_hour"],
                remaining_action_slots=kwargs.get("remaining_action_slots"),
                worker_action_slots=(
                    None if kwargs.get("worker_action_slots") is None else {
                        worker: kwargs["worker_action_slots"][worker]
                        for worker in prefix
                    }
                ),
                worker_inventories=(
                    None if kwargs.get("worker_inventories") is None else {
                        worker: kwargs["worker_inventories"][worker]
                        for worker in prefix
                        if worker in kwargs["worker_inventories"]
                    }
                ),
                shed_stock=kwargs.get("shed_stock"),
                global_resources=kwargs.get("global_resources"),
                enable_row_helpers=kwargs.get("enable_row_helpers", True),
            ))
        whole_fingerprint = ("frontier", tuple(answer_fingerprints))
        shared_packing_seconds = max(0.0, elapsed - sum(nested_answer_seconds))
        _register(
            "hiring_frontier", whole_fingerprint, shared_packing_seconds,
            _context_from_frame(sys._getframe(1)),
        )
        return result

    return timed


def _wrap_frontier_answer(original):
    def timed(candidates, positions, **kwargs):
        fingerprint = _fingerprint((candidates, positions), kwargs)
        started = time.perf_counter()
        try:
            result = original(candidates, positions, **kwargs)
        finally:
            elapsed = time.perf_counter() - started
        if FRONTIER_STACK:
            FRONTIER_STACK[-1].append(elapsed)
        _register(
            "frontier_answer", fingerprint, elapsed,
            _context_from_frame(sys._getframe(1)),
        )
        return result

    return timed


def _wrap_executor_single(original):
    def timed(candidates, positions, **kwargs):
        candidates = tuple(candidates)
        started = time.perf_counter()
        try:
            result = original(candidates, positions, **kwargs)
        finally:
            elapsed = time.perf_counter() - started
        caller = sys._getframe(1).f_code.co_name
        site = "finalization" if caller == "_finalize_day" else f"executor_{caller}"
        _register(
            site,
            _fingerprint((candidates, positions), kwargs),
            elapsed,
            _context_from_frame(sys._getframe(1)),
        )
        return result

    return timed


def main() -> int:
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)
    sh.assign_horizontal_routes = _wrap_hiring_single(sh.assign_horizontal_routes)
    sh.assign_horizontal_routes_frontier = _wrap_frontier(
        sh.assign_horizontal_routes_frontier
    )
    sr.assign_horizontal_routes = _wrap_frontier_answer(sr.assign_horizontal_routes)
    se.assign_horizontal_routes = _wrap_executor_single(se.assign_horizontal_routes)

    started = time.perf_counter()
    game = validation._run_full_game_task(
        (0, SEED, 0), str(CHECKPOINT), validation._sha256_file(CHECKPOINT),
        enable_row_claim_board=False,
        opponent=validation.OPPONENT_SYMETRIC,
        opening_name="standard_mixed_d6h3",
    )
    wall = time.perf_counter() - started
    timing = game.get("timing_seconds") or {}
    controller_cpu = sum(
        value for value in timing.get("controllers", ())
        if isinstance(value, (int, float))
    )

    by_site: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for call in CALLS:
        by_site[call["site"]].append(call)
    duplicate_calls = [call for call in CALLS if call["duplicate"]]
    scoped_duplicate_calls = [call for call in CALLS if call["duplicate_in_scope"]]
    frontier_keys_before_call: dict[Any, int] = {}
    single_answer_duplicates = []
    for index, call in enumerate(CALLS):
        if call["site"] == "frontier_answer":
            frontier_keys_before_call.setdefault(
                (call["scope"], call["fingerprint"]), index
            )
        elif call["site"] != "hiring_frontier":
            prior = frontier_keys_before_call.get(
                (call["scope"], call["fingerprint"])
            )
            if prior is not None:
                single_answer_duplicates.append(call)
    answer_calls = [call for call in CALLS if call["site"] == "frontier_answer"]
    unique_answer_fingerprints = len({call["fingerprint"] for call in answer_calls})
    frontier_duplicate_answers = sum(call["duplicate"] for call in answer_calls)

    summary = {
        "seed": SEED,
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": validation._sha256_file(CHECKPOINT),
        "opening": "standard_mixed_d6h3",
        "row_claim": False,
        "wall_seconds_instrumented": wall,
        "controller_seconds": controller_cpu,
        "final_banks": game.get("final_banks"),
        "turns": game.get("turns"),
        "frontier_builds": len(by_site.get("hiring_frontier", ())),
        "frontier_worker_count_answers": len(answer_calls),
        "unique_frontier_answer_fingerprints": unique_answer_fingerprints,
        "duplicate_frontier_answer_outputs": frontier_duplicate_answers,
        "assignment_frontier_invocations": sum(
            call["site"] != "frontier_answer" for call in CALLS
        ),
        "unique_invocation_fingerprints": len(SEEN),
        "duplicate_invocations": len(duplicate_calls),
        "duplicate_cpu_seconds": sum(call["seconds"] for call in duplicate_calls),
        "duplicate_invocations_within_controller_day": len(scoped_duplicate_calls),
        "duplicate_cpu_seconds_within_controller_day": sum(
            call["seconds"] for call in scoped_duplicate_calls
        ),
        "single_calls_matching_prior_frontier_answers": len(single_answer_duplicates),
        "single_prior_match_cpu_seconds": sum(
            call["seconds"] for call in single_answer_duplicates
        ),
        "frontier_answer_cpu_seconds": sum(call["seconds"] for call in answer_calls),
        "calls_by_site": {},
    }
    for site, calls in sorted(by_site.items()):
        duplicates = [call for call in calls if call["duplicate"]]
        summary["calls_by_site"][site] = {
            "calls": len(calls),
            "unique_fingerprints": len({call["fingerprint"] for call in calls}),
            "duplicate_calls": len(duplicates),
            "duplicate_rate": len(duplicates) / len(calls) if calls else 0.0,
            "total_cpu_seconds": sum(call["seconds"] for call in calls),
            "duplicate_cpu_seconds": sum(call["seconds"] for call in duplicates),
            "duplicates_within_controller_day": sum(
                call["duplicate_in_scope"] for call in calls
            ),
            "duplicate_cpu_seconds_within_controller_day": sum(
                call["seconds"] for call in calls if call["duplicate_in_scope"]
            ),
        }
    summary["calls_by_site"]["frontier_answer"] = {
        "calls": len(answer_calls),
        "unique_fingerprints": unique_answer_fingerprints,
        "duplicate_calls": frontier_duplicate_answers,
        "duplicate_rate": frontier_duplicate_answers / max(1, len(answer_calls)),
        "total_cpu_seconds": sum(call["seconds"] for call in answer_calls),
        "duplicate_cpu_seconds": sum(
            call["seconds"] for call in answer_calls if call["duplicate"]
        ),
        "duplicates_within_controller_day": sum(
            call["duplicate_in_scope"] for call in answer_calls
        ),
        "duplicate_cpu_seconds_within_controller_day": sum(
            call["seconds"] for call in answer_calls if call["duplicate_in_scope"]
        ),
    }
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

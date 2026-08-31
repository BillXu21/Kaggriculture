"""Run two arbitrary primitive controllers through a scalar engine backend.

This module intentionally does not use ``rl_manager.runner``.  Training owns
manager-day batching and trajectories; evaluation needs the smaller contract
``controller(own_observation) -> primitive action`` and permits unrelated
controller implementations on the two seats.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import time
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from oracle.backend import EngineBackend, make_backend


PASS_ACTION: dict[str, Any] = {
    "farmer": ["PASS"],
    "hands": [],
    "market": [],
}
ORIENTATIONS = ("candidate_vs_frozen", "frozen_vs_candidate")


class _FailedController:
    observation_mode = "raw"

    def act(self, observation: Mapping[str, Any]) -> Mapping[str, Any]:
        del observation
        raise RuntimeError("controller failed during startup")

    __call__ = act

    def close(self) -> None:
        return None


@runtime_checkable
class PrimitiveController(Protocol):
    """One stateful controller instance, owned by one seat for one game."""

    observation_mode: str

    def act(self, observation: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def __call__(self, observation: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def close(self) -> None: ...


@runtime_checkable
class ControllerFactory(Protocol):
    """Immutable identity plus a fresh per-game controller constructor."""

    provenance: Mapping[str, Any]

    def create(
        self,
        *,
        seat: int,
        configuration: Mapping[str, Any],
    ) -> PrimitiveController: ...


@dataclass(frozen=True)
class MatchResult:
    """One game, shaped to remain consumable by ``summarize_evaluation``."""

    episode_index: int
    seed: int
    composition: str
    orientation: str
    controller_a_seat: int
    final_banks: list[float]
    margin: float
    winner_seat: int
    outcome: str
    statuses: list[str]
    terminated: bool
    turns: int
    runtime_seconds: float
    trace_digest: str
    controller_errors: list[dict[str, Any]] = field(default_factory=list)
    backend_errors: list[dict[str, Any]] = field(default_factory=list)
    controller_provenance: list[dict[str, Any]] = field(default_factory=list)
    timing_seconds: dict[str, Any] = field(default_factory=dict)
    opening_diagnostics: list[dict[str, Any]] = field(default_factory=list)
    executor_diagnostics: list[dict[str, Any]] = field(default_factory=list)

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def normalize_action(action: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the outer official action shape without changing semantics."""
    if not isinstance(action, Mapping):
        raise TypeError(
            f"controller action must be a mapping, got {type(action).__name__}"
        )
    farmer = action.get("farmer", ["PASS"])
    hands = action.get("hands", [])
    market = action.get("market", [])
    if not isinstance(farmer, (list, tuple)):
        raise TypeError("action.farmer must be a list or tuple")
    if not isinstance(hands, (list, tuple)):
        raise TypeError("action.hands must be a list or tuple")
    if not isinstance(market, (list, tuple)):
        raise TypeError("action.market must be a list or tuple")
    if any(not isinstance(entry, (list, tuple)) for entry in hands):
        raise TypeError("each action.hands entry must be a list or tuple")
    if any(not isinstance(entry, (list, tuple)) for entry in market):
        raise TypeError("each action.market entry must be a list or tuple")
    normalized = {
        "farmer": list(farmer),
        "hands": [list(entry) for entry in hands],
        "market": [list(entry) for entry in market],
    }
    # This catches custom objects, NaN, and other values the JSON-shaped wire
    # contract cannot represent while leaving operation legality to the engine.
    _canonical_json(normalized)
    return normalized


def _agent_configuration(
    backend: EngineBackend,
    requested: Mapping[str, Any],
) -> dict[str, Any]:
    if backend.name == "official":
        configuration = dict(getattr(backend, "env").configuration)
    else:
        environment = getattr(backend, "_env", None)
        configuration = dict(getattr(environment, "configuration", requested))
    configuration.pop("numThreads", None)
    # The pinned official interpreter resolves the episode seed before the
    # first call and exposes configuration.seed=None to competition agents.
    configuration["seed"] = None
    return configuration


def _controller_observation(
    raw: Mapping[str, Any],
    controller: PrimitiveController,
    canonical_farms: Sequence[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    observation = copy.deepcopy(dict(raw))
    mode = getattr(controller, "observation_mode", "raw")
    if mode == "canonical":
        if canonical_farms is None:
            raise RuntimeError("canonical controller observation unavailable")
        observation["farms"] = copy.deepcopy(list(canonical_farms))
        observation.setdefault(
            "step",
            int(observation["day"]) * 24 + int(observation.get("hour", 0)),
        )
    elif mode != "raw":
        raise ValueError(f"unknown controller observation mode {mode!r}")
    return observation


def _call_controller(
    controller: PrimitiveController, observation: Mapping[str, Any]
) -> Mapping[str, Any]:
    action_method = getattr(controller, "act", None)
    if callable(action_method):
        return action_method(observation)
    if callable(controller):
        return controller(observation)
    raise TypeError(f"controller {controller!r} has no act/__call__ method")


def _banks(observations: Sequence[Mapping[str, Any]]) -> list[float]:
    return [float(observations[seat]["farms"][seat]["money"]) for seat in range(2)]


def run_match(
    controller_a: ControllerFactory,
    controller_b: ControllerFactory,
    *,
    seed: int,
    controller_a_seat: int = 0,
    backend_name: str = "fast",
    backend_configuration: Mapping[str, Any] | None = None,
    max_turns: int = 719,
    episode_index: int = 0,
    backend: EngineBackend | None = None,
) -> MatchResult:
    """Run one game and report every outcome from controller A's perspective."""
    if controller_a_seat not in (0, 1):
        raise ValueError("controller_a_seat must be 0 or 1")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an int")
    if max_turns < 1:
        raise ValueError("max_turns must be positive")

    requested_configuration = dict(backend_configuration or {})
    requested_configuration["seed"] = seed
    engine = backend or make_backend(backend_name, requested_configuration)
    started = time.perf_counter()
    observations = engine.reset()
    configuration = _agent_configuration(engine, requested_configuration)
    factories = (
        (controller_a, controller_b)
        if controller_a_seat == 0
        else (controller_b, controller_a)
    )
    controllers: list[PrimitiveController] = []
    controller_errors: list[dict[str, Any]] = []
    backend_errors: list[dict[str, Any]] = []
    timing = {"controllers": [0.0, 0.0], "environment": 0.0}
    trace = hashlib.sha256()
    turns = 0

    try:
        for seat, factory in enumerate(factories):
            try:
                controllers.append(
                    factory.create(seat=seat, configuration=configuration)
                )
            except Exception as exc:  # noqa: BLE001 - game-level boundary
                controller_errors.append(
                    {
                        "seat": seat,
                        "controller": "A" if seat == controller_a_seat else "B",
                        "turn": 0,
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "detail": getattr(exc, "detail", None),
                    }
                )
                controllers.append(_FailedController())
        for turn in range(max_turns):
            canonical_farms = None
            if any(
                getattr(item, "observation_mode", "raw") == "canonical"
                for item in controllers
            ):
                canonical_farms = engine.canonical_state()["farms"]

            actions: list[dict[str, Any]] = []
            for seat, controller in enumerate(controllers):
                call_started = time.perf_counter()
                try:
                    own_observation = _controller_observation(
                        observations[seat], controller, canonical_farms
                    )
                    action = normalize_action(
                        _call_controller(controller, own_observation)
                    )
                except Exception as exc:  # noqa: BLE001 - game-level boundary
                    controller_errors.append(
                        {
                            "seat": seat,
                            "controller": "A" if seat == controller_a_seat else "B",
                            "turn": turn,
                            "type": type(exc).__name__,
                            "message": str(exc),
                            "detail": getattr(exc, "detail", None),
                        }
                    )
                    action = copy.deepcopy(PASS_ACTION)
                finally:
                    timing["controllers"][seat] += time.perf_counter() - call_started
                actions.append(action)
            if controller_errors:
                break

            trace.update(
                _canonical_json({"turn": turn, "actions": actions}).encode("utf-8")
            )
            step_started = time.perf_counter()
            observations, _rewards, statuses = engine.step(actions)
            timing["environment"] += time.perf_counter() - step_started
            turns += 1
            if list(statuses) == ["DONE", "DONE"]:
                break

        validator = getattr(engine, "validate_status_history", None)
        if callable(validator):
            try:
                validator()
            except Exception as exc:  # noqa: BLE001 - preserve panel process
                backend_errors.append(
                    {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
    finally:
        for seat, controller in enumerate(controllers):
            try:
                close_method = getattr(controller, "close", None)
                if callable(close_method):
                    close_method()
            except Exception as exc:  # noqa: BLE001 - cleanup diagnostics
                controller_errors.append(
                    {
                        "seat": seat,
                        "controller": "A" if seat == controller_a_seat else "B",
                        "turn": turns,
                        "type": type(exc).__name__,
                        "message": f"close failed: {exc}",
                    }
                )

    banks = _banks(observations)
    candidate_margin = banks[controller_a_seat] - banks[1 - controller_a_seat]
    winner_seat = 0 if banks[0] > banks[1] else (1 if banks[1] > banks[0] else -1)
    outcome = "W" if candidate_margin > 0 else ("L" if candidate_margin < 0 else "T")
    statuses = list(engine.statuses)
    terminated = (
        statuses == ["DONE", "DONE"] and not controller_errors and not backend_errors
    )
    orientation = ORIENTATIONS[0 if controller_a_seat == 0 else 1]
    diagnostics = [
        {"seat": error["seat"], "runtime_errors": [error]}
        for error in controller_errors
    ]
    opening_diagnostics: list[dict[str, Any]] = []
    executor_diagnostics: list[dict[str, Any]] = []
    for seat, controller in enumerate(controllers):
        diagnostic_fn = getattr(controller, "diagnostics", None)
        if callable(diagnostic_fn):
            try:
                diagnostic = diagnostic_fn()
            except Exception as exc:  # noqa: BLE001 - diagnostic boundary
                diagnostic = {"runtime_error": repr(exc)}
            if diagnostic:
                if diagnostic.get("opening"):
                    opening_diagnostics.append(
                        {"seat": seat, "detail": diagnostic["opening"]}
                    )
                if diagnostic.get("executor"):
                    executor_diagnostics.append(
                        {"seat": seat, "detail": diagnostic["executor"]}
                    )
                if not diagnostic.get("opening") and not diagnostic.get("executor"):
                    diagnostics.append({"seat": seat, **diagnostic})
    diagnostics.extend(executor_diagnostics)
    diagnostics.extend(
        {"seat": None, "runtime_errors": [error]} for error in backend_errors
    )
    provenance = [
        {
            "seat": seat,
            "controller": "A" if seat == controller_a_seat else "B",
            **copy.deepcopy(dict(factories[seat].provenance)),
        }
        for seat in range(2)
    ]
    runtime = time.perf_counter() - started
    if not math.isfinite(runtime):  # pragma: no cover - perf_counter contract
        raise RuntimeError("non-finite match runtime")
    return MatchResult(
        episode_index=episode_index,
        seed=seed,
        composition=orientation,
        orientation=orientation,
        controller_a_seat=controller_a_seat,
        final_banks=banks,
        margin=candidate_margin,
        winner_seat=winner_seat,
        outcome=outcome,
        statuses=statuses,
        terminated=terminated,
        turns=turns,
        runtime_seconds=runtime,
        trace_digest=trace.hexdigest(),
        controller_errors=controller_errors,
        backend_errors=backend_errors,
        controller_provenance=provenance,
        timing_seconds=timing,
        opening_diagnostics=opening_diagnostics,
        executor_diagnostics=diagnostics,
    )


def run_panel(
    controller_a: ControllerFactory,
    controller_b: ControllerFactory,
    *,
    seeds: Sequence[int],
    both_orientations: bool = True,
    backend_name: str = "fast",
    backend_configuration: Mapping[str, Any] | None = None,
    max_turns: int = 719,
) -> list[MatchResult]:
    """Run seeds in input order, with A-seat-0 then A-seat-1 per seed."""
    seats = (0, 1) if both_orientations else (0,)
    results: list[MatchResult] = []
    for seed in seeds:
        for seat in seats:
            results.append(
                run_match(
                    controller_a,
                    controller_b,
                    seed=int(seed),
                    controller_a_seat=seat,
                    backend_name=backend_name,
                    backend_configuration=backend_configuration,
                    max_turns=max_turns,
                    episode_index=len(results),
                )
            )
    return results

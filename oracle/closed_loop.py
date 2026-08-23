"""Independent official-vs-fast closed-loop policy comparison.

Unlike :mod:`oracle.replay`, this runner gives each backend its own fresh
stateful agents.  The agents see only their backend's observations and make
their own decisions; equal observations and equal actions are assertions, not
inputs supplied by the other backend.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

from .backend import EngineBackend, make_backend
from .canonical import FieldDiff, deep_diff
from .canonical import CROPS, SHED_ITEMS, _canonical_farm


class StatefulAgent(Protocol):
    """Kaggle-shaped callable retaining state across primitive turns."""

    def __call__(self, observation: Mapping[str, Any]) -> Mapping[str, Any]: ...


BackendFactory = Callable[[Mapping[str, Any]], EngineBackend]
AgentFactory = Callable[[str, int, Mapping[str, Any]], StatefulAgent]


def _executor_observation(
    observation: Mapping[str, Any], *, from_fast: bool
) -> dict[str, Any]:
    """Adapt wire aliases the existing executor does not consume."""
    view = copy.deepcopy(dict(observation))
    view.setdefault(
        "step", int(view.get("day", 0)) * 24 + int(view.get("hour", 0))
    )
    view["farms"] = [
        _canonical_farm(farm, int(view["day"]), from_fast=from_fast)
        for farm in view.get("farms", [])
    ]
    private = view.get("private")
    if isinstance(private, dict):
        private["shed"] = {
            name: int(private.get("shed", {}).get(name, 0))
            for name in SHED_ITEMS
        }
        private["seeds"] = {
            name: int(private.get("seeds", {}).get(name, 0))
            for name in CROPS
        }
        private["inventories"] = [
            {name: int(inventory.get(name, 0)) for name in SHED_ITEMS}
            for inventory in private.get("inventories", [])
        ]
    return view


@dataclass(frozen=True)
class ClosedLoopDivergenceReport:
    """First closed-loop mismatch with enough context to reproduce it."""

    seed: Any
    step: int
    day: int
    hour: int
    seat: int | None
    field_path: str
    official_value: Any
    fast_value: Any
    official_action: Any
    fast_action: Any
    phase: str  # reset_observation, observation, or action, or next_state
    turn_index: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "step": self.step,
            "day": self.day,
            "hour": self.hour,
            "seat": self.seat,
            "field_path": self.field_path,
            "official_value": self.official_value,
            "fast_value": self.fast_value,
            "official_action": self.official_action,
            "fast_action": self.fast_action,
            "phase": self.phase,
            "turn_index": self.turn_index,
        }

    def render(self) -> str:
        return (
            f"first closed-loop divergence phase={self.phase} "
            f"turn_index={self.turn_index} seed={self.seed!r} "
            f"step={self.step} day={self.day} hour={self.hour} "
            f"seat={self.seat} path={self.field_path} "
            f"official={self.official_value!r} fast={self.fast_value!r} "
            f"official_action={self.official_action!r} "
            f"fast_action={self.fast_action!r}"
        )


class ClosedLoopDivergenceError(AssertionError):
    """Raised before stepping at the first observation/action mismatch."""

    def __init__(self, report: ClosedLoopDivergenceReport) -> None:
        super().__init__(report.render())
        self.report = report


@dataclass(frozen=True)
class ClosedLoopResult:
    """Machine-readable summary of one independent A/B episode."""

    seed: Any
    steps_executed: int
    final_step: int
    official_statuses: list[str]
    fast_statuses: list[str]
    official_rewards: list[float]
    fast_rewards: list[float]
    action_families: dict[str, int]
    observation_comparisons: int
    agent_calls: int
    terminal_step: int | None
    wall_time_seconds: float

    @property
    def terminal(self) -> bool:
        return self.terminal_step is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "steps_executed": self.steps_executed,
            "final_step": self.final_step,
            "official_statuses": list(self.official_statuses),
            "fast_statuses": list(self.fast_statuses),
            "official_rewards": list(self.official_rewards),
            "fast_rewards": list(self.fast_rewards),
            "action_families": dict(sorted(self.action_families.items())),
            "observation_comparisons": self.observation_comparisons,
            "agent_calls": self.agent_calls,
            "terminal_step": self.terminal_step,
            "wall_time_seconds": self.wall_time_seconds,
        }


def make_deterministic_executor_factory(
    *, config: Any | None = None,
) -> AgentFactory:
    """Build fresh existing-``executor_v0`` agents for every A/B seat.

    The fixed plan is deliberately nontrivial (crops, livestock, fertilizer,
    labor, and sells), while ``ExecutorAgent`` supplies the stateful daily
    manager cache, live task regeneration, foreman, and diagnostics path.
    A new provider and agent are constructed for each ``(backend, seat)``.
    """

    from executor_v0.agent import AgentConfig, make_agent
    from executor_v0.manager import FixedPlanProvider
    from executor_v0.smoke import build_fake_plan

    agent_config = config or AgentConfig(strict=True)

    def factory(
        backend_name: str, seat: int, configuration: Mapping[str, Any]
    ) -> StatefulAgent:
        del configuration
        executor = make_agent(
            provider=FixedPlanProvider(build_fake_plan()),
            seat=seat,
            config=agent_config,
        )

        def adapted_agent(observation: Mapping[str, Any]) -> Mapping[str, Any]:
            return executor(_executor_observation(
                observation, from_fast=backend_name == "fast"
            ))

        return adapted_agent

    return factory


def make_checkpoint_executor_factory(
    checkpoint_path: str,
    *,
    device: str = "cpu",
    config: Any | None = None,
) -> AgentFactory:
    """Build fresh existing-executor agents backed by one explicit checkpoint."""

    from executor_v0.agent import AgentConfig, make_agent

    agent_config = config or AgentConfig(strict=True)

    def factory(
        backend_name: str, seat: int, configuration: Mapping[str, Any]
    ) -> StatefulAgent:
        del configuration
        executor = make_agent(
            checkpoint=checkpoint_path,
            device=device,
            seat=seat,
            config=agent_config,
        )

        def adapted_agent(observation: Mapping[str, Any]) -> Mapping[str, Any]:
            return executor(_executor_observation(
                observation, from_fast=backend_name == "fast"
            ))

        return adapted_agent

    return factory


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    try:
        return {str(key): _plain(item) for key, item in value.items()}
    except AttributeError:
        return value


def _normal_observation(
    observation: Mapping[str, Any], *, from_fast: bool,
    default_step: int | None = None
) -> dict[str, Any]:
    """Align wrapper sparsity and fast tile aliases for observation compare."""

    result = _plain(observation)
    if "step" not in result and default_step is not None:
        # The official wrapper omits step from the second seat's observation;
        # it is common public turn metadata, not private policy input.
        result["step"] = default_step
    day = int(result.get("day", 0))
    farms = result.get("farms")
    if isinstance(farms, list):
        result["farms"] = [
            _canonical_farm(farm, day, from_fast=from_fast)
            for farm in farms
        ]
    private = result.get("private")
    if not isinstance(private, dict):
        return result
    private["shed"] = {
        name: int(private.get("shed", {}).get(name, 0))
        for name in SHED_ITEMS
    }
    private["seeds"] = {
        name: int(private.get("seeds", {}).get(name, 0))
        for name in CROPS
    }
    inventories = private.get("inventories", [])
    private["inventories"] = [
        {name: int(inventory.get(name, 0)) for name in SHED_ITEMS}
        for inventory in inventories
    ]
    return result


def _first_diff(left: Any, right: Any, path: str) -> FieldDiff | None:
    diffs = deep_diff(left, right, path=path)
    return diffs[0] if diffs else None


def _action_family(scope: str, action: Any) -> str:
    if isinstance(action, Sequence) and not isinstance(action, (str, bytes)):
        operation = action[0] if action else "EMPTY"
    else:
        operation = "INVALID"
    return f"{scope}.{operation}"


def _record_action_families(
    counts: dict[str, int], action: Mapping[str, Any]
) -> None:
    for scope in ("farmer", "hands"):
        value = action.get(scope, [])
        if scope == "hands" and isinstance(value, Sequence):
            for hand_action in value:
                family = _action_family(scope, hand_action)
                counts[family] = counts.get(family, 0) + 1
        else:
            family = _action_family(scope, value)
            counts[family] = counts.get(family, 0) + 1
    market = action.get("market", [])
    if isinstance(market, Sequence):
        for order in market:
            family = _action_family("market", order)
            counts[family] = counts.get(family, 0) + 1


def _fresh_agents(
    factories: Mapping[str, AgentFactory],
    configuration: Mapping[str, Any],
) -> dict[str, list[StatefulAgent]]:
    agents = {
        backend_name: [
            factories[backend_name](backend_name, seat, configuration)
            for seat in range(2)
        ]
        for backend_name in ("official", "fast")
    }
    identities = [id(agent) for pair in agents.values() for agent in pair]
    if len(set(identities)) != len(identities):
        raise ValueError("closed-loop runner requires four fresh agent instances")
    return agents


def _raise_mismatch(
    *,
    diff: FieldDiff,
    seed: Any,
    step: int,
    day: int,
    hour: int,
    seat: int | None,
    official_action: Any,
    fast_action: Any,
    phase: str,
    turn_index: int,
) -> None:
    raise ClosedLoopDivergenceError(ClosedLoopDivergenceReport(
        seed=seed,
        step=step,
        day=day,
        hour=hour,
        seat=seat,
        field_path=diff.path,
        official_value=diff.official_value,
        fast_value=diff.fast_value,
        official_action=official_action,
        fast_action=fast_action,
        phase=phase,
        turn_index=turn_index,
    ))


def run_closed_loop(
    configuration: Mapping[str, Any] | None = None,
    *,
    seed: Any | None = None,
    max_steps: int = 719,
    terminal_status: str = "DONE",
    backend_factories: Mapping[str, BackendFactory] | None = None,
    agent_factories: Mapping[str, AgentFactory] | None = None,
) -> ClosedLoopResult:
    """Run independent policy instances and compare every A/B boundary.

    ``max_steps`` counts accepted post-reset ``step`` calls.  The default
    therefore preserves the repository's 720-step accounting: one reset
    observation plus 719 transitions ending at canonical step 719.
    ``terminal_status`` controls the paired terminal status required for an
    early successful stop.
    """

    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 0:
        raise ValueError(f"max_steps must be a non-negative integer, got {max_steps!r}")
    if not isinstance(terminal_status, str) or not terminal_status:
        raise ValueError(
            f"terminal_status must be a non-empty string, got {terminal_status!r}"
        )
    config = dict(configuration or {})
    if seed is not None:
        config["seed"] = seed
    run_seed = config.get("seed")
    backend_factories = backend_factories or {
        "official": lambda cfg: make_backend("official", cfg),
        "fast": lambda cfg: make_backend("fast", cfg),
    }
    agent_factories = agent_factories or {
        "official": make_deterministic_executor_factory(),
        "fast": make_deterministic_executor_factory(),
    }
    for name in ("official", "fast"):
        if name not in backend_factories or name not in agent_factories:
            raise ValueError(f"closed-loop factories must provide {name!r}")

    started = time.perf_counter()
    backends = {
        name: backend_factories[name](dict(config))
        for name in ("official", "fast")
    }
    agents = _fresh_agents(agent_factories, config)
    observations = {
        name: backends[name].reset()
        for name in ("official", "fast")
    }
    reset_step = int(backends["official"].canonical_state()["step"])
    observation_comparisons = 0
    action_families: dict[str, int] = {}

    for seat in range(2):
        diff = _first_diff(
            _normal_observation(
                observations["official"][seat], from_fast=False,
                default_step=reset_step
            ),
            _normal_observation(
                observations["fast"][seat], from_fast=True,
                default_step=reset_step
            ),
            f"observation[{seat}]",
        )
        observation_comparisons += 1
        if diff is not None:
            _raise_mismatch(
                diff=diff, seed=run_seed, step=0, day=0, hour=0, seat=seat,
                official_action=None, fast_action=None,
                phase="reset_observation", turn_index=-1,
            )

    reset_diff = _first_diff(
        backends["official"].canonical_state(),
        backends["fast"].canonical_state(),
        "state",
    )
    if reset_diff is not None:
        _raise_mismatch(
            diff=reset_diff, seed=run_seed, step=0, day=0, hour=0, seat=None,
            official_action=None, fast_action=None,
            phase="reset_observation", turn_index=-1,
        )

    steps_executed = 0
    terminal_step: int | None = None
    for turn_index in range(max_steps):
        for seat in range(2):
            current_step = int(backends["official"].canonical_state()["step"])
            diff = _first_diff(
                _normal_observation(
                    observations["official"][seat], from_fast=False,
                    default_step=current_step
                ),
                _normal_observation(
                    observations["fast"][seat], from_fast=True,
                    default_step=current_step
                ),
                f"observation[{seat}]",
            )
            observation_comparisons += 1
            if diff is not None:
                official_state = backends["official"].canonical_state()
                _raise_mismatch(
                    diff=diff, seed=run_seed,
                    step=int(official_state["step"]),
                    day=int(official_state["day"]),
                    hour=int(official_state["hour"]), seat=seat,
                    official_action=None, fast_action=None,
                    phase="observation", turn_index=turn_index,
                )

        actions: dict[str, list[dict[str, Any]]] = {"official": [], "fast": []}
        for backend_name in ("official", "fast"):
            for seat in range(2):
                action = agents[backend_name][seat](
                    copy.deepcopy(observations[backend_name][seat])
                )
                if not isinstance(action, Mapping):
                    raise TypeError(
                        f"{backend_name} seat {seat} agent returned "
                        f"{type(action).__name__}, expected a mapping"
                    )
                actions[backend_name].append(copy.deepcopy(dict(action)))
        for seat in range(2):
            diff = _first_diff(
                actions["official"][seat], actions["fast"][seat],
                f"action[{seat}]",
            )
            if diff is not None:
                state = backends["official"].canonical_state()
                _raise_mismatch(
                    diff=diff, seed=run_seed,
                    step=int(state["step"]), day=int(state["day"]),
                    hour=int(state["hour"]), seat=seat,
                    official_action=actions["official"][seat],
                    fast_action=actions["fast"][seat],
                    phase="action", turn_index=turn_index,
                )
            _record_action_families(action_families, actions["official"][seat])

        observations["official"], _, _ = backends["official"].step(
            copy.deepcopy(actions["official"])
        )
        observations["fast"], _, _ = backends["fast"].step(
            copy.deepcopy(actions["fast"])
        )
        steps_executed = turn_index + 1
        validate_history = getattr(backends["official"], "validate_status_history", None)
        if callable(validate_history):
            validate_history()

        official_state = backends["official"].canonical_state()
        fast_state = backends["fast"].canonical_state()
        diff = _first_diff(official_state, fast_state, "state")
        if diff is not None:
            _raise_mismatch(
                diff=diff, seed=run_seed,
                step=int(official_state["step"]), day=int(official_state["day"]),
                hour=int(official_state["hour"]), seat=None,
                official_action=actions["official"],
                fast_action=actions["fast"],
                phase="next_state", turn_index=turn_index,
            )
        if (
            backends["official"].statuses == [terminal_status, terminal_status]
            and backends["fast"].statuses == [terminal_status, terminal_status]
        ):
            terminal_step = int(official_state["step"])
            break

    official = backends["official"]
    fast = backends["fast"]
    final_step = int(official.canonical_state()["step"])
    return ClosedLoopResult(
        seed=run_seed,
        steps_executed=steps_executed,
        final_step=final_step,
        official_statuses=list(official.statuses),
        fast_statuses=list(fast.statuses),
        official_rewards=list(official.rewards),
        fast_rewards=list(fast.rewards),
        action_families=action_families,
        observation_comparisons=observation_comparisons,
        agent_calls=steps_executed * 4,
        terminal_step=terminal_step,
        wall_time_seconds=time.perf_counter() - started,
    )


__all__ = [
    "AgentFactory",
    "BackendFactory",
    "ClosedLoopDivergenceError",
    "ClosedLoopDivergenceReport",
    "ClosedLoopResult",
    "StatefulAgent",
    "make_checkpoint_executor_factory",
    "make_deterministic_executor_factory",
    "run_closed_loop",
]

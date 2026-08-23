"""Same-action turn-by-turn replay harness: official 1.32.7 vs fast engine.

Primary validation mode. Both engines are reset from the same seed and
configuration; the exact same action pair is submitted to both engines each
turn BEFORE any comparison; canonical full states are compared immediately;
the run stops at the first divergence so later action drift can never obscure
it.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from .backend import EngineBackend, make_backend
from .canonical import FieldDiff, deep_diff

ActionPair = Sequence[Mapping[str, Any]]
ActionSource = Callable[[int, list[dict[str, Any]]], ActionPair]

# Hook applied to the fast canonical state right after each fast step and
# before comparison. Deliberate corruption tests use this tiny seam; production
# replay leaves it as None.
FastMutator = Callable[[dict[str, Any], int], None]


@dataclass(frozen=True)
class DivergenceReport:
    """Compact, deterministic first-divergence context."""

    seed: Any
    step: int
    day: int
    hour: int
    field_path: str
    official_value: Any
    fast_value: Any
    p0_action: Any
    p1_action: Any
    phase: str  # "initial" or "turn"
    turn_index: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "step": self.step,
            "day": self.day,
            "hour": self.hour,
            "field_path": self.field_path,
            "official_value": self.official_value,
            "fast_value": self.fast_value,
            "p0_action": self.p0_action,
            "p1_action": self.p1_action,
            "phase": self.phase,
            "turn_index": self.turn_index,
        }

    def render(self) -> str:
        from .canonical import _render

        return (
            f"first divergence at {self.phase} turn_index={self.turn_index} "
            f"seed={self.seed!r} step={self.step} day={self.day} hour={self.hour} "
            f"path={self.field_path} "
            f"official={_render(self.official_value)} fast={_render(self.fast_value)} "
            f"p0_action={_render(self.p0_action)} p1_action={_render(self.p1_action)}"
        )


class DivergenceError(AssertionError):
    """Raised at the FIRST divergent turn/field; carries a DivergenceReport."""

    def __init__(self, report: DivergenceReport) -> None:
        super().__init__(report.render())
        self.report = report


@dataclass
class ReplayResult:
    turns_executed: int
    final_step: int
    official_statuses: list[str]
    fast_statuses: list[str]
    official_rewards: list[float]
    fast_rewards: list[float]


def _first_diff(diffs: list[FieldDiff]) -> FieldDiff:
    return diffs[0]


def run_same_action_replay(
    configuration: Mapping[str, Any] | None = None,
    actions: Sequence[ActionPair] | ActionSource | None = None,
    *,
    max_turns: int = 720,
    fast_mutator: FastMutator | None = None,
    official_backend: EngineBackend | None = None,
    fast_backend: EngineBackend | None = None,
) -> ReplayResult:
    """Replay identical action pairs on both engines and compare every turn.

    ``actions`` is either a sequence of per-turn ``[action_p0, action_p1]``
    pairs or a callback ``(turn_index, observations) -> pair``. Comparison
    covers step/day/hour, both farms (complete board incl. crop/animal
    lifecycle), hires_today, unlocked quadrants, both seats' private
    shed/seeds/inventories, market inventory/prices(/params), town state with
    duplicate shop multiplicity, rewards, and statuses.

    Raises :class:`DivergenceError` at the first divergent field; official
    status-history anomalies raise
    :class:`oracle.official_backend.OfficialAnomalyError`.
    """
    config = dict(configuration or {})
    seed = config.get("seed")
    official = official_backend or make_backend("official", config)
    fast = fast_backend or make_backend("fast", config)

    if isinstance(actions, Sequence):
        source: ActionSource | None = None
        action_pairs = list(actions)
    else:
        source = actions  # type: ignore[assignment]
        action_pairs = []

    def pair_for(turn: int, observations: list[dict[str, Any]]) -> ActionPair | None:
        if source is not None:
            return source(turn, observations)
        if turn >= len(action_pairs):
            return None
        return action_pairs[turn]

    # Initial comparison (reset is step 0 on both sides).
    official.reset()
    fast.reset()
    initial_official = official.canonical_state()
    initial_fast = fast.canonical_state()
    diffs = deep_diff(initial_official, initial_fast)
    if diffs:
        diff = _first_diff(diffs)
        raise DivergenceError(DivergenceReport(
            seed=seed, step=initial_official["step"], day=initial_official["day"],
            hour=initial_official["hour"], field_path=diff.path,
            official_value=diff.official_value, fast_value=diff.fast_value,
            p0_action=None, p1_action=None, phase="initial", turn_index=-1,
        ))

    turns_executed = 0
    last_pair: ActionPair = ()
    for turn in range(max_turns):
        pair = pair_for(turn, _latest_observations(fast))
        if pair is None:
            break
        last_pair = pair
        # Same pair reaches BOTH engines before any comparison; give each its
        # own copy so neither engine can mutate the other's view.
        official.step([copy.deepcopy(pair[0]), copy.deepcopy(pair[1])])
        fast.step([copy.deepcopy(pair[0]), copy.deepcopy(pair[1])])
        turns_executed = turn + 1

        official.validate_status_history()

        expected_official = official.canonical_state()
        candidate_fast = fast.canonical_state()
        if fast_mutator is not None:
            fast_mutator(candidate_fast, turn)
        diffs = deep_diff(expected_official, candidate_fast)
        if diffs:
            diff = _first_diff(diffs)
            raise DivergenceError(DivergenceReport(
                seed=seed, step=expected_official["step"], day=expected_official["day"],
                hour=expected_official["hour"], field_path=diff.path,
                official_value=diff.official_value, fast_value=diff.fast_value,
                p0_action=copy.deepcopy(pair[0]), p1_action=copy.deepcopy(pair[1]),
                phase="turn", turn_index=turn,
            ))
        if official.statuses == ["DONE", "DONE"] and fast.statuses == ["DONE", "DONE"]:
            break

    return ReplayResult(
        turns_executed=turns_executed,
        final_step=official.canonical_state()["step"],
        official_statuses=official.statuses,
        fast_statuses=fast.statuses,
        official_rewards=official.rewards,
        fast_rewards=fast.rewards,
    )


def _latest_observations(backend: EngineBackend) -> list[dict[str, Any]]:
    """Best-effort latest observation pair for action callbacks."""
    getter = getattr(backend, "observations", None)
    if callable(getter):
        return getter()
    snapshot = getattr(backend, "_env", None)
    state_snapshot = getattr(snapshot, "state_snapshot", None)
    if callable(state_snapshot):
        return state_snapshot()
    raise ValueError("backend does not expose observations for action callbacks")

"""Direct Python-vs-Rust executor parity helpers.

The comparator is deliberately action-first: it reports the first primitive
turn whose complete action object differs and never treats equal banks or
scores as a substitute for action parity.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any


def _digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                       default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ActionDivergence:
    """First mismatch in a primitive action sequence."""

    turn: int
    python_action: Any
    rust_action: Any
    observation_digest: str | None = None
    plan_digest: str | None = None
    history_digest: str | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "turn": self.turn,
            "python_action": self.python_action,
            "rust_action": self.rust_action,
            "observation_digest": self.observation_digest,
            "plan_digest": self.plan_digest,
            "history_digest": self.history_digest,
        }


@dataclass(frozen=True)
class DifferentialResult:
    """Exact result of comparing two deterministic action traces."""

    equal: bool
    turns_compared: int
    divergence: ActionDivergence | None = None
    python_final_banks: tuple[float, ...] | None = None
    rust_final_banks: tuple[float, ...] | None = None
    python_statuses: tuple[str, ...] | None = None
    rust_statuses: tuple[str, ...] | None = None
    diagnostics_equal: bool = False

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "equal": self.equal,
            "turns_compared": self.turns_compared,
            "divergence": (
                None if self.divergence is None
                else self.divergence.to_json_dict()),
            "python_final_banks": self.python_final_banks,
            "rust_final_banks": self.rust_final_banks,
            "python_statuses": self.python_statuses,
            "rust_statuses": self.rust_statuses,
            "diagnostics_equal": self.diagnostics_equal,
        }


def compare_action_sequences(
    python_actions: Sequence[Any],
    rust_actions: Sequence[Any],
    *,
    observations: Sequence[Mapping[str, Any]] | None = None,
    plans: Sequence[Any] | None = None,
    histories: Sequence[Any] | None = None,
) -> ActionDivergence | None:
    """Return the first exact action mismatch, or ``None``.

    Optional context is hashed rather than retained in the result so a
    machine-readable report stays compact while still identifying the exact
    observation/plan/history row to reduce.
    """
    limit = min(len(python_actions), len(rust_actions))
    for turn in range(limit):
        if python_actions[turn] == rust_actions[turn]:
            continue
        return ActionDivergence(
            turn=turn,
            python_action=copy.deepcopy(python_actions[turn]),
            rust_action=copy.deepcopy(rust_actions[turn]),
            observation_digest=(
                None if observations is None else _digest(observations[turn])),
            plan_digest=None if plans is None else _digest(plans[turn]),
            history_digest=(
                None if histories is None else _digest(histories[turn])),
        )
    if len(python_actions) != len(rust_actions):
        return ActionDivergence(
            turn=limit,
            python_action=("<missing>" if limit >= len(python_actions)
                           else copy.deepcopy(python_actions[limit])),
            rust_action=("<missing>" if limit >= len(rust_actions)
                         else copy.deepcopy(rust_actions[limit])),
        )
    return None


def compare_episode_results(
    python_result: Any,
    rust_result: Any,
) -> DifferentialResult:
    """Compare recorded runner results without accepting score equivalence."""
    python_rollout = python_result.rollout
    rust_rollout = rust_result.rollout
    if python_rollout is None or rust_rollout is None:
        raise ValueError("both results must have record_rollout=True")
    python_actions = [
        (step, day, hour, farmer, hands)
        for step, day, hour, farmer, hands in python_rollout.joint_actions]
    rust_actions = [
        (step, day, hour, farmer, hands)
        for step, day, hour, farmer, hands in rust_rollout.joint_actions]
    divergence = compare_action_sequences(python_actions, rust_actions)
    equal = (
        divergence is None
        and python_result.final_banks == rust_result.final_banks
        and python_result.statuses == rust_result.statuses
        and python_result.executor_diagnostics
        == rust_result.executor_diagnostics
    )
    return DifferentialResult(
        equal=equal,
        turns_compared=min(len(python_actions), len(rust_actions)),
        divergence=divergence,
        python_final_banks=tuple(python_result.final_banks),
        rust_final_banks=tuple(rust_result.final_banks),
        python_statuses=tuple(python_result.statuses),
        rust_statuses=tuple(rust_result.statuses),
        diagnostics_equal=(
            python_result.executor_diagnostics
            == rust_result.executor_diagnostics),
    )


__all__ = [
    "ActionDivergence",
    "DifferentialResult",
    "compare_action_sequences",
    "compare_episode_results",
]

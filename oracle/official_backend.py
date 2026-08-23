"""Official 1.32.7 backend adapter — oracle/evaluation use ONLY.

Importing this module (or requesting the ``"official"`` backend from
:mod:`oracle.backend`) lazily imports ``kaggle_environments`` and runs the
provenance guard. The normal fast-engine hot path never touches this module.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .provenance import require_official_modules

EXPECTED_TERMINAL_STATUSES = frozenset({"ACTIVE", "DONE"})


def status_anomalies(history: Sequence[Sequence[str]]) -> list[tuple[int, int, str]]:
    """Statuses outside {ACTIVE, DONE} anywhere in the per-step history.

    The official wrapper's terminal DONE can mask an earlier ERROR/INVALID/
    TIMEOUT inside the same run, so the FULL history decides validity.
    """
    return [
        (step_index, seat, status)
        for step_index, statuses in enumerate(history)
        for seat, status in enumerate(statuses)
        if status not in EXPECTED_TERMINAL_STATUSES
    ]


class OfficialAnomalyError(RuntimeError):
    """Raised when the official wrapper records an ERROR/INVALID/TIMEOUT status.

    The terminal DONE status can mask an earlier anomaly inside the same run;
    callers must therefore validate the FULL status history via
    :meth:`OfficialKaggricultureBackend.status_history`, not just the final
    statuses.
    """


class OfficialKaggricultureBackend:
    """Narrow adapter around ``kaggle_environments.make("kaggriculture")``."""

    name = "official"

    def __init__(self, configuration: Mapping[str, Any] | None = None) -> None:
        make = require_official_modules()
        self.configuration = dict(configuration or {})
        # Raw actions are submitted directly each turn; no agents execute.
        self._env = make("kaggriculture", configuration=self.configuration, debug=False)
        self._env.reset()
        self._rewards: list[float] = [0.0, 0.0]
        self._statuses: list[str] = [str(agent.status) for agent in self._env.state]

    @property
    def env(self) -> Any:
        return self._env

    def reset(self) -> list[dict[str, Any]]:
        self._env.reset()
        self._rewards = [0.0, 0.0]
        self._statuses = [str(agent.status) for agent in self._env.state]
        return self.observations()

    def observations(self) -> list[dict[str, Any]]:
        return [dict(agent.observation) for agent in self._env.state]

    def step(self, actions: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[float], list[str]]:
        if len(actions) != 2:
            raise ValueError("official backend requires exactly two actions")
        state = self._env.step([actions[0], actions[1]])
        self._statuses = [str(agent.status) for agent in state]
        if self._env.done:
            # Interpreter sets reward only at DONE; keep last-known otherwise.
            self._rewards = [float(agent.reward if agent.reward is not None else 0.0)
                             for agent in state]
        return self.observations(), list(self._rewards), list(self._statuses)

    @property
    def rewards(self) -> list[float]:
        return list(self._rewards)

    @property
    def statuses(self) -> list[str]:
        return list(self._statuses)

    def status_history(self) -> list[list[str]]:
        """Full per-step status history including any pre-terminal anomalies."""
        history = []
        for step_states in self._env.steps:
            history.append([str(agent.status) for agent in step_states])
        return history

    def validate_status_history(self) -> None:
        anomalies = status_anomalies(self.status_history())
        if anomalies:
            raise OfficialAnomalyError(
                f"official run invalid; non-expected statuses recorded "
                f"(terminal DONE can mask these): {anomalies}"
            )

    def canonical_state(self) -> dict[str, Any]:
        from .canonical import canonical_state_official

        return canonical_state_official(
            self._env, rewards=self.rewards, statuses=self.statuses
        )

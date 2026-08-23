"""Runtime opening-book wrapper (issue #4, stage 2).

Replays one built-in elite trace literally for days 0-3 (exactly 96 turns),
then delegates unchanged to an injected downstream agent starting exactly at
day 4 hour 0. Guards are fail-closed and minimal:

- observation ``day``/``hour`` must parse and equal the expected trace cursor;
- observed hand count for the configured seat must match the trace action's
  ``hands`` length;
- every emitted action passes Stage 1 shape/market-cap validation
  (``opening_book.trace.validate_action``).

Any guard failure records one divergence (reason + observed turn), captures a
best-effort farm summary, and immediately delegates the same original
observation to downstream. Delegation is permanent: the script never resumes
after divergence or handoff.

Bounded contract: the wrapper never seeds or mutates downstream private
state, never touches ``previous_execution``, and performs no planning,
repair, or heuristic purchases. It is literal playback plus delegation only.
No official-engine claim is made by this module (stage 3 validates).
"""

from __future__ import annotations

import json
from typing import Any, Callable, Mapping

from .trace import (
    DEFAULT_IDENTITY,
    FIRST_DAY,
    LAST_DAY,
    TURNS_PER_DAY,
    TraceError,
    action_for,
    load_built_in_trace,
    validate_action,
)

__all__ = ["OpeningAgent", "make_opening_agent"]

AgentCallable = Callable[[Mapping], Mapping]

_DIAGNOSTICS_SCHEMA_VERSION = 1

# Divergence reasons (stable string contract for stage 3 evaluation logs).
DIVERGENCE_MALFORMED_PHASE = "malformed_phase"
DIVERGENCE_PHASE_MISMATCH = "phase_mismatch"
DIVERGENCE_HAND_COUNT = "hand_count_mismatch"
DIVERGENCE_ACTION_INVALID = "action_invalid"
DIVERGENCE_OBSERVATION_INCOMPLETE = "observation_incomplete"


def _parse_phase(obs: Any) -> tuple[int, int] | None:
    """Return (day, hour) if both fields parse as ints, else None."""
    if not isinstance(obs, Mapping):
        return None
    try:
        day = int(obs["day"])
        hour = int(obs["hour"])
    except (KeyError, TypeError, ValueError):
        return None
    if isinstance(obs["day"], bool) or isinstance(obs["hour"], bool):
        return None
    return day, hour


def _crop_animal_counts(tiles: Any) -> tuple[dict[str, int], dict[str, int]]:
    """Best-effort crop/animal counts over one farm's tiles grid."""
    crops: dict[str, int] = {}
    animals: dict[str, int] = {}
    if not isinstance(tiles, list):
        return crops, animals
    for row in tiles:
        if not isinstance(row, list):
            continue
        for tile in row:
            if not isinstance(tile, dict):
                continue
            animal = tile.get("animal")
            if isinstance(animal, str):
                animals[animal] = animals.get(animal, 0) + 1
            elif tile.get("kind") == "PLANT":
                crop = tile.get("crop")
                if isinstance(crop, str):
                    crops[crop] = crops.get(crop, 0) + 1
    return crops, animals


class OpeningAgent:
    """Stateful callable: literal trace playback for d0-d3, then delegate."""

    def __init__(self, trace: Mapping[str, Any], downstream: AgentCallable,
                 seat: int) -> None:
        if not callable(downstream):
            raise TypeError(f"downstream must be callable, got {downstream!r}")
        if seat not in (0, 1):
            raise ValueError(f"seat must be 0 or 1, got {seat!r}")
        self._trace = trace
        self._identity = trace["identity"]
        self._provenance = dict(trace.get("provenance") or {})
        self._downstream = downstream
        self.seat = seat

        # ---- per-game state --------------------------------------------
        self._cursor: int = 0  # index into trace["turns"]
        self._turns_replayed: int = 0
        self._delegated_calls: int = 0
        self._permanent_delegation: bool = False
        self._clean_handoff_done: bool = False
        self._divergence_reason: str | None = None
        self._divergence_turn: list[int] | None = None
        self._handoff_turn: list[int] | None = None
        self._handoff_farm_summary: dict[str, Any] = {}

    # ------------------------------------------------------------- kaggle
    def __call__(self, obs: Mapping) -> dict[str, Any]:
        if self._permanent_delegation:
            return self._delegate(obs)

        phase = _parse_phase(obs)
        if phase is None:
            return self._diverge(obs, DIVERGENCE_MALFORMED_PHASE)

        day, hour = phase
        if day > LAST_DAY or (day == LAST_DAY and hour >= TURNS_PER_DAY):
            # Beyond the trace horizon: clean handoff (never a divergence).
            return self._handoff(obs, day, hour)

        expected_day = FIRST_DAY + self._cursor // TURNS_PER_DAY
        expected_hour = self._cursor % TURNS_PER_DAY
        if (day, hour) != (expected_day, expected_hour):
            return self._diverge(
                obs, DIVERGENCE_PHASE_MISMATCH,
                observed_turn=[day, hour],
                detail=f"expected ({expected_day}, {expected_hour})",
            )

        try:
            action = action_for(dict(self._trace), day, hour)
        except TraceError as exc:  # pragma: no cover - validated traces only
            return self._diverge(obs, DIVERGENCE_ACTION_INVALID,
                                 observed_turn=[day, hour], detail=str(exc))

        try:
            validate_action(action, label=f"opening (day={day}, hour={hour})")
        except TraceError as exc:
            return self._diverge(obs, DIVERGENCE_ACTION_INVALID,
                                 observed_turn=[day, hour], detail=str(exc))

        farm = None
        if isinstance(obs, Mapping):
            farms = obs.get("farms")
            if isinstance(farms, list) and len(farms) > self.seat \
                    and isinstance(farms[self.seat], Mapping):
                farm = farms[self.seat]
        if farm is None:
            return self._diverge(obs, DIVERGENCE_OBSERVATION_INCOMPLETE,
                                 observed_turn=[day, hour],
                                 detail=f"farms[{self.seat}] missing")

        observed_hands = farm.get("hands") or []
        if not isinstance(observed_hands, list) \
                or len(observed_hands) != len(action["hands"]):
            return self._diverge(
                obs, DIVERGENCE_HAND_COUNT,
                observed_turn=[day, hour],
                detail=(
                    f"observed {len(observed_hands)} hands, "
                    f"trace expects {len(action['hands'])}"
                ),
            )

        # All guards passed: emit the literal scripted action.
        self._cursor += 1
        self._turns_replayed += 1
        return action

    # -------------------------------------------------------------- core
    def _delegate(self, obs: Mapping) -> dict[str, Any]:
        """Pass the observation through untouched; return downstream's result."""
        self._delegated_calls += 1
        return self._downstream(obs)

    def _handoff(self, obs: Mapping, day: int, hour: int) -> dict[str, Any]:
        self._permanent_delegation = True
        self._clean_handoff_done = True
        self._handoff_turn = [day, hour]
        self._handoff_farm_summary = self._farm_summary(obs)
        return self._delegate(obs)

    def _diverge(self, obs: Mapping, reason: str,
                 observed_turn: list[int] | None = None,
                 detail: str = "") -> dict[str, Any]:
        self._permanent_delegation = True
        if self._divergence_reason is None:
            self._divergence_reason = (
                f"{reason}: {detail}" if detail else reason
            )
            self._divergence_turn = observed_turn
        if self._handoff_turn is None:
            phase = _parse_phase(obs)
            self._handoff_turn = list(phase) if phase else None
            self._handoff_farm_summary = self._farm_summary(obs)
        return self._delegate(obs)

    def _farm_summary(self, obs: Mapping) -> dict[str, Any]:
        """Best-effort handoff snapshot; missing fields stay None/empty."""
        summary: dict[str, Any] = {
            "money": None,
            "crops": {},
            "animals": {},
            "land_count": None,
            "shed_wheat": None,
        }
        if not isinstance(obs, Mapping):
            return summary
        farms = obs.get("farms")
        farm = None
        if isinstance(farms, list) and len(farms) > self.seat \
                and isinstance(farms[self.seat], Mapping):
            farm = farms[self.seat]
        if farm is not None:
            money = farm.get("money")
            if isinstance(money, (int, float)) \
                    and not isinstance(money, bool):
                summary["money"] = float(money)
            crops, animals = _crop_animal_counts(farm.get("tiles"))
            summary["crops"] = crops
            summary["animals"] = animals
            quadrants = farm.get("unlocked_quadrants")
            if isinstance(quadrants, list):
                summary["land_count"] = len(quadrants)
        private = obs.get("private")
        if isinstance(private, Mapping):
            shed = private.get("shed")
            if isinstance(shed, Mapping):
                wheat = shed.get("WHEAT")
                if isinstance(wheat, (int, float)) \
                        and not isinstance(wheat, bool):
                    summary["shed_wheat"] = int(wheat)
        return summary

    # ------------------------------------------------------------ diagnostics
    def diagnostics_json(self) -> dict[str, Any]:
        """Deterministic JSON-serializable diagnostics for this game so far."""
        return {
            "schema_version": _DIAGNOSTICS_SCHEMA_VERSION,
            "opening": self._identity,
            "source_provenance": {
                "episode": self._provenance.get("source_episode"),
                "seed": self._provenance.get("source_seed"),
                "player": self._provenance.get("source_player"),
                "source_seat": self._provenance.get("source_seat"),
                "source_replay_sha256":
                    self._provenance.get("source_replay_sha256"),
            },
            "seat": self.seat,
            "turns_replayed": self._turns_replayed,
            "expected_next_turn": (
                [FIRST_DAY + self._cursor // TURNS_PER_DAY,
                 self._cursor % TURNS_PER_DAY]
                if not self._permanent_delegation else None
            ),
            "divergence": {
                "occurred": self._divergence_reason is not None,
                "reason": self._divergence_reason,
                "observed_turn": (
                    list(self._divergence_turn)
                    if self._divergence_turn is not None else None
                ),
            },
            "fallback_active": self._divergence_reason is not None,
            "delegated_calls": self._delegated_calls,
            "handoff": {
                "turn": list(self._handoff_turn)
                if self._handoff_turn is not None else None,
                "clean_d4h0_handoff": self._clean_handoff_done,
                "farm_summary": {
                    key: (dict(value) if isinstance(value, dict) else value)
                    for key, value in self._handoff_farm_summary.items()
                },
            },
        }


def make_opening_agent(opening: str = DEFAULT_IDENTITY,
                       downstream: AgentCallable | None = None,
                       seat: int | None = None) -> OpeningAgent:
    """Build an :class:`OpeningAgent` from a built-in opening identity.

    Rejects unknown identities, missing/non-callable downstream, and seats
    other than 0/1 immediately at construction.
    """
    if downstream is None:
        raise ValueError("downstream agent is required")
    if seat is None:
        raise ValueError("explicit seat is required (0 or 1)")
    trace = load_built_in_trace(opening)
    return OpeningAgent(trace, downstream, seat)

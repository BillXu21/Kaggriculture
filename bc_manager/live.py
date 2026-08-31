"""Live 1.32.7 observation -> BC model-facing input arrays (issue #1, stage 1).

Smallest reusable encoder for one seat/day. It reuses `replay_daily.extractor`
canonical state construction and the authoritative `bc_manager.adapter` array
encoding (via the shared `_input_arrays_from_starts` implementation), so the
live path and the canonical-record -> adapter path cannot drift; exact parity
is enforced by `tests/test_bc_manager_live.py`.

Contract:

- Input: one raw live 1.32.7 observation for one seat, the seat index, the
  prior-day realized labor state (`previous_execution`), an optional explicit
  lifecycle `step` override, and the same `include_opponent` flag semantics as
  `bc_manager.adapter` / `ManagerConfig` (public opponent board only).
- Output: exactly the model-facing input arrays of `bc_manager.adapter` for an
  equivalent canonical daily record, as one-row NumPy arrays with identical
  keys/shapes/dtypes/semantics.
- Day 0 uses deterministic zero previous labor by default; later days must
  carry the exact `workers_hired` / `hire_cost` observed on the previous day.
- Framework-only observation fields are ignored. Opponent private state is
  never read. No metadata/score/name/final-bank values are model inputs.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from replay_daily.extractor import (
    opponent_public_state,
    self_state,
    shared_state,
)
from replay_daily.storage import (
    normalize_public_state,
    normalize_self_state,
    normalize_shared_state,
)

from .adapter import _input_arrays_from_starts
from .economics import (
    E_HISTORY_CORRECTED_V1,
    ECONOMIC_CONTEXT_KEY,
    EconomicHistory,
    economic_context,
    normalize_e_history_version,
    previous_net_cash,
)

__all__ = [
    "encode_live_inputs",
    "validate_previous_execution",
    "EconomicHistory",
]

_PREVIOUS_EXECUTION_KEYS = ("workers_hired", "hire_cost")
_REQUIRED_OBS_KEYS = ("farms", "market", "town", "day", "hour")


def _require_int(value: Any, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{what} must be an integer, got {value!r}")
    if value < 0:
        raise ValueError(f"{what} must be nonnegative, got {int(value)}")
    return int(value)


def validate_previous_execution(
    value: Mapping[str, Any] | None = None,
) -> dict[str, int]:
    """Validate/copy prior-day labor state; None means deterministic zeros."""
    if value is None:
        return {"workers_hired": 0, "hire_cost": 0}
    if not isinstance(value, Mapping):
        raise ValueError(
            f"previous_execution must be a mapping with keys "
            f"{list(_PREVIOUS_EXECUTION_KEYS)}, got {type(value).__name__}")
    unknown = sorted(set(value) - set(_PREVIOUS_EXECUTION_KEYS))
    missing = sorted(set(_PREVIOUS_EXECUTION_KEYS) - set(value))
    if unknown or missing:
        raise ValueError(
            f"previous_execution key mismatch; unknown={unknown}, "
            f"missing={missing}; expected exactly "
            f"{list(_PREVIOUS_EXECUTION_KEYS)}")
    return {
        key: _require_int(value[key], f"previous_execution.{key}")
        for key in _PREVIOUS_EXECUTION_KEYS
    }


def encode_live_inputs(
    obs: Mapping[str, Any],
    seat: int,
    previous_execution: Mapping[str, int] | None = None,
    *,
    include_opponent: bool = False,
    step: int | None = None,
    economic_history: EconomicHistory | None = None,
    economic_prev_start: tuple[int, float] | None = None,
    e_history_version: str | None = None,
) -> dict[str, np.ndarray]:
    """Encode one raw live 1.32.7 observation into one-row BC input arrays.

    `step` resolution order: explicit argument, then `obs["step"]`. Compiled
    replays may omit per-seat `step`; live engine observations carry it. There
    is no silent default because lifecycle timing (`past_lifespan`) depends on
    it.

    Economic context (issue #6 variant E) is emitted only when explicitly
    requested via exactly one of:

    - `economic_history`: an `EconomicHistory` tracker for the current
      episode/seat; it records this daily-start (day, money) observation and
      derives the previous-day delta only from its own earlier recorded
      daily-start state (never from actions). Reset it on a new episode.
    - `economic_prev_start`: an explicit `(prev_day, prev_money)` pair for
      the prior daily-start observation; valid iff `prev_day == day - 1`.

    Passing both raises. With neither, no `economic_context` key is emitted
    and the V0 encoding is byte-identical to before.

    Raises ValueError with a clear message on missing mechanically required
    fields or invalid labor/day/step values.
    """
    if economic_history is not None and economic_prev_start is not None:
        raise ValueError(
            "pass either economic_history or economic_prev_start, not both")
    if e_history_version is not None and economic_history is not None:
        raise ValueError(
            "e_history_version uses explicit runner-owned history; do not also "
            "pass economic_history")
    if e_history_version is not None:
        e_history_version = normalize_e_history_version(e_history_version)
    if not isinstance(obs, Mapping):
        raise ValueError(
            f"obs must be a mapping, got {type(obs).__name__}")
    if seat not in (0, 1):
        raise ValueError(f"seat must be 0 or 1, got {seat!r}")
    for key in _REQUIRED_OBS_KEYS:
        if key not in obs:
            raise ValueError(f"live observation is missing required field {key!r}")
    farms = obs["farms"]
    if not isinstance(farms, Sequence) or isinstance(farms, (str, bytes)) \
            or len(farms) < 2:
        raise ValueError(
            f"obs['farms'] must hold both seats, got {farms!r:.80}")
    for idx in (seat, 1 - seat):
        if not isinstance(farms[idx], Mapping):
            raise ValueError(f"obs['farms'][{idx}] must be a mapping")

    day = _require_int(obs["day"], "obs['day']")
    hour = _require_int(obs["hour"], "obs['hour']")
    resolved_step: int | None
    if step is not None:
        resolved_step = _require_int(step, "step")
    elif obs.get("step") is not None:
        resolved_step = _require_int(obs["step"], "obs['step']")
    else:
        raise ValueError(
            "live observation has no 'step' field; pass step= explicitly "
            "(lifecycle timing depends on it)")
    prev = validate_previous_execution(previous_execution)

    start: dict[str, Any] = {
        "day": day,
        "hour": hour,
        "self": normalize_self_state(
            self_state(dict(obs), seat, day, resolved_step)),
        **normalize_shared_state(shared_state(dict(obs)), "live"),
        "previous_execution": prev,
    }
    if include_opponent:
        start["opponent_public"] = normalize_public_state(
            opponent_public_state(dict(obs), seat, day, resolved_step))
    inputs = _input_arrays_from_starts([start], [day],
                                       include_opponent=include_opponent)
    if (economic_history is not None or economic_prev_start is not None
            or e_history_version is not None):
        money = float(start["self"]["money"])
        unlocked_count = len(start["self"]["unlocked_quadrants"])
        if economic_history is not None:
            delta, valid = economic_history.observe(day, money)
        elif e_history_version is not None:
            delta, valid = previous_net_cash(
                e_history_version, day, money, economic_prev_start)
        else:
            prev_day, prev_money = economic_prev_start
            delta, valid = previous_net_cash(
                E_HISTORY_CORRECTED_V1, day, money,
                (int(prev_day), float(prev_money)))
        inputs[ECONOMIC_CONTEXT_KEY] = economic_context(
            money, unlocked_count, delta if valid else None)[None, :]
    return inputs

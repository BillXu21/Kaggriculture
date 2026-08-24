"""GAE(λ) over Stage-A manager-day trajectory rows (issue #9 B4, req. 10).

Pure NumPy host-side computation over the compact `TrajectoryBuffer` arrays.
Rows are grouped by `(episode_index, seat)` and processed in day order;
Stage-A rows may interleave arbitrarily across episodes/seats. Invalid
sequences fail loudly:

- duplicate days within one (episode, seat) group;
- non-contiguous manager days within one group (gap != 1);
- a mid-group terminal/truncated flag (episode must end at its last row);
- an incomplete episode whose last row is neither terminated nor truncated;
- a truncated episode without explicit bootstrap values.

Terminal episodes bootstrap 0; truncated episodes consume the explicit
`bootstrap_values` entry aligned to their last row. Advantages are
normalized once over the full valid batch by default with an additive
epsilon-stable denominator.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np


def _as_1d(array: object, name: str) -> np.ndarray:
    out = np.asarray(array)
    if out.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {out.shape}")
    return out


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    episode_index: np.ndarray,
    seat: np.ndarray,
    day: np.ndarray,
    *,
    gamma: float,
    gae_lambda: float,
    bootstrap_values: np.ndarray | None = None,
    normalize: bool = True,
    epsilon: float = 1e-8,
) -> dict[str, np.ndarray]:
    """Compute advantages and returns for N trajectory rows.

    Returns `{"advantages": [N] float32, "returns": [N] float32}`.
    `gamma` is the discount and `gae_lambda` the GAE decay; both are plain
    plumbing defaults chosen by the caller (issue #9 defaults 0.99/0.95).
    """
    rewards = _as_1d(rewards, "rewards").astype(np.float64)
    values = _as_1d(values, "values").astype(np.float64)
    terminated = _as_1d(terminated, "terminated")
    truncated = _as_1d(truncated, "truncated")
    episode_index = _as_1d(episode_index, "episode_index")
    seat = _as_1d(seat, "seat")
    day = _as_1d(day, "day")
    n = rewards.shape[0]
    for name, array in (("values", values), ("terminated", terminated),
                        ("truncated", truncated),
                        ("episode_index", episode_index), ("seat", seat),
                        ("day", day)):
        if array.shape[0] != n:
            raise ValueError(
                f"{name} length {array.shape[0]} != rewards length {n}")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"gamma must be in [0, 1], got {gamma}")
    if not 0.0 <= gae_lambda <= 1.0:
        raise ValueError(f"gae_lambda must be in [0, 1], got {gae_lambda}")
    if not np.all(np.isfinite(rewards)) or not np.all(np.isfinite(values)):
        raise ValueError("rewards/values contain non-finite entries")

    groups: dict[tuple, list[int]] = {}
    for row in range(n):
        groups.setdefault((episode_index[row], seat[row]), []).append(row)

    advantages = np.zeros(n, dtype=np.float64)
    for key, rows in groups.items():
        rows.sort(key=lambda r: day[r])
        days = [int(day[r]) for r in rows]
        if any(days[i] == days[i - 1] for i in range(1, len(days))):
            raise ValueError(
                f"episode/seat {key}: duplicate manager day in trajectory "
                f"rows {rows}")
        if any(days[i] - days[i - 1] != 1 for i in range(1, len(days))):
            raise ValueError(
                f"episode/seat {key}: gapped manager days {days} (manager "
                f"days must be contiguous)")
        last = len(rows) - 1
        for position in range(last):
            row = rows[position]
            if terminated[row] or truncated[row]:
                raise ValueError(
                    f"episode/seat {key}: mid-group terminal/truncated flag "
                    f"at day {days[position]}; episodes must end at their "
                    f"last row")
        end_row = rows[last]
        if not (terminated[end_row] or truncated[end_row]):
            raise ValueError(
                f"episode/seat {key}: incomplete episode ends at day "
                f"{days[last]} without terminal/truncated flag")
        if truncated[end_row]:
            if bootstrap_values is None:
                raise ValueError(
                    f"episode/seat {key}: truncated episode requires "
                    f"explicit bootstrap_values; refusing to guess")
            bootstrap = float(np.asarray(bootstrap_values)[end_row])
        else:
            bootstrap = 0.0

        adv_next = 0.0
        for position in range(last, -1, -1):
            row = rows[position]
            v_next = bootstrap if position == last else values[rows[position + 1]]
            delta = rewards[row] + gamma * v_next - values[row]
            adv_next = delta + gamma * gae_lambda * adv_next
            advantages[row] = adv_next

    returns = advantages + values
    if normalize:
        std = float(np.std(advantages))
        advantages = (advantages - np.mean(advantages)) / (std + epsilon)
    return {"advantages": advantages.astype(np.float32),
            "returns": returns.astype(np.float32)}


def advantage_stats(advantages: np.ndarray) -> dict[str, float]:
    """Compact advantage diagnostics for PPO logging."""
    array = np.asarray(advantages, dtype=np.float64)
    return {"adv_mean": float(np.mean(array)),
            "adv_std": float(np.std(array)),
            "adv_min": float(np.min(array)),
            "adv_max": float(np.max(array))}


def valid_trainable_rows(arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    """Indices of candidate/trainable, valid trajectory rows (contiguous)."""
    mask = (np.asarray(arrays["valid"]) == 1) & \
           (np.asarray(arrays["trainable"]) == 1)
    return np.flatnonzero(mask)

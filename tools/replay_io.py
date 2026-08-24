"""Raw Kaggriculture replay loading and episode-configuration derivation.

Raw replays are Kaggle episode JSON dicts with keys ``configuration``,
``info``, ``rewards``, and ``steps`` (720 steps, one entry per player).
This module only reads and validates; it never mutates the replay dict.
"""

from __future__ import annotations

import json
from typing import Any

__all__ = [
    "EXPECTED_EPISODE_STEPS",
    "EXPECTED_TURNS_PER_DAY",
    "episode_configuration",
    "episode_id",
    "load_replay",
]

EXPECTED_TURNS_PER_DAY = 24
EXPECTED_EPISODE_STEPS = 720


def load_replay(path: str) -> dict[str, Any]:
    """Load one raw replay JSON file with clear error messages."""
    try:
        with open(path, encoding="utf-8") as f:
            replay = json.load(f)
    except OSError as error:
        raise ValueError(f"cannot read replay file {path!r}: {error}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"replay file {path!r} is not valid JSON: {error}") from error
    if not isinstance(replay, dict):
        raise ValueError(f"replay file {path!r} must contain a JSON object")
    return replay


def _require_int(value: Any, label: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"configuration {label} must be an integer, got {value!r}"
        ) from error


def episode_configuration(replay: dict[str, Any]) -> dict[str, Any]:
    """Configuration dict suitable for ``fast_env.api.FastKaggricultureEnv``.

    Merges ``replay["configuration"]`` with ``seed`` taken from
    ``replay["info"]["seed"]`` (falling back to ``configuration["seed"]``).
    Validates the pinned default contract: turnsPerDay == 24 and
    episodeSteps == 720; raises ValueError otherwise.
    """
    config = dict(replay.get("configuration") or {})
    info = replay.get("info") or {}
    seed = info.get("seed")
    if seed is None:
        seed = config.get("seed")
    if seed is None:
        raise ValueError(
            "replay has no seed in info.seed nor configuration.seed"
        )
    config["seed"] = seed

    turns_per_day = _require_int(config.get("turnsPerDay"), "turnsPerDay")
    if turns_per_day != EXPECTED_TURNS_PER_DAY:
        raise ValueError(
            f"expected turnsPerDay == {EXPECTED_TURNS_PER_DAY}, got "
            f"{turns_per_day}"
        )
    episode_steps = _require_int(config.get("episodeSteps"), "episodeSteps")
    if episode_steps != EXPECTED_EPISODE_STEPS:
        raise ValueError(
            f"expected episodeSteps == {EXPECTED_EPISODE_STEPS}, got "
            f"{episode_steps}"
        )
    return config


def episode_id(replay: dict[str, Any]) -> int:
    """Integer episode id from ``replay["info"]["EpisodeId"]``."""
    info = replay.get("info") or {}
    raw = info.get("EpisodeId")
    if raw is None:
        raise ValueError("replay info.EpisodeId is missing")
    return int(raw)

"""Evaluation-driven retention of named PPO best checkpoints."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from rl_manager.ppo_checkpoint import save_ppo_checkpoint


def _checkpoint_score(path: Path) -> float | None:
    """Read the prior retained score without reconstructing JAX state."""
    try:
        with np.load(path, allow_pickle=False) as archive:
            meta = json.loads(archive["__meta__"].tobytes().decode("utf-8"))
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot inspect retained checkpoint {path}: {exc}") \
            from exc
    retention = meta.get("provenance", {}).get("best_retention", {})
    score = retention.get("score")
    return None if score is None else float(score)


class BestCheckpointRetention:
    """Maintain overwrite-only named bests using deterministic evaluations.

    Scores are maximized. The checkpoint metadata contains both the exact
    evaluation payload and the source provenance supplied by the caller.
    """

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def path_for(self, name: str) -> Path:
        if not name or Path(name).name != name or name in {".", ".."}:
            raise ValueError(f"best checkpoint name must be a simple filename, got {name!r}")
        return self.directory / f"best_{name}.npz"

    def consider(
        self,
        name: str,
        score: float,
        state,
        config,
        ppo_config,
        *,
        evaluation: Mapping[str, Any],
        provenance: Mapping[str, Any] | None = None,
        model_variant: str = "E",
    ) -> bool:
        """Save ``state`` only when ``score`` strictly improves that named best."""
        score = float(score)
        if not math.isfinite(score):
            raise ValueError(f"best checkpoint score must be finite, got {score}")
        path = self.path_for(name)
        if path.exists():
            previous = _checkpoint_score(path)
            if previous is not None and score <= previous:
                return False
        save_ppo_checkpoint(
            path, state, config, ppo_config, model_variant=model_variant,
            provenance={
                "best_retention": {
                    "name": name,
                    "score": score,
                    "evaluation": dict(evaluation),
                    "source_provenance": (dict(provenance)
                                           if provenance else {}),
                }
            })
        return True


__all__ = ["BestCheckpointRetention"]

"""Explicit reproducible seed streams (issue #9, architecture req. 8).

Every seed is a pure function of `(master_seed, purpose_tag, index)` via
`np.random.SeedSequence` entropy composition, so episode/policy ownership is
independent of worker scheduling, interleaving order, or how many episodes
happen to run before any other one. The derived identifiers are persisted in
trajectory sidecars so any run can be reproduced exactly.
"""

from __future__ import annotations

import numpy as np

_EPISODE_TAG = 1
_POLICY_TAG = 2
_ENVIRONMENT_TAG = 3
_INITIALIZATION_TAG = 4


class SeedStream:
    """Deterministic master-seed -> episode/policy/environment seed stream."""

    def __init__(self, master_seed: int) -> None:
        if isinstance(master_seed, bool) or not isinstance(master_seed, int):
            raise ValueError(f"master_seed must be an int, got {master_seed!r}")
        if master_seed < 0:
            raise ValueError(f"master_seed must be nonnegative, got {master_seed}")
        self.master_seed = int(master_seed)

    def _derive(self, tag: int, index: int) -> int:
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError(f"stream index must be a nonnegative int, got {index!r}")
        sequence = np.random.SeedSequence([self.master_seed, tag, index])
        return int(sequence.generate_state(1, dtype=np.uint32)[0])

    def episode_seed(self, episode_index: int) -> int:
        """Engine seed for one episode; pure in (master_seed, episode_index)."""
        return self._derive(_EPISODE_TAG, episode_index)

    def policy_seed(self, role: str, snapshot_index: int = 0) -> int:
        """PRNG seed identifier for one named policy role (e.g. 'candidate')."""
        if not isinstance(role, str) or not role:
            raise ValueError(f"policy role must be a non-empty string, got {role!r}")
        return self._derive(_POLICY_TAG, snapshot_index)

    def environment_seed(self, env_index: int) -> int:
        """Reserved future knob for per-worker environment streams."""
        return self._derive(_ENVIRONMENT_TAG, env_index)

    def initialization_seed(self, index: int = 0) -> int:
        """Derive a deterministic seed for model initialization."""
        return self._derive(_INITIALIZATION_TAG, index)

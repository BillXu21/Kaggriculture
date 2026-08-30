"""Official-vs-fast differential oracle for Kaggriculture 1.32.7.

Public surface:
- :func:`oracle.make_backend` — narrow official/fast engine seam
- :func:`oracle.run_same_action_replay` — same-action turn-by-turn replay
- :class:`oracle.DivergenceError` / :class:`oracle.DivergenceReport`
- :func:`oracle.verify_official_provenance`
- canonical comparison helpers in :mod:`oracle.canonical`

The official backend is oracle/evaluation use only; importing this package is
lazy about ``kaggle_environments`` and never makes the fast hot path depend on
it.
"""

from .backend import FastBackendAdapter, make_backend
from .canonical import FieldDiff, canonical_state_fast, canonical_state_official, deep_diff
from .provenance import ProvenanceError, verify_official_provenance
from .replay import DivergenceError, DivergenceReport, ReplayResult, run_same_action_replay
from .closed_loop import (
    ClosedLoopDivergenceError,
    ClosedLoopDivergenceReport,
    ClosedLoopResult,
    make_checkpoint_executor_factory,
    make_current_control_factory,
    make_deterministic_executor_factory,
    run_closed_loop,
)

__all__ = [
    "FastBackendAdapter",
    "make_backend",
    "FieldDiff",
    "canonical_state_fast",
    "canonical_state_official",
    "deep_diff",
    "ProvenanceError",
    "verify_official_provenance",
    "DivergenceError",
    "DivergenceReport",
    "ReplayResult",
    "run_same_action_replay",
    "ClosedLoopDivergenceError",
    "ClosedLoopDivergenceReport",
    "ClosedLoopResult",
    "make_checkpoint_executor_factory",
    "make_current_control_factory",
    "make_deterministic_executor_factory",
    "run_closed_loop",
]

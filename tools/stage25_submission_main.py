"""Kaggle entrypoint for a native Stage 2.5 inference archive."""

from pathlib import Path
import sys

from rl_manager.stage25_submission import make_stage25_submission_agent

def _archive_root():
    """Find the extracted archive directory under Kaggle's raw-code loader."""
    candidates = [Path(entry or Path.cwd()) for entry in reversed(sys.path)]
    candidates.append(Path.cwd())
    for candidate in candidates:
        resolved = candidate.resolve()
        if (resolved / "stage25.npz").is_file():
            return resolved
    raise FileNotFoundError(
        "stage25.npz is not present beside the Kaggle submission entrypoint")


_ROOT = _archive_root()
_CHECKPOINT = _ROOT / "stage25.npz"
_agent = None


def agent(observation, configuration=None):
    """Kaggle-compatible one- or two-argument stateful agent callable."""
    global _agent
    if _agent is None:
        seat = int(observation["player"])
        _agent = make_stage25_submission_agent(_CHECKPOINT, seat=seat)
    return _agent(observation, configuration)


def _kaggle_submission_entrypoint(observation, configuration=None):
    return agent(observation, configuration)

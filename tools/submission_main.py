"""Tracked canonical Kaggle entrypoint for the BC-E V0.7 archive."""

import os
from pathlib import Path

import executor_v0
from executor_v0 import AgentConfig, make_agent
from opening_book.agent import make_opening_agent
from oracle.closed_loop import _executor_observation

_ROOT = Path(executor_v0.__file__).resolve().parent.parent
_CHECKPOINT = _ROOT / "best.pt"
_agent = None


def _strict_from_environment() -> bool:
    value = os.environ.get("KAGGRICULTURE_SUBMISSION_STRICT", "0")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def agent(obs, configuration=None):
    """Return one action from one process-local stateful submission agent."""
    del configuration
    global _agent

    adapted = _executor_observation(obs, from_fast=False)
    if _agent is None:
        seat = int(adapted["player"])
        downstream = make_agent(
            checkpoint=str(_CHECKPOINT),
            device="cpu",
            seat=seat,
            config=AgentConfig(
                strict=_strict_from_environment(),
                suppress_expansion_from_prior_debt=True,
                aggressive_sell_all=True,
                optional_spare_watering=True,
            ),
        )
        _agent = make_opening_agent(
            opening="standard_mixed",
            downstream=downstream,
            seat=seat,
        )
    return _agent(adapted)


def _kaggle_submission_entrypoint(obs, configuration=None):
    return agent(obs, configuration)

"""Asymmetric primitive-controller evaluation harness."""

from .agent_match import (
    ControllerFactory,
    MatchResult,
    PrimitiveController,
    run_match,
    run_panel,
)
from .external import ExternalControllerFactory, bundle_digest
from .internal import (
    InternalControllerFactory,
    PassControllerFactory,
    load_internal_factory,
    make_agent_config,
)

__all__ = [
    "ControllerFactory",
    "MatchResult",
    "PrimitiveController",
    "run_match",
    "run_panel",
    "ExternalControllerFactory",
    "bundle_digest",
    "InternalControllerFactory",
    "PassControllerFactory",
    "load_internal_factory",
    "make_agent_config",
]

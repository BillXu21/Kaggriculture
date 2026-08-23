"""Elite opening-book primitive-action traces (issue #4, stage 1).

Stage 1 scope: trace contract, deterministic extraction from one raw replay,
runtime-readable validation/loading, and two committed identities generated
from verified local raw replays. No runtime wrapper/handoff here (stage 2).
"""

from .trace import (
    DEFAULT_IDENTITY,
    ENGINE_VERSION,
    EXPECTED_TURNS,
    FIRST_DAY,
    LAST_DAY,
    MAX_MARKET_ORDERS,
    TURNS_PER_DAY,
    TRACE_FORMAT_VERSION,
    TraceError,
    action_for,
    built_in_identities,
    compute_content_digest,
    load_built_in_trace,
    validate_action,
    validate_trace,
)

__all__ = [
    "DEFAULT_IDENTITY",
    "ENGINE_VERSION",
    "EXPECTED_TURNS",
    "FIRST_DAY",
    "LAST_DAY",
    "MAX_MARKET_ORDERS",
    "TRACE_FORMAT_VERSION",
    "TURNS_PER_DAY",
    "TraceError",
    "action_for",
    "built_in_identities",
    "compute_content_digest",
    "load_built_in_trace",
    "validate_action",
    "validate_trace",
]

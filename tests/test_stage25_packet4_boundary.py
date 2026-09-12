"""Root-owned integration assertions for the Packet 4 lifecycle seam."""

from __future__ import annotations

import pytest

from rl_manager.stage25_provider import (
    Stage25DuplicateDecisionError,
    Stage25PlanProvider,
)

from test_stage25_provider import HOLD, _obs


def test_identity_bearing_cache_read_is_not_a_benign_read() -> None:
    provider = Stage25PlanProvider(7, 0, 3)
    provider.accept_classes(_obs(), HOLD)
    before = provider.export_state()

    # A no-identity read is benign; a new identity on the accepted day is a
    # conflicting duplicate and must not mutate lifecycle state.
    assert provider.daily_plan(_obs(money=999.0), 0) is provider.cached_plan
    with pytest.raises(Stage25DuplicateDecisionError):
        provider.daily_plan(_obs(), 0, decision_id="different-row")
    assert provider.export_state() == before

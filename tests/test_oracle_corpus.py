"""Full-episode same-action parity corpus tests against the pinned official
1.32.7 engine (fixed seeds, deterministic legal-ish generator).

Primitive-turn accounting contract locked here: a default 720-step episode is
ONE reset observation plus exactly 719 accepted primitive ``step`` calls; the
terminal DONE transition lands at canonical step 719 = day 29 hour 23. The
issue's "720-turn episode" terminology therefore means 720 canonical steps
including the reset state, NOT 720 post-reset step calls.

Skipped unless ``kaggle_environments`` passes the provenance guard.
"""

from __future__ import annotations

import pytest

from oracle import DivergenceError, run_same_action_replay
from oracle.action_generator import LegalishActionGenerator
from oracle.provenance import ProvenanceError, verify_official_provenance
from scripts.run_parity_corpus import (
    EXPECTED_DAY_TRANSITIONS,
    EXPECTED_PRIMITIVE_STEPS,
    DayCountingFastBackend,
)

try:
    verify_official_provenance()
    OFFICIAL_AVAILABLE = True
    _SKIP_REASON = ""
except ProvenanceError as error:
    OFFICIAL_AVAILABLE = False
    _SKIP_REASON = str(error)

pytestmark = pytest.mark.skipif(not OFFICIAL_AVAILABLE, reason=_SKIP_REASON)


def _replay(seed: int, max_turns: int):
    generator = LegalishActionGenerator(seed)
    fast_backend = DayCountingFastBackend({"seed": seed})
    result = run_same_action_replay(
        {"seed": seed},
        generator.next_pair,
        max_turns=max_turns,
        fast_backend=fast_backend,
    )
    return generator, fast_backend, result


def test_short_legalish_episode_no_divergence() -> None:
    # Four full days of generated legal-ish play for both seats.
    _, _, result = _replay(42, 96)
    assert result.turns_executed == 96
    assert result.final_step == 96
    assert result.official_statuses == result.fast_statuses == ["ACTIVE", "ACTIVE"]


def test_full_default_episode_terminal_accounting_and_parity() -> None:
    generator, fast_backend, result = _replay(0, EXPECTED_PRIMITIVE_STEPS + 1)
    # Primitive-turn accounting: reset observation + exactly 719 accepted
    # step calls; terminal DONE at canonical step 719 (day 29 hour 23).
    assert result.turns_executed == EXPECTED_PRIMITIVE_STEPS == 719
    assert result.final_step == 719
    assert fast_backend.day_transitions == EXPECTED_DAY_TRANSITIONS == 29
    assert result.official_statuses == result.fast_statuses == ["DONE", "DONE"]
    assert result.official_rewards == result.fast_rewards
    # Broad family coverage must actually be attempted in the episode.
    families = set(generator.coverage)
    assert "market.BUY_SEED" in families and "market.SELL" in families
    assert "unit.PLANT" in families and "unit.WATER" in families


def test_second_seed_full_episode_terminal_parity() -> None:
    _, _, result = _replay(7, EXPECTED_PRIMITIVE_STEPS + 1)
    assert result.turns_executed == 719
    assert result.final_step == 719
    assert result.official_statuses == result.fast_statuses == ["DONE", "DONE"]
    assert result.official_rewards == result.fast_rewards


def test_divergence_still_attributable_with_generator_source() -> None:
    # A corrupted fast canonical state must still stop at the exact first
    # divergent turn with the generator-chosen pair in the report.
    def corrupt(canonical: dict, turn: int) -> None:
        if turn == 30:
            canonical["farms"][0]["hires_today"] = 99

    generator = LegalishActionGenerator(2)
    with pytest.raises(DivergenceError) as excinfo:
        run_same_action_replay(
            {"seed": 2},
            generator.next_pair,
            max_turns=60,
            fast_mutator=corrupt,
        )
    report = excinfo.value.report
    assert report.turn_index == 30
    assert report.seed == 2
    assert report.p0_action is not None and report.p1_action is not None

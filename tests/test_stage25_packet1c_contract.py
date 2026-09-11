"""Cross-packet checks for the authoritative Stage 2.5 Packet 1 contract."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from rl_manager.stage25_config import (
    CURRICULUM_CONFIG_VERSION,
    SHORTFALL_CONFIG_VERSION,
    Stage25CurriculumConfig,
    apply_animal_curriculum,
    apply_crop_curriculum,
    apply_land_curriculum,
)
from rl_manager.stage25_mechanics import (
    ACTION_CLASS_COUNTS,
    ACTION_ORDER,
    ACTION_SCHEMA_VERSION,
    CROP_HOLD_CLASS,
    crop_class_to_delta,
)


ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = ROOT / "docs" / "STAGE25_PACKET1_CONTRACT.md"


def test_authoritative_packet1_document_pins_cross_packet_contract():
    text = AUTHORITY.read_text(encoding="utf-8")
    lowered = text.lower()

    assert "authoritative for stage 2.5 packet 1" in lowered
    for version in ("stage25_physical_v1", "stage25_curriculum_v1",
                    "stage25_crop_shortfall_v1"):
        assert version in lowered
    assert "physical support ∩ curriculum support" in text
    assert "prefix_entropy_surrogate" in text
    assert "physical_batch_size=None" in text
    assert "no permanent `+25` cap" in lowered
    assert "unmasked physically impossible" in lowered

    # Legacy proposal fields and absolute crop-output semantics are not part
    # of the current Stage 2.5 authority, even though legacy modules remain.
    assert "absolute crop" not in lowered
    for legacy_field in ("care_by_animal", "fertilizer_by_crop",
                         "sell_presence", "sell_quantity"):
        assert legacy_field not in lowered


def test_packet1a_and_1b_versions_and_action_schema_agree():
    assert ACTION_SCHEMA_VERSION == "stage25_physical_v1"
    assert CURRICULUM_CONFIG_VERSION == "stage25_curriculum_v1"
    assert SHORTFALL_CONFIG_VERSION == "stage25_crop_shortfall_v1"
    assert ACTION_ORDER == (
        "land", "goose", "cow", "sheep", "wheat", "carrot", "tomato",
        "strawberry", "melon",
    )
    assert ACTION_CLASS_COUNTS == (4, 101, 101, 101, 201, 201, 201, 201, 201)
    # The decoder output/embedding accounting is derived from the actual class
    # tuple, not a separately pinned constant.  Each of the nine step-specific
    # heads contributes one output projection, one bias, and one action
    # embedding per class, so the aggregate is sum(counts) * (2D + 1).
    action_class_total = sum(ACTION_CLASS_COUNTS)
    assert action_class_total == 4 + 3 * 101 + 5 * 201 == 1312
    text = AUTHORITY.read_text(encoding="utf-8")
    assert "4 + 3*101 + 5*201 = 1312" in text
    assert "812" not in text
    assert action_class_total * (2 * 128 + 1) == 337_184
    assert action_class_total * (2 * 256 + 1) == 673_056
    assert CROP_HOLD_CLASS == 100
    assert crop_class_to_delta(CROP_HOLD_CLASS) == 0


def test_disabled_curriculum_is_the_physical_support_identity():
    config = Stage25CurriculumConfig(enabled=False)
    land = (False, True, True, False)
    animal = tuple(index % 3 != 0 for index in range(101))
    physical = tuple(index % 3 != 0 for index in range(201))
    assert apply_land_curriculum(land, 2, config) == land
    assert apply_animal_curriculum(animal, 4, config) == animal
    assert apply_crop_curriculum(physical, config) == physical


def test_stage25_lightweight_modules_import_without_heavy_frameworks():
    result = subprocess.run(
        [sys.executable, "-c", (
            "import sys; import rl_manager.stage25_mechanics, "
            "rl_manager.stage25_config, rl_manager.stage25_data; "
            "assert not ({name.split('.')[0] for name in sys.modules} "
            "& {'torch', 'jax', 'pyarrow'})"
        )],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

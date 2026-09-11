"""Focused Packet 1B tests for curriculum and outcome-proxy contracts.

The data tests intentionally use small logical rows backed by the Packet 1A
10x10 physical representation.  They are skipped until the planned
``rl_manager.stage25_data`` module is present; the curriculum tests remain
independently runnable while Packet 1B is being assembled.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import importlib
import inspect
import math
import subprocess
import sys
from collections.abc import Mapping

import pytest

from rl_manager.stage25_config import (
    CropShortfallConfig,
    Stage25CurriculumConfig,
    apply_animal_curriculum,
    apply_crop_curriculum,
    apply_land_curriculum,
    crop_shortfall_penalty,
)
from rl_manager.stage25_mechanics import physical_context_from_board


def _all_true(size: int) -> tuple[bool, ...]:
    return (True,) * size


def test_disabled_curriculum_preserves_physical_masks_exactly():
    config = Stage25CurriculumConfig(enabled=False)
    land = (False, True, True, False)
    animal = tuple(index % 3 == 0 for index in range(101))
    crop = tuple(index % 5 != 0 for index in range(201))

    assert apply_land_curriculum(land, 2, config) == land
    assert apply_animal_curriculum(animal, 4, config) == animal
    assert apply_crop_curriculum(crop, config) == crop


def test_enabled_curriculum_caps_land_and_animals_relative_to_observed():
    config = Stage25CurriculumConfig(
        enabled=True,
        max_land_expansion_per_decision=1,
        max_animal_additions_per_species_per_decision=2,
    )
    assert apply_land_curriculum(_all_true(4), 2, config) == (
        True, True, True, False)

    animal = apply_animal_curriculum(_all_true(101), 4, config)
    assert animal[:7] == (True,) * 7
    assert not animal[7]
    assert animal[-1] is False


def test_crop_curriculum_caps_positive_delta_but_keeps_contraction():
    config = Stage25CurriculumConfig(enabled=True, max_positive_crop_delta=3)
    mask = apply_crop_curriculum(_all_true(201), config)
    # Class 0 is -100, class 100 is HOLD, and class 200 is +100.
    assert all(mask[index] for index in range(0, 104))
    assert not mask[104]
    assert not mask[200]

    # A curriculum cap may not remove a physically supported full
    # contraction, even when it is the only surviving physical action.
    forced_contraction = tuple(index == 20 for index in range(201))  # delta -80
    assert apply_crop_curriculum(forced_contraction, config) == forced_contraction


def test_curriculum_intersection_is_nonempty_and_never_repairs_physical_support():
    config = Stage25CurriculumConfig(
        enabled=True,
        max_land_expansion_per_decision=0,
        max_animal_additions_per_species_per_decision=0,
        max_positive_crop_delta=0,
    )
    assert apply_land_curriculum((False, True, False, False), 2, config) == (
        False, True, False, False)
    assert apply_animal_curriculum(
        tuple(index == 4 for index in range(101)), 4, config
    ) == tuple(index == 4 for index in range(101))
    assert apply_crop_curriculum(
        tuple(index == 100 for index in range(201)), config
    ) == tuple(index == 100 for index in range(201))

    with pytest.raises(ValueError, match="removed all actions"):
        apply_land_curriculum((False, False, True, False), 1, config)


def test_curriculum_masks_have_no_economic_inputs_or_effects():
    config = Stage25CurriculumConfig(enabled=True, max_land_expansion_per_decision=1)
    physical = (False, True, True, True)
    # The public helper takes only physical support, observed physical state,
    # and config; money/prices/affordability cannot alter its result.
    assert tuple(inspect.signature(apply_land_curriculum).parameters) == (
        "physical_support", "observed_land", "config")
    assert apply_land_curriculum(physical, 1, config) == (
        False, True, False, False)


def test_shortfall_uses_one_shared_tolerance_and_coef_zero_is_noop():
    config = CropShortfallConfig(crop_shortfall_tolerance=5,
                                 crop_shortfall_coef=2.5)
    assert crop_shortfall_penalty(0, config) == 0.0
    assert crop_shortfall_penalty(5, config) == 0.0
    assert crop_shortfall_penalty(6, config) == -2.5
    assert crop_shortfall_penalty(8, config) == -7.5
    assert crop_shortfall_penalty(
        100, CropShortfallConfig(crop_shortfall_tolerance=99,
                                 crop_shortfall_coef=0.0)
    ) == 0.0


@pytest.mark.parametrize(
    "factory",
    [
        lambda: Stage25CurriculumConfig(max_positive_crop_delta=-1),
        lambda: Stage25CurriculumConfig(max_land_expansion_per_decision=True),
        lambda: Stage25CurriculumConfig(version="wrong"),
        lambda: CropShortfallConfig(crop_shortfall_tolerance=-1),
        lambda: CropShortfallConfig(crop_shortfall_coef=-1),
        lambda: CropShortfallConfig(crop_shortfall_coef=math.inf),
        lambda: CropShortfallConfig(crop_shortfall_coef=True),
    ],
)
def test_invalid_curriculum_and_shortfall_configs_fail_loudly(factory):
    with pytest.raises(ValueError):
        factory()


def test_invalid_curriculum_inputs_and_empty_intersections_fail_loudly():
    config = Stage25CurriculumConfig(enabled=True, max_land_expansion_per_decision=0)
    with pytest.raises(ValueError):
        apply_land_curriculum((True,) * 3, 1, config)
    with pytest.raises(ValueError):
        apply_land_curriculum((True,) * 4, 0, config)
    with pytest.raises(ValueError):
        apply_animal_curriculum((True,) * 101, 101, config)
    with pytest.raises(ValueError):
        apply_crop_curriculum((True,) * 200, config)
    with pytest.raises(ValueError):
        crop_shortfall_penalty(-1)


def _board10(*, wheat: int = 1, carrot: int = 0,
             unlocked=("NW",), structures: bool = True,
             sheep: bool = True) -> list[list[object]]:
    board: list[list[object]] = [["LOCKED"] * 10 for _ in range(10)]
    # NW is initially unlocked.  The optional structures and plants use only
    # canonical Packet 1A tile roles; later land is represented by None cells.
    if structures:
        board[0][0] = {"kind": "COOP"}
        board[0][1] = {"kind": "PASTURE"}
        board[0][2] = ({"kind": "PASTURE", "animal": "SHEEP"}
                       if sheep else {"kind": "PASTURE"})

    quadrants = set(unlocked)
    coordinates = [
        (y, x) for y in range(10) for x in range(10)
        if (("NW" if y < 5 and x < 5 else
             "NE" if y < 5 else "SW" if x < 5 else "SE") in quadrants)
    ]
    occupied = {(0, 0), (0, 1), (0, 2)} if structures else set()
    plants = [("WHEAT", wheat), ("CARROT", carrot)]
    for crop, count in plants:
        for y, x in coordinates:
            if count <= 0:
                break
            if (y, x) in occupied:
                continue
            board[y][x] = {"kind": "PLANT", "crop": crop}
            occupied.add((y, x))
            count -= 1
    for y, x in coordinates:
        if (y, x) not in occupied:
            board[y][x] = None
    return board


def _logical_row(day: int, *, crop_counts=(1, 0, 0, 0, 0),
                 unlocked=("NW",),
                 start_wheat: int | None = None,
                 start_carrot: int = 0,
                 end_unlocked=None,
                 start_structures: bool = True,
                 end_structures: bool = True,
                 sheep: bool = True,
                 start_sheep: bool | None = None,
                 end_sheep: bool | None = None) -> dict:
    """Minimal row with both logical labels and canonical physical context."""
    if start_wheat is None:
        start_wheat = crop_counts[0]
    if end_unlocked is None:
        end_unlocked = unlocked
    if start_sheep is None:
        start_sheep = sheep
    if end_sheep is None:
        end_sheep = sheep
    start_board = _board10(
        wheat=start_wheat, carrot=start_carrot, unlocked=unlocked,
        structures=start_structures, sheep=start_sheep)
    end_board = _board10(
        wheat=crop_counts[0], carrot=crop_counts[1],
        unlocked=end_unlocked, structures=end_structures, sheep=end_sheep)
    context = physical_context_from_board(start_board, unlocked)
    assert context.observed_land == len(unlocked)
    return {
        "episode_id": 25,
        "seat": 0,
        "day": day,
        "date": day,
        "step": day * 24,
        "boundary_id": day,
        "score": 1.0,
        "metadata": {
            "episode_id": 25,
            "seat": 0,
            "partition_date": "2026-09-11",
            "min_score": 1.0,
        },
        "start": {
            "self": {
                "board": start_board,
                "unlocked_quadrants": list(unlocked),
                "shed": {},
                "inventories": [],
            }
        },
        "end": {
            "self": {
                "board": end_board,
                "unlocked_quadrants": list(end_unlocked),
            }
        },
    }


def _data_module():
    return pytest.importorskip(
        "rl_manager.stage25_data",
        reason="Packet 1B data module is not present in this checkout",
    )


def _build_labels(module, rows, first_k=2):
    del first_k  # The current API initializes at each contiguous history start.
    return module.build_outcome_proxy_labels(rows)


def _plain(value):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Mapping):
        return dict(value)
    return value


def _result_field(result, names):
    if isinstance(result, Mapping):
        for name in names:
            if name in result:
                return result[name]
    for name in names:
        if hasattr(result, name):
            return getattr(result, name)
    result = _plain(result)
    if isinstance(result, Mapping):
        for name in names:
            if name in result:
                return result[name]
    raise AssertionError(f"result has none of the expected fields: {names}")


def _invalid_mapping(value) -> dict:
    value = _plain(value)
    if isinstance(value, Mapping):
        return dict(value)
    fields = getattr(value, "__dataclass_fields__", {})
    if fields:
        return {name: getattr(value, name) for name in fields}
    raise AssertionError("invalid-count result is not a mapping/dataclass")


def test_outcome_proxy_labels_initialize_first_k_and_propagate_adjacent_rows():
    module = _data_module()
    rows = [
        _logical_row(0, crop_counts=(3, 0, 0, 0, 0), start_wheat=2,
                     end_unlocked=("NW", "NE")),
        _logical_row(1, crop_counts=(4, 1, 0, 0, 0), start_wheat=3,
                     unlocked=("NW", "NE"), end_unlocked=("NW", "NE")),
        _logical_row(2, crop_counts=(5, 1, 0, 0, 0), start_wheat=4,
                     start_carrot=1, unlocked=("NW", "NE"),
                     end_unlocked=("NW", "NE")),
    ]
    result = _build_labels(module, rows, first_k=2)
    labels = _result_field(result, ("labels", "proxy_labels", "outcomes", "rows"))
    invalid = _result_field(
        result, ("invalid_counts", "invalid_count", "counts", "counters")
    )
    assert len(labels) == 3
    invalid = _invalid_mapping(invalid)
    assert all(isinstance(value, int) and value == 0
               for value in invalid.values())

    # The first K boundary is initialized from observed state, while exact
    # adjacency propagates the prior K plus the sampled signed delta.
    assert labels[0].provenance.prior_source == "first_boundary_start_occupancy"
    assert labels[0].provenance.prior_crop_goals == (2, 0, 0, 0, 0)
    assert labels[1].provenance.prior_source == "previous_synthetic_desired_end_goal"
    assert labels[1].provenance.prior_crop_goals == (3, 0, 0, 0, 0)
    assert labels[1].crop_goals == (4, 1, 0, 0, 0)
    assert labels[1].crop_deltas == (1, 1, 0, 0, 0)
    assert labels[1].crop_classes == (101, 101, 100, 100, 100)


def test_outcome_proxy_labels_reject_gaps_with_invalid_counters():
    module = _data_module()
    rows = [
        _logical_row(0),
        _logical_row(2),  # non-adjacent gap
    ]
    result = _build_labels(module, rows, first_k=1)
    invalid = _result_field(
        result, ("invalid_counts", "invalid_count", "counts", "counters")
    )
    assert _invalid_mapping(invalid)["history_reset_gap_rows"] == 1
    assert all(label.provenance.history_reset is (label.row_index == 1)
               for label in result.labels)


def test_outcome_proxy_uses_observed_animal_loss_without_fabricating_transactions():
    module = _data_module()
    rows = [
        _logical_row(0),
        _logical_row(1, start_sheep=True, end_sheep=False),
    ]
    result = _build_labels(module, rows, first_k=1)
    labels = _result_field(result, ("labels", "proxy_labels", "outcomes", "rows"))
    assert labels[-1].animal_labels == (0, 0, None)
    assert "purchase" not in repr(labels[-1]).lower()
    invalid = _invalid_mapping(
        _result_field(result, ("invalid_counts", "invalid_count", "counts", "counters"))
    )
    assert invalid["animal_loss_ambiguity_components"] == 1


def test_outcome_proxy_preserves_class_delta_and_does_not_clip_raw_change():
    module = _data_module()
    constrained = _logical_row(
        1,
        crop_counts=(50, 0, 0, 0, 0),
        start_wheat=20,
        start_structures=False,
        end_unlocked=("NW", "NE"),
        end_structures=False,
        sheep=False,
    )
    # Make most of the current NW footprint sticky housing.  The hypothetical
    # two-quadrant crop capacity is then 27, while the observed end board asks
    # for 50 crops.
    start_board = constrained["start"]["self"]["board"]
    blocked = 0
    for y in range(5):
        for x in range(5):
            if blocked >= 23:
                break
            if start_board[y][x] != "LOCKED":
                start_board[y][x] = {"kind": "COOP"}
                blocked += 1
        if blocked >= 23:
            break
    rows = [
        _logical_row(0, crop_counts=(0, 0, 0, 0, 0)),
        constrained,
    ]
    result = _build_labels(module, rows, first_k=1)
    labels = _result_field(result, ("labels", "proxy_labels", "outcomes", "rows"))
    # The end board asks for 50 crops, but the start physical context has
    # only 27 recoverable cells at the requested land target.  The component
    # is invalid; it must not be repaired or clipped to the capacity.
    invalid = _invalid_mapping(
        _result_field(result, ("invalid_counts", "invalid_count", "counts", "counters"))
    )
    assert (invalid["crop_physical_incompatibility_components"] >= 1 or
            invalid["crop_delta_outside_vocabulary_components"] >= 1)
    assert labels[-1].crop_deltas[0] is None
    assert labels[-1].crop_classes[0] is None


def test_data_module_exports_invalid_count_surface_and_is_framework_free():
    module = _data_module()
    public_names = set(getattr(module, "__all__", dir(module)))
    assert callable(module.build_outcome_proxy_labels)
    assert "InvalidCounters" in public_names
    assert is_dataclass(module.InvalidCounters)
    fields = set(module.InvalidCounters.__dataclass_fields__)
    assert {
        "crop_delta_outside_vocabulary_components",
        "crop_physical_incompatibility_components",
        "history_reset_gap_rows",
        "animal_loss_ambiguity_components",
        "invalid_rows",
        "component_excluded_rows",
    } <= fields

    result = subprocess.run(
        [sys.executable, "-c", (
            "import sys; import rl_manager.stage25_data; "
            "assert not ({name.split('.')[0] for name in sys.modules} "
            "& {'torch', 'jax', 'pyarrow'})"
        )],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_stage25_framework_free_imports_do_not_load_heavy_frameworks():
    if importlib.util.find_spec("rl_manager.stage25_data") is None:
        pytest.skip("Packet 1B data module is not present in this checkout")
    result = subprocess.run(
        [sys.executable, "-c", (
            "import sys; import rl_manager.stage25_mechanics, "
            "rl_manager.stage25_config, rl_manager.stage25_data; "
            "assert not ({name.split('.')[0] for name in sys.modules} "
            "& {'torch', 'jax', 'pyarrow'})"
        )],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

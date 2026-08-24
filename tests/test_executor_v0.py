"""Focused tests for executor_v0 stage 2: plan, manager wrapper, projection.

Covers issue #1 sections 2-3 only. The real-model path is exercised against
the ignored local tiny checkpoint when present; everything else uses the
fake/fixed injection seam and hand-built logits, so no checkpoint dependency.
"""

import copy
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from bc_manager.constants import ANIMAL_ORDER, CROP_ORDER, PRODUCT_ORDER
from executor_v0 import (
    CachingPlanProvider,
    CheckpointPlanProvider,
    DailyPlan,
    FixedPlanProvider,
    clip_sell,
    decode_daily_plan,
    project_plan,
)
from replay_daily.constants import SELL_BIN_ANCHORS

REPO_ROOT = Path(__file__).resolve().parents[1]
TINY_CKPT = REPO_ROOT / "data" / "temp" / "bc-train-smoke" / "ckpt" / "best.pt"


# ------------------------------------------------------------------ helpers


def plan_kwargs(**overrides: Any) -> dict:
    kwargs: dict[str, Any] = {
        "crop_targets": {"WHEAT": 3, "CARROT": 0, "TOMATO": 2,
                         "STRAWBERRY": 0, "MELON": 1},
        "animal_targets": {"GOOSE": 2, "COW": 1, "SHEEP": 0},
        "land_count": 2,
        "fertilizer_by_crop": {"WHEAT": 2, "CARROT": 0, "TOMATO": 1,
                               "STRAWBERRY": 0, "MELON": 0},
        "care_by_animal": {"GOOSE": 1, "COW": 1, "SHEEP": 0},
        "sell_quantities": {
            product: {anchor: 0 for anchor in SELL_BIN_ANCHORS}
            for product in PRODUCT_ORDER
        },
    }
    kwargs["sell_quantities"]["WHEAT"] = {0: 5, 4: 3, 8: 0, 12: 0, 16: 0, 20: 0}
    kwargs["sell_quantities"]["EGG"] = {12: 2, 16: 4, 0: 0, 4: 0, 8: 0, 20: 0}
    kwargs.update(overrides)
    return kwargs


def make_plan(**overrides: Any) -> DailyPlan:
    return DailyPlan.create(**plan_kwargs(**overrides))


def empty_tiles10() -> list[list[None]]:
    return [[None] * 10 for _ in range(10)]


def make_live_obs(day: int, hour: int, step: int) -> dict:
    farm = {
        "farmer": [0, 0],
        "hands": [[1, 1]],
        "hires_today": 1,
        "money": 2500.0,
        "tiles": empty_tiles10(),
        "unlocked_quadrants": ["NW"],
    }
    return {
        "day": day, "hour": hour, "step": step, "player": 0,
        "farms": [farm, copy.deepcopy(farm)],
        "market": {"inventory": {"WHEAT": 7}, "prices": {"WHEAT": 25}},
        "town": {"unlocked_shops": ["BAKERY"]},
        "private": {"shed": {}, "seeds": {}, "inventories": [{}]},
    }


class CountingProvider:
    """Fake provider recording every call (provider-invocation counter)."""

    def __init__(self, plan: DailyPlan) -> None:
        self.plan = plan
        self.calls: list[tuple[int, int, dict[str, int] | None]] = []

    def daily_plan(self, obs, seat, previous_execution=None) -> DailyPlan:
        self.calls.append((int(obs["day"]), int(obs["hour"]),
                           dict(previous_execution)
                           if previous_execution is not None else None))
        return self.plan


# ------------------------------------------------------------------ plan


def test_daily_plan_normalizes_to_canonical_order():
    scrambled = {
        "MELON": 1, "WHEAT": 3, "CARROT": 0, "STRAWBERRY": 0, "TOMATO": 2,
    }
    plan = make_plan(crop_targets=scrambled)
    assert plan.crop_targets == (3, 0, 2, 0, 1)
    assert list(plan.crop_targets_dict) == list(CROP_ORDER)
    assert list(plan.animal_targets_dict) == list(ANIMAL_ORDER)
    assert list(plan.fertilizer_by_crop_dict) == list(CROP_ORDER)
    assert list(plan.care_by_animal_dict) == list(ANIMAL_ORDER)
    # Sell rows are indexed [product][bin]; anchors normalize to bin order.
    wheat_index = PRODUCT_ORDER.index("WHEAT")
    egg_index = PRODUCT_ORDER.index("EGG")
    assert plan.sell_quantities[wheat_index] == (5, 3, 0, 0, 0, 0)
    assert plan.sell_quantities[egg_index][3] == 2  # anchor 12 -> bin 3
    assert plan.sell_quantities[egg_index][4] == 4  # anchor 16 -> bin 4


def test_daily_plan_is_frozen_and_copy_safe():
    plan = make_plan()
    with pytest.raises(Exception):
        plan.land_count = 3  # type: ignore[misc]
    source = {crop: 0 for crop in CROP_ORDER}
    source["WHEAT"] = 1
    plan = make_plan(crop_targets=source)
    source["WHEAT"] = 99  # mutating the input mapping must not affect plan
    assert plan.crop_targets_dict["WHEAT"] == 1


def test_daily_plan_json_round_trip_shape_and_determinism():
    a = make_plan()
    b = make_plan()
    text_a = json.dumps(a.to_json_dict())
    text_b = json.dumps(b.to_json_dict())
    assert text_a == text_b  # deterministic canonical representation
    parsed = json.loads(text_a)
    assert parsed["crop_targets"]["WHEAT"] == 3
    assert parsed["land_count"] == 2
    assert parsed["sell_quantities"]["0"]["WHEAT"] == 5
    assert parsed["sell_quantities"]["12"]["EGG"] == 2
    # Rebuilding from the JSON view reproduces an equal plan.
    parsed_sells = parsed["sell_quantities"]
    rebuilt = DailyPlan.create(
        crop_targets=parsed["crop_targets"],
        animal_targets=parsed["animal_targets"],
        land_count=parsed["land_count"],
        fertilizer_by_crop=parsed["fertilizer_by_crop"],
        care_by_animal=parsed["care_by_animal"],
        sell_quantities={
            product: {int(anchor): parsed_sells[str(anchor)][product]
                      for anchor in SELL_BIN_ANCHORS}
            for product in PRODUCT_ORDER
        },
    )
    assert rebuilt == a


@pytest.mark.parametrize("bad", [
    {"crop_targets": {"WHEAT": -1}},
    {"crop_targets": {"WHEAT": 1.5}},
    {"crop_targets": {"WHEAT": True}},
    {"crop_targets": {"WHEAT": 1, "OATS": 2}},
    {"crop_targets": {}},
    {"animal_targets": {"GOOSE": 1}},
    {"fertilizer_by_crop": {"WHEAT": 1}},
    {"care_by_animal": {}},
    {"land_count": 0},
    {"land_count": 5},
    {"land_count": 2.0},
    {"sell_quantities": {"WHEAT": {0: 1}}},
    {"sell_quantities": {"WHEAT": {0: 1, 4: 0, 8: 0, 12: 0, 16: 0, 20: 0,
                                   24: 9}}},
    {"sell_quantities": {"NO_SUCH_PRODUCT": {a: 0 for a in
                                             SELL_BIN_ANCHORS}}},
])
def test_daily_plan_validation_rejects_bad_fields(bad):
    with pytest.raises(ValueError):
        make_plan(**bad)


# ------------------------------------------------------------- manager seam


def test_fixed_plan_provider_injection_needs_no_checkpoint():
    plan = make_plan()
    provider = FixedPlanProvider(plan)
    obs = make_live_obs(0, 0, 0)
    assert provider.daily_plan(obs, 0) is plan
    assert provider.daily_plan(obs, 1, {"workers_hired": 2,
                                        "hire_cost": 3}) is plan
    with pytest.raises(ValueError):
        FixedPlanProvider({"not": "a plan"})


def test_checkpoint_path_missing_is_clear_error_not_fabrication():
    with pytest.raises(FileNotFoundError, match="checkpoint"):
        CheckpointPlanProvider(REPO_ROOT / "data" / "temp" / "nope.pt")


def test_decode_threshold_shapes_and_quantity_conversion():
    count_max = 100
    b, crops, animals, products, bins = 1, 5, 3, 9, 6
    outputs = {
        "crop_logits": torch.zeros(b, crops, count_max + 1),
        "animal_logits": torch.zeros(b, animals, count_max + 1),
        "land_logits": torch.tensor([[0.0, 5.0, 0.0, 0.0]]),
        "fertilizer_logits": torch.zeros(b, crops, count_max + 1),
        "care_logits": torch.zeros(b, animals, count_max + 1),
        "sell_presence_logits": torch.zeros(b, products, bins),
        "sell_quantity_log1p": torch.zeros(b, products, bins),
    }
    # Count argmax picks class index; land argmax+1 -> 2.
    outputs["crop_logits"][0, 0, 7] = 9.0      # WHEAT -> 7
    outputs["animal_logits"][0, 2, 1] = 4.0    # SHEEP -> 1
    outputs["fertilizer_logits"][0, 1, 3] = 2.0  # CARROT -> 3
    outputs["care_logits"][0, 0, 2] = 1.0      # GOOSE -> 2
    # Presence threshold is strict > 0.5: logit 0.0 (sigmoid exactly 0.5)
    # stays absent; any positive logit is present.
    outputs["sell_presence_logits"][0, 0, 0] = 0.0     # absent at exactly 0.5
    outputs["sell_presence_logits"][0, 0, 1] = 1e-6    # present
    outputs["sell_presence_logits"][0, 1, 3] = 5.0     # present (CARROT)
    # quantity expm1 values: 2.49 -> floor(2.99)=2 ; 2.75 -> floor(3.25)=3
    # (round-half-up on the decoded float); negative raw clamps to 0.
    outputs["sell_quantity_log1p"][0, 0, 1] = math.log1p(2.49)
    outputs["sell_quantity_log1p"][0, 1, 3] = math.log1p(2.75)
    outputs["sell_quantity_log1p"][0, 2, 0] = math.log1p(50.0)  # presence false

    plan = decode_daily_plan(outputs, count_max=count_max)
    assert plan.crop_targets[CROP_ORDER.index("WHEAT")] == 7
    assert plan.animal_targets[ANIMAL_ORDER.index("SHEEP")] == 1
    assert plan.fertilizer_by_crop[CROP_ORDER.index("CARROT")] == 3
    assert plan.care_by_animal[ANIMAL_ORDER.index("GOOSE")] == 2
    assert plan.land_count == 2
    wheat = plan.sell_quantities[PRODUCT_ORDER.index("WHEAT")]
    carrot = plan.sell_quantities[PRODUCT_ORDER.index("CARROT")]
    tomato = plan.sell_quantities[PRODUCT_ORDER.index("TOMATO")]
    assert wheat == (0, 2, 0, 0, 0, 0)   # exact 0.5 absent; 2.49 rounds to 2
    assert carrot == (0, 0, 0, 3, 0, 0)  # 2.75 rounds up to 3
    assert tomato == (0, 0, 0, 0, 0, 0)  # presence false forces zero
    assert all(v >= 0 for row in plan.sell_quantities for v in row)


@pytest.mark.skipif(not TINY_CKPT.exists(),
                    reason="local tiny checkpoint not present")
def test_tiny_checkpoint_load_decode_smoke_and_opponent_config():
    provider = CheckpointPlanProvider(TINY_CKPT, device="cpu")
    assert provider.include_opponent_board == \
        provider.model_config.include_opponent_board
    plan = provider.daily_plan(make_live_obs(0, 0, 0), 0,
                               {"workers_hired": 1, "hire_cost": 1})
    assert isinstance(plan, DailyPlan)
    assert 1 <= plan.land_count <= 4
    assert all(v >= 0 for v in plan.crop_targets)
    assert all(v <= provider.model_config.count_max
               for v in plan.crop_targets)
    json.dumps(plan.to_json_dict())  # serializable end to end
    if not provider.model_config.include_opponent_board:
        with pytest.raises(ValueError, match="opponent"):
            CheckpointPlanProvider(TINY_CKPT, device="cpu",
                                   include_opponent_board=True)


# ------------------------------------------------------------------ caching


def test_caching_provider_computes_once_per_day_and_propagates_labor():
    plan = make_plan()
    inner = CountingProvider(plan)
    cache = CachingPlanProvider(inner)

    first = cache.daily_plan(make_live_obs(0, 0, 10), 0,
                             {"workers_hired": 0, "hire_cost": 0})
    assert first is plan
    assert len(inner.calls) == 1

    # Same day, later hours incl. non-hour0: cached, no new invocation.
    for hour in (1, 7, 23):
        again = cache.daily_plan(make_live_obs(0, hour, 100 + hour), 0,
                                 {"workers_hired": 9, "hire_cost": 9})
        assert again is first
    assert len(inner.calls) == 1

    # New day first seen after hour 0 computes once (deterministic resilience).
    second = cache.daily_plan(make_live_obs(1, 5, 200), 0,
                              {"workers_hired": 2, "hire_cost": 3})
    assert second is plan
    assert len(inner.calls) == 2
    assert inner.calls[-1] == (1, 5, {"workers_hired": 2, "hire_cost": 3})

    # Same new day now caches; never recomputes within a primitive turn loop.
    cache.daily_plan(make_live_obs(1, 6, 201), 0)
    assert len(inner.calls) == 2


# --------------------------------------------------------------- projection


def test_projection_land_never_decreases_and_stays_bounded():
    requested = make_plan(land_count=2)
    result = project_plan(requested, current_land_count=3,
                          current_animals={}, current_crops={})
    assert result.feasible_plan.land_count == 3
    assert result.diagnostics["land"] == {
        "current": 3, "requested": 2, "feasible": 3}
    result = project_plan(requested, current_land_count=1,
                          current_animals={}, current_crops={})
    assert result.feasible_plan.land_count == 2


def test_projection_animals_never_removed_and_deficits_exposed():
    requested = make_plan(animal_targets={"GOOSE": 4, "COW": 0, "SHEEP": 2})
    result = project_plan(requested, current_land_count=1,
                          current_animals={"GOOSE": 2, "SHEEP": 5},
                          current_crops={})
    feasible = result.feasible_plan.animal_targets_dict
    assert feasible["GOOSE"] == 4          # grows toward target
    assert feasible["COW"] == 0            # zero request stays zero
    assert feasible["SHEEP"] == 5          # existing never removed
    diag = result.diagnostics["animals"]
    assert diag["GOOSE"] == {"current": 2, "requested": 4, "feasible": 4,
                             "buy_build_deficit": 2}
    assert diag["SHEEP"] == {"current": 5, "requested": 2, "feasible": 5,
                             "buy_build_deficit": 0}


def test_projection_fertilizer_care_clip_to_eligible_with_shortfall():
    requested = make_plan(
        fertilizer_by_crop={"WHEAT": 5, "CARROT": 1, "TOMATO": 2,
                            "STRAWBERRY": 0, "MELON": 0},
        care_by_animal={"GOOSE": 3, "COW": 1, "SHEEP": 2},
    )
    result = project_plan(requested, current_land_count=1,
                          current_animals={"GOOSE": 1},
                          current_crops={"WHEAT": 4})
    fert = result.feasible_plan.fertilizer_by_crop_dict
    care = result.feasible_plan.care_by_animal_dict
    # Eligibility counts assets the plan itself establishes today
    # (max(current, requested target)), not just the start-of-day snapshot:
    # clipping against hour-0 animals/crops made requests for same-day
    # purchases/plantings permanently infeasible (issue #7).
    assert fert["WHEAT"] == 4      # clipped to planned total max(4, 3) = 4
    assert fert["CARROT"] == 0     # no carrot current or planned: honest zero
    assert fert["TOMATO"] == 2     # plan plants 2 tomatoes today -> eligible
    assert care["GOOSE"] == 2      # plan raises geese to 2 -> eligible 2
    assert care["COW"] == 1        # plan buys 1 cow today -> eligible 1
    assert care["SHEEP"] == 0      # no sheep current or planned
    assert result.diagnostics["fertilizer"]["WHEAT"] == {
        "requested": 5, "eligible": 4, "feasible": 4, "shortfall": 1}
    assert result.diagnostics["fertilizer"]["TOMATO"] == {
        "requested": 2, "eligible": 2, "feasible": 2, "shortfall": 0}
    assert result.diagnostics["care"]["GOOSE"] == {
        "requested": 3, "eligible": 2, "feasible": 2, "shortfall": 1}


def test_projection_keeps_sell_schedule_and_reports_requested_totals():
    requested = make_plan()
    result = project_plan(requested, current_land_count=1,
                          current_animals={}, current_crops={})
    assert result.feasible_plan.sell_quantities == requested.sell_quantities
    totals = result.diagnostics["sells"]["requested_total_by_product"]
    assert totals["WHEAT"] == 8
    assert totals["EGG"] == 6
    assert totals["MILK"] == 0


def test_projection_requested_object_unchanged_and_json_deterministic():
    requested = make_plan()
    before = json.dumps(requested.to_json_dict())
    result = project_plan(requested, current_land_count=2,
                          current_animals={"GOOSE": 1},
                          current_crops={"WHEAT": 1})
    assert json.dumps(result.requested_plan.to_json_dict()) == before
    assert result.requested_plan is requested
    text_one = json.dumps(result.diagnostics, sort_keys=True)
    text_two = json.dumps(
        project_plan(requested, current_land_count=2,
                     current_animals={"GOOSE": 1},
                     current_crops={"WHEAT": 1}).diagnostics,
        sort_keys=True)
    assert text_one == text_two


def test_projection_rejects_invalid_current_state():
    requested = make_plan()
    with pytest.raises(ValueError):
        project_plan(requested, current_land_count=0, current_animals={},
                     current_crops={})
    with pytest.raises(ValueError):
        project_plan(requested, current_land_count=1,
                     current_animals={"GOOSE": -1}, current_crops={})
    with pytest.raises(ValueError):
        project_plan(requested, current_land_count=1, current_animals=None,
                     current_crops={})


def test_clip_sell_semantics():
    assert clip_sell("WHEAT", 5, 3) == (3, 2)
    assert clip_sell("WHEAT", 5, 9) == (5, 0)
    assert clip_sell("WHEAT", 0, 9) == (0, 0)
    assert clip_sell("WHEAT", 5, 0) == (0, 5)
    with pytest.raises(ValueError):
        clip_sell("NOPE", 1, 1)
    with pytest.raises(ValueError):
        clip_sell("WHEAT", -1, 1)
    with pytest.raises(ValueError):
        clip_sell("WHEAT", 1, -1)


@pytest.mark.skipif(not TINY_CKPT.exists(),
                    reason="local tiny checkpoint not present")
def test_real_observation_fake_manager_wrapper_smoke():
    """Bounded live observation + fake manager through the caching wrapper."""
    plan = make_plan()
    wrapper = CachingPlanProvider(FixedPlanProvider(plan))
    obs_day0_hour0 = make_live_obs(0, 0, 0)
    obs_day0_hour1 = make_live_obs(0, 1, 1)
    assert wrapper.daily_plan(obs_day0_hour0, 0) is plan
    assert wrapper.daily_plan(obs_day0_hour1, 0) is plan
    # A real observation through the checkpoint-backed provider decodes into
    # a valid plan shape (tiny weights; plumbing smoke only).
    provider = CheckpointPlanProvider(TINY_CKPT, device="cpu")
    live_plan = CachingPlanProvider(provider).daily_plan(
        make_live_obs(0, 0, 0), 0)
    assert isinstance(live_plan, DailyPlan)

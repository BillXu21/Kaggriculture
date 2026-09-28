"""Focused opt-in claim-board behavior and bounded row assignment."""

from __future__ import annotations

import copy

from executor_v0.strip_claim_board import ClaimPhase, SchedulerMode, build_claim_board
from executor_v0.strip_claim_scheduler import (
    claim_runtime_fragment, evaluate_hypothetical_worker, schedule_claim_board,
)
from executor_v0.strip_executor import StripExecutorConfig, StripExecutorController
from executor_v0.plan import DailyPlan
from executor_v0.strip_routes import WorkerId, route_cursor_invariants_hold
from executor_v0.strip_work import (
    RowSummary, StripWorkPlan, SupplyRequirement, SupplySnapshot,
    WorkDiagnostics, WorkItem, row_key_for_tile,
)


def item(kind: str, tile: tuple[int, int], *, key: str | None = None,
         crop: str | None = None, source: str = "strip_forecast",
         requirements: tuple[SupplyRequirement, ...] = ()) -> WorkItem:
    return WorkItem(key or f"{kind}:{tile[0]},{tile[1]}", kind, tile=tile,
                    crop=crop, source=source, required_supplies=requirements,
                    row_key=row_key_for_tile(tile))


def plan(*items: WorkItem) -> StripWorkPlan:
    rows = sorted({item.row_key for item in items if item.row_key is not None})
    return StripWorkPlan(tuple(items), (), tuple(RowSummary(row, 5, 1, 0, 1)
                                                for row in rows),
                         SupplySnapshot(), WorkDiagnostics(), 0)


def board(work: StripWorkPlan, positions: dict[WorkerId, tuple[int, int]],
          *, carried=None, shed=None, seeds=None):
    claim_board = build_claim_board(work, carried or {}, shed or {}, seeds or {},
                                    epoch_id="test")
    result = schedule_claim_board(claim_board, work, positions, 24, 0)
    return claim_board, result


def test_primary_is_full_horizontal_row_and_coverage_appends():
    work = plan(item("WATER", (0, 0)), item("WATER", (1, 0)))
    claim_board, result = board(work, {WorkerId(0): (0, 0)})
    route = result.assignment.routes[0]
    assert route.segments[0].traversal in (
        tuple((0, x) for x in range(5)), tuple((0, x) for x in range(4, -1, -1)),
    )
    assert len(route.segments) == 2
    assert len(claim_board.owner_by_bundle) == 2


def test_harvest_continuation_and_seed_once():
    work = plan(item("HARVEST", (0, 0), crop="WHEAT", source="routine_harvest"))
    claim_board = build_claim_board(work, {}, {}, {"WHEAT": 1}, epoch_id="test")
    bundle = claim_board.bundles["TILE:0,0"]
    assert bundle.effective_interactions == 3
    assert bundle.global_demand == (("WHEAT", 1),)
    explicit = plan(*work.items, item("PLANT", (0, 0), crop="WHEAT",
                                     requirements=(SupplyRequirement("WHEAT", 1,
                                                                     "global_seed"),)))
    bundle2 = build_claim_board(explicit, {}, {}, {"WHEAT": 1},
                                epoch_id="test").bundles["TILE:0,0"]
    assert bundle2.effective_interactions == 3
    assert bundle2.global_demand == (("WHEAT", 1),)


def test_global_resource_is_exclusive_and_trial_does_not_mutate():
    work = plan(item("PLANT", (0, 0), crop="WHEAT",
                     requirements=(SupplyRequirement("WHEAT", 1, "global_seed"),)),
                item("PLANT", (1, 0), crop="WHEAT",
                     requirements=(SupplyRequirement("WHEAT", 1, "global_seed"),)))
    claim_board = build_claim_board(work, {}, {}, {"WHEAT": 1}, epoch_id="test")
    first = claim_board.trial(WorkerId(0), ("TILE:0,0",))
    assert first is not None
    assert claim_board.available_global()["WHEAT"] == 1
    assert claim_board.claim(first)
    assert claim_board.trial(WorkerId(1), ("TILE:1,0",)) is None
    claim_board.release("TILE:0,0")
    assert claim_board.trial(WorkerId(1), ("TILE:1,0",)) is not None


def test_carried_inventory_can_beat_closer_worker():
    work = plan(item("FEED", (0, 0), requirements=(SupplyRequirement("WHEAT", 1),)))
    _, result = board(work, {WorkerId(0): (0, 0), WorkerId(1): (0, 1)},
                      carried={WorkerId(1): {"WHEAT": 1}})
    assert result.assignment.routes[0].owner == WorkerId(1)


def test_repeatable_schedule_and_no_duplicate_owners():
    work = plan(*(item("CARE", (0, x), key=f"C:{x}:{n}")
                  for x in range(5) for n in range(2)))
    positions = {WorkerId(index): (index, 0) for index in range(3)}
    first_board, first = board(work, positions)
    second_board, second = board(work, positions)
    assert first_board.owner_by_bundle == second_board.owner_by_bundle
    assert [route.to_json_dict() for route in first.assignment.routes] == [
        route.to_json_dict() for route in second.assignment.routes
    ]
    assert len(first_board.owner_by_bundle) == len(set(first_board.owner_by_bundle))


def test_enabled_controller_executes_claimed_route_without_hiring():
    work = plan(item("WATER", (0, 0)))
    farm = {
        "farmer": [0, 0], "hands": [], "money": 0,
        "tiles": [[None] * 10 for _ in range(10)],
        "unlocked_quadrants": ["NW"],
    }
    obs = {
        "day": 3, "hour": 0, "step": 72, "player": 0,
        "farms": [farm, copy.deepcopy(farm)],
        "private": {"shed": {}, "seeds": {}, "inventories": [{}]},
    }
    crops = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
    animals = ("GOOSE", "COW", "SHEEP")
    daily = DailyPlan.create(
        crop_targets={crop: 0 for crop in crops},
        animal_targets={animal: 0 for animal in animals},
        land_count=1,
        fertilizer_by_crop={crop: 0 for crop in crops},
        care_by_animal={animal: 0 for animal in animals},
        sell_quantities={key: {hour: 0 for hour in (0, 4, 8, 12, 16, 20)}
                         for key in (*crops, "EGG", "MILK", "WOOL", "FERTILIZER")},
    )
    controller = StripExecutorController(
        config=StripExecutorConfig(enable_row_claim_board=True),
        work_builder=lambda _obs, _daily, **_kwargs: work,
    )
    result = controller.act(obs, daily)
    assert result.farmer_action == ("WATER",)
    assert result.market_actions == ()
    assert result.diagnostics["row_claim_board"]["owned_bundles"] == {
        "TILE:0,0": "FARMER",
    }
    assert result.diagnostics["submitted_hires"] == 0


def test_optional_on_primary_row_uses_existing_segment():
    work = plan(item("WATER", (0, 0)),
                item("DIG", (0, 1), source="dig_cleanup"))
    claim_board, result = board(work, {WorkerId(0): (0, 0)})
    route = result.assignment.routes[0]
    assert len(route.segments) == 1
    assert claim_board.owner_by_bundle["TILE:0,1"] == WorkerId(0)
    assert route.segments[0].represented_interactions == 2


def test_busy_row_can_have_three_disjoint_tail_owners():
    work = plan(*(item("CARE", (0, x), key=f"CARE:{x}:{n}")
                  for x in range(3) for n in range(8)))
    positions = {WorkerId(index): (index, 0) for index in range(3)}
    claim_board = build_claim_board(work, {}, {}, {}, epoch_id="test")
    result = schedule_claim_board(claim_board, work, positions, 12, 0)
    assert len(set(claim_board.owner_by_bundle.values())) == 3
    assert all(len(segment.traversal) == 1
               for route in result.assignment.routes for segment in route.segments)


def test_nearby_dense_fragment_beats_farther_higher_value():
    work = plan(
        item("WATER", (4, 0)), item("WATER", (4, 1)),
        item("WATER", (0, 4)), item("CARE", (0, 4)),
        item("CARE", (0, 4), key="CARE:0,4:second"),
    )
    claim_board = build_claim_board(work, {}, {}, {}, epoch_id="test")
    segment = claim_runtime_fragment(claim_board, work, WorkerId(0), (4, 0), 20)
    assert segment is not None
    assert segment.traversal == ((4, 0), (4, 1))
    assert set(claim_board.owner_by_bundle) == {"TILE:4,0", "TILE:4,1"}


def test_row_fragment_traverses_empty_tile_between_required_bundles():
    work = plan(item("WATER", (0, 0)), item("WATER", (0, 2)))
    claim_board = build_claim_board(work, {}, {}, {}, epoch_id="test")
    fragment = claim_board.required_fragments()[0]
    assert fragment.bundle_ids == ("TILE:0,0", "TILE:0,2")
    assert fragment.traversal == ((0, 0), (0, 1), (0, 2))
    segment = claim_runtime_fragment(claim_board, work, WorkerId(0), (0, 0), 10)
    assert segment is not None
    assert segment.traversal == fragment.traversal


def test_weed_does_not_displace_required_work_with_no_slack():
    work = plan(item("WATER", (0, 1)), item("DIG", (0, 0),
                                            source="dig_cleanup"))
    claim_board = build_claim_board(work, {}, {}, {}, epoch_id="test")
    schedule_claim_board(claim_board, work, {WorkerId(0): (0, 0)}, 2, 0)
    assert claim_board.owner_by_bundle.get("TILE:0,1") == WorkerId(0)
    assert "TILE:0,0" not in claim_board.owner_by_bundle


def test_distant_weed_does_not_trigger_dedicated_route():
    work = plan(item("DIG", (9, 9), source="dig_cleanup"))
    claim_board, result = board(work, {WorkerId(0): (0, 0)})
    assert result.assignment.routes == ()
    assert claim_board.owner_by_bundle == {}


def test_runtime_optional_claim_when_required_resource_is_unavailable():
    work = plan(
        item("PLANT", (2, 0), crop="WHEAT",
             requirements=(SupplyRequirement("WHEAT", 1, "global_seed"),)),
        item("DIG", (0, 1), source="dig_cleanup"),
    )
    claim_board = build_claim_board(work, {}, {}, {}, epoch_id="test")
    segment = claim_runtime_fragment(claim_board, work, WorkerId(0), (0, 0), 20)
    assert segment is not None
    assert segment.traversal == ((0, 1),)
    assert "TILE:2,0" not in claim_board.owner_by_bundle


def test_explicit_liquidation_mode_skips_future_optional_water():
    work = plan(item("WATER", (0, 0), source="water_optional_spare"))
    claim_board = build_claim_board(work, {}, {}, {}, epoch_id="test")
    result = schedule_claim_board(
        claim_board, work, {WorkerId(0): (0, 0)}, 20, 0,
        mode=SchedulerMode.LIQUIDATION,
    )
    assert result.assignment.routes == ()
    assert claim_board.owner_by_bundle == {}


def test_runtime_refill_appends_new_observed_work():
    work = plan(item("WATER", (0, 0)))
    claim_board, result = board(work, {WorkerId(0): (0, 0)})
    assert result.assignment.routes
    followup = plan(item("WATER", (1, 0)))
    from executor_v0.strip_claim_board import reconcile_claim_board

    reconcile_claim_board(claim_board, followup, {WorkerId(0): {}}, {}, {})
    segment = claim_runtime_fragment(claim_board, followup, WorkerId(0), (0, 0), 20)
    assert segment is not None
    assert segment.traversal == ((1, 0),)
    assert claim_board.owner_by_bundle["TILE:1,0"] == WorkerId(0)


def test_confirmed_pickup_stays_worker_local_without_reserving_shed_twice():
    feed = item("FEED", (0, 0), requirements=(SupplyRequirement("WHEAT", 1),))
    work = plan(feed)
    farmer = WorkerId(0)
    claim_board = build_claim_board(work, {farmer: {}}, {"WHEAT": 1}, {},
                                    epoch_id="test")
    reservation = claim_board.trial(farmer, ("TILE:0,0",))
    assert reservation is not None and claim_board.claim(reservation)
    assert claim_board.available_shed()["WHEAT"] == 0
    claim_board.confirm_pickup(farmer, "WHEAT", 1)
    from executor_v0.strip_claim_board import reconcile_claim_board

    reconcile_claim_board(claim_board, work, {farmer: {"WHEAT": 1}}, {}, {})
    assert claim_board.available_shed() == {}
    assert claim_board.carried_reserved[farmer]["WHEAT"] == 1
    claim_board.phase_by_bundle["TILE:0,0"] = ClaimPhase.IN_PROGRESS
    reconcile_claim_board(claim_board, plan(), {farmer: {}}, {}, {})
    assert claim_board.available_shed() == {}
    assert claim_board.owner_by_bundle == {}


def test_observed_seed_consumption_keeps_continuation_owner():
    farmer = WorkerId(0)
    planting = plan(item("PLANT", (0, 0), crop="WHEAT",
                         requirements=(SupplyRequirement("WHEAT", 1,
                                                         "global_seed"),)))
    claim_board = build_claim_board(planting, {farmer: {}}, {}, {"WHEAT": 1},
                                    epoch_id="test")
    reservation = claim_board.trial(farmer, ("TILE:0,0",))
    assert reservation is not None and claim_board.claim(reservation)
    claim_board.phase_by_bundle["TILE:0,0"] = ClaimPhase.IN_PROGRESS
    from executor_v0.strip_claim_board import reconcile_claim_board

    reconcile_claim_board(claim_board, plan(item("WATER", (0, 0))),
                          {farmer: {}}, {}, {"WHEAT": 0})
    assert claim_board.available_global()["WHEAT"] == 0
    assert claim_board.owner_by_bundle["TILE:0,0"] == farmer


def test_owned_new_successor_reserves_new_inventory_without_stealing():
    from executor_v0.strip_claim_board import reconcile_claim_board

    farmer = WorkerId(0)
    original = plan(item("WATER", (0, 0)))
    claim_board = build_claim_board(original, {farmer: {}}, {"WHEAT": 1}, {},
                                    epoch_id="test")
    reservation = claim_board.trial(farmer, ("TILE:0,0",))
    assert reservation is not None and claim_board.claim(reservation)
    claim_board.phase_by_bundle["TILE:0,0"] = ClaimPhase.IN_PROGRESS
    followup = plan(item("FEED", (0, 0),
                         requirements=(SupplyRequirement("WHEAT", 1),)))
    changed = reconcile_claim_board(claim_board, followup, {farmer: {}},
                                    {"WHEAT": 1}, {})
    assert changed == {farmer}
    assert claim_board.owner_by_bundle["TILE:0,0"] == farmer
    assert claim_board.reservations["TILE:0,0"].shed == (("WHEAT", 1),)
    assert claim_board.available_shed()["WHEAT"] == 0


def test_uncovered_snapshot_hypothetical_worker_is_pure_and_resource_aware():
    work = plan(
        item("PLANT", (0, 0), crop="WHEAT",
             requirements=(SupplyRequirement("WHEAT", 1, "global_seed"),)),
        item("PLANT", (0, 1), crop="WHEAT",
             requirements=(SupplyRequirement("WHEAT", 1, "global_seed"),)),
    )
    claim_board = build_claim_board(work, {}, {}, {"WHEAT": 1}, epoch_id="test")
    snapshot = claim_board.uncovered_snapshot(20)
    first = evaluate_hypothetical_worker(snapshot, (0, 0), 20)
    second = evaluate_hypothetical_worker(snapshot, (0, 0), 20)
    assert first == second
    assert first.effective_interactions == 1
    assert first.reservation_global == (("WHEAT", 1),)
    assert claim_board.owner_by_bundle == {}
    assert claim_board.available_global()["WHEAT"] == 1


def test_enabled_controller_cleans_convenient_weed_after_required_action():
    from tests.test_executor_v0_idle_cleanup import empty_plan, make_obs

    tiles = [[None] * 10 for _ in range(10)]
    tiles[0][1] = "WEED"
    controller = StripExecutorController(
        config=StripExecutorConfig(enable_row_claim_board=True),
        work_builder=lambda obs, _daily, **_kwargs: (
            plan(item("WATER", (0, 0))) if obs["hour"] == 0 else plan()
        ),
    )
    first = controller.act(make_obs(hour=0, farmer=(0, 0), tiles=tiles,
                                    unlocked=("NW",)), empty_plan())
    assert first.farmer_action == ("WATER",)
    second = controller.act(make_obs(hour=1, farmer=(0, 0), tiles=tiles,
                                     unlocked=("NW",)), empty_plan())
    assert second.farmer_action == ("EAST",)
    third = controller.act(make_obs(hour=2, farmer=(1, 0), tiles=tiles,
                                    unlocked=("NW",)), empty_plan())
    assert third.farmer_action == ("DIG",)


def test_enabled_controller_refills_after_primary_route_finishes():
    from tests.test_executor_v0_idle_cleanup import empty_plan, make_obs

    controller = StripExecutorController(
        config=StripExecutorConfig(enable_row_claim_board=True),
        work_builder=lambda obs, _daily, **_kwargs: plan(
            item("WATER", (0, 0) if obs["hour"] == 0 else (1, 0))
        ),
    )
    first = controller.act(make_obs(hour=0, farmer=(0, 0),
                                    unlocked=("NW",)), empty_plan())
    assert first.farmer_action == ("WATER",)
    second = controller.act(make_obs(hour=1, farmer=(0, 0),
                                     unlocked=("NW",)), empty_plan())
    assert second.farmer_action == ("SOUTH",)
    third = controller.act(make_obs(hour=2, farmer=(0, 1),
                                    unlocked=("NW",)), empty_plan())
    assert third.farmer_action == ("WATER",)
    assert controller._routes[WorkerId(0)].route_id.startswith(
        "CLAIM:FARMER:REFILL:"
    )
    assert route_cursor_invariants_hold(controller._routes[WorkerId(0)])


def test_enabled_low_telemetry_preserves_actions_and_reduces_diagnostics():
    from tests.test_executor_v0_idle_cleanup import empty_plan, make_obs

    def builder(_obs, _daily, **_kwargs):
        return plan(item("WATER", (0, 0)))

    full = StripExecutorController(
        config=StripExecutorConfig(enable_row_claim_board=True),
        work_builder=builder,
    )
    low = StripExecutorController(
        config=StripExecutorConfig(enable_row_claim_board=True),
        work_builder=builder, low_telemetry=True,
    )
    obs = make_obs(hour=0, farmer=(0, 0), unlocked=("NW",))
    normal = full.act(obs, empty_plan())
    reduced = low.act(obs, empty_plan())
    assert reduced.action_dict() == normal.action_dict()
    assert reduced.diagnostics["telemetry_mode"] == "reduced"
    assert reduced.diagnostics["row_claim_board"]["owned_bundles"] == {
        "TILE:0,0": "FARMER",
    }


def test_no_legal_work_may_pass_with_recorded_reason():
    from tests.test_executor_v0_idle_cleanup import empty_plan, make_obs

    controller = StripExecutorController(
        config=StripExecutorConfig(enable_row_claim_board=True),
        work_builder=lambda _obs, _daily, **_kwargs: plan(),
    )
    result = controller.act(make_obs(hour=1, farmer=(0, 0),
                                     unlocked=("NW",)), empty_plan())
    assert result.farmer_action == ("PASS",)
    reason = result.diagnostics["row_claim_pass_reasons"]["FARMER"]
    assert reason["reason"] == "NO_LEGAL_REACHABLE_WORK"
    assert reason["scheduler_miss"] is False


def test_enabled_path_skips_legacy_hiring_and_subset_packer(monkeypatch):
    from tests.test_executor_v0_idle_cleanup import empty_plan, make_obs

    def forbidden(*_args, **_kwargs):
        raise AssertionError("legacy global route/hiring search was called")

    monkeypatch.setattr("executor_v0.strip_executor.plan_strip_hiring", forbidden)
    monkeypatch.setattr("executor_v0.strip_executor.assign_horizontal_routes", forbidden)
    monkeypatch.setattr("executor_v0.strip_routes._pack_small_route_sets", forbidden)
    controller = StripExecutorController(
        config=StripExecutorConfig(enable_row_claim_board=True),
        work_builder=lambda _obs, _daily, **_kwargs: plan(item("WATER", (0, 0))),
    )
    result = controller.act(make_obs(hour=1, farmer=(0, 0),
                                     unlocked=("NW",)), empty_plan())
    assert result.farmer_action == ("WATER",)


def test_resource_blocked_required_row_gives_stable_positioning_target():
    from tests.test_executor_v0_idle_cleanup import empty_plan, make_obs

    work = plan(item("PLANT", (2, 0), crop="WHEAT",
                     requirements=(SupplyRequirement("WHEAT", 1,
                                                     "global_seed"),)))
    controller = StripExecutorController(
        config=StripExecutorConfig(enable_row_claim_board=True),
        work_builder=lambda _obs, _daily, **_kwargs: work,
    )
    obs = make_obs(hour=1, farmer=(0, 0), unlocked=("NW",))
    controller._finalize_day(obs, empty_plan(), work)
    assert controller._claim_refill(WorkerId(0), (0, 0), work, obs) is None
    assert controller._claim_stage(WorkerId(0), (0, 0), obs) == ("SOUTH",)


def test_enabled_retained_harvest_chain_stays_with_original_worker():
    from tests.test_executor_v0_idle_cleanup import empty_plan, make_obs

    def builder(obs, _daily, **_kwargs):
        hour = obs["hour"]
        if hour == 0:
            return plan(item("HARVEST", (0, 0), crop="WHEAT",
                             source="routine_harvest"))
        if hour == 1:
            return plan(item("PLANT", (0, 0), crop="WHEAT",
                             requirements=(SupplyRequirement("WHEAT", 1,
                                                             "global_seed"),)))
        return plan(item("WATER", (0, 0), crop="WHEAT"))

    controller = StripExecutorController(
        config=StripExecutorConfig(enable_row_claim_board=True),
        work_builder=builder,
    )
    actions = []
    for hour in range(3):
        obs = make_obs(hour=hour, farmer=(0, 0), unlocked=("NW",))
        obs["private"]["seeds"] = {"WHEAT": 0 if hour == 2 else 1}
        actions.append(controller.act(obs, empty_plan()).farmer_action)
        assert controller._claim_board is not None
        assert controller._claim_board.owner_by_bundle["TILE:0,0"] == WorkerId(0)
    assert actions == [("HARVEST",), ("PLANT", "WHEAT"), ("WATER",)]


def test_enabled_supply_pickup_is_observation_confirmed_once():
    from tests.test_executor_v0_idle_cleanup import empty_plan, make_obs

    completed = False

    def builder(_obs, _daily, **_kwargs):
        if completed:
            return plan()
        return plan(item("FEED", (0, 0),
                         requirements=(SupplyRequirement("WHEAT", 1),)))

    controller = StripExecutorController(
        config=StripExecutorConfig(enable_row_claim_board=True),
        work_builder=builder,
    )
    x, y = 0, 0
    shed = {"WHEAT": 1}
    inventory = {}
    actions = []
    for hour in range(24):
        obs = make_obs(hour=hour, farmer=(x, y), unlocked=("NW",))
        obs["private"]["shed"] = dict(shed)
        obs["private"]["inventories"] = [dict(inventory)]
        action = controller.act(obs, empty_plan()).farmer_action
        actions.append(action)
        if action == ("NORTH",):
            y -= 1
        elif action == ("SOUTH",):
            y += 1
        elif action == ("EAST",):
            x += 1
        elif action == ("WEST",):
            x -= 1
        elif action[0] == "PICKUP":
            inventory[action[1]] = inventory.get(action[1], 0) + action[2]
            shed[action[1]] -= action[2]
        elif action == ("FEED",):
            completed = True
            inventory["WHEAT"] -= 1
            break
    assert completed
    assert sum(action[0] == "PICKUP" for action in actions) == 1
    assert controller._claim_board is not None
    assert controller._claim_board.available_shed().get("WHEAT", 0) == 0


def test_observed_new_owned_stage_starts_a_supply_pickup():
    from tests.test_executor_v0_idle_cleanup import empty_plan, make_obs

    def builder(obs, _daily, **_kwargs):
        if obs["hour"] == 0:
            return plan(item("WATER", (0, 0)))
        return plan(item("FEED", (0, 0),
                         requirements=(SupplyRequirement("WHEAT", 1),)))

    controller = StripExecutorController(
        config=StripExecutorConfig(enable_row_claim_board=True),
        work_builder=builder,
    )
    first = make_obs(hour=0, farmer=(0, 0), unlocked=("NW",))
    first["private"]["shed"] = {"WHEAT": 1}
    assert controller.act(first, empty_plan()).farmer_action == ("WATER",)
    second = make_obs(hour=1, farmer=(0, 0), unlocked=("NW",))
    second["private"]["shed"] = {"WHEAT": 1}
    action = controller.act(second, empty_plan()).farmer_action
    assert action in {("SOUTH",), ("EAST",)}
    route = controller._routes[WorkerId(0)]
    assert controller._supply_plans[route.route_id].reserved_from_shed == (
        ("WHEAT", 1),
    )


def test_replaced_claim_route_does_not_protect_stale_supply_demand():
    from tests.test_executor_v0_idle_cleanup import empty_plan, make_obs

    controller = StripExecutorController(
        config=StripExecutorConfig(enable_row_claim_board=True),
        work_builder=lambda _obs, _daily, **_kwargs: plan(
            item("FEED", (0, 0),
                 requirements=(SupplyRequirement("WHEAT", 1),))
        ),
    )
    obs = make_obs(hour=0, farmer=(0, 0), unlocked=("NW",))
    obs["private"]["shed"] = {"WHEAT": 1}
    controller.act(obs, empty_plan())
    assert controller._outstanding_reservations() == {"WHEAT": 1}
    controller._routes.clear()
    assert controller._outstanding_reservations() == {}

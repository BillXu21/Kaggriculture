"""Focused opt-in claim-board behavior and bounded row assignment."""

from __future__ import annotations

import copy

from executor_v0.strip_claim_board import ClaimPhase, SchedulerMode, build_claim_board
from executor_v0.strip_claim_scheduler import (
    claim_runtime_fragment, evaluate_hypothetical_worker, schedule_claim_board,
)
from executor_v0.strip_executor import StripExecutorConfig, StripExecutorController
from executor_v0.plan import DailyPlan
from executor_v0.strip_routes import (
    WorkerId, assign_horizontal_routes, generate_horizontal_route_candidates,
    remaining_day_action_slots, route_cursor_invariants_hold,
)
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
    assert set(claim_board.claim_source_by_bundle.values()) == {"PRIMARY_TETSUYA"}


def test_primary_tetsuya_preserves_exact_small_packing_and_multirow_order():
    work = plan(item("WATER", (0, 0)), item("WATER", (2, 0)))
    positions = {WorkerId(0): (0, 0)}
    candidates = generate_horizontal_route_candidates(work)
    base = assign_horizontal_routes(
        candidates, positions, assignment_hour=0,
        remaining_action_slots=24,
        worker_action_slots={WorkerId(0): 24},
        worker_inventories={}, shed_stock={}, global_resources={},
    )
    claim_board = build_claim_board(work, {}, {}, {}, epoch_id="test")
    result = schedule_claim_board(claim_board, work, positions, 24, 0)

    assert len(base.routes[0].segments) == 2
    route = next(route for route in result.assignment.routes
                 if route.owner == WorkerId(0))
    assert [segment.segment_id for segment in route.segments[:2]] == [
        segment.segment_id for segment in base.routes[0].segments
    ]
    assert [segment.traversal for segment in route.segments[:2]] == [
        segment.traversal for segment in base.routes[0].segments
    ]
    assert claim_board.diagnostics()["base_row_candidates"] == [
        candidate.route_id for candidate in candidates
    ]
    assert claim_board.diagnostics()["base_assigned_row_ids_by_worker"]["FARMER"] == [
        segment.physical_row_id for segment in base.routes[0].segments
    ]


def test_primary_tetsuya_preserves_allocator_orientation():
    work = plan(*(item("WATER", (0, x)) for x in range(5)))
    positions = {WorkerId(0): (0, 9)}
    base = assign_horizontal_routes(
        generate_horizontal_route_candidates(work), positions,
        assignment_hour=0, remaining_action_slots=24,
        worker_action_slots={WorkerId(0): 24},
        worker_inventories={}, shed_stock={}, global_resources={},
    )
    claim_board = build_claim_board(work, {}, {}, {}, epoch_id="test")
    result = schedule_claim_board(claim_board, work, positions, 24, 0)
    actual = next(route for route in result.assignment.routes
                  if route.owner == WorkerId(0))

    assert base.routes[0].segments[0].traversal == tuple(
        (0, x) for x in range(4, -1, -1)
    )
    assert actual.segments[0].traversal == base.routes[0].segments[0].traversal


def test_primary_import_conflict_reserves_resource_once_and_leaves_work_uncovered():
    seed = (SupplyRequirement("WHEAT", 1, "global_seed"),)
    work = plan(
        item("WATER", (0, 0)),
        item("UNROUTED", (0, 0), key="SEED:0", requirements=seed),
        item("WATER", (2, 0)),
        item("UNROUTED", (2, 0), key="SEED:2", requirements=seed),
    )
    worker = WorkerId(0)
    positions = {worker: (0, 0)}
    claim_board = build_claim_board(work, {worker: {}}, {}, {"WHEAT": 1},
                                   epoch_id="test")
    result = schedule_claim_board(claim_board, work, positions, 24, 0)

    reserved = [reservation for reservation in claim_board.reservations.values()
                if dict(reservation.global_resources).get("WHEAT", 0)]
    assert len(reserved) == 1
    assert sum(dict(value.global_resources).get("WHEAT", 0)
               for value in claim_board.reservations.values()) == 1
    assert len(claim_board.diagnostics()["base_assigned_row_ids_by_worker"]["FARMER"]) == 2
    assert any(failure["reason"] == "RESOURCE_LEDGER_CONFLICT"
               for failure in claim_board.diagnostics()["primary_import_failures"])
    assert claim_board.diagnostics()["uncovered_required_bundles_after_primary_import"]
    assert "TILE:2,0" in claim_board.diagnostics()["uncovered_required"]
    assert result.assignment.routes


def test_required_tail_appends_after_imported_base_assignment(monkeypatch):
    work = plan(item("WATER", (0, 0)), item("WATER", (2, 0)))
    positions = {WorkerId(0): (0, 0)}
    normal_assign = assign_horizontal_routes

    def leave_second_candidate_for_tail(candidates, *args, **kwargs):
        candidates = tuple(candidates)
        return normal_assign(
            tuple(candidate for candidate in candidates
                  if candidate.row_key.global_row == 0),
            *args, **kwargs,
        )

    monkeypatch.setattr(
        "executor_v0.strip_claim_scheduler.assign_horizontal_routes",
        leave_second_candidate_for_tail,
    )
    claim_board = build_claim_board(work, {}, {}, {}, epoch_id="test")
    result = schedule_claim_board(claim_board, work, positions, 24, 0)
    route = result.assignment.routes[0]

    assert claim_board.claim_source_by_bundle == {
        "TILE:0,0": "PRIMARY_TETSUYA",
        "TILE:2,0": "REQUIRED_TAIL",
    }
    assert len(route.segments) == 2
    assert ":REQUIRED_TAIL:" in route.segments[1].segment_id
    assert claim_board.tail_claims_added_after_primary_import == 1


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
    assert first_board.diagnostics() == second_board.diagnostics()
    route_owner_by_bundle = {}
    for route in first.assignment.routes:
        for tile in route.traversal:
            bundle_id = f"TILE:{tile[0]},{tile[1]}"
            if first_board.owner_by_bundle.get(bundle_id) == route.owner:
                assert bundle_id not in route_owner_by_bundle
                route_owner_by_bundle[bundle_id] = route.owner
    assert route_owner_by_bundle == first_board.owner_by_bundle
    assert set(first_board.reservations) == set(first_board.owner_by_bundle)
    assert all(first_board.reservations[bundle_id].worker == owner
               for bundle_id, owner in first_board.owner_by_bundle.items())


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


def test_feature_off_keeps_base_route_assignment_behavior():
    from tests.test_executor_v0_idle_cleanup import empty_plan, make_obs

    work = plan(item("WATER", (0, 0)), item("WATER", (2, 0)))
    obs = make_obs(hour=0, farmer=(0, 9), unlocked=("NW",))
    controller = StripExecutorController(
        config=StripExecutorConfig(enable_row_claim_board=False),
        work_builder=lambda _obs, _daily, **_kwargs: work,
    )
    positions = controller._worker_positions(obs)
    inventories = {
        worker: controller._worker_inventory(obs, worker) for worker in positions
    }
    slots = remaining_day_action_slots(obs)
    expected = assign_horizontal_routes(
        generate_horizontal_route_candidates(work), positions,
        assignment_hour=0, remaining_action_slots=slots,
        worker_action_slots={worker: slots for worker in positions},
        worker_inventories=inventories,
        shed_stock=obs["private"]["shed"],
        global_resources=obs["private"]["seeds"],
    )

    controller._finalize_day(obs, empty_plan(), work)

    assert controller._claim_board is None
    assert [route.to_json_dict() for route in controller._assignment.routes] == [
        route.to_json_dict() for route in expected.routes
    ]


def test_optional_on_primary_row_uses_existing_segment():
    work = plan(item("WATER", (0, 0)),
                item("DIG", (0, 1), source="dig_cleanup"))
    claim_board, result = board(work, {WorkerId(0): (0, 0)})
    route = result.assignment.routes[0]
    assert len(route.segments) == 1
    assert claim_board.owner_by_bundle["TILE:0,1"] == WorkerId(0)
    assert route.segments[0].represented_interactions == 2
    assert claim_board.claim_source_by_bundle["TILE:0,1"] == "PRIMARY_TETSUYA"
    assert claim_board.claim_type_by_bundle["TILE:0,1"] == "OPTIONAL_CLEANUP"
    assert claim_board.diagnostics()["optional_cleanup_bundles"] == ["TILE:0,1"]


def test_primary_dense_row_matches_tetsuya_assignment():
    work = plan(*(item("CARE", (0, x), key=f"CARE:{x}:{n}")
                  for x in range(3) for n in range(8)))
    positions = {WorkerId(index): (index, 0) for index in range(3)}
    base = assign_horizontal_routes(
        generate_horizontal_route_candidates(work), positions,
        assignment_hour=0, remaining_action_slots=12,
        worker_action_slots={worker: 12 for worker in positions},
        worker_inventories={}, shed_stock={}, global_resources={},
    )
    claim_board = build_claim_board(work, {}, {}, {}, epoch_id="test")
    result = schedule_claim_board(claim_board, work, positions, 12, 0)

    assert [route.owner for route in result.assignment.routes] == [
        route.owner for route in base.routes
    ]
    assert [
        [segment.to_json_dict() for segment in route.segments]
        for route in result.assignment.routes
    ] == [
        [segment.to_json_dict() for segment in route.segments]
        for route in base.routes
    ]
    assert set(claim_board.owner_by_bundle.values()) == {
        route.owner for route in base.routes
    }
    assert set(claim_board.claim_source_by_bundle.values()) == {"PRIMARY_TETSUYA"}


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


def test_primary_assignment_claims_required_and_cleanup_bundles_together():
    work = plan(item("WATER", (0, 1)), item("DIG", (0, 0),
                                            source="dig_cleanup"))
    claim_board = build_claim_board(work, {}, {}, {}, epoch_id="test")
    schedule_claim_board(claim_board, work, {WorkerId(0): (0, 0)}, 2, 0)
    assert claim_board.owner_by_bundle.get("TILE:0,1") == WorkerId(0)
    assert claim_board.owner_by_bundle.get("TILE:0,0") == WorkerId(0)
    assert claim_board.claim_type_by_bundle["TILE:0,0"] == "OPTIONAL_CLEANUP"


def test_distant_cleanup_keeps_the_base_horizontal_row_shape():
    work = plan(item("DIG", (9, 9), source="dig_cleanup"))
    claim_board, result = board(work, {WorkerId(0): (0, 0)})
    assert len(result.assignment.routes) == 1
    segment = result.assignment.routes[0].segments[0]
    assert segment.traversal == tuple((9, x) for x in range(5, 10))
    assert claim_board.owner_by_bundle == {"TILE:9,9": WorkerId(0)}
    assert claim_board.claim_source_by_bundle["TILE:9,9"] == "PRIMARY_TETSUYA"
    assert claim_board.claim_type_by_bundle["TILE:9,9"] == "OPTIONAL_CLEANUP"


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


def test_explicit_mode_preserves_base_assignment_of_optional_work():
    work = plan(item("WATER", (0, 0), source="water_optional_spare"))
    claim_board = build_claim_board(work, {}, {}, {}, epoch_id="test")
    result = schedule_claim_board(
        claim_board, work, {WorkerId(0): (0, 0)}, 20, 0,
        mode=SchedulerMode.LIQUIDATION,
    )
    assert len(result.assignment.routes) == 1
    assert claim_board.owner_by_bundle == {"TILE:0,0": WorkerId(0)}
    assert claim_board.claim_source_by_bundle["TILE:0,0"] == "PRIMARY_TETSUYA"


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
    assert claim_board.claim_source_by_bundle["TILE:1,0"] == "RUNTIME_REFILL"


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


def test_hypothetical_hire_charges_later_supply_pickup_before_first_row():
    from executor_v0.strip_claim_scheduler import hypothetical_worker_route

    work = plan(
        item("WATER", (0, 0)),
        item("FEED", (9, 9),
             requirements=(SupplyRequirement("WHEAT", 1),)),
    )
    claim_board = build_claim_board(work, {}, {"WHEAT": 1}, {}, epoch_id="test")
    coverage = evaluate_hypothetical_worker(
        claim_board.uncovered_snapshot(24), (0, 0), 24
    )
    assert tuple(bundle for fragment in coverage.claimed_fragments
                 for bundle in fragment.bundle_ids) == ("TILE:0,0",)
    assert coverage.incremental_turns == 1
    route = hypothetical_worker_route(
        claim_board, work, WorkerId(1), (0, 0), coverage, 24, 0
    )
    assert route is not None

    feed_only = plan(item("FEED", (9, 9),
                          requirements=(SupplyRequirement("WHEAT", 1),)))
    feed_board = build_claim_board(
        feed_only, {}, {"WHEAT": 1}, {}, epoch_id="test"
    )
    feed_coverage = evaluate_hypothetical_worker(
        feed_board.uncovered_snapshot(24), (0, 0), 24
    )
    assert feed_coverage.incremental_turns == 20
    assert feed_coverage.reservation_shed == (("WHEAT", 1),)
    assert hypothetical_worker_route(
        feed_board, feed_only, WorkerId(1), (0, 0), feed_coverage, 24, 0
    ) is not None


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


def test_no_legal_work_moves_toward_center_staging_before_pass():
    from tests.test_executor_v0_idle_cleanup import empty_plan, make_obs

    controller = StripExecutorController(
        config=StripExecutorConfig(enable_row_claim_board=True),
        work_builder=lambda _obs, _daily, **_kwargs: plan(),
    )
    result = controller.act(make_obs(hour=1, farmer=(0, 0),
                                     unlocked=("NW",)), empty_plan())
    assert result.farmer_action == ("SOUTH",)
    assert "FARMER" not in result.diagnostics["row_claim_pass_reasons"]


def test_legitimate_pass_at_center_staging_is_recorded():
    from tests.test_executor_v0_idle_cleanup import empty_plan, make_obs

    controller = StripExecutorController(
        config=StripExecutorConfig(enable_row_claim_board=True),
        work_builder=lambda _obs, _daily, **_kwargs: plan(),
    )
    result = controller.act(make_obs(hour=1, farmer=(4, 4),
                                     unlocked=("NW",)), empty_plan())
    assert result.farmer_action == ("PASS",)
    reason = result.diagnostics["row_claim_pass_reasons"]["FARMER"]
    assert reason["reason"] == "ALREADY_AT_STAGING_TARGET"
    assert reason["scheduler_miss"] is False


def test_enabled_path_reuses_base_row_allocator(monkeypatch):
    from tests.test_executor_v0_idle_cleanup import empty_plan, make_obs

    base_assign = assign_horizontal_routes
    calls = []

    def tracked_assignment(candidates, *args, **kwargs):
        candidates = tuple(candidates)
        result = base_assign(candidates, *args, **kwargs)
        calls.append((candidates, result))
        return result

    def forbidden_hiring(*_args, **_kwargs):
        raise AssertionError("claim mode must use its unchanged claim-hiring path")

    monkeypatch.setattr("executor_v0.strip_executor.plan_strip_hiring", forbidden_hiring)
    monkeypatch.setattr(
        "executor_v0.strip_claim_scheduler.assign_horizontal_routes",
        tracked_assignment,
    )
    controller = StripExecutorController(
        config=StripExecutorConfig(enable_row_claim_board=True),
        work_builder=lambda _obs, _daily, **_kwargs: plan(item("WATER", (0, 0))),
    )
    result = controller.act(make_obs(hour=1, farmer=(0, 0),
                                     unlocked=("NW",)), empty_plan())
    assert result.farmer_action == ("WATER",)
    assert len(calls) == 1
    assert calls[0][0][0].source_shape == "horizontal_quadrant_row"
    assert result.diagnostics["row_claim_board"]["primary_bundles_imported"] == 1


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


def _claim_hiring_case(work, obs):
    from tests.test_executor_v0_idle_cleanup import empty_plan

    controller = StripExecutorController(
        config=StripExecutorConfig(enable_row_claim_board=True),
        work_builder=lambda _obs, _daily, **_kwargs: work,
    )
    daily = empty_plan()
    controller._start_day(obs, daily)
    positions = controller._worker_positions(obs)
    inventories = {
        worker: controller._worker_inventory(obs, worker) for worker in positions
    }
    controller._finalize_claim_day(obs, work, positions, inventories)
    orders = controller._plan_claim_hires(obs, work, positions)
    return controller, orders


def _intensive_row_work(*rows, interactions=12):
    return plan(*(
        item("CARE", (row, 0), key=f"CARE:{row}:{index}")
        for row in rows for index in range(interactions)
    ))


def test_claim_hiring_noops_when_existing_workers_cover_required_work():
    from tests.test_executor_v0_idle_cleanup import make_obs

    work = plan(item("WATER", (0, 0)))
    controller, orders = _claim_hiring_case(
        work, make_obs(hour=0, farmer=(0, 0), money=100)
    )
    assert orders == ()
    assert controller._claim_hiring_diagnostics["hire_stop_reason"] == "NO_REQUIRED_LEFTOVERS"
    assert controller._claim_board is not None
    assert not controller._claim_board.uncovered_snapshot(23).fragments


def test_claim_hire_covers_contiguous_row_tail_and_is_observation_confirmed():
    from tests.test_executor_v0_idle_cleanup import empty_plan, make_obs

    work = plan(item("WATER", (0, 0)), item("WATER", (5, 5)))
    controller = StripExecutorController(
        config=StripExecutorConfig(enable_row_claim_board=True),
        work_builder=lambda _obs, _daily, **_kwargs: work,
    )
    first_obs = make_obs(hour=19, farmer=(0, 0), money=100)
    first = controller.act(first_obs, empty_plan())
    assert first.market_actions == (("HIRE",),)
    assert controller._claim_hiring_diagnostics["wanted_hires"] == 1
    assert controller._claim_board is not None
    assert controller._claim_board.owner_by_bundle["TILE:5,5"] == WorkerId(1)
    hired_route = controller._routes[WorkerId(1)]
    assert len(hired_route.segments) == 1
    assert hired_route.segments[0].traversal == ((5, 5),)
    assert ":REQUIRED_TAIL:" in hired_route.segments[0].segment_id

    waiting = controller.act(first_obs, empty_plan())
    assert waiting.farmer_action == ("PASS",)
    assert waiting.diagnostics["row_claim_pass_reasons"]["FARMER"]["reason"] == (
        "AWAITING_OBSERVATION_CONFIRMATION"
    )

    confirmed = make_obs(
        hour=20, farmer=(0, 0), hands=((4, 4),), money=99
    )
    result = controller.act(confirmed, empty_plan())
    assert controller._routes_finalized
    assert WorkerId(1) in controller._routes
    assert controller._claim_board.owner_by_bundle["TILE:5,5"] == WorkerId(1)
    assert result.farmer_action != ("PASS",)
    assert result.diagnostics["claim_hiring"]["wanted_hires"] == 1
    assert result.diagnostics["wanted_hires"] == 1
    assert len(result.diagnostics["claim_hiring"]["planned_workers"]) == 1


def test_unconfirmed_claim_hire_releases_claim_before_retry():
    from tests.test_executor_v0_idle_cleanup import empty_plan, make_obs

    work = plan(item("WATER", (0, 0)), item("WATER", (5, 5)))
    controller = StripExecutorController(
        config=StripExecutorConfig(enable_row_claim_board=True),
        work_builder=lambda _obs, _daily, **_kwargs: work,
    )
    first = controller.act(make_obs(hour=19, farmer=(0, 0), money=100), empty_plan())
    assert first.market_actions == (("HIRE",),)
    assert controller._claim_board is not None
    assert controller._claim_board.owner_by_bundle["TILE:5,5"] == WorkerId(1)

    retry = controller.act(make_obs(hour=20, farmer=(0, 0), money=100), empty_plan())
    assert retry.market_actions == (("HIRE",),)
    assert controller._claim_board.owner_by_bundle["TILE:5,5"] == WorkerId(1)
    assert retry.diagnostics["claim_hiring"]["wanted_hires"] == 1
    assert retry.diagnostics["failed_hires"] == 1


def test_claim_hiring_plans_two_sequential_spawns_and_escalating_costs():
    from tests.test_executor_v0_idle_cleanup import make_obs

    work = _intensive_row_work(0, 2, 4, 6)
    controller, orders = _claim_hiring_case(
        work, make_obs(hour=0, farmer=(0, 0), money=100)
    )
    assert orders == (("HIRE",), ("HIRE",), ("HIRE",))
    planned = controller._claim_hiring_diagnostics["planned_workers"]
    assert [record["spawn"] for record in planned] == [[4, 4], [4, 5], [5, 4]]
    assert controller._claim_hiring_diagnostics["sequential_hire_costs"] == [1, 1, 2]
    assert len({tuple(record["bundle_ids"]) for record in planned}) == 3
    assert set(controller._claim_board.owner_by_bundle.values()) == {
        WorkerId(0), WorkerId(1), WorkerId(2), WorkerId(3),
    }


def test_claim_hiring_cash_allows_first_but_not_second_hire():
    from tests.test_executor_v0_idle_cleanup import make_obs

    controller, orders = _claim_hiring_case(
        _intensive_row_work(0, 2, 4),
        make_obs(hour=0, farmer=(0, 0), money=1),
    )
    assert orders == (("HIRE",),)
    assert controller._claim_hiring_diagnostics["hire_stop_reason"] == "CASH"
    assert controller._claim_hiring_diagnostics["sequential_hire_costs"] == [1]
    assert controller._claim_board is not None
    assert len(controller._claim_board.owner_by_bundle) < 3


def test_claim_hiring_respects_per_turn_order_cap():
    from tests.test_executor_v0_idle_cleanup import make_obs

    obs = make_obs(hour=0, farmer=(0, 0), money=100)
    obs["configuration"] = {"maxMarketOrdersPerTurn": 1, "boardSize": 10}
    controller, orders = _claim_hiring_case(_intensive_row_work(0, 2, 4), obs)
    assert orders == (("HIRE",),)
    assert controller._claim_hiring_diagnostics["hire_stop_reason"] == "ORDER_CAP"


def test_claim_hiring_does_not_hire_for_optional_only_leftovers():
    from tests.test_executor_v0_idle_cleanup import make_obs

    work = plan(item("DIG", (0, 1), source="dig_cleanup"))
    controller, orders = _claim_hiring_case(
        work, make_obs(hour=22, farmer=(0, 0), money=100)
    )
    assert orders == ()
    assert controller._claim_hiring_diagnostics["wanted_hires"] == 0


def test_claim_hiring_does_not_add_worker_for_shared_resource_block():
    from tests.test_executor_v0_idle_cleanup import make_obs

    seeds = (SupplyRequirement("WHEAT", 1, "global_seed"),)
    work = plan(
        item("PLANT", (0, 0), crop="WHEAT", requirements=seeds),
        item("PLANT", (5, 5), crop="WHEAT", key="PLANT:5:5", requirements=seeds),
    )
    obs = make_obs(hour=0, farmer=(0, 0), money=100)
    obs["private"]["seeds"] = {"WHEAT": 1}
    controller, orders = _claim_hiring_case(work, obs)
    assert orders == ()
    assert controller._claim_hiring_diagnostics["hire_stop_reason"] == (
        "RESOURCE_BLOCKED_REQUIRED_LEFTOVERS"
    )
    assert controller._claim_board is not None
    assert controller._claim_board.owner_by_bundle == {"TILE:0,0": WorkerId(0)}


def test_claim_hiring_is_deterministic_for_same_observation():
    from tests.test_executor_v0_idle_cleanup import make_obs

    work = _intensive_row_work(0, 2, 4)
    obs = make_obs(hour=0, farmer=(0, 0), money=100)
    first, first_orders = _claim_hiring_case(work, copy.deepcopy(obs))
    second, second_orders = _claim_hiring_case(work, copy.deepcopy(obs))
    assert first_orders == second_orders
    assert first._claim_hiring_diagnostics == second._claim_hiring_diagnostics
    assert first._claim_board.owner_by_bundle == second._claim_board.owner_by_bundle


def test_route_less_worker_claims_unclaimed_required_work_before_pass():
    from tests.test_executor_v0_idle_cleanup import empty_plan, make_obs

    hand = WorkerId(1)
    controller = StripExecutorController(
        config=StripExecutorConfig(enable_row_claim_board=True),
        work_builder=lambda obs, _daily, **_kwargs: (
            plan(item("FEED", (5, 5), requirements=(SupplyRequirement("WHEAT", 1),)))
            if obs["hour"] > 0 else plan()
        ),
    )
    initial = make_obs(hour=0, farmer=(0, 0), hands=((5, 5),), unlocked=("NW",))
    initial["private"]["inventories"] = [{}, {"WHEAT": 1}]
    controller.act(initial, empty_plan())
    observed = make_obs(hour=1, farmer=(0, 0), hands=((5, 5),), unlocked=("NW",))
    observed["private"]["inventories"] = [{}, {"WHEAT": 1}]
    result = controller.act(observed, empty_plan())
    assert result.hands_actions[0] == ("FEED",)
    assert result.hands_actions[0] != ("PASS",)
    assert controller._claim_board.owner_by_bundle["TILE:5,5"] == hand

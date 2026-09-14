"""Additional block-reason coverage for the strip forecast contract."""

from test_strip_work import obs, plan, plant

from executor_v0.strip_work import BlockReason, WorkStatus, build_strip_work_plan


def test_seed_and_fertilizer_shortages_are_distinct_global_and_inventory_blocks():
    seed_short = build_strip_work_plan(obs(), plan(crop_targets={"WHEAT": 1}))
    planting = next(item for item in seed_short.items if item.kind == "PLANT")
    assert planting.status == WorkStatus.BLOCKED
    assert planting.block_reason == BlockReason.MISSING_GLOBAL_RESOURCE

    board = [[None] * 10 for _ in range(10)]
    board[0][0] = plant("WHEAT", planted_day=1)
    fertilizer_short = build_strip_work_plan(
        obs(board, day=3), plan(fertilizer_by_crop={"WHEAT": 1})
    )
    treatment = next(item for item in fertilizer_short.items if item.kind == "FERTILIZE")
    water = next(
        item for item in fertilizer_short.items
        if item.kind == "WATER" and item.tile == (0, 0)
    )
    assert treatment.block_reason == BlockReason.MISSING_SUPPLY
    assert water.block_reason == BlockReason.DEPENDENCY_BLOCKED


def test_unresolved_demand_reports_no_spatial_slot_without_fabricating_a_tile():
    board = [[None] * 10 for _ in range(10)]
    for y in range(5):
        for x in range(5):
            board[y][x] = "LOCKED"
    result = build_strip_work_plan(obs(board), plan(crop_targets={"WHEAT": 1}))
    unresolved = next(
        item for item in result.items
        if item.id == "UNRESOLVED_PLANT:WHEAT"
    )
    assert unresolved.tile is None
    assert unresolved.block_reason == BlockReason.NO_SPATIAL_SLOT
    assert result.diagnostics.unresolved_crop_delta_dict["WHEAT"] == 1


def test_animal_chain_orders_purchase_before_build_before_place():
    result = build_strip_work_plan(obs(), plan(animal_targets={"GOOSE": 1}))
    chain = next(item for item in result.chains if item.kind == "ANIMAL_EXPANSION")
    kinds = [next(work for work in result.items if work.id == item_id).kind
             for item_id in chain.item_ids]
    assert kinds == ["BUY_ANIMAL", "BUILD_COOP", "PLACE"]

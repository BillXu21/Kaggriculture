from __future__ import annotations

from dataclasses import replace
from itertools import product

import pytest

from executor_v0.strip_cost import RouteCostSegment, RouteCostWork, simulate_route_cost
from executor_v0.strip_prefix_trie import RouteCostTrie
from executor_v0.strip_work import BlockReason, WorkStatus


def work(
    work_id,
    *,
    inventory=(),
    global_requirements=(),
    continuation=0,
    continuation_global=(),
    depends=(),
    status=WorkStatus.READY,
    reason=None,
    turns=1,
):
    return RouteCostWork(
        work_id,
        "FEED",
        turns,
        continuation,
        True,
        inventory,
        global_requirements,
        continuation_global,
        depends,
        status,
        reason,
    )


def oriented(segment_id, traversal, work_by_tile):
    forward = RouteCostSegment(segment_id, traversal, work_by_tile)
    return forward, replace(
        forward,
        traversal=tuple(reversed(traversal)),
        work_by_tile=tuple(reversed(work_by_tile)),
    )


def metrics(result):
    return (
        result.total_turns,
        result.setup_travel_turns,
        result.pickup_travel_turns,
        result.inter_segment_travel_turns,
        result.effective_interactions_completed_before_deadline,
        result.segments_completed_before_deadline,
        result.effective_interactions_missed,
    )


@pytest.mark.parametrize("budget", [0, 1, 8, 9, 10, 30, None])
@pytest.mark.parametrize("shed", [None, {}, {"WHEAT": 1}, {"WHEAT": 8}])
@pytest.mark.parametrize("global_resources", [None, {}, {"CARROT": 2, "WHEAT": 1}])
def test_all_orders_orientations_and_subsets_match_original(
    budget, shed, global_resources
):
    segments = (
        oriented(
            "a",
            ((0, 0), (0, 1)),
            (
                (work("feed", inventory=(("WHEAT", 1),)),),
                (
                    work(
                        "care",
                        global_requirements=(("CARROT", 1),),
                        depends=("feed",),
                        status=WorkStatus.BLOCKED,
                        reason=BlockReason.DEPENDENCY_BLOCKED,
                    ),
                ),
            ),
        ),
        oriented(
            "b",
            ((1, 0), (1, 1)),
            (
                (work("harvest", continuation=2, continuation_global=(("WHEAT", 1),)),),
                (work("feed2", inventory=(("WHEAT", 2),)),),
            ),
        ),
        oriented(
            "c",
            ((2, 0),),
            (
                (
                    work(
                        "blocked",
                        status=WorkStatus.BLOCKED,
                        reason=BlockReason.DEPENDENCY_BLOCKED,
                        depends=("care",),
                    ),
                ),
            ),
        ),
    )
    carried = {"WHEAT": 1}
    trie = RouteCostTrie(
        segments,
        (0, 0),
        remaining_action_slots=budget,
        carried_inventory=carried,
        shed_stock=shed,
        global_resources=global_resources,
    )
    for mask in range(1, 8):
        indices = tuple(i for i in range(3) if mask & (1 << i))
        for order in (indices, tuple(reversed(indices))):
            for sides in product((0, 1), repeat=len(order)):
                path = tuple((i, side, 0) for i, side in zip(order, sides))
                expected = simulate_route_cost(
                    (0, 0),
                    tuple(segments[i][side] for i, side, _ in path),
                    remaining_action_slots=10**9 if budget is None else budget,
                    carried_inventory=carried,
                    shed_stock=shed,
                    global_resources=global_resources,
                    include_segment_results=False,
                )
                actual = trie.evaluate(mask, path)
                assert metrics(actual) == metrics(expected)
                nodes = trie.nodes_created
                assert trie.evaluate(mask, path) is actual
                assert trie.nodes_created == nodes


def test_siblings_own_resource_ledgers_and_prefix_state_is_unhashable():
    segments = (
        oriented("a", ((0, 0),), ((work("a", global_requirements=(("CARROT", 1),)),),)),
        oriented("b", ((1, 0),), ((work("b", global_requirements=(("CARROT", 1),)),),)),
    )
    trie = RouteCostTrie(
        segments, (0, 0), remaining_action_slots=10, global_resources={"CARROT": 1}
    )
    first = trie.evaluate(3, ((0, 0, 0), (1, 0, 0)))
    reverse = trie.evaluate(3, ((1, 0, 0), (0, 0, 0)))
    assert first.effective_interactions_completed_before_deadline == 1
    assert reverse.effective_interactions_completed_before_deadline == 1
    root = trie._mask_roots[3]
    assert root.state.global_resources == {"CARROT": 1}
    assert root.children[(0, 0)].state.global_resources == {"CARROT": 0}
    assert root.children[(1, 0)].state.global_resources == {"CARROT": 0}
    with pytest.raises(TypeError):
        hash(root.state)


def test_future_demand_cannot_reuse_a_different_pickup_root():
    segments = (
        oriented("a", ((0, 0),), ((work("a"),),)),
        oriented("b", ((1, 0),), ((work("b", inventory=(("WHEAT", 1),)),),)),
    )
    trie = RouteCostTrie(
        segments, (0, 0), remaining_action_slots=10, shed_stock={"WHEAT": 1}
    )
    trie.evaluate(1, ((0, 0, 0),))
    trie.evaluate(3, ((0, 0, 0), (1, 0, 0)))
    assert trie._mask_roots[1] is not trie._mask_roots[3]
    assert trie.root_nodes_created == 2


def test_invalid_budget_is_rejected():
    with pytest.raises(ValueError):
        RouteCostTrie((), (0, 0), remaining_action_slots=-1)


@pytest.mark.parametrize("packing", ["small", "large", "large_frontier"])
def test_packing_releases_nodes_while_preserving_plan_caches(monkeypatch, packing):
    import executor_v0.strip_routes as routes
    from executor_v0.strip_work import row_key_for_tile

    count = 3 if packing == "small" else 9
    candidates = tuple(
        routes.HorizontalRouteCandidate(
            f"release:{row}",
            row_key_for_tile((row, 0)),
            ((row, 0),),
            1,
            1,
            0,
            tile_interactions=(1,),
        )
        for row in range(count)
    )
    workers = (routes.WorkerId(0), routes.WorkerId(1))
    positions = {worker: (0, 0) for worker in workers}
    slots = {worker: 20 for worker in workers}
    contexts = []
    original_create = routes._RoutePlanContext.create

    def create(cls, values):
        context = original_create(values)
        contexts.append(context)
        return context

    monkeypatch.setattr(routes._RoutePlanContext, "create", classmethod(create))
    routes._cached_small_chain_plan_for_context.cache_clear()
    routes._cached_chain_plan_for_context.cache_clear()
    if packing == "small":
        routes._pack_small_route_sets_frontier(
            candidates, workers, positions, (1, 2), slots
        )
    elif packing == "large":
        routes._pack_large_route_set(candidates, workers, positions, slots)
    else:
        routes._pack_large_route_set_frontier(
            candidates, workers, positions, (1, 2), slots
        )
    assert contexts
    assert all(not context.prefix_tries for context in contexts)
    small_cache = routes._cached_small_chain_plan_for_context.cache_info()
    large_cache = routes._cached_chain_plan_for_context.cache_info()
    assert small_cache.maxsize == large_cache.maxsize == 4096
    assert (small_cache.currsize if packing == "small" else large_cache.currsize) > 0

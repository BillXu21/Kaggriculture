"""Assignment-local structural reuse of the original chain cost simulation.

The ordered/oriented path is the identity. Simulator state is node-owned data,
never a cache key. Only fields used by chain selection are materialized.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Mapping, Sequence

from executor_v0.strip_cost import (
    RouteCostSegment,
    _CostSegmentSummary,
    _distance,
    _summarize_cost_segment,
    _work_can_progress,
    nearest_shed_access,
)


@dataclass(slots=True)
class _State:
    position: tuple[int, int]
    elapsed: int
    pickup_travel: int
    completed: int = 0
    feasible: int = 0
    segments_completed: int = 0
    inventory: dict[str, int] = field(default_factory=dict)
    global_resources: dict[str, int] | None = None
    feasible_ids: set[str] = field(default_factory=set)
    setup_travel: int = 0
    inter_segment_travel: int = 0
    segment_count: int = 0


@dataclass(frozen=True, slots=True)
class _ChainCost:
    total_turns: int
    setup_travel_turns: int
    pickup_travel_turns: int
    inter_segment_travel_turns: int
    effective_interactions_completed_before_deadline: int
    segments_completed_before_deadline: int
    effective_interactions_missed: int


@dataclass(slots=True)
class _Node:
    state: _State
    children: dict[tuple[int, int], _Node] = field(default_factory=dict)
    result: _ChainCost | None = None


class RouteCostTrie:
    """One frozen candidate set, worker start, budget, and resource context.

    Whole-chain inventory pickup precedes any segment. Subsets with different
    initial inventories or pickup counts therefore use distinct roots. Roots
    are compared once per subset without hashing ledgers; edges use only the
    existing candidate index and orientation. Pickup order cannot change chain
    metrics: the original simulator charges one action per picked item before
    segment evaluation and makes the entire planned inventory available.
    """

    def __init__(
        self,
        oriented_segments: tuple[tuple[RouteCostSegment, RouteCostSegment], ...],
        start_position: tuple[int, int],
        *,
        remaining_action_slots: int | None = None,
        carried_inventory: Mapping[str, int] | None = None,
        shed_stock: Mapping[str, int] | None = None,
        global_resources: Mapping[str, int] | None = None,
        profile: bool = False,
    ) -> None:
        if remaining_action_slots is not None and remaining_action_slots < 0:
            raise ValueError("remaining_action_slots must be nonnegative")
        self.segments = oriented_segments
        self.start = start_position
        self.budget = (
            10**9 if remaining_action_slots is None else remaining_action_slots
        )
        self.carried = dict(carried_inventory or {})
        self.shed = None if shed_stock is None else dict(shed_stock)
        self.global_resources = (
            None
            if global_resources is None
            else {
                str(item): max(0, int(quantity))
                for item, quantity in global_resources.items()
            }
        )
        self.profile = profile
        self._summaries: dict[tuple[int, int], _CostSegmentSummary] = {}
        self._mask_roots: dict[int, _Node] = {}
        self._roots: list[tuple[dict[str, int], int, _Node]] = []
        self.segment_visits_before = 0
        self.nodes_created = 0
        self.root_nodes_created = 0
        self.lookup_hits = 0
        self.state_extension_seconds = 0.0
        self.state_copying_seconds = 0.0
        self.trie_lookup_creation_seconds = 0.0
        self.result_construction_seconds = 0.0
        self.context_preparation_seconds = 0.0

    def _summary(self, key: tuple[int, int]) -> _CostSegmentSummary:
        summary = self._summaries.get(key)
        if summary is None:
            summary = _summarize_cost_segment(self.segments[key[0]][key[1]])
            self._summaries[key] = summary
        return summary

    def _root(self, mask: int) -> _Node:
        root = self._mask_roots.get(mask)
        if root is not None:
            return root
        started = perf_counter() if self.profile else 0.0
        demand: dict[str, int] = {}
        remaining = mask
        while remaining:
            bit = remaining & -remaining
            index = bit.bit_length() - 1
            for item, quantity in self._summary((index, 0)).inventory_demand:
                demand[item] = demand.get(item, 0) + quantity
            remaining ^= bit
        inventory: dict[str, int] = {}
        pickup_count = 0
        for item, quantity in demand.items():
            carried = min(quantity, max(0, int(self.carried.get(item, 0))))
            need = quantity - carried
            take = (
                need
                if self.shed is None
                else min(need, max(0, int(self.shed.get(item, 0))))
            )
            pickup_count += int(take > 0)
            if carried + take > 0:
                inventory[item] = carried + take
        for existing_inventory, existing_count, candidate in self._roots:
            if pickup_count == existing_count and inventory == existing_inventory:
                root = candidate
                break
        else:
            position = nearest_shed_access(self.start) if pickup_count else self.start
            travel = _distance(self.start, position)
            root = _Node(
                _State(
                    position,
                    travel + pickup_count,
                    travel,
                    inventory=inventory.copy(),
                    global_resources=(
                        None
                        if self.global_resources is None
                        else self.global_resources.copy()
                    ),
                )
            )
            self._roots.append((inventory, pickup_count, root))
            self.root_nodes_created += 1
        self._mask_roots[mask] = root
        if self.profile:
            self.context_preparation_seconds += perf_counter() - started
        return root

    def _extend(self, parent: _State, key: tuple[int, int]) -> _State:
        started = perf_counter() if self.profile else 0.0
        state = _State(
            parent.position,
            parent.elapsed,
            parent.pickup_travel,
            parent.completed,
            parent.feasible,
            parent.segments_completed,
            parent.inventory.copy(),
            None if parent.global_resources is None else parent.global_resources.copy(),
            parent.feasible_ids.copy(),
            parent.setup_travel,
            parent.inter_segment_travel,
            parent.segment_count,
        )
        if self.profile:
            now = perf_counter()
            self.state_copying_seconds += now - started
            started = now
        segment = self.segments[key[0]][key[1]]
        summary = self._summary(key)
        # Hot loop: this runs once per trie node, so the accumulators and the
        # ledger containers are hoisted into locals. The containers themselves
        # are still the same objects, so in-place mutation is unchanged, and
        # ``elapsed``/``feasible``/``completed`` are written back below.
        budget = self.budget
        travel = _distance(state.position, segment.traversal[0])
        elapsed = state.elapsed + travel
        if state.segment_count:
            state.inter_segment_travel += travel
        else:
            state.setup_travel = travel
        segment_feasible = segment_completed = 0
        segment_blocked = False
        feasible = state.feasible
        completed_total = state.completed
        feasible_ids = state.feasible_ids
        inventory = state.inventory
        global_resources = state.global_resources
        movement_turns = summary.tile_movement_turns
        # ``segment.traversal`` is only needed for its endpoints here; the
        # previous zip/enumerate over it built a tuple per tile for nothing.
        for tile_index, tile_work in enumerate(segment.work_by_tile):
            if tile_index:
                elapsed += movement_turns[tile_index]
            for work in tile_work:
                if not _work_can_progress(
                    work, feasible_ids, inventory, global_resources
                ):
                    segment_blocked = True
                    continue
                feasible_ids.add(work.work_id)
                feasible_turns = work.represented_turns
                continuation = work.continuation_global_requirements
                # An explicit loop avoids building a generator per work item in
                # the innermost loop; the branch structure is unchanged, so a
                # blocked continuation with no continuation turns still leaves
                # the segment unblocked.
                if global_resources is None:
                    feasible_turns += work.continuation_turns
                else:
                    shortfall = False
                    for item, quantity in continuation:
                        if global_resources.get(item, 0) < quantity:
                            shortfall = True
                            break
                    if shortfall:
                        if work.continuation_turns:
                            segment_blocked = True
                    else:
                        for item, quantity in continuation:
                            global_resources[item] -= quantity
                        feasible_turns += work.continuation_turns
                feasible += feasible_turns
                segment_feasible += feasible_turns
                if feasible_turns:
                    done = min(feasible_turns, max(0, budget - elapsed))
                    completed_total += done
                    segment_completed += done
                    elapsed += feasible_turns
        state.elapsed = elapsed
        state.feasible = feasible
        state.completed = completed_total
        state.position = segment.traversal[-1]
        state.segment_count += 1
        state.segments_completed += int(
            not segment_blocked
            and segment_completed == segment_feasible
            and state.elapsed <= self.budget
        )
        if self.profile:
            self.state_extension_seconds += perf_counter() - started
        return state

    def evaluate(self, mask: int, path: Sequence[tuple[int, int, int]]) -> _ChainCost:
        node = self._root(mask)
        self.segment_visits_before += len(path)
        if self.profile:
            for index, side, _distance_to_entry in path:
                started = perf_counter()
                key = (index, side)
                child = node.children.get(key)
                self.trie_lookup_creation_seconds += perf_counter() - started
                if child is None:
                    state = self._extend(node.state, key)
                    started = perf_counter()
                    child = _Node(state)
                    node.children[key] = child
                    self.nodes_created += 1
                    self.trie_lookup_creation_seconds += perf_counter() - started
                else:
                    self.lookup_hits += 1
                node = child
        else:
            # Unprofiled hot path: the instrumentation above costs a clock read
            # per path element, which dominates a walk that is otherwise one
            # dict lookup. Counters are folded into locals and flushed once;
            # nothing reads them between the loop and the flush.
            extend = self._extend
            created = hits = 0
            for index, side, _distance_to_entry in path:
                key = (index, side)
                child = node.children.get(key)
                if child is None:
                    node.children[key] = child = _Node(extend(node.state, key))
                    created += 1
                else:
                    hits += 1
                node = child
            self.nodes_created += created
            self.lookup_hits += hits
        if node.result is None:
            if self.profile:
                started = perf_counter()
            state = node.state
            node.result = _ChainCost(
                state.elapsed,
                state.setup_travel,
                state.pickup_travel,
                state.inter_segment_travel,
                state.completed,
                state.segments_completed,
                state.feasible - state.completed,
            )
            if self.profile:
                self.result_construction_seconds += perf_counter() - started
        return node.result

    def statistics(self) -> dict[str, int | float]:
        return {
            "segment_visits_before": self.segment_visits_before,
            "segment_visits_after": self.nodes_created,
            "trie_nodes_created": self.nodes_created + self.root_nodes_created,
            "root_nodes_created": self.root_nodes_created,
            "lookup_hits": self.lookup_hits,
            "state_extension_seconds": self.state_extension_seconds,
            "state_copying_seconds": self.state_copying_seconds,
            "trie_lookup_creation_seconds": self.trie_lookup_creation_seconds,
            "result_construction_seconds": self.result_construction_seconds,
            "context_preparation_seconds": self.context_preparation_seconds,
        }

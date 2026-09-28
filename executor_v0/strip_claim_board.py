"""Exclusive tile claims and one shared resource ledger for strip scheduling."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping

from executor_v0.strip_cost import route_cost_segment_from_items
from executor_v0.strip_routes import WorkerId
from executor_v0.strip_work import (
    BlockReason, RowKey, StripWorkPlan, WorkItem, WorkStatus,
    forecast_effective_interactions, row_key_for_tile,
)


class SchedulerMode(StrEnum):
    NORMAL = "NORMAL"
    LIQUIDATION = "LIQUIDATION"


class ServiceClass(StrEnum):
    HARD_REQUIRED = "HARD_REQUIRED"
    REQUIRED = "REQUIRED"
    OPTIONAL = "OPTIONAL"


class ClaimPhase(StrEnum):
    UNCLAIMED = "UNCLAIMED"
    CLAIMED = "CLAIMED"
    IN_PROGRESS = "IN_PROGRESS"
    DONE = "DONE"
    FAILED = "FAILED"


@dataclass(frozen=True)
class TileServiceBundle:
    bundle_id: str
    tile: tuple[int, int]
    row_key: RowKey
    items: tuple[WorkItem, ...]
    continuation_stages: tuple[tuple[str, tuple[str, ...]], ...]
    service_class: ServiceClass
    effective_interactions: int
    inventory_demand: tuple[tuple[str, int], ...]
    global_demand: tuple[tuple[str, int], ...]
    source_rank: int
    claimable: bool


@dataclass(frozen=True)
class RowBoard:
    row_key: RowKey
    tiles: tuple[tuple[int, int], ...]
    bundle_ids_by_tile: tuple[str | None, ...]


@dataclass(frozen=True)
class RowFragment:
    row_key: RowKey
    bundle_ids: tuple[str, ...]
    traversal: tuple[tuple[int, int], ...]
    role: str


@dataclass(frozen=True)
class ClaimReservation:
    worker: WorkerId
    bundle_ids: tuple[str, ...]
    carried: tuple[tuple[str, int], ...]
    shed: tuple[tuple[str, int], ...]
    global_resources: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class BundleCoverageView:
    bundle_id: str
    tile: tuple[int, int]
    effective_interactions: int
    inventory_demand: tuple[tuple[str, int], ...]
    global_demand: tuple[tuple[str, int], ...]
    hard_required: bool


@dataclass(frozen=True)
class UncoveredRequiredWork:
    epoch_id: str
    observation_version: int
    fragments: tuple[RowFragment, ...]
    bundle_views: tuple[BundleCoverageView, ...]
    remaining_shed: tuple[tuple[str, int], ...]
    remaining_global: tuple[tuple[str, int], ...]
    horizon_slots: int

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "epoch_id": self.epoch_id,
            "observation_version": self.observation_version,
            "fragments": [
                {"row_key": fragment.row_key.to_json_dict(),
                 "bundle_ids": list(fragment.bundle_ids),
                 "traversal": [list(tile) for tile in fragment.traversal]}
                for fragment in self.fragments
            ],
            "bundles": [
                {"bundle_id": view.bundle_id, "tile": list(view.tile),
                 "effective_interactions": view.effective_interactions,
                 "inventory_demand": dict(view.inventory_demand),
                 "global_demand": dict(view.global_demand),
                 "hard_required": view.hard_required}
                for view in self.bundle_views
            ],
            "remaining_shed": dict(self.remaining_shed),
            "remaining_global": dict(self.remaining_global),
            "horizon_slots": self.horizon_slots,
        }


_REPAIRABLE = {
    BlockReason.DEPENDENCY_BLOCKED,
    BlockReason.MISSING_SUPPLY,
    BlockReason.MISSING_GLOBAL_RESOURCE,
    BlockReason.MISSING_PURCHASE,
}


def _pairs(values: Mapping[str, int]) -> tuple[tuple[str, int], ...]:
    return tuple(sorted((key, value) for key, value in values.items() if value > 0))


def _bundle(tile: tuple[int, int], items: tuple[WorkItem, ...],
            fertilizer_ids: frozenset[str]) -> TileServiceBundle:
    forecast = forecast_effective_interactions(items)
    segment = route_cost_segment_from_items("BUNDLE", (tile,), items,
                                            fertilizer_item_ids=fertilizer_ids)
    inventory: dict[str, int] = defaultdict(int)
    global_demand: dict[str, int] = defaultdict(int)
    for work in segment.work_by_tile[0]:
        for key, amount in work.inventory_requirements:
            inventory[key] += amount
        for key, amount in work.global_requirements + work.continuation_global_requirements:
            global_demand[key] += amount
    optional = all(
        item.source in {"optional_deferrable", "fertilizer_policy",
                        "fertilizer_linked_productive", "dig_cleanup",
                        "water_optional_spare"}
        or item.id in fertilizer_ids or item.kind == "FERTILIZE"
        for item in items
    )
    service = (ServiceClass.HARD_REQUIRED
               if any(item.source == "survival_weed_prevention" for item in items)
               else ServiceClass.OPTIONAL if optional else ServiceClass.REQUIRED)
    claimable = any(
        item.status == WorkStatus.READY or item.block_reason in _REPAIRABLE
        for item in items
    )
    return TileServiceBundle(
        f"TILE:{tile[0]},{tile[1]}", tile, row_key_for_tile(tile), items,
        forecast.continuation_stages_by_work_item, service,
        forecast.effective_interactions, _pairs(inventory), _pairs(global_demand),
        min((0 if item.source == "survival_weed_prevention" else 1)
            for item in items), claimable,
    )


@dataclass
class ClaimBoard:
    bundles: dict[str, TileServiceBundle]
    rows: dict[RowKey, RowBoard]
    observed_shed: dict[str, int]
    observed_global: dict[str, int]
    worker_carried: dict[WorkerId, dict[str, int]]
    epoch_id: str
    observation_version: int = 0
    owner_by_bundle: dict[str, WorkerId] = field(default_factory=dict)
    phase_by_bundle: dict[str, ClaimPhase] = field(default_factory=dict)
    reservations: dict[str, ClaimReservation] = field(default_factory=dict)
    remaining_shed: dict[str, int] = field(default_factory=dict)
    remaining_global: dict[str, int] = field(default_factory=dict)
    carried_reserved: dict[WorkerId, dict[str, int]] = field(default_factory=dict)
    resource_shortfalls: dict[str, int] = field(default_factory=dict)
    claims_considered: int = 0

    def __post_init__(self) -> None:
        for bundle_id in self.bundles:
            self.phase_by_bundle.setdefault(bundle_id, ClaimPhase.UNCLAIMED)
        self.remaining_shed = dict(self.observed_shed)
        self.remaining_global = dict(self.observed_global)

    def available_shed(self) -> dict[str, int]:
        return dict(self.remaining_shed)

    def available_global(self) -> dict[str, int]:
        return dict(self.remaining_global)

    def trial(self, worker: WorkerId, bundle_ids: tuple[str, ...]) -> ClaimReservation | None:
        """Return a reservation proposal without changing any authoritative state."""
        self.claims_considered += 1
        if not bundle_ids or any(
            self.phase_by_bundle.get(bundle_id) != ClaimPhase.UNCLAIMED
            or not self.bundles[bundle_id].claimable for bundle_id in bundle_ids
        ):
            return None
        carried = dict(self.worker_carried.get(worker, {}))
        for key, amount in self.carried_reserved.get(worker, {}).items():
            carried[key] = carried.get(key, 0) - amount
        shed = self.available_shed()
        global_stock = self.available_global()
        taken_carried: dict[str, int] = defaultdict(int)
        taken_shed: dict[str, int] = defaultdict(int)
        taken_global: dict[str, int] = defaultdict(int)
        for bundle_id in bundle_ids:
            bundle = self.bundles[bundle_id]
            for key, amount in bundle.inventory_demand:
                local = min(max(0, carried.get(key, 0)), amount)
                carried[key] = carried.get(key, 0) - local
                taken_carried[key] += local
                needed = amount - local
                if shed.get(key, 0) < needed:
                    return None
                shed[key] = shed.get(key, 0) - needed
                taken_shed[key] += needed
            for key, amount in bundle.global_demand:
                if global_stock.get(key, 0) < amount:
                    return None
                global_stock[key] = global_stock.get(key, 0) - amount
                taken_global[key] += amount
        return ClaimReservation(worker, bundle_ids, _pairs(taken_carried),
                                _pairs(taken_shed), _pairs(taken_global))

    def claim(self, reservation: ClaimReservation) -> bool:
        if self.trial(reservation.worker, reservation.bundle_ids) != reservation:
            return False
        # Commit a fragment atomically, then keep tile-sized records so a
        # confirmed completed tile can release only its own unused stock.
        for bundle_id in reservation.bundle_ids:
            part = self.trial(reservation.worker, (bundle_id,))
            if part is None:
                raise AssertionError("resource allocation changed during claim")
            self.reservations[bundle_id] = part
            for key, amount in part.shed:
                self.remaining_shed[key] = self.remaining_shed.get(key, 0) - amount
            for key, amount in part.global_resources:
                self.remaining_global[key] = self.remaining_global.get(key, 0) - amount
            local = self.carried_reserved.setdefault(reservation.worker, {})
            for key, amount in part.carried:
                local[key] = local.get(key, 0) + amount
            self.owner_by_bundle[bundle_id] = reservation.worker
            self.phase_by_bundle[bundle_id] = ClaimPhase.CLAIMED
        return True

    def release(self, bundle_id: str) -> None:
        reservation = self.reservations.pop(bundle_id, None)
        if reservation is not None:
            for key, amount in reservation.shed:
                self.remaining_shed[key] = self.remaining_shed.get(key, 0) + amount
            for key, amount in reservation.global_resources:
                self.remaining_global[key] = self.remaining_global.get(key, 0) + amount
            local = self.carried_reserved.get(reservation.worker, {})
            for key, amount in reservation.carried:
                local[key] = local.get(key, 0) - amount
        self.owner_by_bundle.pop(bundle_id, None)
        self.phase_by_bundle[bundle_id] = ClaimPhase.UNCLAIMED

    def confirm_pickup(self, worker: WorkerId, item: str, quantity: int) -> None:
        """Move observed shed reservations into that worker's carried ledger."""
        remaining = max(0, quantity)
        for bundle_id, reservation in tuple(self.reservations.items()):
            if reservation.worker != worker or remaining <= 0:
                continue
            shed = dict(reservation.shed)
            moved = min(remaining, shed.get(item, 0))
            if not moved:
                continue
            shed[item] -= moved
            carried = dict(reservation.carried)
            carried[item] = carried.get(item, 0) + moved
            self.reservations[bundle_id] = ClaimReservation(
                worker, reservation.bundle_ids, _pairs(carried), _pairs(shed),
                reservation.global_resources,
            )
            self.remaining_shed[item] = self.remaining_shed.get(item, 0) + moved
            local = self.carried_reserved.setdefault(worker, {})
            local[item] = local.get(item, 0) + moved
            remaining -= moved

    def release_failed_pickup(self, worker: WorkerId, item: str, quantity: int) -> None:
        """Return only unacquired stock after confirmed pickup failure."""
        remaining = max(0, quantity)
        for bundle_id, reservation in tuple(self.reservations.items()):
            if reservation.worker != worker or remaining <= 0:
                continue
            shed = dict(reservation.shed)
            released = min(remaining, shed.get(item, 0))
            if not released:
                continue
            shed[item] -= released
            self.reservations[bundle_id] = ClaimReservation(
                worker, reservation.bundle_ids, reservation.carried, _pairs(shed),
                reservation.global_resources,
            )
            self.remaining_shed[item] = self.remaining_shed.get(item, 0) + released
            remaining -= released

    def sync_owned_reservations(self) -> None:
        """Refresh a locked tile's demand after observation adds successors."""
        for bundle_id, old in tuple(self.reservations.items()):
            bundle = self.bundles[bundle_id]
            carried = dict(old.carried)
            shed = dict(old.shed)
            global_held = dict(old.global_resources)
            desired_inventory = dict(bundle.inventory_demand)
            desired_global = dict(bundle.global_demand)
            local_reserved = self.carried_reserved.setdefault(old.worker, {})
            observed_carried = self.worker_carried.get(old.worker, {})
            for key in sorted(set(carried) | set(shed) | set(desired_inventory)):
                needed = desired_inventory.get(key, 0)
                held = carried.get(key, 0) + shed.get(key, 0)
                if held > needed:
                    extra = held - needed
                    give_shed = min(extra, shed.get(key, 0))
                    shed[key] = shed.get(key, 0) - give_shed
                    self.remaining_shed[key] = self.remaining_shed.get(key, 0) + give_shed
                    extra -= give_shed
                    if extra:
                        carried[key] = carried.get(key, 0) - extra
                        local_reserved[key] = local_reserved.get(key, 0) - extra
                elif held < needed:
                    missing = needed - held
                    available_local = max(
                        0, observed_carried.get(key, 0) - local_reserved.get(key, 0)
                    )
                    take_local = min(missing, available_local)
                    carried[key] = carried.get(key, 0) + take_local
                    local_reserved[key] = local_reserved.get(key, 0) + take_local
                    missing -= take_local
                    take_shed = min(missing, max(0, self.remaining_shed.get(key, 0)))
                    shed[key] = shed.get(key, 0) + take_shed
                    self.remaining_shed[key] = self.remaining_shed.get(key, 0) - take_shed
            for key in sorted(set(global_held) | set(desired_global)):
                delta = desired_global.get(key, 0) - global_held.get(key, 0)
                if delta < 0:
                    global_held[key] += delta
                    self.remaining_global[key] = self.remaining_global.get(key, 0) - delta
                elif delta > 0:
                    take = min(delta, max(0, self.remaining_global.get(key, 0)))
                    global_held[key] = global_held.get(key, 0) + take
                    self.remaining_global[key] = self.remaining_global.get(key, 0) - take
            self.reservations[bundle_id] = ClaimReservation(
                old.worker, old.bundle_ids, _pairs(carried), _pairs(shed),
                _pairs(global_held),
            )

    def unclaimed(self, service: ServiceClass | None = None) -> tuple[str, ...]:
        return tuple(sorted(
            bundle_id for bundle_id, bundle in self.bundles.items()
            if self.phase_by_bundle[bundle_id] == ClaimPhase.UNCLAIMED
            and bundle.claimable
            and (service is None or bundle.service_class == service)
        ))

    def required_fragments(self) -> tuple[RowFragment, ...]:
        fragments: list[RowFragment] = []
        for row in sorted(self.rows):
            run: list[str] = []
            def flush() -> None:
                if not run:
                    return
                first = self.bundles[run[0]].tile
                last = self.bundles[run[-1]].tile
                fragments.append(RowFragment(
                    row, tuple(run),
                    tuple((row.global_row, x) for x in range(first[1], last[1] + 1)),
                    "REQUIRED_TAIL",
                ))
                run.clear()

            for bundle_id in self.rows[row].bundle_ids_by_tile:
                if bundle_id is None:
                    continue
                bundle = self.bundles[bundle_id]
                if (bundle is not None and bundle.service_class != ServiceClass.OPTIONAL
                        and bundle.claimable and
                        self.phase_by_bundle[bundle_id] == ClaimPhase.UNCLAIMED):
                    run.append(bundle_id)
                elif self.phase_by_bundle[bundle_id] in {
                    ClaimPhase.CLAIMED, ClaimPhase.IN_PROGRESS,
                }:
                    flush()
            flush()
        return tuple(fragments)

    def uncovered_snapshot(self, horizon_slots: int) -> UncoveredRequiredWork:
        fragments = self.required_fragments()
        ids = {bundle_id for fragment in fragments for bundle_id in fragment.bundle_ids}
        return UncoveredRequiredWork(
            self.epoch_id, self.observation_version, fragments,
            tuple(
                BundleCoverageView(
                    bundle_id, bundle.tile, bundle.effective_interactions,
                    bundle.inventory_demand, bundle.global_demand,
                    bundle.service_class == ServiceClass.HARD_REQUIRED,
                )
                for bundle_id, bundle in sorted(self.bundles.items())
                if bundle_id in ids
            ),
            _pairs(self.available_shed()), _pairs(self.available_global()), horizon_slots,
        )

    def diagnostics(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "epoch_id": self.epoch_id,
            "observation_version": self.observation_version,
            "owned_bundles": {key: worker.label for key, worker in sorted(
                self.owner_by_bundle.items())},
            "uncovered_required": [bundle_id for fragment in self.required_fragments()
                                   for bundle_id in fragment.bundle_ids],
            "optional_leftovers": list(self.unclaimed(ServiceClass.OPTIONAL)),
            "claims_considered": self.claims_considered,
            "remaining_shed": self.available_shed(),
            "remaining_global": self.available_global(),
            "resource_shortfalls": dict(self.resource_shortfalls),
        }


def build_claim_board(work_plan: StripWorkPlan,
                      worker_carried: Mapping[WorkerId, Mapping[str, int]],
                      shed: Mapping[str, int], seeds: Mapping[str, int],
                      *, epoch_id: str) -> ClaimBoard:
    fertilizer_ids = frozenset(
        item_id for chain in work_plan.chains
        if chain.kind == "FERTILIZER_UPKEEP" or chain.source == "fertilizer_policy"
        for item_id in chain.item_ids
    )
    by_tile: dict[tuple[int, int], list[WorkItem]] = defaultdict(list)
    for item in work_plan.items:
        if item.tile is not None:
            by_tile[item.tile].append(item)
    bundles = {
        f"TILE:{tile[0]},{tile[1]}": _bundle(tile, tuple(items), fertilizer_ids)
        for tile, items in sorted(by_tile.items())
    }
    rows: dict[RowKey, RowBoard] = {}
    for bundle in bundles.values():
        key = bundle.row_key
        if key not in rows:
            tiles = tuple((key.global_row, x) for x in range(key.x_start, key.x_end + 1))
            rows[key] = RowBoard(key, tiles, tuple(
                f"TILE:{tile[0]},{tile[1]}" if tile in by_tile else None
                for tile in tiles
            ))
    return ClaimBoard(bundles, rows,
                      {str(k): max(0, int(v)) for k, v in shed.items()},
                      {str(k): max(0, int(v)) for k, v in seeds.items()},
                      {worker: dict(stock) for worker, stock in worker_carried.items()},
                      epoch_id)


def reconcile_claim_board(board: ClaimBoard, work_plan: StripWorkPlan,
                          worker_carried: Mapping[WorkerId, Mapping[str, int]],
                          shed: Mapping[str, int], seeds: Mapping[str, int]) -> set[WorkerId]:
    """Refresh current bundles by stable tile key and retain existing owners."""
    refreshed = build_claim_board(work_plan, worker_carried, shed, seeds,
                                  epoch_id=board.epoch_id)
    changed_owners = {
        owner for bundle_id, owner in board.owner_by_bundle.items()
        if (bundle_id not in refreshed.bundles
            or board.bundles[bundle_id].items != refreshed.bundles[bundle_id].items)
    }
    for key, before in board.observed_global.items():
        consumed = max(0, before - refreshed.observed_global.get(key, 0))
        for bundle_id, reservation in tuple(board.reservations.items()):
            if consumed <= 0:
                break
            if board.phase_by_bundle[bundle_id] != ClaimPhase.IN_PROGRESS:
                continue
            global_demand = dict(reservation.global_resources)
            used = min(consumed, global_demand.get(key, 0))
            if used:
                global_demand[key] -= used
                board.reservations[bundle_id] = ClaimReservation(
                    reservation.worker, reservation.bundle_ids,
                    reservation.carried, reservation.shed, _pairs(global_demand),
                )
                consumed -= used
    for worker, previous in board.worker_carried.items():
        current = refreshed.worker_carried.get(worker, {})
        for key, before in previous.items():
            consumed = max(0, before - current.get(key, 0))
            for bundle_id, reservation in tuple(board.reservations.items()):
                if consumed <= 0:
                    break
                if (reservation.worker != worker or
                        board.phase_by_bundle[bundle_id] != ClaimPhase.IN_PROGRESS):
                    continue
                carried = dict(reservation.carried)
                used = min(consumed, carried.get(key, 0))
                if used:
                    carried[key] -= used
                    board.reservations[bundle_id] = ClaimReservation(
                        worker, reservation.bundle_ids, _pairs(carried),
                        reservation.shed, reservation.global_resources,
                    )
                    consumed -= used
    previously_done = {bundle_id for bundle_id, phase in board.phase_by_bundle.items()
                       if phase == ClaimPhase.DONE}
    for bundle_id in tuple(board.bundles):
        if bundle_id not in refreshed.bundles:
            board.release(bundle_id)
            board.phase_by_bundle[bundle_id] = ClaimPhase.DONE
    board.bundles.update(refreshed.bundles)
    board.rows.update(refreshed.rows)
    board.worker_carried = refreshed.worker_carried
    board.carried_reserved = {}
    for reservation in board.reservations.values():
        local = board.carried_reserved.setdefault(reservation.worker, {})
        for key, amount in reservation.carried:
            local[key] = local.get(key, 0) + amount
    board.observed_shed = refreshed.observed_shed
    board.observed_global = refreshed.observed_global
    board.remaining_shed = dict(board.observed_shed)
    board.remaining_global = dict(board.observed_global)
    for reservation in board.reservations.values():
        for key, amount in reservation.shed:
            board.remaining_shed[key] = board.remaining_shed.get(key, 0) - amount
        for key, amount in reservation.global_resources:
            board.remaining_global[key] = board.remaining_global.get(key, 0) - amount
    board.sync_owned_reservations()
    for bundle_id in tuple(reversed(board.reservations)):
        if (not any(amount < 0 for amount in board.remaining_shed.values())
                and not any(amount < 0 for amount in board.remaining_global.values())):
            break
        bundle = board.bundles[bundle_id]
        if (bundle.service_class == ServiceClass.HARD_REQUIRED
                or board.phase_by_bundle[bundle_id] == ClaimPhase.IN_PROGRESS):
            continue
        board.release(bundle_id)
    board.resource_shortfalls = {
        f"shed:{key}": -amount for key, amount in board.remaining_shed.items()
        if amount < 0
    } | {
        f"global:{key}": -amount for key, amount in board.remaining_global.items()
        if amount < 0
    }
    for bundle_id in board.bundles:
        board.phase_by_bundle.setdefault(bundle_id, ClaimPhase.UNCLAIMED)
        if bundle_id in refreshed.bundles and bundle_id in previously_done:
            board.phase_by_bundle[bundle_id] = ClaimPhase.UNCLAIMED
    board.observation_version += 1
    return changed_owners

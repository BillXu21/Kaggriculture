"""Observed-state farm utilization and land purchase diagnostics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

QUADRANT_ORDER = ("NW", "NE", "SW", "SE")


def farm_utilization_snapshot(
    farm: Mapping[str, Any],
    *,
    day: int,
    episode: int | None = None,
    seat: int | None = None,
    boundary: str = "daily",
) -> dict[str, Any]:
    """Count productive tiles from the observed canonical board only."""
    tiles: Sequence[Any] = [tile for row in farm["tiles"] for tile in row]
    unlocked = [tile for tile in tiles if tile != "LOCKED"]
    crop_squares = sum(
        isinstance(tile, Mapping) and tile.get("kind") == "PLANT"
        for tile in unlocked)
    animal_squares = sum(
        isinstance(tile, Mapping) and tile.get("animal") is not None
        for tile in unlocked)
    unlocked_squares = len(unlocked)
    productive_squares = crop_squares + animal_squares
    if unlocked_squares <= 0:
        raise ValueError("observed farm must have at least one unlocked square")
    if productive_squares > unlocked_squares:
        raise ValueError(
            f"productive squares {productive_squares} exceed unlocked squares "
            f"{unlocked_squares}")
    occupancy = productive_squares / unlocked_squares
    if not 0.0 <= occupancy <= 1.0:
        raise ValueError(f"productive occupancy outside [0, 1]: {occupancy}")
    result: dict[str, Any] = {
        "day": int(day),
        "bank": float(farm["money"]),
        "land_quadrants_owned": len(farm["unlocked_quadrants"]),
        "unlocked_squares": unlocked_squares,
        "crop_squares": int(crop_squares),
        "animal_squares": int(animal_squares),
        "productive_squares": int(productive_squares),
        "productive_occupancy": float(occupancy),
        "boundary": str(boundary),
    }
    if episode is not None:
        result["episode"] = int(episode)
    if seat is not None:
        result["seat"] = int(seat)
    return result


def observed_land_purchase_events(
    previous_farm: Mapping[str, Any],
    current_farm: Mapping[str, Any],
    *,
    episode: int,
    seat: int,
    day: int,
    hour: int,
) -> list[dict[str, Any]]:
    """Record only quadrants newly visible in the post-step observation."""
    before = set(str(value) for value in previous_farm["unlocked_quadrants"])
    after = set(str(value) for value in current_farm["unlocked_quadrants"])
    new_quadrants = after - before
    if not new_quadrants:
        return []
    unexpected = new_quadrants - set(QUADRANT_ORDER)
    if unexpected:
        raise ValueError(f"unknown observed unlocked quadrants: {sorted(unexpected)}")
    expected_new = [quadrant for quadrant in QUADRANT_ORDER
                    if quadrant in new_quadrants]
    return [
        {
            "episode": int(episode),
            "seat": int(seat),
            "quadrant": quadrant,
            "submitted_day": int(day),
            "submitted_hour": int(hour),
            "causal_day": int(day),
            "causal_hour": int(hour),
        }
        for quadrant in expected_new
    ]


__all__ = [
    "QUADRANT_ORDER",
    "farm_utilization_snapshot",
    "observed_land_purchase_events",
]

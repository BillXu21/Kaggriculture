"""Diagnostic-only BC probe: carrot_start prefix through d2 -> BC at d3h0.

This intentionally does NOT change the opening-book production contract.  The
committed ``carrot_start`` identity remains a 96-turn d0-d3 opening with its
normal d4h0 handoff.  This tool reuses only its first 72 literal actions so BC-E
can be observed while the source replay's early CARROT plants are still live.

Example::

    python -m tools.probe_carrot_d3_handoff \
        --checkpoint C:/path/to/best.pt --seeds 7,42,2026 --seats 0,1
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from typing import Any

from executor_v0.smoke import detect_engine
from opening_book.eval import adapt_one_arg, make_checkpoint_downstream_factory, pass_action
from opening_book.trace import action_for, load_built_in_trace, validate_action
from oracle.provenance import ProvenanceError, verify_official_provenance

ENGINE_ENV_ID = "kaggriculture"
HANDOFF_DAY = 3
PREFIX_TURNS = HANDOFF_DAY * 24


def _csv_ints(text: str, *, name: str) -> list[int]:
    try:
        values = [int(part.strip()) for part in text.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must be comma-separated ints") from exc
    if not values:
        raise argparse.ArgumentTypeError(f"{name} must not be empty")
    return values


def _farm_summary(obs: Mapping[str, Any], seat: int) -> dict[str, Any]:
    farm = obs["farms"][seat]
    crops: dict[str, int] = {}
    animals: dict[str, int] = {}
    for row in farm.get("tiles") or []:
        for tile in row:
            if not isinstance(tile, Mapping):
                continue
            if isinstance(tile.get("animal"), str):
                species = str(tile["animal"])
                animals[species] = animals.get(species, 0) + 1
            elif tile.get("kind") == "PLANT" and isinstance(tile.get("crop"), str):
                crop = str(tile["crop"])
                crops[crop] = crops.get(crop, 0) + 1
    private = obs.get("private") if isinstance(obs.get("private"), Mapping) else {}
    return {
        "day": int(obs["day"]),
        "hour": int(obs["hour"]),
        "money": farm.get("money"),
        "crops": crops,
        "animals": animals,
        "seeds": dict(private.get("seeds") or {}),
        "shed": dict(private.get("shed") or {}),
        "town": list((obs.get("town") or {}).get("unlocked_shops") or []),
    }


class CarrotD3PrefixAgent:
    """Play carrot_start d0-d2 literally, then permanently delegate at d3h0."""

    def __init__(self, downstream: Any, seat: int) -> None:
        self.trace = load_built_in_trace("carrot_start")
        self.downstream = downstream
        self.seat = seat
        self.cursor = 0
        self.handoff: dict[str, Any] | None = None

    def __call__(self, obs: Mapping[str, Any]) -> dict[str, Any]:
        day = int(obs["day"])
        hour = int(obs["hour"])
        if day >= HANDOFF_DAY:
            if self.handoff is None:
                if (day, hour) != (HANDOFF_DAY, 0):
                    raise RuntimeError(f"expected clean d3h0 handoff, got {(day, hour)}")
                if self.cursor != PREFIX_TURNS:
                    raise RuntimeError(
                        f"expected {PREFIX_TURNS} prefix turns, replayed {self.cursor}"
                    )
                self.handoff = _farm_summary(obs, self.seat)
            return self.downstream(obs)

        expected = (self.cursor // 24, self.cursor % 24)
        if (day, hour) != expected:
            raise RuntimeError(f"prefix phase mismatch: expected {expected}, got {(day, hour)}")
        action = action_for(self.trace, day, hour)
        validate_action(action, label=f"carrot_d3_prefix day={day} hour={hour}")
        farm = obs["farms"][self.seat]
        hands = farm.get("hands") or []
        if len(hands) != len(action["hands"]):
            raise RuntimeError(
                f"hand-count mismatch at {(day, hour)}: observed {len(hands)}, "
                f"trace expects {len(action['hands'])}"
            )
        self.cursor += 1
        return action


def _requested_day(diag: Mapping[str, Any], day: int) -> Mapping[str, Any]:
    days = diag.get("days") or {}
    entry = days.get(str(day)) if isinstance(days, Mapping) else None
    if entry is None and isinstance(days, Mapping):
        entry = days.get(day)
    if not isinstance(entry, Mapping) or not isinstance(entry.get("requested"), Mapping):
        raise RuntimeError(f"downstream diagnostics missing requested plan for day {day}")
    return entry["requested"]


def _run_one(checkpoint: str, device: str, seed: int, seat: int) -> dict[str, Any]:
    import kaggle_environments

    downstream = make_checkpoint_downstream_factory(checkpoint, device, seat)()
    wrapper = CarrotD3PrefixAgent(downstream, seat)

    def opponent(obs, config):  # noqa: ARG001
        return pass_action(obs)

    agents: list[Any] = [opponent, opponent]
    agents[seat] = adapt_one_arg(wrapper)

    env = kaggle_environments.make(ENGINE_ENV_ID, configuration={"seed": seed})
    env.reset()
    steps = env.run(agents)
    if wrapper.handoff is None:
        raise RuntimeError("game completed without d3h0 handoff")

    diagnostics = downstream.diagnostics_json()
    requested = dict(_requested_day(diagnostics, HANDOFF_DAY))
    final_farms = steps[-1][0].observation.farms
    return {
        "seed": seed,
        "seat": seat,
        "prefix_turns": wrapper.cursor,
        "handoff": wrapper.handoff,
        "requested": requested,
        "final_bank": final_farms[seat]["money"],
    }


def _print_table(rows: list[dict[str, Any]]) -> None:
    header = (
        "Seed Seat | H_Carrot H_Wheat H_Strawb H_Melon | "
        "BC_Carrot BC_Wheat BC_Strawb BC_Melon | BC_Goose BC_Cow BC_Sheep"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        h = row["handoff"]["crops"]
        plan = row["requested"]
        crops = plan["crop_targets"]
        animals = plan["animal_targets"]
        print(
            f"{row['seed']:4d} {row['seat']:4d} | "
            f"{int(h.get('CARROT', 0)):8d} {int(h.get('WHEAT', 0)):7d} "
            f"{int(h.get('STRAWBERRY', 0)):7d} {int(h.get('MELON', 0)):7d} | "
            f"{int(crops['CARROT']):9d} {int(crops['WHEAT']):8d} "
            f"{int(crops['STRAWBERRY']):8d} {int(crops['MELON']):8d} | "
            f"{int(animals['GOOSE']):8d} {int(animals['COW']):6d} "
            f"{int(animals['SHEEP']):8d}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", default="7,42,2026")
    parser.add_argument("--seats", default="0,1")
    parser.add_argument("--out", default=None, help="optional JSON output path")
    args = parser.parse_args(argv)

    if not os.path.isfile(args.checkpoint):
        parser.error(f"checkpoint not found: {args.checkpoint}")
    seeds = _csv_ints(args.seeds, name="seeds")
    seats = _csv_ints(args.seats, name="seats")
    if any(seat not in (0, 1) for seat in seats):
        parser.error("seats must contain only 0 and/or 1")

    engine = detect_engine()
    if not engine.get("available"):
        raise RuntimeError(f"official engine unavailable: {engine.get('reason')}")
    if engine.get("version") != "1.32.7":
        raise RuntimeError(f"expected kaggle-environments 1.32.7, got {engine.get('version')}")
    try:
        provenance = verify_official_provenance()
    except ProvenanceError as exc:
        raise RuntimeError(f"official provenance mismatch: {exc}") from exc

    rows = [
        _run_one(args.checkpoint, args.device, seed, seat)
        for seat in seats
        for seed in seeds
    ]
    _print_table(rows)

    if args.out:
        payload = {
            "experiment": "carrot_start_d3h0_bc_probe",
            "engine": engine,
            "provenance": provenance,
            "rows": rows,
        }
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

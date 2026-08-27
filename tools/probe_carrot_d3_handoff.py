"""Diagnostic-only BC probe: carrot_start prefix through d2 -> BC at d3h0.

This intentionally does NOT change the opening-book production contract. The
committed ``carrot_start`` identity remains a 96-turn d0-d3 opening with its
normal d4h0 handoff. This tool reuses only its first 72 literal actions so BC-E
can be observed while the source replay's early CARROT plants are still live.

By default the downstream executor uses the promoted Stage-4 economics:
prior-debt suppression ON, aggressive sell-all ON, strict mode ON, cleanup OFF.
Use ``--no-aggressive-sell-all`` only for an explicit diagnostic comparison.
Full official-engine replay JSON can be emitted with ``--replay-dir``.

Example::

    python -m tools.probe_carrot_d3_handoff \
        --checkpoint C:/path/to/best.pt --seeds 7,42,2026 --seats 0,1 \
        --replay-dir artifacts/carrot-d3-replays
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from typing import Any

from executor_v0.agent import AgentConfig, make_agent
from executor_v0.smoke import detect_engine
from opening_book.eval import adapt_one_arg, pass_action
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


def _write_replay(env: Any, path: str) -> None:
    payload = env.toJSON()
    with open(path, "w", encoding="utf-8") as fh:
        if isinstance(payload, str):
            fh.write(payload)
            if not payload.endswith("\n"):
                fh.write("\n")
        else:
            json.dump(payload, fh, sort_keys=True)
            fh.write("\n")


def _run_one(
    checkpoint: str,
    device: str,
    seed: int,
    seat: int,
    *,
    aggressive_sell_all: bool,
    replay_dir: str | None,
) -> dict[str, Any]:
    import kaggle_environments

    config = AgentConfig(
        strict=True,
        suppress_expansion_from_prior_debt=True,
        aggressive_sell_all=aggressive_sell_all,
        optional_idle_cleanup=False,
        optional_spare_watering=False,
    )
    downstream = make_agent(
        checkpoint=checkpoint,
        device=device,
        seat=seat,
        config=config,
    )
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
    row = {
        "seed": seed,
        "seat": seat,
        "prefix_turns": wrapper.cursor,
        "aggressive_sell_all": aggressive_sell_all,
        "handoff": wrapper.handoff,
        "requested": requested,
        "final_bank": final_farms[seat]["money"],
        "executor_config": diagnostics.get("config", {}),
        "fallback_errors": diagnostics.get("fallback_errors", []),
    }
    if replay_dir:
        os.makedirs(replay_dir, exist_ok=True)
        replay_path = os.path.join(
            replay_dir,
            f"carrot-d3-seed{seed}-seat{seat}-sell{int(aggressive_sell_all)}.json",
        )
        _write_replay(env, replay_path)
        row["replay_path"] = replay_path
    return row


def _print_table(rows: list[dict[str, Any]]) -> None:
    header = (
        "Seed Seat Shop            | H_Carrot Cash | "
        "BC_Carrot BC_Strawb BC_Melon | FinalBank"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        h = row["handoff"]
        crops = row["requested"]["crop_targets"]
        shops = ",".join(h.get("town") or []) or "-"
        print(
            f"{row['seed']:4d} {row['seat']:4d} {shops[:15]:15s} | "
            f"{int(h['crops'].get('CARROT', 0)):8d} {int(h.get('money') or 0):4d} | "
            f"{int(crops['CARROT']):9d} {int(crops['STRAWBERRY']):8d} "
            f"{int(crops['MELON']):8d} | {int(row['final_bank']):9d}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", default="7,42,2026")
    parser.add_argument("--seats", default="0,1")
    parser.add_argument("--out", default=None, help="optional JSON output path")
    parser.add_argument(
        "--aggressive-sell-all",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use the Stage-4 aggressive sell override (default: on)",
    )
    parser.add_argument(
        "--replay-dir",
        default=None,
        help="optional directory for full official-engine replay JSONs",
    )
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
        _run_one(
            args.checkpoint,
            args.device,
            seed,
            seat,
            aggressive_sell_all=args.aggressive_sell_all,
            replay_dir=args.replay_dir,
        )
        for seat in seats
        for seed in seeds
    ]
    _print_table(rows)

    if args.out:
        payload = {
            "experiment": "carrot_start_d3h0_bc_probe",
            "engine": engine,
            "provenance": provenance,
            "aggressive_sell_all": args.aggressive_sell_all,
            "rows": rows,
        }
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Diagnostic A/B for the carrot d3 handoff under executor sell configuration.

Runs the same replay-derived d0-d2 carrot prefix, then hands BC-E control at
d3h0 in two arms:

- default: executor default selling (aggressive_sell_all=False)
- aggressive: the Stage-4 diagnostic baseline setting (aggressive_sell_all=True)

This tool is research-only. It does not change opening-book or executor defaults.
It prints final banks and writes full official replay JSON for visual inspection.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from executor_v0.agent import AgentConfig, make_agent
from executor_v0.smoke import detect_engine
from opening_book.eval import adapt_one_arg, pass_action
from oracle.provenance import ProvenanceError, verify_official_provenance
from tools.probe_carrot_d3_handoff import CarrotD3PrefixAgent, HANDOFF_DAY, _requested_day

ENGINE_ENV_ID = "kaggriculture"


def _csv_ints(text: str, *, name: str) -> list[int]:
    try:
        values = [int(part.strip()) for part in text.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must be comma-separated ints") from exc
    if not values:
        raise argparse.ArgumentTypeError(f"{name} must not be empty")
    return values


def _run_one(
    checkpoint: str,
    device: str,
    seed: int,
    seat: int,
    aggressive_sell_all: bool,
    replay_dir: Path,
) -> dict[str, Any]:
    import kaggle_environments

    config = AgentConfig(
        strict=True,
        suppress_expansion_from_prior_debt=True,
        aggressive_sell_all=aggressive_sell_all,
    )
    downstream = make_agent(
        checkpoint=checkpoint,
        device=device,
        seat=seat,
        config=config,
    )
    wrapper = CarrotD3PrefixAgent(downstream, seat)

    def opponent(obs, configuration):  # noqa: ARG001
        return pass_action(obs)

    agents: list[Any] = [opponent, opponent]
    agents[seat] = adapt_one_arg(wrapper)

    env = kaggle_environments.make(ENGINE_ENV_ID, configuration={"seed": seed})
    env.reset()
    steps = env.run(agents)
    if wrapper.handoff is None:
        raise RuntimeError("game completed without d3h0 handoff")

    diag = downstream.diagnostics_json()
    requested = dict(_requested_day(diag, HANDOFF_DAY))
    final_farms = steps[-1][0].observation.farms
    bank = float(final_farms[seat]["money"])

    mode = "aggressive" if aggressive_sell_all else "default"
    replay_dir.mkdir(parents=True, exist_ok=True)
    replay_path = replay_dir / f"carrot-d3-{mode}-seed{seed}-seat{seat}.json"
    replay = env.toJSON()
    replay_path.write_text(json.dumps(replay), encoding="utf-8")

    return {
        "seed": seed,
        "seat": seat,
        "mode": mode,
        "aggressive_sell_all": aggressive_sell_all,
        "handoff": wrapper.handoff,
        "requested": requested,
        "final_bank": bank,
        "replay": str(replay_path),
        "fallback_errors": list(diag.get("errors") or []),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", default="7,42,2026")
    parser.add_argument("--seats", default="0,1")
    parser.add_argument(
        "--replay-dir",
        default="artifacts/carrot-d3-visual",
        help="directory for full official replay JSON files",
    )
    parser.add_argument("--out", default="artifacts/carrot-d3-sell-ab.json")
    args = parser.parse_args(argv)

    if not os.path.isfile(args.checkpoint):
        parser.error(f"checkpoint not found: {args.checkpoint}")
    seeds = _csv_ints(args.seeds, name="seeds")
    seats = _csv_ints(args.seats, name="seats")
    if any(seat not in (0, 1) for seat in seats):
        parser.error("seats must contain only 0 and/or 1")

    engine = detect_engine()
    if not engine.get("available") or engine.get("version") != "1.32.7":
        raise RuntimeError(f"expected official 1.32.7 engine, got {engine}")
    try:
        provenance = verify_official_provenance()
    except ProvenanceError as exc:
        raise RuntimeError(f"official provenance mismatch: {exc}") from exc

    replay_dir = Path(args.replay_dir)
    rows: list[dict[str, Any]] = []
    for seat in seats:
        for seed in seeds:
            for aggressive in (False, True):
                rows.append(
                    _run_one(
                        args.checkpoint,
                        args.device,
                        seed,
                        seat,
                        aggressive,
                        replay_dir,
                    )
                )

    print("Seed Seat | DefaultBank AggressiveBank Delta | Shops")
    print("---------------------------------------------------------")
    keyed = {(r["seed"], r["seat"], r["mode"]): r for r in rows}
    for seat in seats:
        for seed in seeds:
            d = keyed[(seed, seat, "default")]
            a = keyed[(seed, seat, "aggressive")]
            shops = ",".join(d["handoff"].get("town") or [])
            print(
                f"{seed:4d} {seat:4d} | {d['final_bank']:11.0f} "
                f"{a['final_bank']:14.0f} {a['final_bank'] - d['final_bank']:+7.0f} | {shops}"
            )

    payload = {
        "experiment": "carrot_d3_sell_mode_ab",
        "engine": engine,
        "provenance": provenance,
        "rows": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    print(f"replays: {replay_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

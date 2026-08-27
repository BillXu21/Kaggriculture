"""Exact d3h0 shop counterfactual for the carrot opening.

Replays carrot_start through d2, captures the real d3h0 observation, then asks
fresh copies of the same BC-E checkpoint for a plan after changing only
``town.unlocked_shops``. Counterfactual actions are never submitted to the
environment; the real game continues with the unmodified observation.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from collections.abc import Mapping
from typing import Any

from executor_v0.agent import AgentConfig, make_agent
from executor_v0.smoke import detect_engine
from opening_book.eval import adapt_one_arg, pass_action
from oracle.provenance import ProvenanceError, verify_official_provenance
from tools.probe_carrot_d3_handoff import (
    HANDOFF_DAY,
    CarrotD3PrefixAgent,
    _csv_ints,
    _requested_day,
)

ENGINE_ENV_ID = "kaggriculture"
DEFAULT_SHOPS = ("FARMERS_MARKET", "YARN_STORE")


def _stage4_config() -> AgentConfig:
    return AgentConfig(
        strict=True,
        suppress_expansion_from_prior_debt=True,
        aggressive_sell_all=True,
        optional_idle_cleanup=False,
        optional_spare_watering=False,
    )


def _replace_shops(obs: Mapping[str, Any], shops: list[str]) -> dict[str, Any]:
    cloned = copy.deepcopy(dict(obs))
    town = copy.deepcopy(dict(cloned.get("town") or {}))
    town["unlocked_shops"] = list(shops)
    cloned["town"] = town
    return cloned


def _one_plan(checkpoint: str, device: str, seat: int,
              obs: Mapping[str, Any], shop: str) -> dict[str, Any]:
    probe = make_agent(
        checkpoint=checkpoint,
        device=device,
        seat=seat,
        config=_stage4_config(),
    )
    cf_obs = _replace_shops(obs, [shop])
    probe(cf_obs)  # action intentionally discarded; this is inference-only.
    requested = _requested_day(probe.diagnostics_json(), HANDOFF_DAY)
    return {
        "shop": shop,
        "crop_targets": dict(requested["crop_targets"]),
        "animal_targets": dict(requested["animal_targets"]),
    }


class CounterfactualPrefixAgent(CarrotD3PrefixAgent):
    def __init__(self, downstream: Any, seat: int, *, checkpoint: str,
                 device: str, shops: list[str]) -> None:
        super().__init__(downstream, seat)
        self.checkpoint = checkpoint
        self.device = device
        self.shops = shops
        self.counterfactuals: list[dict[str, Any]] | None = None

    def __call__(self, obs: Mapping[str, Any]) -> dict[str, Any]:
        day, hour = int(obs["day"]), int(obs["hour"])
        if (day, hour) == (HANDOFF_DAY, 0) and self.counterfactuals is None:
            self.counterfactuals = [
                _one_plan(self.checkpoint, self.device, self.seat, obs, shop)
                for shop in self.shops
            ]
        return super().__call__(obs)


def _run_one(checkpoint: str, device: str, seed: int, seat: int,
             shops: list[str]) -> dict[str, Any]:
    import kaggle_environments

    downstream = make_agent(
        checkpoint=checkpoint,
        device=device,
        seat=seat,
        config=_stage4_config(),
    )
    wrapper = CounterfactualPrefixAgent(
        downstream,
        seat,
        checkpoint=checkpoint,
        device=device,
        shops=shops,
    )

    def opponent(obs, config):  # noqa: ARG001
        return pass_action(obs)

    agents: list[Any] = [opponent, opponent]
    agents[seat] = adapt_one_arg(wrapper)
    env = kaggle_environments.make(ENGINE_ENV_ID, configuration={"seed": seed})
    env.reset()
    env.run(agents)

    if wrapper.handoff is None or wrapper.counterfactuals is None:
        raise RuntimeError("missing d3h0 handoff/counterfactual result")

    actual_requested = _requested_day(downstream.diagnostics_json(), HANDOFF_DAY)
    return {
        "seed": seed,
        "seat": seat,
        "actual_shops": list(wrapper.handoff.get("town") or []),
        "handoff_money": wrapper.handoff.get("money"),
        "handoff_crops": dict(wrapper.handoff.get("crops") or {}),
        "actual_crop_targets": dict(actual_requested["crop_targets"]),
        "counterfactuals": wrapper.counterfactuals,
    }


def _print(rows: list[dict[str, Any]], shops: list[str]) -> None:
    labels = " | ".join(f"{shop[:8]:>8s}" for shop in shops)
    header = f"Seed Seat ActualShop      H_Carrot | Actual | {labels}"
    print(header)
    print("-" * len(header))
    for row in rows:
        actual_shop = ",".join(row["actual_shops"]) or "-"
        h_carrot = int(row["handoff_crops"].get("CARROT", 0))
        actual = int(row["actual_crop_targets"]["CARROT"])
        cf_by_shop = {
            item["shop"]: int(item["crop_targets"]["CARROT"])
            for item in row["counterfactuals"]
        }
        values = " | ".join(f"{cf_by_shop[s]:8d}" for s in shops)
        print(
            f"{row['seed']:4d} {row['seat']:4d} {actual_shop[:15]:15s} "
            f"{h_carrot:8d} | {actual:6d} | {values}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", default="7,2026")
    parser.add_argument("--seats", default="0,1")
    parser.add_argument(
        "--shops",
        default=",".join(DEFAULT_SHOPS),
        help="comma-separated unlocked-shop counterfactuals",
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    if not os.path.isfile(args.checkpoint):
        parser.error(f"checkpoint not found: {args.checkpoint}")
    seeds = _csv_ints(args.seeds, name="seeds")
    seats = _csv_ints(args.seats, name="seats")
    shops = [part.strip().upper() for part in args.shops.split(",") if part.strip()]
    if any(seat not in (0, 1) for seat in seats):
        parser.error("seats must contain only 0 and/or 1")
    if not shops:
        parser.error("shops must not be empty")

    engine = detect_engine()
    if not engine.get("available"):
        raise RuntimeError(f"official engine unavailable: {engine.get('reason')}")
    if engine.get("version") != "1.32.7":
        raise RuntimeError(
            f"expected kaggle-environments 1.32.7, got {engine.get('version')}"
        )
    try:
        provenance = verify_official_provenance()
    except ProvenanceError as exc:
        raise RuntimeError(f"official provenance mismatch: {exc}") from exc

    rows = [
        _run_one(args.checkpoint, args.device, seed, seat, shops)
        for seat in seats
        for seed in seeds
    ]
    _print(rows, shops)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "experiment": "carrot_d3_shop_counterfactual",
                    "engine": engine,
                    "provenance": provenance,
                    "shops": shops,
                    "rows": rows,
                },
                fh,
                indent=2,
                sort_keys=True,
            )
            fh.write("\n")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

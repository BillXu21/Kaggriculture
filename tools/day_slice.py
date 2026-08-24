"""Isolated ONE-DAY replay-slice harness.

Replays a recorded prefix up to a day boundary in the exact-parity fast
engine, verifies the boundary observation against the recorded replay, then
executes exactly one day (24 turns) with a pluggable agent on one seat while
the opponent replays recorded actions. Emits a JSON-safe ``SliceResult``.

Sibling-module contract (imported lazily so this harness stays importable
while those modules are being delivered):
- ``tools.replay_io``: load_replay / episode_configuration / episode_id
- ``tools.expert_plan``: extract_daily_plan / boundary_observation
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from dataclasses import dataclass, field, fields
from typing import Any, Callable, Mapping, Sequence

__all__ = [
    "SliceResult",
    "normalize_obs",
    "first_diff",
    "run_day_slice",
    "make_expert_executor_agent",
    "run_slices",
    "summarize",
]

# Structure tile kinds that can hold an animal; empty = present without one.
STRUCTURE_KINDS = ("COOP", "PASTURE")

_FLOAT_TOL = 1e-6

AgentFactory = Callable[[int], Callable[[Mapping], Mapping]]


# --------------------------------------------------------------------- obs


def _is_zero(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) \
        and value == 0


def _drop_zero_values(node: Any) -> Any:
    """Recursively drop dict entries whose value is 0/0.0 (bools preserved)."""
    if isinstance(node, Mapping):
        stripped = {key: _drop_zero_values(val) for key, val in node.items()}
        return {key: val for key, val in stripped.items() if not _is_zero(val)}
    if isinstance(node, list):
        return [_drop_zero_values(item) for item in node]
    return node


def normalize_obs(obs: Mapping[str, Any]) -> dict[str, Any]:
    """Deep-copied normalization for boundary comparison.

    - removes top-level ``remainingOverageTime`` and ``step``;
    - within every farm's tiles, removes ``age`` and ``placed_day`` from every
      dict tile (cosmetic fast-engine vs replay differences);
    - drops zero-valued (0/0.0) dict entries recursively so missing-vs-present
      zero entries compare equal;
    - everything else is preserved verbatim.
    """
    out = copy.deepcopy(dict(obs))
    out.pop("remainingOverageTime", None)
    out.pop("step", None)
    farms = out.get("farms")
    if isinstance(farms, list):
        for farm in farms:
            if not isinstance(farm, Mapping):
                continue
            tiles = farm.get("tiles")
            if not isinstance(tiles, list):
                continue
            for row in tiles:
                if not isinstance(row, list):
                    continue
                for tile in row:
                    if isinstance(tile, Mapping):
                        tile.pop("age", None)
                        tile.pop("placed_day", None)
    result = _drop_zero_values(out)
    return result if isinstance(result, dict) else {}


def first_diff(a: Any, b: Any, path: str = "") -> str | None:
    """Deterministic textual first difference between normalized observations.

    Floats (and mixed int/float numerics) compare with tolerance 1e-6.
    Returns None when equal.
    """
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        for key in sorted(set(a) | set(b)):
            if key not in a:
                return f"{path}.{key}: missing in a (b={b[key]!r})"
            if key not in b:
                return f"{path}.{key}: missing in b (a={a[key]!r})"
            diff = first_diff(a[key], b[key], f"{path}.{key}")
            if diff is not None:
                return diff
        return None
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return f"{path}: length {len(a)} != {len(b)}"
        for index, (x, y) in enumerate(zip(a, b)):
            diff = first_diff(x, y, f"{path}[{index}]")
            if diff is not None:
                return diff
        return None
    both_numeric = (
        isinstance(a, (int, float)) and not isinstance(a, bool)
        and isinstance(b, (int, float)) and not isinstance(b, bool))
    if both_numeric:
        if abs(float(a) - float(b)) <= _FLOAT_TOL:
            return None
        return f"{path}: {a!r} != {b!r}"
    if type(a) is not type(b):
        return (f"{path}: type {type(a).__name__} != {type(b).__name__} "
                f"({a!r} vs {b!r})")
    if a != b:
        return f"{path}: {a!r} != {b!r}"
    return None


# ----------------------------------------------------------------- metrics


def _count_tiles(farm: Mapping[str, Any]) -> tuple[dict[str, int], dict[str, int], int, int]:
    """(crops by name, animals by species, weed count, empty structure count)."""
    crops: dict[str, int] = {}
    animals: dict[str, int] = {}
    weeds = 0
    empty_structures = 0
    for row in farm.get("tiles", []):
        if not isinstance(row, list):
            continue
        for tile in row:
            if not isinstance(tile, Mapping):
                continue
            kind = tile.get("kind")
            if kind == "PLANT":
                crop = str(tile.get("crop"))
                crops[crop] = crops.get(crop, 0) + 1
            elif "animal" in tile:
                species = str(tile["animal"])
                animals[species] = animals.get(species, 0) + 1
            elif kind == "WEED":
                weeds += 1
            elif kind in STRUCTURE_KINDS:
                empty_structures += 1
    return crops, animals, weeds, empty_structures


_ANIMAL_COSTS = {"GOOSE": 300, "COW": 400, "SHEEP": 500}
_CROP_SEED_COSTS = {"WHEAT": 10, "CARROT": 20, "TOMATO": 50,
                    "STRAWBERRY": 100, "MELON": 80}


def _wealth(obs_seat: Mapping[str, Any], seat: int) -> float:
    """Cash + inventories at observed market prices + assets at cost.

    One-day cash deltas hide same-day investment (seeds bought, animals
    placed, weeds reclaimed); pairing on identical start states makes this
    deterministic wealth the fairer primary comparison (issue #7).
    """
    farm = obs_seat["farms"][seat]
    private = obs_seat.get("private") or {}
    prices = (obs_seat.get("market") or {}).get("prices") or {}
    total = float(farm["money"])
    shed = private.get("shed") or {}
    for product, qty in shed.items():
        total += int(qty) * float(prices.get(product, 0))
    for crop, qty in (private.get("seeds") or {}).items():
        total += int(qty) * _CROP_SEED_COSTS.get(crop, 0)
    for inv in private.get("inventories") or []:
        for item, qty in inv.items():
            total += int(qty) * float(prices.get(item, 0))
    crops, animals, _, _ = _count_tiles(farm)
    for crop, count in crops.items():
        total += count * _CROP_SEED_COSTS.get(crop, 0)
    for species, count in animals.items():
        total += count * _ANIMAL_COSTS.get(species, 0)
    return total


def _farm_metrics(obs_seat: Mapping[str, Any], seat: int) -> dict[str, Any]:
    farm = obs_seat["farms"][seat]
    crops, animals, weeds, empty_structures = _count_tiles(farm)
    return {
        "cash": float(farm["money"]),
        "crops": crops,
        "animals": animals,
        "weeds": weeds,
        "empty_structures": empty_structures,
        "unlocked": len(farm.get("unlocked_quadrants", [])),
        "hands": len(farm.get("hands", [])),
        "hires_today": int(farm.get("hires_today", 0)),
        "wealth": _wealth(obs_seat, seat),
    }


def _accumulate_action_families(
    action: Mapping[str, Any],
    foreman_families: dict[str, int],
    market_families: dict[str, int],
) -> None:
    unit_entries: list[Any] = [action.get("farmer")]
    unit_entries.extend(action.get("hands") or [])
    for entry in unit_entries:
        if isinstance(entry, (list, tuple)) and entry:
            name = str(entry[0])
            foreman_families[name] = foreman_families.get(name, 0) + 1
    for order in action.get("market") or []:
        if isinstance(order, (list, tuple)) and order:
            name = str(order[0])
            market_families[name] = market_families.get(name, 0) + 1


# -------------------------------------------------------------- result


@dataclass(frozen=True)
class SliceResult:
    """JSON-safe outcome of one isolated day slice."""

    episode_id: int = 0
    seed: int = 0
    seat: int = 0
    day: int = 0
    boundary_verified: bool = False
    boundary_first_diff: str | None = None
    cash_start: float = 0.0
    cash_end: float = 0.0
    wealth_start: float = 0.0
    wealth_end: float = 0.0
    crops_start: dict[str, int] = field(default_factory=dict)
    crops_end: dict[str, int] = field(default_factory=dict)
    animals_start: dict[str, int] = field(default_factory=dict)
    animals_end: dict[str, int] = field(default_factory=dict)
    weeds_start: int = 0
    weeds_end: int = 0
    empty_structures_start: int = 0
    empty_structures_end: int = 0
    unlocked_start: int = 0
    unlocked_end: int = 0
    hands_end: int = 0
    hires_today_max: int = 0
    foreman_action_families: dict[str, int] = field(default_factory=dict)
    market_op_families: dict[str, int] = field(default_factory=dict)
    agent_diagnostics: dict[str, Any] = field(default_factory=dict)
    turns_executed: int = 0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


# ---------------------------------------------------------------- runner


def run_day_slice(
    replay_path_or_replay: Any,
    day: int,
    seat: int,
    agent_factory: AgentFactory,
    *,
    verify_boundary: bool = True,
    max_turns: int | None = None,
    strict: bool = False,
) -> SliceResult:
    """Run one isolated day slice; never raises unless ``strict=True``."""
    try:
        return _run_day_slice_impl(
            replay_path_or_replay, day, seat, agent_factory,
            verify_boundary=verify_boundary, max_turns=max_turns)
    except Exception as exc:  # noqa: BLE001 - API contract: report, don't raise
        if strict:
            raise
        return SliceResult(seat=seat, day=day,
                           error=f"{type(exc).__name__}: {exc}")


def _run_day_slice_impl(
    replay_path_or_replay: Any,
    day: int,
    seat: int,
    agent_factory: AgentFactory,
    *,
    verify_boundary: bool,
    max_turns: int | None,
) -> SliceResult:
    from fast_env.api import FastKaggricultureEnv
    from tools.replay_io import episode_configuration, episode_id

    replay: Any = replay_path_or_replay
    if isinstance(replay, (str, os.PathLike)):
        from tools.replay_io import load_replay
        replay = load_replay(replay)

    cfg = episode_configuration(replay)
    env = FastKaggricultureEnv(cfg)
    obs = env.reset()
    steps = replay["steps"]

    # Prefix: recorded actions transform steps[t-1] into steps[t]; after the
    # loop the engine state matches the start-of-day boundary steps[day*24].
    boundary_step = day * 24
    for t in range(1, boundary_step + 1):
        obs, _, _ = env.step([steps[t][0]["action"], steps[t][1]["action"]])

    boundary_verified = False
    boundary_first_diff: str | None = None
    if verify_boundary:
        recorded = steps[boundary_step][seat]["observation"]
        boundary_first_diff = first_diff(
            normalize_obs(obs[seat]), normalize_obs(recorded))
        boundary_verified = boundary_first_diff is None

    from tools.expert_plan import extract_daily_plan
    plan = extract_daily_plan(replay, seat, day)
    assert plan is not None  # availability contract; the agent uses its own provider

    agent = agent_factory(seat)
    opp = 1 - seat
    total_turns = 24 if max_turns is None else int(max_turns)

    start = _farm_metrics(obs[seat], seat)
    hires_today_max = start["hires_today"]
    foreman_families: dict[str, int] = {}
    market_families: dict[str, int] = {}
    turns_executed = 0
    statuses: Sequence[str] = ("ACTIVE", "ACTIVE")

    for h in range(total_turns):
        hires_today_max = max(
            hires_today_max, _farm_metrics(obs[seat], seat)["hires_today"])
        action = agent(obs[seat])
        _accumulate_action_families(action, foreman_families, market_families)
        pair: list[Any] = [None, None]
        pair[seat] = action
        pair[opp] = steps[boundary_step + h + 1][opp]["action"]
        obs, _, statuses = env.step(pair)
        turns_executed += 1
        if statuses[0] == "DONE" and statuses[1] == "DONE":
            break

    end = _farm_metrics(obs[seat], seat)
    hires_today_max = max(hires_today_max, end["hires_today"])

    diagnostics: dict[str, Any] = {}
    diagnostics_fn = getattr(agent, "diagnostics_json", None)
    if callable(diagnostics_fn):
        diagnostics = diagnostics_fn()

    return SliceResult(
        episode_id=int(episode_id(replay)),
        seed=int(cfg.get("seed", 0)),
        seat=seat,
        day=day,
        boundary_verified=boundary_verified,
        boundary_first_diff=boundary_first_diff,
        cash_start=start["cash"],
        cash_end=end["cash"],
        wealth_start=start["wealth"],
        wealth_end=end["wealth"],
        crops_start=start["crops"],
        crops_end=end["crops"],
        animals_start=start["animals"],
        animals_end=end["animals"],
        weeds_start=start["weeds"],
        weeds_end=end["weeds"],
        empty_structures_start=start["empty_structures"],
        empty_structures_end=end["empty_structures"],
        unlocked_start=start["unlocked"],
        unlocked_end=end["unlocked"],
        hands_end=end["hands"],
        hires_today_max=hires_today_max,
        foreman_action_families=foreman_families,
        market_op_families=market_families,
        agent_diagnostics=diagnostics,
        turns_executed=turns_executed,
        error=None,
    )


# ------------------------------------------------------------ factories


def make_expert_executor_agent(replay: Mapping[str, Any], day: int, *,
                               checkpoint: str | None = None) -> AgentFactory:
    """Build an agent_factory over seats using the expert's daily plan.

    Each constructed agent is ``executor_v0.agent.make_agent`` with a
    ``FixedPlanProvider(extract_daily_plan(replay, seat, day))`` and default
    config. ``executor_v0`` is imported lazily inside the closure factory.
    """
    if checkpoint is not None:
        raise ValueError(
            "make_expert_executor_agent only supports the FixedPlanProvider "
            "path; checkpoint is reserved and must stay None")

    def _factory(seat: int) -> Callable[[Mapping], Mapping]:
        from executor_v0.agent import make_agent
        from executor_v0.manager import FixedPlanProvider
        from tools.expert_plan import extract_daily_plan

        plan = extract_daily_plan(replay, seat, day)
        return make_agent(provider=FixedPlanProvider(plan), seat=seat)

    return _factory


def run_slices(specs: Sequence[tuple[Any, int, int]],
               agent_factory: AgentFactory) -> list[SliceResult]:
    """Run (replay_path_or_replay, day, seat) specs sequentially."""
    return [run_day_slice(replay, day, seat, agent_factory)
            for replay, day, seat in specs]


def summarize(results: Sequence[SliceResult]) -> dict[str, Any]:
    """Compact aggregate over slice results."""
    count = len(results)
    if count == 0:
        return {"slices": 0, "cash_delta_mean": 0.0, "weeds_end_mean": 0.0,
                "boundary_verified_count": 0, "total_turns": 0,
                "error_count": 0, "per_slice": []}
    deltas = [r.cash_end - r.cash_start for r in results]
    return {
        "slices": count,
        "cash_delta_mean": sum(deltas) / count,
        "weeds_end_mean": sum(r.weeds_end for r in results) / count,
        "boundary_verified_count": sum(1 for r in results if r.boundary_verified),
        "total_turns": sum(r.turns_executed for r in results),
        "error_count": sum(1 for r in results if r.error),
        "per_slice": [
            {"episode_id": r.episode_id, "seat": r.seat, "day": r.day,
             "cash_delta": r.cash_end - r.cash_start}
            for r in results
        ],
    }


# ------------------------------------------------------------------ CLI


def main(argv: Sequence[str] | None = None) -> int:
    # Script-mode execution (`python tools/day_slice.py`) puts tools/ on
    # sys.path instead of the repo root; restore the root so the lazy
    # `tools.*` and repo-package imports resolve.
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    parser = argparse.ArgumentParser(
        description="Run one isolated replay day slice with the expert "
                    "executor-v0 agent.")
    parser.add_argument("--replay", required=True, help="Path to replay JSON")
    parser.add_argument("--day", required=True, type=int)
    parser.add_argument("--seat", required=True, type=int, choices=(0, 1))
    args = parser.parse_args(argv)

    from tools.replay_io import load_replay

    replay = load_replay(args.replay)
    factory = make_expert_executor_agent(replay, args.day)
    result = run_day_slice(replay, args.day, args.seat, factory)
    print(json.dumps(result.to_dict()))
    return 0 if result.error is None else 1


if __name__ == "__main__":
    raise SystemExit(main())

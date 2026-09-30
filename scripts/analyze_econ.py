"""Offline analysis of a saved economic diagnostics document.

Reads the JSON written by ``scripts/econ_diagnostics.py`` and quantifies where
interactions are lost: feasible-but-not-completed interaction turns, missed
interactions, supply/global shortages, and deadline-infeasible routes. No game
is executed.
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1
            else r"artifacts\overnight\econ_41003.json")
d = json.loads(path.read_text(encoding="utf-8"))
print(f"document: {path.name}  seed={d['seed']}  banks={d['final_banks']}")

for seat, payload in sorted(d["seats"].items()):
    days = payload["days"]
    if not days:
        continue
    print(f"\n{'=' * 74}\n=== seat {seat}: {len(days)} manager days "
          f"===\n{'=' * 74}")

    cost_rows: list[dict] = []
    for key in sorted(days, key=lambda k: int(k)):
        day = days[key]
        for route_id, cost in (day.get("canonical_route_costs") or {}).items():
            row = dict(cost)
            row["_day"] = int(key)
            row["_route"] = route_id
            cost_rows.append(row)

    if not cost_rows:
        print("  no canonical_route_costs")
        continue
    print(f"  canonical route costs: {len(cost_rows):,}")

    def total(field: str) -> int:
        return sum(int(r.get(field) or 0) for r in cost_rows)

    feasible = total("feasible_effective_interaction_turns")
    completed = total("effective_interactions_completed_before_deadline")
    missed = total("effective_interactions_missed")
    represented = total("represented_interaction_turns")
    eff_turns = total("effective_interaction_turns")
    print(f"\n  represented interaction turns      : {represented:,}")
    print(f"  effective interaction turns        : {eff_turns:,}")
    print(f"  feasible effective interaction turns: {feasible:,}")
    print(f"  COMPLETED before deadline          : {completed:,}")
    print(f"  MISSED                            : {missed:,}")
    if feasible:
        print(f"  --> feasible-but-not-completed gap : "
              f"{feasible - completed:,} "
              f"({100 * (feasible - completed) / feasible:.1f}% of feasible)")
    if eff_turns:
        print(f"  effective completion rate          : "
              f"{100 * completed / eff_turns:.1f}%")

    # feasibility / shortage flags
    flags = Counter()
    for r in cost_rows:
        flags["resource_feasible" if r.get("resource_feasible") else
              "resource_INFEASIBLE"] += 1
        flags["route_complete" if r.get("route_complete_before_deadline") else
              "route_INCOMPLETE"] += 1
        flags["timing_complete" if r.get("timing_complete_before_deadline")
              else "timing_INCOMPLETE"] += 1
        if r.get("supply_shortage"):
            flags["has_supply_shortage"] += 1
        if r.get("global_shortage"):
            flags["has_global_shortage"] += 1
        stage = str(r.get("first_hire_driving_stage") or "none")
        flags[f"hire_stage_{stage}"] += 1
    print("\n  flags:")
    for key, value in sorted(flags.items()):
        print(f"    {value:>6,}  {key}")

    # where feasible work is lost, by day
    print("\n  per-day feasible-vs-completed (worst 10):")
    per_day: list[tuple[int, int, int, int, int]] = []
    for key in sorted(days, key=lambda k: int(k)):
        rows = [r for r in cost_rows if r["_day"] == int(key)]
        if not rows:
            continue
        f = sum(int(r.get("feasible_effective_interaction_turns") or 0)
                for r in rows)
        c = sum(int(r.get("effective_interactions_completed_before_deadline")
                     or 0) for r in rows)
        m = sum(int(r.get("effective_interactions_missed") or 0) for r in rows)
        n_inf = sum(1 for r in rows if not r.get("route_complete_before_deadline"))
        per_day.append((int(key), f, c, m, n_inf))
    per_day.sort(key=lambda t: -(t[1] - t[2]))
    print(f"    {'day':>5}{'feasible':>11}{'completed':>11}{'missed':>9}"
          f"{'incomplete':>11}{'gap%':>8}")
    for day, f, c, m, n_inf in per_day[:10]:
        gap = 100 * (f - c) / f if f else 0.0
        print(f"    {day:>5}{f:>11,}{c:>11,}{m:>9,}{n_inf:>11,}{gap:>7.1f}%")
    tot_f = sum(t[1] for t in per_day)
    tot_c = sum(t[2] for t in per_day)
    print(f"    {'ALL':>5}{tot_f:>11,}{tot_c:>11,}"
          f"{sum(t[3] for t in per_day):>9,}"
          f"{sum(t[4] for t in per_day):>11,}"
          f"{100 * (tot_f - tot_c) / tot_f if tot_f else 0:>7.1f}%")

    # supply shortage detail (some fields are dicts, some are item tuples)
    def items_of(value):
        if isinstance(value, dict):
            return value.items()
        if isinstance(value, (list, tuple)):
            out = []
            for entry in value:
                if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    out.append((entry[0], entry[1]))
                elif isinstance(entry, dict):
                    out.extend(entry.items())
            return out
        return ()

    shortages = Counter()
    required = Counter()
    for r in cost_rows:
        for item, qty in items_of(r.get("supply_shortage")):
            shortages[f"supply:{item}"] += int(qty or 0)
        for item, qty in items_of(r.get("global_shortage")):
            shortages[f"global:{item}"] += int(qty or 0)
        for item, qty in items_of(r.get("global_quantities_required")):
            required[f"global_required:{item}"] += int(qty or 0)
    if shortages:
        print("\n  shortage quantities (item -> total units short):")
        for key, value in shortages.most_common(12):
            print(f"    {value:>8,}  {key}")
    if required:
        print("\n  global quantities required (item -> total):")
        for key, value in required.most_common(12):
            print(f"    {value:>8,}  {key}")

    # which stages are blocked when resource-infeasible
    blocked = Counter()
    allowed = Counter()
    for r in cost_rows:
        stage = str(r.get("first_hire_driving_stage") or "none")
        if r.get("resource_feasible"):
            allowed[stage] += 1
        else:
            blocked[stage] += 1
    print("\n  routes blocked by resources, by first hire-driving stage:")
    for stage, value in blocked.most_common(10):
        print(f"    {value:>5,} blocked / {allowed[stage]:>5,} allowed  {stage}")

    # movement vs interaction split
    mv = sum(int(r.get("horizontal_sweep_turns") or 0) for r in cost_rows)
    setup = sum(int(r.get("setup_travel_turns") or 0) for r in cost_rows)
    inter = sum(int(r.get("inter_segment_travel_turns") or 0) for r in cost_rows)
    pick = sum(int(r.get("pickup_action_turns") or 0) for r in cost_rows)
    tt = sum(int(r.get("total_turns") or 0) for r in cost_rows)
    print(f"\n  turn budget: total {tt:,} = interactions {eff_turns:,} "
          f"+ setup {setup:,} + inter-seg {inter:,} + horizontal sweep {mv:,}"
          f" + pickup {pick:,}")
    if tt:
        print(f"  non-interaction share: "
              f"{100 * (tt - eff_turns) / tt:.1f}%")
    print(f"  median total_turns per route: "
          f"{statistics.median([int(r.get('total_turns') or 0) for r in cost_rows]):.0f}")

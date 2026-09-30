"""Offline analysis of market behaviour from a saved economic document.

Reads the JSON written by ``scripts/econ_diagnostics.py`` and reports what the
executor actually ordered, per day and per seat: buy/sell split, item mix,
quantities, and cash. The diagnosis so far is that ~25% of routes are dropped as
``resource_feasible=False`` with PLANT-stage routes blocked 15/15, so the
question is whether seed/animal purchase is late or too small.

No game is executed.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1
            else r"artifacts\overnight\econ_41003.json")
d = json.loads(path.read_text(encoding="utf-8"))

for seat, payload in sorted(d["seats"].items()):
    days = payload["days"]
    if not days:
        continue
    print(f"\n{'=' * 74}\n=== seat {seat} market behaviour ===\n{'=' * 74}")

    sample = next(iter(days.values()))
    market = sample.get("market_diagnostics")
    print("  market_diagnostics keys: "
          f"{sorted(market) if isinstance(market, dict) else type(market)}")
    if isinstance(market, dict):
        for key, value in market.items():
            kind = type(value).__name__
            extra = ""
            if isinstance(value, (list, tuple)):
                extra = f" len={len(value)} sample={value[:2]}"
            elif isinstance(value, dict):
                extra = f" keys={sorted(value)[:8]}"
            else:
                extra = f" = {value}"
            print(f"    {kind:<6} {key}{extra}")

    buys: Counter[str] = Counter()
    sells: Counter[str] = Counter()
    buy_qty: Counter[str] = Counter()
    per_day_buys: dict[int, Counter] = defaultdict(Counter)
    per_day_sells: dict[int, Counter] = defaultdict(Counter)
    cash_series: list[tuple[int, float]] = []
    order_kinds: Counter[str] = Counter()

    for key in sorted(days, key=lambda k: int(k)):
        day = int(key)
        doc = days[key].get("market_diagnostics") or {}
        orders = doc.get("market_orders") or ()
        for order in orders:
            if not isinstance(order, dict):
                order_kinds[str(type(order))] += 1
                continue
            kind = str(order.get("kind") or order.get("type") or "?")
            order_kinds[kind] += 1
            item = str(order.get("item") or order.get("item_id")
                       or order.get("target") or "?")
            qty = order.get("quantity") or order.get("amount") or 0
            try:
                qty = int(qty)
            except (TypeError, ValueError):
                qty = 0
            if kind.startswith("SELL") or kind == "SELL":
                sells[item] += 1
                per_day_sells[day][item] += qty
            else:
                buys[f"{kind}:{item}"] += 1
                buy_qty[f"{kind}:{item}"] += qty
                per_day_buys[day][item] += qty
        for field in ("cash", "money", "cash_after", "balance"):
            if field in doc:
                try:
                    cash_series.append((day, float(doc[field])))
                except (TypeError, ValueError):
                    pass
                break

    print(f"\n  total orders: {sum(order_kinds.values())}")
    print("  order kinds: "
          + ", ".join(f"{k}={v}" for k, v in order_kinds.most_common(12)))
    print(f"\n  BUY order kinds x items ({sum(buys.values())} orders):")
    for key, value in buys.most_common(20):
        print(f"    {value:>5,} orders  qty {buy_qty[key]:>7,}  {key}")
    print(f"\n  SELL items ({sum(sells.values())} orders):")
    for key, value in sells.most_common(12):
        qty = sum(per_day_sells[day][key] for day in per_day_sells)
        print(f"    {value:>5,} orders  qty {qty:>7,}  {key}")

    if cash_series:
        print(f"\n  cash series (day -> value):")
        print("    " + ", ".join(f"{d_}:{v:,.0f}" for d_, v in cash_series))
        print(f"    min {min(v for _, v in cash_series):,.0f}  "
              f"max {max(v for _, v in cash_series):,.0f}  "
              f"final {cash_series[-1][1]:,.0f}")

    print("\n  per-day BUY quantities (non-zero days):")
    for day in sorted(per_day_buys):
        items = ", ".join(f"{k}={v}" for k, v in per_day_buys[day].most_common())
        print(f"    day {day:>3}: {items}")
    print("\n  per-day SELL quantities (non-zero days):")
    for day in sorted(per_day_sells):
        items = ", ".join(f"{k}={v}" for k, v in per_day_sells[day].most_common())
        print(f"    day {day:>3}: {items}")

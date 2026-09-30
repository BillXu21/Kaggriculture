"""Offline analysis of a saved strip controller profile. No game execution."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1
            else r"artifacts\overnight\profile_41003_base.json")
d = json.loads(path.read_text(encoding="utf-8"))
rows = d["all_rows"]

print(f"profile: {path.name}  seed={d['seed']}  banks={d['final_banks']} "
      f"digest={str(d['trace_digest'])[:16]}")
print(f"profiled wall: {d['profiled_wall_seconds']:.1f}s   "
      f"total tottime: {d['total_profiled_tottime_seconds']:.1f}s")

total = sum(r["tottime"] for r in rows)
print(f"\n=== TOP 30 by tottime (of {total:.1f}s total) ===")
cum = 0.0
for r in rows[:30]:
    cum += r["tottime"]
    fn = r["function"].replace("\\", "/").split("/")[-1]
    print(f"  {r['tottime']:7.2f}s {100*r['tottime']/total:5.2f}% "
          f"cum={100*cum/total:5.1f}%  {r['calls']:>11,}  {fn}")

print("\n=== copy.py rows in detail ===")
for r in rows:
    if "copy.py" in r["function"].replace("\\", "/"):
        fn = r["function"].replace("\\", "/").split("/")[-1]
        us = 1e6 * r["tottime"] / max(1, r["calls"])
        print(f"  {r['tottime']:7.2f}s {r['calls']:>11,} calls "
              f"{us:>9.1f}us/call  {fn}")

print("\n=== C builtins ('~') rows: which are the huge call counts? ===")
bu = [r for r in rows if r["function"].replace("\\", "/").endswith("/~")
      or r["function"] == "~"]
bu.sort(key=lambda r: -r["calls"])
for r in bu[:14]:
    print(f"  {r['calls']:>12,} calls {r['tottime']:7.2f}s  "
          f"cum={r['cumtime']:7.2f}s  {r['function'][:70]}")

print("\n=== <string> (dataclass-generated) rows ===")
gen = [r for r in rows if r["function"].startswith("<")]
gen.sort(key=lambda r: -r["tottime"])
for r in gen[:10]:
    print(f"  {r['tottime']:7.2f}s {r['calls']:>12,}  {r['function'][:80]}")

print("\n=== cumulative callers of deepcopy (top by cumtime, copy-related) ===")
for r in sorted(rows, key=lambda r: -r["cumtime"]):
    fn = r["function"].replace("\\", "/").split("/")[-1]
    if any(k in fn for k in ("deepcopy", "copy(", "snapshot", "canonical")):
        print(f"  cum={r['cumtime']:8.2f}s tot={r['tottime']:7.2f}s "
              f"{r['calls']:>9,}  {fn}")

print("\n=== executor_v0 cumulative entry points (cumtime) ===")
for r in sorted(rows, key=lambda r: -r["cumtime"]):
    fn = r["function"].replace("\\", "/")
    if "executor_v0" in fn and r["cumtime"] > 1.0:
        print(f"  cum={r['cumtime']:8.2f}s tot={r['tottime']:7.2f}s "
              f"{r['calls']:>10,}  {fn.split('executor_v0/')[-1][:75]}")

print("\n=== AGGREGATE by owner package ===")
agg: dict[str, float] = defaultdict(float)
aggn: dict[str, int] = defaultdict(int)
for r in rows:
    fn = r["function"].replace("\\", "/")
    if "executor_v0/" in fn:
        owner = "executor_v0"
    elif "/site-packages/jax" in fn or fn.startswith("jax"):
        owner = "jax"
    elif "rl_manager/" in fn:
        owner = "rl_manager"
    elif "fast_env/" in fn:
        owner = "fast_env"
    elif "/Lib/copy.py" in fn:
        owner = "stdlib:copy"
    elif fn.startswith("<") or fn.endswith("/~"):
        owner = "<builtin/generated>"
    else:
        owner = "other:" + fn.split("/")[-1]
    agg[owner] += r["tottime"]
    aggn[owner] += r["calls"]
for owner, v in sorted(agg.items(), key=lambda kv: -kv[1]):
    print(f"  {v:7.2f}s {100*v/total:5.2f}%  {aggn[owner]:>12,} calls  {owner}")

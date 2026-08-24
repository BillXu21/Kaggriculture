"""Run a fixed one-day slice suite and write compact JSON results (issue #7).

Scenario set: failure specimen + three elite replays across seats/days.
Baseline and candidate runs must use the identical scenario list so paired
comparison stays exact.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.day_slice import make_expert_executor_agent, run_day_slice, summarize
from tools.replay_io import load_replay

PRIMARY_SAMPLES = Path(
    r"C:\Users\liuyi\VSCodeProjecs\Kaggriculture\Kaggriculture\data\samples")

SPECIMEN = "datasamplesbc_e_mirror_98178196.json"

SCENARIOS = [
    (SPECIMEN, 1, 0), (SPECIMEN, 1, 1),
    (SPECIMEN, 3, 0), (SPECIMEN, 3, 1),
    (SPECIMEN, 5, 0), (SPECIMEN, 5, 1),
    ("98093786.json", 3, 0), ("98093786.json", 3, 1),
    ("98093786.json", 7, 0), ("98093786.json", 7, 1),
    ("97879422.json", 3, 0), ("97879422.json", 8, 1),
    ("98184881.json", 3, 0), ("98184881.json", 6, 1),
]

# Broader accumulated-candidate set: every elite replay contributes two
# mid-game days on alternating seats (deterministic, no cherry-picking).
_ELITE = ["97879422.json", "97927291.json", "97968292.json", "98004787.json",
          "98045895.json", "98089225.json", "98093786.json", "98134768.json",
          "98137050.json", "98139342.json", "98139344.json", "98141707.json",
          "98141712.json", "98184881.json", "98185000.json", "98189541.json",
          "98189542.json", "98198569.json"]
EXPANDED_SCENARIOS = [(SPECIMEN, 2, 0), (SPECIMEN, 4, 1)] + [
    (_ELITE[i % len(_ELITE)], (3, 6, 9)[(i // len(_ELITE)) % 3],
     (i // len(_ELITE)) % 2)
    for i in range(2 * len(_ELITE))
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--label", default="run")
    parser.add_argument("--set", choices=("smoke", "expanded"),
                        default="smoke")
    args = parser.parse_args()

    scenarios = SCENARIOS if args.set == "smoke" else EXPANDED_SCENARIOS
    for name, _, _ in scenarios:
        path = PRIMARY_SAMPLES / name
        if not path.exists():
            raise SystemExit(f"missing replay sample: {path}")

    started = time.perf_counter()
    factories: dict[tuple[str, int], object] = {}
    results = []
    for name, day, seat in scenarios:
        key = (name, day)
        if key not in factories:
            factories[key] = make_expert_executor_agent(
                load_replay(str(PRIMARY_SAMPLES / name)), day)
        results.append(run_day_slice(str(PRIMARY_SAMPLES / name), day, seat,
                                     factories[key]))
    payload = {
        "schema_version": 1,
        "label": args.label,
        "elapsed_s": round(time.perf_counter() - started, 1),
        "summary": summarize(results),
        "slices": [r.to_dict() for r in results],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"wrote {out} ({len(results)} slices) in {payload['elapsed_s']}s")


if __name__ == "__main__":
    main()

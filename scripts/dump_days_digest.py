"""Dump a stable hash of the retained per-day executor diagnostics.

PARITY TOOL. Runs one fixed-seed full game and emits a deterministic digest of
``Stage25StripExecutorAgent.diagnostics_json()["days"]`` for both seats, so an
optimization of the diagnostics-retention path can be proven byte-identical
rather than merely action-identical.

Only ``days`` is hashed: the rest of the document includes provider telemetry
that is not part of the retention contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts import parallel_full_game_validation as harness  # noqa: E402

DEFAULT_CHECKPOINT = Path(
    r"C:\Users\liuyi\VSCodeProjecs\Kaggriculture\stage25_bc_7m_best_inference.npz")


def stable_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=repr).encode("utf-8")
    ).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--seeds", type=int, nargs="+", default=[41003])
    parser.add_argument("--opening", default=harness.OPENING_NAME)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    checkpoint = harness._resolve_checkpoint_path(args.checkpoint)
    checkpoint_sha = harness._sha256_file(checkpoint)

    import rl_manager.executor_factory as factory_module

    captured: dict[int, dict] = {}
    original_init = factory_module.Stage25StripExecutorAgent.__init__

    def patched_init(self, **kwargs):
        original_init(self, **kwargs)
        captured.setdefault(id(self), self)

    factory_module.Stage25StripExecutorAgent.__init__ = patched_init
    try:
        for seed in args.seeds:
            record = harness._run_full_game_task(
                (0, seed, 0), str(checkpoint), checkpoint_sha,
                enable_row_claim_board=False,
                opponent=harness.OPPONENT_SYMETRIC,
                opening_name=args.opening,
            )
            agents = sorted(captured.values(), key=lambda a: a.seat)
            print(f"seed {seed}: banks={record['final_banks']} "
                  f"action_digest={str(record.get('trace_digest'))[:16]}")
            result: dict[str, object] = {
                "seed": seed,
                "final_banks": [int(v) for v in record["final_banks"]],
                "trace_digest": record.get("trace_digest"),
                "turns": record.get("turns"),
                "seats": {},
            }
            for agent in agents:
                document = agent.diagnostics_json()
                days = document.get("days", {})
                result["seats"][str(agent.seat)] = {
                    "day_keys": sorted(days),
                    "day_count": len(days),
                    "days_sha256": stable_digest(days),
                    "schema_version": document.get("schema_version"),
                }
                print(f"  seat {agent.seat}: {len(days)} day keys, "
                      f"days sha256={stable_digest(days)[:32]}")
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(
                    json.dumps(result, indent=2, sort_keys=True),
                    encoding="utf-8")
    finally:
        factory_module.Stage25StripExecutorAgent.__init__ = original_init
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

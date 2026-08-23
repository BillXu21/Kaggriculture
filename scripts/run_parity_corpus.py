"""Full-episode same-action parity corpus runner (official 1.32.7 vs fast).

Runs complete default-configuration 720-step episodes (reset observation plus
719 accepted primitive ``step`` transitions; terminal DONE at canonical step
719, day 29 hour 23) for a fixed seed list. Each episode is driven by
``oracle.action_generator.LegalishActionGenerator`` with the seed as the
generator RNG seed: ONE pre-transition-chosen action pair goes to BOTH
engines every turn, and the run stops at the first divergent field with the
standard :class:`oracle.DivergenceReport`.

Usage (repo root, oracle venv):

    python scripts/run_parity_corpus.py                       # 8 fixed seeds
    python scripts/run_parity_corpus.py --seeds 0 --output out.json

The JSON report records per-seed turn counts, terminal outcomes, day
boundaries, the action-family coverage histogram, wall time, and — on
divergence — the exact DivergenceReport plus the ``(generator_seed,
turn_index)`` needed to reproduce it deterministically.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from oracle import DivergenceError, run_same_action_replay  # noqa: E402
from oracle.action_generator import LegalishActionGenerator  # noqa: E402
from oracle.backend import FastBackendAdapter  # noqa: E402
from oracle.official_backend import OfficialAnomalyError  # noqa: E402

SCHEMA_VERSION = 1
DEFAULT_SEEDS = (0, 1, 2, 7, 17, 42, 123, 999)
EXPECTED_PRIMITIVE_STEPS = 719  # reset observation + 719 accepted step calls
EXPECTED_DAY_TRANSITIONS = 29  # day 0 -> day 29


class DayCountingFastBackend(FastBackendAdapter):
    """Fast adapter that counts day-boundary transitions across steps."""

    def __init__(self, configuration: Any = None) -> None:
        super().__init__(configuration)
        self.day_transitions = 0
        self._last_day: int | None = None

    def step(self, actions):  # type: ignore[override]
        observations, rewards, statuses = super().step(actions)
        day = int(observations[0]["day"])
        if self._last_day is not None and day != self._last_day:
            self.day_transitions += 1
        self._last_day = day
        return observations, rewards, statuses


def run_seed(seed: int, max_turns: int) -> dict[str, Any]:
    configuration = {"seed": seed}
    generator = LegalishActionGenerator(seed)
    fast_backend = DayCountingFastBackend(configuration)
    started = time.perf_counter()
    error: dict[str, Any] | None = None
    result = None
    try:
        result = run_same_action_replay(
            configuration,
            generator.next_pair,
            max_turns=max_turns,
            fast_backend=fast_backend,
        )
    except DivergenceError as exc:
        error = {
            "kind": "DivergenceError",
            "report": exc.report.to_dict(),
            "repro": {
                "generator_seed": seed,
                "turn_index": exc.report.turn_index,
                "note": f"rerun scripts/run_parity_corpus.py --seeds {seed}; "
                "the divergence reproduces deterministically at this turn_index",
            },
        }
    except OfficialAnomalyError as exc:
        error = {"kind": "OfficialAnomalyError", "message": str(exc)}
    finally:
        wall_seconds = time.perf_counter() - started
    payload: dict[str, Any] = {
        "seed": seed,
        "wall_seconds": round(wall_seconds, 2),
        "coverage": generator.coverage,
        "day_transitions": fast_backend.day_transitions,
    }
    if error is not None:
        payload["error"] = error
        return payload
    payload.update({
        "turns_executed": result.turns_executed,
        "final_step": result.final_step,
        "official_statuses": result.official_statuses,
        "fast_statuses": result.fast_statuses,
        "official_rewards": result.official_rewards,
        "fast_rewards": result.fast_rewards,
    })
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
        help="comma-separated fixed generator/episode seeds",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=720,
        help="primitive step-call budget per episode (default 720)",
    )
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "research" / "parity_corpus_report.json"),
        help="JSON report output path",
    )
    args = parser.parse_args()

    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    results: list[dict[str, Any]] = []
    failures = 0
    for seed in seeds:
        print(f"[corpus] seed {seed}: running full episode ...", flush=True)
        payload = run_seed(seed, args.max_turns)
        if "error" in payload:
            failures += 1
            print(
                f"[corpus] seed {seed}: DIVERGENCE/ANOMALY "
                f"{json.dumps(payload['error'], sort_keys=True)[:600]}",
                flush=True,
            )
            results.append(payload)
            continue
        ok_terminal = (
            payload["turns_executed"] >= EXPECTED_PRIMITIVE_STEPS - 1
            and payload["final_step"] == EXPECTED_PRIMITIVE_STEPS
            and payload["official_statuses"] == ["DONE", "DONE"]
            and payload["fast_statuses"] == ["DONE", "DONE"]
            and payload["official_rewards"] == payload["fast_rewards"]
        )
        status = "OK" if ok_terminal else "UNEXPECTED-Terminal"
        print(
            f"[corpus] seed {seed}: {status} turns={payload['turns_executed']} "
            f"final_step={payload['final_step']} "
            f"rewards={payload['official_rewards']} "
            f"days+={payload['day_transitions']} "
            f"families={len(payload['coverage'])} "
            f"wall={payload['wall_seconds']}s",
            flush=True,
        )
        results.append(payload)

    coverage_union: dict[str, int] = {}
    for payload in results:
        for family, count in payload["coverage"].items():
            coverage_union[family] = coverage_union.get(family, 0) + count

    report = {
        "schema_version": SCHEMA_VERSION,
        "engine_identity": "kaggle-environments 1.32.7 @ 28b6d8af3ce73926b3d0fda1410c1ddd8384ab8c",
        "episode_contract": (
            "default configuration, 720-step episodes = reset observation + "
            f"{EXPECTED_PRIMITIVE_STEPS} accepted primitive step calls; terminal "
            "DONE at canonical step 719 (day 29 hour 23)"
        ),
        "seeds": seeds,
        "episodes_completed": len(results),
        "episodes_failed": failures,
        "zero_divergence": failures == 0 and len(results) == len(seeds),
        "expected_primitive_steps": EXPECTED_PRIMITIVE_STEPS,
        "expected_day_transitions": EXPECTED_DAY_TRANSITIONS,
        "coverage_union": dict(sorted(coverage_union.items())),
        "results": results,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[corpus] report written to {output_path}")
    print(
        f"[corpus] zero_divergence={report['zero_divergence']} "
        f"completed={report['episodes_completed']}/{len(seeds)} "
        f"families={len(coverage_union)}"
    )
    return 0 if report["zero_divergence"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Bounded Stage 2.5 Packet 4 native-provider/executor smoke.

This is plumbing evidence only. It creates a tiny native checkpoint, runs two
profiled Stage 2.5 executors against the real local engine, and crosses two
manager-day boundaries. It does not train or measure competitive strength.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

from executor_v0.agent import ExecutorAgent
from oracle.backend import canonical_observations, make_backend
from rl_manager.executor_factory import make_stage25_executor_factory
from rl_manager.stage25_checkpoint import save_stage25_inference_checkpoint
from rl_manager.stage25_policy import init_stage25_params, tiny_stage25_config
from rl_manager.stage25_provider import Stage25PlanProvider


def run_smoke(*, engine: str, seed: int, days: int, mode: str) -> dict:
    if days < 2:
        raise ValueError("days must be at least 2 to cross a manager boundary")
    config = tiny_stage25_config()
    params = init_stage25_params(config, seed=seed)
    with tempfile.TemporaryDirectory(prefix="stage25-packet4-") as directory:
        checkpoint = Path(directory) / "tiny-stage25.npz"
        save_stage25_inference_checkpoint(
            checkpoint, params, config, seed=seed,
            executor={"profile": "stage25_executor_v1"},
        )
        factory = make_stage25_executor_factory()
        providers = [
            Stage25PlanProvider(
                episode_id="packet4-smoke", seat=seat,
                manager_start_day=0, native_checkpoint=checkpoint,
                mode=mode, seed=seed,
            )
            for seat in (0, 1)
        ]
        agents: list[ExecutorAgent] = [
            factory.create(
                backend_name=engine, seat=seat,
                configuration={"seed": seed}, provider=providers[seat],
            )
            for seat in (0, 1)
        ]
        backend = make_backend(engine, {"seed": seed})
        # Route every observation through the same public canonicalization seam
        # the runner uses: canonical farms (no fast-engine `age` alias) and a
        # resolved absolute step for both seats.
        def seat_animals(views: list[dict]) -> int:
            return sum(
                1
                for seat in (0, 1)
                for row in views[seat]["farms"][seat]["tiles"]
                for tile in row
                if isinstance(tile, dict) and "animal" in tile
            )

        observations = canonical_observations(backend, backend.reset())
        boundaries: list[int] = []
        max_steps = days * 24
        steps_run = 0
        buy_animal_orders = 0
        max_seat_animals = seat_animals(observations)
        for _ in range(max_steps):
            if int(observations[0]["hour"]) == 0:
                boundaries.append(int(observations[0]["day"]))
            actions = [
                agents[seat](observations[seat]) for seat in (0, 1)
            ]
            for action in actions:
                for order in (action.get("market") or []):
                    if order and order[0] == "BUY_ANIMAL":
                        buy_animal_orders += 1
            raw_observations, _, statuses = backend.step(actions)
            observations = canonical_observations(backend, raw_observations)
            max_seat_animals = max(max_seat_animals, seat_animals(observations))
            steps_run += 1
            if len(boundaries) >= days:
                break
            if all(status not in ("ACTIVE", "INACTIVE") for status in statuses):
                break
        if len(boundaries) < 2:
            raise RuntimeError(
                f"smoke did not cross a manager boundary: {boundaries!r}")
        engine_identity = {
            "backend": engine,
            "module": (
                "fast_env" if engine == "fast" else "kaggle_environments"
            ),
        }
        final_seat_animals = seat_animals(observations)
        return {
            "engine": engine_identity,
            "seed": seed,
            "mode": mode,
            "seats": [0, 1],
            "boundaries": boundaries,
            "boundary_count": len(boundaries),
            "steps": steps_run,
            "provider_days": [provider.last_accepted_day for provider in providers],
            "executor_profiles": [
                agent.effective_profile["version"] for agent in agents
            ],
            "statuses": list(statuses),
            "buy_animal_orders": buy_animal_orders,
            "final_seat_animals": final_seat_animals,
            "max_seat_animals": max_seat_animals,
            "plumbing_only": True,
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the bounded Stage 2.5 Packet 4 native lifecycle smoke")
    parser.add_argument("--engine", choices=("fast", "official"), default="fast")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--days", type=int, default=2)
    parser.add_argument("--mode", choices=("deterministic", "stochastic"),
                        default="deterministic")
    args = parser.parse_args(argv)
    print(json.dumps(run_smoke(
        engine=args.engine, seed=args.seed, days=args.days, mode=args.mode,
    ), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

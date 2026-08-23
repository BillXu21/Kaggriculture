"""Optional engine smoke harness for the V0 executor agent (issue #1 §9).

Runs one full local game through ``kaggle_environments`` when the engine
(1.32.x, env id ``kaggriculture``) is already installed. Nothing is ever
installed or vendored here; when the package is missing the harness prints a
clear report and exits with code 3 so CI can distinguish skip from failure.

Future Kaggle command interface (for final docs): build a submission whose
agent callable is ``executor_v0.agent.make_agent(checkpoint=<best.pt path>,
seat=...)`` — i.e. ``def agent(obs): return _AGENT(obs)`` with ``_AGENT``
constructed once at module import — and submit it as a normal Kaggle agent;
no network access or extra files are required at run time.

Example:
    python -m executor_v0.smoke --seed 7 --manager fake --opponent pass
    python -m executor_v0.smoke --manager checkpoint --checkpoint best.pt
"""

import argparse
import sys
from typing import Any

from .agent import AgentConfig, ExecutorAgent, make_agent
from .plan import DailyPlan

__all__ = ["detect_engine", "build_fake_plan", "run_smoke_game", "main"]

_ENGINE_ENV_ID = "kaggriculture"


def detect_engine() -> dict[str, Any]:
    """Return {'available': bool, 'version': str|None, 'reason': str}."""
    try:
        import kaggle_environments
    except ImportError as exc:
        return {"available": False, "version": None,
                "reason": f"kaggle_environments not installed ({exc})"}
    # 1.32.7 exposes the installed package version as ``__version__``.
    # A module-level ``version`` name may also exist as the imported
    # importlib.metadata function, so do not report that callable as the
    # engine version.
    version = getattr(kaggle_environments, "__version__", None)
    if not isinstance(version, str) or not version:
        version = "unknown"
    return {"available": True, "version": version, "reason": ""}


def build_fake_plan() -> DailyPlan:
    """Small fixed plan for plumbing smokes; no model involved."""
    return DailyPlan.create(
        crop_targets={"WHEAT": 4, "CARROT": 2, "TOMATO": 0,
                      "STRAWBERRY": 0, "MELON": 0},
        animal_targets={"GOOSE": 1, "COW": 0, "SHEEP": 0},
        land_count=1,
        fertilizer_by_crop={"WHEAT": 1, "CARROT": 0, "TOMATO": 0,
                            "STRAWBERRY": 0, "MELON": 0},
        care_by_animal={"GOOSE": 1, "COW": 0, "SHEEP": 0},
        sell_quantities={
            product: {anchor: (2 if product == "WHEAT" else 0)
                      for anchor in (0, 4, 8, 12, 16, 20)}
            for product in ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY",
                            "MELON", "EGG", "MILK", "WOOL", "FERTILIZER")
        },
    )


def build_agent(args: argparse.Namespace) -> ExecutorAgent:
    if args.manager == "fake":
        from .manager import FixedPlanProvider
        provider = FixedPlanProvider(build_fake_plan())
        return make_agent(provider=provider, seat=args.seat,
                          config=AgentConfig(strict=False))
    if not args.checkpoint:
        raise SystemExit("--manager checkpoint requires --checkpoint PATH")
    return make_agent(checkpoint=args.checkpoint, device=args.device,
                      seat=args.seat, config=AgentConfig(strict=False))


def run_smoke_game(agent: ExecutorAgent, opponent: Any, seed: int) -> dict:
    """Run one full game; returns a small machine-readable summary."""
    import kaggle_environments

    env = kaggle_environments.make(_ENGINE_ENV_ID,
                                   configuration={"seed": seed})
    env.reset()
    steps = env.run([agent, opponent])
    final_state = steps[-1]
    return {
        "env_id": _ENGINE_ENV_ID,
        "engine_version": detect_engine()["version"],
        "seed": seed,
        "steps": len(steps),
        "rewards": [state.reward for state in final_state],
        "statuses": [str(state.status) for state in final_state],
        "diagnostics": agent.diagnostics_json(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m executor_v0.smoke",
        description="One-game smoke of the executor_v0 V0 agent. Requires "
                    "kaggle_environments 1.32.x with the 'kaggriculture' "
                    "environment already installed; never installs anything.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--manager", choices=("fake", "checkpoint"),
                        default="fake",
                        help="fake=fixed plan provider (no model); "
                             "checkpoint=real BC manager from --checkpoint")
    parser.add_argument("--checkpoint", default=None,
                        help="path to the D-019 best.pt checkpoint")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seat", type=int, default=None,
                        choices=(None, 0, 1),
                        help="explicit seat; derived from obs when omitted")
    parser.add_argument("--opponent", choices=("pass",), default="pass",
                        help="only the trivial pass opponent is wired for V0")
    args = parser.parse_args(argv)

    engine = detect_engine()
    if not engine["available"]:
        print(f"SKIP: {engine['reason']}")
        print("Install kaggle-environments 1.32.x to run this smoke.")
        return 3
    print(f"engine: kaggle_environments {engine['version']}")

    agent = build_agent(args)
    opponent = lambda obs, config: {"farmer": ["PASS"], "hands": [],
                                    "market": []}  # noqa: E731
    summary = run_smoke_game(agent, opponent, seed=args.seed)
    print(f"steps={summary['steps']} rewards={summary['rewards']} "
          f"statuses={summary['statuses']}")
    fallbacks = len(summary["diagnostics"]["fallback_errors"])
    print(f"fallback_errors={fallbacks}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

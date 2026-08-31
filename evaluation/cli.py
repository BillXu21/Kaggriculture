"""Command line entrypoint for asymmetric controller panels.

Examples::

    python -m evaluation.cli --a ppo@promotions/promotion_003.npz \
      --b bc@artifacts/local/bc-v1-E/best.pt --b-name BC-E-aggressive \
      --b-aggressive-sell --seeds 6000..6031 --output artifacts/panel.json

    python -m evaluation.cli --a ppo@promotions/promotion_003.npz \
      --b external@/path/to/downloaded_agent --b-entrypoint agent.py:agent \
      --seeds 7,17 --output artifacts/external.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rl_manager.provenance import backend_provenance
from rl_manager.evaluation import summarize_evaluation

from evaluation.agent_match import run_panel
from evaluation.external import ExternalControllerFactory
from evaluation.internal import (
    PassControllerFactory,
    load_internal_factory,
    make_agent_config,
)


def _seed_values(values: list[str]) -> list[int]:
    seeds: list[int] = []
    for value in values:
        for token in value.split(","):
            token = token.strip()
            if ".." in token:
                first, last = token.split("..", 1)
                start, stop = int(first), int(last)
                if stop < start:
                    raise ValueError("seed ranges must be ascending")
                seeds.extend(range(start, stop + 1))
            elif token:
                seeds.append(int(token))
    if not seeds:
        raise ValueError("at least one seed is required")
    return seeds


def _config(path: str | None, aggressive: bool) -> Any:
    payload = None
    if path:
        with Path(path).open(encoding="utf-8") as stream:
            payload = json.load(stream)
    return make_agent_config(
        payload,
        aggressive_sell_all=True if aggressive else None,
    )


def _factory(
    spec: str,
    *,
    name: str | None,
    entrypoint: str,
    timeout: float | None,
    config_path: str | None,
    aggressive: bool,
    opening: str,
) -> Any:
    if spec == "pass":
        return PassControllerFactory(display_name=name or "PASS")
    if "@" not in spec:
        raise ValueError(
            f"controller spec {spec!r} must be pass, bc@PATH, bc-legacy@PATH, "
            "ppo@PATH, ppo-legacy@PATH, "
            "or external@PATH"
        )
    kind, path = spec.split("@", 1)
    if not path:
        raise ValueError(f"controller spec {spec!r} has an empty path")
    if kind == "external":
        return ExternalControllerFactory(
            source=path,
            entrypoint=entrypoint,
            timeout_seconds=timeout,
            display_name=name or Path(path).stem,
        )
    if kind not in {"bc", "bc-legacy", "ppo", "ppo-legacy",
                    "snapshot", "snapshot-legacy", "ppo-snapshot"}:
        raise ValueError(f"unknown controller kind {kind!r}")
    return load_internal_factory(
        kind,
        path,
        executor_config=_config(config_path, aggressive),
        display_name=name,
        opening_name=opening,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", default="pass", help="controller A spec")
    parser.add_argument("--b", default="pass", help="controller B spec")
    parser.add_argument("--a-name")
    parser.add_argument("--b-name")
    parser.add_argument("--a-entrypoint", default="agent.py:agent")
    parser.add_argument("--b-entrypoint", default="agent.py:agent")
    parser.add_argument("--a-config")
    parser.add_argument("--b-config")
    parser.add_argument("--a-aggressive-sell", action="store_true")
    parser.add_argument("--b-aggressive-sell", action="store_true")
    parser.add_argument("--a-opening", default="standard_mixed")
    parser.add_argument("--b-opening", default="standard_mixed")
    parser.add_argument("--external-timeout", type=float, default=None)
    parser.add_argument("--backend", choices=("fast", "official"), default="fast")
    parser.add_argument("--seeds", nargs="+", default=["0"], metavar="SEED|START..END")
    parser.add_argument("--seat0-only", action="store_true")
    parser.add_argument("--max-turns", type=int, default=719)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-threads", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    seeds = _seed_values(args.seeds)
    if args.num_threads < 1:
        raise ValueError("--num-threads must be positive")
    factory_a = _factory(
        args.a,
        name=args.a_name,
        entrypoint=args.a_entrypoint,
        timeout=args.external_timeout,
        config_path=args.a_config,
        aggressive=args.a_aggressive_sell,
        opening=args.a_opening,
    )
    factory_b = _factory(
        args.b,
        name=args.b_name,
        entrypoint=args.b_entrypoint,
        timeout=args.external_timeout,
        config_path=args.b_config,
        aggressive=args.b_aggressive_sell,
        opening=args.b_opening,
    )
    backend_configuration = {"seed": 0, "numThreads": args.num_threads}
    print(
        f"evaluation: backend={args.backend} games="
        f"{len(seeds) * (1 if args.seat0_only else 2)} "
        f"seeds={len(seeds)} orientations="
        f"{'seat0' if args.seat0_only else 'both'}"
    )
    results = run_panel(
        factory_a,
        factory_b,
        seeds=seeds,
        both_orientations=not args.seat0_only,
        backend_name=args.backend,
        backend_configuration=backend_configuration,
        max_turns=args.max_turns,
    )
    summary = summarize_evaluation(
        results,
        expected_seeds=seeds,
        expected_orientations=(
            ("candidate_vs_frozen",)
            if args.seat0_only
            else ("candidate_vs_frozen", "frozen_vs_candidate")
        ),
        provenance={
            "controller_a": dict(factory_a.provenance),
            "controller_b": dict(factory_b.provenance),
        },
    )
    output = {
        "schema_version": 1,
        "engine_provenance": backend_provenance(args.backend, backend_configuration),
        "panel_config": {
            "backend": args.backend,
            "seeds": seeds,
            "orientations": ["candidate_vs_frozen"]
            if args.seat0_only
            else ["candidate_vs_frozen", "frozen_vs_candidate"],
            "max_turns": args.max_turns,
            "num_threads": args.num_threads,
        },
        "controller_a": dict(factory_a.provenance),
        "controller_b": dict(factory_b.provenance),
        "games": [result.to_json_dict() for result in results],
        "evaluation": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        json.dump(output, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    wlt = summary["wlt"]
    print(
        f"result: {wlt['W']}W/{wlt['L']}L/{wlt['T']}T "
        f"mean_margin={summary['mean_margin']} "
        f"fatal={len(summary['fatal_anomalies'])} output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

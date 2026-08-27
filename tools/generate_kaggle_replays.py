"""Generate stock Kaggle replay JSON/HTML for exact V0.7-vs-PASS cases.

This utility intentionally follows the same manual action loop as
``tools.run_executor_v07_panel`` while stepping the pinned official
``kaggle_environments`` backend.  It does not use the custom debug viewer.

The JSON output is ``env.toJSON()`` and the HTML output is
``env.render(mode='html')``, so both are native Kaggle replay artifacts.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from executor_v0.agent import AgentConfig, make_agent
from opening_book.agent import make_opening_agent
from opening_book.trace import validate_action
from oracle.closed_loop import _executor_observation
from oracle.official_backend import OfficialKaggricultureBackend
from tools.run_executor_v07_panel import (
    DEFAULT_MAX_TRANSITIONS,
    DEFAULT_OPENING,
    _pass_action,
)


def _parse_case(value: str) -> tuple[int, int]:
    try:
        seed_text, seat_text = value.split(":", 1)
        seed = int(seed_text)
        seat = int(seat_text)
    except (AttributeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            f"case must be SEED:SEAT, got {value!r}"
        ) from exc
    if seed < 0:
        raise argparse.ArgumentTypeError("seed must be nonnegative")
    if seat not in (0, 1):
        raise argparse.ArgumentTypeError("seat must be 0 or 1")
    return seed, seat


def _parse_expected(value: str) -> tuple[tuple[int, int], float]:
    try:
        case_text, bank_text = value.split("=", 1)
        case = _parse_case(case_text)
        bank = float(bank_text)
    except (ValueError, argparse.ArgumentTypeError) as exc:
        raise argparse.ArgumentTypeError(
            f"expected bank must be SEED:SEAT=BANK, got {value!r}"
        ) from exc
    return case, bank


def _run_case(
    checkpoint: Path,
    seed: int,
    seat: int,
    *,
    output_dir: Path,
    expected_bank: float | None,
) -> dict[str, Any]:
    backend = OfficialKaggricultureBackend({"seed": seed})
    config = AgentConfig(
        strict=True,
        turn_trace=False,
        suppress_expansion_from_prior_debt=True,
    )
    downstream = make_agent(checkpoint=str(checkpoint), seat=seat, config=config)
    opening_agent = make_opening_agent(
        opening=DEFAULT_OPENING,
        downstream=downstream,
        seat=seat,
    )

    observations = backend.reset()
    terminal = False
    transitions = 0
    for transition in range(1, DEFAULT_MAX_TRANSITIONS + 1):
        tested_observation = _executor_observation(
            copy.deepcopy(observations[seat]), from_fast=False
        )
        opponent_observation = _executor_observation(
            copy.deepcopy(observations[1 - seat]), from_fast=False
        )

        tested_action = opening_agent(tested_observation)
        if not isinstance(tested_action, Mapping):
            raise RuntimeError(
                f"executor returned {type(tested_action).__name__} for {seed}:{seat}"
            )
        tested_action = copy.deepcopy(dict(tested_action))
        validate_action(tested_action, label=f"tested action {seed}:{seat} t={transition}")

        opponent_action = _pass_action(opponent_observation, 1 - seat)
        validate_action(opponent_action, label=f"PASS action {seed}:{seat} t={transition}")

        actions: list[Mapping[str, Any]] = [tested_action, opponent_action]
        if seat == 1:
            actions = [opponent_action, tested_action]

        observations, _, statuses = backend.step(copy.deepcopy(actions))
        backend.validate_status_history()
        transitions = transition
        terminal = list(statuses) == ["DONE", "DONE"]
        if terminal:
            break

    if not terminal or transitions != DEFAULT_MAX_TRANSITIONS:
        raise RuntimeError(
            f"case {seed}:{seat} did not complete the expected "
            f"{DEFAULT_MAX_TRANSITIONS} transitions: transitions={transitions}, "
            f"statuses={backend.statuses!r}"
        )

    opening_diagnostics = opening_agent.diagnostics_json()
    if opening_diagnostics.get("divergence", {}).get("occurred"):
        raise RuntimeError(f"opening divergence in {seed}:{seat}: {opening_diagnostics}")
    if opening_diagnostics.get("fallback_active"):
        raise RuntimeError(f"opening fallback in {seed}:{seat}: {opening_diagnostics}")
    if not opening_diagnostics.get("handoff", {}).get("clean_d4h0_handoff"):
        raise RuntimeError(f"unclean d4h0 handoff in {seed}:{seat}: {opening_diagnostics}")

    final_state = backend.canonical_state()
    farm = (final_state.get("farms") or [])[seat]
    if not isinstance(farm, Mapping):
        raise RuntimeError(f"missing final tested farm for {seed}:{seat}")
    bank = float(farm["money"])
    if expected_bank is not None and bank != float(expected_bank):
        raise RuntimeError(
            f"panel-bank mismatch for {seed}:{seat}: expected {expected_bank}, got {bank}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"seed_{seed}_seat_{seat}_v07_vs_pass"
    json_path = output_dir / f"{stem}.json"
    html_path = output_dir / f"{stem}.html"

    replay = backend.env.toJSON()
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(replay, handle, ensure_ascii=False, separators=(",", ":"))
        handle.write("\n")

    html = backend.env.render(mode="html")
    if not isinstance(html, str) or not html.strip():
        raise RuntimeError(f"Kaggle HTML renderer returned no HTML for {seed}:{seat}")
    html_path.write_text(html, encoding="utf-8")

    return {
        "seed": seed,
        "seat": seat,
        "bank": bank,
        "json": str(json_path),
        "html": str(html_path),
        "transitions": transitions,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--case",
        dest="cases",
        action="append",
        required=True,
        type=_parse_case,
        metavar="SEED:SEAT",
        help="Replay case; repeat for multiple cases.",
    )
    parser.add_argument(
        "--expected-bank",
        action="append",
        default=[],
        type=_parse_expected,
        metavar="SEED:SEAT=BANK",
        help="Optional exact bank assertion; repeat as needed.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/kaggle_replays"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    checkpoint = args.checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    if len(set(args.cases)) != len(args.cases):
        raise ValueError(f"duplicate cases are not allowed: {args.cases}")

    expected = dict(args.expected_bank)
    unknown_expected = sorted(set(expected) - set(args.cases))
    if unknown_expected:
        raise ValueError(
            f"--expected-bank supplied for cases not requested: {unknown_expected}"
        )

    summaries = []
    for seed, seat in args.cases:
        summary = _run_case(
            checkpoint,
            seed,
            seat,
            output_dir=args.output_dir,
            expected_bank=expected.get((seed, seat)),
        )
        summaries.append(summary)
        print(
            f"replay seed={seed} seat={seat} bank={summary['bank']:.1f} "
            f"json={summary['json']} html={summary['html']}"
        )

    manifest = args.output_dir / "manifest.json"
    manifest.write_text(
        json.dumps({"schema_version": 1, "games": summaries}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"manifest={manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

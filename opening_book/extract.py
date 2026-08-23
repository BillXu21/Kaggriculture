"""Deterministic single-replay opening-trace extractor (issue #4, stage 1).

Extracts the exact submitted primitive actions of one seat for days 0-3 from
one explicit raw replay file and writes canonical compact JSON. It never scans
a corpus and never touches the network.

Action alignment follows ``replay_daily/extractor.py``: ``steps[i][seat].action``
transforms observation ``i-1`` into observation ``i``, so the action acting on
the (day, hour) observation is read at step index + 1 of that observation's
first occurrence (same semantics as
``research/analyze_elite_openings.py::seat_hour_index``).

CLI::

    python -m opening_book.extract \
        --replay data/samples/2026-08-20/<episode>.json \
        --seat 0 --identity standard_mixed --out <path>.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from typing import Any

from .trace import (
    ENGINE_VERSION,
    FIRST_DAY,
    LAST_DAY,
    TURNS_PER_DAY,
    TRACE_FORMAT_VERSION,
    TraceError,
    canonical_json_bytes,
    compute_content_digest,
    validate_action,
    validate_trace,
)


def _seat_hour_index(replay: dict[str, Any], seat: int) -> dict[tuple[int, int], int]:
    """Map (day, hour) -> first step index whose observation starts that hour."""
    idx: dict[tuple[int, int], int] = {}
    for i, step in enumerate(replay["steps"]):
        obs = step[seat].get("observation") or {}
        key = (obs.get("day"), obs.get("hour"))
        if key not in idx:
            idx[key] = i
    return idx


def extract_opening_trace(
    replay_path: str, seat: int, identity: str
) -> dict[str, Any]:
    """Extract a validated trace document from one explicit raw replay file.

    Raises :class:`TraceError` on wrong engine version, invalid seat, missing
    horizon coverage, or any action-shape violation.
    """
    if seat not in (0, 1):
        raise TraceError(f"seat must be 0 or 1, got {seat!r}")

    with open(replay_path, "rb") as f:
        raw = f.read()
    replay_sha256 = hashlib.sha256(raw).hexdigest()
    replay = json.loads(raw.decode("utf-8"))

    module_version = replay.get("module_version")
    if module_version != ENGINE_VERSION:
        raise TraceError(
            f"{replay_path}: expected module_version {ENGINE_VERSION!r}, "
            f"got {module_version!r}"
        )

    info = replay.get("info") or {}
    episode_id = info.get("EpisodeId")
    team_names = info.get("TeamNames") or []
    seed = info.get("seed")
    player = team_names[seat] if len(team_names) > seat else None
    if not isinstance(episode_id, int) or not isinstance(seed, int) or not player:
        raise TraceError(
            f"{replay_path}: incomplete provenance in replay info "
            f"(EpisodeId={episode_id!r}, seed={seed!r}, player={player!r})"
        )

    steps = replay["steps"]
    idx = _seat_hour_index(replay, seat)

    turns: list[dict[str, Any]] = []
    for day in range(FIRST_DAY, LAST_DAY + 1):
        for hour in range(TURNS_PER_DAY):
            key = (day, hour)
            i = idx.get(key)
            if i is None or i + 1 >= len(steps):
                raise TraceError(
                    f"{replay_path}: no action step found for (day={day}, hour={hour})"
                )
            action = steps[i + 1][seat].get("action")
            validate_action(action, label=f"(day={day}, hour={hour})")
            turns.append({"day": day, "hour": hour, "action": action})

    doc: dict[str, Any] = {
        "format_version": TRACE_FORMAT_VERSION,
        "identity": identity,
        "module_version": ENGINE_VERSION,
        "horizon": {
            "first_day": FIRST_DAY,
            "last_day": LAST_DAY,
            "turns_per_day": TURNS_PER_DAY,
        },
        "provenance": {
            "source_episode": episode_id,
            "source_seat": seat,
            "source_seed": seed,
            "source_player": player,
            "source_replay_sha256": replay_sha256,
        },
        "content_digest": compute_content_digest(turns),
        "turns": turns,
    }
    validate_trace(doc)
    return doc


def write_trace(doc: dict[str, Any], out_path: str) -> None:
    """Write canonical deterministic compact JSON plus a trailing newline."""
    with open(out_path, "wb") as f:
        f.write(canonical_json_bytes(doc))
        f.write(b"\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract an opening trace from one raw replay file."
    )
    parser.add_argument("--replay", required=True, help="path to one raw replay JSON")
    parser.add_argument("--seat", required=True, type=int, choices=(0, 1))
    parser.add_argument(
        "--identity", required=True, choices=("standard_mixed", "pasture_heavy")
    )
    parser.add_argument("--out", required=True, help="output trace JSON path")
    args = parser.parse_args(argv)
    try:
        doc = extract_opening_trace(args.replay, args.seat, args.identity)
    except TraceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    write_trace(doc, args.out)
    print(
        f"wrote {args.out}: identity={doc['identity']} "
        f"episode={doc['provenance']['source_episode']} "
        f"seat={doc['provenance']['source_seat']} "
        f"turns={len(doc['turns'])} digest={doc['content_digest']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

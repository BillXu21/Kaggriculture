"""Command-line interface for the canonical daily replay extractor.

Usage:
    python -m replay_daily extract --input REPLAY.json [--input DIR ...]
        --output OUT.jsonl [--manifest MANIFEST.csv]
        [--source-dataset SLUG] [--partition-date YYYY-MM-DD]
        [--on-version-mismatch {skip,fail}]

    python -m replay_daily inspect RECORDS.jsonl [--episode ID] [--seat {0,1}]
        [--day D]

Single-process, local files only. Each input replay is parsed exactly once and
appended to OUTPUT as one JSON record per line per (episode, seat, day).
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .constants import ENGINE_VERSION
from .extractor import VersionMismatch, extract_replay, load_manifest


def _iter_input_files(inputs: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in inputs:
        p = Path(raw)
        if p.is_dir():
            files.extend(sorted(
                f for f in p.glob("*.json") if f.name != "manifest.csv"
            ))
        else:
            files.append(p)
    return files


def cmd_extract(args: argparse.Namespace) -> int:
    manifest: dict[int, dict[str, Any]] = {}
    if args.manifest:
        manifest = load_manifest(args.manifest)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {"parsed": 0, "skipped": 0, "records": 0}
    with out_path.open("w", encoding="utf-8", newline="\n") as out:
        for path in _iter_input_files(args.input):
            try:
                with open(path, encoding="utf-8") as f:
                    replay = json.load(f)
            except json.JSONDecodeError as exc:
                print(f"ERROR {path}: invalid JSON ({exc})", file=sys.stderr)
                return 2
            row = manifest.get(int((replay.get("info") or {}).get("EpisodeId", -1)))
            try:
                records = extract_replay(
                    replay,
                    source_dataset=args.source_dataset,
                    partition_date=args.partition_date,
                    source_path=str(path),
                    manifest_row=row,
                )
            except VersionMismatch as exc:
                if args.on_version_mismatch == "fail":
                    print(f"ERROR {path}: {exc}", file=sys.stderr)
                    return 3
                print(f"SKIP  {path}: {exc}")
                stats["skipped"] += 1
                continue
            for record in records:
                out.write(json.dumps(record, sort_keys=True, ensure_ascii=False))
                out.write("\n")
            stats["parsed"] += 1
            stats["records"] += len(records)

    print(
        f"extracted {stats['records']} records from {stats['parsed']} replay(s) "
        f"({stats['skipped']} skipped on version) -> {out_path}"
    )
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    with open(args.records, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            meta = rec["metadata"]
            if args.episode is not None and meta.get("episode_id") != args.episode:
                continue
            if args.seat is not None and meta.get("seat") != args.seat:
                continue
            if args.day is not None and rec.get("day") != args.day:
                continue
            print(json.dumps(rec, indent=2, sort_keys=True, ensure_ascii=False))
            return 0
    print("no matching record", file=sys.stderr)
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m replay_daily",
        description=(
            f"Extract canonical (episode, seat, day) records from "
            f"Kaggriculture {ENGINE_VERSION} replays into inspectable JSONL."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_extract = sub.add_parser("extract", help="parse replays and write JSONL records")
    p_extract.add_argument("--input", action="append", required=True,
                           help="replay .json file or directory (repeatable)")
    p_extract.add_argument("--output", required=True, help="output JSONL path")
    p_extract.add_argument("--manifest", help="daily partition manifest.csv for score metadata")
    p_extract.add_argument("--source-dataset", help="source dataset slug recorded in metadata")
    p_extract.add_argument("--partition-date", help="partition date recorded in metadata")
    p_extract.add_argument("--on-version-mismatch", choices=("skip", "fail"),
                           default="fail",
                           help="fail on non-1.32.7 replays (default) or skip them")
    p_extract.set_defaults(func=cmd_extract)

    p_inspect = sub.add_parser("inspect", help="pretty-print one record from a JSONL file")
    p_inspect.add_argument("records", help="JSONL produced by `extract`")
    p_inspect.add_argument("--episode", type=int, default=None)
    p_inspect.add_argument("--seat", type=int, choices=(0, 1), default=None)
    p_inspect.add_argument("--day", type=int, default=None)
    p_inspect.set_defaults(func=cmd_inspect)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

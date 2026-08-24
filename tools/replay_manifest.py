"""Manifest builder over a directory of raw Kaggriculture replay samples.

One JSON entry per ``*.json`` file (sorted by filename) with episode
metadata and a coarse classification. Unreadable files yield an error
entry instead of aborting the whole manifest.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

try:  # package context (tools.replay_manifest)
    from .replay_io import load_replay
except ImportError:  # script context (python tools/replay_manifest.py)
    from replay_io import load_replay

__all__ = [
    "FAILURE_SPECIMEN_EPISODE_ID",
    "HIGH_REWARD_MIN",
    "build_manifest",
    "classify",
    "write_manifest",
]

FAILURE_SPECIMEN_EPISODE_ID = 98178196
HIGH_REWARD_MIN = 40000


def classify(episode_id: int, rewards: list) -> str:
    """Coarse specimen classification for one episode."""
    if episode_id == FAILURE_SPECIMEN_EPISODE_ID:
        return "failure_specimen"
    if rewards and min(rewards) >= HIGH_REWARD_MIN:
        return "high_reward"
    return "unknown"


def build_manifest(sample_dir: str | Path) -> list[dict[str, Any]]:
    """Build one manifest entry per ``*.json`` file in ``sample_dir``."""
    entries: list[dict[str, Any]] = []
    for path in sorted(Path(sample_dir).glob("*.json")):
        entry: dict[str, Any] = {"file": path.name}
        try:
            replay = load_replay(str(path))
            info = replay.get("info") or {}
            config = replay.get("configuration") or {}
            raw_episode_id = info.get("EpisodeId")
            if raw_episode_id is None:
                raise ValueError("info.EpisodeId is missing")
            episode_id = int(raw_episode_id)
            seed = info.get("seed")
            teams = info.get("TeamNames")
            rewards = [float(r) for r in (replay.get("rewards") or [])]
            entry.update({
                "episode_id": episode_id,
                "seed": int(seed) if seed is not None else None,
                "teams": [str(t) for t in teams] if teams else None,
                "rewards": rewards,
                "engine_version": (
                    replay.get("version") or replay.get("module_version")
                ),
                "turns_per_day": int(config["turnsPerDay"]),
                "classification": classify(episode_id, rewards),
            })
        except Exception as error:  # robust per-file; never abort the batch
            entries.append({"file": path.name, "error": str(error)})
            continue
        entries.append(entry)
    return entries


def write_manifest(manifest: list[dict[str, Any]], path: str | Path) -> None:
    """Deterministic JSON dump (indent=2, sort_keys=True)."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 2:
        print(
            "usage: python tools/replay_manifest.py <sample_dir> <out.json>",
            file=sys.stderr,
        )
        return 2
    sample_dir, out_path = argv
    manifest = build_manifest(sample_dir)
    write_manifest(manifest, out_path)
    counts = Counter(
        str(entry.get("classification", "error")) for entry in manifest
    )
    print(f"entries={len(manifest)} out={out_path}")
    for name in sorted(counts):
        print(f"{name}={counts[name]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

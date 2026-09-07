"""Paired replay + executor capture for the Stage 2.5 upkeep experiment.

Instrumentation only. This module never touches policy behavior, heuristics,
task priorities, routing, or worker assignment: it only serializes artifacts
the runner and executors already produced during the game.

Capture layout per game (all JSON, large files gzipped)::

    <capture-dir>/<variant>/episode_<id>_seed_<seed>_seat_<seat>/
        meta.json                 # pairing identity + outcome + provenance
        debug_trace.json.gz       # canonical per-turn states + joint actions
                                  # + per-turn executor snapshots (both seats)
        rollout.json.gz           # submitted primitive actions, sampled
                                  # manager plans, request digests, handoff
        executor_seat0.json.gz    # full per-day executor diagnostics, seat 0
        executor_seat1.json.gz    # full per-day executor diagnostics, seat 1
        replay.json.gz            # official kaggle replay (official backend)
        status_history.json       # full official status history (official)
        capture_error.json        # only when capture itself failed; partial
                                  # files are kept alongside it

Task matching key: regenerated tasks carry stable ``key`` strings
(``KIND:...`` / ``SELL:...`` / ``BUY_...``); a task that disappears and
reappears under the same key is a regenerated equivalent, while a new key
is a genuinely new task. The audit treats keys accordingly and never
invents completion explanations beyond the recorded foreman ``reason``,
``unassigned_reasons``, and generator ``unresolved`` strings.
"""

from __future__ import annotations

import copy
import gzip
import json
import traceback
from pathlib import Path
from typing import Any, Mapping

CAPTURE_SCHEMA_VERSION = 1


def game_dir(capture_dir: Path, variant: str, episode_id: int,
             seed: int, seat: int) -> Path:
    """Deterministic per-game directory; safe to call before the game runs."""
    return (Path(capture_dir) / str(variant)
            / f"episode_{int(episode_id)}_seed_{int(seed)}_seat_{int(seat)}")


def _dumps(document: Any) -> bytes:
    return (json.dumps(document, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n"
            ).encode("utf-8")


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(payload)
    tmp.replace(path)


def write_json(path: Path, document: Any) -> None:
    _write_bytes(path, _dumps(document))


def write_json_gz(path: Path, document: Any) -> None:
    _write_bytes(path, gzip.compress(_dumps(document), compresslevel=6))


def rollout_to_json_dict(rollout: Any) -> dict[str, Any]:
    """Serialize a runner RolloutRecord without importing runner types."""
    joint = []
    for entry in rollout.joint_actions or []:
        step, day, hour, action0, action1 = entry
        joint.append({
            "step": int(step), "day": int(day), "hour": int(hour),
            "seat_0": copy.deepcopy(dict(action0)),
            "seat_1": copy.deepcopy(dict(action1)),
        })
    plans = {
        f"{int(seat)}/{int(day)}": copy.deepcopy(plan)
        for (seat, day), plan in (rollout.plans or {}).items()
    }
    digests = {
        f"{int(seat)}/{int(day)}": str(digest)
        for (seat, day), digest in
        (rollout.manager_input_digests or {}).items()
    }
    return {
        "seed": int(rollout.seed),
        "backend_name": str(rollout.backend_name),
        "composition": str(rollout.composition),
        "joint_actions": joint,
        "manager_input_digests": digests,
        "plans": plans,
        "opening_handoff": copy.deepcopy(rollout.opening_handoff or []),
    }


def write_game_capture(
    directory: Path,
    *,
    meta: Mapping[str, Any],
    debug_trace: Mapping[str, Any] | None,
    rollout: Any | None,
    executor_full_diagnostics: Any | None,
    official_replay: Mapping[str, Any] | None,
    status_history: Any | None,
) -> dict[str, Any]:
    """Write one game's capture incrementally; keep partials on failure.

    Files are written one at a time (atomic tmp+rename each), so an
    interrupted game still leaves the earlier files behind. ``meta.json``
    is written last with ``complete: true``. Capture failures are recorded
    in ``capture_error.json`` and returned -- never raised -- so a capture
    problem can never turn a completed game into a lost result.
    """
    directory = Path(directory)
    written: list[str] = []
    try:
        if debug_trace is not None:
            write_json_gz(directory / "debug_trace.json.gz",
                           copy.deepcopy(dict(debug_trace)))
            written.append("debug_trace.json.gz")
        if rollout is not None:
            write_json_gz(directory / "rollout.json.gz",
                           rollout_to_json_dict(rollout))
            written.append("rollout.json.gz")
        diags = list(executor_full_diagnostics or [])
        for seat_index, diagnostics in enumerate(diags):
            name = f"executor_seat{seat_index}.json.gz"
            write_json_gz(directory / name,
                           copy.deepcopy(dict(diagnostics)))
            written.append(name)
        if official_replay is not None:
            write_json_gz(directory / "replay.json.gz",
                           copy.deepcopy(dict(official_replay)))
            written.append("replay.json.gz")
        if status_history is not None:
            write_json(directory / "status_history.json",
                        [[str(s) for s in entry] for entry in status_history])
            written.append("status_history.json")
        final_meta = dict(meta)
        final_meta.update(
            capture_schema_version=CAPTURE_SCHEMA_VERSION,
            complete=True, files=list(written),
            has_official_replay=official_replay is not None)
        write_json(directory / "meta.json", final_meta)
        written.append("meta.json")
        return {"directory": str(directory), "files": written,
                "complete": True}
    except Exception as exc:  # noqa: BLE001 - partial artifacts are the point
        try:
            write_json(directory / "capture_error.json", {
                "error_type": type(exc).__name__,
                "message": str(exc),
                "files_written_before_error": written,
                "traceback": traceback.format_exc(limit=5),
            })
        except Exception:  # noqa: BLE001 - best effort only
            pass
        try:
            partial_meta = dict(meta)
            partial_meta.update(
                capture_schema_version=CAPTURE_SCHEMA_VERSION,
                complete=False,
                files=list(written),
                has_official_replay=official_replay is not None,
                capture_error=f"{type(exc).__name__}: {exc}")
            write_json(directory / "meta.json", partial_meta)
        except Exception:  # noqa: BLE001 - best effort only
            pass
        return {"directory": str(directory), "files": written,
                "complete": False,
                "capture_error": f"{type(exc).__name__}: {exc}"}


def read_json_gz(path: Path) -> Any:
    with gzip.open(path, "rb") as handle:
        return json.loads(handle.read().decode("utf-8"))


def iter_captured_games(capture_dir: Path) -> list[dict[str, Any]]:
    """List captures that finished with a complete meta.json."""
    games = []
    root = Path(capture_dir)
    if not root.is_dir():
        return games
    for meta_path in sorted(root.glob("*/*/meta.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if meta.get("complete") is True:
            games.append({"meta": meta, "directory": str(meta_path.parent)})
    return games

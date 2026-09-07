"""Lightweight audit of Stage 2.5 paired captures.

Postprocessor only: reads capture artifacts plus ``games.jsonl`` and emits a
compact Markdown report with machine-readable tables. It never reruns the
scheduler, never optimizes routes, and never invents reasons.

Reading cautions (applied throughout, not just listed here):

- movement is not automatically wasted;
- a queued task is not necessarily affordable, reachable, or eligible;
- unfinished work is not necessarily economically harmful;
- no missed-production-deadline claim is made unless mechanics and timing
  establish it;
- where scheduler eligibility cannot be recovered cheaply, the report says
  "idle with queued work", never "avoidable idle".

Task completion below is an *observed-interaction heuristic*: a tile task key
present at turn ``t``, followed by a matching interaction primitive at its
tile, then absent at ``t+1``, counts as "completed (observed)". A key that
ends with no matching interaction counts as "ended without observed
interaction". Matching uses the stable task ``key`` (``KIND:...``); a key
that reappears later is a regenerated equivalent, not a new task.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from rl_manager.debug_trace import validate_trace
from rl_manager.stage25_capture import iter_captured_games, read_json_gz

MOVEMENT_OPS = frozenset({"NORTH", "SOUTH", "EAST", "WEST"})
# Worker-side primitives that move inventory without field work.
LOGISTICS_OPS = frozenset({"PICKUP", "DROP"})
CAUTIONS = (
    "movement is not automatically wasted; a queued task is not necessarily "
    "affordable, reachable, or eligible; unfinished work is not necessarily "
    "economically harmful; no missed-production-deadline claim is made; "
    "PASS-while-queued is reported as \"idle with queued work\", never as "
    "\"avoidable idle\"."
)


def _is_mapping(value: Any) -> bool:
    return isinstance(value, Mapping)


def _canon_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def classify_worker_op(action: Sequence[Any]) -> str:
    """Bucket one worker primitive into a stable audit family."""
    op = str(action[0]) if action else "EMPTY"
    if op == "PASS":
        return "pass"
    if op in MOVEMENT_OPS:
        return "movement"
    if op in LOGISTICS_OPS:
        return "logistics"
    return f"interaction:{op}"


def _tile_counts(tiles: Any) -> tuple[dict[str, int], dict[str, int]]:
    """Count crops/animals on canonical farm tiles; tolerant to shape."""
    crops: Counter[str] = Counter()
    animals: Counter[str] = Counter()
    if not isinstance(tiles, list):
        return dict(crops), dict(animals)
    for row in tiles:
        if not isinstance(row, list):
            continue
        for tile in row:
            if not _is_mapping(tile):
                continue
            if "animal" in tile and isinstance(tile.get("animal"), str):
                animals[tile["animal"]] += 1
            elif tile.get("kind") == "PLANT" and isinstance(
                    tile.get("crop"), str):
                crops[tile["crop"]] += 1
    return dict(crops), dict(animals)


def _task_kind(key: Any) -> str:
    text = str(key) if isinstance(key, str) else ""
    return text.split(":", 1)[0] if ":" in text else text or "UNKNOWN"


def _task_tile(key: Any) -> tuple[int, int] | None:
    """Trailing ``y,x`` tile of a task key, or None for non-tile tasks."""
    if not isinstance(key, str):
        return None
    tail = key.rsplit(":", 1)[-1]
    try:
        y_text, x_text = tail.split(",", 1)
        return int(y_text), int(x_text)
    except (ValueError, AttributeError):
        return None


class AuditError(ValueError):
    """Raised when capture inputs are missing or not auditable."""


def load_game(directory: Path) -> dict[str, Any]:
    """Load one captured game directory into plain dicts."""
    directory = Path(directory)
    try:
        meta = json.loads((directory / "meta.json").read_text(
            encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise AuditError(f"{directory}: unreadable meta.json: {exc}") from exc
    if meta.get("complete") is not True:
        raise AuditError(f"{directory}: capture incomplete; not audited")
    try:
        trace = read_json_gz(directory / "debug_trace.json.gz")
    except (OSError, ValueError) as exc:
        raise AuditError(
            f"{directory}: unreadable debug_trace.json.gz: {exc}") from exc
    try:
        validate_trace(trace)
    except ValueError as exc:
        raise AuditError(f"{directory}: invalid debug trace: {exc}") from exc
    rollout = None
    rollout_path = directory / "rollout.json.gz"
    if rollout_path.is_file():
        rollout = read_json_gz(rollout_path)
    executors = {}
    for seat in (0, 1):
        path = directory / f"executor_seat{seat}.json.gz"
        if path.is_file():
            executors[seat] = read_json_gz(path)
    return {"meta": meta, "trace": trace, "rollout": rollout,
            "executors": executors, "directory": str(directory)}


def analyze_game(game: Mapping[str, Any]) -> dict[str, Any]:
    """Compute per-game and per-day measurements for both seats."""
    meta = game["meta"]
    turns = game["trace"]["turns"]
    executors = game.get("executors") or {}
    candidate_seat = int(meta.get("candidate_seat", 0))

    action_counts = {0: Counter(), 1: Counter()}
    worker_turns = {0: 0, 1: 0}
    market_submitted = {0: Counter(), 1: Counter()}
    completed = {0: Counter(), 1: Counter()}
    ended_unobserved = {0: Counter(), 1: Counter()}
    assignment_changes = {0: 0, 1: 0}
    change_after_interaction = {0: 0, 1: 0}
    change_without_interaction = {0: 0, 1: 0}
    coassigned_turns = {0: 0, 1: 0}
    idle_with_queued = {0: 0, 1: 0}
    care_assignments = {0: 0, 1: 0}
    fertilizer_assignments = {0: 0, 1: 0}

    prev_keys: dict[int, dict[int, str | None]] = {0: {}, 1: {}}
    prev_interaction: dict[int, dict[int, bool]] = {0: {}, 1: {}}
    live_keys: dict[int, set[str]] = {0: set(), 1: set()}
    interacted_keys: dict[int, set[str]] = {0: set(), 1: set()}

    day_rows: dict[int, dict[str, Any]] = {}

    def day_row(day: int) -> dict[str, Any]:
        return day_rows.setdefault(day, {
            "day": day,
            "money_0": None, "money_1": None,
            "land_0": None, "land_1": None,
            "crops_0": {}, "crops_1": {}, "animals_0": {}, "animals_1": {},
            "prices": {},
            "movement_0": 0, "movement_1": 0, "pass_0": 0, "pass_1": 0,
            "productive_0": 0, "productive_1": 0,
            "assign_changes_0": 0, "assign_changes_1": 0,
            "idle_queued_0": 0, "idle_queued_1": 0,
            "sells_0": 0, "sells_1": 0, "buys_0": 0, "buys_1": 0,
            "hires_0": 0, "hires_1": 0,
        })

    for turn in turns:
        day = int(turn["day"])
        row = day_row(day)
        state = turn.get("canonical_state") or {}
        farms = state.get("farms") or []
        for seat in (0, 1):
            if seat < len(farms) and _is_mapping(farms[seat]):
                farm = farms[seat]
                if row[f"money_{seat}"] is None:
                    row[f"money_{seat}"] = farm.get("money")
                    row[f"land_{seat}"] = len(
                        farm.get("unlocked_quadrants") or [])
                    crops, animals = _tile_counts(farm.get("tiles"))
                    row[f"crops_{seat}"] = crops
                    row[f"animals_{seat}"] = animals
        market = state.get("market") or {}
        if not row["prices"] and _is_mapping(market.get("prices")):
            row["prices"] = dict(market["prices"])

        joint = turn.get("joint_actions")
        debug = turn.get("executor_debug") or {}
        if not _is_mapping(joint):
            continue  # terminal snapshot: state only, no decision
        for seat in (0, 1):
            actions = joint.get(str(seat)) or {}
            worker_actions = [actions.get("farmer") or ["PASS"]]
            worker_actions.extend(actions.get("hands") or [])
            worker_turns[seat] += len(worker_actions)
            for action in worker_actions:
                family = classify_worker_op(
                    action if isinstance(action, (list, tuple)) else [])
                action_counts[seat][family] += 1
                if family == "movement":
                    row[f"movement_{seat}"] += 1
                elif family == "pass":
                    row[f"pass_{seat}"] += 1
                elif family.startswith("interaction:"):
                    row[f"productive_{seat}"] += 1
            for order in actions.get("market") or []:
                kind = str(order[0]) if order else "EMPTY"
                if kind == "SELL":
                    market_submitted[seat]["SELL"] += 1
                    row[f"sells_{seat}"] += 1
                elif kind.startswith("BUY"):
                    market_submitted[seat][kind] += 1
                    row[f"buys_{seat}"] += 1
                elif kind == "HIRE":
                    market_submitted[seat]["HIRE"] += 1
                    row[f"hires_{seat}"] += 1

            snapshot = debug.get(str(seat)) or {}
            assignments = snapshot.get("assignments") or []
            current: dict[int, str | None] = {}
            interacted_now: dict[int, bool] = {}
            seen_keys: Counter[str] = Counter()
            for assignment in assignments:
                if not _is_mapping(assignment):
                    continue
                worker = assignment.get("worker_index")
                key = assignment.get("task_key")
                action = assignment.get("action") or []
                op = str(action[0]) if action else "PASS"
                is_interaction = op not in ("PASS", *MOVEMENT_OPS,
                                            *LOGISTICS_OPS)
                if isinstance(worker, int):
                    current[worker] = key if isinstance(key, str) else None
                    interacted_now[worker] = bool(is_interaction)
                    if isinstance(key, str):
                        seen_keys[key] += 1
                        if key.startswith("CARE:"):
                            care_assignments[seat] += 1
                        elif key.startswith("FERTILIZE:"):
                            fertilizer_assignments[seat] += 1
                        if is_interaction and _task_tile(key) is not None:
                            target = assignment.get("target")
                            tile = _task_tile(key)
                            if (isinstance(target, list) and len(target) == 2
                                    and tile is not None
                                    and tuple(target) == (tile[1], tile[0])):
                                # Executor targets are [x, y]; task tiles
                                # are canonical [y, x].
                                interacted_keys[seat].add(key)
                            elif op == _task_kind(key):
                                interacted_keys[seat].add(key)
            for key, count in seen_keys.items():
                if count > 1:
                    coassigned_turns[seat] += 1
                    break
            for worker, key in current.items():
                previous = prev_keys[seat].get(worker)
                if previous is not None and key != previous:
                    assignment_changes[seat] += 1
                    row[f"assign_changes_{seat}"] += 1
                    if prev_interaction[seat].get(worker):
                        change_after_interaction[seat] += 1
                    else:
                        change_without_interaction[seat] += 1
            task_keys = {a.get("task_key") for a in assignments
                         if _is_mapping(a)
                         and isinstance(a.get("task_key"), str)}
            unassigned = ((snapshot.get("unassigned") or {}).get("task_keys")
                          or [])
            queued = bool(task_keys or unassigned)
            for assignment in assignments:
                if not _is_mapping(assignment):
                    continue
                action = assignment.get("action") or []
                if action and str(action[0]) == "PASS" and queued:
                    idle_with_queued[seat] += 1
                    row[f"idle_queued_{seat}"] += 1
                    break
            # Completion heuristic over regenerated task keys.
            known = {t.get("key") for t in (snapshot.get("tasks") or [])
                     if _is_mapping(t) and isinstance(t.get("key"), str)}
            for key in live_keys[seat] - known:
                kind = _task_kind(key)
                if _task_tile(key) is None:
                    continue
                if key in interacted_keys[seat]:
                    completed[seat][kind] += 1
                else:
                    ended_unobserved[seat][kind] += 1
            live_keys[seat] = {k for k in known if _task_tile(k) is not None}
            interacted_keys[seat] &= live_keys[seat]
            prev_keys[seat] = current
            prev_interaction[seat] = interacted_now

    day_debt = {}
    for seat in (0, 1):
        diagnostics = executors.get(seat) or {}
        days = diagnostics.get("days") or {}
        for day_text, record in days.items():
            try:
                day = int(day_text)
            except (TypeError, ValueError):
                continue
            debt = record.get("end_of_day_work_debt") or {}
            entry = day_debt.setdefault(day, {})
            entry[f"unfinished_{seat}"] = len(
                (record.get("unfinished_tasks") or []))
            entry[f"missed_maintenance_{seat}"] = len(
                record.get("missed_maintenance") or [])
            entry[f"debt_all_{seat}"] = list(debt.get("all") or [])
            entry[f"debt_survival_{seat}"] = list(debt.get("survival") or [])
            entry[f"debt_maintenance_{seat}"] = list(
                debt.get("maintenance") or [])
            entry[f"debt_productive_{seat}"] = list(
                debt.get("productive") or [])
            entry[f"debt_manager_{seat}"] = list(debt.get("manager") or [])
            entry[f"care_observed_{seat}"] = record.get(
                "care_completed_observed")
            entry[f"fertilizer_observed_{seat}"] = record.get(
                "fertilizer_completed_observed")
            entry[f"hires_submitted_{seat}"] = (record.get("hires") or {}).get(
                "submitted")
            pending = record.get("pending_task_turns") or {}
            entry[f"top_pending_{seat}"] = sorted(
                pending.items(), key=lambda kv: (-kv[1], kv[0]))[:5]

    game_row: dict[str, Any] = {
        "variant": meta.get("variant"),
        "seed": meta.get("seed"),
        "seat": meta.get("seat"),
        "episode_id": meta.get("episode_id"),
        "candidate_seat": candidate_seat,
        "bank": (meta.get("final_banks") or [None, None])[candidate_seat],
        "opponent_bank": (meta.get("final_banks") or [None, None])[
            1 - candidate_seat],
        "terminated": meta.get("terminated"),
        "statuses": meta.get("statuses"),
        "has_official_replay": meta.get("has_official_replay"),
    }
    game_row["margin"] = (
        game_row["bank"] - game_row["opponent_bank"]
        if isinstance(game_row["bank"], (int, float))
        and isinstance(game_row["opponent_bank"], (int, float)) else None)
    for seat in (0, 1):
        tag = "cand" if seat == candidate_seat else "opp"
        counts = action_counts[seat]
        game_row[f"{tag}_movement"] = (
            counts.get("movement", 0))
        game_row[f"{tag}_pass"] = counts.get("pass", 0)
        game_row[f"{tag}_logistics"] = counts.get("logistics", 0)
        game_row[f"{tag}_productive"] = sum(
            n for k, n in counts.items() if k.startswith("interaction:"))
        for op, count in sorted(counts.items()):
            if op.startswith("interaction:"):
                game_row[f"{tag}_{op.replace('interaction:', 'do_')}"] = count
        game_row[f"{tag}_worker_turns"] = worker_turns[seat]
        game_row[f"{tag}_assign_changes"] = assignment_changes[seat]
        game_row[f"{tag}_changes_after_interaction"] = \
            change_after_interaction[seat]
        game_row[f"{tag}_changes_without_interaction"] = \
            change_without_interaction[seat]
        game_row[f"{tag}_coassigned_turns"] = coassigned_turns[seat]
        game_row[f"{tag}_idle_with_queued"] = idle_with_queued[seat]
        game_row[f"{tag}_care_assignments"] = care_assignments[seat]
        game_row[f"{tag}_fertilizer_assignments"] = fertilizer_assignments[seat]
        game_row[f"{tag}_completed"] = dict(sorted(completed[seat].items()))
        game_row[f"{tag}_ended_unobserved"] = dict(
            sorted(ended_unobserved[seat].items()))
        game_row[f"{tag}_market"] = dict(sorted(market_submitted[seat].items()))

    ordered_days = []
    for day in sorted(day_row for day_row in day_rows):
        row = dict(day_rows[day])
        debt = day_debt.get(day, {})
        row.update(debt)
        row["variant"] = meta.get("variant")
        row["seed"] = meta.get("seed")
        row["seat"] = meta.get("seat")
        row["episode_id"] = meta.get("episode_id")
        ordered_days.append(row)
    return {"game": game_row, "days": ordered_days,
            "turns": turns, "rollout": game.get("rollout")}


def _plans_by_key(rollout: Mapping[str, Any] | None) -> dict[str, Any]:
    if not _is_mapping(rollout):
        return {}
    plans = rollout.get("plans") or {}
    return dict(plans) if _is_mapping(plans) else {}


def divergence_between(baseline: Mapping[str, Any],
                       other: Mapping[str, Any]) -> dict[str, Any]:
    """First divergences between two analyzed games (paired by identity)."""
    base_turns = baseline["turns"]
    other_turns = other["turns"]
    common = min(len(base_turns), len(other_turns))
    action_div = state_div = None
    for index in range(common):
        base, theirs = base_turns[index], other_turns[index]
        if action_div is None and _is_mapping(base.get("joint_actions")) \
                and _is_mapping(theirs.get("joint_actions")):
            if _canon_json(base["joint_actions"]) != _canon_json(
                    theirs["joint_actions"]):
                action_div = {"turn": index, "step": base.get("step"),
                              "day": base.get("day"), "hour": base.get("hour")}
        if state_div is None:
            if _canon_json(base.get("canonical_state")) != _canon_json(
                    theirs.get("canonical_state")):
                state_div = {"turn": index, "step": base.get("step"),
                             "day": base.get("day"), "hour": base.get("hour")}
        if action_div is not None and state_div is not None:
            break
    base_plans = _plans_by_key(baseline.get("rollout"))
    other_plans = _plans_by_key(other.get("rollout"))
    plan_div = None
    for key in sorted(set(base_plans) | set(other_plans)):
        if _canon_json(base_plans.get(key)) != _canon_json(
                other_plans.get(key)):
            seat_text, day_text = key.split("/", 1)
            plan_div = {"seat": int(seat_text), "day": int(day_text),
                        "plan_key": key}
            break
    return {"first_action_divergence": action_div,
            "first_state_divergence": state_div,
            "first_plan_divergence": plan_div,
            "compared_turns": common,
            "turn_count_baseline": len(base_turns),
            "turn_count_other": len(other_turns)}


def _delta(game_row: Mapping[str, Any], base_row: Mapping[str, Any],
           field: str) -> float | None:
    value, base = game_row.get(field), base_row.get(field)
    if isinstance(value, (int, float)) and isinstance(base, (int, float)):
        return value - base
    return None


def pair_report(variant: str, analyses: Mapping[tuple, dict[str, Any]],
                baseline_variant: str = "baseline") -> list[dict[str, Any]]:
    """Paired per-(seed, seat) comparison of one variant against baseline."""
    pairs = []
    keys = sorted({k[1:] for k in analyses if k[0] == variant})
    for seed, seat in keys:
        game = analyses.get((variant, seed, seat))
        base = analyses.get((baseline_variant, seed, seat))
        if game is None or base is None:
            pairs.append({"variant": variant, "seed": seed, "seat": seat,
                          "paired": False,
                          "reason": "missing baseline or variant capture"})
            continue
        grow, brow = game["game"], base["game"]
        divergence = divergence_between(base, game)
        entry: dict[str, Any] = {
            "variant": variant, "seed": seed, "seat": seat,
            "episode_id": grow.get("episode_id"), "paired": True,
            "bank": grow.get("bank"), "bank_delta": _delta(
                grow, brow, "bank"),
            "opponent_bank": grow.get("opponent_bank"),
            "opponent_bank_delta": _delta(grow, brow, "opponent_bank"),
            "margin": grow.get("margin"),
            "margin_delta": (_delta(grow, brow, "margin")),
            "cand_movement_delta": _delta(grow, brow, "cand_movement"),
            "cand_pass_delta": _delta(grow, brow, "cand_pass"),
            "cand_productive_delta": _delta(grow, brow, "cand_productive"),
            "cand_assign_changes_delta": _delta(
                grow, brow, "cand_assign_changes"),
            "cand_idle_with_queued_delta": _delta(
                grow, brow, "cand_idle_with_queued"),
            "cand_care_delta": _delta(grow, brow, "cand_care_assignments"),
            "cand_fertilizer_delta": _delta(
                grow, brow, "cand_fertilizer_assignments"),
            "divergence": divergence,
        }
        pairs.append(entry)
    return pairs


def _fmt_bank(value: Any) -> str:
    return f"{value:,.0f}" if isinstance(value, (int, float)) else "n/a"


def _fmt_delta(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "n/a"
    return f"{value:+,.0f}"


def _fmt_div(label: str, div: Mapping[str, Any] | None) -> str:
    if not div:
        return f"{label}: none within compared turns"
    if "turn" in div:
        return (f"{label}: turn {div['turn']} "
                f"(step {div.get('step')}, day {div.get('day')} "
                f"hour {div.get('hour')})")
    return (f"{label}: seat {div.get('seat')} day {div.get('day')} "
            f"(plan key {div.get('plan_key')})")


def render_markdown(analyses: Mapping[tuple, dict[str, Any]],
                    pairs_by_variant: Mapping[str, list[dict[str, Any]]],
                    games_rows: list[dict[str, Any]],
                    baseline_variant: str = "baseline",
                    focus_pairs: int = 4) -> str:
    """Compact report; every bottleneck links game/day/turn evidence."""
    lines = ["# Stage 2.5 upkeep capture audit", "",
             f"Baseline arm: `{baseline_variant}`. {CAUTIONS}", "",
             "## Panel means (candidate perspective)", ""]
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in games_rows:
        by_variant[str(row.get("variant"))].append(row)
    lines.append("| variant | games | mean bank | mean opp bank | mean margin |"
                 " mean movement | mean pass | mean productive |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for variant in sorted(by_variant):
        rows = by_variant[variant]
        def mean(field: str) -> str:
            values = [r[field] for r in rows
                      if isinstance(r.get(field), (int, float))]
            return f"{sum(values) / len(values):,.0f}" if values else "n/a"
        lines.append(
            f"| {variant} | {len(rows)} | {mean('bank')} | "
            f"{mean('opponent_bank')} | {mean('margin')} | "
            f"{mean('cand_movement')} | {mean('cand_pass')} | "
            f"{mean('cand_productive')} |")
    lines += ["", "## Paired divergences and deltas", ""]
    focus: list[dict[str, Any]] = []
    for variant, pairs in pairs_by_variant.items():
        if variant == baseline_variant:
            continue
        for pair in pairs:
            if pair.get("paired"):
                focus.append(pair)
    focus.sort(key=lambda p: abs(p.get("bank_delta") or 0), reverse=True)
    if not focus:
        lines.append("No paired games available.")
    for pair in focus[:max(0, focus_pairs)]:
        div = pair["divergence"]
        lines.append(
            f"### {pair['variant']} seed {pair['seed']} seat {pair['seat']} "
            f"(episode {pair.get('episode_id')})")
        lines.append(
            f"Bank {_fmt_bank(pair['bank'])} ({_fmt_delta(pair['bank_delta'])} "
            f"vs {baseline_variant}); opponent {_fmt_bank(pair['opponent_bank'])} "
            f"({_fmt_delta(pair['opponent_bank_delta'])}); margin "
            f"{_fmt_bank(pair['margin'])} ({_fmt_delta(pair['margin_delta'])}).")
        lines.append(_fmt_div("First primitive-action divergence",
                              div.get("first_action_divergence")) + ".")
        lines.append(_fmt_div("First state divergence",
                              div.get("first_state_divergence")) + ".")
        lines.append(_fmt_div("First manager-plan divergence",
                              div.get("first_plan_divergence")) + ".")
        action_div = div.get("first_action_divergence") or {}
        plan_div = div.get("first_plan_divergence") or {}
        if action_div and (not plan_div or action_div.get("turn", 0) <= 0
                           or plan_div.get("day", 0) * 24
                           > action_div.get("turn", 0)):
            lines.append("Read: state changed because a different action "
                         "occurred first; do not imply the manager changed "
                         "first.")
        lines.append(
            "Travel delta "
            f"{_fmt_delta(pair['cand_movement_delta'])} worker-turns; "
            f"productive delta {_fmt_delta(pair['cand_productive_delta'])}; "
            f"PASS delta {_fmt_delta(pair['cand_pass_delta'])}; assignment "
            f"changes delta {_fmt_delta(pair['cand_assign_changes_delta'])}; "
            "idle-with-queued-work delta "
            f"{_fmt_delta(pair['cand_idle_with_queued_delta'])} "
            "(eligibility unverified).")
        lines.append(
            "Care-assignment delta "
            f"{_fmt_delta(pair['cand_care_delta'])}; fertilizer-assignment "
            f"delta {_fmt_delta(pair['cand_fertilizer_delta'])}.")
        lines.append("")
    lines += ["## Combined vs fertilizer-only (observations, not causes)", ""]
    both = [p for p in focus
            if p["variant"] in ("combined", "fertilizer")]
    by_identity: dict[tuple, dict[str, dict[str, Any]]] = defaultdict(dict)
    for pair in both:
        by_identity[(pair["seed"], pair["seat"])][pair["variant"]] = pair
    if not by_identity:
        lines.append("No combined/fertilizer pairs captured yet.")
    complete = 0
    for (seed, seat), variants in sorted(by_identity.items()):
        comb = variants.get("combined")
        fert = variants.get("fertilizer")
        if comb is None or fert is None:
            continue
        complete += 1
        lines.append(f"### seed {seed} seat {seat}")
        for label, pair in (("combined", comb), ("fertilizer-only", fert)):
            div = pair["divergence"]
            lines.append(
                f"- {label}: bank {_fmt_bank(pair['bank'])} "
                f"({_fmt_delta(pair['bank_delta'])}); "
                + _fmt_div("action divergence",
                           div.get("first_action_divergence")) + "; "
                + _fmt_div("plan divergence",
                           div.get("first_plan_divergence")) + ".")
        lines.append("- Compare day-end unfinished planting/harvest work, "
                     "farm-composition trajectories, and opponent-bank "
                     "trajectories in `days.csv`/`games.csv` before "
                     "hypothesizing; correlations here are not causal.")
        lines.append("")
    if not complete:
        lines.append("No complete combined/fertilizer identity shares this "
                     "slice; any cross-arm correlation is not causal "
                     "evidence.")
        lines.append("")
    lines += ["## Evidence-linked bottleneck candidates (hypotheses only)",
              "",
              "Each item below names the game/day/turn evidence to open; "
              "none of these is implemented or claimed as a fix.", ""]
    signals: list[tuple[float, str]] = []
    for key, analysis in analyses.items():
        variant, seed, seat = key
        for day in analysis["days"]:
            for tag in ("cand", "opp"):
                idle = day.get(f"idle_queued_{tag}", 0) or 0
                if idle >= 10:
                    signals.append((idle, f"- {variant} seed {seed} seat "
                                          f"{seat} day {day['day']}: {idle} "
                                          f"idle-with-queued-work "
                                          f"worker-turns ({tag}); open the "
                                          f"day's executor_debug turns and "
                                          f"unassigned_reasons before "
                                          f"concluding anything."))
                unfinished = day.get(f"unfinished_{tag}")
                if isinstance(unfinished, int) and unfinished >= 5:
                    signals.append((unfinished, f"- {variant} seed {seed} "
                                                f"seat {seat} day "
                                                f"{day['day']}: {unfinished} "
                                                f"unfinished tasks at day end "
                                                f"({tag}); check debt "
                                                f"categories and economics "
                                                f"before calling it harmful."))
    signals.sort(key=lambda item: -item[0])
    lines.extend([text for _, text in signals[:20]] or
                 ["No strong workload signals in this capture slice."])
    lines += ["", "## Reading cautions", "",
              f"{CAUTIONS} No route optimizer or counterfactual scheduler "
              "was used.", ""]
    return "\n".join(lines) + "\n"


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns,
                                extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            flat = {k: (_canon_json(v) if isinstance(v, (dict, list)) else v)
                    for k, v in row.items()}
            writer.writerow(flat)


def run_audit(*, capture_dir: Path, output_dir: Path,
              baseline_variant: str = "baseline",
              variants: Sequence[str] | None = None,
              focus_pairs: int = 4) -> dict[str, Any]:
    """Audit every complete capture; returns artifact paths + counts."""
    found = iter_captured_games(capture_dir)
    if variants is not None:
        wanted = set(variants)
        found = [g for g in found if g["meta"].get("variant") in wanted]
    analyses: dict[tuple, dict[str, Any]] = {}
    skipped: list[dict[str, str]] = []
    for entry in found:
        meta = entry["meta"]
        key = (meta.get("variant"), meta.get("seed"), meta.get("seat"))
        try:
            game = load_game(Path(entry["directory"]))
        except AuditError as exc:
            skipped.append({"directory": entry["directory"],
                            "reason": str(exc)})
            continue
        analyses[key] = analyze_game(game)
    if not analyses:
        raise AuditError(f"no complete captures under {capture_dir}")
    games_rows = [a["game"] for a in analyses.values()]
    days_rows: list[dict[str, Any]] = []
    for analysis in analyses.values():
        days_rows.extend(analysis["days"])
    pairs_by_variant = {
        variant: pair_report(variant, analyses,
                             baseline_variant=baseline_variant)
        for variant in sorted({k[0] for k in analyses})}
    divergences: list[dict[str, Any]] = []
    for variant, pairs in pairs_by_variant.items():
        for pair in pairs:
            if not pair.get("paired"):
                continue
            div = pair["divergence"]
            divergences.append({
                "variant": variant, "seed": pair["seed"], "seat": pair["seat"],
                "episode_id": pair.get("episode_id"),
                "bank_delta": pair.get("bank_delta"),
                "opponent_bank_delta": pair.get("opponent_bank_delta"),
                "margin_delta": pair.get("margin_delta"),
                "first_action_turn": (div.get("first_action_divergence") or {})
                .get("turn"),
                "first_action_day": (div.get("first_action_divergence") or {})
                .get("day"),
                "first_state_turn": (div.get("first_state_divergence") or {})
                .get("turn"),
                "first_state_day": (div.get("first_state_divergence") or {})
                .get("day"),
                "first_plan_seat": (div.get("first_plan_divergence") or {})
                .get("seat"),
                "first_plan_day": (div.get("first_plan_divergence") or {})
                .get("day"),
            })
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "games.csv", games_rows)
    _write_csv(output_dir / "days.csv", days_rows)
    pairs_rows = []
    for variant, pairs in pairs_by_variant.items():
        for pair in pairs:
            flat = {k: (_canon_json(v) if isinstance(v, (dict, list)) else v)
                    for k, v in pair.items() if k != "divergence"}
            pairs_rows.append(flat)
    _write_csv(output_dir / "pairs.csv", pairs_rows)
    _write_csv(output_dir / "divergences.csv", divergences)
    (output_dir / "divergences.json").write_text(
        json.dumps(divergences, indent=2) + "\n", encoding="utf-8")
    if skipped:
        (output_dir / "skipped.json").write_text(
            json.dumps(skipped, indent=2) + "\n", encoding="utf-8")
    markdown = render_markdown(analyses, pairs_by_variant, games_rows,
                               baseline_variant=baseline_variant,
                               focus_pairs=focus_pairs)
    (output_dir / "audit.md").write_text(markdown, encoding="utf-8")
    return {"games": len(games_rows), "day_rows": len(days_rows),
            "pairs": len(pairs_rows), "skipped": skipped,
            "report": str(output_dir / "audit.md")}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--baseline", default="baseline")
    parser.add_argument("--variants", nargs="*", default=None)
    parser.add_argument("--focus-pairs", type=int, default=4)
    args = parser.parse_args(argv)
    if args.focus_pairs < 0:
        parser.error("--focus-pairs must be nonnegative")
    summary = run_audit(capture_dir=args.capture_dir, output_dir=args.output_dir,
                        baseline_variant=args.baseline, variants=args.variants,
                        focus_pairs=args.focus_pairs)
    print(json.dumps({k: v for k, v in summary.items()
                      if k != "skipped"}, indent=2))
    if summary["skipped"]:
        print(f"skipped {len(summary['skipped'])} captures; "
              "see skipped.json")
    print(f"report: {summary['report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
